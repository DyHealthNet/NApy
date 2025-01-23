#include <stats.hpp>
#include <cmath>

// Computes NAN-aware Spearman correlation of two vectors.
std::pair<double, double> pairwise_nan_spearman(const DataMatrix& data, 
    int row1, int row2, const vec2d& ranksRow1, 
    const vec2d& ranksRow2, double na_value)
{
    // Iterate over each row and extract subvector of pairwise non-NAN features.
    const int numCols = data.cols();
    int numNonNAs = 0;

    std::vector<double> updatedRanks1(numCols, -1);

    // Iterate through rank maps of each row and account for NAs in other vector.
    double rankSum1 = 0.0;
    double rankSumSquared1 = 0.0;
    int subtractRight = 0;
    for (size_t iV = 0; iV < ranksRow1.size(); ++iV)
    {
        const int rank = iV+1;
        // const std::vector<int> rowSet = ranksRow1[iV];
        // Check if rank actually contains values, if not skip this iteration.
        if (ranksRow1[iV].size()>0)
        {
            // Iterate through "bin" and sort out NA indices of second vector.
            int subsetSize = 0;
            for (const auto& index : ranksRow1[iV])
            {
                if (data(row2, index)!= na_value)
                {
                    ++subsetSize;
                    ++numNonNAs;
                }
            }

            // Updated ranks compute as average over updated ascending sequence of integers.
            if (subsetSize > 0)
            {
                const double startRank = rank - subtractRight;
                const double average = 1.0/subsetSize * (subsetSize*startRank + 0.5*subsetSize*(subsetSize-1));
                // Set averaged rank to all remaining elements in subset of row.
                for (const auto& el : ranksRow1[iV])
                {
                    if (data(row2, el)!=na_value)
                    {
                        updatedRanks1[el] = average;
                        rankSum1 += average;
                        rankSumSquared1 += average*average;
                    }
                } 
            }
            // Number of elements in current container that were deleted.
            const int subtract_extra = ranksRow1[iV].size() - subsetSize;
            subtractRight += subtract_extra;
        }
    }

    subtractRight = 0;
    double rankSum2 = 0.0;
    double rankSumSquared2 = 0.0;
    double one_times_two = 0.0;

    for (size_t iV = 0; iV < ranksRow2.size(); ++iV)
    {
        const int rank = iV+1;
        // const std::vector<int> rowSet = ranksRow2[iV];
        // Check if rank actually contains values, if not skip this iteration.
        if (ranksRow2[iV].size()>0)
        {
            // Iterate through "bin" and sort out NA indices of second vector.
            int subsetSize = 0;
            for (const auto& index : ranksRow2[iV])
            {
                if (data(row1, index)!= na_value)
                {
                    ++subsetSize;
                }
            }

            // Updated ranks compute as average over updated ascending sequence of integers.
            if (subsetSize > 0)
            {
                const double startRank = rank - subtractRight;
                const double average = 1.0/subsetSize * (subsetSize*startRank + 0.5*subsetSize*(subsetSize-1));
                // Set averaged rank to all remaining elements in subset of row.
                for (const auto& el : ranksRow2[iV])
                {
                    // updatedRanks2[el] = average;
                    rankSum2 += average;
                    rankSumSquared2 += average*average;
                    // Also directly compute index-matching product for nominator of Pearson correlation.
                    one_times_two += average * updatedRanks1[el];
                } 
            }
            // Number of elements in current container that were deleted.
            const int subtract_extra = ranksRow2[iV].size() - subsetSize;
            subtractRight += subtract_extra;
        }
    }

    // Compute Pearson Correlation on rank-transformed data.
    const double average1 = rankSum1 / numNonNAs;
    const double average2 = rankSum2 / numNonNAs;
    const double nominator = one_times_two - average2*rankSum1 - average1*rankSum2 + numNonNAs*average1*average2;
    double variance_x = rankSumSquared1 - 2*average1*rankSum1 + numNonNAs*average1*average1;
    double variance_y = rankSumSquared2 - 2*average2*rankSum2 + numNonNAs*average2*average2;
    
    const double correlation = nominator / std::sqrt(variance_x * variance_y);

    // If number of paired no-NAs is smaller than 3, Pvalue is not well-defined.
    if (numNonNAs < 2)
    {
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }
    else if (numNonNAs < 3)
    {
        return std::make_pair(correlation, std::numeric_limits<double>::quiet_NaN());
    }
    // Compute p-value based on Student's t distribution CDF.
    double pvalue = 0.0;
    // Correlation value might not be equal to 1.0, hence allow tolerance of 1E-14, 
    // in the cases of corr=1.0, the difference was in the magnitude of E-15.
    if (abs(correlation-1.0) < 1E-14 || abs(correlation+1.0) < 1E-14 )
    {
        pvalue = 0.0;
    }
    else
    {
        const int numDOFs = numNonNAs - 2;
        const double t = correlation * std::sqrt(numDOFs / ( (correlation + 1.0) * (1.0 - correlation) ));
        boost::math::students_t dist(numDOFs);
        const double transformed_t = -1.0*abs(t);
        if (std::isnan(transformed_t))
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        pvalue = 2.0 * cdf(dist, -1.0*abs(t));
    }

    return std::make_pair(correlation, pvalue);
}

// NAN-aware Spearman Correlation on given DataMatrix.
std::pair<DataMatrix, DataMatrix> statistics::spearman_with_nans(const DataMatrix& data, 
    double na_value)
{
    const int num_rows = data.rows();
    const int num_cols = data.cols();
    DataMatrix correlations(num_rows, num_rows);
    DataMatrix pvalues(num_rows, num_rows);
    
    vec3d row_maps(num_rows);

    // Sort and compute ranks of each row before correlation computation.
    # pragma omp parallel for
    for (int iR = 0; iR < num_rows; ++iR)
    {
        // Directly sort indices of row values.
        std::vector<int> indices(num_cols);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b){return data(iR, a)<data(iR, b);});
        size_t vecCounter = 0;
        int rankCounter = 1;
        
        vec2d row_map(num_cols, std::vector<int>());
        // Iterate through whole sorted indices vector.
        while (vecCounter < indices.size())
        {
            // If value is NA, simply ignore it.
            if (data(iR, indices[vecCounter])==na_value)
            {
                // Vector element counter increases only by one.
                ++vecCounter;
            }
            else // Value is non-NA.
            {
                std::vector<int> rank_bin;
                int forward_counter = 1;
                // Count how many subsequent data points are equal, i.e. how many ties there are.
                while((vecCounter+forward_counter) < indices.size() && data(iR,indices[vecCounter]) == data(iR,indices[vecCounter+forward_counter]))
                {
                    ++forward_counter;
                }
                // Put ties into same rank bin.
                for (size_t iP = vecCounter; iP < vecCounter+forward_counter; ++iP)
                {
                    rank_bin.push_back(indices[iP]);
                }
                // Add all indices with same rank bin into map with key of current rankCounter value.
                row_map[rankCounter-1] = rank_bin;
                // Rank counter increases by number of values that have been averaged over.
                rankCounter += forward_counter;
                // Vector index counter increases by number of values that have been averaged over.
                vecCounter += forward_counter;
            }
            
        }

        // Store map with ranks and bins for each row.
        row_maps[iR] = row_map;
    }

    // Compute each pairwise correlation of rows.
    #pragma omp parallel for
    for (int iR = 0; iR < num_rows; ++iR)
    {
        for (int jR = iR; jR < num_rows; ++jR)
        {
            std::pair<double, double> results = pairwise_nan_spearman(data, iR, jR, 
                row_maps[iR], row_maps[jR], na_value);
            correlations(iR, jR) = std::get<0>(results);
            correlations(jR, iR) = std::get<0>(results);
            double pvalue = std::get<1>(results);
            pvalues(iR, jR) = pvalue;
            pvalues(jR, iR) = pvalue;
        }
    }

    return std::make_pair(correlations, pvalues);
}
