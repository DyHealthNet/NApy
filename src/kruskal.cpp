#include <stats.hpp>
#include <cmath>

std::tuple<double, double, double> get_nans_kruskal()
{
    return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
}

std::tuple<double, double, double> pairwise_nan_kruskal(const DataMatrix& cat_data, 
    int iCat, int category_groups, const vec2d& rowRanks, 
    double na_value)
{
    // Iterate over each row and extract subvector of pairwise non-NAN features.
    int numDataPoints = 0;

    std::vector<double> groupRankSums(category_groups, 0.0);
    std::vector<int> groupSizes(category_groups, 0);
    double tie_correction = 0.0;

    // Iterate through rank maps of each row and account for NAs in other vector.
    int subtractRight = 0;
    for(size_t iR = 0; iR < rowRanks.size(); ++iR)
    {
        const int rank = iR+1;
        // const std::vector<int>& rowSet = rowRanks[iR];
        // Check if rank actually contains values, if not skip this iteration.
        if (rowRanks[iR].size()>0)
        {
            // Iterate through "bin" and sort out NA indices of second vector.
            int subsetSize = 0;
            for (const auto& index : rowRanks[iR])
            {
                if (cat_data(iCat, index)!= na_value)
                {
                    ++subsetSize;
                    ++numDataPoints;
                    groupSizes[cat_data(iCat, index)] += 1;
                }
            }
            // Again iterate over "bin" and update rank sums of corresponding categories.
            if (subsetSize > 0)
            {
                const double startRank = rank - subtractRight;
                const double average = 1.0/subsetSize * (subsetSize*startRank + 0.5*subsetSize*(subsetSize-1));
                for (const auto& index : rowRanks[iR])
                {
                    if (cat_data(iCat, index)!= na_value)
                    {
                        // Add averaged rank to rank sum of corresponding group.
                        groupRankSums[cat_data(iCat, index)] += average;
                    }
                }
                // Compute correction term for denominator in H statistic in case of ties.
                if (subsetSize > 1)
                {
                    tie_correction += subsetSize*subsetSize*subsetSize - subsetSize;
                }
            }

            // Number of elements in current container that were deleted.
            const int subtract_extra = rowRanks[iR].size() - subsetSize;
            subtractRight += subtract_extra;
        }
    }

    // Compute H statistic value by aggregating per-category rank sums.
    double h_statistic = 0.0;
    for (int iC = 0; iC < category_groups; ++iC)
    {
        if (groupSizes[iC] == 0)
            return get_nans_kruskal();

        h_statistic += groupRankSums[iC]*groupRankSums[iC]/groupSizes[iC];
    }
    h_statistic *= 12.0/(numDataPoints*numDataPoints + numDataPoints);
    h_statistic -= 3.0*(numDataPoints+1);

    // Correct statistic value for tiebreaks.
    tie_correction /= (numDataPoints*numDataPoints*numDataPoints - numDataPoints);
    tie_correction = 1 - tie_correction;
    h_statistic /= tie_correction;

    // Compute eta-squared effect size if valid.
    double eta_squared_value;
    if (numDataPoints - category_groups <= 0)
        eta_squared_value = std::numeric_limits<double>::quiet_NaN();
    else 
        eta_squared_value = (h_statistic-category_groups +1)/(numDataPoints - category_groups);

    // Compute Pvalue with survival function of chi-squared distribution if well-defined.
    double pvalue;
    if(category_groups < 2)
        pvalue = std::numeric_limits<double>::quiet_NaN();
    else 
    {
        const int num_dofs = category_groups-1;
        boost::math::chi_squared dist(num_dofs);
        if (h_statistic < 0 || std::isnan(h_statistic) || std::isinf(h_statistic))
            return get_nans_kruskal();
        else
            pvalue = 1 - cdf(dist, h_statistic);
    }
    return std::make_tuple(pvalue, h_statistic, eta_squared_value);
}

std::map<std::string, DataMatrix> statistics::kruskal_wallis_with_nans(const DataMatrix& cat_data,
    const DataMatrix& cont_data, const std::vector<int>& category_groups, double na_value, 
    const std::set<std::string>& return_types)
{
    const int num_cat_variables = cat_data.rows();
    const int num_cont_variables = cont_data.rows();
    const int num_samples = cat_data.cols();
    // Check how many and which output matrices have to be created.
    bool compute_pval = false;
    bool compute_h = false;
    bool compute_eta2 = false;
    DataMatrix pvalues(0,0), h_stat(0,0), eta2(0,0);
    
    if (return_types.count("p_unadjusted"))
    {
        pvalues = DataMatrix(num_cat_variables, num_cont_variables);
        compute_pval = true;
    }
    if (return_types.count("H"))
    {
        h_stat = DataMatrix(num_cat_variables, num_cont_variables);
        compute_h = true;
    }
    if (return_types.count("eta2"))
    {
        eta2 = DataMatrix(num_cat_variables, num_cont_variables);
        compute_eta2 = true;
    }

    #pragma omp parallel for 
    for (int iCont = 0; iCont < num_cont_variables; ++iCont)
    {
        // Sort and rank values of continuous variable first.
        std::vector<int> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b){return cont_data(iCont, a)<cont_data(iCont, b);});
        size_t vecCounter = 0;
        int rankCounter = 1;
        
        vec2d row_map(num_samples, std::vector<int>());
        // Iterate through whole sorted indices vector.
        while (vecCounter < indices.size())
        {
            // If value is NA, simply ignore it.
            if (cont_data(iCont, indices[vecCounter])==na_value)
            {
                // Vector element counter increases only by one.
                ++vecCounter;
            }
            else // Value is non-NA.
            {
                int forward_counter = 1;
                // Count how many subsequent values in ordered sequence are equal, i.e. ties.
                while((vecCounter+forward_counter) < indices.size() && cont_data(iCont,indices[vecCounter]) == cont_data(iCont,indices[vecCounter+forward_counter]))
                {
                    ++forward_counter;
                }
                // Store indices of aggregated rank bins.
                for (size_t iP = vecCounter; iP < vecCounter+forward_counter; ++iP)
                {
                    row_map[rankCounter-1].push_back(indices[iP]);
                }
                // Rank counter increases by number of values that have been averaged over.
                rankCounter += forward_counter;
                // Vector index counter increases by number of values that have been averaged over.
                vecCounter += forward_counter;
            }
        }

        for (int iCat = 0; iCat < num_cat_variables; ++iCat)
        {
            std::tuple<double, double, double> results = pairwise_nan_kruskal(cat_data, 
                iCat, category_groups[iCat], row_map, na_value);
            if (compute_pval)
                pvalues(iCat, iCont) = std::get<0>(results);
            if (compute_h)
                h_stat(iCat, iCont) = std::get<1>(results);
            if (compute_eta2)
                eta2(iCat, iCont) = std::get<2>(results);
        }
    }    

    // Combine desired return matrices into one output.
    std::map<std::string, DataMatrix> output;
    if (compute_pval)
        output.insert(std::make_pair("p_unadjusted", pvalues));
    if (compute_h)
        output.insert(std::make_pair("H", h_stat));
    if (compute_eta2)
        output.insert(std::make_pair("eta2", eta2));
    
    return output;
}
