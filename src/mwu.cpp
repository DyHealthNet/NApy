#include <stats.hpp>
#include <boost/math/special_functions/beta.hpp>

std::tuple<double, double, double> get_nans_mwu()
{
    return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
}

// Helper function for computing binomial coefficients.
long binom_coeff(int n, int k)
{
    return std::lround(1 / ((n + 1) * boost::math::beta(n - k + 1, k + 1)));
}

// Efficient computation of exact pvalues by Andreas Loeffler, translated into C++ from
// https://aakinshin.net/posts/mw-loeffler/.
double compute_exact_pvalue(int n, int m, int u)
{
    u = n*m - u;
    std::vector<int> sigma(u+1);

    for (int d=1; d<=n; ++d)
    {
        for (int i=d; i<=u; i+=d)
        {
            sigma[i] += d;
        }
    }

    for (int d=m+1; d <= m+n; ++d)
    {
        for (int i=d; i <= u; i+=d)
        {
            sigma[i] -= d;
        }
    }

    std::vector<long> p(u+1);
    p[0] = 1;
    for (int a=1; a <= u; ++a)
    {
        for (int i=0; i < a; ++i)
        {
            p[a] += p[i] * sigma[a-i];
        }
        p[a] /= a;
    }

    // Normalize each frequency by maximal number of possible frequencies.
    std::vector<double> normalized_p(u+1);
    const double total = static_cast<double>(binom_coeff(n+m, n));
    for (int iP = 0; iP < u+1; ++iP)
    {
        normalized_p[iP] = p[iP] / total;
    }
    // Compute cumulative sum of normalized frequencies.
    std::vector<double> p_cum(u+1);
    p_cum[0] = normalized_p[0];
    for (int iP = 1; iP < u+1; ++iP)
    {
        p_cum[iP] = p_cum[iP-1] + normalized_p[iP];
    }

    double pvalue = 1.0 - p_cum[u] + normalized_p[u];
    pvalue = std::min(1.0, 2*pvalue);
    return pvalue;
}

std::tuple<double, double, double> pairwise_nan_mwu(const DataMatrix& bin_data, int iBin, 
    const vec2d& rowRanks, double na_value, const std::string& mode)
{
    // Compute number of elements per category and rank sums.
    //int numDataPoints = 0;

    std::array<double, 2> groupRankSums = {0.0, 0.0};
    std::array<int, 2> groupSizes = {0,0};
    bool exist_ties = false;
    double tie_correction = 0.0;

    // Iterate through rank maps of each row and account for NAs in other vector.
    int subtractRight = 0;
    for (size_t iV = 0; iV < rowRanks.size(); ++iV)
    {
        const int rank = iV+1;
        // Check if rank actually contains values, if not skip this iteration.
        if (rowRanks[iV].size()>0)
        {
            // Iterate through "bin" and sort out NA indices of second vector.
            int subsetSize = 0;
            for (const auto& index : rowRanks[iV])
            {
                if (bin_data(iBin, index)!= na_value)
                {
                    ++subsetSize;
                    //++numDataPoints;
                    groupSizes[bin_data(iBin, index)] += 1;
                }
            }
            // Again iterate over "bin" and update rank sums of corresponding categories.
            if (subsetSize > 0)
            {
                const double startRank = rank - subtractRight;
                const double average = 1.0/subsetSize * (subsetSize*startRank + 0.5*subsetSize*(subsetSize-1));
                for (const auto& index : rowRanks[iV])
                {
                    if (bin_data(iBin, index)!= na_value)
                    {
                        // Add averaged rank to rank sum of corresponding group.
                        groupRankSums[bin_data(iBin, index)] += average;
                    }
                }
                // Compute correction term for denominator in H statistic in case of ties.
                if (subsetSize > 1)
                {
                    exist_ties = true;
                    tie_correction += subsetSize*subsetSize*subsetSize - subsetSize;
                }
            }

            // Number of elements in current container that were deleted.
            const int subtract_extra = rowRanks[iV].size() - subsetSize;
            subtractRight += subtract_extra;
        }
    }

    // Compute U-statistic value.
    int larger_index = (groupRankSums[0]>=groupRankSums[1]) ? 0 : 1 ;
    const int n1 = groupSizes[larger_index];
    const int n2 = groupSizes[1-larger_index];
    const double R1 = groupRankSums[larger_index];
    const double U = n1*n2 + 0.5*n1*(n1+1) - R1;

    // Edge case checks.
    if (n1 == 0 || n2 == 0)
        return get_nans_mwu();

    // Compute pvalue value based on chosen mode.
    bool is_exact_possible = false;
    if (!exist_ties && (groupSizes[0]<8 || groupSizes[1]<8))
        is_exact_possible = true;

    // Compute z-value from U-statistic value.
    const double mu = 0.5*n1*n2;
    const double n = n1 + n2;
    const double sigma = std::sqrt((1.0/12.0)*n1*n2*((n+1) - tie_correction/(n*(n-1))));
    const double z_value = (U - mu) / sigma;

    double r_effect;
    if (sigma == 0.0)
        r_effect = std::numeric_limits<double>::quiet_NaN();
    else
        r_effect = std::abs(z_value) / std::sqrt(n);

    if (mode == "asymptotic" || (mode == "auto" && !is_exact_possible))
    {
        // Compute two-sided pvalues from standard normal distribution.
        boost::math::normal dist(0,1);
        if (std::isnan(abs(z_value)))
            return get_nans_mwu();
        const double pvalue = 2.0 * (1-cdf(dist, abs(z_value)));
        return std::make_tuple(pvalue, U, r_effect);
        
    }
    else if (mode == "exact" || (mode == "auto" && is_exact_possible))
    {
        // Use efficient dynamic programming approach by Andreas Loeffler to compute exact pvalues.
        const int rounded_u = static_cast<int>(std::round(U));
        const double pvalue = compute_exact_pvalue(n1, n2, rounded_u);
        return std::make_tuple(pvalue, rounded_u, r_effect);
    }
    else 
    {
        std::cerr << "MWU: unknown input mode, skipping and returning NA..." << std::endl;
        return get_nans_mwu();
    }

}

std::map<std::string, DataMatrix> statistics::mwu_with_nans(const DataMatrix& bin_data, 
        const DataMatrix& cont_data, double na_value, const std::set<std::string>& return_types,
        const std::string& mode)
{
    const int num_bin_variables = bin_data.rows();
    const int num_cont_variables = cont_data.rows();
    const int num_samples = bin_data.cols();
    // Check how many and which output matrices have to be created.
    bool compute_pval = false;
    bool compute_u = false;
    bool compute_r = false;
    DataMatrix pvalues(0,0), u_stat(0,0), r_effect(0,0);
    
    if (return_types.count("p_unadjusted"))
    {
        pvalues = DataMatrix(num_bin_variables, num_cont_variables);
        compute_pval = true;
    }
    if (return_types.count("U"))
    {
        u_stat = DataMatrix(num_bin_variables, num_cont_variables);
        compute_u = true;
    }
    if (return_types.count("r"))
    {
        r_effect = DataMatrix(num_bin_variables, num_cont_variables);
        compute_r = true;
    }

    #pragma omp parallel for 
    for (int iCont = 0; iCont < num_cont_variables; ++iCont)
    {
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
                std::vector<int> rank_bin;
                int forward_counter = 1;
                // Count how many subsequent values in ordered sequence are equal, i.e. ties.
                while((vecCounter+forward_counter) < indices.size() && cont_data(iCont,indices[vecCounter]) == cont_data(iCont,indices[vecCounter+forward_counter]))
                {
                    ++forward_counter;
                }
                // Store indices of aggregated rank bins.
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

        for (int iBin = 0; iBin < num_bin_variables; ++iBin)
        {
            std::tuple<double, double, double> results = pairwise_nan_mwu(bin_data, 
                iBin, row_map, na_value, mode);
            if (compute_pval)
                pvalues(iBin, iCont) = std::get<0>(results);
            if (compute_u)
                u_stat(iBin, iCont) = std::get<1>(results);
            if (compute_r)
                r_effect(iBin, iCont) = std::get<2>(results);
        }
    }    

    // Combined desired return matrices into one output.
    std::map<std::string, DataMatrix> output;
    if (compute_pval)
        output.insert(std::make_pair("p_unadjusted", pvalues));
    if (compute_u)
        output.insert(std::make_pair("U", u_stat));
    if (compute_r)
        output.insert(std::make_pair("r", r_effect));
    
    return output;
}
