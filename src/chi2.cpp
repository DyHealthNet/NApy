#include <stats.hpp>

std::tuple<double, double, double, double> get_nan_four_tuple()
{
    return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
}

std::tuple<double, double, double, double> pairwise_nan_chi_squared(const DataMatrix& data, int row1, int row2,
        int num_categories1, int num_categories2, int na_value)
{
    const int num_samples = data.cols();
    int num_non_nas = 0;

    std::vector<int> var1_frequencies(num_categories1, 0);
    std::vector<int> var2_frequencies(num_categories2, 0);
    std::vector<std::vector<int>> cont_table(num_categories1, std::vector<int>(num_categories2, 0));
    // Iterate over both rows simulateneously and count frequencies for contingency table.
    for (int iC = 0; iC < num_samples; ++iC)
    {
        const int entry1 = static_cast<int>(data(row1, iC));
        const int entry2 = static_cast<int>(data(row2, iC));
        if (entry1 != na_value && entry2 != na_value)
        {
            ++num_non_nas;
            var1_frequencies[entry1] += 1;
            var2_frequencies[entry2] += 1;
            cont_table[entry1][entry2] += 1;
        }
    }

    if (num_non_nas == 0)
        return get_nan_four_tuple();
    // Compute chi-squared test statistic by summing over contingency table and computing
    // expected frequencies.
    double statistics_value = 0.0;
    for (int iR = 0; iR < num_categories1; ++iR)
    {
        for (int iC = 0; iC < num_categories2; ++iC)
        {
            const double expected_freq = var1_frequencies[iR] * var2_frequencies[iC] / static_cast<double>(num_non_nas);
            if (expected_freq == 0.0)
            {
                return get_nan_four_tuple();
            }
            const int actual_freq = cont_table[iR][iC];
            statistics_value += (actual_freq - expected_freq)*(actual_freq - expected_freq) / expected_freq;
        }
    }

    // Compute phi effect size.
    const double phi_value = std::sqrt(statistics_value / num_non_nas);

    // If only one category present, P-value and Cramers' V are undefined.
    if (num_categories1 == 1 || num_categories2 == 1)
        return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), statistics_value, 
                              phi_value, std::numeric_limits<double>::quiet_NaN());
    
    // Compute Cramer's V effect size.
    const double cramers_v = std::sqrt(statistics_value / (num_non_nas*std::min(num_categories1-1, num_categories2-1)));

    // Compute two-sided P-value based on chi2 statistic.
    const int num_dofs = (num_categories1 - 1) * (num_categories2 - 1);
    boost::math::chi_squared dist(num_dofs);
    if (std::isnan(statistics_value) || statistics_value < 0)
        return get_nan_four_tuple();
    const double pvalue = 1 - cdf(dist, statistics_value);

    return std::make_tuple(pvalue, statistics_value, phi_value, cramers_v);
}

std::map<std::string, DataMatrix> statistics::chi_squared_with_nans(const DataMatrix& data,
        const std::vector<int>& category_groups, double na_value, const std::set<std::string>& return_types)
{
    const int num_variables = data.rows();
    // Check how many and which output matrices have to be created.
    bool compute_pval = false;
    bool compute_chi2 = false;
    bool compute_phi = false;
    bool compute_cramers = false;
    DataMatrix pvalues(0,0), chi2(0,0), phi(0,0), cramers(0,0);
    
    if (return_types.count("p_unadjusted"))
    {
        pvalues = DataMatrix(num_variables, num_variables);
        compute_pval = true;
    }
    if (return_types.count("chi2"))
    {
        chi2 = DataMatrix(num_variables, num_variables);
        compute_chi2 = true;
    }
    if (return_types.count("phi"))
    {
        phi = DataMatrix(num_variables, num_variables);
        compute_phi = true;
    }
    if (return_types.count("cramers_v"))
    {
        cramers = DataMatrix(num_variables, num_variables);
        compute_cramers = true;
    }

    #pragma omp parallel for 
    for (int iR = 0; iR < num_variables; ++iR)
    {
        for (int jR = iR; jR < num_variables; ++jR)
        {
            std::tuple<double, double, double, double> results = pairwise_nan_chi_squared(data, iR, jR, 
                category_groups[iR], category_groups[jR], na_value);
            if (compute_pval)
            {
                pvalues(iR, jR) = std::get<0>(results);
                pvalues(jR, iR) = std::get<0>(results);
            }
            if (compute_chi2)
            {
                chi2(iR, jR) = std::get<1>(results);
                chi2(jR, iR) = std::get<1>(results);
            }
            if (compute_phi)
            {
                phi(iR, jR) = std::get<2>(results);
                phi(jR, iR) = std::get<2>(results);
            }
            if (compute_cramers)
            {
                cramers(iR, jR) = std::get<3>(results);
                cramers(jR, iR) = std::get<3>(results);
            }
        }
    }

    // Combine desired return matrices into one output.
    std::map<std::string, DataMatrix> output;
    if (compute_pval)
        output.insert(std::make_pair("p_unadjusted", pvalues));
    if (compute_chi2)
        output.insert(std::make_pair("chi2", chi2));
    if (compute_phi)
        output.insert(std::make_pair("phi", phi));
    if (compute_cramers)
        output.insert(std::make_pair("cramers_v", cramers));
    
    return output;
}
