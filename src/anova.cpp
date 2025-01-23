#include <stats.hpp>
#include <cmath>

std::tuple<double, double, double> get_nan_three_tuple()
{
    return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
}

std::tuple<double, double, double> pairwise_nan_anova(const DataMatrix& cat_data, const DataMatrix& cont_data,
    int rowCat, int rowCont, int numCats, double na_value)
{
    const int num_samples = cat_data.cols();
    
    // Create per-group sums and count statistics.
    std::vector<double> group_sums(numCats, 0.0);
    std::vector<int> group_counts(numCats, 0);
    double total_sum = 0.0;
    double total_sum_squared = 0.0;
    int total_count = 0;
    
    for (int iCol = 0; iCol < num_samples; ++iCol)
    {
        const double entryCat = cat_data(rowCat, iCol);
        const double entryCont = cont_data(rowCont, iCol);
        if (entryCat != na_value && entryCont != na_value)
        {
            const int category = static_cast<int>(entryCat);
            group_sums[category] += entryCont;
            ++group_counts[category];
            total_sum += entryCont;
            total_sum_squared += entryCont*entryCont;
            total_count += 1;
        }
    }

    if (total_count == 0)
    {
        return get_nan_three_tuple();
    }
    const double ss_total = total_sum_squared - (total_sum*total_sum)/total_count;

    // Aggregate per-group sums and sum of squares.
    double ss_bg = 0.0;
    for (int iC = 0; iC < numCats; ++iC)
    {
        if (group_counts[iC]==0)
        {
            return get_nan_three_tuple();
        }
        ss_bg += group_sums[iC]*group_sums[iC] / group_counts[iC];
    }
    ss_bg -= total_sum*total_sum / total_count;

    // Compute within group variances as difference of totals and between groups.
    const double ss_wg = ss_total - ss_bg;

    // Compute eta-squared effect size if valid.
    double np2_value;
    if (ss_bg + ss_wg != 0.0)
        np2_value = ss_bg / (ss_bg + ss_wg);
    else
        np2_value = std::numeric_limits<double>::quiet_NaN();

    // Check well-definedness of DOFs.
    if (numCats <= 1 || total_count - numCats <= 0)
    {
        return get_nan_three_tuple();
    }
    const int dof_bg = numCats - 1;
    const int dof_wg = total_count - numCats;

    const double ms_bg = ss_bg / dof_bg;
    const double ms_wg = ss_wg / dof_wg;
    // Compute F statistic value if valid.
    double statistic;
    double pvalue;
    if (ms_wg <= 0.0)
    {
        // Check if all group sums are equal.
        if (std::adjacent_find(group_sums.begin(), group_sums.end(), std::not_equal_to<>()) == group_sums.end())
        {
            return get_nan_three_tuple();
        }
        else // Group sums are not equal, return inf for F distribution value and zero P-value.
        {
            statistic = std::numeric_limits<double>::infinity();
            pvalue = 0.0;
            return std::make_tuple(pvalue, statistic, np2_value);
        }
    }
    else
    {
        statistic = ms_bg / ms_wg;
    }

    // Compute corresponding Pvalue from F distribution.
    // First check validity for F distribution input.
    if (statistic < 0.0 || std::isnan(statistic))
    {
        return get_nan_three_tuple();
    }
    boost::math::fisher_f dist(dof_bg, dof_wg);
    pvalue = 1 - cdf(dist, statistic);

    return std::make_tuple(pvalue, statistic, np2_value);
}

std::map<std::string, DataMatrix> statistics::anova_with_nans(const DataMatrix& cat_data,
    const DataMatrix& cont_data, const std::vector<int>& category_groups,
    double na_value, const std::set<std::string>& return_types)
{
    const int num_cat_variables = cat_data.rows();
    const int num_cont_variables = cont_data.rows();
    // Check how many and which output matrices have to be created.
    bool compute_pval = false;
    bool compute_f = false;
    bool compute_np2 = false;
    DataMatrix pvalues(0,0), f_stat(0,0), np2(0,0);
    
    if (return_types.count("p_unadjusted"))
    {
        pvalues = DataMatrix(num_cat_variables, num_cont_variables);
        compute_pval = true;
    }
    if (return_types.count("F"))
    {
        f_stat = DataMatrix(num_cat_variables, num_cont_variables);
        compute_f = true;
    }
    if (return_types.count("np2"))
    {
        np2 = DataMatrix(num_cat_variables, num_cont_variables);
        compute_np2 = true;
    }

    #pragma omp parallel for 
    for (int iCat = 0; iCat < num_cat_variables; ++iCat)
    {
        for (int iCont = 0; iCont < num_cont_variables; ++iCont)
        {
            std::tuple<double, double, double> results = pairwise_nan_anova(cat_data, cont_data, 
                iCat, iCont, category_groups[iCat], na_value);
            if (compute_pval)
                pvalues(iCat, iCont) = std::get<0>(results);
            if (compute_f)
                f_stat(iCat, iCont) = std::get<1>(results);
            if (compute_np2)
                np2(iCat, iCont) = std::get<2>(results);
        }
    }   

    // Combined desired return matrices into one output.
    std::map<std::string, DataMatrix> output;
    if (compute_pval)
        output.insert(std::make_pair("p_unadjusted", pvalues));
    if (compute_f)
        output.insert(std::make_pair("F", f_stat));
    if (compute_np2)
        output.insert(std::make_pair("np2", np2));
    
    return output;
}
