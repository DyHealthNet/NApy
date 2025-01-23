#include <stats.hpp>
#include <cmath>

std::tuple<double, double, double> get_ttest_nans()
{
    return std::make_tuple(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
}

std::tuple<double, double, double> pairwise_nan_ttest(const DataMatrix& bin_data, const DataMatrix& cont_data,
    int rowCat, int rowCont, double na_value, bool use_welch)
{
    const int num_samples = bin_data.cols();
    
    // Create per-group sums and count statistics.
    std::array<double, 2> group_sums = {0.0, 0.0};
    std::array<double, 2> group_sums_squared = {0.0, 0.0};
    std::array<int, 2> group_counts = {0, 0};
    
    for (int iCol = 0; iCol < num_samples; ++iCol)
    {
        const double entryBin = bin_data(rowCat, iCol);
        const double entryCont = cont_data(rowCont, iCol);
        if (entryBin != na_value && entryCont != na_value)
        {
            const int category = static_cast<int>(entryBin);
            group_sums[category] += entryCont;
            group_sums_squared[category] += entryCont*entryCont;
            ++group_counts[category];
        }
    }

    // Check if one of the two categories contains zero elements after NA removal.
    if (group_counts[0] <= 1 || group_counts[1] <= 1)
        return get_ttest_nans();

    // Total number of non-NA elements needs to be at least three.
    if (group_counts[0] + group_counts[1] < 3)
        return get_ttest_nans();
    
    // Compute means and standard deviations.
    std::array<double, 2> means;
    means[0] = group_sums[0] / group_counts[0];
    means[1] = group_sums[1] / group_counts[1];
    std::array<double, 2> variance;
    variance[0] = (1.0/(group_counts[0]-1)) * (group_sums_squared[0] - 2*means[0]*group_sums[0] + group_counts[0]*means[0]*means[0]);
    variance[1] = (1.0/(group_counts[1]-1)) * (group_sums_squared[1] - 2*means[1]*group_sums[1] + group_counts[1]*means[1]*means[1]);
    
    double statistic_value, cohens_value;
    int dofs;
    if (use_welch == false)
    {   // Perform Student's t-test.
        double pooled_standard_dev = (group_counts[0]-1)*variance[0] + (group_counts[1]-1)*variance[1];
        pooled_standard_dev /= (group_counts[0] + group_counts[1] - 2);
        pooled_standard_dev = std::sqrt(pooled_standard_dev);

        if (pooled_standard_dev == 0.0)
            return get_ttest_nans();
            
        statistic_value = (means[0] - means[1]);
        statistic_value /= pooled_standard_dev * std::sqrt(1.0/group_counts[0] + 1.0/group_counts[1]);
        dofs = group_counts[0] + group_counts[1] - 2; 
        cohens_value =  (means[0]-means[1])/pooled_standard_dev;
    }
    else // Use Welch's t-test in case of non-equal variances.
    {
        statistic_value = means[0]-means[1];
        std::array<double, 2> std_err;
        std_err[0] = std::sqrt(variance[0]/group_counts[0]);
        std_err[1] = std::sqrt(variance[1]/group_counts[1]);
        statistic_value /= std::sqrt(std_err[0]*std_err[0] + std_err[1]*std_err[1]);

        double dof_nom = (variance[0]/group_counts[0] + variance[1]/group_counts[1])*(variance[0]/group_counts[0] + variance[1]/group_counts[1]);
        double dof_denom = variance[0]*variance[0] / (group_counts[0]*group_counts[0]*(group_counts[0]-1));
        dof_denom += variance[1]*variance[1] / (group_counts[1]*group_counts[1]*(group_counts[1]-1));
        
        dofs = static_cast<int>(std::round(dof_nom / dof_denom));

        if (variance[0]+variance[1] == 0.0)
            cohens_value = std::numeric_limits<double>::quiet_NaN();
        else
            cohens_value = (means[0] - means[1]) / std::sqrt((variance[0]+variance[1])/2.0);
    }

    double pvalue;
    // Compute Pvalue based on evaluating survival function of t-distribution.
    if (dofs > 0)
    {
        boost::math::students_t dist(dofs);
        if (std::isnan(abs(statistic_value)))
            return get_ttest_nans();
        pvalue = 2.0 * (1-cdf(dist, abs(statistic_value)));
    }
    else
        pvalue = std::numeric_limits<double>::quiet_NaN();
    
    return std::make_tuple(pvalue, statistic_value, cohens_value);
}

std::map<std::string, DataMatrix> statistics::ttest(const DataMatrix& bin_data, const DataMatrix& cont_data,
        double na_value, const std::set<std::string>& return_types, bool use_welch)
{
    const int num_bin_variables = bin_data.rows();
    const int num_cont_variables = cont_data.rows();
    // Check how many and which output matrices have to be created.
    bool compute_pval = false;
    bool compute_t = false;
    bool compute_cohens = false;
    DataMatrix pvalues(0,0), t_stat(0,0), cohens(0,0);
    
    if (return_types.count("p_unadjusted"))
    {
        pvalues = DataMatrix(num_bin_variables, num_cont_variables);
        compute_pval = true;
    }
    if (return_types.count("t"))
    {
        t_stat = DataMatrix(num_bin_variables, num_cont_variables);
        compute_t = true;
    }
    if (return_types.count("cohens_d"))
    {
        cohens = DataMatrix(num_bin_variables, num_cont_variables);
        compute_cohens = true;
    }

    #pragma omp parallel for 
    for (int iCat = 0; iCat < num_bin_variables; ++iCat)
    {
        for (int iCont = 0; iCont < num_cont_variables; ++iCont)
        {
            std::tuple<double, double, double> results = pairwise_nan_ttest(bin_data, cont_data, 
                iCat, iCont, na_value, use_welch);
            if (compute_pval)
                pvalues(iCat, iCont) = std::get<0>(results);
            if (compute_t)
                t_stat(iCat, iCont) = std::get<1>(results);
            if (compute_cohens)
                cohens(iCat, iCont) = std::get<2>(results);
        }
    }    

    // Combined desired return matrices into one output.
    std::map<std::string, DataMatrix> output;
    if (compute_pval)
        output.insert(std::make_pair("p_unadjusted", pvalues));
    if (compute_t)
        output.insert(std::make_pair("t", t_stat));
    if (compute_cohens)
        output.insert(std::make_pair("cohens_d", cohens));
    
    return output;
}