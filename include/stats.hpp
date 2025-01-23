#include <omp.h>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/normal.hpp>
#include <utility>
#include <numeric>
#include <matrix.hpp>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <iostream>
#include <limits>

#pragma once

using vec3d = std::vector<std::vector<std::vector<int>>>;
using vec2d = std::vector<std::vector<int>>;

namespace statistics
{

    // NAN-aware Pearson Correlation on given DataMatrix.
    std::pair<DataMatrix, DataMatrix> pearson_with_nans(const DataMatrix& data, double na_value);
    
    // NAN-aware Spearman Correlation on given DataMatrix.
    std::pair<DataMatrix, DataMatrix> spearman_with_nans(const DataMatrix& data, double na_value);
    
    // NAN-aware Chi-squared test on independence.
    std::map<std::string, DataMatrix> chi_squared_with_nans(const DataMatrix& data,
            const std::vector<int>& category_groups, double na_value, 
            const std::set<std::string>& effect_size);

    // NAN-aware one-way ANOVA test.
    std::map<std::string, DataMatrix> anova_with_nans(const DataMatrix& cat_data, 
        const DataMatrix& cont_data, const std::vector<int>& category_groups,
        double na_value, const std::set<std::string>& effect_size);

    // NAN-aware Kruskal-Wallis test.
    std::map<std::string, DataMatrix> kruskal_wallis_with_nans(const DataMatrix& cat_data, 
        const DataMatrix& cont_data, const std::vector<int>& category_groups, 
        double na_value, const std::set<std::string>& effect_size);

    // NAN-aware Student's and Welch's T-test.
    std::map<std::string, DataMatrix> ttest(const DataMatrix& bin_data, const DataMatrix& cont_data,
        double na_value, const std::set<std::string>& effect_size, bool use_welch);

    // NAN-aware Mann-Whitney-U test.
    std::map<std::string, DataMatrix> mwu_with_nans(const DataMatrix& bin_data, 
        const DataMatrix& cont_data, double na_value, const std::set<std::string>& return_types, 
        const std::string& mode);
}