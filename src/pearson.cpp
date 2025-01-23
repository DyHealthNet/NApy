#include <stats.hpp>
#include <cmath>

// Computes NAN-aware Pearson correlation of two vectors.
std::pair<double, double> pairwise_nan_correlation(const DataMatrix& data, 
    int row1, int row2, double na_value)
{
    double x_sum = 0.0;
    double y_sum = 0.0;
    double x_squared_sum = 0.0;
    double y_squared_sum = 0.0;
    double x_times_y = 0.0;
    int numNonNAs = 0;

    const int numCols = data.cols();
    // Compute averages of both vectors on non-NAN matched subvectors.
    for (int iC = 0; iC < numCols; ++iC)
    {
        const double entryX = data(row1, iC);
        const double entryY = data(row2, iC);
        if (entryX != na_value && entryY != na_value)
        {
            x_sum += entryX;
            y_sum += entryY;
            x_squared_sum += entryX*entryX;
            y_squared_sum += entryY*entryY;
            x_times_y += entryX*entryY;
            numNonNAs += 1;
        }
    }

    const double avgX = x_sum / numNonNAs;
    const double avgY = y_sum / numNonNAs;

    // Compute nominator and denominator based on simplified Pearson correlation formula.
    const double nominator = x_times_y - avgY*x_sum - avgX*y_sum + numNonNAs*avgX*avgY;
    const double denominatorX = x_squared_sum - 2*avgX*x_sum + numNonNAs*avgX*avgX;
    const double denominatorY = y_squared_sum - 2*avgY*y_sum + numNonNAs*avgY*avgY;

    const double corr = nominator/std::sqrt(denominatorX*denominatorY);

    if (numNonNAs <= 1)
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

    if (numNonNAs <= 2)
        return std::make_pair(corr, std::numeric_limits<double>::quiet_NaN());

    // Compute P-value based on Beta-Distribution CDF evaluation.
    const double distParam = static_cast<double>(numNonNAs)/2.0 - 1;
    boost::math::beta_distribution<> beta_dist(distParam, distParam);
    const double scaled_corr = (-1.0*abs(corr)+1)/2.0;
    if (scaled_corr < 0 || std::isnan(scaled_corr))
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    const double pval = 2.0*cdf(beta_dist, scaled_corr);

    return std::make_pair(corr, pval);
}

// NAN-aware Pearson Correlation on given DataMatrix.
std::pair<DataMatrix, DataMatrix> statistics::pearson_with_nans(const DataMatrix& data, 
    double na_value)
{
    const int num_rows = data.rows();
    DataMatrix correlations(num_rows, num_rows);
    DataMatrix pvalues(num_rows, num_rows);

    // Compute each pairwise correlation of rows.
    #pragma omp parallel for
    for (int iR = 0; iR < num_rows; ++iR)
    {
        for (int jR = iR; jR < num_rows; ++jR)
        {
            std::pair<double, double> results = pairwise_nan_correlation(data, iR, jR, na_value);
            correlations(iR, jR) = std::get<0>(results);
            correlations(jR, iR) = std::get<0>(results);
            double pvalue = std::get<1>(results);
            pvalues(iR, jR) = pvalue;
            pvalues(jR, iR) = pvalue;
        }
    }

    return std::make_pair(correlations, pvalues);
}