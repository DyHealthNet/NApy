import numpy as np
from numba import njit, prange, set_num_threads
from numba.core import types
from numba.typed import List
import scipy as sc

@njit(parallel=True, fastmath=False, nogil=True)
def pearson_numba(data : np.ndarray, nan_value : float, num_threads : int):
    """
    Compute numba-optimized Pearson Correlation with pairwise NAN-removal.
    Args:
        data: Data matrix with rows as features and columns as samples.
        nan_value: Float value representing missing value.
        num_threads: Number of numba-threads to use in parallel computation.

    Returns: Pairwise R-squared values matrix and P-values matrix.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_rows = data.shape[0]
    num_samples = data.shape[1]
    # Initialize output matrices.
    corr_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    pvalue_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    # Compute all pairwise Pearson correlations of variables (i.e. rows).
    for row1 in prange(num_rows):
        for row2 in range(row1, num_rows):
            # Compute Pearson Correlation for given pair of variables.
            sum1 = sum2 = 0.0
            sq_sum1 = sq_sum2 = 0.0
            one_times_two = 0.0
            num_values = 0
            for col in range(num_samples):
                # Only consider data if both values are non-NA.
                if data[row1, col] != nan_value and data[row2, col] != nan_value:
                    sum1 = sum1 + data[row1, col]
                    sum2 = sum2 + data[row2, col]
                    sq_sum1 = sq_sum1 + data[row1, col] * data[row1, col]
                    sq_sum2 = sq_sum2 + data[row2, col] * data[row2, col]
                    one_times_two = one_times_two + data[row1, col] * data[row2, col]
                    num_values = num_values + 1

            # Check invalid cases.
            if num_values <= 1:
                pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = np.nan
                corr_matrix[row1, row2] = corr_matrix[row2, row1] = np.nan
            else:
                avg1 = sum1 / num_values
                avg2 = sum2 / num_values
                nominator = one_times_two - avg2 * sum1 - avg1 * sum2 + num_values * avg1 * avg2
                denom1 = sq_sum1 - 2 * avg1 * sum1 + num_values * avg1 * avg1
                denom2 = sq_sum2 - 2 * avg2 * sum2 + num_values * avg2 * avg2
                corr = nominator / np.sqrt(denom1 * denom2)
                if num_values == 2:
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = np.nan
                    corr_matrix[row1, row2] = corr_matrix[row2, row1] = corr
                else:
                    # Compute two-sided Pv-alues based on beta distribution.
                    a = b = num_values / 2.0 - 1
                    x = (-1.0 * np.abs(corr) + 1.0) / 2.0
                    # Clip value into [0,1] interval.
                    if x < 0.0:
                        x = 0.0
                    if x > 1.0:
                        x = 1.0
                    pvalue = 2.0 * sc.special.betainc(a, b, x)
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = pvalue
                    corr_matrix[row1, row2] = corr_matrix[row2, row1] = corr

    return corr_matrix, pvalue_matrix

@njit(parallel=True, fastmath=False, nogil=True)
def spearman_numba(data : np.ndarray, nan_value : float, num_threads : int):
    """
    Computes numba-optimized Spearman rank correlation with pairwise removal of NAN-values.

        data: Data matrix with rows as features and columns as samples.
        nan_value: Float value representing missing value.
        num_threads: Number of numba-threads to use in parallel computation.

    Returns: Pairwise rho-values matrix and P-values matrix.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_rows = data.shape[0]
    num_samples = data.shape[1]
    # Initialize output matrices.
    corr_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    pvalue_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    row_rank_bins = np.zeros((num_rows, num_samples, num_samples), dtype=np.int32)

    # Pre-sort and pre-process rows in parallel.
    for iR in prange(num_rows):
        sorted_indices = np.argsort(data[iR])
        # Iterate thru sorted indices array and put data elements into rank-based bins.
        rank_counter = 1
        array_counter = 0
        row_bins = np.full((num_samples, num_samples), -1, dtype=np.int32)
        while array_counter < num_samples:
            # If data is marked as NA, simply ignore it.
            if data[iR, sorted_indices[array_counter]] == nan_value:
                array_counter = array_counter + 1
            else: # Value is not NA.
                rank_bin = np.full((1, num_samples), -1, dtype=np.int32)
                forward_counter = 1
                # Count how many subsequent data points are equal, i.e. how many subsequent ties there are.
                while array_counter+forward_counter < num_samples and data[iR, sorted_indices[array_counter]] == data[iR, sorted_indices[array_counter+forward_counter]]:
                    forward_counter = forward_counter + 1
                # Put ties into same rank bin.
                rank_bin_counter = 0
                for iP in range(array_counter, array_counter+forward_counter):
                    rank_bin[0, rank_bin_counter] = sorted_indices[iP]
                    rank_bin_counter = rank_bin_counter+1
                # Save per-rank bin into per-row bins.
                row_bins[rank_counter-1, :] = rank_bin
                # Update rank counter and array counter based on number of detected ties.
                rank_counter = rank_counter + forward_counter
                array_counter = array_counter + forward_counter

        # Save per-row rank bins into global data structure.
        row_rank_bins[iR, :] = row_bins

    # Use pre-computed row-wise rank bins to efficiently compute pairwise Spearman correlations and P-values.
    for row1 in prange(num_rows):
        for row2 in range(row1, num_rows):
            number_non_nas = 0

            ### Iterate thru rank bins of first row and account for NAs in second row.
            updated_ranks1 = np.full((1, num_samples), -1.0, dtype=np.float64)
            rank_sum1 = 0.0
            rank_sum_squared1 = 0.0
            subtract_right = 0
            for iR in range(num_samples):
                # Rank indices start at 1.
                rank = iR+1
                row_rank_bin = row_rank_bins[row1, iR, :]
                # Check if rank actually contains a bin.
                if row_rank_bin[0] != -1:
                    # Iterate thru rank-bin and sort out NA indices of second row.
                    rank_subset = np.full((1, num_samples), -1, dtype=np.int32)
                    row_rank_size = 0
                    rank_subset_size = 0
                    for el in row_rank_bin:
                        if el == -1:
                            break
                        row_rank_size = row_rank_size + 1
                        if data[row2, el] != nan_value:
                            rank_subset[0, rank_subset_size] = el
                            rank_subset_size = rank_subset_size + 1
                            number_non_nas = number_non_nas + 1

                    # Number of elements in current rank bin that have been deleted.
                    subtract_extra = row_rank_size - rank_subset_size
                    # Update tie-ranks as average over ascending sequence of integers.
                    if rank_subset_size > 0:
                        start_rank = rank - subtract_right
                        average = 1.0/rank_subset_size * (rank_subset_size*start_rank + 0.5*rank_subset_size*(rank_subset_size-1))
                        # Set averaged rank to all remaining elements in subset of rank bin.
                        for iS in range(rank_subset_size):
                            index = rank_subset[0, iS]
                            updated_ranks1[0, index] = average
                            rank_sum1 = rank_sum1 + average
                            rank_sum_squared1 = rank_sum_squared1 + average*average

                    subtract_right = subtract_right + subtract_extra

            ### Iterate thru rank bins of second row and account for NAs in first row.
            subtract_right = 0
            rank_sum2 = 0.0
            rank_sum_squared2 = 0.0
            first_times_second = 0.0
            for iR in range(num_samples):
                row_rank_bin = row_rank_bins[row2, iR, :]
                rank = iR+1
                # Check if rank bin actually contains an element.
                if row_rank_bin[0] != -1:
                    # Iterate thru rank-bin and sort out NA indices of first row.
                    rank_subset = np.full((1, num_samples), -1, dtype=np.int32)
                    row_rank_size = 0
                    rank_subset_size = 0
                    for el in row_rank_bin:
                        if el == -1:
                            break
                        row_rank_size = row_rank_size + 1
                        if data[row1, el] != nan_value:
                            rank_subset[0, rank_subset_size] = el
                            rank_subset_size = rank_subset_size + 1

                    # Number of elements in current rank bin that have been deleted.
                    subtract_extra = row_rank_size - rank_subset_size
                    # Update tie-ranks as average over ascending sequence of integers.
                    if rank_subset_size > 0:
                        start_rank = rank - subtract_right
                        average = 1.0 / rank_subset_size * (
                                    rank_subset_size * start_rank + 0.5 * rank_subset_size * (rank_subset_size - 1))
                        # Set averaged rank to all remaining elements in subset of rank bin.
                        for iS in range(rank_subset_size):
                            index = rank_subset[0, iS]
                            rank_sum2 = rank_sum2 + average
                            rank_sum_squared2 = rank_sum_squared2 + average * average
                            first_times_second = first_times_second + average * updated_ranks1[0, index]

                    subtract_right = subtract_right + subtract_extra

            # Compute Pearson Correlation on rank-transformed data.
            average1 = rank_sum1 / number_non_nas
            average2 = rank_sum2 / number_non_nas
            nominator = first_times_second - average2*rank_sum1 - average1*rank_sum2 + number_non_nas*average1*average2
            variance1 = rank_sum_squared1 - 2.0*average1*rank_sum1 + number_non_nas*average1*average1
            variance2 = rank_sum_squared2 - 2.0*average2*rank_sum2 + number_non_nas*average2*average2
            correlation = nominator / np.sqrt(variance1*variance2)

            # Check edge cases where correlation value is undefined.
            if number_non_nas < 2:
                corr_value = np.nan
                pvalue = np.nan
            elif number_non_nas < 3:
                corr_value = correlation
                pvalue = np.nan
            else:
                corr_value = correlation
                # Compute two-sided pvalue based on Student-t-distribution.
                num_dofs = number_non_nas - 2.0
                t_value = correlation * np.sqrt(num_dofs / ((correlation+1.0)*(1.0-correlation)))
                pvalue = 2.0 * sc.special.stdtr(num_dofs, -1.0*np.abs(t_value))

            corr_matrix[row1, row2] = corr_value
            corr_matrix[row2, row1] = corr_value
            pvalue_matrix[row1, row2] = pvalue
            pvalue_matrix[row2, row1] = pvalue

    return corr_matrix, pvalue_matrix

@njit(parallel=True, fastmath=False, nogil=True)
def chi2_numba(data : np.ndarray, categories_per_var : np.ndarray, nan_value : int,
               is_pvalue : bool, is_chi2 : bool, is_phi : bool, 
               is_cramers : bool, num_threads : int):
    """
    Computes numba-optimized Chi-squared tests with pairwise removal of NAN-values.

        data: Data matrix with rows as features and columns as samples' values.
        nan_value: Float value representing missing value.
        num_threads: Number of numba-threads to use in parallel computation.
        categories_per_var: Number of categories contained in each row of the data matrix.
        is_pvalue: Whether or not return pvalue matrix.
        is_chi2: Whether or not to return chi2 statistics matrix.
        is_phi: Whether or not to return phi effect size matrix.
        is_cramer: Whether or not to return Cramer's V effect size matrix.

    Returns: Dictionary storing specified return data matrices.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_rows = data.shape[0]
    num_samples = data.shape[1]
    #pvalue_matrix = np.zeros((0,0), dtype=np.float64)
    #effect_matrix = np.zeros((0, 0), dtype=np.float64)
    #phi_matrix = np.zeros((0, 0), dtype=np.float64)
    #cramers_matrix = np.zeros((0, 0), dtype=np.float64)
    # Initialize output matrices.
    if is_pvalue:
        pvalue_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    if is_chi2:
        effect_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    if is_phi:
        phi_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    if is_cramers:
        cramers_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)
    
    # Cast data matrix from float to int to represent categories.
    data = data.astype(np.int32)

    for row1 in prange(num_rows):
        for row2 in range(row1, num_rows):
            number_non_nas = 0.0
            num_cats1 = categories_per_var[row1]
            num_cats2 = categories_per_var[row2]
            # Initialize data structures to count occurences of categories.
            row1_frequencies = np.full(num_cats1, 0, dtype=np.uint32)
            row2_frequencies = np.full(num_cats2, 0, dtype=np.uint32)
            cont_table = np.full((num_cats1, num_cats2), 0, dtype=np.uint32)
            # Iterate over both rows simulateneously and count category frequencies for contingency table.
            for iS in range(num_samples):
                cat1 = data[row1, iS]
                cat2 = data[row2, iS]
                if cat1 != nan_value and cat2 != nan_value:
                    number_non_nas = number_non_nas + 1.0
                    row1_frequencies[cat1] = row1_frequencies[cat1] + 1
                    row2_frequencies[cat2] = row2_frequencies[cat2] + 1
                    cont_table[cat1, cat2] = cont_table[cat1, cat2] + 1

            if number_non_nas == 0.0:
                if is_chi2:
                    effect_matrix[row1, row2] = effect_matrix[row2, row1] = np.nan
                if is_pvalue:
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = np.nan
                if is_phi:
                    phi_matrix[row1, row2] = phi_matrix[row2, row1] = np.nan
                if is_cramers:
                    cramers_matrix[row1, row2] = cramers_matrix[row2, row1] = np.nan
                continue

            # Compute Chi-squared test statistic by iterating over contingency table.
            statistic_value = 0.0
            empty_cat = False
            for iR in range(num_cats1):
                for iC in range(num_cats2):
                    expected_freq = row1_frequencies[iR] * row2_frequencies[iC] / number_non_nas
                    # Check if given category actually exists in both rows.
                    if expected_freq == 0.0:
                        empty_cat = True
                    else:
                        actual_freq = cont_table[iR, iC]
                        statistic_value = statistic_value + (actual_freq - expected_freq)*(actual_freq - expected_freq) / expected_freq
            # Check if chi-squared statistic is well-defined, i.e.
            if empty_cat:
                if is_chi2:
                    effect_matrix[row1, row2] = effect_matrix[row2, row1] = np.nan
                if is_pvalue:
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = np.nan
                if is_phi:
                    phi_matrix[row1, row2] = phi_matrix[row2, row1] = np.nan
                if is_cramers:
                    cramers_matrix[row1, row2] = cramers_matrix[row2, row1] = np.nan
                continue
            
            # Compute phi effect size.
            phi_value = np.sqrt(statistic_value / number_non_nas)

            # Check if Pvalue computation and Cramer's V are well-defined.
            if num_cats1 == 1 or num_cats2 == 1:                
                if is_chi2:
                    effect_matrix[row1, row2] = effect_matrix[row2, row1] = statistic_value
                if is_pvalue:
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = np.nan
                if is_phi:
                    phi_matrix[row1, row2] = phi_matrix[row2, row1] = phi_value
                if is_cramers:
                    cramers_matrix[row1, row2] = cramers_matrix[row2, row1] = np.nan
            else:
                # Two-sided P-value computation based on chi-squared distribution.
                num_dofs = (num_cats1-1.0)*(num_cats2-1.0)
                pvalue = sc.special.chdtrc(num_dofs, statistic_value)
                cramer_value = np.sqrt(statistic_value / (number_non_nas * np.min(np.array([num_cats1-1, num_cats2-1]))))
                if is_chi2:
                    effect_matrix[row1, row2] = effect_matrix[row2, row1] = statistic_value
                if is_pvalue:
                    pvalue_matrix[row1, row2] = pvalue_matrix[row2, row1] = pvalue
                if is_phi:
                    phi_matrix[row1, row2] = phi_matrix[row2, row1] = phi_value
                if is_cramers:
                    cramers_matrix[row1, row2] = cramers_matrix[row2, row1] = cramer_value

    return pvalue_matrix, effect_matrix, phi_matrix, cramers_matrix

@njit(parallel=True, fastmath=False, nogil=True)
def kruskal_wallis_numba(cat_data : np.ndarray, cont_data : np.ndarray, nan_value : float, category_groups : np.ndarray,
                         num_threads : int, compute_pvalue : bool, compute_h : bool, compute_np2 : bool):
    """
    Computes numba-optimized Kruskal-Wallis test with pairwise removal of NAN-values.

        data: Data matrix with rows as features and columns as samples.
        nan_value: Float value representing missing value.
        num_threads: Number of numba-threads to use in parallel computation.

    Returns: Unadjusted P-value matrix and desired selection of effect size matrices.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_cat_variables = cat_data.shape[0]
    num_cont_variables = cont_data.shape[0]
    num_samples = cat_data.shape[1]
    cat_nan_value = int(nan_value)
    # Initialize output matrices.
    if compute_pvalue:
        pvalue_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)
    if compute_h:
        h_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)
    if compute_np2:
        np2_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)

    # Cast data matrix from float to int to represent categories.
    cat_data = cat_data.astype(np.int32)
    row_rank_bins = np.zeros((num_cont_variables, num_samples, num_samples), dtype=np.int32)

    # Pre-sort and pre-process rows in parallel.
    for iR in prange(num_cont_variables):
        sorted_indices = np.argsort(cont_data[iR])
        # Iterate thru sorted indices array and put data elements into rank-based bins.
        rank_counter = 1
        array_counter = 0
        row_bins = np.full((num_samples, num_samples), -1, dtype=np.int32)
        while array_counter < num_samples:
            # If data is marked as NA, simply ignore it.
            if cont_data[iR, sorted_indices[array_counter]] == nan_value:
                array_counter = array_counter + 1
            else: # Value is not NA.
                rank_bin = np.full((1, num_samples), -1, dtype=np.int32)
                forward_counter = 1
                # Count how many subsequent data points are equal, i.e. how many subsequent ties there are.
                while array_counter+forward_counter < num_samples and cont_data[iR, sorted_indices[array_counter]] == cont_data[iR, sorted_indices[array_counter+forward_counter]]:
                    forward_counter = forward_counter + 1
                # Put ties into same rank bin.
                rank_bin_counter = 0
                for iP in range(array_counter, array_counter+forward_counter):
                    rank_bin[0, rank_bin_counter] = sorted_indices[iP]
                    rank_bin_counter = rank_bin_counter+1
                # Save per-rank bin into per-row bins.
                row_bins[rank_counter-1, :] = rank_bin
                # Update rank counter and array counter based on number of detected ties.
                rank_counter = rank_counter + forward_counter
                array_counter = array_counter + forward_counter

        # Save per-row rank bins into global data structure.
        row_rank_bins[iR, :] = row_bins

    for cont_row in prange(num_cont_variables):
        for cat_row in range(num_cat_variables):
            number_non_nas = 0
            ### Iterate thru rank bins of first row and account for NAs in second row.
            group_rank_sums = np.full(category_groups[cat_row], 0.0, dtype=np.float64)
            group_sizes = np.full(category_groups[cat_row], 0, dtype=np.int32)
            tie_correction = 0.0
            subtract_right = 0
            for iR in range(num_samples):
                # Rank indices start at 1.
                rank = iR+1
                row_rank_bin = row_rank_bins[cont_row, iR, :]
                # Check if rank actually contains a bin.
                if row_rank_bin[0] != -1:
                    # Iterate thru rank-bin and sort out NA indices of second row.
                    rank_subset = np.full((1, num_samples), -1, dtype=np.int32)
                    row_rank_size = 0
                    rank_subset_size = 0
                    for el in row_rank_bin:
                        if el == -1:
                            break
                        row_rank_size = row_rank_size + 1
                        if cat_data[cat_row, el] != cat_nan_value:
                            rank_subset[0, rank_subset_size] = el
                            rank_subset_size = rank_subset_size + 1
                            number_non_nas = number_non_nas + 1
                            group_sizes[cat_data[cat_row, el]] = group_sizes[cat_data[cat_row, el]] + 1

                    # Number of elements in current rank bin that have been deleted.
                    subtract_extra = row_rank_size - rank_subset_size
                    # Update tie-ranks as average over ascending sequence of integers.
                    if rank_subset_size > 0:
                        start_rank = rank - subtract_right
                        average = 1.0/rank_subset_size * (rank_subset_size*start_rank + 0.5*rank_subset_size*(rank_subset_size-1))
                        # Set averaged rank to all remaining elements in subset of rank bin.
                        for iS in range(rank_subset_size):
                            index = rank_subset[0, iS]
                            group_rank_sums[cat_data[cat_row, index]] = group_rank_sums[cat_data[cat_row, index]] + average
                        # Compute correction term for denominator in H statistic in case of ties.
                        if rank_subset_size > 1:
                            tie_correction = tie_correction + (rank_subset_size*rank_subset_size*rank_subset_size - rank_subset_size)
                    subtract_right = subtract_right + subtract_extra

            # Compute H statistic value by aggregating per-category rank sums.
            h_statistic = 0.0
            is_empty_category = False
            for iC in range(category_groups[cat_row]):
                if group_sizes[iC] == 0:
                    # H statistic is not well-defined and return NAs on all outputs.
                    is_empty_category = True
                else:
                    h_statistic = h_statistic + group_rank_sums[iC]*group_rank_sums[iC]/group_sizes[iC]

            if is_empty_category:
                if compute_pvalue:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                if compute_h:
                    h_matrix[cat_row, cont_row] = np.nan
                if compute_np2:
                    np2_matrix[cat_row, cont_row] = np.nan
                continue

            h_statistic = h_statistic * 12.0/(number_non_nas * number_non_nas + number_non_nas)
            h_statistic = h_statistic - 3.0*(number_non_nas + 1)
            # Correct H statistic for ties.
            tie_correction = tie_correction / (number_non_nas * number_non_nas * number_non_nas - number_non_nas)
            tie_correction = 1 - tie_correction
            h_statistic = h_statistic / tie_correction

            # Compute eta-squared effect size if valid.
            if compute_np2:
                if number_non_nas - category_groups[cat_row] <= 0:
                    np2_matrix[cat_row, cont_row] = np.nan
                else:
                    np2_matrix[cat_row, cont_row] = (h_statistic-category_groups[cat_row]+1)/(number_non_nas - category_groups[cat_row])

            if compute_pvalue:
                if category_groups[cat_row] < 2:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                else:
                    num_dofs = category_groups[cat_row] - 1.0
                    pvalue_matrix[cat_row, cont_row] = sc.special.chdtrc(num_dofs, h_statistic)

            if compute_h:
                h_matrix[cat_row, cont_row] = h_statistic

    return pvalue_matrix, h_matrix, np2_matrix

@njit(parallel=True, fastmath=False, nogil=True)
def ttest_numba(bin_data : np.ndarray, cont_data : np.ndarray,  nan_value : float, compute_pvalues : bool,
                compute_t : bool, compute_cohens : bool, use_welch : bool, num_threads : int):
    """
    Computes pairwise two-sample ttest for all combinations of variables in binary and cotinuous data.
    Args:
        bin_data: Data matrix with rows representing variables and columns indicating category membership (0/1).
        cont_data: Data matrix with rows representing continuous variables and columns 
        nan_value: Value representing missing data.
        compute_pvalues: Whether or not to compute and return pvalues.
        compute_t : Wheter or not to compute and return t-statistic values.
        compute_cohens: Whether or not compute and return cohens_d effect size values.
        use_welch: Whether to use Student's t-test or Welch's t-test.
        num_threads : Number of threads to use in parallel computation.

    Returns:
        Pairwise return type matrix and P-values matrix.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_bin_variables = bin_data.shape[0]
    num_cont_variables = cont_data.shape[0]
    num_samples = bin_data.shape[1]
    bin_nan_value = int(nan_value)

    if compute_pvalues:
        pvalue_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)
    if compute_t:
        t_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)
    if compute_cohens:
        cohen_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)

    # Cast data matrix from float to int to represent categories.
    bin_data = bin_data.astype(np.int32)

    for cont_row in prange(num_cont_variables):
        for bin_row in range(num_bin_variables):
            # Compute per-group sums and count statistics.
            group_sums = np.zeros(2, dtype=np.float64)
            group_sums_squared = np.zeros(2, dtype=np.float64)
            group_counts = np.zeros(2, dtype=np.int32)

            for iS in range(num_samples):
                entry_bin = bin_data[bin_row, iS]
                entry_cont = cont_data[cont_row, iS]
                # Both data points need to be non-missing values.
                if entry_bin != bin_nan_value and entry_cont != nan_value:
                    group_sums[entry_bin] = group_sums[entry_bin] + entry_cont
                    group_sums_squared[entry_bin] = group_sums_squared[entry_bin] + entry_cont*entry_cont
                    group_counts[entry_bin] = group_counts[entry_bin] + 1

            # Check if category contains no more elements after NA removal.
            if group_counts[0] <= 1 or group_counts[1] <= 1:
                if compute_pvalues:
                    pvalue_matrix[bin_row, cont_row] = np.nan
                if compute_t:
                    t_matrix[bin_row, cont_row] = np.nan
                if compute_cohens:
                    cohen_matrix[bin_row, cont_row] = np.nan
                continue
            # Check if total number of non-NA elements is still at least three.
            if group_counts[0] + group_counts[1] < 3:
                if compute_pvalues:
                    pvalue_matrix[bin_row, cont_row] = np.nan
                if compute_t:
                    t_matrix[bin_row, cont_row] = np.nan
                if compute_cohens:
                    cohen_matrix[bin_row, cont_row] = np.nan
                continue

            # Compute means and variances of categories.
            means = np.zeros(2, dtype=np.float64)
            variance = np.zeros(2, dtype=np.float64)
            means[0] = group_sums[0] / group_counts[0]
            means[1] = group_sums[1] / group_counts[1]
            variance[0] = (1.0 / (group_counts[0] - 1)) * (
                        group_sums_squared[0] - 2 * means[0] * group_sums[0] + group_counts[0] * means[0] * means[0])
            variance[1] = (1.0 / (group_counts[1] - 1)) * (
                        group_sums_squared[1] - 2 * means[1] * group_sums[1] + group_counts[1] * means[1] * means[1])

            # Perform Student's-t-test.
            if use_welch == False:
                pooled_standard_dev = (group_counts[0]-1)*variance[0] + (group_counts[1]-1)*variance[1]
                pooled_standard_dev = pooled_standard_dev / (group_counts[0] + group_counts[1] - 2)
                pooled_standard_dev = np.sqrt(pooled_standard_dev)

                if pooled_standard_dev == 0.0:
                    if compute_pvalues:
                        pvalue_matrix[bin_row, cont_row] = np.nan
                    if compute_t:
                        t_matrix[bin_row, cont_row] = np.nan
                    if compute_cohens:
                        cohen_matrix[bin_row, cont_row] = np.nan
                    continue
                else:
                    cohens_value = (means[0]-means[1])/pooled_standard_dev
                    statistic_value = (means[0] - means[1])
                    statistic_value = statistic_value / (pooled_standard_dev * np.sqrt(1.0 / group_counts[0] + 1.0 / group_counts[1]))
                    dofs = group_counts[0] + group_counts[1] - 2
            else: # Perform Welch's t-test for non-equal variances.
                statistic_value = means[0] - means[1]
                std_err = np.zeros(2, dtype=np.float64)
                std_err[0] = np.sqrt(variance[0] / group_counts[0])
                std_err[1] = np.sqrt(variance[1] / group_counts[1])
                statistic_value = statistic_value / np.sqrt(std_err[0] * std_err[0] + std_err[1] * std_err[1])

                dof_nom = (variance[0] / group_counts[0] + variance[1] / group_counts[1]) * (
                            variance[0] / group_counts[0] + variance[1] / group_counts[1])
                dof_denom = variance[0] * variance[0] / (group_counts[0] * group_counts[0] * (group_counts[0] - 1))
                dof_denom = dof_denom + variance[1] * variance[1] / (group_counts[1] * group_counts[1] * (group_counts[1] - 1))

                dofs = np.round(dof_nom / dof_denom)
                if variance[0]+variance[1] == 0.0:
                    cohens_value = np.nan
                else:
                    cohens_value = (means[0] - means[1]) / np.sqrt((variance[0] + variance[1]) / 2.0)

            # Compute P-value based on survival function of t-distribution.
            pvalue = 2.0 * (0.5 * sc.special.betainc(0.5 * dofs, 0.5, dofs / (dofs + np.abs(statistic_value)**2)))
            if compute_pvalues:
                pvalue_matrix[bin_row, cont_row] = pvalue
            if compute_t:
                t_matrix[bin_row, cont_row] = statistic_value
            if compute_cohens:
                cohen_matrix[bin_row, cont_row] = cohens_value

    return pvalue_matrix, t_matrix, cohen_matrix

@njit
def compute_exact_pvalue(n : int, m : int, u : int):
    """
    Helper function for computing exact P-values based on efficient dynammic programming approach as presented by
    Andreas Loeffler.
    Args:
        n: Size of first population.
        m: Size of second population.
        u: Value of U statistic.

    Returns:
        Exact P-value corresponding to value of U statistic.
    """
    u = n*m - u
    sigma = np.full(u+1, 0, dtype=np.int32)

    for d in range(1, n+1):
        for i in range(d, u+1, d):
            sigma[i] = sigma[i] + d

    for d in range(m+1, m+n+1):
        for i in range(d, u+1, d):
            sigma[i] = sigma[i] - d

    p = np.full(u+1, 0, dtype=np.int64)
    p[0] = 1
    for a in range(1, u+1):
        for i in range(0, a):
            p[a] = p[a] + p[i]*sigma[a-i]
        p[a] = p[a] / a

    normalized_p = np.full(u+1, 0.0, dtype=np.float64)
    total = sc.special.binom(np.float64(n+m), np.float64(n))
    for iP in range(0, u+1):
        normalized_p[iP] = p[iP] / total

    p_cum = np.full(u+1, 0.0, dtype=np.float64)
    p_cum[0] = normalized_p[0]
    for iP in range(1, u+1):
        p_cum[iP] = p_cum[iP-1] + normalized_p[iP]

    pvalue = 2*(1.0 - p_cum[u] + normalized_p[u])
    if pvalue > 1.0:
        return 1.0
    else:
        return pvalue


@njit(parallel=True, fastmath=False, nogil=True)
def mann_whitney_numba(bin_data : np.ndarray, cont_data : np.ndarray,  nan_value : float, compute_pvalues : bool,
                compute_u : bool, compute_r : bool, num_threads : int, mode : int):
    """
        Computes pairwise MWU tests for all combinations of variables in binary and continuous data.
        Args:
            bin_data: Data matrix with rows representing variables and columns indicating category membership (0/1).
            cont_data: Data matrix with rows representing continuous variables and columns
            nan_value: Value representing missing data.
            compute_pvalues: Whether or not to compute and return pvalues.
            compute_u : Wheter or not to compute and return U-statistic values.
            compute_r: Whether or not compute and return Pearson's r effect size values.
            num_threads : Number of threads to use in parallel computation.
            mode: Which mode to use for computation of U statistic (0=='auto', 1=='exact', 2=='asymptotic').
        Returns:
            Pairwise return type and P-values matrices.
        """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_bin_variables = bin_data.shape[0]
    num_cont_variables = cont_data.shape[0]
    num_samples = bin_data.shape[1]
    bin_nan_value = int(nan_value)

    if compute_pvalues:
        pvalue_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)
    if compute_u:
        u_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)
    if compute_r:
        r_matrix = np.zeros((num_bin_variables, num_cont_variables), dtype=np.float64)

    # Cast data matrix from float to int to represent categories.
    bin_data = bin_data.astype(np.int32)

    row_rank_bins = np.zeros((num_cont_variables, num_samples, num_samples), dtype=np.int32)

    # Pre-sort and pre-process rows in parallel.
    for iR in prange(num_cont_variables):
        sorted_indices = np.argsort(cont_data[iR])
        # Iterate thru sorted indices array and put data elements into rank-based bins.
        rank_counter = 1
        array_counter = 0
        row_bins = np.full((num_samples, num_samples), -1, dtype=np.int32)
        while array_counter < num_samples:
            # If data is marked as NA, simply ignore it.
            if cont_data[iR, sorted_indices[array_counter]] == nan_value:
                array_counter = array_counter + 1
            else:  # Value is not NA.
                rank_bin = np.full((1, num_samples), -1, dtype=np.int32)
                forward_counter = 1
                # Count how many subsequent data points are equal, i.e. how many subsequent ties there are.
                while array_counter + forward_counter < num_samples and cont_data[iR, sorted_indices[array_counter]] == cont_data[iR, sorted_indices[array_counter + forward_counter]]:
                    forward_counter = forward_counter + 1
                # Put ties into same rank bin.
                rank_bin_counter = 0
                for iP in range(array_counter, array_counter + forward_counter):
                    rank_bin[0, rank_bin_counter] = sorted_indices[iP]
                    rank_bin_counter = rank_bin_counter + 1
                # Save per-rank bin into per-row bins.
                row_bins[rank_counter - 1, :] = rank_bin
                # Update rank counter and array counter based on number of detected ties.
                rank_counter = rank_counter + forward_counter
                array_counter = array_counter + forward_counter

        # Save per-row rank bins into global data structure.
        row_rank_bins[iR, :] = row_bins

    for cont_row in prange(num_cont_variables):
        for bin_row in range(num_bin_variables):
            number_non_nas = 0
            ### Iterate thru rank bins of first row and account for NAs in second row.
            group_rank_sums = np.full(2, 0.0, dtype=np.float64)
            group_sizes = np.full(2, 0, dtype=np.int32)
            tie_correction = 0.0
            exist_ties = False
            subtract_right = 0
            for iR in range(num_samples):
                # Rank indices start at 1.
                rank = iR+1
                row_rank_bin = row_rank_bins[cont_row, iR, :]
                # Check if rank actually contains a bin.
                if row_rank_bin[0] != -1:
                    # Iterate thru rank-bin and sort out NA indices of second row.
                    rank_subset = np.full((1, num_samples), -1, dtype=np.int32)
                    row_rank_size = 0
                    rank_subset_size = 0
                    for el in row_rank_bin:
                        if el == -1:
                            break
                        row_rank_size = row_rank_size + 1
                        if bin_data[bin_row, el] != bin_nan_value:
                            rank_subset[0, rank_subset_size] = el
                            rank_subset_size = rank_subset_size + 1
                            number_non_nas = number_non_nas + 1
                            group_sizes[bin_data[bin_row, el]] = group_sizes[bin_data[bin_row, el]] + 1

                    # Number of elements in current rank bin that have been deleted.
                    subtract_extra = row_rank_size - rank_subset_size
                    # Update tie-ranks as average over ascending sequence of integers.
                    if rank_subset_size > 0:
                        start_rank = rank - subtract_right
                        average = 1.0/rank_subset_size * (rank_subset_size*start_rank + 0.5*rank_subset_size*(rank_subset_size-1))
                        # Set averaged rank to all remaining elements in subset of rank bin.
                        for iS in range(rank_subset_size):
                            index = rank_subset[0, iS]
                            group_rank_sums[bin_data[bin_row, index]] = group_rank_sums[bin_data[bin_row, index]] + average
                        # Compute correction term for U statistic in case of ties.
                        if rank_subset_size > 1:
                            exist_ties = True
                            tie_correction = tie_correction + (rank_subset_size*rank_subset_size*rank_subset_size - rank_subset_size)
                    subtract_right = subtract_right + subtract_extra

            # Compute U statistic value by aggregating per-category rank sums.
            larger_index = 1
            if group_rank_sums[0]>=group_rank_sums[1]:
                larger_index = 0
            n1 = group_sizes[larger_index]
            n2 = group_sizes[1 - larger_index]
            R1 = group_rank_sums[larger_index]
            U = n1*n2 + 0.5*n1*(n1+1) - R1

            # Check for empty category.
            if n1 == 0 or n2 == 0:
                if compute_pvalues:
                    pvalue_matrix[bin_row, cont_row] = np.nan
                if compute_u:
                    u_matrix[bin_row, cont_row] = np.nan
                if compute_r:
                    r_matrix[bin_row, cont_row] = np.nan
                continue

            # Check and determine chosen computation mode.
            is_exact_possible = False
            if (not exist_ties) and (group_sizes[0] < 8 or group_sizes[1] < 8):
                is_exact_possible = True
            # Compute z-value from U statistic value.
            mu = 0.5*n1*n2
            n = n1+n2
            sigma = np.sqrt((1.0/12.0)*n1*n2*((n+1) - tie_correction/(n*(n-1))))
            z_value = (U - mu) / sigma

            # Compute Pearson's r effect size if well-defined.
            if sigma == 0.0:
                r_effect = np.nan
            else:
                r_effect = z_value / np.sqrt(n)

            # Compute P-value based on asymptotic mode if desired and possible.
            if mode == 2 or (mode == 0 and not is_exact_possible):
                # Compute two-sided P-values from standard normal distribution.
                pvalue = 2.0 * (0.5 * sc.special.erfc(np.abs(z_value) / np.sqrt(2)))
                if compute_pvalues:
                    pvalue_matrix[bin_row, cont_row] = pvalue
                if compute_u:
                    u_matrix[bin_row, cont_row] = U
                if compute_r:
                    r_matrix[bin_row, cont_row] = r_effect
            elif mode == 1 or (mode == 0 and is_exact_possible):
                # Compute exact P-values based on efficient dynamic programming approach presented by Andreas Loeffler.
                rounded_u = int(U)
                pvalue = compute_exact_pvalue(n1, n2, rounded_u)
                if compute_pvalues:
                    pvalue_matrix[bin_row, cont_row] = pvalue
                if compute_u:
                    u_matrix[bin_row, cont_row] = rounded_u
                if compute_r:
                    r_matrix[bin_row, cont_row] = r_effect

    return pvalue_matrix, u_matrix, r_matrix

@njit(parallel=True, fastmath=False, nogil=True)
def anova_numba(cat_data : np.ndarray, cont_data : np.ndarray, category_groups : np.ndarray,
                nan_value : float, compute_pvalue : bool, compute_f : bool,
                compute_np2 : bool, num_threads : int):
    """
    Compute pairwise one-way-ANOVA test for all combinations of categorical and continuous variables.
    Args:
        cat_data: Data matrix with rows representing variables and columns indicating category membership.
        cont_data: Data matrix with rows representing continuous variables.
        category_groups: Array storing number of categories per cotegorical variable (i.e. row).
        nan_value: Value representing missing data.
        num_threads : Number of threads to use in parallel computation.
        compute_pvalue: Whether or not to return Pvalue matrix.
        compute_f : Whether or not to return F statistic value.
        compute_np2: Whether or not to return np2 effect size matrix.

    Returns:
        Pairwise return type matrix and P-values matrix.
    """
    # Set number of desired numba threads.
    set_num_threads(num_threads)
    # Extract number of features and samples.
    num_cat_variables = cat_data.shape[0]
    num_cont_variables = cont_data.shape[0]
    num_samples = cat_data.shape[1]
    cat_nan_value = int(nan_value)
    # Initialize output matrices.
    if compute_pvalue:
        pvalue_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)
    if compute_f:
        f_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)
    if compute_np2:
        np2_matrix = np.zeros((num_cat_variables, num_cont_variables), dtype=np.float64)

    # Cast data matrix from float to int to represent categories.
    cat_data = cat_data.astype(np.int32)

    for cont_row in prange(num_cont_variables):
        for cat_row in range(num_cat_variables):
            # Compute in-group and out-group sums and squared sums.
            num_categories = category_groups[cat_row]
            group_sums = np.zeros(num_categories, dtype=np.float64)
            group_counts = np.zeros(num_categories, dtype=np.int32)
            total_sum = 0.0
            total_sum_squared = 0.0
            total_count = 0
            for iS in range(num_samples):
                entry_cat = cat_data[cat_row, iS]
                entry_cont = cont_data[cont_row, iS]
                if entry_cat != cat_nan_value and entry_cont != nan_value:
                    group_sums[entry_cat] = group_sums[entry_cat] + entry_cont
                    group_counts[entry_cat] = group_counts[entry_cat] + 1
                    total_sum = total_sum + entry_cont
                    total_sum_squared = total_sum_squared + entry_cont*entry_cont
                    total_count = total_count + 1

            # Check if any elements remain after NA removal.
            if total_count == 0:
                if compute_pvalue:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                if compute_f:
                    f_matrix[cat_row, cont_row] = np.nan
                if compute_np2:
                    np2_matrix[cat_row, cont_row] = np.nan
                continue

            # Aggregate per-group sums and sums of squares.
            ss_total = total_sum_squared - (total_sum * total_sum) / total_count
            ss_bg = 0.0
            is_empty_category = False
            for iC in range(num_categories):
                # If empty category is present after NA removal, test results are undefined.
                if group_counts[iC] == 0:
                    is_empty_category = True
                else:
                    ss_bg = ss_bg + group_sums[iC]*group_sums[iC] / group_counts[iC]

            if is_empty_category:
                if compute_pvalue:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                if compute_f:
                    f_matrix[cat_row, cont_row] = np.nan
                if compute_np2:
                    np2_matrix[cat_row, cont_row] = np.nan
                continue

            ss_bg = ss_bg - total_sum*total_sum / total_count
            # Compute within-group variances as difference of totals and between groups.
            ss_wg = ss_total - ss_bg

            # Compute eta-squared effect size if valid.
            if ss_bg + ss_wg != 0.0:
                np2_value = ss_bg / (ss_bg + ss_wg)
            else:
                np2_value = np.nan

            # Check if input number of categories and number of categories after NA removal are still valid.
            if num_categories <= 1 or total_count - num_categories <= 0:
                if compute_pvalue:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                if compute_f:
                    f_matrix[cat_row, cont_row] = np.nan
                if compute_np2:
                    np2_matrix[cat_row, cont_row] = np.nan
                continue

            # Compute DOFs and F statistic value.
            dof_bg = float(num_categories - 1)
            dof_wg = float(total_count - num_categories)

            ms_bg = ss_bg / dof_bg
            ms_wg = ss_wg / dof_wg

            # Check edge case of within-group variances being zero.
            if ms_wg <= 0.0:
                # If all group sums are equal, return NA, else return np.inf (as implemented in scipy).
                if np.all(group_sums == group_sums[0]):
                    if compute_pvalue:
                        pvalue_matrix[cat_row, cont_row] = np.nan
                    if compute_f:
                        f_matrix[cat_row, cont_row] = np.nan
                    if compute_np2:
                        np2_matrix[cat_row, cont_row] = np.nan
                    continue
                else:
                    if compute_pvalue:
                        pvalue_matrix[cat_row, cont_row] = 0.0
                    if compute_f:
                        f_matrix[cat_row, cont_row] = np.infty
                    if compute_np2:
                        np2_matrix[cat_row, cont_row] = np2_value
                    continue

            statistic = ms_bg / ms_wg

            # Check validity of F statistic value.
            if statistic < 0.0:
                if compute_pvalue:
                    pvalue_matrix[cat_row, cont_row] = np.nan
                if compute_f:
                    f_matrix[cat_row, cont_row] = np.nan
                if compute_np2:
                    np2_matrix[cat_row, cont_row] = np.nan
                continue
            
            # Compute corresponding P-value from F distribution CDF.
            pvalue = sc.special.fdtrc(dof_bg, dof_wg, statistic)
            if compute_pvalue:
                pvalue_matrix[cat_row, cont_row] = pvalue
            if compute_f:
                f_matrix[cat_row, cont_row] = statistic
            if compute_np2:
                np2_matrix[cat_row, cont_row] = np2_value

    return pvalue_matrix, f_matrix, np2_matrix
