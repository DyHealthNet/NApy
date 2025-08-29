import libnapy
import libnapy_numba
import numpy as np
import scipy as sc
import torch
import pandas as pd

def _adjust_pvalues_bonferroni(pval_matrix : np.array, ignore_diag : bool) -> np.array:
    """
    Corrects Pvalue matrix for multiple testing by using Bonferroni correction.
    Args:
        pval_matrix: Pvalue matrix with Pvalues to be corrected.
        ignore_diag : Whether or not Pvalues on diagonal correspond to "self-tests" of variables, and should hence be
            ignored in multiple testing correction.
    Returns:
        Corrected Pvalue matrix. Number of performed tests ignores NAs in the input matrix and "self-tests" on
        the diagonal of the input matrix. Pvalues on the diagonal are therefore set to NA in the resulting output.
    """
    submatrix = np.array([])
    num_tests = 0
    mask = np.array([], dtype=bool)
    if ignore_diag:
        # Count number of non-NA entries in Pvalue matrix by ignoring "self-tests" on the diagonal and accounting for
        # symmetry of Pvalue matrix.
        diag_mask = np.eye(pval_matrix.shape[0], dtype=bool)
        pval_matrix[diag_mask] = np.nan
        na_mask = ~np.isnan(pval_matrix)
        mask = ~diag_mask & na_mask
        submatrix = pval_matrix[mask]
        num_tests = len(submatrix)/2
    else:
        # Count number of non-NA entries in Pvalue matrix.
        mask = ~np.isnan(pval_matrix)
        submatrix = pval_matrix[mask]
        num_tests = len(submatrix)

    # Multiply each entry in matrix by number of performed tests and take min of element and 1.
    submatrix = submatrix * num_tests
    corrected_array = np.clip(submatrix, a_min=0.0, a_max=1.0)
    pval_matrix[mask] = corrected_array
    return pval_matrix

def _adjust_pvalues_fdr_control(pval_matrix : np.array, method : str, ignore_diag : bool) -> np.array:
    """
    Correct Pvalue matrix for multiple testing by using Benjamini-Hochberg correction.
    Args:
        pval_matrix: Pvalue matrix with Pvalues to be corrected.
        method : Which FDR control to use (either 'bh' or 'by').
        ignore_diag:

    Returns:
        Corrected Pvalue matrix in same numpy.array format. Number of performed tests ignores NAs in the input matrix
        and "self-tests" on the diagonal. Pvalues on the diagonal are therefore set to NA in the resulting output.
    """
    pvalues_array = np.array([])
    mask = np.array([], dtype=bool)
    if ignore_diag:
        # Add NAs on diagonal to ignore "self-tests".
        diagonal_mask = np.eye(pval_matrix.shape[0], dtype=bool)
        pval_matrix[diagonal_mask] = np.nan

        # Subset non-NA values of upper triangular pvalue matrix into flattened 1D array.
        upper_tri_mask = np.triu(np.ones(pval_matrix.shape, dtype=bool), k=1)
        non_na_mask = ~np.isnan(pval_matrix)
        mask = upper_tri_mask & non_na_mask
        pvalues_array = pval_matrix[mask]

    else:
        # Take diagonal into account as well.
        mask = ~np.isnan(pval_matrix)
        pvalues_array = pval_matrix[mask]

    # Apply scipy's false discovery rate control on flattened pvalue matrix.
    pvalues_adjusted = sc.stats.false_discovery_control(pvalues_array, axis=0, method=method)
    pvalues_adjusted = np.clip(pvalues_adjusted, a_min=0.0, a_max=1.0)

    # Put adjusted Pvalues back into submatrix of non-NA entries.
    pval_matrix[mask] = pvalues_adjusted

    if ignore_diag:
        # Copy adjusted Pvalues from upper triangular matrix to lower triangular matrix.
        lower_indices = np.tril_indices(pval_matrix.shape[0], -1)
        pval_matrix[lower_indices] = pval_matrix.T[lower_indices]

    return pval_matrix

def _check_input_data_single_matrix(data : np.ndarray, threads : int, axis : int):
    """
    Check validity of input arguments.
    Args:
        data: Continuous input data matrix.
        threads: Number of threads to use in parallel computations.
        axis: Whether to consider rows as variables (axis=0) or columns.

    Returns: None if input is valid.
    """
    if not isinstance(threads, int) or threads < 1:
        raise ValueError(f"Invalid number of threads: {threads}.")

    if axis!=0 and axis!=1:
        raise ValueError(f"Invalid axis parameter: {axis}.")
    # Check data type of input matrix.
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data matrix needs to be of class numpy.ndarray.")

    if len(data.shape) != 2:
        raise ValueError(f"Input data matrix needs to be 2D, current shape is: {data.shape}.")

def _check_input_data_two_matrices(cont_data : np.ndarray, cat_data : np.ndarray, threads : int,
                                   axis : int):
    """Check validity of input arguments

    Args:
        cont_data (np.ndarray): Continuous input data matrix.
        cat_data (np.ndarray): Categorical input data matrix.
        threads (int): Number of threads to use in parallel computation.
        axis (int): Whether to consider rows as variables (axis=0) or columns.
    """
    if not isinstance(threads, int) or threads < 1:
        raise ValueError(f"Invalid number of threads: {threads}.")

    if axis!=0 and axis!=1:
        raise ValueError(f"Invalid axis parameter: {axis}.")
    # Check data type of input matrix.
    if not isinstance(cat_data, np.ndarray) or not isinstance(cont_data, np.ndarray):
        raise ValueError("Input data matrices need to be of class numpy.ndarray.")

    if len(cat_data.shape) != 2:
        raise ValueError(f"Input data matrix needs to be 2D, current shape is: {cat_data.shape}.")
    if len(cont_data.shape) != 2:
        raise ValueError(f"Input data matrix needs to be 2D, current shape is: {cont_data.shape}.")
    # Check if dimensions of data matrices are matching.
    if axis==0 and cat_data.shape[1] != cont_data.shape[1]:
        raise ValueError(f"Column dimensions of categorical and continuous data do not match.")
    if axis==1 and cont_data.shape[0] != cat_data.shape[0]:
        raise ValueError(f"Row dimensions of categorical and continuous data do not match.")

def parse_input_single_matrix(cont_data):
    if isinstance(cont_data, torch.Tensor):
        return cont_data.numpy().astype(np.float64)
    elif isinstance(cont_data, pd.DataFrame):
        return cont_data.copy().to_numpy(dtype=np.float64)
    elif isinstance(cont_data, np.ndarray):
        return cont_data.copy().astype(np.float64)
    else:
        return raiseValueError("Input error: data has invalid datatype, needs to be either torch.Tensor, pd.DataFrame, or np.ndarray!")

def transform_output(output_dic, axis, data_one, data_two=None):
    if isinstance(data_one, torch.Tensor):
        # Transform all output matrices in dictionary to torch tensors.
        for k, v in output_dic.items():
            output_dic[k] = torch.from_numpy(v)
        return output_dic
    elif isinstance(data_one, pd.DataFrame):
        # Transform to pandas dataframe, with corresponding variables labels.
        for k, v in output_dic.items():
            if axis==0:
                if data_two is None: # Result matrix is symmetric with same rows and columns.
                    output_dic[k] = pd.DataFrame(v, index=data_one.index, columns=data_one.index)
                else: # Result matrix's index is data_one's index and columns are data_two's index.
                    output_dic[k] = pd.DataFrame(v, index=data_one.index, columns=data_two.index)
            else:
                if data_two is None:
                    output_dic[k] = pd.DataFrame(v, index=data_one.columns, columns=data_one.columns)
                else: # Result matrix's index is data_one's index and columns are data_two's index.
                    output_dic[k] = pd.DataFrame(v, index=data_one.columns, columns=data_two.columns)
        return output_dic
    else:
        return output_dic

def joint_multiple_testing_correction(cont_matrix: np.ndarray, cat_matrix : np.ndarray, mixed_matrix : np.ndarray,
                                      method : str = ['bonferroni', 'benjamini-hb', 'benjamini-yek']):
    # For Bonferroni correction, simply count number of performed tests.
    if method == 'bonferroni':
        number_of_tests = 0
        # Counter number of performed tests for cont matrix.
        diag_mask = np.eye(cont_matrix.shape[0], dtype=bool)
        cont_matrix[diag_mask] = np.nan
        na_mask = ~np.isnan(cont_matrix)
        cont_mask = ~diag_mask & na_mask
        cont_submatrix = cont_matrix[cont_mask]
        number_of_tests = len(cont_submatrix) / 2
        # Count number of performed tests for cat matrix.
        diag_mask = np.eye(cat_matrix.shape[0], dtype=bool)
        cat_matrix[diag_mask] = np.nan
        na_mask = ~np.isnan(cat_matrix)
        cat_mask = ~diag_mask & na_mask
        cat_submatrix = cat_matrix[cat_mask]
        number_of_tests = number_of_tests + len(cat_submatrix) / 2
        # Count number of performed tests for mixed matrix.
        mixed_mask = ~np.isnan(mixed_matrix)
        mixed_submatrix = mixed_matrix[mixed_mask]
        number_of_tests = number_of_tests + len(mixed_submatrix)
        ## Multiply each matrixs' pvalues by total number of performed tests.
        cont_submatrix = cont_submatrix * number_of_tests
        corrected_cont = np.clip(cont_submatrix, a_min=0.0, a_max=1.0)
        cont_matrix[cont_mask] = corrected_cont
        cat_submatrix = cat_submatrix * number_of_tests
        corrected_cat = np.clip(cat_submatrix, a_min=0.0, a_max=1.0)
        cat_matrix[cat_mask] = corrected_cat
        mixed_submatrix = mixed_submatrix * number_of_tests
        corrected_mixed = np.clip(mixed_submatrix, a_min=0.0, a_max=1.0)
        mixed_matrix[mixed_mask] = corrected_mixed
        return cont_matrix, cat_matrix, mixed_matrix
    else:
        # Extract relevant Pvalues from cont matrix.
        diagonal_mask = np.eye(cont_matrix.shape[0], dtype=bool)
        cont_matrix[diagonal_mask] = np.nan
        # Subset non-NA values of upper triangular pvalue matrix into flattened 1D array.
        cont_upper_tri_mask = np.triu(np.ones(cont_matrix.shape, dtype=bool), k=1)
        cont_non_na_mask = ~np.isnan(cont_matrix)
        cont_mask = cont_upper_tri_mask & cont_non_na_mask
        cont_array = cont_matrix[cont_mask]
        # Extract relevant Pvalues from cat matrix.
        diagonal_mask = np.eye(cat_matrix.shape[0], dtype=bool)
        cat_matrix[diagonal_mask] = np.nan
        # Subset non-NA values of upper triangular pvalue matrix into flattened 1D array.
        cat_upper_tri_mask = np.triu(np.ones(cat_matrix.shape, dtype=bool), k=1)
        cat_non_na_mask = ~np.isnan(cat_matrix)
        cat_mask = cat_upper_tri_mask & cat_non_na_mask
        cat_array = cat_matrix[cat_mask]
        # Extract relevant Pvalues from mixed matrix.
        mixed_mask = ~np.isnan(mixed_matrix)
        mixed_array = mixed_matrix[mixed_mask]
        # Concatenate three Pvalues arrays.
        joint_pvalues = np.concatenate([cont_array, cat_array, mixed_array])
        # Apply scipy's false discovery rate control on flattened matrix.
        joint_adjusted = sc.stats.false_discovery_control(joint_pvalues, axis=0, method=method)
        joint_adjusted = np.clip(joint_adjusted, a_min=0.0, a_max=1.0)
        # Put adjusted Pvalues back into cont matrix.
        cont_length = len(cont_array)
        cont_matrix[cont_mask] = joint_adjusted[:cont_length]
        lower_indices = np.tril_indices(cont_matrix.shape[0], -1)
        cont_matrix[lower_indices] = cont_matrix.T[lower_indices]
        # Put adjusted Pvalues back into cat matrix.
        cat_length = len(cat_array)
        cat_matrix[cat_mask] = joint_adjusted[cont_length:(cont_length+cat_length)]
        lower_indices = np.tril_indices(cat_matrix.shape[0], -1)
        cat_matrix[lower_indices] = cat_matrix.T[lower_indices]
        # Put adjusted Pvalues back into mixed matrix.
        mixed_length = len(mixed_array)
        mixed_matrix[mixed_mask] = joint_adjusted[(cont_length+cat_length):(cont_length+cat_length+mixed_length)]
        return cont_matrix, cat_matrix, mixed_matrix


def pearsonr(data : np.array, nan_value : float = -999, axis : int = 0, threads : int = 1,
             use_numba : bool = True, return_types : list[str] = []):
    """Computes NA-aware Pearson correlation on given data matrix between all
    combinations of variables and returns r-squared values as well as pvalues.

    Args:
        data (np.array): Given data matrix. Per default, rows are considered as variables and columns
            as samples/measurements.
        nan_value (float, optional): Value indicating missing measurement in data. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        use_numba (bool, optional): If set to True, use numba based python implementation instead of CPP version.
        return_types (list[str], optional): List of data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', and 'r2'. If an empty list is
        passed, every possible data matrix is returned.
    """
    input_data = data
    data = parse_input_single_matrix(data)

    # Check validity of input data.
    _check_input_data_single_matrix(data, threads, axis)

    # Check input of Pvalue adjustment method.
    if not set(return_types).issubset({'r2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['r2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']

    # Transpose data if necessary.
    if axis==1:
        data = data.T.copy()
    
    # Ensure float datatype on matrix.
    data = np.array(data, copy=False, dtype=np.float64)
    nan_value = float(nan_value)
    
    # Use CPP-based correlation computation with OpenMP.
    if not use_numba:
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        data_mat = libnapy.DataMatrix(data)
        corr_mat, pvalue_mat = libnapy.pearson_with_nans(data_mat, nan_value)
        corr_mat = np.array(corr_mat, copy=False)
        pvalue_mat = np.array(pvalue_mat, copy=False)

    else: # Use numba-based python implementation.
        corr_mat, pvalue_mat = libnapy_numba.pearson_numba(data, nan_value, threads)
    
    # Clip values to range 0 and 1 (rounding errors)
    pvalue_mat = np.clip(pvalue_mat, a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'r2' in return_types:
        output_dic["r2"] = corr_mat

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=True)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=True)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=True)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_data)

    return output_dic


def spearmanr(data : np.array, nan_value : float = -999, axis : int = 0, threads : int = 1,
              use_numba : bool = False, return_types : list[str] = []):
    """Computes NA-aware Spearman rank correlation on given data matrix between all
    combinations of variables and return r-squared value matrix as well as pvalue matrix.

    Args:
        data (np.array): Given data matrix. Per default, rows are considered as variables and columns
            as samples/measurements.
        nan_value (float, optional): Value indicating missing measurement in data. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        use_numba (bool, optional): If set to True, use numba based python implementation instead of CPP version.
        return_types (list[str], optional): List of data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', and 'rho'. If an empty list is
        passed, every possible data matrix is returned.
    """
    input_data = data
    data = parse_input_single_matrix(data)

    _check_input_data_single_matrix(data, threads, axis)
    # Check input of return types list.
    if not set(return_types).issubset({'rho', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['rho', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']

    # Transpose data if necessary.
    if axis==1:
        data = data.T.copy()
    
    # Ensure float datatype on matrix.
    data = np.array(data, copy=False, dtype=np.float64)
    nan_value = float(nan_value)

    if not use_numba:
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        # Convert into wrapper object.
        data_mat = libnapy.DataMatrix(data)
        corr_mat, pvalue_mat = libnapy.spearman_with_nans(data_mat, nan_value)
        corr_mat = np.array(corr_mat, copy=False)
        pvalue_mat = np.array(pvalue_mat, copy=False)
    else:
        corr_mat, pvalue_mat = libnapy_numba.spearman_numba(data, nan_value, threads)
        
    # Clip values to range 0 and 1 (rounding errors)
    pvalue_mat = np.clip(pvalue_mat, a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'rho' in return_types:
        output_dic["rho"] = corr_mat

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=True)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=True)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=True)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_data)

    return output_dic


def chi_squared(data : np.array, nan_value : float = -999, axis : int = 0, threads : int = 1,
                check_data : bool = False, return_types : list[str] = [], use_numba : bool = True):
    """Run chi-square test on indepence between all pairs of given variables in matrix. Computes 
    and returns test statistic values as well as associated pvalues. 

    Args:
        data (np.array): Data matrix storing variables and respective measurement of samples.
        nan_value (float, optional): Integer value representing NA values. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        check_data (bool, optional): Whether to perform additional consistency checks on input data. Defaults to False.
        return_types (list[str], optional): List of result data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', 'chi2', 'phi', and 'cramers_v'.
        If an empty list is passed, every possible data matrix is returned.
        use_numba (bool, optional): Whether to use numba-parallelized python implementation.
    """
    input_data = data
    data = parse_input_single_matrix(data)

    _check_input_data_single_matrix(data, threads, axis)
    # Check input of return types list.
    if not set(return_types).issubset({'chi2', 'phi', 'cramers_v', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['chi2', 'phi', 'cramers_v', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']
    
    # Tranpose data if necessary.
    if axis==1:
        data = data.T.copy()
        
    # Ensure float datatype on matrix.
    data = np.array(data, copy=False, dtype=np.float64)
    
    # Count number of categories per variable.
    nan_value = float(nan_value)
    categories_per_var = [len(np.unique(row))-1 if nan_value in row else len(np.unique(row)) for row in data]
    
    # Perform additional consistency checks on categories of input data.
    if check_data:
        for index, row in enumerate(data):
            uniques_per_var = np.unique(row)
            uniques_wo_nan = np.setdiff1d(uniques_per_var, [nan_value])
            if not 0.0 in uniques_wo_nan:
                raise ValueError(f"Input variable {index} does not contain category 0.")
            if np.max(uniques_wo_nan) != len(uniques_wo_nan)-1:
                raise ValueError(f"Input variable {index}'s maximum category does not match number of categories.")

    if any(s.startswith('p_') for s in return_types):
        return_types_mod = {x for x in return_types if not x.startswith('p_')}
        return_types_mod.add('p_unadjusted')
    else:
        return_types_mod = set(return_types)

    if not use_numba:
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        # Convert into wrapper object.
        data_mat = libnapy.DataMatrix(data)
        result_dict = libnapy.chi_squared_with_nans(data_mat, categories_per_var, nan_value, return_types_mod)
    else:
        nan_value = int(nan_value)
        compute_pvalues = 'p_unadjusted' in return_types_mod
        compute_chi2 = 'chi2' in return_types_mod
        compute_phi = 'phi' in return_types_mod
        compute_cramers = 'cramers_v' in return_types_mod
        pvalue_mat, chi2_mat, phi_mat, cramers_mat = libnapy_numba.chi2_numba(data, np.array(categories_per_var), nan_value,
                                                            compute_pvalues, compute_chi2, compute_phi,
                                                            compute_cramers, threads)
        result_dict = dict()
        result_dict["p_unadjusted"] = pvalue_mat
        result_dict["chi2"] = chi2_mat
        result_dict["phi"] = phi_mat
        result_dict["cramers_v"] = cramers_mat
    
    # Clip values to 0 and 1
    result_dict["p_unadjusted"] =  np.clip(result_dict["p_unadjusted"], a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'chi2' in return_types:
        output_dic["chi2"] = np.array(result_dict["chi2"], copy=False)

    if 'phi' in return_types:
        output_dic["phi"] = np.array(result_dict["phi"], copy=False)

    if 'cramers_v' in return_types:
        output_dic["cramers_v"] = np.array(result_dict["cramers_v"], copy=False)
    
    # Check if P-value results are desired.
    if 'p_unadjusted' in result_dict.keys():
        pvalue_mat = np.array(result_dict['p_unadjusted'], copy=False)

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=True)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=True)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=True)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_data)

    return output_dic


def anova(cat_data : np.array, cont_data : np.array, nan_value : float = -999, axis : int = 0,
          threads : int = 1, check_data : bool = False, return_types : list[str] = [],
          use_numba : bool = True):
    """Runs ANOVA between all pairwise combinations of categorical and continuous input data variables
    and returns paiwise statistic values as well as pairwise P-values.

    Args:
        cat_data (np.array): Data matrix storing categorical variables.
        cont_data (np.array): Data matrix storing continuous variables.
        nan_value (float, optional): Value indicating missing value. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        check_data (bool, optional): Whether to perform additional consistency checks on input data. Defaults to False.
        return_types (list[str], optional): List of result data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', 'F', 'np2'.
        If an empty list is passed, every possible data matrix is returned.
        use_numba (bool, optional): Whether or not to use numba-based python implementation.
    """
    input_cat = cat_data
    input_cont = cont_data

    cont_data = parse_input_single_matrix(cont_data)
    cat_data = parse_input_single_matrix(cat_data)

    _check_input_data_two_matrices(cont_data, cat_data, threads, axis)
    # Check input of return types list.
    if not set(return_types).issubset({'F', 'np2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['F', 'np2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']
    
    # Transpose data if necessary.
    if axis==1:
        cat_data = cat_data.T.copy()
        cont_data = cont_data.T.copy()
    
    # Ensure float datatype on matrix.
    cat_data = np.array(cat_data, copy=False, dtype=np.float64)
    cont_data = np.array(cont_data, copy=False, dtype=np.float64)
    
    # Count number of categories per variable.
    nan_value = float(nan_value)
    categories_per_var = [len(np.unique(row))-1 if nan_value in row else len(np.unique(row)) for row in cat_data]
    
    # Perform additional consistency checks on categories of input data.
    if check_data:
        for index, row in enumerate(cat_data):
            uniques_per_var = np.unique(row)
            uniques_wo_nan = np.setdiff1d(uniques_per_var, [nan_value])
            if not 0.0 in uniques_wo_nan:
                raise ValueError(f"Input variable {index} does not contain category 0.")
            if np.max(uniques_wo_nan) != len(uniques_wo_nan)-1:
                raise ValueError(f"Input variable {index}'s maximum category does not match number of categories.")

    if any(s.startswith('p_') for s in return_types):
        return_types_mod = {x for x in return_types if not x.startswith('p_')}
        return_types_mod.add('p_unadjusted')
    else:
        return_types_mod = set(return_types)

    if not use_numba:
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        # Convert into wrapper object.
        cat_data_mat = libnapy.DataMatrix(cat_data)
        cont_data_mat = libnapy.DataMatrix(cont_data)
        result_dict = libnapy.anova_with_nans(cat_data_mat, cont_data_mat, categories_per_var, nan_value, return_types_mod)
    else:
        nan_value = int(nan_value)
        compute_pvalues = 'p_unadjusted' in return_types_mod
        compute_f = 'F' in return_types_mod
        compute_np2 = 'np2' in return_types_mod
        pvalue_mat, f_mat, np2_mat = libnapy_numba.anova_numba(cat_data, cont_data, np.array(categories_per_var), nan_value,
                                                            compute_pvalues, compute_f, compute_np2, threads)
        result_dict = dict()
        result_dict["p_unadjusted"] = pvalue_mat
        result_dict["F"] = f_mat
        result_dict["np2"] = np2_mat
        
    # Clip values to 0 and 1
    result_dict["p_unadjusted"] =  np.clip(result_dict["p_unadjusted"], a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'F' in return_types:
        output_dic["F"] = np.array(result_dict["F"], copy=False)

    if 'np2' in return_types:
        output_dic["np2"] = np.array(result_dict["np2"], copy=False)
    
    # Check if P-value results are desired.
    if 'p_unadjusted' in result_dict.keys():
        pvalue_mat = np.array(result_dict['p_unadjusted'], copy=False)

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=False)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=False)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=False)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_cat, input_cont)

    return output_dic


def kruskal_wallis(cat_data : np.array, cont_data : np.array, nan_value : float = -999, axis : int = 0,
          threads : int = 1, check_data : bool = False, return_types : list[str] = [], use_numba : bool = False):
    """Runs Kruskal-Wallis tests between all pairwise combinations of categorical and continuous input data variables
    and returns pairwise statistic values as well as pairwise P-values.

    Args:
        cat_data (np.array): Data matrix storing categorical variables.
        cont_data (np.array): Data matrix storing continuous variables.
        nan_value (float, optional): Value indicating missing value. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        check_data (bool, optional): Whether to perform additional consistency checks on input data. Defaults to False.
        return_types (list[str], optional): List of result data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', 'H', 'eta2'.
        If an empty list is passed, every possible data matrix is returned.
        use_numba (bool, optional): Whether or not to use numba-based implementation.
    """
    input_cat = cat_data
    input_cont = cont_data

    cont_data = parse_input_single_matrix(cont_data)
    cat_data = parse_input_single_matrix(cat_data)

    _check_input_data_two_matrices(cont_data, cat_data, threads, axis)
    # Check input of return types list.
    if not set(return_types).issubset(
            {'H', 'eta2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['H', 'eta2', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']
    
    # Transpose data if necessary.
    if axis==1:
        cat_data = cat_data.T.copy()
        cont_data = cont_data.T.copy()
    
    # Ensure float datatype on matrix.
    cat_data = np.array(cat_data, copy=False, dtype=np.float64)
    cont_data = np.array(cont_data, copy=False, dtype=np.float64)
    
    # Count number of categories per variable.
    nan_value = float(nan_value)
    categories_per_var = [len(np.unique(row))-1 if nan_value in row else len(np.unique(row)) for row in cat_data]
    
    # Perform additional consistency checks on categories of input data.
    if check_data:
        for index, row in enumerate(cat_data):
            uniques_per_var = np.unique(row)
            uniques_wo_nan = np.setdiff1d(uniques_per_var, [nan_value])
            if not 0.0 in uniques_wo_nan:
                raise ValueError(f"Input variable {index} does not contain category 0.")
            if np.max(uniques_wo_nan) != len(uniques_wo_nan)-1:
                raise ValueError(f"Input variable {index}'s maximum category does not match number of categories.")

    if any(s.startswith('p_') for s in return_types):
        return_types_mod = {x for x in return_types if not x.startswith('p_')}
        return_types_mod.add('p_unadjusted')
    else:
        return_types_mod = set(return_types)

    if not use_numba:
        # Convert into wrapper object.
        cat_data_mat = libnapy.DataMatrix(cat_data)
        cont_data_mat = libnapy.DataMatrix(cont_data)
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        # Run tests.
        result_dict = libnapy.kruskal_wallis_with_nans(cat_data_mat, cont_data_mat, categories_per_var, nan_value, return_types_mod)
    else:
        nan_value = int(nan_value)
        compute_pvalues = 'p_unadjusted' in return_types_mod
        compute_h = 'H' in return_types_mod
        compute_eta2 = 'eta2' in return_types_mod
        pvalue_mat, h_mat, eta2_mat = libnapy_numba.kruskal_wallis_numba(cat_data, cont_data, nan_value, np.array(categories_per_var),
                                                            threads, compute_pvalues, compute_h, compute_eta2)
        result_dict = dict()
        result_dict["p_unadjusted"] = pvalue_mat
        result_dict["H"] = h_mat
        result_dict["eta2"] = eta2_mat
        
    # Clip values to 0 and 1
    result_dict["p_unadjusted"] =  np.clip(result_dict["p_unadjusted"], a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'H' in return_types:
        output_dic["H"] = np.array(result_dict["H"], copy=False)

    if 'eta2' in return_types:
        output_dic["eta2"] = np.array(result_dict["eta2"], copy=False)

    # Check if P-value results are desired.
    if 'p_unadjusted' in result_dict.keys():
        pvalue_mat = np.array(result_dict['p_unadjusted'], copy=False)

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=False)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=False)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=False)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_cat, input_cont)

    return output_dic


def ttest(bin_data : np.array, cont_data : np.array, nan_value : float = -999, axis : int = 0,
          threads : int = 1, check_data : bool = False, return_types : list[str] = [],
          equal_var : bool = True, use_numba : bool = True):
    """Runs t-tests on independence between all pairwise combinations of binary and continuous 
       input data variables and return pairwise statistic / effect size values as well as 
       pairwise P-values.

    Args:
        bin_data (np.array): Data matrix storing binary variables.
        cont_data (np.array): Data matrix storing continuous variables.
        nan_value (float, optional): Value indicating missing value. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        check_data (bool, optional): Whether to perform additional consistency checks on input data. Defaults to False.
        return_types (list[str], optional): List of result data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', 't', 'cohens_d'.
        equal_var (bool, optional): Whether or not to assume that variances of pairwise variables
            are equal (in which case to perform Student's t-test) or not (in which case to perform
            Welch's t-test). Defaults to True.
        use_numba (bool, optional): Whether or not to use numba-based python implementation. Defaults to True.
    """
    input_bin = bin_data
    input_cont = cont_data

    cont_data = parse_input_single_matrix(cont_data)
    bin_data = parse_input_single_matrix(bin_data)

    _check_input_data_two_matrices(cont_data, bin_data, threads, axis)
    # Check input of return types list.
    if not set(return_types).issubset(
            {'t', 'cohens_d', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['t', 'cohens_d', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']
    
    # Transpose data if necessary.
    if axis==1:
        bin_data = bin_data.T.copy()
        cont_data = cont_data.T.copy()
    
    # Ensure float datatype on matrix.
    bin_data = np.array(bin_data, copy=False, dtype=np.float64)
    cont_data = np.array(cont_data, copy=False, dtype=np.float64)

    nan_value = float(nan_value)
    use_welch = not equal_var
    
    # Perform additional consistency checks on categories of input data.
    if check_data:
        for index, row in enumerate(bin_data):
            uniques_per_var = np.unique(row)
            uniques_wo_nan = np.setdiff1d(uniques_per_var, [nan_value])
            if not 0.0 in uniques_wo_nan:
                raise ValueError(f"Input variable {index} does not contain category 0.")
            if np.max(uniques_wo_nan) != len(uniques_wo_nan)-1:
                raise ValueError(f"Input variable {index}'s maximum category does not match number of categories.")
            if len(uniques_wo_nan) != 2:
                raise ValueError(f"Input variable {index} is not binary. T-test requires binary variables.")

    if any(s.startswith('p_') for s in return_types):
        return_types_mod = {x for x in return_types if not x.startswith('p_')}
        return_types_mod.add('p_unadjusted')
    else:
        return_types_mod = set(return_types)

    if not use_numba:
        # Set number of desired threads for computation.
        libnapy.set_num_threads(threads)
        # Convert into wrapper object.
        bin_data_mat = libnapy.DataMatrix(bin_data)
        cont_data_mat = libnapy.DataMatrix(cont_data)
        # Run t-test.
        result_dict = libnapy.t_test_with_nans(bin_data_mat, cont_data_mat, nan_value, return_types_mod, use_welch)
    else:
        nan_value = int(nan_value)
        compute_pvalues = 'p_unadjusted' in return_types_mod
        compute_t = 't' in return_types_mod
        compute_cohens = 'cohens_d' in return_types_mod
        pvalue_mat, t_mat, cohens_mat = libnapy_numba.ttest_numba(bin_data, cont_data, nan_value, compute_pvalues,
                                                                compute_t, compute_cohens, use_welch, threads)
        result_dict = dict()
        result_dict["p_unadjusted"] = pvalue_mat
        result_dict["t"] = t_mat
        result_dict["cohens_d"] = cohens_mat
    
    # Clip values to 0 and 1
    result_dict["p_unadjusted"] =  np.clip(result_dict["p_unadjusted"], a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 't' in return_types:
        output_dic["t"] = np.array(result_dict["t"], copy=False)

    if 'cohens_d' in return_types:
        output_dic["cohens_d"] = np.array(result_dict["cohens_d"], copy=False)

    # Check if P-value results are desired.
    if 'p_unadjusted' in result_dict.keys():
        pvalue_mat = np.array(result_dict['p_unadjusted'], copy=False)

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=False)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=False)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=False)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_bin, input_cont)

    return output_dic

def mwu(bin_data: np.array, cont_data: np.array, nan_value: float = -999, axis: int = 0,
          threads: int = 1, check_data: bool = False, return_types: list[str] = [],
          mode : str = 'auto', use_numba : bool = False):
    """Runs Mann-Whitney-U tests on independence between all pairwise combinations of binary and continuous
       input data variables and return pairwise statistic / effect size values as well as
       pairwise P-values.

    Args:
        bin_data (np.array): Data matrix storing binary variables.
        cont_data (np.array): Data matrix storing continuous variables.
        nan_value (float, optional): Value indicating missing value. Defaults to -999.
        axis (int, optional): Whether to consider rows as variables (axis=0) or columns (axis=1). Defaults to 0.
        threads (int, optional): Number of threads to be used in parallel computation. Defaults to 1.
        check_data (bool, optional): Whether to perform additional consistency checks on input data. Defaults to False.
        return_types (str, optional): List of result data matrices to return. Can be any subset of
        'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek', 'U', 'r'.
        If an empty list is passed, every possible data matrix is returned.
        mode (str, optional): Which method to use for the P-value calculation. Can be one of 'exact', 'asymptotic', or
        'auto'. In the first case, the computationally expensive exact calculation is used, in case of 'asymptotic' we
        make use of the z-value based approximation of U. In both cases, we use averaging for tiebreaking.
        When using 'auto', if there are no ties in the data, and the sample size of one of the two groups is less than 8,
        we use 'exact', otherwise we run 'asymptotic'. Note that if there are ties in the data, but 'exact'
        mode is chosen, this may lead to inaccurate test results, only chose mode 'exact' manually if working
        on small data that you can manually check.
        use_numba (bool, optional): Whether or not to use numba based implementation of MWU.
    """
    input_bin = bin_data
    input_cont = cont_data

    cont_data = parse_input_single_matrix(cont_data)
    bin_data = parse_input_single_matrix(bin_data)

    _check_input_data_two_matrices(cont_data, bin_data, threads, axis)
    if not mode in ['auto', 'asymptotic', 'exact']:
        raise ValueError(f"Invalid Mann-Whitney-U test mode : {mode}.")

    if not set(return_types).issubset(
            {'U', 'r', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek'}):
        raise ValueError(f"Unknown return type in input list: {return_types}.")

    if len(return_types) == 0:
        return_types = ['U', 'r', 'p_unadjusted', 'p_bonferroni', 'p_benjamini_hb', 'p_benjamini_yek']

    # Transpose data if necessary.
    if axis == 1:
        bin_data = bin_data.T.copy()
        cont_data = cont_data.T.copy()

    # Ensure float datatype on matrix.
    bin_data = np.array(bin_data, copy=False, dtype=np.float64)
    cont_data = np.array(cont_data, copy=False, dtype=np.float64)

    nan_value = float(nan_value)

    # Perform additional consistency checks on categories of input data.
    if check_data:
        for index, row in enumerate(bin_data):
            uniques_per_var = np.unique(row)
            uniques_wo_nan = np.setdiff1d(uniques_per_var, [nan_value])
            if not 0.0 in uniques_wo_nan:
                raise ValueError(f"Input variable {index} does not contain category 0.")
            if np.max(uniques_wo_nan) != len(uniques_wo_nan) - 1:
                raise ValueError(f"Input variable {index}'s maximum category does not match number of categories.")
            if len(uniques_wo_nan) != 2:
                raise ValueError(f"Input variable {index} is not binary. Mann-Whitney-U test requires binary variables.")

    if any(s.startswith('p_') for s in return_types):
        return_types_mod = {x for x in return_types if not x.startswith('p_')}
        return_types_mod.add('p_unadjusted')
    else:
        return_types_mod = set(return_types)

    # Set number of desired threads for computation.
    if not use_numba:
        # Convert into wrapper object.
        bin_data_mat = libnapy.DataMatrix(bin_data)
        cont_data_mat = libnapy.DataMatrix(cont_data)
        libnapy.set_num_threads(threads)
        result_dict = libnapy.mwu_with_nans(bin_data_mat, cont_data_mat, nan_value, return_types_mod, mode)
    else:
        if mode == "auto":
            mode_int = 0
        elif mode == "exact":
            mode_int = 1
        else:
            mode_int = 2
        compute_pvalues = 'p_unadjusted' in return_types_mod
        compute_u = 'U' in return_types_mod
        compute_r = 'r' in return_types_mod
        pvalue_mat, u_mat, r_mat = libnapy_numba.mann_whitney_numba(bin_data, cont_data, nan_value, compute_pvalues,
                                                                     compute_u, compute_r, threads, mode_int)
        result_dict = dict()
        result_dict["p_unadjusted"] = pvalue_mat
        result_dict["U"] = u_mat
        result_dict["r"] = r_mat
    
    # Clip values to 0 and 1
    result_dict["p_unadjusted"] =  np.clip(result_dict["p_unadjusted"], a_min=0.0, a_max=1.0)

    output_dic = dict()
    # Check which effect sizes and Pvalues to return.
    if 'U' in return_types:
        output_dic["U"] = np.array(result_dict["U"], copy=False)

    if 'r' in return_types:
        output_dic["r"] = np.array(result_dict["r"], copy=False)

    # Check if P-value results are desired.
    if 'p_unadjusted' in result_dict.keys():
        pvalue_mat = np.array(result_dict['p_unadjusted'], copy=False)

    if 'p_bonferroni' in return_types:
        pvalue_mat_bonf = _adjust_pvalues_bonferroni(pvalue_mat.copy(), ignore_diag=False)
        output_dic["p_bonferroni"] = pvalue_mat_bonf

    if 'p_benjamini_hb' in return_types:
        pvalue_mat_benj_hb = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'bh', ignore_diag=False)
        output_dic['p_benjamini_hb'] = pvalue_mat_benj_hb

    if 'p_benjamini_yek' in return_types:
        pvalue_mat_benj_yek = _adjust_pvalues_fdr_control(pvalue_mat.copy(), 'by', ignore_diag=False)
        output_dic['p_benjamini_yek'] = pvalue_mat_benj_yek

    if 'p_unadjusted' in return_types:
        output_dic["p_unadjusted"] = pvalue_mat

    output_dic = transform_output(output_dic, axis, input_bin, input_cont)

    return output_dic

if __name__ == "__main__":
    cont_unadjusted = np.array([[0.        , 0.21975876, 0.74907518, 0.49731136],
       [0.21975876, 0.        , 0.34668469, 0.54832602],
       [0.74907518, 0.34668469, 0.        , 0.94735348],
       [0.49731136, 0.54832602, 0.94735348, 0.        ]])
    cat_unadjusted = np.array([[0.01735127, 0.13533528, 0.13533528],
       [0.13533528, 0.01430588, 0.01430588],
       [0.13533528, 0.01430588, 0.01430588]])
    mixed_unadjusted = np.array([[0.56471812, 0.65143906, 0.8668779 , 0.18009231],
       [0.27523352, 0.51269076, 0.51269076, 0.82725935],
       [0.27523352, 0.51269076, 0.51269076, 0.82725935]])
    out = joint_multiple_testing_correction(cont_unadjusted, cat_unadjusted, mixed_unadjusted, method='bh')
