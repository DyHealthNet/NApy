# NApy: Efficient Statistics in Python for Large-Scale Heterogeneous Data with Enhanced Support for Missing Data

A fast python tool providing statistical tests and effect sizes for a more comprehensive and informative analysis of mixed type data in the presence of missingness. Written both in C++ and numba and parallelized with OpenMP.

![NApy_Overview](https://github.com/user-attachments/assets/4330d368-a962-493a-9b6d-b26554fabf5b)

# Installation

## Installation on Linux

On a Linux system with the `conda` package manager installed, you can simply install NApy by running `source install.sh`. You can then reassure that the installation was succesful by starting an interactive python session in your shell (e.g. `python -i`) and there running
```python
import napy
import numpy as np
data = np.random.rand(4, 10)
res = napy.spearmanr(data)
```

In case the above installation does not work for you, or should give erroneous results, you can also manually install NApy by running:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/module"
conda env create -f environment.yml
conda activate napy
mkdir build
cd build
cmake ..
make
cp libnapy.cpython<...>.so ../module
```
In the last step above you need to move the resulting `.so` file in the `build/` directory into the directory `module/`.

For easy usage, we recommend adding the path to `module/` to your python include path permanently. Under Linux, this can be done by adding the line `export PYTHONPATH="${PYTHONPATH}:$(pwd)/module"` to the `.bashrc` file. You can then simply use NApy from any python program by putting the line `import napy` at the top of your implementation. Note that for NApy to run, you need to have the installed environment `napy` activated.

## Installation on Mac

On a MacOS system, you need to handle the installation more manually. First you need to install following packages:

```bash
brew install libomp
brew install lapack
```

Then you need to ensure that the paths are correctly exported. Open your ~/.zshrc file, and add the following lines:

```bash
export OpenMP_ROOT=$(brew --prefix)/opt/libomp

export LDFLAGS="-L/opt/homebrew/opt/lapack/lib"
export CPPFLAGS="-I/opt/homebrew/opt/lapack/include"

export PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig"
```
Save the file and run zsh to apply changes. Now you are ready to build a conda environment:

```bash
conda env create -n napy -f environment_mac.yml
conda activate napy
```

After this, you need to execute the following lines to build the desired module.

```bash
mkdir build
cd build
cmake -DCMAKE_CXX_STANDARD=17 ..
make
```

The resulting .so file will be located in your build/ directory. In the last step above you need to move the resulting `.so` file from the `build/` directory into the directory `module/`. For easy usage, we recommend adding the path to `module/` to your python include path permanently. You can then simply use NApy from any python program by putting the line `import napy` at the top of your implementation. Note that for NApy to run, you need to have the installed environment `napy` activated.

```python
import napy
import numpy as np
data = np.random.rand(4, 10)
res = napy.spearmanr(data)
```
# How To Integrate New Tests into NApy
<img width="450" alt="NApy_NewTests" src="https://github.com/user-attachments/assets/ca15aed7-91fa-48d5-9904-0c6936410075" align="right" />

If you want to integrate new tests into NApy, fork the repository, implement the following steps, and please write correctness tests to compare your implementation with existing Python libraries!

1. Become familiar with mathematical foundations of your statistical test procedure
2. Is the test procedure efficiently implementable in Numba, i.e. does the statistical test procedure mainly consist of single-passes over input data structures, and computation of sums, averages, ...?
3. Are all necessary mathematical operations compatible with Numba?
4. If 2. and 3. are true for your statistical test procedure, implement the test in the <b>Python part of NApy with Numba</b> (more conveniant, less requirement on C++ knowledge):
    * Add new function in `module/library_numba.py` with corresponding function signature
5. If 2. and 3. are not true for your statistical test procedure, implement the test in the <b>C++ part of NApy</b>:
    * Add new source file for respective test in `src/` (e.g. src/new_test.cpp)
    * Add name of new source file in line 4 of `src/CMakeLists.txt`
    * Declare function signature in namespace statistics within file `include/stats.hpp`
    * Recompile NApy after addding the test to its C++ part
6. Implement Python "wrapper" function in `module/napy.py` (which is supposed to be called from outside user when importing NApy as a Python module):
    * If Numba-based: call your Python-based implementation of your newly added test with `libnapy_numba.new_test(...)`
    * If C++-based: call your C++ version of your test with `libnapy.new_test(...)`
7. Implement correctness tests in the [NApy_benchmark](https://github.com/DyHealthNet/NApy_benchmark) repository under `tests/test_napy.py`. Ideally, they should include testing of all implemented parameters and comparisons against at least two further statistical tools (e.g. Python-based and R-based).

If you run into any problems while implementing your test, please open an issue in this GitHub repository so we can try to help you.


# Examples for Interoperability with scikit-learn and PyTorch

We provide two examples showing NApy's interoperability with other libraries:
* one example for the case of correlation-based hierarchical clustering using scikit-learn's AgglomerativeClustering (`examples/sklearn_clustering.py`)
* one example for computing principal components on correlation structures using PyTorch's linalg.eig.function (`examples/torch_principal_components.py`)

# User Manual

We here provide an overview of the usability of the implemented tests and their respective input and output.

## Quantitative vs. quantitative data

- **Pearson Correlation:** The function `napy.pearsonr(data, nan_value=-999.0, axis=0, threads=1, return_types=[], use_numba=True)` computes Pearson's r-squared value on all pairs of given variables, in combination with associated two-sided P-values. The P-values roughly indicate the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from the given pairwise data. Missing values are pairwisely ignored. In case two given input variables have length less than three (which can also happen after removal of NAs), P-values are not well-defined and `np.nan` is returned instead. The function takes the following arguments:

  - data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing continuous data values.
  - nan_value [float, default=-999.0]: Float value representing missing values in the given input data.
  - axis [int, default=0]: Whether to consider rows as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computation.
  - return_types [list[str], default=[]]: Which calculation results (correlation value, P-values) to return. Can be a list containing any of the following entries: `'r2'` for Pearson's r-squared correlation values, `'p_unadjusted'` for uncorrected two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned. In the P-value correction procedure we ignore "self-tests" on the diagonal of the P-value matrix and therefore set the P-values after applying the desired multiple testing correction method to `numpy.nan`. In this case, since the resulting P-value matrix is symmetric, the number of actually performed tests corresponds to half the size of the P-value matrix minus the diagonal. With any applied multiple testing correction case, `numpy.nan` values based on the unadjusted P-value matrix are ignored and not counted as performed tests.
  - use_numba [bool, default=True]: Whether to use the numba-based or C++ implementation of the test.

  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.

  Example call:

  ```python
  import napy
  import numpy as np
  data = np.array([[1,2,3,4,5], [2,3,4,5,-99]])
  NAN_VALUE = -99.0
  result_dict = napy.pearsonr(data, nan_value=NAN_VALUE, axis=0, threads=1, return_types=['r2', 'p_benjamini_hb'])
  # result_dict['r2'] stores Pearson correlation values, and result_dict['p_benjamini_hb'] stores corrected P-values
  ```

- **Spearman rank correlation:**  The function `nanpy.spearmanr(data, nan_value=999.0, axis=0, threads=1, return_types=[], use_numba=False)`  computes Spearman's rank correlation coefficients and associated two-sided P-values for all pairs of variables in the given data. The P-value roughly indicates the probability of an uncorrelated system producing datasets that have a Spearman correlation at least as extreme as the one computed from the given pairwise data. Missing values are pairwisely ignored in the calculations. In case two given input variables have length less than three (which can also happen after removal of NAs), P-values are not well-defined and `np.nan` is returned instead. The function takes the following arguments:

  - data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing data to compute correlation and pvalues for.
  - nan_value [float, default=-999.0]: Float value representing missing values in the given input data.
  - axis [int, default=0]: Whether to consider rows as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computation.
  - return_types [list[str], default=[]]: Which calculation results (correlation values, P-values) to return. Can be a list containing any of the following entries: `'rho'` for Spearman rank correlation values, `'p_unadjusted'` for uncorrected two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned. In the P-value correction procedure we ignore "self-tests" on the diagonal of the P-value matrix and therefore set the P-values after applying the desired multiple testing correction method to `numpy.nan`. In this case, since the resulting P-value matrix is symmetric, the number of actually performed tests corresponds to half the size of the P-value matrix minus the diagonal. With any applied multiple testing correction case, `numpy.nan` values based on the unadjusted P-value matrix are ignored and not counted as performed tests.
  - use_numba [bool, default=False]: Whether to use the numba-based or C++ implementation of the test.
  
  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.
  
  Example call:
  
  ```python
  import napy
  import numpy as np
  data = np.array([[1,2,3,4,5], [2,3,4,5,-99]])
  NAN_VALUE = -99.0
  result_dict = napy.spearmanr(data, nan_value=NAN_VALUE, axis=0, threads=1, return_types=['rho', 'p_unadjusted'])
  # result_dict['rho'] stores Spearman rank correlation values, result_dict['p_unadjusted'] stores unadjusted P-values
  ```
  
  

## Categorical vs. categorical data

- **Chi-squared test on independence:** The function `napy.chi_squared(data, nan_value=-999.0, axis=0, threads=1, check_data=False, return_types=[], use_numba=True)` runs chi-squared tests on independence on all pairs of given variables. It returns the statistics, effect sizes and P-values declared in the parameter `return_types`. Missing values are pairwisely ignored in the calculation. For each variable, categories need to be integer-encoded by integers in ascending order starting from zero. In case some category should no longer be present due to missing values, this will lead to the chi-squared statistic being undefined and will hence return `numpy.nan` for the respective pair of variables. This also happens in case a variable should only consist of one category. The function takes the following arguments:

  - data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing data to compute correlation and pvalues for.
  - nan_value [float, default=-999.0]: Float value representing missing values in the given input data.
  - axis [int, default=0]: Whether to consider rows as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computation.
  - check_data [bool, default=False]: Whether or not to perform additional checks on the format of input data. It introduces a slight overhead in runtime.
  - return_types [list[str], default=[]]: Which statistic results to return. Can be a list containing any of the following entries: `'chi2'` for the corresponding value of the chi-squared statistic, `'phi'` and `'cramers_v'` for the respective effect sizes, `'p_unadjusted'` for unadjusted two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned. In the P-value correction procedure we ignore "self-tests" on the diagonal of the P-value matrix and therefore set the P-values after applying the desired multiple testing correction method to `np.nan`. In this case, since the resulting P-value matrix is symmetric, the number of actually performed tests corresponds to half the size of the P-value matrix minus the diagonal. With any applied multiple testing correction case, `np.nan` values based on the unadjusted P-value matrix are ignored and not counted as performed tests.
  - use_numba [bool, default=True]: Whether to use the numba-based or C++ implementation of the test.
  
  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.
  
  
  Example usage:
  
  ```python
  import napy 
  import numpy as np
  data = np.array([[1,1,0,0,1], [1,2,0,2,0], [1,3,2,0,-99]])
  NAN_VALUE = -99.0
  result_dict = napy.chi_squared(data, nan_value=NAN_VALUE, axis=0, threads=1, check_data=False, return_types=['phi', 'p_bonferroni'])
  # result_dict['phi'] stores effect size values, result_dict['p_bonferroni'] stores Bonferroni-corrected P-values
  ```
  



## Quantitative vs. categorical data

- **One-way-ANOVA:** The function `napy.anova(cat_data, cont_data, nan_value=-999.0, axis=0, threads=1, check_data=False, return_types=[], use_numba=True)` runs ANOVA tests between all pairwise combinations of variables from `cat_data` and `cont_data`. It returns the effect sizes, statistic values, and P-values as declared in the parameter `return_types` (with two-sided P-values based on the computed F statistic value). Missing values are pairwisely ignored. Categories need to be integer-encoded in ascending order starting from zero. In case some category should no longer be present due to missing values, this will lead to the F-statistic being undefined and will hence return `numpy.nan` for the respective pair of variables. The same happens in case a variable should only consist of one category. The function takes the following arguments:
  
  - cat_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing categorical variables data.
  - cont_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing continuous variables data.
  - axis [int, default=0]: Whether to consider rows in both input matrices as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computations.
  - check_data [bool, default=False]: Whether or not to perform additional checks on the format of categorical input data. It introduces a slight overhead in runtime.
  - return_types [list[str], default=[]]: Which statistic results to return. Can be a list containing any of the following entries: `'F'` for the corresponding value of the F statistic,  and `'np2'` for the partial-eta-squared effect size, `'p_unadjusted'` for unadjusted two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned.
  - use_numba [bool, default=True]: Whether to use the numba-based or C++ implementation of the test.
  
  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.

​	Example usage:	

```python
import napy 
import numpy as np
cat_data = np.array([[1,1,0,1], [1,2,2,0], [1,0,0,-99]])
cont_data = np.array([[-99,3,3,2], [1,4,0,7], [2,2,1,-3]])
NAN_VALUE = -99.0
result_dict = napy.anova(cat_data, cont_data, nan_value=NAN_VALUE, axis=0, threads=1, check_data=False, return_types=['np2', 'p_unadjsted'])
# result_dict['np2'] stores effect size values, result_dict['p_unadjusted'] stores unadjusted P-values
```



- **Kruskal-Wallis test:** The function `napy.kruskal_wallis(cat_data, cont_data, nan_value=-999.0, axis=0, threads=1, check_data=False, return_types=[], use_numba=False)` runs Kruskal-Wallis test between all pairwise combinations of variables from `cat_data` and `cont_data`. It computes the effect size and statistic declared in parameter `return_types` and P-values equal to the survival function of the chi-square distribution evaluated at H. Missing values are pairwisely ignored. Categories need to be integer-encoded in ascending order starting from zero. In case some category should no longer be present due to missing values, this will lead to the H-statistic being undefined and will hence return `numpy.nan` for the respective pair of variables. The same happens in case a variable should only consist of one category. The function takes the following arguments:
  
  - cat_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing categorical variables data.
  - cont_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table array storing continuous variables data.
  - axis [int, default=0]: Whether to consider rows in both input matrices as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computations.
  - check_data [bool, default=False]: Whether or not to perform additional checks on the format of categorical input data. It introduces a slight overhead in runtime.
  - return_types [list[str], default=[]]: Which statistic results to return. Can be a list containing any of the following entries: `'H'` for the corresponding value of the H statistic,  and `'eta2'` for the eta-squared effect size, `'p_unadjusted'` for unadjusted two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned.
  - use_numba [bool, default=False]: Whether to use the numba-based or C++ implementation of the test.
  
  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.

​	Example usage:

```python
import napy 
import numpy as np
cat_data = np.array([[1,1,0,1], [1,2,2,0], [1,0,0,-99]])
cont_data = np.array([[-99,3,3,2], [1,4,0,7], [2,2,1,-3]])
NAN_VALUE = -99.0
result_dict = napy.kruskal_wallis(cat_data, cont_data, nan_value=NAN_VALUE, axis=0, threads=1, check_data=False, return_types=['H', 'eta2'])
# result_dict['H'] stores H statistic values, result_dict['eta2'] stores eta-squared effect size values
```



- **Student's and Welch's t-test:** The function `napy.ttest(bin_data, cont_data, nan_value = -999.0, axis=0, threads=1, check_data=False, return_types=[], equal_var=True, use_numba=True)` runs t-tests between all pairwise combinations of variables from `bin_data` and `cont_data`. Depending whether equal variances of binary sample groups are assumed, we either run Student's t-test (`equal_var=True`) or Welch's t-test (`equal_var=False`). It computes the effect size / statistic value declared in parameter `return_types` and P-values equal to twice the value of the survival function of the t-distribution at the absolute value of the corresponding t-value. Missing values are pairwisely ignored, i.e. if a missing value occurs in the binary variable, the matching position in the continuous data is also ignored. Binary categories need to be integer-encoded (`0,1`). In case one category should no longer be present due to removal of missing values, this will lead to the t-statistic being undefined and will hence return `numpy.nan` for the respective pair of variables. The same happens in case a variable should only consist of one category. The function takes the following arguments:

  - bin_data [np.ndarray, torch.Tensor, pd.DataFrame] : 2D table storing binary variables data.
  - cont_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing continuous variable data.
  - axis [int, default=0]: Whether to consider rows in both input matrices as variables (axis=0) or columns (axis=1).
  - threads [int, default=1]: How many threads to use in parallel computations.
  - check_data [bool, default=False]: Whether or not to perform additional checks on the format of categorical input data. It introduces a slight overhead in runtime.
  - return_types [list[str], default=[]]: Which statistic results to return. Can be a list containing any of the following entries: `'t'` for the corresponding value of the t statistic,  and `'cohens_d'` for Cohen's D effect size, `'p_unadjusted'` for unadjusted two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned.
  - use_numba [bool, default=True]: Whether to use the numba-based or C++ implementation of the test.
  - equal_var [bool, default=True]: Whether or not to assume equal variances within binary groups of variables. If `equal_var=True`, we run Student's t-test, otherwise Welch's t-test.

  Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.

  Example call:

  ```python
  import napy 
  import numpy as np
  bin_data = np.array([[1,1,0,1,0], [-99,0,0,1,1]])
  cont_data = np.array([[3,2,4,-99,1], [3,1,2,5,4]])
  NAN_VALUE = -99.0
  result_dict = napy.ttest(bin_data, cont_data, nan_value=NAN_VALUE, axis=0, threads=1, check_data=False, return_types=['p_unadjusted'], equal_var=False)
  # result_dict['p_unadjsted'] stores unadjusted P-values
  ```




- **Mann-Whitney-U test:** The function `napy.mwu(bin_data, cont_data, nan_value = -999.0, axis=0, threads=1, check_data=False, return_types=[], mode='auto')` runs Mann-Whitney-U tests between all pairwise combinations of variables from `bin_data` and `cont_data`. Depending on the chosen `mode` and the input data, either the exact P-value calculation or the much faster asymptotic approximation based on the z-value is used. The function computes the effect size / statistic value declared in parameter `return_types` and two-sided P-values either from the exact calculation of the U-distribution or the survival function of the standard normal distribution, depending on `mode`. Missing values are pairwisely ignored, i.e. if a missing value occurs in the binary variable, the matching position in the continuous data is also ignored. Binary categories need to be integer-encoded (`0,1`). In case one category should no longer be present due to removal of missing values, this will lead to the U-statistic being undefined and will hence return `numpy.nan` for the respective pair of variables. The same happens in case a variable should only consist of one category. The function takes the following arguments:

  - bin_data [np.ndarray, torch.Tensor, pd.DataFrame] : 2D table storing binary variables data.

  - cont_data [np.ndarray, torch.Tensor, pd.DataFrame]: 2D table storing continuous variable data.

  - axis [int, default=0]: Whether to consider rows in both input matrices as variables (axis=0) or columns (axis=1).

  - threads [int, default=1]: How many threads to use in parallel computations.

  - check_data [bool, default=False]: Whether or not to perform additional checks on the format of categorical input data. It introduces a slight overhead in runtime.

  - return_types [list[str], default=[]]: Which statistic results to return. Can be a list containing any of the following entries: `'t'` for the corresponding value of the t statistic,  and `'cohens_d'` for Cohen's D effect size, `'p_unadjusted'` for unadjusted two-sided P-values, `'p_bonferroni'` for Bonferroni-corrected P-values,  `'p_benjamini_hb'` for Benjamini-Hochberg correction, and `p_benjamini_yek'` for Bejamini-Yekutieli correction. If the list is emtpy, all available results will be returned.

  - use_numba [bool, default=False]: Whether to use the numba-based or C++ implementation of the test.

  - mode [str, default='auto']: Which method to use for the two-sided P-value calculation. Can be one of `auto, exact, asymptotic`. In the `exact` calculation, the P-value is computed from the survival function of the exact distribution of U, using a [fast algorithmic implementation](https://aakinshin.net/posts/mw-loeffler/) of the generally time-consuming procedure. With the `asymptotic` mode, the much faster calculation via the surival function of the standard normal distribution evaluated at the z-value is used. In the `auto` mode, for each pair of variables it is checked whether there are no ties in the corresponding data and at least one of the two samples has less than eight elements, and in such case the `exact` mode is chosen. Otherwise, the `asymptotic` mode is run. Note that we correct for ties by rank-averaging over equal values, which makes the test result invalid in case the user runs `mode='exact'` even if there are ties in the data. We generally encourage usage of the `mode=auto`.

    Based on the values specified in the return_types list, a dictionary storing the specified data matrices will be returned.

    Example call:

    ```python
    import napy 
    import numpy as np
    bin_data = np.array([[1,1,0,1,0], [-99,0,0,1,1]])
    cont_data = np.array([[3,2,4,-99,1], [3,1,2,5,4]])
    NAN_VALUE = -99.0
    statistics, pvalues = napy.mwu(bin_data, cont_data, nan_value=NAN_VALUE, axis=0, threads=1, check_data=False, return_types=['r', 'p_benjamini_hb'], mode='auto')
    # result_dict['p_benjamini_hb'] stores Benjamini-Hochberg corrected P-values, result_dict['r'] stores effect size values
    ```

# Citation

In case you find our tool useful, please cite our corresponding [manuscript](https://doi.org/10.1093/gigascience/giaf140):

Fabian Woller, Lis Arend, Christian Fuchsberger, Markus List, David B Blumenthal, NApy: Efficient Statistics in Python for Large-Scale Heterogeneous Data with Enhanced Support for Missing Data, GigaScience, 2025;, giaf140, https://doi.org/10.1093/gigascience/giaf140

```
@article{10.1093/gigascience/giaf140,
    author = {Woller, Fabian and Arend, Lis and Fuchsberger, Christian and List, Markus and Blumenthal, David B},
    title = {NApy: Efficient Statistics in Python for Large-Scale Heterogeneous Data with Enhanced Support for Missing Data},
    journal = {GigaScience},
    pages = {giaf140},
    year = {2025},
    month = {11},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf140},
    url = {https://doi.org/10.1093/gigascience/giaf140},
}
```


