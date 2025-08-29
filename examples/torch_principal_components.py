import pandas as pd
import napy
import torch

# Read numerical data.
df = pd.read_csv('example_numerical.csv', index_col=0)
data_tensor = torch.tensor(df.to_numpy())

# Compute NA-aware Pearson Correlation with NApy.
spearman_results = napy.spearmanr(data=data_tensor,
                                axis=1,
                                nan_value=-99.0,
                                threads=1)
correlation_tensor = spearman_results['rho']

# Eigen-decomposition of correlation matrix.
eigenvalues, eigenvectors = torch.linalg.eig(correlation_tensor)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Sort eigenvalues in descending order.
sorted_indices = torch.argsort(eigenvalues, descending=True)
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Select most explanatory components.
k = 5
top_eigenvalues = eigenvalues_sorted[:k]
top_eigenvectors = eigenvectors_sorted[:, :k]

print("Top k eigenvalues:", top_eigenvalues)
print("Top k principal components:")
print(top_eigenvectors)



