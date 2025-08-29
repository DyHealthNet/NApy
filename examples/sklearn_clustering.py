import pandas as pd
import napy
from sklearn.cluster import AgglomerativeClustering

# Read numerical data.
df = pd.read_csv('example_numerical.csv', index_col=0)

# Compute NA-aware Pearson Correlation with NApy.
pearson_results = napy.pearsonr(data=df,
                                nan_value=-99.0,
                                threads=1)
correlation_matrix = pearson_results['r2'].to_numpy()

# Turn correlations into distances and run hierarchical clustering.
distance_matrix = 1 - correlation_matrix
clustering = AgglomerativeClustering(
    n_clusters=3,
    metric="precomputed",
    linkage="average"
)
labels = clustering.fit_predict(distance_matrix)
print(labels)
