from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sample.csv")
sampled_df = df.sample(n=3000, random_state=42)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(sampled_df[['delta', 'history_correct']])

Z = linkage(scaled_features, method='ward')

# # find the best number of clusters using dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(Z)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data points')
# plt.ylabel('Euclidean distance')
# plt.show()

# agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
sampled_df['cluster'] = agg_clustering.fit_predict(scaled_features)

plt.figure(figsize=(10,7))
plt.scatter(sampled_df['delta'], sampled_df['history_correct'], c=sampled_df['cluster'], cmap='rainbow', alpha=0.6, edgecolors='w', s=100)
plt.title('Clusters from Hierarchical clustering')
plt.xlabel('Delta (Standardized)')
plt.ylabel('History Correct (Standardized)')
plt.colorbar(label='Cluster label')
plt.show()