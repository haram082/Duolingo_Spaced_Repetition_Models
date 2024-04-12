import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# loading the data
sample_df = pd.read_csv("sample.csv")

# feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(sample_df[['delta', 'history_correct']])
sample_df[['delta', 'history_correct']] = scaled_features

# elbow method 
# inertia = []
# k_range = range(1, 11)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(scaled_features)
#     inertia.append(kmeans.inertia_)

# #plotting elbow
# plt.figure(figsize=(8,5))
# plt.plot(k_range, inertia, marker='o', color='blue')
# plt.title('Elbow Method for optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.xticks(k_range)
# plt.show()

#applying kmeans
kmeans = KMeans(n_clusters=3, random_state=0)
sample_df['cluster'] = kmeans.fit_predict(scaled_features)

# plotting clusters
plt.figure(figsize=(8,5))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = sample_df[sample_df['cluster'] == i]
    plt.scatter(cluster_data['delta'], cluster_data['history_correct'], c=colors[i], label=f'Cluster {i}', alpha=0.5)
plt.title('User clusters')
plt.xlabel('Delta')
plt.ylabel('History Correct')
plt.legend()
plt.show()

