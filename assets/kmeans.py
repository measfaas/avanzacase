# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:12:21 2024

@author: faas
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

file_path="C:/temp2/user-faas/avanza/Dataset till case - (2024).csv"

autokunder_data = pd.read_csv(file_path, delimiter=';')



# Data preparation
features = autokunder_data[['Totalt kapital på Avanza', 'Totalt kapital i Auto', 
                            'Kapital i aktier', 'Kapital i fonder (inklusive Auto)', 
                            'Inloggade dagar senaste månaden']].fillna(0)

# Standardizing features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Using the Elbow Method and Silhouette Score to determine optimal clusters
wcss = []  # Within-cluster sum of squares
silhouette_scores = []

# Testing different numbers of clusters
cluster_range = range(2, 11)
for k in cluster_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans_test.fit_predict(features_scaled)
    wcss.append(kmeans_test.inertia_)  # Sum of squared distances to closest cluster center
    
    # Calculate silhouette score for k > 1
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting the Elbow Method
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, wcss, marker='o', label='WCSS (Elbow Method)')
plt.title('Optimal antal kluster med Elbow-metoden')
plt.xlabel('Antal kluster')
plt.ylabel('WCSS')
plt.legend()
plt.grid()
plt.show()

# Plotting the Silhouette Score
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', color='orange', label='Silhouette Score')
plt.title('Optimal antal kluster med Silhouette Score')
plt.xlabel('Antal kluster')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid()
plt.show()



# Applying K-means clustering with 6 clusters
kmeans_9 = KMeans(n_clusters=9, random_state=42)
clusters_9 = kmeans_9.fit_predict(features_scaled)

# Perform PCA again for visualization with 6 clusters
features_pca_9 = PCA(n_components=2).fit_transform(features_scaled)

# Visualizing the 6 clusters
plt.figure(figsize=(10, 8))
for cluster_id in range(kmeans_9.n_clusters):
    plt.scatter(features_pca_9[clusters_9 == cluster_id, 0], 
                features_pca_9[clusters_9 == cluster_id, 1], 
                label=f'Cluster {cluster_id}', alpha=0.7)
plt.title('Kundsegmentering med K-means (9 Kluster, 2D PCA)')
plt.xlabel('PCA Komponent 1')
plt.ylabel('PCA Komponent 2')
plt.legend()
plt.grid()
plt.show()

# Assign clusters to the original dataset for further analysis
autokunder_data['Cluster'] = clusters_9