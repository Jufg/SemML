import re
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*\n\r\t]', '', filename).strip()


# Well-separated Clusters
# Generate a data set with clearly separated clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# Applying K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=42, method='pam')  # Methode: Partitionierung um Medoide (PAM)
kmedoids.fit(X)
labels = kmedoids.labels_
medoids = kmedoids.cluster_centers_

# Visualisation of the clusters
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", s=50)

# mark Medoids
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids')

title = "Well-separated Clusters (K-Medoids)"
plt.title(title)
plt.legend(loc='best')

plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300, pad_inches=0, bbox_inches='tight')
plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
plt.show()

# Center-based Clusters
# Generate data: 4 spherical clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.5, center_box=(-10, 10), random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
centers = kmeans.cluster_centers_

# Visualisation
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

title = 'Center-based Clusters (K-Means)'
plt.title(title)
plt.legend()

plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300, pad_inches=0, bbox_inches='tight')
plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
plt.show()

# Contiguous Clusters
# Generate a data set with related clusters
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Hierarchical clustering (single linkage)
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = hierarchical.fit_predict(X)

# Visualisation
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", s=50)

title = 'Contiguous Clusters (Hierarchical Clustering - Single Linkage)'
plt.title(title)
plt.legend(loc='best')

plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300, pad_inches=0, bbox_inches='tight')
plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
plt.show()

# Density-based Clusters
# Generate data: Clusters with different density and noise
centers = [[2, 2], [8, 8], [15, 1]]
cluster_std = [0.5, 1.5, 0.2]
X_clusters, y_clusters = make_blobs(n_samples=[300, 150, 50],
                                    centers=centers,
                                    cluster_std=cluster_std,
                                    random_state=42)

# Add noise (random points)
X_noise = np.random.uniform(low=-2, high=18, size=(50, 2))
X = np.vstack([X_clusters, X_noise])

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.0, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualisation
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        color = 'black'
        label_name = "Noise"
    else:
        color = plt.cm.viridis(label / max(unique_labels))
        label_name = f"Cluster {label}"
    plt.scatter(X[labels == label, 0], X[labels == label, 1],
                c=[color], label=label_name, s=50)

title = 'Density-based Clusters (DBSCAN) with Noise'
plt.title(title)
plt.legend(loc='best')

plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300, pad_inches=0, bbox_inches='tight')
plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
plt.show()
