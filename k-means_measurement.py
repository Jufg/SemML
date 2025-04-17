import random
import re
import numpy as np
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
# threshold
t = 1.2


# Plots
def get_cluster_colors(num_clusters):
    """
    Generates list of cluster colors.
    """
    colormap = matplotlib.colormaps["Set2"]
    return [colormap(i / num_clusters) for i in range(num_clusters)]


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*\n\r\t]', '', filename).strip()


def plot_kmeans_results(data_points, cluster_labels, centers, title="K-Means Clustering Results"):
    """
    Visualises the K-Means results with the cluster assignments and centres.
    """
    x_all = np.array([d[0] for d in data_points])
    y_all = np.array([d[1] for d in data_points])

    # Scatterplot of the data points with cluster assignments
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each cluster separately to assign labels
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = get_cluster_colors(len(unique_clusters))

    for cluster, color in zip(unique_clusters, cluster_colors):
        cluster_points = (x_all[cluster_labels == cluster], y_all[cluster_labels == cluster])
        ax.scatter(*cluster_points, label=f"Cluster {int(cluster)}", color=color, s=20, alpha=0.7)

    # Mark the centroids
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")

    ax.set_xlabel("Brown Scale (0 to 1)")
    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.legend()
    plt.grid(alpha=0.3)

    # plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    # plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


# K-means Implementation
def rand_init(data_points, k, random_state):
    np.random.seed(random_state)

    return np.random.choice(data_points.shape[0], k, replace=False)


def kmeans_plus_plus_init(data_points, k, random_state):
    """
    Initialises cluster centres with the K-Means++ algorithm.
    Returns the indices of the selected cluster centres.
    """
    np.random.seed(random_state)

    # Choose the first centre at random from the data points
    centers_indices = [np.random.choice(data_points.shape[0])]

    # Repeat the selection of centres
    for _ in range(1, k):
        # Calculate the distances from each point to the nearest centre
        distances = np.min(np.linalg.norm(data_points[:, np.newaxis] - data_points[centers_indices], axis=2), axis=1)

        # Select the next point with a probability proportional to the square of the distance
        probs = distances ** 2
        probs /= probs.sum()

        # Select a new centre point based on the probabilities
        new_center_index = np.random.choice(data_points.shape[0], p=probs)
        centers_indices.append(new_center_index)

    return centers_indices


def closest_centroid(x, centroids, K):
    """Finds and returns the index of the closest centroid for a given vector x"""
    distances = np.linalg.norm(centroids - x, axis=1)
    return np.argmin(distances)


def create_clusters(centroids, K, X):
    """Returns an array of cluster indices for all the data samples"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def compute_means(cluster_idx, K, X):
    """Computes and returns the new centroids of the clusters"""
    return np.array([X[cluster_idx == i].mean(axis=0) for i in range(K)])


def converge_to_clusters(centroids, k, data_points, i=0):
    while True:
        # Create clusters
        clusters = create_clusters(centroids, k, data_points)
        prev_centroids = centroids

        # Calculate new cluster centroids as the mean value of the associated points
        centroids = compute_means(clusters, k, data_points)

        if np.array_equal(centroids, prev_centroids):
            break

        i = i + 1

    return centroids, clusters


def k_means(data_points, k, init_func, random_state=42):
    data_points = np.array(data_points)
    n_samples = data_points.shape[0]

    # Ensure that k is never bigger than the number of points
    k = min(k, n_samples)

    # Initialise cluster centroids with the given init function
    centroids = data_points[init_func(data_points, k, random_state)]
    i = 0

    centroids, clusters = converge_to_clusters(centroids, k, data_points, i)

    return centroids, clusters


def closest_neighbour(c, centroids):
    # Compute all distances from c to the centroids
    distances = np.linalg.norm(centroids - c, axis=1)

    # Set the distance to itself to infinity to exclude it from the minimum
    distances[np.isclose(distances, 0)] = np.inf

    # Find the index of the minimum distance
    closest_idx = np.argmin(distances)

    return centroids[closest_idx]


def avg_dist(centroids):
    # avgDist = 1/k * sum(d(c_i, d_neigh))
    return 1 / len(centroids) * np.array([np.linalg.norm(c - closest_neighbour(c, centroids)) for c in centroids]).sum()


def is_conflicting(c_i, c_neigh, avgDist):
    return np.linalg.norm(c_i - c_neigh) < (avgDist / t)


def get_conflicts(centroids):
    conflicts = []
    avg_Dist = avg_dist(centroids)

    for c_idx in range(len(centroids)):
        c_i = centroids[c_idx]
        closest = closest_neighbour(c_i, centroids)

        if is_conflicting(c_i, closest, avg_Dist):
            conflicts.append(c_idx)

    return np.array(conflicts)


def cluster_variance(cluster_points):
    mean_point = np.mean(cluster_points, axis=0)
    squared_deviations = np.sum((cluster_points - mean_point) ** 2, axis=1)

    # Variance with Bessel correction (n-1)
    variance = np.sum(squared_deviations) / (cluster_points.shape[0] - 1)

    return variance


def get_mega_cluster(k, cluster_idx, data_points):
    clusters = [[] for _ in range(k)]

    for point, cluster_index in zip(data_points, cluster_idx):
        clusters[int(cluster_index)].append(point)

    return clusters[np.argmax([cluster_variance(np.array(cluster)) for cluster in clusters])]


def k_means_improved(data_points, k, init_func, random_state=42, iterations=10):
    data_points = np.array(data_points)
    n_samples = data_points.shape[0]

    # Ensure that k is never bigger than the number of points
    k = min(k, n_samples)

    # Initialise cluster centroids wth the given init-function
    centroids = data_points[init_func(data_points, k, random_state)]

    clusters = []

    for i in range(iterations):
        centroids, clusters = converge_to_clusters(centroids, k, data_points, i)
        conflicts = get_conflicts(centroids)
        mega_cluster = get_mega_cluster(k, clusters, data_points)

        # print("Conflicting clusters: {}".format(conflicts))
        if len(conflicts) == 0:
            break

        # plot_clusters_with_conflicts(centroids, clusters, conflicts, data_points,
        #                             "Cluster with conflicts; iteration: {}".format(i))
        # plot_clusters_with_mega_cluster(centroids, clusters, mega_cluster, data_points,
        #                                "Mega cluster; iteration: {}".format(i))

        conflicting_centroid = random.choice(conflicts)
        random_instance = random.choice(mega_cluster)

        if i < iterations - 1:
            centroids[conflicting_centroid] = random_instance

    return centroids, clusters

# read data
s1_data = pd.read_csv("data/S-Sets/s3.txt", header=None, sep='\s+')
s1_data.columns = ['x', 'y']

s1_labels = pd.read_csv("data/S-Sets/s-originals/s3-label.pa",
                        header=None,
                        skiprows=5,
                        names=["label"])

assert len(s1_labels) == len(s1_data)

# Format data
X = s1_data[['x', 'y']].dropna()
true_labels = s1_labels["label"].tolist()

dataset_name = "S4 - S-Sets"
num_runs = 10
k = 15

# Prepare results lists
silhouette_scores = {"kmeans": [], "improved": [], "kmeans++": []}
homogeneity_scores = {"kmeans": [], "improved": [], "kmeans++": []}

# Method configuration: (name, function, initializer)
methods = [
    ("kmeans", k_means, rand_init),
    ("kmeans++", k_means, kmeans_plus_plus_init),
    ("improved", k_means_improved, rand_init)
]

# Measurements
for i in tqdm(range(num_runs), desc="Runs", ncols=100):
    rand_state = i

    for name, method_func, init_func in methods:
        centroids, labels = method_func(X, k, init_func, rand_state)

        silhouette = silhouette_score(X, labels)
        homogeneity = homogeneity_score(true_labels, labels)

        silhouette_scores[name].append(silhouette)
        homogeneity_scores[name].append(homogeneity)


# Calculation and output of average scores as well as differences and variances
# Output of metadata
print(f"\n=== Dataset: {dataset_name} ===")
print(f"⟶ k: {k}")
print(f"⟶ Number of runs: {num_runs}")
print(f"⟶ Threshold: {t}")


for name in silhouette_scores:
    avg_sil = np.mean(silhouette_scores[name])
    avg_hom = np.mean(homogeneity_scores[name])

    # Calculate differences (nominal and percentage)
    if name == "improved":
        sil_diff = avg_sil - np.mean(silhouette_scores["kmeans++"])
        hom_diff = avg_hom - np.mean(homogeneity_scores["kmeans++"])

        sil_diff_percent = (sil_diff / np.mean(silhouette_scores["kmeans++"])) * 100
        hom_diff_percent = (hom_diff / np.mean(homogeneity_scores["kmeans++"])) * 100
    else:
        sil_diff = hom_diff = sil_diff_percent = hom_diff_percent = 0

    # Calc variance
    var_sil = np.var(silhouette_scores[name])
    var_hom = np.var(homogeneity_scores[name])

    # Output of the results
    print(f"\n=== {name.upper()} ===")
    print(f"⟶ Average Silhouette-Score: {avg_sil:.4f}")
    print(f"⟶ Average Homogeneity-Score: {avg_hom:.4f}")

    # Differences (nominal and percentage)
    if name == "improved":
        print(f"⟶ Difference in Silhouette-Score (nominal) vs. K-Means++: {sil_diff:.4f}")
        print(f"⟶ Difference in Homogeneity-Score (nominal) vs. K-Means++: {hom_diff:.4f}")
        print(f"⟶ Difference in Silhouette-Score (percentage) vs. K-Means++: {sil_diff_percent:.2f}%")
        print(f"⟶ Difference in Homogeneity-Score (percentage) vs. K-Means++: {hom_diff_percent:.2f}%")

    # Variance
    print(f"⟶ Variance of the Silhouette-Scores: {var_sil}")
    print(f"⟶ Variance of the Homogeneity-Scores: {var_hom}")

