import re
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Seed for reproducibility
np.random.seed(42)

# Cluster centers for bakery products (Brown Scale, Area in cm²)
clusters_spec = {
    "Semmel": {"mean": (0.3, 60), "std": (0.05, 5), "count": 50},
    "Poppy seed semmel": {"mean": (0.5, 60), "std": (0.05, 5), "count": 50},
    "Croissants": {"mean": (0.5, 80), "std": (0.05, 8), "count": 50},
    "Rye bread": {"mean": (0.8, 120), "std": (0.05, 10), "count": 150},
    "Wheat bread": {"mean": (0.55, 120), "std": (0.03, 10), "count": 50},
}

# threshold t for conflicting clusters
t = 1.4


# Data functions

def get_cluster_colors(num_clusters):
    """
    Generates list of cluster colors.
    """
    colormap = matplotlib.colormaps["Set2"]
    return [colormap(i / num_clusters) for i in range(num_clusters)]


def generate_data(clusters):
    """
    Generates synthetic data based on the defined cluster parameters.
    """
    data = []
    labels = []
    for label, params in clusters.items():
        x = np.random.normal(params["mean"][0], params["std"][0], params["count"])
        y = np.random.normal(params["mean"][1], params["std"][1], params["count"])
        data.append((x, y))
        labels.extend([label] * params["count"])

    return data, labels


def standardize_data(data):
    # Combine data into a 2D array
    x_all = np.concatenate([d[0] for d in data])
    y_all = np.concatenate([d[1] for d in data])
    points = np.column_stack((x_all, y_all))

    # Standardisation of data
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Break down standardised data back into separate lists
    standardized_data = []
    start = 0
    for d in data:
        count = len(d[0])
        standardized_data.append((points_scaled[start:start + count, 0], points_scaled[start:start + count, 1]))
        start += count

    return standardized_data, scaler


# PLot functions

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*\n\r\t]', '', filename).strip()


def plot_data(data_points, color_map, title="Plot of Bakery Data"):
    """
    Plots the scatter plot of the data points.
    """
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_all, y_all, c=x_all, cmap=color_map, edgecolor='k', s=80)

    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Add horizontal colorbar for Brown Scale
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0)
    cbar.set_label("Brown Scale (0 to 1)")

    plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


def plot_clusters(data_points, clusters, cluster_colors, color_map, title="Clustered Bakery Products"):
    """
    Plots the data points and highlights clusters with their centers and boundaries.
    """
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_all, y_all, c=x_all, cmap=color_map, edgecolor='k', s=80)

    # Mark cluster centers with circles
    for (label, params), color in zip(clusters.items(), cluster_colors):
        center_x, center_y = params["mean"]
        radius_x, radius_y = params["std"]
        circle = patches.Ellipse(
            (center_x, center_y), width=radius_x * 6, height=radius_y * 6,
            edgecolor=color, facecolor='none', lw=2, linestyle='--', label=f'{label} Cluster'
        )
        ax.add_patch(circle)
        ax.scatter(center_x, center_y, color=color, marker='x', s=100, zorder=5)

    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # Add horizontal colorbar for Brown Scale
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0)
    cbar.set_label("Brown Scale (0 to 1)")

    plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


def plot_kmeans_results(data_points, cluster_labels, centers, title="K-Means Clustering Results"):
    """
    Visualises the K-Means results with the cluster assignments and centres.
    """
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])

    # Scatterplot of the data points with cluster assignments
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each cluster separately to assign labels
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = get_cluster_colors(len(unique_clusters))

    for cluster, color in zip(unique_clusters, cluster_colors):
        cluster_points = (x_all[cluster_labels == cluster], y_all[cluster_labels == cluster])
        ax.scatter(*cluster_points, label=f"Cluster {int(cluster)}", color=color, s=80, edgecolor='k')

    # Mark the centroids
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")

    ax.set_xlabel("Brown Scale (0 to 1)")
    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.legend()
    plt.grid(alpha=0.3)

    plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


def plot_clusters_with_conflicts(centroids, clusters, conflicts, data_points, title="Clusters with Conflicts"):
    """
    Plots the clusters and highlights conflicting centroids.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of all data points
    cluster_colors = get_cluster_colors(len(centroids))

    for cluster_idx, color in zip(range(len(centroids)), cluster_colors):
        cluster_points = data_points[np.array(clusters) == cluster_idx]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f"Cluster {cluster_idx}")

    # Plot all centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=100, label="Centroids")

    # Highlight conflicting centroids
    for conflict in conflicts:
        ax.scatter(centroids[conflict, 0], centroids[conflict, 1], color="red", marker="o", s=150,
                   label=f"Conflicting Centroid {conflict}")

    ax.set_title(title)
    ax.set_xlabel("Brown scale (standardised)")
    ax.set_ylabel("Area (standardised)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


def plot_clusters_with_mega_cluster(centroids, clusters, mega_cluster, data_points, title="Clusters with Mega Cluster"):
    """
    Plots the clusters and highlights the mega cluster.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of all data points
    cluster_colors = get_cluster_colors(len(centroids))

    for cluster_idx, color in zip(range(len(centroids)), cluster_colors):
        cluster_points = data_points[np.array(clusters) == cluster_idx]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f"Cluster {cluster_idx}")

    # Plot all centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=100, label="Centroids")

    # Highlight mega cluster
    mega_cluster = np.array(mega_cluster)  # Ensure it's an array for easy handling
    ax.scatter(mega_cluster[:, 0], mega_cluster[:, 1], color="gold", marker="o", s=150, alpha=0.6,
               label="Mega Cluster")

    ax.set_title(title)
    ax.set_xlabel("Brown scale (standardised)")
    ax.set_ylabel("Area (standardised)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.savefig("plots/png/" + sanitize_filename(title) + ".png", format="png", dpi=300)
    plt.savefig("plots/svg/" + sanitize_filename(title) + ".svg", format="svg")
    plt.show()


# K-Means

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
    distances = np.zeros(K)
    for i in range(K):
        distances[i] = np.linalg.norm(centroids[i] - x)
    return np.argmin(distances)


def create_clusters(centroids, K, X):
    """Returns an array of cluster indices for all the data samples"""
    m, _ = np.shape(X)
    cluster_idx = np.empty(m)
    for i in range(m):
        cluster_idx[i] = closest_centroid(X[i], centroids, K)
    return cluster_idx


def compute_means(cluster_idx, K, X):
    """Computes and returns the new centroids of the clusters"""
    _, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        points = X[cluster_idx == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


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
    init_centroids = centroids
    i = 0

    centroids, clusters = converge_to_clusters(centroids, k, data_points, i)

    return centroids, clusters, init_centroids, create_clusters(init_centroids, k, data_points)


def run_kmeans_ref(data_points, n_clusters=5, random_state=42):
    """
        Executes the K-Means algorithm on the given data set.
    """

    # Combine all data points into a 2D array
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])
    points = np.column_stack((x_all, y_all))  # Data in (n_samples, 2)-Format

    # Execute K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(points)

    return kmeans.cluster_centers_, kmeans.labels_


def closest_neighbour(c, centroids):
    other_centroids = np.array([centroid for centroid in centroids if not np.array_equal(centroid, c)])
    distances = np.linalg.norm(other_centroids - c, axis=1)

    return other_centroids[np.argmin(distances)]


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
    init_centroids = centroids

    clusters = []

    for i in range(iterations):
        centroids, clusters = converge_to_clusters(centroids, k, data_points, i)
        conflicts = get_conflicts(centroids)
        mega_cluster = get_mega_cluster(k, clusters, data_points)

        if i == 0:
            plot_kmeans_results(std_data, clusters, centroids, "K-Means Result")

        print("Conflicting clusters: {}".format(conflicts))
        if len(conflicts) == 0:
            break

        plot_clusters_with_conflicts(centroids, clusters, conflicts, data_points,
                                     "Cluster with conflicts; iteration: {}".format(i))
        plot_clusters_with_mega_cluster(centroids, clusters, mega_cluster, data_points,
                                        "Mega cluster; iteration: {}".format(i))

        conflicting_centroid = random.choice(conflicts)
        random_instance = random.choice(mega_cluster)

        if i < iterations - 1:
            centroids[conflicting_centroid] = random_instance

    return centroids, clusters, init_centroids, create_clusters(init_centroids, k, data_points)


def k_means_improved_2(data_points, k, init_func, random_state=42, iterations=10):
    data_points = np.array(data_points)
    n_samples = data_points.shape[0]

    # Ensure that k is never bigger than the number of points
    k = min(k, n_samples)

    # Initialise cluster centroids wth the given init-function
    centroids = data_points[init_func(data_points, k, random_state)]

    clusters = []

    for i in range(iterations):
        clusters = create_clusters(centroids, k, data_points)

        conflicts = get_conflicts(centroids)
        mega_cluster = get_mega_cluster(k, clusters, data_points)

        print("Conflicting clusters: {}".format(conflicts))
        if len(conflicts) == 0:
            break

        plot_clusters_with_conflicts(centroids, clusters, conflicts, data_points,
                                     "Cluster with conflicts; iteration: {}".format(i))
        plot_clusters_with_mega_cluster(centroids, clusters, mega_cluster, data_points,
                                        "Mega cluster; iteration: {}".format(i))

        conflicting_centroid = random.choice(conflicts)
        random_instance = random.choice(mega_cluster)

        if i < iterations - 1:
            centroids[conflicting_centroid] = random_instance

    init_centroids = centroids
    centroids, clusters = converge_to_clusters(centroids, k, data_points, i)

    return centroids, clusters, init_centroids, create_clusters(init_centroids, k, data_points)


# Generate the data
data, _ = generate_data(clusters_spec)
std_data, scaler = standardize_data(data)
flat_data = [(x, y) for x_array, y_array in std_data for x, y in zip(x_array, y_array)]
# Main Code
if __name__ == "__main__":
    rand_state = random.randint(0, 1000)

    # Define colormap for Brown Scale
    cmap = matplotlib.colormaps["YlOrBr"]

    # Plot the data
    # plot_data(data, cmap, title="Scatter Plot of Bakery Data")

    # Plot the clusters with boundaries
    # plot_clusters(data, clusters_spec, get_cluster_colors(len(clusters_spec)), cmap, title="Clustered Bakery Products")

    # Calculate Clusters with k-means
    # centroids, clusters, init_centroids, init_clusters = k_means(flat_data,
    #                                                             len(clusters_spec),
    #                                                             rand_init,
    #                                                             rand_state)

    # Plot calculated clusters
    # plot_kmeans_results(data, init_clusters, scaler.inverse_transform(init_centroids), title="K-Means Init")
    # plot_kmeans_results(data, clusters, scaler.inverse_transform(centroids), title="K-Means Clustering Results")

    # Calculate Clusters with K-means++
    # centroids_plus, clusters_plus, init_centroids_plus, init_clusters_plus = k_means(flat_data,
    #                                                                              len(clusters_spec),
    #                                                                               kmeans_plus_plus_init,
    #                                                                                rand_state)

    # Plot calculated clusters with K-means++
    # plot_kmeans_results(data, init_clusters_plus, scaler.inverse_transform(init_centroids_plus), title="K-Means++ Init")
    # plot_kmeans_results(data, clusters_plus, scaler.inverse_transform(centroids_plus),
    #                    title="K-Means++ Clustering Results")

    # Calculate Clusters with K-means improved
    centroids_imp, clusters_imp, init_centroids_imp, init_clusters_imp = k_means_improved_2(flat_data,
                                                                                            len(clusters_spec),
                                                                                            rand_init,
                                                                                            rand_state)

    # Plot calculated clusters with K-means improved
    plot_kmeans_results(data, init_clusters_imp, scaler.inverse_transform(init_centroids_imp),
                        title="K-means improved Init")
    plot_kmeans_results(data, clusters_imp, scaler.inverse_transform(centroids_imp),
                        title="K-means improved Clustering Results")

    # Calculate Clusters with K-means reference impl
    # centroids, clusters = run_kmeans_ref(std_data, len(clusters_spec), rand_state)

    # Plot calculated clusters with reference impl
    # plot_kmeans_results(data, clusters, scaler.inverse_transform(centroids), title="Ref Clustering Results")
