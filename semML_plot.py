import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
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
    # Kombiniere Daten zu einem 2D-Array
    x_all = np.concatenate([d[0] for d in data])
    y_all = np.concatenate([d[1] for d in data])
    points = np.column_stack((x_all, y_all))  # Daten in (n_samples, 2)-Format

    # Standardisierung der Daten
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Zerlege standardisierte Daten zurück in getrennte Listen
    standardized_data = []
    start = 0
    for d in data:
        count = len(d[0])
        standardized_data.append((points_scaled[start:start + count, 0], points_scaled[start:start + count, 1]))
        start += count

    return standardized_data, scaler


def plot_data(data_points, color_map, title="Scatter Plot of Bakery Data"):
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

    plt.show()


def plot_kmeans_results(data_points, cluster_labels, centers, title="K-Means Clustering Results"):
    """
    Visualises the K-Means results with the cluster assignments and centres.
    """
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])

    # Scatterplot der Datenpunkte mit Clusterzuweisungen
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_all, y_all, c=cluster_labels, cmap="tab10", s=80, edgecolor='k')

    # Markiere die Zentren
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centroids")

    ax.set_xlabel("Brown Scale (0 to 1)")
    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.legend()
    plt.grid(alpha=0.3)
    plt.show()


#data, _ = generate_data(clusters_spec)


def rand_init(data_points, k, random_state):
    np.random.seed(random_state)

    return np.random.choice(data_points.shape[0], k, replace=False)


def kmeans_plus_plus_init(data_points, k, random_state):
    """
    Initialisiert Clusterzentren mit dem K-Means++ Algorithmus.
    Gibt die Indizes der ausgewählten Clusterzentren zurück.
    """
    np.random.seed(random_state)

    # 1. Wähle das erste Zentrum zufällig aus den Datenpunkten
    centers_indices = [np.random.choice(data_points.shape[0])]

    # 2. Wiederhole die Auswahl der Zentren
    for _ in range(1, k):
        # Berechne die Abstände jedes Punktes zum nächsten Zentrum
        distances = np.min(np.linalg.norm(data_points[:, np.newaxis] - data_points[centers_indices], axis=2), axis=1)

        # Wähle den nächsten Punkt mit einer Wahrscheinlichkeit proportional zum Quadrat des Abstands
        probs = distances ** 2
        probs /= probs.sum()  # Normiere die Wahrscheinlichkeiten

        # Wähle einen neuen Mittelpunkt basierend auf den Wahrscheinlichkeiten
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
        points = X[cluster_idx == i]  # gather points for the cluster i
        centroids[i] = np.mean(points, axis=0)  # use axis=0 to compute means across points
    return centroids


def k_means(data_points, k, init_func, random_state=42):
    data_points = np.array(data_points)
    n_samples = data_points.shape[0]

    # Ensure that k is never bigger than the number of points
    k = min(k, n_samples)

    # Initialise cluster centres randomly from the data points
    centroids = data_points[init_func(data_points, k, random_state)]
    i = 0

    while True:
        # Create clusters
        clusters = create_clusters(centroids, k, data_points)
        prev_centroids = centroids

        #plot_kmeans_results(data, clusters, centroids, title=(str(i) + " iteration"))

        # Calculate new cluster centres as the mean value of the associated points
        # new_centres = np.array([data_points[cluster_labels == k].mean(axis=0) for k in range(k)])
        centroids = compute_means(clusters, k, data_points)

        if np.array_equal(centroids, prev_centroids):
            break

        i = i + 1

    return centroids, clusters


def run_kmeans_ref(data_points, n_clusters=5, random_state=42):
    """
    Führt den K-Means Algorithmus auf den gegebenen Datensatz aus.
    """
    # Kombiniere alle Datenpunkte zu einem 2D-Array
    x_all = np.concatenate([d[0] for d in data_points])
    y_all = np.concatenate([d[1] for d in data_points])
    points = np.column_stack((x_all, y_all))  # Daten in (n_samples, 2)-Format

    # Führe den K-Means-Algorithmus aus
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(points)

    # Ergebnisse des K-Means
    labels = kmeans.labels_  # Clusterzuweisungen für jeden Punkt
    centers = kmeans.cluster_centers_  # Berechnete Clusterzentren

    return centers, labels


# Main Code
if __name__ == "__main__":
    rand_state = random.randint(0, 1000)

    # Generate the data
    data, _ = generate_data(clusters_spec)
    std_data, scaler = standardize_data(data)

    # Define colormap for Brown Scale
    cmap = matplotlib.colormaps["YlOrBr"]

    # Plot the data
    plot_data(data, cmap, title="Scatter Plot of Bakery Data")

    # Plot the clusters with boundaries
    plot_clusters(data, clusters_spec, get_cluster_colors(len(clusters_spec)), cmap, title="Clustered Bakery Products")

    # Calculate Clusters with k-means
    # flat data
    flat_data = [(x, y) for x_array, y_array in std_data for x, y in zip(x_array, y_array)]
    centroids, clusters = k_means(flat_data, len(clusters_spec), rand_init, rand_state)

    # Plot calculated clusters
    plot_kmeans_results(data, clusters, scaler.inverse_transform(centroids), title="K-Means Clustering Results")

    # Calculate Clusters with K-means++
    centroids, clusters = k_means(flat_data, len(clusters_spec), kmeans_plus_plus_init, rand_state)

    # Plot calculated clusters with K-means++
    plot_kmeans_results(data, clusters, scaler.inverse_transform(centroids), title="K-Means++ Clustering Results")

    # Calculate Clusters with K-means reference impl
    centroids, clusters = run_kmeans_ref(std_data, len(clusters_spec), rand_state)

    # Plot calculated clusters with reference impl
    plot_kmeans_results(data, clusters, scaler.inverse_transform(centroids), title="Ref Clustering Results")
