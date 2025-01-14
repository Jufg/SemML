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
    "Croissants": {"mean": (0.5, 80), "std": (0.05, 8), "count": 30},
    "Rye bread": {"mean": (0.8, 120), "std": (0.05, 10), "count": 150},
    "Wheat bread": {"mean": (0.55, 120), "std": (0.03, 10), "count": 30},
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

def plot_data(data, cmap, title="Scatter Plot of Bakery Data"):
    """
    Plots the scatter plot of the data points.
    """
    x_all = np.concatenate([d[0] for d in data])
    y_all = np.concatenate([d[1] for d in data])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_all, y_all, c=x_all, cmap=cmap, edgecolor='k', s=80)

    ax.set_ylabel("Area (cm$^2$)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Add horizontal colorbar for Brown Scale
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0)
    cbar.set_label("Brown Scale (0 to 1)")

    plt.show()

def plot_clusters(data, clusters, cluster_colors, cmap, title="Clustered Bakery Products"):
    """
    Plots the data points and highlights clusters with their centers and boundaries.
    """
    x_all = np.concatenate([d[0] for d in data])
    y_all = np.concatenate([d[1] for d in data])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_all, y_all, c=x_all, cmap=cmap, edgecolor='k', s=80)

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

# Main Code
if __name__ == "__main__":
    # Generate the data
    data, labels = generate_data(clusters_spec)

    # Define colormap for Brown Scale
    cmap = matplotlib.colormaps["YlOrBr"]

    # Plot the data
    plot_data(data, cmap, title="Scatter Plot of Bakery Data")

    # Plot the clusters with boundaries
    plot_clusters(data, clusters_spec, get_cluster_colors(len(clusters_spec)), cmap, title="Clustered Bakery Products")
