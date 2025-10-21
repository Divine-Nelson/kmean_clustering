import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_dbscan_boundaries(dbscan, X, resolution=500, show_xlabels=True, show_ylabels=True):
    labels = dbscan.labels_
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    core_points = X[core_mask]
    core_labels = labels[core_mask]

    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Use NearestNeighbors to assign each grid point to the nearest core point
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(core_points)
    nearest_core_idx = nn.kneighbors(grid_points, return_distance=False).flatten()
    Z = core_labels[nearest_core_idx]
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap="Pastel1", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], c="red", marker="x", s=80, label="Noise")

    plt.title(f"DBSCAN Decision Regions (eps={dbscan.eps}, min_samples={dbscan.min_samples})")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=12)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=12)
    plt.legend()