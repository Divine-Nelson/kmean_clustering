import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from decision_boundary import plot_decision_boundaries

#Data preparation
df = pd.read_csv("housing.csv")
X = df[["longitude", "latitude", "median_income"]].copy()

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans for k=1..9
"""kmeans_per_k = [
    KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    for k in range(1, 10)
]

# Compute silhouette scores for k=2..9
silhouette_scores = [
    silhouette_score(X_scaled, model.labels_)
    for model in kmeans_per_k[1:]
]

# Plot silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("Number of clusters (k)", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.title("Silhouette Analysis for K-Means")
plt.grid(True)
plt.show()
"""



"""best_k = 2

kmeans = KMeans(n_clusters=best_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)
"""
"""plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"], df["latitude"], df["median_income"],
    c=df["cluster"], cmap="tab10", s=50, alpha=0.6
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"K-Means Clustering (k={best_k}) on Housing Data")
plt.show()"""

X_2d = X_scaled[:, :2]


# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_2d)

# Plot decision boundaries
plt.figure(figsize=(8, 6))
plot_decision_boundaries(kmeans, X_2d, resolution=500)
plt.title("K-Means Decision Boundaries (Longitude vs Latitude)")
plt.show()
