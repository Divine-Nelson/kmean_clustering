import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from decision_boundary import plot_dbscan_boundaries

# --- Data Preparation ---
df = pd.read_csv("housing.csv")
X = df[["longitude", "latitude", "median_income"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_2d = X_scaled[:, :2] 

# --- Apply DBSCAN ---
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan.fit(X_2d)

# --- Plot Decision Regions ---
plt.figure(figsize=(8, 6))
plot_dbscan_boundaries(dbscan, X_2d)
plt.title("DBSCAN Decision Boundaries (Longitude vs Latitude)")
plt.show()
