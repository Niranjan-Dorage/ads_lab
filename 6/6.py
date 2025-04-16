# pip install pandas scikit-learn matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

# Step 1: Load the data from CSV
data = pd.read_csv('data.csv')

# Step 2: KNN-based Outlier Detection
X = data[['Feature1', 'Feature2']].values

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X)
distances, _ = knn.kneighbors(X)

# Calculate the mean distance for each point and consider points with high distance as outliers
mean_distances = distances.mean(axis=1)
threshold_knn = np.percentile(mean_distances, 95)  # Outliers are above the 95th percentile
outliers_knn = mean_distances > threshold_knn

# Step 3: LOF-based Outlier Detection
lof = LocalOutlierFactor(n_neighbors=10)
outliers_lof = lof.fit_predict(X)
outliers_lof = outliers_lof == -1  # LOF labels outliers as -1

# Step 4: Plot the data and highlight outliers
plt.figure(figsize=(10, 6))

# Plot all points
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Normal Points')

# Plot KNN-based outliers in red
plt.scatter(X[outliers_knn, 0], X[outliers_knn, 1], color='red', label='KNN Outliers')

# Plot LOF-based outliers in green
plt.scatter(X[outliers_lof, 0], X[outliers_lof, 1], color='green', label='LOF Outliers')

plt.title('Outlier Detection using KNN and LOF')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
