import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data (3 correlated variables)
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)

# Display results
print("Transformed Data:")
print(X_pca)

# Visualize the data in 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA: 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()