import numpy as np
from sklearn.decomposition import PCA

class PCACurve:
    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.loadings = None
        self.explained_variance = None
        self.fitted = False

    def fit(self, yield_data: np.ndarray) -> None:
        """Fit PCA model to historical yield data (2D array: time points x maturities)."""
        self.pca.fit(yield_data)
        self.loadings = self.pca.components_
        self.explained_variance = self.pca.explained_variance_ratio_
        self.fitted = True

    def transform(self, yield_data: np.ndarray) -> np.ndarray:
        """Transform observed yield data into the space of principal components."""
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        return self.pca.transform(yield_data)

    def inverse_transform(self, components: np.ndarray) -> np.ndarray:
        """Reconstruct the yield data from principal components."""
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        return self.pca.inverse_transform(components)

    def reconstruct_yield(self, ytms: np.ndarray) -> np.ndarray:
        """Reconstruct the original YTMs from the principal components."""
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        ytms_transformed = self.transform(ytms.reshape(1, -1))  # Reshape for a single time point
        return self.inverse_transform(ytms_transformed)[0]  # Flatten the reconstructed yield

    def __call__(self, ytms: np.ndarray) -> np.ndarray:
        """Convenience method to call reconstruct_yield."""
        return self.reconstruct_yield(ytms)
