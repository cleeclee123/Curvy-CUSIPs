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
        self.pca.fit(yield_data)
        self.loadings = self.pca.components_
        self.explained_variance = self.pca.explained_variance_ratio_
        self.fitted = True

    def transform(self, yield_data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        return self.pca.transform(yield_data)

    def inverse_transform(self, components: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        return self.pca.inverse_transform(components)

    def yield_curve(self, tau: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PCA model has not been fitted yet.")
        components = self.transform(tau)
        return self.inverse_transform(components)

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        return self.yield_curve(tau)
