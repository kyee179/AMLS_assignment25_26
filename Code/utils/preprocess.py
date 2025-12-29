import numpy as np
import torch
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


class ImagePreprocessor:
    """
    Handles Feature Engineering (HOG, PCA) for flattened inputs (SVM).
    """

    def __init__(self, use_hog=True, use_pca=True, pca_components=50):
        self.use_hog = use_hog
        self.use_pca = use_pca
        self.pca = PCA(n_components=pca_components)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def extract_hog(self, images):
        hog_features = []
        for img in images:
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=False,
            )
            hog_features.append(features)
        return np.array(hog_features)

    def fit_transform(self, X):
        if self.use_hog:
            X_processed = self.extract_hog(X)
        else:
            X_processed = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.fit_transform(X_processed)

        if self.use_pca:
            X_reduced = self.pca.fit_transform(X_scaled)
            self._is_fitted = True
            return X_reduced

        self._is_fitted = True
        return X_scaled

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data first.")

        if self.use_hog:
            X_processed = self.extract_hog(X)
        else:
            X_processed = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X_processed)

        if self.use_pca:
            return self.pca.transform(X_scaled)
        return X_scaled


class AddGaussianNoise(object):
    """
    Custom PyTorch transform to add Gaussian noise to a tensor.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_torch_augmentations():
    """
    Returns SPECIFIC augmentations requested:
    1. Brightness/Contrast
    2. Gaussian Noise
    """
    return {
        "pil": [
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
        ],
        "tensor": [
            AddGaussianNoise(mean=0.0, std=0.1),
        ],
    }


class NumpyAugmentations:
    @staticmethod
    def add_gaussian_noise(images, mean=0, std=10):
        noisy_images = images.copy().astype(np.float32)
        noise = np.random.normal(mean, std, images.shape)
        noisy_images += noise
        return np.clip(noisy_images, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_contrast(images, alpha=1.5):
        adjusted = images.astype(np.float32) * alpha
        return np.clip(adjusted, 0, 255).astype(np.uint8)
