from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_extraction.feature_extraction import extract_all_features


def prepare_features(images, labels, n_components=75):
    handcrafted = extract_all_features(images)
    scaled = StandardScaler().fit_transform(handcrafted)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)

    return reduced, labels