import numpy as np
import torch, random
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_extraction.feature_extraction import extract_all_features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_image_dataset(image_folder, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)

    images, labels = [], []
    for img_pil, label in dataset:
        images.append(img_pil.squeeze().numpy())
        labels.append(label)

    return images, labels


def prepare_features(images, labels, n_components=75):
    handcrafted = extract_all_features(images)
    scaled = StandardScaler().fit_transform(handcrafted)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)

    return reduced, labels


def split_data(features, labels, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)