import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.train import train_model
from utils.evaluate import test_model

from models.model import FeatureFusionDUNetCBAM

from utils.set_seed import set_seed
from utils.split_data import split_data
from utils.feature_ds import FeatureDataset
from utils.gt_img_ds import get_image_dataset
from utils.prepare_fe import prepare_features


from config import (
    IMAGE_FOLDER, BEST_MODEL_PATH,
    IMAGE_SIZE,
    N_PCA_COMPONENTS, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_LR, SEED
)


if __name__ == '__main__':
    set_seed(SEED)

    images, labels = get_image_dataset(IMAGE_FOLDER, image_size=IMAGE_SIZE)

    features, labels = prepare_features(images, labels, n_components=N_PCA_COMPONENTS)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(features, labels)

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=16, shuffle=False)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FeatureFusionDUNetCBAM(
        num_classes=len(set(labels)),
        feature_dim=N_PCA_COMPONENTS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=PATIENCE, verbose=True, min_lr=MIN_LR
    )


    train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        NUM_EPOCHS, device=device,
    )

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_model(model, test_loader, device=device)
    print("Training and evaluation complete.")
    print("Best model saved as 'best_model.pth'.")