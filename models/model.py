import torch
import numpy as np
import torch.nn as nn
from models.resnet_enc import ResNetEncoder
from models.decoder import Decoder
from models.cbam import CBAM
from models.aspp import ASPP





class FeatureFusionDUNetCBAM(nn.Module):
    def __init__(self, num_classes=2, feature_dim=50, image_channels=1, image_size=(128, 128)):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.encoder = ResNetEncoder(in_channels=64)
        self.decoder = Decoder()
        self.cbam = CBAM(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.aspp = ASPP(128, 64)

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        )

        self.final_fc = nn.Linear(64, num_classes)

    def forward(self, features):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        features = features.to(next(self.parameters()).device)

        feat_map = self.feature_proj(features).view(-1, 64, 32, 32)

        x3, x2, x0 = self.encoder(feat_map)
        x3 = self.aspp(x3)
        x = self.decoder(x3, x2, x0)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.final_fc(x)

