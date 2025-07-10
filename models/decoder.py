import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.cb3 = self._conv_block(64+64, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.cb4 = self._conv_block(64+64, 64)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x3, x2, x0):
        x = self.cb3(torch.cat([self.up3(x3), x2], dim=1))
        return self.cb4(torch.cat([self.up4(x), x0], dim=1))