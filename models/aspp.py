import torch
import torch.nn as nn

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 1)
        self.c6 = nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6)
        self.c12 = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12)
        self.c18 = nn.Conv2d(in_c, out_c, 3, padding=18, dilation=18)
        self.proj = nn.Conv2d(out_c * 4, out_c, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c6(x)
        x3 = self.c12(x)
        x4 = self.c18(x)
        return self.proj(torch.cat([x1, x2, x3, x4], dim=1))
