"""Small early-fusion U-Net for a binary, directional land-cover transition."""

import torch
from torch import nn


def conv(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU())


class ChangeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.full = conv(12, 16)
        self.half_res = conv(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(conv(32, 64), conv(64, 64))
        self.up_half = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decode_half = conv(64, 32)
        self.up_full = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decode_full = conv(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, pair):
        full = self.full(pair)
        half = self.half_res(self.pool(full))
        coarse = self.bridge(self.pool(half))
        half_out = self.decode_half(torch.cat((self.up_half(coarse), half), dim=1))
        full_out = self.decode_full(torch.cat((self.up_full(half_out), full), dim=1))
        return self.head(full_out).squeeze(1)
