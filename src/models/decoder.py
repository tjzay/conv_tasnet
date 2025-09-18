import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ..utils.constants import SR, L, N

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels = N, # from learned basis features
            out_channels = 1, # to mono audio
            kernel_size = L, # length of each filter
            stride = L//2, 
            bias = False
        )
    
    def forward(self,x):
        x = self.decoder_conv(x)
        return x
