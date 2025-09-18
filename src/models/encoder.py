import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ..utils.constants import SR, L, N

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Conv1d(
            in_channels = 1, # as audio is mono
            out_channels = N, # number of learned basis features
            kernel_size = L, # length of each filter
            stride = L//2, 
            bias = False
        )
        self.encoder_act = nn.ReLU()
    
    def forward(self,x):
        x = self.encoder_conv(x) # (batch, 1, time) -> (batch, N, (time-L)/stride + 1)
        x = self.encoder_act(x)
        return x
