import torch
import torch.nn as nn
from .blocks import TCN_block
from ..utils.constants import N, B, X, R

# Create bottleneck layer (converts from N features (encoder output) -> B)
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Conv1d(
            in_channels = N,
            out_channels = B,
            kernel_size = 1,
            bias = False
        )
    
    def forward(self, x):
        return self.bottleneck(x)
    

class Separator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = Bottleneck()
        blocks = []
        for _ in range(R):
            for i in range(X):
                blocks.append(TCN_block(dilation=2 ** i))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.bottleneck(x)
        skip_sum = None
        for blk in self.blocks:
            x, skip = blk(x)
            skip_sum = skip_sum + skip
        return skip_sum
        