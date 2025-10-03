import torch
import torch.nn as nn
from .blocks import TCN_block
from ..utils.constants import N, B, X, R, C

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

        self.mask = nn.Conv1d(
            in_channels = B,
            out_channels = N*C,
            kernel_size = 1,
            bias = True # True here so that it can easily bias the sigmoid without just relying on learning a large magnitude weight
        )

    def forward(self, x):
        # bottleneck layer
        x = self.bottleneck(x)

        # run thru separator and accumulate skip sum
        skip_sum = 0
        for blk in self.blocks:
            x, skip = blk(x)
            skip_sum = skip_sum + skip

        # run thru mask, sigmoid, reshape
        mask_abs = self.mask(skip_sum)
        mask_sig = torch.sigmoid(mask_abs)
        masks = mask_sig.view(
            x.size(0),  # number of batches
            C,          # number of sources
            N,          # number of features
            x.size(2)   # number of encoder time steps
        ).contiguous()  # force a contiguous in memory copy

        return masks
        