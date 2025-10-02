import torch
import torch.nn as nn
from ..utils.constants import B, H, P

class TCN_block(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv1x1_in = nn.Conv1d(
            in_channels = B,
            out_channels = H,
            kernel_size = 1,
            bias = False
        )

        self.pad = (P - 1) * self.dilation
        self.depthwise = nn.Conv1d(
            in_channels = H,
            out_channels = H,
            kernel_size = P,
            dilation = self.dilation,
            groups = H,
            padding = self.pad,
            bias = False
        )

        self.norm = nn.GroupNorm(1,H)
        self.prelu = nn.PReLU()

        self.residual = nn.Conv1d(
            in_channels = H,
            out_channels = B,
            kernel_size = 1,
            bias = False
        )

        self.skip = nn.Conv1d(
            in_channels = H,
            out_channels = B,
            kernel_size = 1,
            bias = False
        )

    def forward(self,x):
        out = self.conv1x1_in(x)
        out = self.depthwise(out)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        out = self.norm(out)
        out = self.prelu(out)
        residual = self.residual(out)
        skip = self.skip(out)

        return x + residual, skip

