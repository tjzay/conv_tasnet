## AUDIO PARAMETERS
SR = 16000 # samples per second
FRAME_MS = 20 # ms
HOP_MS = 10 # ms
FRAME = SR * FRAME_MS // 1000 # number of samples 
HOP = SR * HOP_MS // 1000 # number of samples 

## MODEL PARAMETERS
"""N,B,H,X,R affect parameter count. See parameter_count.py for details"""

# size of the encoder in samples (smaller L -> better time resolution; larger L -> better frequency resolution)
# this also affects latency (L=32 -> 2ms)
L = 32 # kernel = L, stride = L//2
# CONSTRAINT: FRAME % (L//2) == 0 for overlap-add to align with encoder. i.e. each frame produces an integer number of encoder values

# number of encoder output channels (larger N is richer representation but more resources)
N = 256

# number of bottleneck channels after encoder before TCN (usually B=N)
B = 256 # reduce this for memory later

# number of hidden channels inside each TCN block
H = 512

# kernel size of depthwise convolution in each block
P = 3

# number of blocks per repeat
# X=6 means dilations [1,2,4,8,16,32]
X = 7

# number of repeats
R = 3

# number of output sources
C = 2
