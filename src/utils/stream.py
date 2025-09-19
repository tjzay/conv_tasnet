"""This script defines FrameCutter and OLA classes. 
FrameCutter splits continuous audio into frames of size FRAME.
OLA takes the individual frames and stitches them together into continuous audio.
"""

from .constants import FRAME, HOP
import torch
import matplotlib.pyplot as plt

class FrameCutter:
    def __init__(self,frame,hop):
        self.frame = frame
        self.hop = hop
        self.buffer = None

    def push(self, x):
        assert x.ndim == 1, f"push expects 1D mono, got shape {tuple(x.shape)}"
        # Add incoming stream to the buffer
        if self.buffer is None:
            # create empty buffer on same device/dtype as x
            self.buffer = x.new_zeros(0)
        self.buffer = torch.cat([self.buffer, x])

    def pull(self):
        if self.buffer is None or self.buffer.numel() < self.frame :
            return None
        # output a frame and advance buffer by hop
        y = self.buffer[:self.frame]
        self.buffer = self.buffer[self.hop:]
        return y
    
class OLA:
    def __init__(self,frame,hop):
        self.frame = frame
        self.hop = hop
        self.window = None
        self.stream = None
        self.write_pos = 0
        self.read_pos = 0


    def add(self, frame):
        if self.window is None:
            self.window = torch.hann_window(self.frame, periodic=True,
                                            device=frame.device, dtype=frame.dtype)
        # Classic OLA algorithm
        if self.window is not None:
            frame = frame * self.window
        if self.stream is None:
            self.stream = frame.clone()
            self.write_pos = self.hop
            return
        
        end_pos = self.write_pos + self.frame
        if end_pos > self.stream.numel():
            extra = end_pos - self.stream.numel()
            self.stream = torch.cat([self.stream, frame.new_zeros(extra)])
        
        self.stream[self.write_pos:end_pos] += frame
        self.write_pos += self.hop

    
    def pending(self):
        # Samples that are finalized and safe to read:
        # everything up to write_pos - (frame - hop)
        finalized_end = self.write_pos - (self.frame - self.hop)
        return max(0, finalized_end - self.read_pos)
    
    def read(self, n):
        # returns n contiguous samples if ready, leaves the rest for later
        # if not enough ready samples, return the maximum
        avail = self.pending()
        if avail <= 0:
            return None
        
        n = min(n, avail)

        y = self.stream[self.read_pos:self.read_pos+n].clone()
        self.read_pos += n

        # compact the buffer to avoid unbounded growth
        if self.read_pos > 4 * self.frame:
            # keep only the unread tail [read_pos:write_pos]
            tail = self.stream[self.read_pos:self.write_pos].contiguous()
            self.stream = tail
            self.write_pos -= self.read_pos
            self.read_pos = 0

        return y
    
    def flush(self):
        return self.read(self.pending())
    
    def reset(self):
        self.window = None
        self.stream = None
        self.write_pos = 0
        self.read_pos = 0

def main():
    FRAME, HOP = 320, 160
    n = torch.arange(100000)
    x = torch.sin(n)

    fc  = FrameCutter(FRAME, HOP)
    ola = OLA(FRAME, HOP)

    for i in range(0, len(x), HOP):
        fc.push(x[i:i+HOP])
        f = fc.pull()
        if f is not None:
            ola.add(f)

    y = ola.flush()
    print(len(x))
    print(len(y))
    y = y[:len(x)]
    x = x[:len(y)]

    k = len(y)

    plt.figure()
    plt.plot(n[:k],x[:k], label = 'original')
    plt.plot(n[:k],y[:k], label = 'processed')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

