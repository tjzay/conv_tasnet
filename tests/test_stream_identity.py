import torch
from src.utils.constants import SR, FRAME, HOP
from src.utils.stream import FrameCutter, OLA
from tests.test_enc_dec import snr_db

def run_identity(seed = 0):
    torch.manual_seed(seed)

    x = torch.randn(SR)

    fc = FrameCutter(frame=FRAME,hop=HOP)
    ola = OLA(frame=FRAME,hop=HOP)

    out_chunks = []
    # Weird chunk sizes to stress buffering
    CHUNK_SIZES = [257, 511, 73, 1024, 89, 640] 

    i = 0
    while i < x.numel():
        # select current n
        n = CHUNK_SIZES[i % len(CHUNK_SIZES)]
        chunk = x[i:i+n]
        fc.push(chunk)
        i += chunk.numel()

        # pull as many frames as ready
        while True:
            f = fc.pull()
            if f is None:
                break
            ola.add(f)

        y_hop = ola.read(HOP)
        if y_hop is not None and y_hop.numel() > 0:
            out_chunks.append(y_hop)

    # drain the end
    tail = ola.flush()
    if tail is not None and tail.numel() > 0:
            out_chunks.append(tail)

    y = torch.cat(out_chunks, dim=0)
    y = y[: x.numel()]
    x = x[: y.numel()]

    s=snr_db(x,y).item()
    print(f"Identity SNR: {s:.2f} dB")
    return s

def main():
     s = run_identity()

if __name__ == "__main__":
     main()
