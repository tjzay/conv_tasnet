import torch, math
from src.utils.constants import SR, L, N
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.utils.audio_check import generate_sine

def snr_db(x,y):
    num = (x**2).sum()
    den = ((x-y)**2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(num / den)

def main():
    B, T = 2, SR
    # generate B batches of the same sine wave
    x_sine = generate_sine().repeat(B,1) # (B,T)
    x_noise = torch.randn(B,T) * 0.2 # (B,T)

    enc, dec = Encoder(), Decoder()

    for name, x in [("sine", x_sine), ("noise", x_noise)]:
        x = x.unsqueeze(1) # (B,1,T)
        E = enc(x) # (B,N,T')
        Tprime = (T-L) // (L//2) + 1
        assert E.shape == (B, N, Tprime), f"E {E.shape} vs {(B,N,Tprime)}"

        y = dec(E)                                  # -> (B,1,T_out)
        Tout = (Tprime-1) * (L//2) + L
        assert y.shape == (B, 1, Tout), f"y {y.shape} vs {(B,1,Tout)}"
        assert Tout == T, "Length mismatch (choose T aligned with stride=L//2)"

        s = snr_db(x,y).mean().item()
        print(f"{name} SNR: {s:.2f} dB")

    print("Encoder/Decoder round-trip ran ✅")

if __name__ == "__main__":
    main()
