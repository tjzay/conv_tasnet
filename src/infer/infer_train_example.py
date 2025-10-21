import torch
import soundfile as sf
import os
import random
import numpy as np

from src.data.mix_dataset import MUSDBDataset
from src.models.encoder import Encoder
from src.models.separator import Separator
from src.models.decoder import Decoder
from src.utils.constants import N

@torch.no_grad()
def infer_one_example(
    ckpt_path="checkpoints/checkpoint_last.pt",
    root="/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data/musdb18",
    sr=16000,
    segment_seconds=10,
    device="cpu",
    out_dir="/Users/tobyzayontz/Documents/Uni/conv_tasnet/inference_out/third_training",
    seed=1337,
):
    # --- reproducible random crop selection ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS backend doesn't expose per-device seeding, but global torch.manual_seed covers it.
    # For stronger determinism during inference, enforce deterministic algorithms if available.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # --- load checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device)
    enc, sep, dec = Encoder().to(device), Separator().to(device), Decoder().to(device)
    enc.load_state_dict(ckpt["enc"]); sep.load_state_dict(ckpt["sep"]); dec.load_state_dict(ckpt["dec"])
    enc.eval(); sep.eval(); dec.eval()

    # --- pick one training example ---
    ds = MUSDBDataset(root=root, split="train", sr=sr, segment_seconds=segment_seconds)
    mix, sources = ds[10]   # mix: (1,T), sources: (2,T)
    mix = mix.unsqueeze(0).to(device).float()     # (B=1,1,T)
    sources = sources.unsqueeze(0).to(device).float() # (1,2,T)

    # --- forward pass ---
    z = enc(mix)                         # (1, N, T′)
    masks = sep(z)                       # (1, C, N, T′)
    masked = masks * z.unsqueeze(1)      # (1, C, N, T′)
    B, C_, N_, Tp = masked.shape
    yhat = dec(masked.view(B*C_, N_, Tp))# (B*C,1,T)
    if yhat.ndim == 3 and yhat.size(1) == 1:
        yhat = yhat.squeeze(1)           # (B*C, T)
    yhat = yhat.view(B, C_, -1)          # (1,2,T)

    # --- crop to match target length (keep on-device for math) ---
    T = min(yhat.size(-1), sources.size(-1), mix.size(-1))
    yhat = yhat[..., :T]      # (1,2,T) on `device`
    mix  = mix[..., :T]       # (1,T)   on `device`

    # --- write out audio ---
    os.makedirs(out_dir, exist_ok=True)

    # compute mixture-consistency error on-device (same dtype/device), then move to CPU for saving
    err = (mix - (yhat[0,0] + yhat[0,1])).abs().mean().item()
    print("mix-sum(sources) MAE:", err)

    # move cropped tensors to CPU only for writing
    mix_cpu = mix.squeeze().detach().cpu().numpy()
    vox_cpu = yhat[0,0].detach().cpu().numpy()
    acc_cpu = yhat[0,1].detach().cpu().numpy()

    sf.write(os.path.join(out_dir, "mix.wav"), mix_cpu, sr)
    sf.write(os.path.join(out_dir, "vocals.wav"), vox_cpu, sr)
    sf.write(os.path.join(out_dir, "accompaniment.wav"), acc_cpu, sr)

    print(f"Saved inference results to {out_dir}/")

if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    infer_one_example(device=dev, seed=1337)
