import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import argparse

# ——— import your stuff ———
from src.data.mix_dataset import MUSDBDataset
from src.models.encoder import Encoder
from src.models.separator import Separator
from src.models.decoder import Decoder
from src.utils.constants import N, C

# ---------- SI-SNR + PIT ----------
def si_snr(est, ref, eps=1e-8):
    # est, ref: (B, T)
    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    ref_energy = (ref**2).sum(dim=1, keepdim=True)
    proj = ((est * ref).sum(dim=1, keepdim=True) / (ref_energy + eps)) * ref
    noise = est - proj
    ratio = (proj**2).sum(dim=1) / ((noise**2).sum(dim=1) + eps)
    return 10 * torch.log10(ratio + eps)

def pit_l1_loss(est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    # est, tgt: (B, C, T), C=2
    l01 = (est[:,0] - tgt[:,0]).abs().mean() + (est[:,1] - tgt[:,1]).abs().mean()
    l10 = (est[:,0] - tgt[:,1]).abs().mean() + (est[:,1] - tgt[:,0]).abs().mean()
    return torch.min(l01, l10)

def pit_neg_si_snr_loss(est_sources, tgt_sources):
    # Permutation invariant loss for training
    # est_sources, tgt_sources: (B, C, T) with C=2 → only 2 perms
    # Perm 0: [0,1], Perm 1: [1,0]
    loss01 = -si_snr(est_sources[:,0], tgt_sources[:,0]).mean() \
             -si_snr(est_sources[:,1], tgt_sources[:,1]).mean()
    loss10 = -si_snr(est_sources[:,0], tgt_sources[:,1]).mean() \
             -si_snr(est_sources[:,1], tgt_sources[:,0]).mean()
    return torch.minimum(loss01, loss10)

@torch.no_grad()
def evaluate(dl, enc, sep, dec, device="cpu"):
    enc.eval(); sep.eval(); dec.eval()
    scores = []
    for mix, sources in dl:
        mix = mix.to(device).float()
        sources = sources.to(device).float()
        z = enc(mix)
        masks = sep(z)
        masked = masks * z.unsqueeze(1)
        B, C_, N_, Tp = masked.shape
        yhat = dec(masked.view(B*C_, N_, Tp))
        if yhat.ndim == 3 and yhat.size(1) == 1:
            yhat = yhat.squeeze(1)
        yhat = yhat.view(B, C_, -1)

        # align
        T = min(yhat.size(-1), sources.size(-1))
        yhat = yhat[..., :T]
        tgt = sources[..., :T]

        # SI-SDR score
        score01 = si_snr(yhat[:,0], tgt[:,0]).mean() + si_snr(yhat[:,1], tgt[:,1]).mean()
        score10 = si_snr(yhat[:,0], tgt[:,1]).mean() + si_snr(yhat[:,1], tgt[:,0]).mean()
        best_score = torch.max(score01, score10)  # permutation invariant
        scores.append(best_score.item())
    enc.train(); sep.train(); dec.train()
    return sum(scores) / max(1, len(scores))


def save_checkpoint(path, enc, sep, dec, optim, epoch, train_cfg, val_sisdr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "enc": enc.state_dict(),
        "sep": sep.state_dict(),
        "dec": dec.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "val_sisdr_db": val_sisdr,
        "train_cfg": train_cfg,
    }, path)

# ---------- One-batch demo ----------
def demo_realdata_one_step(
    root="/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data/musdb18",
    sr=16000, segment_seconds=4, batch_size=2, lr=1e-4, max_norm=5.0, device="cpu"
):
    # 1) Data
    ds = MUSDBDataset(root=root, split="train", sr=sr, segment_seconds=segment_seconds)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mix, sources = next(iter(dl))                  # mix: (B,1,T), sources: (B,2,T)
    mix = mix.to(device).float()
    sources = sources.to(device).float()

    # 2) Model
    enc, sep, dec = Encoder().to(device), Separator().to(device), Decoder().to(device)
    optim = torch.optim.Adam(list(enc.parameters())+list(sep.parameters())+list(dec.parameters()), lr=lr)

    enc.train(); sep.train(); dec.train()

    # 3) Forward (non-streaming)
    z = enc(mix)                                   # (B, N, T′)
    masks = sep(z)                                 # (B, C, N, T′)
    masked = masks * z.unsqueeze(1)                # (B, C, N, T′)
    B, C_, N_, Tp = masked.shape
    masked_bc = masked.view(B*C_, N_, Tp)          # (B*C, N, T′)

    yhat = dec(masked_bc)                          # (B*C, 1, T) or (B*C, T)
    if yhat.ndim == 3 and yhat.size(1) == 1:
        yhat = yhat.squeeze(1)                     # (B*C, T)
    yhat = yhat.view(B, C_, -1)                    # (B, C, T)

    # 4) (Optional) align T just in case decoder rounding differs
    T = min(yhat.size(-1), sources.size(-1))
    yhat = yhat[..., :T]
    tgt = sources[..., :T]

    # 5) Loss + backward
    loss = pit_l1_loss(yhat, tgt)
    optim.zero_grad()
    loss.backward()
    clip_grad_norm_(list(enc.parameters())+list(sep.parameters())+list(dec.parameters()), max_norm=max_norm)
    optim.step()

    print("batch mix:", mix.shape, "encoder out:", z.shape, "masks:", masks.shape, "yhat:", yhat.shape)
    print(f"PIT -SI-SNR loss: {loss.item():.3f}")

def train_epochs(
    root="/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data/musdb18",
    sr=16000,
    segment_seconds=4,
    batch_size=2,
    lr=1e-4,
    max_norm=5.0,
    device="cpu",
    epochs=1,
    steps_per_epoch=100,
    val_steps=None,
    ckpt_dir="checkpoints",
    early_stop_patience=5,
):
    # Dataset & Loader
    ds = MUSDBDataset(root=root, split="train", sr=sr, segment_seconds=segment_seconds)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    ds_val = MUSDBDataset(root=root, split="test", sr=sr, segment_seconds=segment_seconds)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    it = iter(dl)

    # Models
    enc, sep, dec = Encoder().to(device), Separator().to(device), Decoder().to(device)
    optim = torch.optim.Adam(list(enc.parameters()) + list(sep.parameters()) + list(dec.parameters()), lr=lr)

    enc.train(); sep.train(); dec.train()

    best_sisdr = float("-inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        ma_loss = 0.0
        for step in range(steps_per_epoch):
            try:
                mix, sources = next(it)
            except StopIteration:
                it = iter(dl)
                mix, sources = next(it)

            mix = mix.to(device).float()         # (B,1,T)
            sources = sources.to(device).float() # (B,2,T)

            # Forward
            z = enc(mix)                                   # (B,N,T')
            masks = sep(z)                                 # (B,C,N,T')
            masked = masks * z.unsqueeze(1)                # (B,C,N,T')
            Bsz, C_, N_, Tp = masked.shape
            yhat = dec(masked.view(Bsz * C_, N_, Tp))      # (B*C,1,T) or (B*C,T)
            if yhat.ndim == 3 and yhat.size(1) == 1:
                yhat = yhat.squeeze(1)                     # (B*C,T)
            yhat = yhat.view(Bsz, C_, -1)                  # (B,C,T)

            # Align lengths just in case
            T = min(yhat.size(-1), sources.size(-1))
            yhat = yhat[..., :T]
            tgt = sources[..., :T]

            # Loss & step
            loss = pit_l1_loss(yhat, tgt)
            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(list(enc.parameters()) + list(sep.parameters()) + list(dec.parameters()), max_norm=max_norm)
            optim.step()

            # moving avg for display
            ma_loss = 0.98 * ma_loss + 0.02 * loss.item() if step > 0 else loss.item()
            if (step + 1) % 10 == 0:
                print(f"epoch {epoch+1} step {step+1:04d}/{steps_per_epoch}  loss {loss.item():.3f}  ma {ma_loss:.3f}")

        # ---- end of epoch: validate & checkpoint ----
        if len(dl_val) > 0:
            val_sisdr = evaluate(dl_val, enc, sep, dec, device=device)  # higher is better (dB)
        else:
            val_sisdr = float("nan")
        print(f"epoch {epoch+1} complete  train_ma {ma_loss:.3f}  val_sisdr_db {val_sisdr:.3f}")

        # save last checkpoint
        train_cfg = {
            "sr": sr,
            "segment_seconds": segment_seconds,
            "batch_size": batch_size,
            "lr": lr,
            "max_norm": max_norm,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
        }
        save_checkpoint(os.path.join(ckpt_dir, "checkpoint_last.pt"), enc, sep, dec, optim, epoch+1, train_cfg, val_sisdr)

        # save best (maximize SI-SDR)
        if val_sisdr > best_sisdr:
            best_sisdr = val_sisdr
            save_checkpoint(os.path.join(ckpt_dir, "checkpoint_best.pt"), enc, sep, dec, optim, epoch+1, train_cfg, val_sisdr)
            print(f"  ✓ new best checkpoint saved (val_sisdr_db {val_sisdr:.3f})")

        # ---- early stopping on no SI-SDR improvement ----
        # Only count if val_sisdr is finite (skip NaN cases)
        if torch.isfinite(torch.tensor(val_sisdr)):
            if val_sisdr > best_sisdr:
                # already handled above; reset counter
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  (no improvement) patience {epochs_no_improve}/{early_stop_patience}")
                if epochs_no_improve >= early_stop_patience:
                    print("Early stopping: no SI-SDR improvement.")
                    break

    print("Training loop finished.")
    return enc, sep, dec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data/musdb18")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_seconds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_norm", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--no_demo", action="store_true")
    args = parser.parse_args()

    dev = (
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    print("Using device:", dev)

    if not args.no_demo:
        demo_realdata_one_step(device=dev, root=args.root, sr=args.sr, segment_seconds=args.segment_seconds, batch_size=args.batch_size, lr=args.lr, max_norm=args.max_norm)

    train_epochs(
        root=args.root,
        sr=args.sr,
        segment_seconds=args.segment_seconds,
        batch_size=args.batch_size,
        lr=args.lr,
        max_norm=args.max_norm,
        device=dev,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        ckpt_dir=args.ckpt_dir,
        early_stop_patience=args.early_stop_patience,
    )
