import os
import stempeg
import soundfile as sf
import numpy as np

# Paths
MUSDB_ROOT = "/Users/tobyzayontz/musdb18"
OUT_ROOT = "/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data"

splits = ["train", "test"]

def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    """Convert stereo [T,2] to mono [T]."""
    if x.ndim == 2 and x.shape[1] == 2:
        return x.mean(axis=1)
    return x

for split in splits:
    in_dir = os.path.join(MUSDB_ROOT, split)
    out_mix = os.path.join(OUT_ROOT, split, "mix")
    out_voice = os.path.join(OUT_ROOT, split, "voice")
    out_acc = os.path.join(OUT_ROOT, split, "accompaniment")

    os.makedirs(out_mix, exist_ok=True)
    os.makedirs(out_voice, exist_ok=True)
    os.makedirs(out_acc, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.endswith(".stem.mp4"):
            continue

        path = os.path.join(in_dir, fname)
        # load stems (S, T, 2) = (5, time, stereo)
        stems, rate = stempeg.read_stems(path)

        # Extract
        mix = stereo_to_mono(stems[0])
        drums = stereo_to_mono(stems[1])
        bass = stereo_to_mono(stems[2])
        other = stereo_to_mono(stems[3])
        voice = stereo_to_mono(stems[4])

        # Out filenames
        base = os.path.splitext(fname)[0]
        sf.write(os.path.join(out_mix, base + "_mix.wav"), voice+other, rate)
        sf.write(os.path.join(out_voice, base + "_voice.wav"), voice, rate)
        sf.write(os.path.join(out_acc, base + "_acc.wav"), other, rate)

        print(f"Processed {fname}")
