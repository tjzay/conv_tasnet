import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MUSDBDataset(Dataset):
    def __init__(self, root, split="train", sr=16000, segment_seconds=10):
        """
        Args:
            root: base data folder (e.g. ".../conv_tasnet/src/data")
            split: "train" or "test"
            sr: target sample rate
            segment_seconds: fixed segment length in seconds
        """
        self.root = root
        self.split = split
        self.sr = sr
        self.segment_len = sr * segment_seconds

        # 1. collect file list from mix folder
        mix_dir = os.path.join(root, split, "mix")
        mix_files = sorted(glob.glob(os.path.join(mix_dir, "*.wav")))

        self.items = []
        for mix_path in mix_files:
            base = os.path.basename(mix_path).replace("_mix.wav", "")
            voice_path = os.path.join(root, split, "voice", base + "_voice.wav")
            acc_path   = os.path.join(root, split, "accompaniment", base + "_acc.wav")

            # 2. check files exist
            assert os.path.exists(voice_path), f"Missing {voice_path}"
            assert os.path.exists(acc_path), f"Missing {acc_path}"

            self.items.append({
                "mix": mix_path,
                "voice": voice_path,
                "acc": acc_path
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # 3. load audio with torchaudio
        mix, sr_mix = torchaudio.load(item["mix"])       # (channels, T)
        voice, sr_voice = torchaudio.load(item["voice"])  # (channels, T)
        acc, sr_acc   = torchaudio.load(item["acc"])      # (channels, T)

        # 4. resample all to target sr (16 kHz by default)
        if sr_mix != self.sr:
            resample_mix = T.Resample(orig_freq=sr_mix, new_freq=self.sr)
            mix = resample_mix(mix)
        if sr_voice != self.sr:
            resample_voice = T.Resample(orig_freq=sr_voice, new_freq=self.sr)
            voice = resample_voice(voice)
        if sr_acc != self.sr:
            resample_acc = T.Resample(orig_freq=sr_acc, new_freq=self.sr)
            acc = resample_acc(acc)

        # Ensure identical length after resampling (trim to shortest)
        T_min = min(mix.size(-1), voice.size(-1), acc.size(-1))
        mix = mix[..., :T_min]
        voice = voice[..., :T_min]
        acc = acc[..., :T_min]

        # 4b. convert stereo to mono if needed
        if mix.size(0) == 2:
            mix = mix.mean(dim=0, keepdim=True)
        if voice.size(0) == 2:
            voice = voice.mean(dim=0, keepdim=True)
        if acc.size(0) == 2:
            acc = acc.mean(dim=0, keepdim=True)

        # 5. crop/pad to segment_len
        t_len = mix.size(-1)
        if t_len >= self.segment_len:
            start = torch.randint(0, t_len - self.segment_len + 1, (1,)).item()
            mix = mix[:, start:start+self.segment_len]
            voice = voice[:, start:start+self.segment_len]
            acc = acc[:, start:start+self.segment_len]
        else:
            pad = self.segment_len - t_len
            mix = F.pad(mix, (0, pad))
            voice = F.pad(voice, (0, pad))
            acc = F.pad(acc, (0, pad))

        # 6. stack sources -> (C=2, T)
        sources = torch.cat([voice, acc], dim=0)

        return mix, sources

def main():  
    # To test the dataloader class
    root = "/Users/tobyzayontz/Documents/Uni/conv_tasnet/src/data/musdb18"

    ds = MUSDBDataset(root, split="train", sr=16000, segment_seconds=4)
    print("Number of items:", len(ds))
    mix, sources = ds[0]
    print("mix:", mix.shape)        # expect (1, T)
    print("sources:", sources.shape) # expect (2, T)

    dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)

    for mix_b, sources_b in dl:
        print("batch mix:", mix_b.shape)        # expect (B, 1, T)
        print("batch sources:", sources_b.shape) # expect (B, 2, T)
        break

    err = (mix - sources.sum(dim=0, keepdim=True)).abs().mean().item()
    print("mix ≈ sum(sources), mean abs error:", err)

    mix, sources = ds[0]
    plt.plot(mix[0].numpy(), label="mix")
    plt.plot(sources[0].numpy(), label="voice")
    plt.plot(sources[1].numpy(), label="accompaniment")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
