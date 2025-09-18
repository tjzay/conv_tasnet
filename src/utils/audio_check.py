""" This script generates 1 second of a 1 kHz sine wave 
sampled at 16 kHz. It saves it as a .mp4 file."""
import torch, math
import soundfile as sf
import matplotlib.pyplot as plt

SR = 16000 # sample rate
DURATION = 1.0 # seconds
FREQ = 440.0 # Hz

def generate_sine(freq=FREQ, dur = DURATION, sr = SR):
    n = torch.arange(dur*sr) # number of samples
    w = (freq/sr) * 2 * math.pi # radians per sample
    x = torch.sin(w * n) # output sine wave
    return x

def main():
    audio = generate_sine()
    sf.write("sine_test.wav", audio, sr)

    plt.plot(audio[:500])  # show first 500 samples (≈30 ms)
    plt.title("1 kHz Sine Wave")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    main()
