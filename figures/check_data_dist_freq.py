import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt


def get_peak_freq(x, sr=256):
    X = np.fft.rfft(x, axis=-1)
    psd = (np.abs(X)**2).mean(axis=0)
    freqs = np.fft.rfftfreq(x.shape[-1], 1/sr)
    return freqs[np.argmax(psd)]


# Load a few samples from train/test pickles
train_files = glob.glob(
    "/scratch/chntzi001/khula/processedSynthetic_0.0001/train/*.pkl")
test_files = glob.glob(
    "/scratch/chntzi001/khula/processedSynthetic_0.0001/test/*.pkl")

train_freqs, test_freqs = [], []

for f in train_files[:200]:
    d = pickle.load(open(f, "rb"))
    x = d["X"]  # [channels, timepoints]
    train_freqs.append(get_peak_freq(x))

for f in test_files[:200]:
    d = pickle.load(open(f, "rb"))
    x = d["X"]
    test_freqs.append(get_peak_freq(x))

# Plot histograms
plt.hist(train_freqs, bins=30, alpha=0.6, label="train")
plt.hist(test_freqs, bins=30, alpha=0.6, label="test")
plt.xlabel("Dominant frequency (Hz)")
plt.ylabel("Count")
plt.legend()
plt.title("Train vs Test dominant frequencies")
plt.savefig("train_vs_test_freqs.png", dpi=150)
print("Saved plot to train_vs_test_freqs.png")
