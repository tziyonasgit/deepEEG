import numpy as np
import mne
from mne.time_frequency import psd_array_welch
print(mne.__version__)

# extract bandpower frequencies i.e. calculate mean power across channels per frequency range
fmin, fmax = 1, 45
n_fft = 512
n_overlap = 256
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 45),
}

full_file_path = "/Users/cccohen/1_191_51545547_24_20240228_022942002_processed.set"
data = mne.io.read_epochs_eeglab(full_file_path)
data = data.copy().filter(1, 45)

spec = data.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                        n_fft=n_fft, n_overlap=n_overlap, picks="eeg")
psds, freqs = spec.get_data(return_freqs=True)

# if there are epochs, average across them
if psds.ndim == 3:
    psds = psds.mean(axis=0)

bandpowers = {}
for band, (lo, hi) in bands.items():
    idx = (freqs >= lo) & (freqs <= hi)
    bandpowers[band] = psds[:, idx].mean(axis=1)


X = np.vstack(list(bandpowers.values())).T

print(X.shape)  # Each channel has its associated bandpower (99, 5)
print(X)
