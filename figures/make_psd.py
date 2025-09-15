import mne
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
egi_to_1020 = {
    'E3': 'AF4',
    'E4': 'F2',
    'E6': 'FCZ',
    'E9': 'FP2',
    'E11': 'FZ',
    'E13': 'FC1',
    'E15': 'FPZ',
    'E19': 'F1',
    'E23': 'AF3',
    'E24': 'F3',
    'E28': 'FC5',
    'E29': 'FC3',
    'E30': 'C1',
    'E33': 'F7',
    'E34': 'FT7',
    'E36': 'C3',
    'E37': 'CP1',
    'E41': 'C5',
    'E42': 'CP3',
    'E45': 'T7',
    'E46': 'TP7',
    'E47': 'CP5',
    'E51': 'P5',
    'E52': 'P3',
    'E55': 'CPZ',
    'E58': 'P7',
    'E60': 'P1',
    'E62': 'PZ',
    'E65': 'PO7',
    'E67': 'PO3',
    'E70': 'O1',
    'E72': 'POZ',
    'E75': 'OZ',
    'E77': 'PO4',
    'E83': 'O2',
    'E85': 'P2',
    'E87': 'CP2',
    'E90': 'PO8',
    'E92': 'P4',
    'E93': 'CP4',
    'E97': 'P6',
    'E96': 'P8',
    'E98': 'CP6',
    'E102': 'TP8',
    'E103': 'C6',
    'E104': 'C4',
    'E105': 'C2',
    'E108': 'T8',
    'E111': 'FC4',
    'E112': 'FC2',
    'E116': 'FT8',
    'E117': 'FC6',
    'E122': 'F8',
    'E124': 'F4'
}
removeChannels = ['E2', 'E5', 'E7', 'E10', 'E12', 'E16', 'E18', 'E20', 'E22', 'E26', 'E27', 'E31', 'E35', 'E39', 'E40', 'E50', 'E53', 'E54', 'E57', 'E59', 'E61', 'E63', 'E64',
                  'E66', 'E69', 'E71', 'E74', 'E76', 'E78', 'E79', 'E80', 'E82', 'E84', 'E86', 'E89', 'E91', 'E95', 'E100', 'E101', 'E106', 'E109', 'E110', 'E115', 'E118', 'E123']


# sample24 = "/Users/cccohen/1_191_51545547_24_20240228_022942002_processed.set"
# sample3 = "/Users/cccohen/1_191_55191581_3_20220516_125802002_processed.set"
# sample24 = "/Users/cccohen/1_191_91801361_24_20240229_095806002_processed.set"
# sample3 = "/Users/cccohen/1_191_54683158_3_20221027_020338002_processed.set"
# epochs3 = mne.io.read_epochs_eeglab(sample3)
# epochs24 = mne.io.read_epochs_eeglab(sample24)
# epochs3.resample(256)
# epochs24.resample(256)
# epochs3.drop_channels(removeChannels)
# epochs3.rename_channels(egi_to_1020)
# epochs24.drop_channels(removeChannels)
# epochs24.rename_channels(egi_to_1020)
# n_times_3 = len(epochs3.times)
# n_times_24 = len(epochs24.times)
# print("epochs3:", n_times_3)
# print("epochs24:", n_times_24)
# psd3 = epochs3.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
#                            n_per_seg=N_PER_SEG, n_overlap=N_OVERLAP, n_fft=1024)
# psd24 = epochs24.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
#                              n_per_seg=N_PER_SEG, n_overlap=N_OVERLAP, n_fft=1024)
# freqs = psd3.freqs
# P3 = psd3.get_data()
# P24 = psd24.get_data()
# P3_mean = P3.mean(axis=(0, 1))
# P24_mean = P24.mean(axis=(0, 1))

# ===== PSD params =====
SFREQ = 256
SEG_SEC = 2
N_FFT = 512
N_PER_SEG = int(SEG_SEC * SFREQ)
N_OVERLAP = N_PER_SEG // 2
FMIN, FMAX = 0.5, 60.0


def getAdult():
    sampleAdult = "/home/chntzi001/deepEEG/psd/sub-AB10_task-gonogo_run-1_eeg.set"
    rawAdult = mne.io.read_raw_eeglab(sampleAdult)
    removeAdult = drop = ['VEO', 'CPz', 'R-Dia-Y-(mm)', 'PO5', 'F6', 'Pz', 'CB2', 'POz', 'PO6',
                          'CB1', 'FCz', 'Oz', 'M2', 'R-Dia-X-(mm)', 'Cz', 'M1', 'EKG', 'HEO', 'F5']
    rawAdult.resample(256)
    rawAdult.drop_channels(removeAdult)
    rawAdult.rename_channels({ch: ch.upper() for ch in rawAdult.ch_names})
    print("adult:", rawAdult.n_times)
    psdAdult = rawAdult.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                                    n_per_seg=N_PER_SEG, n_overlap=N_OVERLAP, n_fft=1024)
    PAdult = psdAdult.get_data()

    PAdult_mean = PAdult.mean(axis=(0))

    return psdAdult.freqs, PAdult_mean


def read_epochs_eeglab_safe(path):
    try:
        return mne.io.read_epochs_eeglab(path, verbose="ERROR")
    except Exception as e1:
        print(
            f"[EEGLAB EPOCHS FAILED] {path}\n{type(e1).__name__}: {e1}", flush=True)
        try:
            return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        except Exception as e2:
            print(
                f"[EEGLAB RAW FAILED] {path}\n{type(e2).__name__}: {e2}", flush=True)
            return None


def mean_psd_in_folder(folder):
    files = glob.glob(os.path.join(folder, "*.set"), recursive=True)
    per_file_means = []
    skipped = 0
    freqs_ref = None
    for f in files:
        try:
            ep = mne.io.read_epochs_eeglab(f)
        except Exception as e:
            print("ep is None !!!!!!!!!!!!!!!!!", flush=True)
            ep = None
        if ep is None:
            skipped += 1
            continue
        ep.resample(256)
        ep.drop_channels(removeChannels)
        ep.rename_channels(egi_to_1020)

        if len(ep.times) < N_PER_SEG:
            print(len(ep.times), flush=True)
            print("len(ep.times) too small !!!!!!!!!!!!", flush=True)
            skipped += 1
            continue

        psd = ep.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                             n_per_seg=N_PER_SEG, n_overlap=N_OVERLAP, n_fft=1024, picks="data", verbose="ERROR"
                             )
        freqs = psd.freqs
        P = psd.get_data()
        print("00000000000000000000000000000000000")
        print("P is:", P, flush=True)
        if P.ndim == 3:          # (n_epochs, n_channels, n_freqs)
            P_mean_file = P.mean(axis=(0, 1))
        elif P.ndim == 2:        # (n_channels, n_freqs)
            P_mean_file = P.mean(axis=0)
        else:
            skipped += 1
            continue
        if freqs_ref is None:
            freqs_ref = freqs
        elif not np.array_equal(freqs_ref, freqs):
            skipped += 1
            continue
        per_file_means.append(P_mean_file)

    if len(per_file_means) == 0:
        raise RuntimeError(
            f"No usable .set files in {folder}. Skipped={skipped}")

    A = np.vstack(per_file_means)      # (n_files, n_freqs)
    mean = A.mean(axis=0)
    return freqs_ref, mean, len(per_file_means), skipped


FOLDER_3M = "/scratch/chntzi001/khula/3M"
FOLDER_24M = "/scratch/chntzi001/khula/24M"
freqs, P3_mean, n_times_3, sk3 = mean_psd_in_folder(FOLDER_3M)
freqs, P24_mean, n_times_24, sk24 = mean_psd_in_folder(FOLDER_24M)
print(
    f"3M: used {n_times_3}, skipped {sk3} | 24M: used {n_times_24}, skipped {sk24}")


sampleAdult = "/home/chntzi001/deepEEG/psd/sub-AB10_task-gonogo_run-1_eeg.set"
rawAdult = mne.io.read_raw_eeglab(sampleAdult)
removeAdult = drop = ['VEO', 'CPz', 'R-Dia-Y-(mm)', 'PO5', 'F6', 'Pz', 'CB2', 'POz', 'PO6',
                      'CB1', 'FCz', 'Oz', 'M2', 'R-Dia-X-(mm)', 'Cz', 'M1', 'EKG', 'HEO', 'F5']
rawAdult.resample(256)
rawAdult.drop_channels(removeAdult)
rawAdult.rename_channels({ch: ch.upper() for ch in rawAdult.ch_names})

freqsAdult, PAdult_mean = getAdult()

assert np.array_equal(freqs, freqsAdult), "Frequency grids differ."

# ======= Plot PSDs =======
plt.figure(figsize=(8, 5))
plt.semilogy(freqs, P3_mean, label="3M", color="blue")
plt.semilogy(freqs, P24_mean, label="24M", color="red")
plt.semilogy(freqs, PAdult_mean, label="Adult", color="orange")

plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.title("PSD Overlay: 3M vs 24M")
plt.grid(True, which="both", ls=":")
delta_band = plt.axvspan(0.5, 4, color="lightblue",
                         alpha=0.3, label="Delta (0.5–4 Hz)")
theta_band = plt.axvspan(4, 8,  color="lightgreen",
                         alpha=0.3, label="Theta (4–8 Hz)")
alpha_band = plt.axvspan(8, 13, color="violet",
                         alpha=0.3, label="Alpha (8–12 Hz)")
beta_band = plt.axvspan(13, 30, color="gold", alpha=0.2,
                        label="Beta (12–30 Hz)")
plt.legend(framealpha=1, facecolor="white")
plt.tight_layout()
plt.savefig("v2psd_overlay.png", dpi=300)
plt.show()
