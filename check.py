
checkpointChannels = set(['FP1', 'FPZ', 'FP2',
                          'AF3', 'AF4',
                          'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                          'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                          'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                          'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                          'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                          'PO7', 'PO3', 'POZ',  'PO4', 'PO8',
                          'O1', 'OZ', 'O2', ])
acceptedChannels = set([
    "PO12", "CCP2H", "FFC5H", "OI1", "PO7", "CPPZ",
    "TP7", "PO2", "FC3", "FTT7H", "PPO8", "CCP4H",
    "P11", "FCC5H", "FFC4H", "FP1", "CPP2H", "FFT7H",
    "P1", "I2", "AFF6H", "FZ", "PO4", "FCC2H",
    "F8", "FT9", "CP2", "AF3", "FCZ", "POO11H",
    "FPZ", "F3", "P8", "FC2", "F1", "CCP3H",
    "CP6", "PO1", "C1", "AFZ", "C3", "CB1",
    "FTT8H", "POO12H", "TP9", "I1", "FP2", "POO10H",
    "CPP1H", "CPP4H", "TTP8H", "AFF5H", "PO10", "POO9H",
    "POO3", "CP5", "PO3", "FC6", "FTT9H", "PPOZ",
    "TPP5H", "POO4", "CB2", "FT7", "CPZ", "CP1",
    "PPO1", "CP3", "CCP5H", "O2", "FCC1H", "CP4",
    "FT8", "T9", "PO5", "P2", "P5", "POZ",
    "FC1", "CPP3H", "C5", "P9", "P10", "PO6",
    "FFT8H", "CCP1H", "C2", "POOZ", "T7", "POO7",
    "FFC3H", "F6", "FCCZ", "TPP8H", "F7", "P4",
    "P3", "AF8", "PPO2", "AF4", "FFC2H", "FFC1H",
    "P6", "F2", "C6", "P12", "TP10", "CZ",
    "IZ", "CCP6H", "TP8", "PO11", "OI2", "FC5",
    "TTP7H", "CPP5H", "F5", "POO8", "CPP6H", "OZ",
    "PO9", "AF7", "PZ", "O1", "FC4", "PO8",
    "F4", "FCC3H", "T10", "P7", "FT10", "FCC4H",
    "FCC6H", "T8", "PPO7", "C4", "FFC6H", "FTT10H"
])

chOrder_standard = set(['P1', 'PO1', 'F8', 'C2', 'CZ', 'PO2', 'FPZ', 'F3', 'CP4', 'CP3', 'PO3', 'C5', 'FC6', 'PO10', 'FP2', 'FC4', 'FT7', 'PO8', 'CP5', 'F2', 'P4', 'AFZ', 'P6', 'O2', 'P2', 'FC5', 'FC1', 'TP9', 'T7', 'C4',
                        'P8', 'T8', 'OZ', 'AF4', 'CP1', 'FCZ', 'TP7', 'PO4', 'AF3', 'C3', 'O1', 'P7', 'F4', 'F1', 'FT8', 'CP2', 'CP6', 'PO7', 'P9', 'P5', 'P3', 'C6', 'PZ', 'FC2', 'PO9', 'POZ', 'C1', 'TP8', 'FZ', 'F7', 'P10', 'TP10', 'FC3'])

ch_names = set(['P1', 'PO1', 'F8', 'C2', 'CZ', 'PO2', 'FPZ', 'F3', 'CP4', 'CP3', 'PO3', 'C5', 'FC6', 'PO10', 'FP2', 'FC4', 'FT7', 'PO8', 'CP5', 'F2', 'P4', 'AFZ', 'P6', 'O2', 'P2', 'FC5', 'FC1', 'TP9', 'T7', 'C4',
                'P8', 'T8', 'OZ', 'AF4', 'CP1', 'FCZ', 'TP7', 'PO4', 'AF3', 'C3', 'O1', 'P7', 'F4', 'F1', 'FT8', 'CP2', 'CP6', 'PO7', 'P9', 'P5', 'P3', 'C6', 'PZ', 'FC2', 'PO9', 'POZ', 'C1', 'TP8', 'FZ', 'F7', 'P10', 'TP10', 'FC3'])


missing_channels = ch_names - acceptedChannels
extra_channels = acceptedChannels - ch_names

if not missing_channels:
    print("✅ All data channels are in the allowed list.")
else:
    print(
        f"❌ These data channels are NOT in the allowed list:\n{sorted(missing_channels)}")

if extra_channels:
    print(
        f"❌ Tdddhese data channels are NOT in the allowed list:\n{sorted(extra_channels)}")
