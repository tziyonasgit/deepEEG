import csv
import mne

rows = []
with open('khula_file_list.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader, None)  # skips header
    for row in csv_reader:
        data_file = row[0]
        try:
            raw = mne.io.read_epochs_eeglab(data_file, verbose=False)
        except ValueError as e:
            if "trials less than 2" in str(e):
                print(f"⚠️ Skipping non-epoched file: {data_file}")

        except OSError as e:
            print(f"❌ Cannot read file (corrupt?): {data_file}")
            print(f"  → Error: {e}")
