import glob
import csv
import os

# Define where your Khula .set files are stored
khula_root = "/scratch/chntzi001/khula"

# Recursively get all .set files
set_files = glob.glob(os.path.join(khula_root, "**/*.set"), recursive=True)

# Output CSV path
output_csv = "khula_file_list.csv"

# Write CSV with headers: file_path,label
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'label'])  # ✅ include both headers

    for path in sorted(set_files):
        filename = os.path.basename(path)
        try:
            age = int(filename.split('_')[3])  # e.g., 3, 6, 12, 24
        except (IndexError, ValueError):
            print(f"⚠️ Skipping malformed filename: {filename}")
            continue

        writer.writerow([path, age])

print(f"✅ {len(set_files)} .set files written to {output_csv}")
