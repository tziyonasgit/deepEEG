import pandas as pd
from collections import Counter

# Load your Excel or CSV file
# df = pd.read_excel('your_file.xlsx')  # for Excel
df = pd.read_csv('khulasubs.csv')       # for CSV

# Extract the session columns
session_cols = ['3M', '6M', '12M', '24M']
sessions = df[session_cols]

# Stack all IDs from all sessions into one column with corresponding session
subject_sessions = []

for col in session_cols:
    subject_sessions.extend([(sub_id, col)
                            for sub_id in sessions[col].dropna().unique()])

# Create a mapping: subject_id -> set of sessions
subject_to_sessions = {}
for sub_id, sess in subject_sessions:
    if sub_id not in subject_to_sessions:
        subject_to_sessions[sub_id] = set()
    subject_to_sessions[sub_id].add(sess)

# Count how many sessions each subject appeared in
session_counts = [len(sessions) for sessions in subject_to_sessions.values()]
count_distribution = Counter(session_counts)

# Print result
print("Subject session participation distribution:")
for i in range(1, 5):
    print(f"{i} session(s): {count_distribution.get(i, 0)} subjects")
