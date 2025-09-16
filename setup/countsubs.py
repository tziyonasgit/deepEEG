import pandas as pd
from collections import Counter


df = pd.read_csv('khulasubs.csv')

session_cols = ['3M', '6M', '12M', '24M']
sessions = df[session_cols]
subject_sessions = []

for col in session_cols:
    subject_sessions.extend([(sub_id, col)
                            for sub_id in sessions[col].dropna().unique()])

# create a mapping of subject_id to the set of sessions they are in
subject_to_sessions = {}
for sub_id, sess in subject_sessions:
    if sub_id not in subject_to_sessions:
        subject_to_sessions[sub_id] = set()
    subject_to_sessions[sub_id].add(sess)

# count how many sessions each subject appeared in
session_counts = [len(sessions) for sessions in subject_to_sessions.values()]
count_distribution = Counter(session_counts)

print("Subject session participation distribution:")
for i in range(1, 5):
    print(f"{i} session(s): {count_distribution.get(i, 0)} subjects")
