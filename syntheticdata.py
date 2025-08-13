import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 200

# True distributions
process_A = np.random.normal(loc=0, scale=1, size=n)
process_B = np.random.normal(loc=3, scale=1, size=n)

# Force sample means to match (e.g., both to 1.5)
target_mean = 1.5
process_A_adj = process_A - np.mean(process_A) + target_mean
process_B_adj = process_B - np.mean(process_B) + target_mean

print("True means:", 0, 3)
print("Sample means:", np.mean(process_A_adj), np.mean(process_B_adj))

plt.plot(process_A_adj, label="Process A (adj)")
plt.plot(process_B_adj, label="Process B (adj)")
plt.legend()
plt.show()
