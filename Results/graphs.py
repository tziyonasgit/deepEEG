import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your log file
log_path = "log.txt"

# Store results
records = []

with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        records.append({
            'Epoch': data['epoch'],
            'ValAccuracy': data['val_accuracy'],
            'ValLoss': data['val_loss']
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# Melt the dataframe to long format
data_melted = df.melt(id_vars='Epoch', value_vars=['ValAccuracy', 'ValLoss'],
                      var_name='Metric', value_name='Value')

# Create the lineplot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=data_melted, x='Epoch', y='Value', hue='Metric', marker='o')

# Add labels and title
plt.title("Val Metrics Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend(title='Metric')

lr = 5e-4
batch_size = 64
epochs = 10
model = "EEGPT-large"

# Add hyperparameter block
hyperparams = f"Model: {model}\nLR: {lr}\nBatch Size: {batch_size}\nEpochs: {epochs}"
plt.gcf().text(0.8, 0.4, hyperparams, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


plt.tight_layout()
plt.show()
