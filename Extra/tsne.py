import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Generate dummy data with random values
num_rows = 1000

# Create a DataFrame with random values and specific column names
dummy_data = pd.DataFrame({
    'Name': [np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Isabella', 'Jack', 'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter', 'Quinn', 'Rachel', 'Sam', 'Taylor']
) for _ in range(num_rows)],
    'Condition': np.random.choice(['Good', 'Bad'], size=num_rows),
    'Latency_Wifi': np.random.normal(loc=1, scale=0.2, size=num_rows),  # 'Good' condition has lower latency
    'Loss_Wifi': np.random.normal(loc=0.05, scale=0.02, size=num_rows),   # 'Good' condition has lower loss
    'Latency_Gaming': np.random.normal(loc=1, scale=0.2, size=num_rows),
    'Loss_Gaming': np.random.normal(loc=0.05, scale=0.02, size=num_rows),
    'Latency_Video': np.random.normal(loc=1, scale=0.2, size=num_rows),
    'Loss_Video': np.random.normal(loc=0.05, scale=0.02, size=num_rows),
    'Latency_WFH': np.random.normal(loc=1, scale=0.2, size=num_rows),
    'Loss_WFH': np.random.normal(loc=0.05, scale=0.02, size=num_rows),
})

features = dummy_data.drop(['Name',
 'Condition'], axis=1)
print("data: ",features)
print("shape of features", features.shape) # shape of features (1000, 8)
#[1000 rows x 8 columns]
# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)
print("data scaled: ",data_scaled)

tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)
data_scaled = scaler.fit_transform(data_tsne)

df = dummy_data

features = dummy_data.drop(['Name',
 'Condition'], axis=1)
columns_of_interest =  features.columns.to_list()

# Apply K-means on the t-SNE components
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data_tsne)

# Add t-SNE components and cluster labels to the original DataFrame
df['TSNE_Component_1'] = data_tsne[:, 0]
df['TSNE_Component_2'] = data_tsne[:, 1]
df['Cluster'] = labels

# Get the centroid coordinates
# centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columns_of_interest)
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['TSNE_Component_1', 'TSNE_Component_2'])

# Display the main features for each centroid
for cluster_num in range(n_clusters):
    centroid_features = centroids.iloc[cluster_num]
    main_features = centroid_features.abs().sort_values(ascending=False).head(3)  # Display top 3 features
    print(f"Cluster {cluster_num + 1}: Main Features - {main_features.index.tolist()}")

# Count the number of users in each cluster
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Number_of_Users']

# Select the top 10 clusters based on the highest number of users
top_clusters = cluster_counts.nlargest(10, 'Number_of_Users')['Cluster'].tolist()

# Filter the DataFrame for the top clusters
df_top_clusters = df[df['Cluster'].isin(top_clusters)]



# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(df['TSNE_Component_1'], df['TSNE_Component_2'],
#                       c=df['Cluster'], cmap='tab10', s=30, alpha=0.7)

# plt.title('t-SNE with KMeans Clusters')
# plt.xlabel('TSNE Component 1')
# plt.ylabel('TSNE Component 2')
# plt.colorbar(scatter, label='Cluster')
# plt.grid(True)
# plt.show()
