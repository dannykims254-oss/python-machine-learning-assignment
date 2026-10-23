# Customer Segmentation Using K-Means

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Prepare customer data
# Each row contains [monthly spending, monthly visits]
customers = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]
]

# 2. Scale the features
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)

# 3. Create the K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# 4. Train the model and assign clusters
clusters = model.fit_predict(scaled_customers)

# 5. Display each customer's cluster
for i, cluster in enumerate(clusters):
    print(f"Customer {i + 1} belongs to Cluster {cluster}")

# 6. Count customers in each cluster
print("\nTotal customers in each cluster:")

for cluster in range(3):
    count = sum(clusters == cluster)
    print(f"Cluster {cluster}: {count} customers")
