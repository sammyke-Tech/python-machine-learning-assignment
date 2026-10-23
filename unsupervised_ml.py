# Customer Segmentation Using K-Means
# Unsupervised Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Step 1: Prepare customer dataset
# Each customer has [monthly spending, monthly visits]
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
# Step 2: Scale the customer data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customers)
# Step 3: Create the K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
# Step 4: Train the model
clusters = model.fit_predict(scaled_data)
# Step 5: Display cluster assignments
print("CUSTOMER CLUSTER ASSIGNMENTS")
print("=" * 25)

for i, cluster in enumerate(clusters):
    print(f"Customer {i + 1} belongs to Cluster {cluster}")
    # Step 6: Display total customers in each cluster
print("\nTOTAL CUSTOMERS IN EACH CLUSTER")
print("=" * 25)

for cluster in range(3):
    count = sum(clusters == cluster)
    print(f"Cluster {cluster}: {count} customers")
# Step 7: Visualize customer clusters
for cluster in range(3):
    points = [
        customer
        for customer, label in zip(customers, clusters)
        if label == cluster
    ]

    plt.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        label=f"Cluster {cluster}"
    )

plt.xlabel("Monthly Spending (KES)")
plt.ylabel("Monthly Visits")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.grid(True)
plt.show()

