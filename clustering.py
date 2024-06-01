import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dname = 'df_fuzzy_names_ncvoter2017_100k_lsh200-50-100'
dataset2 = pd.read_pickle("dataset/"+ dname + ".pkl")

print(dataset2.head())

##Norm Data
sign_norms = dataset2["Fuzzy Signature_Norm-200"]
data_norms = np.array([x for x in sign_norms.to_numpy()])

scaler = StandardScaler()
data_norms_scaled = scaler.fit_transform(data_norms)

import numpy as np

def kmeans_dot_product(data, k, max_iterations=20, tol=1e-4):
    #centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    # Get cluster centroids (Slower but more precise)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    
    # Initialize centroids randomly (it runs faster, both results are little different.)
    #centroids = data[np.random.choice(len(data), k, replace=False), :]
    
    labels = np.zeros(len(data))

    for index in range(max_iterations):
        #print(index)
        # Assign each data point to the nearest centroid using dot product
        distances = np.dot(data, centroids.T)
        new_labels = np.argmax(distances, axis=1)
        #print(new_labels)

        # Check for convergence
        if np.all(new_labels == labels):
            break

        # Update centroids
        for i in range(k):
            if np.sum(new_labels == i) > 0:
                centroids[i, :] = np.mean(data[new_labels == i, :], axis=0)
                #print(i, "update")

        labels = new_labels

    return centroids, labels

# Example usage:
np.random.seed(42)

# Set the number of clusters
no_of_clusters = 50
name = dname + "_c" + str(no_of_clusters)
print(name)

# Run K-means clustering
cluster_centroids, cids = kmeans_dot_product(data_norms_scaled, no_of_clusters)
dataset2['Cluster_Id'] = cids

count = {i: 0 for i in range(no_of_clusters)}
for i in cids:
    count[i] += 1    
    
_min = count[min(count, key=count.get)]
_max = count[max(count, key=count.get)]
print(count, _min, _max)

completed_items = []
for i in range(_max):
    count = {k: v - 1 if v > 0 else 0 for k, v in count.items()}
    completed_items.append(100000 - sum(count.values()))

print(completed_items)

plt.hist(cids, bins=np.arange(cids.min(), cids.max(), 1) )
plt.savefig("out/" + name + '.eps', format='eps')

print("Clusteing Done")

# Padding
max_cluster_size = dataset2.groupby('Cluster_Id').size().max()
print("max_cluster_size", max_cluster_size)

# Initialize a DataFrame with the necessary columns
columns = ['Cluster_Id'] + [f'Item_{i}' for i in range(max_cluster_size)]
cluster_dataset2 = pd.DataFrame(columns=columns)
cluster_dataset2_IDs = pd.DataFrame(columns=columns)

dummy_element = np.ones(50)

for cluster_num in range(no_of_clusters):  # Cluster number
    cluster_items = dataset2[dataset2['Cluster_Id'] == cluster_num]['Fuzzy Signature_Norm-50'].tolist()
    cluster_items_IDs = dataset2[dataset2['Cluster_Id'] == cluster_num]['ID'].tolist()    
    #print(cluster_num, len(cluster_items))
    
    while len(cluster_items) < max_cluster_size:
    #_len = max_cluster_size - len(cluster_items)
        cluster_items.append(dummy_element)
        cluster_items_IDs.append("NULL")
    
    # Assign the cluster items to separate columns
    data = [cluster_num] + cluster_items
    cluster_dataset2.loc[cluster_num] = data
    
    IDs = [cluster_num] + cluster_items_IDs
    cluster_dataset2_IDs.loc[cluster_num] = IDs

cluster_dataset2.to_pickle("out/" + name + ".pkl")
cluster_dataset2_IDs.to_pickle("out/" + name + "_IDs.pkl")
np.save("out/" + name + "_centroids.npy", cluster_centroids)

print("Padding Done")