import numpy as np
from sklearn.cluster import KMeans

def calculate_activation(x, receptor, spread):
    # Calculate the squared Euclidean distance
    distance_squared = np.sum((x - receptor)**2)

    # Calculate the RBF activation
    activation = np.exp(-distance_squared / (2 * spread**2))

    return activation


def rbf_input_middle_layers(X, num_clusters, num_neighbors):
    # Step 1: Use k-means clustering to find the receptors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    receptors = kmeans.cluster_centers_
    # print(receptors)
    # Step 2: Determine the spread of each receptor based on the nearest p neighbors
    nearest_neighbors = []
    for receptor in receptors:
        distances = np.linalg.norm(X - receptor, axis=1)
        # print('distances')
        # print(distances)
        nearest = np.argsort(distances)[:num_neighbors]
        # print('nearest')
        # print(nearest)
        # print(distances[nearest[0]])
        # print(distances[nearest[1]])
        nearest_neighbors.append(nearest)



    # Calculate the spread as the average distance to the nearest p neighbors
    # print(nearest_neighbors)
    spreads = []
    nearest_neighbors = np.array(nearest_neighbors,dtype=int)
    for i in range(num_clusters):
        # print('i and nearest_neighbors[i]')
        # print(i)
        # print(nearest_neighbors[i])
        spread = np.mean(np.linalg.norm(X[nearest_neighbors[i,:]] - receptors[i], axis=1))
        spreads.append(spread)

    return receptors, spreads
