import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx as nx
from scipy.spatial import distance_matrix

url = "https://raw.githubusercontent.com/SakibHossain26/fortnitemap/main/chest_data.json"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    raw_coords = data['chests'][0]['coords']
    chest_matrix = np.array(raw_coords)
    
    # --- STEP 1: CLUSTERING ---
    dbscan = DBSCAN(eps=8.0, min_samples=7)
    clusters = dbscan.fit_predict(chest_matrix)
    
    unique_clusters, counts = np.unique(clusters[clusters >= 0], return_counts=True)
    best_cluster_id = unique_clusters[np.argmax(counts)]
    
    # Extract only the coordinates for the best cluster
    best_x = chest_matrix[clusters == best_cluster_id, 0]
    best_y = chest_matrix[clusters == best_cluster_id, 1]
    best_coords = np.column_stack((best_x, best_y))
    
    # --- STEP 2: ROUTING (GRAPH THEORY) ---
    print(f"Calculating optimal route for {len(best_coords)} chests...")
    
    # Build a mathematical distance matrix for just this POI
    dist_matrix = distance_matrix(best_coords, best_coords)
    
    # Turn the coordinates into a Graph (Nodes and Edges)
    G = nx.Graph()
    for i in range(len(best_coords)):
        for j in range(i + 1, len(best_coords)):
            G.add_edge(i, j, weight=dist_matrix[i, j])
            
    # Run the Traveling Salesperson Problem (TSP) solver
    # cycle=False means you don't have to walk back to the first chest you opened
    tsp_path = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=False)
    
    # Order the coordinates based on the calculated optimal path
    optimal_route = best_coords[tsp_path]
    route_x = optimal_route[:, 0]
    route_y = optimal_route[:, 1]
    
    # --- STEP 3: VISUALIZATION ---
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # Fade out the rest of the map
    plt.scatter(chest_matrix[:, 0], chest_matrix[:, 1], c='gray', s=10, alpha=0.2)
    
    # Plot the optimal route as a glowing line
    plt.plot(route_x, route_y, c='cyan', linewidth=2, zorder=2, label="Optimal Path")
    
    # Emphasize the chests in the route
    plt.scatter(best_x, best_y, c='gold', s=30, zorder=3, label="Chests")
    
    # Mark exactly where to drop (Start) and where you finish (End)
    plt.scatter(route_x[0], route_y[0], c='lime', s=100, marker='s', zorder=4, label="Drop Spot (Start)")
    plt.scatter(route_x[-1], route_y[-1], c='red', s=100, marker='X', zorder=4, label="End Route")
    
    plt.title(f"Optimal Loot Route - POI Density: {len(best_coords)} Chests")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    
else:
    print("Failed to fetch data.")
