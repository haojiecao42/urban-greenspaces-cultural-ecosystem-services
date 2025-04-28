import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from math import pi
import matplotlib


# Load data
data = pd.read_csv('C:\\Users\\haojiecao\\Documents\\UGS\\ner\\correlation_matrix\\all_pro.csv')
numeric_data = data.drop('UGS', axis=1)
column_names = numeric_data.columns

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(numeric_data)

# Set global font to Calibri
matplotlib.rcParams['font.family'] = 'Calibri'
matplotlib.rcParams['font.size'] = 14

# Parameters for experiment
som_dimensions = [3, 4, 5, 6]  # You can adjust this based on your specific needs
random_seeds = [42, 123, 456, 789]
iterations = 1050
initial_learning_rate = 0.05
final_learning_rate = 0.01
results = []

# Run experiments with different configurations and random seeds
for seed in random_seeds:
    np.random.seed(seed)
    for som_dimension in som_dimensions:
        som = MiniSom(som_dimension, som_dimension, len(X_scaled[0]), sigma=1.0, learning_rate=0.05)

        # Train SOM with decaying learning rate
        for i in range(iterations):
            t = i / iterations
            current_learning_rate = final_learning_rate + (initial_learning_rate - final_learning_rate) * (1 - t)
            random_index = np.random.randint(len(X_scaled))
            random_sample = X_scaled[random_index]
            som.update(random_sample, som.winner(random_sample), i, current_learning_rate)

        # Evaluate clusters using silhouette score
        cluster_assignments = [som.winner(x) for x in X_scaled]
        cluster_indices = np.ravel_multi_index(np.array(cluster_assignments).T, (som_dimension, som_dimension))
        swi = silhouette_score(X_scaled, cluster_indices)
        results.append({'seed': seed, 'dimension': som_dimension, 'SWI': swi})

# Analyzing results to find the best configuration
results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['SWI'].idxmax()]
best_seed = int(best_result['seed'])
best_dimension = int(best_result['dimension'])
print(f"Best SWI: {best_result['SWI']} at dimension {best_dimension} with seed {best_seed}")

# Train SOM with the best configuration to visualize
np.random.seed(best_seed)
best_som = MiniSom(best_dimension, best_dimension, len(X_scaled[0]), sigma=1.0, learning_rate=0.05)
for i in range(iterations):
    t = i / iterations
    current_learning_rate = final_learning_rate + (initial_learning_rate - final_learning_rate) * (1 - t)
    random_index = np.random.randint(len(X_scaled))
    random_sample = X_scaled[random_index]
    best_som.update(random_sample, best_som.winner(random_sample), i, current_learning_rate)

# Get unique cluster indices from the best configuration
best_cluster_assignments = [best_som.winner(x) for x in X_scaled]
best_cluster_indices = np.ravel_multi_index(np.array(best_cluster_assignments).T, (best_dimension, best_dimension))
unique_cluster_indices = set(best_cluster_indices)

# Function to plot radar charts
def make_radar_chart(name, stats, feature_names):
    num_vars = len(stats)
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += [angles[0]]
    stats = stats.tolist() + [stats[0]]
    feature_names = [name.replace('(%)', '') for name in feature_names]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='grey', alpha=0.5)
    ax.plot(angles, stats, color='black', linewidth=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, color='white', size=-1, fontweight='bold', verticalalignment='top')

    ax.set_yticklabels([])

    # plt.title(name, size=20, color='red', y=1.1)
    plt.savefig(f'C:\\Users\\haojiecao\\Documents\\UGS\\ner\\correlation_matrix\\Cluster_{name}.png',bbox_inches='tight')
    plt.close()

# Plotting radar charts for each unique cluster center
for cluster_index in unique_cluster_indices:
    cluster_coords = np.unravel_index(cluster_index, (best_dimension, best_dimension))
    weights = best_som.get_weights()[cluster_coords[0], cluster_coords[1]]
    make_radar_chart(f"Cluster {cluster_index}", weights, column_names)

# Adding cluster index to each data point
data['Cluster_Index'] = [np.ravel_multi_index(best_som.winner(x), (best_dimension, best_dimension)) for x in X_scaled]

# Export the data with cluster indices to a new CSV file
output_path = 'C:\\Users\\haojiecao\\Documents\\UGS\\ner\\correlation_matrix\\clustered_data_all_pro.csv'
data.to_csv(output_path, index=False)
print(f"Data with cluster indices has been saved to {output_path}")