import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Calibri'

# Data for global importance
global_importance_data = {
    "Variable": ['Total area of UGS', 'Land cover: Tree', 'Land cover: Grassland', 'Land cover: Water', 'Land cover: Herbaceous wetland',
                 'Land cover: Mangrove', 'Mean NDVI of UGS', 'Mean NDWI of UGS', 'Human utility: Picnic Area',
                 'Human utility: Playground', 'Human utility: Body of water', 'Human utility: Walk Path',
                 'Human utility: Athletic facility', 'Human utility: Nature preserve', 'Human utility: Dog park',
                 'Human utility: Fitness center', 'Biodiversity utility'],
    "Importance": [281.663597, 224.832143, 91.725641, 105.178493, 75.655582,
                   4.558466, 98.609390, 129.186881, 27.327247, 22.008327,
                   29.898055, 15.333269, 13.368712, 13.891128, 11.198632,
                   8.778511, 172.682437]
}

# Data for local mean importance
local_importance_mean_data = {
    "Variable": ['Total area of UGS', 'Land cover: Tree', 'Land cover: Grassland', 'Land cover: Water', 'Land cover: Herbaceous wetland',
                 'Land cover: Mangrove', 'Mean NDVI of UGS', 'Mean NDWI of UGS', 'Human utility: Picnic Area',
                 'Human utility: Playground', 'Human utility: Body of water', 'Human utility: Walk Path',
                 'Human utility: Athletic facility', 'Human utility: Nature preserve', 'Human utility: Dog park',
                 'Human utility: Fitness center', 'Biodiversity utility'],
    "Importance": [5.1285843, 4.9195102, 2.9959396, 2.1713576, 2.9242892,
                   0.2942298, 3.1432307, 3.3157908, 0.7730455, 0.8053452,
                   1.3179296, 0.7325500, 0.8272430, 0.6880646, 0.2032272,
                   0.6209912, 4.3245778]
}

current_font_size = plt.rcParams['font.size']

color_map_hex = {
    'Total area of UGS': '#D9EDBF',
    'Land cover: ': '#FF9800',
    'Mean NDVI of UGS': '#2C7865',
    'Mean NDWI of UGS': '#90D26D',
    'Human utility: ': '#EADFB4',
    'Biodiversity utility': '#9BB0C1'
}

def assign_color(variable, color_map):
    for key, color in color_map.items():
        if variable.startswith(key) or variable == key:
            return color
    return 'gray'

# Convert dictionaries to pandas DataFrame
global_df = pd.DataFrame(global_importance_data).sort_values(by='Importance', ascending=True)
local_df = pd.DataFrame(local_importance_mean_data).sort_values(by='Importance', ascending=True)

# Assign colors based on variable name using the new hex color map
global_df['Color'] = global_df['Variable'].apply(lambda x: assign_color(x, color_map_hex))
local_df['Color'] = local_df['Variable'].apply(lambda x: assign_color(x, color_map_hex))

# Global Importance Plot
plt.figure(figsize=(10, 8))
ax = plt.gca()
global_bars = ax.barh(global_df['Variable'], global_df['Importance'], color=global_df['Color'])
ax.set_xlabel('Global Variable Importance', fontweight='bold', fontsize=18)
ax.set_ylabel('Variables', fontweight='bold', fontsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Set limits for annotations
buffer_space = max(global_df['Importance']) * 0.05
ax.set_xlim(0, max(global_df['Importance']) + buffer_space)

# Annotations for Global Importance
for bar, value in zip(global_bars, global_df['Importance']):
    ax.text(value + buffer_space * 0.1, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center', fontsize=16)

# Spine and layout adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('C:\\Users\\haojiecao\\Documents\\UGS\\ner\\importance\\importance_python\\global_importance_plot_color.jpg', format='jpg', dpi=300)

# Local Importance Plot
plt.figure(figsize=(10, 8))
local_bars = plt.barh(local_df['Variable'], local_df['Importance'], color=local_df['Color'])
plt.xlabel('Local Mean Variable Importance', fontweight='bold', fontsize=18)
plt.ylabel('Variables', fontweight='bold', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Annotations for Local Importance
for bar, value in zip(local_bars, local_df['Importance']):
    plt.text(value + max(local_df['Importance']) * 0.01, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center', fontsize=16)

# Spine and layout adjustments
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('C:\\Users\\haojiecao\\Documents\\UGS\\ner\\importance\\importance_python\\local_importance_plot_color.jpg', format='jpg', dpi=300)
