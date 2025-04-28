import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import matplotlib

# Set global font to Calibri
matplotlib.rcParams['font.family'] = 'Calibri'
matplotlib.rcParams['font.size'] = 14

# Load the file
file_path = 'C:\\Users\\haojiecao\\Documents\\UGS\\ner\\ces_text.xlsx'
data = pd.read_excel(file_path, sheet_name='pro_all')

# Extract all columns with 'Sum' in their name
percentage_columns = [col for col in data.columns if '%' in col]

renamed_columns = [col.replace('(%)', '').strip() for col in percentage_columns]
data_renamed = data[percentage_columns].copy()
data_renamed.columns = renamed_columns


# Define a function to calculate p-values for each pair of variables
def calculate_p_values(df):
    p_values = pd.DataFrame(data=np.ones((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                _, p_value = pearsonr(df[col1], df[col2])
                p_values.loc[col1, col2] = p_value
            else:
                p_values.loc[col1, col2] = np.nan  # Diagonal will be NaN
    return p_values

# Calculate the correlation matrix and the p-value matrix
correlation_matrix_renamed = data_renamed.corr(method='spearman')
p_values_matrix_renamed = calculate_p_values(data_renamed)

rows, cols = correlation_matrix_renamed.shape
annotations_renamed = pd.DataFrame('', index=correlation_matrix_renamed.index, columns=correlation_matrix_renamed.columns)

for r in range(rows):
    for c in range(cols):
        correlation_value = correlation_matrix_renamed.iloc[r, c]
        p_value = p_values_matrix_renamed.iloc[r, c]
        significance = ''
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        annotations_renamed.iloc[r, c] = f'{correlation_value:.2f}{significance}'

# Visualization
plt.figure(figsize=(12, 10))

# Find the min and max values across the correlation matrix to define the color range symmetrically around 0
vmin, vmax = correlation_matrix_renamed.min().min(), correlation_matrix_renamed.max().max()
vmax = max(abs(vmin), abs(vmax))
vmin = -vmax


# Create the heatmap with combined annotations
sns.heatmap(correlation_matrix_renamed, annot=annotations_renamed, fmt='', cmap='coolwarm', cbar=True, square=True, vmin=vmin, vmax=vmax, center=0)

plt.xticks(rotation=45, ha='right', fontweight='bold', fontsize=15)
plt.yticks(rotation=0, fontweight='bold', fontsize=15)

plt.tight_layout()

# Save the plot as a jpg file
plt.savefig("C:\\Users\\haojiecao\\Documents\\UGS\\ner\\correlation_matrix\\correlation_matrix_pro_all_with_p_values_and_confidence_spearman.jpg", format='jpg', dpi=300)

plt.show()