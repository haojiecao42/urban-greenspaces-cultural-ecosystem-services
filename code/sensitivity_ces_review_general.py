import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data
data = pd.read_excel('C:\\Users\\haojiecao\\Documents\\UGS\\ner\\ces_text.xlsx', sheet_name='no_religious')

# Set a random seed for reproducibility
np.random.seed(42)

# Step 1: Group Data by "UGS"
groups = data.groupby('UGS')

# Step 2 and 3: Sample and Count Non-Zero Columns
results = []

for name, group in groups:
    max_samples = len(group)
    for percentage in np.linspace(0, 1, 101):  # Creates a sequence from 0% to 100%
        sample_size = int(max_samples * percentage)
        if sample_size > 0:
            sample = group.sample(n=sample_size, random_state=42)
            non_zero_columns = (sample.iloc[:, 1:11] != 0).any(axis=0).sum()  # Adjust the column indices as needed
            results.append((sample_size, non_zero_columns))

# Convert results to DataFrame for easier handling
results_df = pd.DataFrame(results, columns=['SampleSize', 'NonZeroColumns'])

# Aggregate the results to find the minimum number of reviews for each NonZeroCES value
# This involves grouping by NonZeroCES and finding the minimum SampleSize across all UGS groups
min_reviews_per_ces_sensitivity = results_df.groupby('NonZeroColumns')['SampleSize'].min().reset_index()


plt.figure(figsize=(16, 6))  # Set total figure size to accommodate both plots side by side
plt.rcParams['font.family'] = 'Calibri'

# Create a subplot grid of 1 row by 2 columns
# Then, plot the scatter plot on the left (subplot 1)
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.scatter(results_df['SampleSize'], results_df['NonZeroColumns'], alpha=0.5, color='black', s=60)
current_font_size = 16
plt.xlabel('CES-related review subset', fontweight='bold', fontsize=current_font_size + 2)
plt.ylabel('Number of CES categories of UGS', fontweight='bold', fontsize=current_font_size + 2)
plt.tick_params(axis='both', which='major', labelsize=current_font_size)  # Set font size for x and y tick labels
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Now, plot the bar plot on the right (subplot 2)
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
bars = plt.barh(min_reviews_per_ces_sensitivity['NonZeroColumns'], min_reviews_per_ces_sensitivity['SampleSize'], color='gray')
current_font_size = 16
plt.ylabel('Number of CES categories of UGS', fontweight='bold', fontsize=current_font_size + 1)
plt.xlabel('Minimum CES-related review subset size', fontweight='bold', fontsize=current_font_size + 1)
plt.tick_params(axis='both', which='major', labelsize=current_font_size)  # Adjusting font size for tick labels
# Placing labels on the bars correctly
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)}', va='center', ha='left', fontsize=current_font_size)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()  # Adjust layout to not overlap
plt.savefig('C:\\Users\\haojiecao\\Documents\\UGS\\ner\\sensitivity\\sensitivity.jpg', format='jpg', dpi=300)  # Save the combined figure
plt.show()
