import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set Display Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Specify the path to the Excel file
excel_file_path = Path("data") / "Synapt Settings Diagnostic Database.xlsm"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# List of columns to omit
columns_to_omit = ['Cone', 'Acc1b', 'Src. Offset', 'IE', 'Ref.', 'Quad Pr.', 'Date Modified', 'Puller', 'ADC Base', 'Amp. Th.', 'IA Th.', 'Capillary', 'IMS Vac', 'ToF Vac']

# Omit specified columns
df.drop(columns=columns_to_omit, inplace=True)

# Extract 'File' column for filtering later
files = df['File']

# Separate data into POS and NEG
df_pos = df[df['File'].str.contains('POS')]
df_neg = df[df['File'].str.contains('NEG')]

# Drop the 'File' column before applying PCA (since it's non-numeric)
df_numeric_pos = df_pos.drop(columns=['File'])
df_numeric_neg = df_neg.drop(columns=['File'])

# Standardize the data for POS
scaler_pos = StandardScaler()
df_standardized_pos = scaler_pos.fit_transform(df_numeric_pos)

# Standardize the data for NEG
scaler_neg = StandardScaler()
df_standardized_neg = scaler_neg.fit_transform(df_numeric_neg)

# Initialize PCA with the desired number of components
n_components = 3  # You can change this based on your requirements

# Fit and transform the standardized data for POS
pca_pos = PCA(n_components=n_components)
principal_components_pos = pca_pos.fit_transform(df_standardized_pos)

# Fit and transform the standardized data for NEG
pca_neg = PCA(n_components=n_components)
principal_components_neg = pca_neg.fit_transform(df_standardized_neg)

# Create a new DataFrame with the principal components for POS
columns_pos = [f'PC{i}' for i in range(1, n_components + 1)]
df_pca_pos = pd.DataFrame(data=principal_components_pos, columns=columns_pos, index=df_pos.index)

# Create a new DataFrame with the principal components for NEG
columns_neg = [f'PC{i}' for i in range(1, n_components + 1)]
df_pca_neg = pd.DataFrame(data=principal_components_neg, columns=columns_neg, index=df_neg.index)

# Print the explained variance ratio for POS
print("Explained Variance Ratio (POS):", pca_pos.explained_variance_ratio_)

# Print the explained variance ratio for NEG
print("Explained Variance Ratio (NEG):", pca_neg.explained_variance_ratio_)

# Create a table for POS
table_pos = pd.DataFrame({
    'Principal Component': [f'PC{i}' for i in range(1, n_components + 1)],
    'Eigenvalue': pca_pos.explained_variance_,
    'Proportion of Variation': pca_pos.explained_variance_ratio_,
    'Cumulative Percentage': np.cumsum(pca_pos.explained_variance_ratio_)
})

# Create a table for NEG
table_neg = pd.DataFrame({
    'Principal Component': [f'PC{i}' for i in range(1, n_components + 1)],
    'Eigenvalue': pca_neg.explained_variance_,
    'Proportion of Variation': pca_neg.explained_variance_ratio_,
    'Cumulative Percentage': np.cumsum(pca_neg.explained_variance_ratio_)
})

# Display the tables
print("Table for POS:")
print(table_pos)

print("\nTable for NEG:")
print(table_neg)

# Get the loading scores for POS
loading_scores_pos = pca_pos.components_

# Get the loading scores for NEG
loading_scores_neg = pca_neg.components_

# Create a table for POS
table_correlation_pos = pd.DataFrame(data=loading_scores_pos.T, columns=[f'PC{i}' for i in range(1, n_components + 1)])
table_correlation_pos.index = df_numeric_pos.columns
table_correlation_pos.index.name = 'Variable'
table_correlation_pos.columns.name = 'Principal Component (POS)'

# Create a table for NEG
table_correlation_neg = pd.DataFrame(data=loading_scores_neg.T, columns=[f'PC{i}' for i in range(1, n_components + 1)])
table_correlation_neg.index = df_numeric_neg.columns
table_correlation_neg.index.name = 'Variable'
table_correlation_neg.columns.name = 'Principal Component (NEG)'

# Display the tables
print("Table for POS (Correlations between Principal Components and Variables):")
print(table_correlation_pos)

print("\nTable for NEG (Correlations between Principal Components and Variables):")
print(table_correlation_neg)

# Initialize arrays to store top contributors for each sample type
top_contributors_pos = []
top_contributors_neg = []

# Loop through POS and NEG
for sample_type, df_standardized, df_original in zip(['POS', 'NEG'], [df_standardized_pos, df_standardized_neg], [df_pos, df_neg]):
    # Fit and transform the standardized data
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_standardized)

    # Get the loading scores for the first principal component
    loading_scores_pc1 = pca.components_[0, :]

    # Find the indices of the top n rows with the highest loading scores
    n_top_rows = 5  # Change this based on how many top rows you want to identify
    top_rows_indices = np.abs(loading_scores_pc1).argsort()[-n_top_rows:][::-1]

    # Display the rows with the highest loading scores for PC1 and their corresponding 'File' values
    top_contributors = df_original.iloc[top_rows_indices]
    print(f"Top Contributors to PC1 ({sample_type}):")
    print(top_contributors[['File'] + list(df_original.columns)])

    # Store the top contributors for later use
    if sample_type == 'POS':
        top_contributors_pos.append(top_contributors)
    else:
        top_contributors_neg.append(top_contributors)

    # Store the top contributors for later use
    if sample_type == 'POS':
        top_contributors_pos.append(top_contributors)
    else:
        top_contributors_neg.append(top_contributors)

# Plot the scree plot to visualize explained variance for POS
plt.bar(range(1, n_components + 1), pca_pos.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (POS)')
plt.title('Scree Plot (POS)')
plt.show()

# Plot the scree plot to visualize explained variance for NEG
plt.bar(range(1, n_components + 1), pca_neg.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (NEG)')
plt.title('Scree Plot (NEG)')
plt.show()

# Optional: Plot the data in the new PCA space for POS
fig_pos = plt.figure(figsize=(8, 6))
ax_pos = fig_pos.add_subplot(111, projection='3d')
ax_pos.scatter(df_pca_pos['PC1'], df_pca_pos['PC2'], df_pca_pos['PC3'])
ax_pos.set_xlabel('Principal Component 1 (POS)')
ax_pos.set_ylabel('Principal Component 2 (POS)')
ax_pos.set_zlabel('Principal Component 3 (POS)')
ax_pos.set_title('3D Scatter Plot of PCA (POS)')
plt.show()

# Optional: Plot the data in the new PCA space for NEG
fig_neg = plt.figure(figsize=(8, 6))
ax_neg = fig_neg.add_subplot(111, projection='3d')
ax_neg.scatter(df_pca_neg['PC1'], df_pca_neg['PC2'], df_pca_neg['PC3'])
ax_neg.set_xlabel('Principal Component 1 (NEG)')
ax_neg.set_ylabel('Principal Component 2 (NEG)')
ax_neg.set_zlabel('Principal Component 3 (NEG)')
ax_neg.set_title('3D Scatter Plot of PCA (NEG)')
plt.show()

