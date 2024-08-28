import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set Display Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Specify the paths to the Excel files
excel_file_path_1 = Path("data") / "Synapt Settings Diagnostic Database.xlsm"
excel_file_path_2 = Path("data") / "Failing Test Data.xlsx"

# Read the Excel files into DataFrames
df_1 = pd.read_excel(excel_file_path_1)
df_2 = pd.read_excel(excel_file_path_2)

# List of columns to omit
columns_to_omit = ['P. Offset', 'Ref. Grid','Cone', 'Acc1b', 'Src. Offset', 'IE', 'Ref.', 'Quad Pr.', 'Date Modified', 'Puller', 'ADC Base', 'Amp. Th.', 'IA Th.', 'Capillary', 'IMS Vac', 'ToF Vac']

# Omit specified columns from both DataFrames
df_1.drop(columns=columns_to_omit, inplace=True)
df_2.drop(columns=columns_to_omit, inplace=True)

# Extract 'File' column for filtering later
files_1 = df_1['File']
files_2 = df_2['File']

# Concatenate the two DataFrames vertically
df_combined = pd.concat([df_1, df_2], ignore_index=True)

# Separate data into POS and NEG for the combined DataFrame
df_pos_combined = df_combined[df_combined['File'].str.contains('POS')]
df_neg_combined = df_combined[df_combined['File'].str.contains('NEG')]

# Drop the 'File' column before applying PCA (since it's non-numeric)
df_numeric_pos_combined = df_pos_combined.drop(columns=['File'])
df_numeric_neg_combined = df_neg_combined.drop(columns=['File'])

# Standardize the data for POS
scaler_pos_combined = StandardScaler()
df_normalized_pos_combined = scaler_pos_combined.fit_transform(df_numeric_pos_combined)

# Standardize the data for NEG
scaler_neg_combined = StandardScaler()
df_normalized_neg_combined = scaler_neg_combined.fit_transform(df_numeric_neg_combined)

# Initialize PCA with the desired number of components
n_components_combined = 3  # You can change this based on your requirements

# Fit and transform the standardized data for POS
pca_pos_combined = PCA(n_components=n_components_combined)
principal_components_pos_combined = pca_pos_combined.fit_transform(df_normalized_pos_combined)

# Fit and transform the standardized data for NEG
pca_neg_combined = PCA(n_components=n_components_combined)
principal_components_neg_combined = pca_neg_combined.fit_transform(df_normalized_neg_combined)

# Create a new DataFrame with the principal components for POS
columns_pos_combined = [f'PC{i}' for i in range(1, n_components_combined + 1)]
df_pca_pos_combined = pd.DataFrame(data=principal_components_pos_combined, columns=columns_pos_combined, index=df_pos_combined.index)

# Create a new DataFrame with the principal components for NEG
columns_neg_combined = [f'PC{i}' for i in range(1, n_components_combined + 1)]
df_pca_neg_combined = pd.DataFrame(data=principal_components_neg_combined, columns=columns_neg_combined, index=df_neg_combined.index)

# Print the explained variance ratio for POS
print("Explained Variance Ratio (POS):", pca_pos_combined.explained_variance_ratio_)

# Print the explained variance ratio for NEG
print("Explained Variance Ratio (NEG):", pca_neg_combined.explained_variance_ratio_)

# Create a table for POS
table_pos = pd.DataFrame({
    'Principal Component': [f'PC{i}' for i in range(1, n_components_combined + 1)],
    'Eigenvalue': pca_pos_combined.explained_variance_,
    'Proportion of Variation': pca_pos_combined.explained_variance_ratio_,
    'Cumulative Percentage': np.cumsum(pca_pos_combined.explained_variance_ratio_)
})

# Create a table for NEG
table_neg = pd.DataFrame({
    'Principal Component': [f'PC{i}' for i in range(1, n_components_combined + 1)],
    'Eigenvalue': pca_neg_combined.explained_variance_,
    'Proportion of Variation': pca_neg_combined.explained_variance_ratio_,
    'Cumulative Percentage': np.cumsum(pca_neg_combined.explained_variance_ratio_)
})

# Display the tables
print("Table for POS:")
print(table_pos)

print("\nTable for NEG:")
print(table_neg)

# Get the loading scores for POS
loading_scores_pos = pca_pos_combined.components_

# Get the loading scores for NEG
loading_scores_neg = pca_neg_combined.components_

# Create a table for POS
table_correlation_pos = pd.DataFrame(data=loading_scores_pos.T, columns=[f'PC{i}' for i in range(1, n_components_combined + 1)])
table_correlation_pos.index = df_numeric_pos_combined.columns
table_correlation_pos.index.name = 'Variable'
table_correlation_pos.columns.name = 'Principal Component (POS)'

# Create a table for NEG
table_correlation_neg = pd.DataFrame(data=loading_scores_neg.T, columns=[f'PC{i}' for i in range(1, n_components_combined + 1)])
table_correlation_neg.index = df_numeric_neg_combined.columns
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
for sample_type, df_standardized, df_original in zip(['POS', 'NEG'], [df_normalized_pos_combined, df_normalized_neg_combined], [df_pos_combined, df_neg_combined]):
    # Fit and transform the standardized data
    pca = PCA(n_components=n_components_combined)
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

# Optional: Plot the data in the new PCA space for POS
# Create an interactive 3D scatter plot
fig_pos_combined = px.scatter_3d(df_pca_pos_combined, x='PC1', y='PC2', z='PC3', title='3D Scatter Plot of PCA (POS - Combined)',
                                  color_discrete_sequence=['blue'])

# Optional: Color only the data points from 'Failing Test Data.xlsx' in red
fig_pos_combined.update_traces(marker=dict(color=np.where(df_pos_combined['File'] == 'POS_TEST', 'red', 'blue')))

fig_pos_combined.show()

# Optional: Plot the data in the new PCA space for NEG
# Create an interactive 3D scatter plot
fig_neg_combined = px.scatter_3d(df_pca_neg_combined, x='PC1', y='PC2', z='PC3', title='3D Scatter Plot of PCA (NEG - Combined)',
                                  color_discrete_sequence=['blue'])

# Optional: Color only the data points from 'Failing Test Data.xlsx' in red
fig_neg_combined.update_traces(marker=dict(color=np.where(df_neg_combined['File'] == 'NEG_TEST', 'red', 'blue')))

fig_neg_combined.show()

def create_table_trace(dataframe, title, include_index=True):
    if include_index:
        trace = go.Table(
            header=dict(values=[dataframe.index.name] + list(dataframe.columns)),
            cells=dict(values=[dataframe.index] + [dataframe[col] for col in dataframe.columns]),
            name=title
        )
    else:
        trace = go.Table(
            header=dict(values=list(dataframe.columns)),
            cells=dict(values=[dataframe[col] for col in dataframe.columns]),
            name=title
        )
    return trace

# Create table traces
table_pos_trace = create_table_trace(table_pos, 'Table for POS', include_index=False)
table_neg_trace = create_table_trace(table_neg, 'Table for NEG', include_index=False)
table_correlation_pos_trace = create_table_trace(table_correlation_pos, 'Table for POS (Correlations between Principal Components and Variables)')
table_correlation_neg_trace = create_table_trace(table_correlation_neg, 'Table for NEG (Correlations between Principal Components and Variables)')

# Create figures for each table
fig_pos_table = go.Figure(table_pos_trace)
fig_neg_table = go.Figure(table_neg_trace)
fig_correlation_pos_table = go.Figure(table_correlation_pos_trace)
fig_correlation_neg_table = go.Figure(table_correlation_neg_trace)

# Update layout for better visibility
fig_pos_table.update_layout(title_text="Table for POS")
fig_neg_table.update_layout(title_text="Table for NEG")
fig_correlation_pos_table.update_layout(title_text="Table for POS (Correlations)")
fig_correlation_neg_table.update_layout(title_text="Table for NEG (Correlations)")

# Show the figures
fig_pos_table.show()
fig_neg_table.show()
fig_correlation_pos_table.show()
fig_correlation_neg_table.show()

from plotly.subplots import make_subplots

# Function to create a table trace
def create_table_trace(dataframe, title, include_index=True):
    if include_index:
        trace = go.Table(
            header=dict(values=[dataframe.index.name] + list(dataframe.columns)),
            cells=dict(values=[dataframe.index] + [dataframe[col] for col in dataframe.columns]),
            name=title
        )
    else:
        trace = go.Table(
            header=dict(values=list(dataframe.columns)),
            cells=dict(values=[dataframe[col] for col in dataframe.columns]),
            name=title
        )
    return trace

# Create table traces
table_pos_trace = create_table_trace(table_pos, 'Table for POS', include_index=False)
table_neg_trace = create_table_trace(table_neg, 'Table for NEG', include_index=False)
table_correlation_pos_trace = create_table_trace(table_correlation_pos, 'Table for POS (Correlations between Principal Components and Variables)')
table_correlation_neg_trace = create_table_trace(table_correlation_neg, 'Table for NEG (Correlations between Principal Components and Variables)')

# Create subplots for all figures
fig_combined = make_subplots(rows=2, cols=2,
                             subplot_titles=["Table for POS", "Table for NEG", "Table for POS (Correlations)", "Table for NEG (Correlations)"])

# Add traces to subplots
fig_combined.add_trace(table_pos_trace, row=1, col=1)
fig_combined.add_trace(table_neg_trace, row=1, col=2)
fig_combined.add_trace(table_correlation_pos_trace, row=2, col=1)
fig_combined.add_trace(table_correlation_neg_trace, row=2, col=2)

# Update layout for better visibility
fig_combined.update_layout(title_text="Combined Tables")

# Show the combined figure
fig_combined.show()