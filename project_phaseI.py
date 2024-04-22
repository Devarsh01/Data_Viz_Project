#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import shapiro

#%%
title_style = {'fontname': 'serif', 'color': 'blue', 'size': 'large'}
label_style = {'fontname': 'serif', 'color': 'darkred'}

#%%
df = pd.read_csv("C:/Users/DEVARSH SHETH/Desktop/GWU/Data_Visulization/Project/Data_Viz_Project/exported_data.csv")  # Replace "your_dataset.csv" with the actual file path
# Display non-null count and dtype of each column in the uncleaned dataset
print("Non-null count and dtype of each column in the uncleaned dataset:")
print(df.info())

# Display the count of NaN values in each column
nan_count = df.isnull().sum()
print("Count of NaN values in each column:")
print(nan_count)
# %%

empty_developers = df[df['Developers'].isnull()]

# Retrieve the corresponding revenue values
empty_developers_revenue = empty_developers['Revenue']

print("Revenue corresponding to empty values in the 'Developers' column:")
print(empty_developers_revenue.to_string())
# %%
df_cleaned = df.dropna()

num_records_before = len(df)

# Remove rows with null values
df_cleaned = df.dropna()

# Number of records after removing null values
num_records_after = len(df_cleaned)

# Number of records removed due to null values
num_records_removed = num_records_before - num_records_after

# Print the results
print("Number of records before removing null values:", num_records_before)
print("Number of records after removing null values:", num_records_after)
print("Number of records removed due to null values:", num_records_removed)

#%%
# Display the first few observations of the cleaned dataset
print("\nFirst few observations of the cleaned dataset:")
print(df_cleaned.head())

# %%
print("\nStatistics of the cleaned dataset:")
print(df_cleaned.describe())
# %%
df_cleaned['Age'] = df_cleaned['Age'].astype(int)

numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Create box plots for each numerical column to visualize outliers
plt.figure(figsize=(16, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df_cleaned[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# %%
features = df_cleaned[numerical_columns]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform PCA
pca = PCA()
pca.fit(scaled_features)

# Explained Variance by Component
explained_variance = pca.explained_variance_ratio_
print("Explained Variance by Component:", explained_variance)

# Cumulative Variance Explained
cumulative_variance = np.cumsum(explained_variance)
print("Cumulative Variance Explained:", cumulative_variance)

# Singular Values
singular_values = pca.singular_values_
print("Singular Values:", singular_values)

# Compute the condition number
condition_number = np.linalg.cond(scaled_features)
print("Condition Number:", condition_number)

# %%
# Perform Shapiro-Wilk test for normality
statistic, p_value = shapiro(features)

# Print the test statistic and p-value
print("Shapiro-Wilk Test Statistic:", statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05  # significance level
if p_value > alpha:
    print("The data looks Gaussian (fail to reject H0)")
else:
    print("The data does not look Gaussian (reject H0)")

# %%

correlation_matrix = df_cleaned[numerical_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Coefficient Matrix", **title_style)
plt.xlabel("Variables", **label_style)
plt.ylabel("Variables", **label_style)
plt.gca().tick_params(axis='both', which='major', labelsize=10)
plt.show()

# %%

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))

# Plot KDE plots for each numerical variable
for i, column in enumerate(features.columns):
    if i >= 12:  # Plot only 12 KDE plots
        break
    sns.kdeplot(data=df_cleaned, x=column, fill=True, ax=axes.flatten()[i])
    axes.flatten()[i].set_title(f"{column} Distribution", **title_style)
    axes.flatten()[i].set_xlabel(column, **label_style)
    axes.flatten()[i].set_ylabel("Density", **label_style)
    axes.flatten()[i].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.show()
# %%
