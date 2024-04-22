#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

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
numerical_columns = ['Age', 'Average In-game purchase', 'Negative', 'Peak CCU', 
                     'Positive', 'Price', 'Revenue', 'User score']

# Create box plots for each numerical column to visualize outliers
plt.figure(figsize=(16, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_cleaned[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# %%
features = df_cleaned.select_dtypes(include=['float64', 'int64'])

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
from scipy.stats import shapiro

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

