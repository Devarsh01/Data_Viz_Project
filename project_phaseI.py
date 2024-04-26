#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import shapiro, normaltest, anderson
from scipy.stats import normaltest
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

#%%
title_style = {'fontname': 'serif', 'color': 'blue', 'size': 'large'}
label_style = {'fontname': 'serif', 'color': 'darkred'}

#%%
df = pd.read_csv("https://raw.githubusercontent.com/Devarsh01/Data_Viz_Project/main/exported_data.csv")
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

numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64', 'int32']).columns

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

plt.figure(figsize=(8, 5))

# Creating the bar plot for individual explained variance
bar = plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
              label='Individual explained variance', color='blue')

# Creating the line plot for cumulative explained variance
line = plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='red',
                label='Cumulative explained variance')

plt.title('Scree Plot of Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

# %%
# Perform Shapiro-Wilk test for normality
shapiro_stat, shapiro_p_value = shapiro(features)

# Print the test statistic and p-value for Shapiro-Wilk Test
print("Shapiro-Wilk Test Statistic:", shapiro_stat)
print("P-value:", shapiro_p_value)

for column in features.columns:
    stat, p = normaltest(features[column])
    print(f"Feature: {column}")
    print('D\'Agostino\'s K-squared test Statistic:', stat)
    print('P-value:', p)
    alpha = 0.05  # significance level
    if p > alpha:
        print(f"The data in {column} looks Gaussian (fail to reject H0).")
    else:
        print(f"The data in {column} does not look Gaussian (reject H0).")
    print("\n")

# Interpret the results for Shapiro-Wilk Test
alpha = 0.05  # significance level
if shapiro_p_value > alpha:
    print("\nThe data looks Gaussian according to Shapiro-Wilk Test (fail to reject H0)")
else:
    print("\nThe data does not look Gaussian according to Shapiro-Wilk Test (reject H0)")

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

# Display datatypes of each column in df_cleaned
print(df_cleaned.dtypes)

# %%

# Extract unique genres from the dataset
unique_genres = set()

# Iterate over each row and split the genres
for genres_str in df_cleaned['Genres']:
    genres_list = genres_str.split(',')
    unique_genres.update(genres_list)

# Display the unique genres
print("Unique Genres:")
for genre in unique_genres:
    print(genre)

# %%
# Line-Plot
selected_genres = ['Indie', 'Action', 'Racing', 'Strategy', 'Adventure']

# Filter the dataframe for the selected genres
filtered_df = df_cleaned[df_cleaned['Genres'].str.contains('|'.join(selected_genres))]

# Group the data by genre and calculate the mean Peak CCU for each genre
genre_peak_ccu = filtered_df.groupby('Genres')['Peak CCU'].mean().sort_values(ascending=False)

# Plot the line plot for Peak CCU for the selected genres
plt.figure(figsize=(10, 6))
for genre in selected_genres:
    plt.plot(filtered_df[filtered_df['Genres'].str.contains(genre)].groupby('Year')['Peak CCU'].mean(), label=genre)

# Add labels and title
plt.xlabel('Year', **label_style)
plt.ylabel('Peak CCU', **label_style)
plt.title('Peak CCU for Selected Genres Over Time', **title_style)
plt.legend()
plt.grid(True)
plt.show()

# %%
from prettytable import PrettyTable

# Create a PrettyTable object
table = PrettyTable()

# Define the table headers
table.field_names = ["Genre", "Mean Peak CCU"]

# Add data to the table for all selected genres
for genre in selected_genres:
    mean_peak_ccu = genre_peak_ccu.loc[genre] if genre in genre_peak_ccu.index else "N/A"
    table.add_row([genre, mean_peak_ccu])

# Print the table
print(table)

print('''- Indie: 5.68 (approximately): This indicates that, on average, indie games have around 5.68 peak concurrent users.
- Action: 369.93 (approximately): This suggests that action games have a significantly higher average peak concurrent user count, with around 369.93 concurrent users on average.
- Racing: 56.56 (approximately): Racing games have an average peak concurrent user count of around 56.56 users.
- Strategy: 230.04 (approximately): Strategy games also show a relatively high average peak concurrent user count, with around 230.04 concurrent users on average.
- Adventure: 23.37 (approximately): Adventure games have a lower average peak concurrent user count compared to other genres, with around 23.37 concurrent users on average.''')

# %%
#Stacked Bar Plot
# Extract individual genres from the combined string format
df_cleaned['Genre'] = df_cleaned['Genres'].str.split(',')

# Create a new dataframe by exploding the 'Genre' column
df_expanded = df_cleaned.explode('Genre')

games_by_genre = df_expanded['Genre'].value_counts().head(5).index

# Filter the data to include only the top 5 genres
top_genres_df = df_expanded[df_expanded['Genre'].isin(games_by_genre)]

# Group the filtered data by genre and playtime category and count the number of games in each group
games_by_genre_playtime = top_genres_df.groupby(['Genre', 'Playtime']).size().unstack(fill_value=0)
games_by_genre_playtime.drop(columns=['LOW'], inplace=True)

# Plot a stacked bar plot
plt.figure(figsize=(12, 8))
games_by_genre_playtime.plot(kind='bar', stacked=True)
plt.title('Number of Games by Genre and Playtime Category (Top 5 Genres)', **title_style)
plt.xlabel('Genre', **label_style)
plt.ylabel('Number of Games', **label_style)
plt.xticks(rotation=45, ha='right', **label_style)
plt.legend(title='Playtime Category')
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
# Bar Plot Group
games_by_developers = df_cleaned['Developers'].value_counts()

top_publishers = games_by_developers.nlargest(5).index

# Retrieve the count of games and Peak CCU for the top 10 publishers
games_counts = games_by_developers.loc[top_publishers]
peak_ccu_values = df_cleaned[df_cleaned['Developers'].isin(top_publishers)].groupby('Developers')['Peak CCU'].sum()

# Plot a grouped bar plot
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(top_publishers))

plt.bar(index, games_counts, bar_width, color='skyblue', label='Number of Games Developed')
plt.bar(index + bar_width, peak_ccu_values, bar_width, color='orange', label='Peak CCU')

plt.title('Top 5 Developers by Number of Games Developed and Peak CCU', **title_style)
plt.xlabel('Developers', **label_style)
plt.ylabel('Count / Peak CCU', **label_style)
plt.xticks(index + bar_width / 2, top_publishers, rotation=45, ha='right', color='black')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
# Count Plot
# Plot the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='Quarter', palette='Set2')
plt.title('Number of Games Released by Quarter', **title_style)
plt.xlabel('Quarter', **label_style)
plt.ylabel('Number of Games Released', **label_style)
plt.xticks(color='black')
plt.yticks(color='black')
plt.tight_layout()
plt.show()

# %%
# pie Plot

year_intervals = pd.cut(df_cleaned['Year'], bins=range(2005, 2026, 3), right=False)

# Count the number of games released in each interval
games_by_year_interval = df_cleaned.groupby(year_intervals).size()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(games_by_year_interval, labels=games_by_year_interval.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Games Released by Year Interval', **title_style)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.legend()
plt.show()
# %%
# Dist plot

plt.figure(figsize=(10, 6))
sns.distplot(df_cleaned['Price'], kde=True, color='skyblue')
plt.title('Distribution of Price', **title_style)
plt.xlabel('Price', **label_style)
plt.ylabel('Density', **label_style)
plt.tight_layout()
plt.show()
# %%
# Pair Plot

selected_features = ['Peak CCU', 'Price', 'Age']

# Set the style for the plot
sns.set_style("whitegrid")

# Create the pair plot
sns.pairplot(df_cleaned[selected_features])
plt.suptitle('Pair Plot of Selected Numerical Features', **title_style)
plt.tight_layout()
plt.show()
# %%
# histogram plot with KDE

plt.figure(figsize=(10, 6))
sns.distplot(df_cleaned['Positive'], kde=True, color='skyblue')
plt.title('Distribution of Positive Reviews', **title_style)
plt.xlabel('Positive Reviews', **label_style)
plt.ylabel('Density', **label_style)
plt.tight_layout()
plt.show()

# %%
# QQ-Plot
features = ['Average playtime forever', 'Revenue', 'Price', 'Negative']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot QQ plots for each feature
for i, feature in enumerate(features):
    # Fit a QQ plot comparing the distribution of the feature with a normal distribution
    sm.qqplot(df_cleaned[feature], line='45', ax=axes[i])
    axes[i].set_title(f'QQ Plot: {feature} vs. Normal Distribution', **title_style)
    axes[i].set_xlabel('Theoretical Quantiles', **label_style)
    axes[i].set_ylabel('Sample Quantiles', **label_style)

plt.tight_layout()
plt.show()
# %%
# KDE Fill

custom_palette = ["#ff7f0e"]

# Plot KDE plot with filled areas and custom palette
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_cleaned, x='Price', fill=True, palette=custom_palette)
plt.title('Kernel Density Estimation of Price', **title_style)
plt.xlabel('Price', **label_style)
plt.ylabel('Density', **label_style)
plt.grid(True)
plt.show()

# %%
# reg Line

plt.figure(figsize=(10, 6))
sns.regplot(data=df_cleaned, x='Peak CCU', y='Revenue', scatter_kws={'alpha':0.5})
plt.title('Regression of Revenue against Peak CCU', **title_style)
plt.xlabel('Peak CCU', **label_style)
plt.ylabel('Revenue', **label_style)
plt.grid(True)
plt.show()
# %%
# Multi Box Plot

df_cleaned['Categories'] = df_cleaned['Categories'].str.split(',')

# Explode the dataframe to create a row for each category
df_expanded = df_cleaned.explode('Categories')

# Plot the boxen plot
plt.figure(figsize=(12, 8))
sns.boxenplot(data=df_expanded, x='Categories', y='Price', palette='Set2')
plt.title('Price Distribution Across Different Categories', **title_style)
plt.xlabel('Categories', **label_style)
plt.ylabel('Price', **label_style)
plt.xticks(rotation=45, ha='right', color='black')
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
# Area Plot

cumulative_positive_reviews = df_cleaned.groupby('Year')['Positive'].sum().cumsum()

# Plot the area plot
plt.figure(figsize=(12, 8))
plt.fill_between(cumulative_positive_reviews.index, cumulative_positive_reviews, color='skyblue', alpha=0.5)

# Add labels and title
plt.title('Cumulative Positive Reviews Over Time', **title_style)
plt.xlabel('Year', **label_style)
plt.ylabel('Cumulative Positive Reviews', **label_style)
plt.xticks(**label_style)
plt.yticks(**label_style)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# violin Plot

playtime_counts = df_cleaned['Playtime'].value_counts()

# Plot a bar plot
plt.figure(figsize=(12, 8))
sns.violinplot(data=df_cleaned, x='Playtime', y='Average playtime forever', inner='quartile', palette='muted')
plt.title('Violin Plots of Average Playtime by Playtime Category', **title_style)
plt.xlabel('Playtime Category', **label_style)
plt.ylabel('Average Playtime', **label_style)
plt.tight_layout()
plt.show()

# %%
# Joint KDE

plt.figure(figsize=(8, 6))
sns.jointplot(data=df_cleaned, x='Price', y='Average In-game purchase', kind='scatter', color='purple')
plt.title('Joint Distribution of Price, Average In-game Purchase, and Revenue', **title_style)
plt.xlabel('Price', **label_style)
plt.ylabel('Average In-game Purchase', **label_style)
plt.tight_layout()
plt.show()
# %%
# Rug Plot Revenue

plt.figure(figsize=(8, 6))
sns.kdeplot(data=df_cleaned, x='Revenue', fill=True, color='orange', linewidth=2)
sns.rugplot(data=df_cleaned, x='Revenue', height=0.1, color='black', alpha=0.5)
plt.title('KDE Plot of Revenue with Rug Plots', **title_style)
plt.xlabel('Revenue', **label_style)
plt.ylabel('Density', **label_style)
plt.tight_layout()
plt.show()
# %%
# Cluster Map

selected_features = df_cleaned[['Peak CCU','Price', 'Positive', 'Negative', 'Revenue']]

selected_features_sampled = selected_features.sample(n=1000, random_state=1)
# Plot a cluster map
plt.figure(figsize=(10, 8))
sns.clustermap(selected_features_sampled, cmap='viridis', figsize=(12, 10))
plt.title('Cluster Map of Game Features')
plt.show()

# %%
# Hexbin Plot

plt.figure(figsize=(10, 8))
plt.hexbin(df_cleaned['Peak CCU'], df_cleaned['Average In-game purchase'], gridsize=50, cmap='YlGnBu', edgecolors='none')
plt.colorbar(label='Count in Bin')
plt.xlabel('Peak CCU', **label_style)
plt.ylabel('Average In-game Purchase', **label_style)
plt.title('Hexbin Plot of Peak CCU versus Average In-game Purchase', **title_style)
plt.show()

# %%
# Strip Plot

plt.figure(figsize=(10, 8))
sns.stripplot(x='Playtime', y='Price', data=df_cleaned, jitter=True)
plt.xlabel('Playtime Category', **label_style)
plt.ylabel('Price', **label_style)
plt.title('Distribution of Price within each Playtime Category', **title_style)
plt.show()

# %%
# Swarm Plot

df_sampled = df_cleaned.sample(n=1000, random_state=1)

plt.figure(figsize=(10, 8))
sns.swarmplot(x='Playtime', y='Price', data=df_sampled)
plt.xlabel('Playtime Category', **label_style)
plt.ylabel('Price', **label_style)
plt.title('Distribution of Price within each Playtime Category', **title_style)
plt.show()


# %%
# 3-D Plot

# Create a 3D scatter plot
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df_cleaned['Average In-game purchase'], df_cleaned['Price'], df_cleaned['Revenue'], c='blue', marker='o')
ax1.set_xlabel('Average In-game Purchase', **label_style)
ax1.set_ylabel('Price', **label_style)
ax1.set_zlabel('Revenue', **label_style)
ax1.set_title('3D Scatter Plot of Price, Average In-game Purchase, and Revenue', **title_style)

# Create a contour plot
ax2 = fig.add_subplot(122, projection='3d')
# For contour plots, we typically need grid data. For demonstration, let's use a simple trisurf plot which doesn't require grid data.
ax2.plot_trisurf(df_cleaned['Average In-game purchase'], df_cleaned['Price'], df_cleaned['Revenue'], cmap='viridis', edgecolor='none')
ax2.set_xlabel('Average In-game Purchase', **label_style)
ax2.set_ylabel('Price', **label_style)
ax2.set_zlabel('Revenue', **label_style)
ax2.set_title('Contour Plot of Price, Average In-game Purchase, and Revenue', **title_style)

plt.tight_layout()
plt.show()
# %%
