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
plt.xlabel('Year')
plt.ylabel('Peak CCU')
plt.title('Peak CCU for Selected Genres Over Time')
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
