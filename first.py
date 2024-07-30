# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris = pd.read_csv('IRIS.csv')

# Inspect the data
# Display the first few rows
print(iris.head())

# Display basic information about the dataset
print(iris.info())

# Display summary statistics
print(iris.describe())

# Visualize Key Statistics and Distributions
# Pairplot
# A pairplot is useful to visualize the relationship between different features:
sns.pairplot(iris, hue='species')
plt.show()

# Histograms
# Visualize the distribution of each feature:
iris.hist(edgecolor='black', linewidth=1.2)
plt.show()

# Boxplots
# Boxplots are helpful to understand the distribution and outliers of each feature:
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris)
plt.show()

# Correlation Heatmap
# A correlation heatmap shows the relationship between different numerical features:
plt.figure(figsize=(8, 6))
sns.heatmap(iris.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.show()

# Violin Plots
# Violin plots are useful to visualize the distribution and probability density of the data at different values:
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal_length', data=iris)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal_width', data=iris)
plt.show()

# Summary and Insights
# Summarize your findings from the visualizations and basic statistics:
# Distribution and Range: Note the range and distribution of each feature (sepal length, sepal width, petal length, petal width) for different species.
# Relationships: Identify any notable relationships between features, such as correlations or patterns that distinguish different species.
# Outliers: Identify any outliers or unusual data points in the dataset.
