import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("sample_properties.csv")

# Drop null values
df.dropna(inplace=True)

# Show dataset summary
print(df.info())
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Example scatter plot (size vs price)
sns.scatterplot(x='size', y='price_in_thousands', data=df)
plt.title("Size vs Price")
plt.show()

