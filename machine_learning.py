# ✅ Updated Script for Your Dataset (with 'area_sqft' and 'price_in_thousands')
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("sample_properties.csv")

# Preview data
print("\n[INFO] Dataset Preview:")
print(df.head())

# Drop missing values
df = df.dropna()

# Visualize correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Scatter plot of area vs price
plt.figure(figsize=(6, 4))
sns.scatterplot(x='area_sqft', y='price_in_thousands', data=df)
plt.title("Area vs Price")
plt.savefig("scatter_area_price.png")
plt.close()

# Select features and target
X = df[['bedrooms', 'bathrooms', 'area_sqft']]
y = df['price_in_thousands']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n[INFO] Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Save model
joblib.dump(model, "linear_model.pkl")
print("\n[INFO] Model saved as 'linear_model.pkl'")
