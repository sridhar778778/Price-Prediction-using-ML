import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

# Upload dataset (for Google Colab users)
uploaded = files.upload()

# Load the dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Preprocessing: Extract BHK from 'size' column
df['bhk'] = df['size'].str.extract('(\d+)').astype(float)

# Handle missing values
df = df.dropna(subset=['location', 'total_sqft', 'bath', 'price', 'bhk'])

# Convert 'location' to numerical values
df['location_id'] = df['location'].factorize()[0]

# Convert 'total_sqft' to numerical (handling ranges like "1200-1500")
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            vals = list(map(float, x.split('-')))
            return sum(vals) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])

# Define features (X) and target (y)
X = df[['location_id', 'bhk', 'total_sqft', 'bath', 'balcony']].fillna(0)
y = df['price']

# Split dataset into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
lr_y_pred = lr_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)

# Compute R-squared and RMSE
lr_r2 = r2_score(y_test, lr_y_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))

rf_r2 = r2_score(y_test, rf_y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))

# Print model performance
print(f"Linear Regression R-squared: {lr_r2:.2f}")
print(f"Linear Regression RMSE: {lr_rmse:.2f}")

print(f"Random Forest R-squared: {rf_r2:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")

# User Input for Prediction
location_mapping = dict(zip(df['location'], df['location_id']))
location_name = input("Enter location name: ")

if location_name in location_mapping:
    try:
        location_id = location_mapping[location_name]
        bhk = float(input("Enter number of BHK: "))
        sqft = float(input("Enter total square feet: "))
        bath = float(input("Enter number of bathrooms: "))
        balcony = float(input("Enter number of balconies: "))

        # Predict using both models
        lr_pred = lr_model.predict([[location_id, bhk, sqft, bath, balcony]])
        rf_pred = rf_model.predict([[location_id, bhk, sqft, bath, balcony]])

        print(f"\n Predicted price using Linear Regression:** {lr_pred[0]:.2f} Lakhs")
        print(f"Predicted price using Random Forest:** {rf_pred[0]:.2f} Lakhs")

    except ValueError:
        print(" Invalid input. Please enter numerical values for BHK, square feet, bathrooms, and balconies.")
else:
    print(" Location not found.")

# Data Visualization

# Distribution of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=30, kde=True, color='green')
plt.title("House Price Distribution in Bengaluru", fontsize=14)
plt.xlabel("Price (in Lakhs)", fontsize=12)
plt.ylabel("Number of Houses", fontsize=12)
plt.show()

#  Scatter Plot: Total Sqft vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['total_sqft'], y=df['price'], hue=df['bhk'], palette='coolwarm', alpha=0.7)
plt.title("House Price vs Total Area", fontsize=14)
plt.xlabel("Total Area (in Sqft)", fontsize=12)
plt.ylabel("Price (in Lakhs)", fontsize=12)
plt.legend(title="Number of BHKs", fontsize=10, title_fontsize=12)
plt.show()

# Top 10 Locations by Number of Listings
top_locations = df['location'].value_counts().head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='muted')
plt.title("Top 10 Most Common Property Locations", fontsize=14)
plt.xlabel("Number of Listings", fontsize=12)
plt.ylabel("Location Names", fontsize=12)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation metrics
models = ['Linear Regression', 'Random Forest']
r2_scores = [lr_r2, rf_r2]

# Bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=r2_scores, palette='Blues_d')
plt.title('R-squared Comparison')
plt.ylabel('R-squared Score')
plt.ylim(0, 1)
plt.show()

# Feature importances from Random Forest
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Residuals
lr_residuals = y_test - lr_y_pred
rf_residuals = y_test - rf_y_pred

plt.figure(figsize=(12, 5))

# Linear Regression Residuals
plt.subplot(1, 2, 1)
sns.histplot(lr_residuals, kde=True, color='blue')
plt.title("Linear Regression Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

# Random Forest Residuals
plt.subplot(1, 2, 2)
sns.histplot(rf_residuals, kde=True, color='green')
plt.title("Random Forest Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

sample_range = 50  # you can change this for more data points

plt.figure(figsize=(12, 5))
plt.plot(range(sample_range), y_test.values[:sample_range], label='Actual', color='black', marker='o')
plt.plot(range(sample_range), lr_y_pred[:sample_range], label='Linear Regression', color='blue', linestyle='--', marker='x')
plt.plot(range(sample_range), rf_y_pred[:sample_range], label='Random Forest', color='green', linestyle='-.', marker='s')

plt.title("Actual vs Predicted Prices (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Price (in Lakhs)")
plt.legend()
plt.show()
