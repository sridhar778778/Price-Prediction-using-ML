import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from google.colab import files
uploaded = files.upload()

# Load the dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Preprocess the data
# Extract numerical values for BHK from 'size' ('2 BHK' -> 2)
df['bhk'] = df['size'].str.extract('(\d+)').astype(float)

# Handle missing data
df = df.dropna(subset=['location', 'total_sqft', 'bath', 'price', 'bhk'])

# Factorize location to numerical ID
df['location_id'] = df['location'].factorize()[0]

# Convert total_sqft to a numeric average ('1200-1500' -> 1350)
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

# Features and target
X = df[['location_id', 'bhk', 'total_sqft', 'bath', 'balcony']].fillna(0)
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

accuracy = lr_model.score(X_test, y_test)
print(f"accuracy: {accuracy}")

# Evaluate the model
y_pred = lr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R-squared: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# User prediction
location_mapping = dict(zip(df['location'], df['location_id']))
location_name = input("Enter location name: ")
if location_name in location_mapping:
    try:
        location_id = location_mapping[location_name]
        bhk = float(input("Enter number of BHK: "))
        sqft = float(input("Enter total square feet: "))
        bath = float(input("Enter number of bathrooms: "))
        balcony = float(input("Enter number of balconies: "))
        predicted_price = lr_model.predict([[location_id, bhk, sqft, bath, balcony]])
        print(f"Predicted price: {predicted_price[0]:.2f} Lakhs")
    except ValueError:
        print("Invalid input. Please enter numerical values for BHK, square feet, bathrooms, and balconies.")
else:
    print("Location not found.")

# 1. Distribution of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=30, kde=True, color='green')
plt.title("House Price Distribution in Bengaluru", fontsize=14)
plt.xlabel("Price (in Lakhs)", fontsize=12)
plt.ylabel("Number of Houses", fontsize=12)
plt.show()
# 2. Scatter Plot: Total Sqft vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['total_sqft'], y=df['price'], hue=df['bhk'], palette='coolwarm', alpha=0.7)
plt.title("House Price vs Total Area", fontsize=14)
plt.xlabel("Total Area (in Sqft)", fontsize=12)
plt.ylabel("Price (in Lakhs)", fontsize=12)
plt.legend(title="Number of BHKs", fontsize=10, title_fontsize=12)
plt.show()
# 4. Top 10 Locations by Count
top_locations = df['location'].value_counts().head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='muted')
plt.title("Top 10 Most Common Property Locations", fontsize=14)
plt.xlabel("Number of Listings", fontsize=12)
plt.ylabel("Location Names", fontsize=12)
plt.show()
