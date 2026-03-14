# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# Step 2: Load dataset
data = pd.read_csv("student_scores.csv")


# Step 3: Data cleaning
data = data.dropna()


# Step 4: Select features and target
X = data[['Hours','Sleep','Practice']]
y = data['Score']


# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 7: Polynomial Features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# Step 8: Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)


# Step 9: Make predictions
predictions = model.predict(X_test_poly)


# Step 10: Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# Step 11: Visualization (Hours vs Score only)
# Step 11: Visualization (Polynomial curve)

# scatter original data
plt.scatter(data['Hours'], y, color="blue", label="Actual Data")

# create smooth hours range
hours_range = np.linspace(data['Hours'].min(), data['Hours'].max(), 100)

# create dummy values for other features
sleep_mean = data['Sleep'].mean()
practice_mean = data['Practice'].mean()

# build dataframe
curve_df = pd.DataFrame({
    "Hours": hours_range,
    "Sleep": sleep_mean,
    "Practice": practice_mean
})

# scale
curve_scaled = scaler.transform(curve_df)

# polynomial transform
curve_poly = poly.transform(curve_scaled)

# predict
curve_predictions = model.predict(curve_poly)

# plot curve
plt.plot(hours_range, curve_predictions, color="red", label="Polynomial Regression")

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Polynomial Regression Fit")

plt.legend()
plt.show()