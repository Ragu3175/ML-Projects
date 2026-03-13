# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Step 2: Load dataset
data = pd.read_csv("student_scores.csv")

# dataset example:
# Hours,Score
# 2.5,21
# 5.1,47
# 3.2,27
# ...


# Step 3: Data cleaning
data = data.dropna()   # remove missing values


# Step 4: Select features and target
X = data[['Hours']]    # feature
y = data['Score']      # target


# Step 5: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 7: Make predictions
predictions = model.predict(X_test)


# Step 8: Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# Step 9: Print learned equation
m = model.coef_[0]
b = model.intercept_

print(f"Learned Equation: Score = {m:.2f} * Hours + {b:.2f}")


# Step 10: Visualization
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction")

plt.show()