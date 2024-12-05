from sklearn.linear_model import LinearRegression
import numpy as np

# Define the X and Y arrays
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).reshape(-1, 1)
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, Y)

# Output the intercept and slope
print("Using Scikit-Learn:")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")
