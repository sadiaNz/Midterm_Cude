from sklearn.linear_model import LinearRegression
import numpy as np

from cuml.linear_model import LinearRegression as cuLinearRegression


# Define the X and Y arrays
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).reshape(-1, 1)
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# ==============================================================================
# Linear Regression using scikit-learn
# ==============================================================================

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, Y)

# Output the results
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

print("\n")  # Space between methods

# ==============================================================================
# Linear Regression using cuML (Optional Bonus)
# ==============================================================================

# Convert data to float32 for cuML compatibility
X_cuml = X.astype(np.float32)
Y_cuml = Y.astype(np.float32)

# Fit linear regression using cuML
cuml_model = cuLinearRegression()
cuml_model.fit(X_cuml, Y_cuml)

# Get intercept and slope
cuml_intercept = cuml_model.intercept_
cuml_slope = cuml_model.coef_[0]

print("\n---Linear Regression Using cuML---")
print(f"Intercept: {cuml_intercept}")
print(f"Slope: {cuml_slope}")
