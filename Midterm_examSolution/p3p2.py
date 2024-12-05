from sklearn.linear_model import LinearRegression
import numpy as np

from cuml.linear_model import LinearRegression as cuLinearRegression
import cupy as cp

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
print("Intercept: {}".format(model.intercept_))
print("Slope: {}".format(model.coef_[0]))

# Convert the arrays to GPU-supported data types
X_gpu = cp.array(X)
Y_gpu = cp.array(Y)

# Create and fit the linear regression model using cuML
gpu_model = cuLinearRegression()
gpu_model.fit(X_gpu, Y_gpu)

# Output the intercept and slope
print("\nUsing cuML:")
print("Intercept: {}".format(gpu_model.intercept_))
print("Slope: {}".format(gpu_model.coef_[0]))
