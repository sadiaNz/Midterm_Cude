# Import required libraries
from sklearn.linear_model import LinearRegression
from cuml.linear_model import LinearRegression as cuLinearRegression
import numpy as np
import warnings

# Suppress warnings from cuML (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# Data
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).reshape(-1, 1)
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# ---------------- Using Scikit-learn ----------------
try:
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, Y)
    sklearn_intercept = sklearn_model.intercept_
    sklearn_slope = sklearn_model.coef_[0]
    print("\n---Linear Regression Using Scikit-learn---")
    print(f"Intercept: {sklearn_intercept}")
    print(f"Slope: {sklearn_slope}")
except Exception as e:
    print("\nScikit-learn part failed:")
    print(e)
    sklearn_intercept = None
    sklearn_slope = None

# ---------------- Using cuML ----------------
try:
    X_cuml = X.astype(np.float32)
    Y_cuml = Y.astype(np.float32)
    cuml_model = cuLinearRegression()
    cuml_model.fit(X_cuml, Y_cuml)
    cuml_intercept = cuml_model.intercept_
    cuml_slope = cuml_model.coef_[0]
    print("\n---Linear Regression Using cuML---")
    print(f"Intercept: {cuml_intercept}")
    print(f"Slope: {cuml_slope}")
except Exception as e:
    print("\ncuML part failed:")
    print(e)
    cuml_intercept = None
    cuml_slope = None

# ---------------- Comparison ----------------
if sklearn_intercept is not None and cuml_intercept is not None:
    print("\n---Comparison of Scikit-learn and cuML Outputs---")
    print(f"Scikit-learn Intercept: {sklearn_intercept}, Slope: {sklearn_slope}")
    print(f"cuML Intercept: {cuml_intercept}, Slope: {cuml_slope}")
else:
    print("\nUnable to compare results due to an error.")
