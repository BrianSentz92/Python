import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
from sklearn.linear_model import LinearRegression

sns.set()

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
data = pd.read_csv('1.02.Multiple Linear Regression.csv')
print("First few rows:")
print(data.head())
print()

# ------------------------------------------------------------
# 2. Define features (X) and target (y)
# ------------------------------------------------------------
# Two input features: SAT, Rand 1,2,3
X = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print()

# ------------------------------------------------------------
# 3. Create and fit the regression model
# ------------------------------------------------------------
reg = LinearRegression()
reg.fit(X, y)

# ------------------------------------------------------------
# 4. Display model parameters
# ------------------------------------------------------------
print("Intercept:", reg.intercept_)
print("Coefficients (aligned with X columns):")
print("SAT coefficient:", reg.coef_[0])
print("Rand 1,2,3 coefficient:", reg.coef_[1])
print()

# ------------------------------------------------------------
# 5. Build a grid of SAT and Rand values for the regression plane
# ------------------------------------------------------------
sat_range = np.linspace(data['SAT'].min(), data['SAT'].max(), 10)
rand_range = np.linspace(data['Rand 1,2,3'].min(), data['Rand 1,2,3'].max(), 10)

SAT_grid, RAND_grid = np.meshgrid(sat_range, rand_range)

# Calculate predicted GPA (Z) for every point on the grid using the model
Z = reg.intercept_ + reg.coef_[0] * SAT_grid + reg.coef_[1] * RAND_grid

# ------------------------------------------------------------
# 6. Create a 3D scatter plot and regression plane
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual observed data points
ax.scatter(data['SAT'], data['Rand 1,2,3'], y, color='blue', label='Actual Data')

# Regression plane
ax.plot_surface(SAT_grid, RAND_grid, Z, alpha=0.4)

ax.set_xlabel('SAT Score')
ax.set_ylabel('Rand 1,2,3')
ax.set_zlabel('GPA')
ax.set_title('3D Multiple Linear Regression Plane')

plt.legend()
plt.show()

