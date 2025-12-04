import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Set seaborn style
sns.set()

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
# Make sure the CSV file is in your working directory OR provide full path
data = pd.read_csv('1.01.Simple Linear Regression.csv')

print("First five rows of the dataset:")
print(data.head())
print()

# ------------------------------------------------------------
# 2. Define feature (X) and target (y)
# ------------------------------------------------------------
X = data['SAT']
y = data['GPA']

# ------------------------------------------------------------
# 3. Reshape X into a 2D array for scikit-learn
# ------------------------------------------------------------
# scikit-learn requires a 2D matrix for features
X_matrix = X.values.reshape(-1, 1)

print("Shape of X before reshape:", X.shape)
print("Shape of X_matrix after reshape:", X_matrix.shape)
print()

# ------------------------------------------------------------
# 4. Create the Linear Regression model
# ------------------------------------------------------------
reg = LinearRegression()

# ------------------------------------------------------------
# 5. Fit the model
# ------------------------------------------------------------
reg.fit(X_matrix, y)

# ------------------------------------------------------------
# 6. Print the learned parameters
# ------------------------------------------------------------
print("Model Intercept:", reg.intercept_)
print("Model Coefficient:", reg.coef_)
print()

# ------------------------------------------------------------
# 7. OPTIONAL: Visualize the regression line
# ------------------------------------------------------------
plt.scatter(X, y, color='blue', label='Data Points')

# Predict y values for plotting line
y_pred = reg.predict(X_matrix)

plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.title('Simple Linear Regression (scikit-learn)')
plt.legend()
plt.show()
