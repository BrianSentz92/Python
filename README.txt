README: Simple Linear Regression with Scikit-Learn
Predicting Student GPA from SAT Scores

Overview
This project demonstrates how to build and interpret a simple linear regression model using Python and scikit-learn. It is designed as an introductory step into machine learning concepts such as features, targets, supervised learning, model training, and prediction.

The goal is to predict college GPA based on SAT scores, using a clean machine learning workflow.

This project was developed in PyCharm using a standalone virtual environment.

Key Concepts Demonstrated
1. Supervised Learning
The model is trained using historical labeled data:
- Feature (X): SAT score
- Target (y): College GPA

2. Machine Learning Workflow (scikit-learn)
The project follows the standard ML pipeline:
- Load and explore data
- Define features and target
- Reshape data for scikit-learn
- Create the LinearRegression model
- Fit the model on the data
- Extract slope and intercept
- Visualize results

3. Feature Reshaping (ML Requirement)
scikit-learn requires input features to be a 2D array.
The SAT score column is reshaped from a 1D vector into a 2D matrix using:
X_matrix = X.values.reshape(-1, 1)

This is a common practical step in ML pipelines.

Project Files
ScLearnLinearRegression.py      # Main Python script
1.02. Multiple linear regression.csv   # Dataset used (if uploaded)
README.txt                       # Project documentation

Requirements
pip install numpy pandas matplotlib seaborn scikit-learn

Or inside a virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Code Summary
Load Data
Reads SAT and GPA values from CSV into a DataFrame.

Prepare Features and Target
Separates the input variable (SAT) and output variable (GPA).

Reshape Data
Converts X into a 2D array so scikit-learn can accept it.

Train the Model
reg.fit(X_matrix, y)

Extract Results
print("Intercept:", reg.intercept_)
print("Coefficient:", reg.coef_)

Visualization
Scatter plot + regression line.

Results and Interpretation
The model learns:
- Intercept: baseline GPA
- Coefficient: GPA increase per SAT point

Example Interpretation:
For every additional SAT point, GPA increases by approximately 0.0017.

Why This Project Matters
This project demonstrates:
- Understanding of ML workflows
- Ability to use scikit-learn
- Competence with Python, pandas, visualization
- Handling preprocessing and model fitting
- Debugging real-world environment issues

Future Improvements
- Add multiple regression
- Compare scikit-learn vs statsmodels
- Add train/test splits
- Add predictions for new SAT scores
- Wrap code in reusable functions

Contact
Project created by Brian Sentz
