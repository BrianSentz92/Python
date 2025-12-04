README: Multiple Linear Regression with Scikit-Learn
Predicting GPA Using SAT and an Additional Feature (3D Visualization)

Overview
This project demonstrates a multiple linear regression model using Python and scikit-learn, with a 3D regression plane visualization. The model predicts a student’s GPA based on two features:
1. SAT score
2. Rand 1,2,3 (a randomly assigned value acting as a noise feature)

This project extends basic linear regression concepts and introduces multi-feature modeling commonly seen in real machine learning workflows.

Key Concepts Demonstrated
1. Multiple Linear Regression
Uses two input features to predict a single output. The model learns one coefficient per feature and an intercept.

2. Supervised Learning Structure
Features (X): SAT, Rand 1,2,3
Target (y): GPA
The model is trained using historical samples and learns the best-fitting regression plane.

3. 3D Visualization
The project creates a 3D scatterplot of the dataset and overlays a regression plane learned by the model. This visually demonstrates how multi-feature linear models work.

4. Feature Relevance
SAT should be predictive of GPA.
Rand 1,2,3 is intentionally random, demonstrating how irrelevant variables behave in a regression model.

Project Files
MultipleLinearRegression3D.py
1.02. Multiple linear regression.csv
requirements.txt
README.txt

Requirements
Install dependencies with:
pip install numpy pandas matplotlib seaborn scikit-learn

Or install from requirements.txt:
pip install -r requirements.txt

How to Run
1. Place the CSV in the same directory as the script.
2. Install dependencies.
3. Run the script:
python3 MultipleLinearRegression3D.py

Expected Results
- Prints model coefficients.
- Displays a 3D scatterplot with the fitted regression plane.
- Shows how multiple linear regression uses more than one input to predict an output.

Why This Project Matters
This project demonstrates:
- Multi-feature modeling
- scikit-learn integration
- Data visualization skills
- Practical ML workflow used in real projects
- A foundation for feature engineering and model evaluation

Future Improvements
- Add train/test split
- Add model evaluation metrics
- Visualize residuals
- Compare results with statsmodels
- Add predictions for new samples

Contact
Project created by Brian Sentz.
