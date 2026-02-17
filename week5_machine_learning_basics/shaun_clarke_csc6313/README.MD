In this project, you will develop a supervised machine learning pipeline to predict patient health outcomes using Scikit-Learn.
You will transition from "Classical ML" theory to a functional Python implementation that handles data generation, cleaning, model training, and real-time inference.

Core Requirements
1. Data Generation & Persistence
Automated Dataset: Your script must programmatically generate a synthetic dataset and save it as patient_health_data.csv in your Current Working Directory (CWD).

Schema: The CSV must contain at least 500 rows with the features: age, bmi, blood_sugar_level, and a target health_risk_score.

2. Preprocessing & Clean Output
Data Cleaning: Implement Median Imputation for missing values and Standardization (Z-score scaling).

Zero Warning Policy: Your code must be robust enough to handle new data without triggering UserWarnings (e.g., "X does not have valid feature names").

3. The Bias-Variance Challenge
Model Comparison: You must train and evaluate three distinct regression models:

Underfit (High Bias): A model using only one input feature.

Overfit (High Variance): A model using PolynomialFeatures with a degree of 10 or higher.

Optimal: A Multiple Linear Regression model using all three scaled features.

Evaluation: Display the Mean Squared Error (MSE) and R-squared (R^2) for all three models in the terminal.

4. Interactive Terminal Inference
Your program must be designed to run as an interactive tool in the terminal:

Prompts: The script must ask the user to input an Age, BMI, and Blood Sugar Level.

Real-time Prediction: After input, the script must process the data through the saved scaler and output:

Predicted Health Risk Score (0-100 scale).

Probability of Risk (Percentage via Logistic Regression).

Final Diagnosis (Either "HEALTHY" or "AT RISK" based on a 60-point threshold).