# Diabetes Prediction Using Machine Learning

A beginner-friendly machine learning project that predicts whether a person is diabetic based on basic health measurements, using **Logistic Regression** in scikit-learn.

## Dataset
The notebook expects a file named `diabetes.csv` (the Kaggle "Diabetes Prediction Dataset"). It is not included in this repo, so download it and place it in the project folder.

Features used: `age`, `hypertension`, `heart_disease`, `bmi`, `HbA1c_level`, `blood_glucose_level`
Target: `diabetes` (0 = non-diabetic, 1 = diabetic)

## What the notebook does
1. Loads and explores the data (shape, missing values, statistics)
2. Drops the `gender` and `smoking_history` columns
3. Balances the classes by under-sampling the non-diabetic group
4. Splits the data into train/test sets (80/20, stratified)
5. Standardizes features with `StandardScaler`, fitted on the training set only to avoid data leakage
6. Trains a Logistic Regression model
7. Evaluates it with accuracy, a classification report, and a confusion matrix
8. Builds a simple predictive system and saves the trained model (`trained_model.sav`) and the scaler (`scaler.sav`) with `pickle`

## How to run
```bash
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
jupyter notebook Diabetes_Prediction_Using_Machine_Learning.ipynb
```

## Tech stack
Python, NumPy, pandas, scikit-learn, seaborn, matplotlib

## Disclaimer
This is a learning project and is **not** a medical tool. Do not use it for real diagnosis.
