# Telco Customer Churn Prediction

A machine learning project that predicts customer churn in the telecommunications industry using Decision Tree and Artificial Neural Network classifiers. Built as part of CM2604 – Machine Learning at IIT in collaboration with Robert Gordon University.

## Overview

Customer churn is one of the most costly problems in telecoms. This project takes the Telco Customer Churn dataset, cleans and engineers it, trains two models, tunes them, and compares their performance to identify the most reliable approach for churn prediction.

## Dataset

The Telco Customer Churn dataset contains 7043 rows and 21 attributes covering customer demographics, subscribed services, billing information, account tenure, and churn status. The target variable indicates whether a customer discontinued the service.

Class distribution: 73.46% no churn, 26.54% churn. SMOTE and class weighting were used to handle this imbalance during training.

One notable data issue was the TotalCharges column, which contained blank strings instead of NaN for new customers with zero tenure. These were identified by converting the column to numeric and imputed as TotalCharges = MonthlyCharges to preserve billing logic.

## What was done

**Data Preparation**
Handled hidden missing values, analyzed class imbalance, and explored distributions across numerical and categorical features.

**EDA**
Month-to-month contract customers and those with fiber optic internet showed significantly higher churn rates. Tenure was strongly negatively correlated with churn, and higher monthly charges correlated positively.

**Feature Engineering**
Seven new features were created to improve model performance including AvgChargesPerMonth, TenureGroup, ContractValue, ChargesToTenureRatio, HasPremiumSupport, IsStreamingCustomer, and TotalServices.

**Preprocessing**
LabelEncoder for categorical variables, StandardScaler for numerical features.

**Models**
Two classifiers were implemented and tuned using GridSearchCV and manual experimentation.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree (Tuned) | 0.735 | 0.500 | 0.791 | 0.613 | 0.819 |
| Neural Network (Tuned) | 0.614 | 0.402 | 0.930 | 0.561 | 0.836 |

The Neural Network achieved a higher AUC (0.836) and significantly better recall, making it the better choice when minimising missed churners is the priority. The Decision Tree offered better overall accuracy and a cleaner F1 score.

## Tech stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)
- Jupyter Notebook

## Author

Sandali Balasooriya - AI & Data Science undergraduate at IIT  
[LinkedIn](https://www.linkedin.com/in/sandalibalasooriya) 