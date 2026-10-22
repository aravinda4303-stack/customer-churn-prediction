# 📊 Customer Churn Intelligence

An explainable machine learning application that predicts customer churn using **XGBoost**, **Streamlit**, and **SHAP**.

## Overview

Customer churn occurs when customers stop using a company's products or services. This project provides an interactive web application that estimates the probability of a telecommunications customer leaving a service.

Users can enter customer details, subscribed services, contract type, and billing information to receive a prediction and understand which features influenced the model's output.

## Features

- Customer churn prediction using a trained XGBoost classifier.
- Interactive web application built with Streamlit.
- Simplified 16-feature customer input form.
- Churn probability estimation.
- Individual prediction explanations using SHAP.
- Feature contribution visualization.
- Detailed explanation table for individual predictions.
- Monthly and yearly billing input options.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- SHAP

## Model Performance

Previously recorded evaluation results for the reduced 16-feature model:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 75.02% |
| Precision | 51.90% |
| Recall    | 80.21% |
| F1-score  | 63.03% |
| ROC-AUC   | 85.30% |

These metrics reflect the recorded evaluation on the test set. Performance may differ on new data.

## Explainable AI with SHAP

The application uses SHAP to explain how input features influence individual predictions.

- **Positive SHAP contributions:** Push the model output toward churn.
- **Negative SHAP contributions:** Push the model output away from churn.

These explanations describe model behavior and do not establish causation.

## Dataset

The model was trained using a **US telecommunications customer churn dataset**. The original training data uses US billing values and represents the customer population included in that dataset.

The model has not been validated for Indian telecommunications customers. Its predictions should therefore be treated as estimates for educational purposes, not as validated predictions for the Indian market.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── customer_churn_xgboost_reduced16.json
├── customer_churn_preprocessor_reduced16.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd customer-churn-prediction
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Limitations

- The model was trained on US telecommunications data.
- Predictions are estimates, not guarantees of future customer behavior.
- SHAP explanations describe model behavior and do not prove causation.
- Further validation is required before using the model for real business decisions.

## Author

**Aravind S**

B.Tech ECE – Data Science
SRM Institute of Science and Technology, Tiruchirappalli

## Disclaimer

This project was developed for educational and portfolio purposes. Predictions should not be used as the sole basis for customer retention or business decisions.
