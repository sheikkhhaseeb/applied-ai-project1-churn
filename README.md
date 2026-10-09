# ML-Powered Customer Analytics

**Student:** Muhammad Haseeb Bilal
**Project:** Project 1 – ML-Powered Customer Analytics
**Dataset:** Telco Customer Churn

## Project Overview

This project focuses on analyzing telecom customer data, building machine learning models to predict customer churn, optimizing model performance, identifying customer segments, and deploying the final model through an interactive web application.

The project is completed across four weeks:

* **Week 1:** Exploratory Data Analysis
* **Week 2:** Machine Learning Models
* **Week 3:** Model Optimization and Unsupervised Learning
* **Week 4:** Final Application and Deployment

The main objective is to understand customer churn patterns, develop a reliable prediction model, discover meaningful customer segments, and make churn predictions accessible through a web application.

---

# Week 1 — Exploratory Data Analysis

## Objective

The goal of Week 1 was to understand the Telco Customer Churn dataset, examine data quality, and identify patterns associated with customer churn.

## Work Completed

* Loaded and inspected the dataset.
* Examined data types and missing values.
* Converted `TotalCharges` to numeric.
* Analyzed the distribution of churn.
* Studied customer tenure and monthly charges.
* Compared churn across contract types.
* Analyzed internet service types and payment methods.
* Investigated relationships between customer characteristics and churn.

## Key Findings

### Contract Type

* Month-to-month: **42.71% churn**
* One year: **11.27% churn**
* Two year: **2.83% churn**

Customers with month-to-month contracts had a substantially higher churn rate than customers with longer-term contracts.

### Customer Tenure

* Customers with less than 6 months of tenure: approximately **54.27% churn**
* Customers within their first 12 months: approximately **47.44% churn**

Newer customers were more likely to churn.

### Internet Service

* Fiber optic: **41.89% churn**
* DSL: **18.96% churn**
* No internet service: **7.40% churn**

### Payment Method

Customers paying through electronic checks had approximately **45.29% churn**, while automatic payment methods had lower churn rates.

## Week 1 Learning Outcome

Exploratory Data Analysis identified important relationships between churn and contract type, tenure, internet service, and payment method. These findings provided a foundation for the machine learning models developed in the following weeks.

---

# Week 2 — Machine Learning Models

## Objective

The goal of Week 2 was to prepare the dataset, build baseline machine learning models, and evaluate their predictive performance.

## Data Preparation

* Removed the `customerID` column.
* Converted categorical variables using one-hot encoding.
* Converted `Churn` into a binary target variable.
* Split the data into training and test sets.
* Applied `StandardScaler` with Logistic Regression.

## Models Explored

* Logistic Regression
* Random Forest
* XGBoost

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* ROC AUC
* Recall
* Precision
* F1-score

## Week 2 Learning Outcome

Different evaluation metrics provide different insights into model performance. Accuracy alone may not be sufficient for customer churn prediction. ROC AUC measures how effectively a model distinguishes between classes, while recall measures the proportion of actual churners correctly identified.

---

# Week 3 — Model Optimization and Unsupervised Learning

## Objective

Week 3 focused on evaluating model stability, comparing models through cross-validation, tuning hyperparameters, performing customer segmentation, applying PCA, and selecting a final model.

## 3.1 Split-to-Split Variability

The same Logistic Regression model was evaluated using 20 different random splits.

| Metric                   |                 Result |
| ------------------------ | ---------------------: |
| Minimum accuracy         |                  78.0% |
| Maximum accuracy         |                  82.8% |
| Mean accuracy            |                 80.02% |
| Standard deviation       |                 0.0104 |
| Approximate 95% interval | ±2.1 percentage points |

**Finding:** Model performance varied across random splits. This demonstrated why evaluating a model using only one train-test split may provide an incomplete picture of its performance.

## 3.2 Five-Fold Cross-Validation

| Model               |       ROC AUC |        Recall |      F1-score |
| ------------------- | ------------: | ------------: | ------------: |
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 ± 0.030 |
| Random Forest       | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |

Cross-validation provided a more reliable comparison by evaluating each model across multiple folds.

## 3.3 Hyperparameter Tuning

### Random Forest — Grid Search

* Parameter combinations: **24**
* Five-fold CV fits: **120**
* Best CV AUC: **0.8468**
* Execution time: **101 seconds**

Best parameters:

* `max_depth = 8`
* `max_features = sqrt`
* `min_samples_leaf = 20`

### Random Forest — Random Search

* Search iterations: **24**
* Five-fold CV fits: **120**
* Best CV AUC: **0.8464**
* Execution time: **109 seconds**

Best parameters:

* `max_depth = 15`
* `max_features ≈ 0.213`
* `min_samples_leaf = 15`

Grid Search achieved a slightly higher CV AUC than Random Search in this experiment.

## 3.4 XGBoost Optimization

XGBoost was optimized using randomized hyperparameter search.

The tuned model achieved:

* **CV AUC: 0.8502**
* **CV AUC standard deviation: 0.0117**

### Final Cross-Validation Comparison

| Model               |     CV AUC | CV Standard Deviation |
| ------------------- | ---------: | --------------------: |
| Logistic Regression |     0.8464 |                0.0129 |
| Random Forest       |     0.8464 |                0.0114 |
| **XGBoost**         | **0.8502** |            **0.0117** |

XGBoost was selected as the final model because it achieved the highest cross-validation AUC among the listed candidate models.

## 3.5 Customer Segmentation Using K-Means

K-Means clustering was applied to identify customer groups with similar characteristics.

The clustering analysis used:

* Tenure
* Monthly Charges
* Total Charges
* Number of Services

Features were standardized before clustering, and four customer clusters were selected.

### Customer Segments

| Customer Segment            | Customers | Average Tenure | Average Monthly Charge | Average Services | Churn |
| --------------------------- | --------: | -------------: | ---------------------: | ---------------: | ----: |
| High-Risk New High-Spenders |     2,157 |          18.38 |                  80.41 |             3.28 |   43% |
| New Low-Service Customers   |     1,918 |           8.96 |                  37.71 |             1.20 |   32% |
| Loyal High-Value Customers  |     1,938 |          59.83 |                  92.09 |             5.06 |   14% |
| Loyal Low-Cost Customers    |     1,030 |          53.61 |                  30.96 |             1.48 |    5% |

### Suggested Business Actions

**High-Risk New High-Spenders**

* Offer targeted discounts and suitable contract options.
* Provide proactive customer support.

**New Low-Service Customers**

* Improve customer onboarding.
* Recommend relevant service bundles.

**Loyal High-Value Customers**

* Offer loyalty rewards and premium support.

**Loyal Low-Cost Customers**

* Maintain reliable service.
* Offer optional upgrades when appropriate.

These segments can help businesses develop more targeted customer retention strategies.

## 3.6 Principal Component Analysis (PCA)

PCA was applied to study feature redundancy and dimensionality reduction.

### Results

* Original feature count: **30**
* Components explaining 90% of the variance: **15**
* The strongest PC1 loadings were mainly associated with customers having no internet service.
* The two-dimensional PCA visualization showed substantial overlap between churned and non-churned customers.

**Finding:** PCA reduced the dimensionality of the dataset while retaining most of its variance. However, the first two principal components did not clearly separate churned and non-churned customers.

## 3.7 Final Model Evaluation

The final model was selected using cross-validation and evaluated on the held-out test set.

**Selected Model:** Tuned XGBoost

| Metric    | Test Result |
| --------- | ----------: |
| ROC AUC   |  **0.8483** |
| Recall    |   **0.521** |
| Precision |   **0.659** |

The test ROC AUC of 0.8483 was close to the cross-validation AUC of 0.8502, indicating consistent discrimination performance on unseen data.

The trained model was saved as `churn_model.joblib` for use in the Week 4 application.

## Week 3 Learning Outcome

Week 3 demonstrated the importance of cross-validation, hyperparameter optimization, model evaluation, unsupervised learning, and dimensionality reduction. The optimized model and its saved artifacts were then used for the application stage.

---

# Week 4 — Final Application and Deployment

## Objective

The goal of Week 4 was to make the trained customer churn model accessible through an interactive Streamlit web application.

The application uses the saved model and metadata produced during the model development stage.

## 4.1 Application Development

The application was developed using Streamlit and Python. It loads the trained model and uses customer information to generate churn predictions.

### Main Features

**1. Single Customer Prediction**

* Enter customer details through an interactive interface.
* Generate a churn prediction for an individual customer.
* View the predicted churn probability.

**2. Churn Risk Assessment**

* Display the model's estimated churn probability.
* Categorize the customer into a risk band using the application's configured threshold.

**3. What-If Analysis**

* Change selected customer attributes.
* Recalculate the predicted churn risk.
* Examine how changes in customer information affect the model's prediction.

**4. Batch Customer Scoring**

* Upload a CSV file containing multiple customer records.
* Generate predictions for multiple customers.
* Review the resulting customer-level predictions.

**5. Model Information**

* Display available model evaluation information.
* Use saved metadata for the configured feature columns and prediction threshold.

## 4.2 Model Artifacts

The application uses the following files:

| File                   | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `churn_model.joblib`   | Saved trained machine learning model                                   |
| `model_meta.json`      | Model metadata, feature columns, evaluation information, and threshold |
| `sample_customers.csv` | Sample customer records for testing batch predictions                  |
| `app.py`               | Streamlit application                                                  |
| `requirements.txt`     | Python package dependencies                                            |

## 4.3 Deployment

The application is designed to run using Streamlit and can be deployed as a web application.

**Live Streamlit Application:** https://haseebsheikh.streamlit.app/

**GitHub Repository:** https://github.com/sheikkhhaseeb/applied-ai-project1-churn

## 4.4 Week 4 Learning Outcomes

* Learned how to connect a saved machine learning model to a web application.
* Developed an interactive interface for customer churn prediction.
* Used model probabilities to assess predicted churn risk.
* Explored what-if analysis for changes in customer attributes.
* Prepared batch scoring functionality for CSV data.
* Organized application files, dependencies, and documentation for deployment.

## Week 4 Deliverable

The Week 4 deliverable is an interactive customer churn prediction application supported by the saved model artifacts and project documentation.

---

# Final Project Lessons

1. Exploratory Data Analysis helps identify important patterns before model development.
2. A single random train-test split may not reliably represent model performance.
3. Cross-validation supports more reliable model comparison.
4. Hyperparameter tuning can improve model selection.
5. ROC AUC, recall, precision, and F1-score measure different aspects of classification performance.
6. K-Means clustering can identify customer segments without using the target variable during clustering.
7. PCA can reduce dimensionality while preserving a large proportion of the variance.
8. Final model evaluation should use a held-out test set.
9. Saved model artifacts make it possible to use a trained model outside the original notebook.
10. A web application provides an accessible interface for machine learning predictions.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Joblib
* Jupyter Notebook
* Kaggle
* Streamlit

---

# Repository Structure

```text
applied-ai-project1-churn/
│
├── week1-eda.ipynb
├── week2-ml-models.ipynb
├── week3-optimization.ipynb
│
├── app.py
├── requirements.txt
├── churn_model.joblib
├── model_meta.json
├── sample_customers.csv
│
└── README.md
```

The repository contains the notebooks for the project workflow, the saved model artifacts, and the Streamlit application files. Keep one correctly named notebook per week and remove duplicate notebook copies.

---

# Final Model Summary

* **Selected Model:** Tuned XGBoost
* **Cross-Validation ROC AUC:** 0.8502
* **Test ROC AUC:** 0.8483
* **Test Recall:** 0.521
* **Test Precision:** 0.659

The project covers the complete workflow from exploratory analysis and model development to optimization, customer segmentation, and an interactive prediction application.
