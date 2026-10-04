# ML-Powered Customer Analytics

**Student:** Muhammad Haseeb Bilal
**Project:** Project 1 – ML-Powered Customer Analytics
**Dataset:** Telco Customer Churn

---

## Project Overview

This project focuses on analyzing telecom customer data and building machine learning models to predict customer churn.

The project is completed across four weeks:

* **Week 1:** Exploratory Data Analysis
* **Week 2:** Machine Learning Models
* **Week 3:** Model Optimization and Unsupervised Learning
* **Week 4:** Final Deployment / Application

The goal is to understand customer churn, build reliable prediction models, identify important customer groups, and prepare the final model for practical use.

---

# Week 1 — Exploratory Data Analysis

## Objective

The goal of Week 1 was to understand the Telco Customer Churn dataset and identify patterns related to customer churn.

### Work Completed

* Loaded and inspected the dataset.
* Checked data types and missing values.
* Converted `TotalCharges` to numeric.
* Performed exploratory data analysis.
* Analyzed churn distribution.
* Studied customer tenure.
* Compared churn across contract types.
* Analyzed internet service types.
* Analyzed payment methods.
* Investigated relationships between customer characteristics and churn.

### Key Findings

**Contract Type**

* Month-to-month: **42.71% churn**
* One year: **11.27% churn**
* Two year: **2.83% churn**

Customers with month-to-month contracts were much more likely to churn.

**Tenure**

Customers with less than 6 months of tenure had approximately **54.27% churn**.

Customers in the first 12 months had approximately **47.44% churn**.

**Internet Service**

* Fiber optic: **41.89%**
* DSL: **18.96%**
* No internet service: **7.40%**

**Payment Method**

Electronic check customers had approximately **45.29% churn**, while automatic payment methods had much lower churn.

### Week 1 Lesson

Customer churn is strongly related to contract type, tenure, internet service, and payment method. Exploratory analysis helped identify these patterns before machine learning.

---

# Week 2 — Machine Learning Models

## Objective

The goal of Week 2 was to build baseline machine learning models and evaluate their performance.

### Data Preparation

* Removed `customerID`.
* Converted categorical variables using one-hot encoding.
* Converted `Churn` into a binary target.
* Split data into training and test sets.
* Used `StandardScaler` with Logistic Regression.

### Models

The project explored:

* Logistic Regression
* Random Forest
* XGBoost

### Evaluation Metrics

The models were evaluated using:

* Accuracy
* ROC AUC
* Recall
* Precision
* F1-score

### Key Lesson

Accuracy alone is not enough for a churn problem. ROC AUC measures overall ranking ability, while recall is particularly useful for identifying customers who are likely to churn.

---

# Week 3 — Model Optimization and Unsupervised Learning

## Objective

Week 3 focused on reliable model evaluation, hyperparameter tuning, customer segmentation, and PCA.

---

## 3.1 Split-to-Split Variability

The same Logistic Regression model was evaluated using 20 different random splits.

Results:

* Minimum accuracy: **78.0%**
* Maximum accuracy: **82.8%**
* Mean accuracy: **80.02%**
* Standard deviation: **0.0104**
* 95% theoretical interval: **±2.1 percentage points**

### Lesson

Changing only the random split changed the model accuracy considerably. This showed why relying on a single split can give misleading conclusions.

---

## 3.2 Five-Fold Cross-Validation

| Model               |       ROC AUC |        Recall |            F1 |
| ------------------- | ------------: | ------------: | ------------: |
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 ± 0.030 |
| Random Forest       | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |

Cross-validation provided a more reliable comparison than using one random split.

---

## 3.3 Hyperparameter Tuning

### Random Forest Grid Search

* Combinations: **24**
* 5-fold CV fits: **120**
* Best CV AUC: **0.8468**
* Time: **101 seconds**

Best parameters:

```text
max_depth = 8
max_features = sqrt
min_samples_leaf = 20
```

### Random Forest Random Search

* Iterations: **24**
* 5-fold CV fits: **120**
* Best CV AUC: **0.8464**
* Time: **109 seconds**

Best parameters:

```text
max_depth = 15
max_features ≈ 0.213
min_samples_leaf = 15
```

Grid Search achieved a slightly higher CV AUC in this experiment.

---

## 3.4 XGBoost

XGBoost was tuned using randomized hyperparameter search.

The tuned model achieved:

**CV AUC = 0.8502**

This was the highest CV AUC among the final candidate models.

### Final CV Comparison

| Model               |     CV AUC |     CV Std |
| ------------------- | ---------: | ---------: |
| Logistic Regression |     0.8464 |     0.0129 |
| Random Forest       |     0.8464 |     0.0114 |
| **XGBoost**         | **0.8502** | **0.0117** |

XGBoost was selected as the final model based on the highest CV AUC.

---

## 3.5 Customer Segmentation

K-means clustering was applied using:

* Tenure
* Monthly Charges
* Total Charges
* Number of services

Features were standardized before clustering.

Four clusters were selected based on clustering quality and business usefulness.

### Customer Segments

| Segment                     | Customers | Avg Tenure | Monthly Charge | Services |   Churn |
| --------------------------- | --------: | ---------: | -------------: | -------: | ------: |
| High-Risk New High-Spenders |     2,157 |      18.38 |          80.41 |     3.28 | **43%** |
| New Low-Service Customers   |     1,918 |       8.96 |          37.71 |     1.20 | **32%** |
| Loyal High-Value Customers  |     1,938 |      59.83 |          92.09 |     5.06 | **14%** |
| Loyal Low-Cost Customers    |     1,030 |      53.61 |          30.96 |     1.48 |  **5%** |

### Business Actions

**High-Risk New High-Spenders:**
Offer targeted discounts, proactive support, and attractive contract offers.

**New Low-Service Customers:**
Improve onboarding and offer service bundles.

**Loyal High-Value Customers:**
Provide loyalty rewards and premium support.

**Loyal Low-Cost Customers:**
Maintain reliable service and offer optional upgrades.

---

## 3.6 PCA

PCA was applied to study feature redundancy and dimensionality reduction.

The dataset contained **30 features**.

Results:

**15 of 30 components explain 90% of the variance.**

The strongest PC1 loadings were mainly related to customers with no internet service.

In the 2D PCA visualization, churned and non-churned customers were mostly mixed together, showing that the first two principal components do not clearly separate the two groups.

---

## 3.7 Final Model Evaluation

The final model was selected using cross-validation and then evaluated once on the untouched test set.

### Final Model

**Tuned XGBoost**

### Test Results

| Metric    |      Score |
| --------- | ---------: |
| AUC       | **0.8483** |
| Recall    |  **0.521** |
| Precision |  **0.659** |

The test AUC of **0.8483** was very close to the CV AUC of **0.8502**, showing that the model performed consistently on unseen data.

### Saved Model

The final model was saved as:

```text
churn_model.joblib
```

---

# Week 4 — Final Application

**Status:** To be completed.

The final model from Week 3 will be used for the Week 4 application/deployment stage.

---

# Final Project Lessons

The main lessons from this project are:

1. Exploratory Data Analysis helps identify important patterns before modeling.
2. A single random split can give an unreliable estimate of model performance.
3. Cross-validation provides a more trustworthy model comparison.
4. Hyperparameter tuning can improve model performance.
5. ROC AUC, recall, precision, and F1 provide different information about a classifier.
6. K-means can identify meaningful customer segments without using the target variable.
7. PCA can reduce dimensionality while preserving most of the information.
8. Model performance should always be evaluated on an untouched test set.

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

---

# Repository Structure

```text
applied-ai-project1-churn/
│
├── week1-eda.ipynb
├── week2-ml-models.ipynb
├── week3-optimization.ipynb
├── churn_model.joblib
└── README.md
```

---

# Final Model

**Tuned XGBoost**

**Test AUC: 0.8483**

**Test Recall: 0.521**

**Test Precision: 0.659**
