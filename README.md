# ML-Powered Customer Analytics

**Author:** Haseeb Sheikh
**Project:** Project 1 – ML-Powered Customer Analytics
**Dataset:** Telco Customer Churn
**Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, Joblib, Kaggle

---

## 📌 Project Overview

This project focuses on analyzing telecom customer data and building machine learning models to predict customer churn.

Customer churn means a customer stops using a company's service. Predicting churn can help a company identify customers who are likely to leave and take preventive actions such as discounts, better support, or service offers.

The project was completed in three weeks:

* **Week 1:** Exploratory Data Analysis (EDA)
* **Week 2:** Baseline Machine Learning and Model Evaluation
* **Week 3:** Model Optimization and Unsupervised Learning

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Understand the customer churn dataset.
2. Perform exploratory data analysis.
3. Identify important patterns related to customer churn.
4. Build classification models to predict churn.
5. Compare different machine learning models.
6. Use cross-validation for more reliable evaluation.
7. Tune machine learning hyperparameters.
8. Identify customer segments using K-means clustering.
9. Apply PCA for dimensionality reduction.
10. Select and save the final churn prediction model.

---

# 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about telecom customers, including:

* Customer demographics
* Contract information
* Internet services
* Phone services
* Payment methods
* Monthly charges
* Total charges
* Tenure
* Churn status

### Target Variable

The target variable is:

**Churn**

* `Yes` → Customer left the company
* `No` → Customer stayed with the company

For machine learning:

* `Yes = 1`
* `No = 0`

---

# 🗓️ Week 1 — Exploratory Data Analysis

## Objective

The goal of Week 1 was to understand the dataset before applying machine learning.

The following steps were performed:

* Loaded the dataset using Pandas.
* Checked dataset shape and data types.
* Checked missing values.
* Converted `TotalCharges` into numeric format.
* Filled missing `TotalCharges` values.
* Analyzed the churn distribution.
* Studied customer tenure.
* Compared churn across contract types.
* Compared churn across internet service types.
* Analyzed payment methods and monthly charges.

---

## 🔍 Important EDA Findings

### Contract Type

Customers with month-to-month contracts had a much higher churn rate than customers with one-year or two-year contracts.

* Month-to-month: **42.71%**
* One year: **11.27%**
* Two year: **2.83%**

This shows that customers with longer contracts are much more likely to remain with the company.

---

### Customer Tenure

New customers had a higher probability of leaving.

Customers with less than 6 months of tenure had a churn rate of approximately:

**54.27%**

Customers in the first 12 months also showed high churn:

**47.44%**

This suggests that early customer experience is very important for retention.

---

### Internet Service

Fiber optic customers showed higher churn compared with DSL customers.

* Fiber optic: **41.89%**
* DSL: **18.96%**
* No internet service: **7.40%**

This indicates that internet service type is an important feature for churn analysis.

---

### Payment Method

Electronic check customers had a high churn rate:

**45.29%**

Customers using automatic payment methods had much lower churn:

* Credit card automatic: **15.24%**
* Bank transfer automatic: **16.00%**

This suggests that automatic payment methods may be associated with stronger customer retention.

---

# 🤖 Week 2 — Baseline Machine Learning

## Objective

Week 2 focused on building baseline classification models and evaluating their performance.

The dataset was divided into:

* Training set: **80%**
* Test set: **20%**

Stratified splitting was used to preserve the churn class distribution.

---

## Feature Preparation

The following preprocessing steps were used:

* Removed `customerID`.
* Converted categorical variables using one-hot encoding.
* Converted the target `Churn` into binary values.
* Used `StandardScaler` for Logistic Regression.

---

## Models

The main models explored during the project included:

### Logistic Regression

Logistic Regression was used as a simple and interpretable baseline classification model.

### Random Forest

Random Forest was used to capture nonlinear relationships between customer features.

### XGBoost

XGBoost was later introduced as a powerful gradient boosting model for improved prediction performance.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* ROC AUC
* Recall
* Precision
* F1-score

### Why ROC AUC?

ROC AUC measures how well the model separates customers who churn from customers who stay across different classification thresholds.

### Why Recall?

Recall is important because missing a customer who is likely to churn can result in a lost customer.

---

# ⚙️ Week 3 — Model Optimization and Unsupervised Learning

## Objective

Week 3 focused on improving model evaluation, hyperparameter tuning, customer segmentation, and dimensionality reduction.

---

# 1. Split-to-Split Variability

The same Logistic Regression model was evaluated using 20 different random train-validation splits.

Results:

* **Minimum accuracy:** 78.0%
* **Maximum accuracy:** 82.8%
* **Mean accuracy:** 80.02%
* **Standard deviation:** 0.0104
* **Theoretical standard error:** 0.0107
* **95% CI:** ±2.1 percentage points

The results showed that model accuracy can change significantly simply because the random split changes.

### Lesson

A single train-test split can give an unstable estimate of model performance. This is why cross-validation is more reliable.

---

# 2. 5-Fold Cross-Validation

Stratified 5-fold cross-validation was used to compare Logistic Regression and Random Forest.

| Model               |       ROC AUC |        Recall |            F1 |
| ------------------- | ------------: | ------------: | ------------: |
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 ± 0.030 |
| Random Forest       | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |

Logistic Regression achieved slightly higher ROC AUC and recall than the baseline Random Forest.

### Important Lesson

ROC AUC alone does not tell the complete story. Recall tells us how many of the actual churners the model successfully identifies.

---

# 3. Hyperparameter Tuning

## Logistic Regression

A validation curve was used to test different values of the regularization parameter `C`.

The goal was to find a value that provided a good balance between underfitting and overfitting.

---

## Random Forest Grid Search

The following parameters were tested:

* `max_depth`
* `min_samples_leaf`
* `max_features`

Grid Search evaluated:

**24 parameter combinations × 5 folds = 120 fits**

Results:

* Best CV AUC: **0.8468**
* Time: **101 seconds**

Best parameters:

```text
max_depth = 8
max_features = sqrt
min_samples_leaf = 20
```

---

## Random Search

Random Search evaluated 24 randomly selected parameter combinations.

Results:

* Best CV AUC: **0.8464**
* Time: **109 seconds**

Best parameters:

```text
max_depth = 15
max_features ≈ 0.213
min_samples_leaf = 15
```

### Grid vs Random Search

In this experiment, Grid Search achieved a slightly higher AUC:

**0.8468 vs 0.8464**

However, Random Search can be more useful when there are many hyperparameters because it can explore a large search space without testing every possible combination.

---

# 4. XGBoost Optimization

XGBoost was trained using:

* Learning rate
* Tree depth
* Number of estimators
* Subsampling
* Column sampling
* Minimum child weight
* Regularization

Early stopping was also used to prevent unnecessary boosting rounds and reduce overfitting.

Randomized Search was then used to tune the XGBoost model.

### Tuned XGBoost CV AUC

**0.8502**

XGBoost achieved the highest cross-validation AUC among the final candidate models.

---

# 🏆 Final Model Comparison

| Model               |     CV AUC |     CV Std |
| ------------------- | ---------: | ---------: |
| Logistic Regression |     0.8464 |     0.0129 |
| Random Forest       |     0.8464 |     0.0114 |
| **XGBoost**         | **0.8502** | **0.0117** |

XGBoost was selected as the final model because it achieved the highest cross-validation AUC.

However, the improvement was small, so the models performed quite similarly.

---

# 5. Customer Segmentation with K-Means

Unsupervised learning was used to identify different types of customers.

The following features were used:

* Tenure
* Monthly Charges
* Total Charges
* Number of services

The features were standardized before applying K-means.

---

## Choosing the Number of Clusters

Values of `k` from 2 to 8 were tested.

The silhouette scores were:

|  k | Silhouette |
| -: | ---------: |
|  2 |     0.4575 |
|  3 |     0.4076 |
|  4 |     0.4015 |
|  5 |     0.3825 |
|  6 |     0.3718 |
|  7 |     0.3622 |
|  8 |     0.3712 |

Although `k=2` had the highest silhouette score, **k=4** was selected because it provided more useful and actionable customer segments while still giving a reasonable clustering quality.

---

# 👥 Customer Segments

Four customer segments were identified.

| Segment                     | Customers | Avg Tenure | Monthly Charge | Avg Services |   Churn |
| --------------------------- | --------: | ---------: | -------------: | -----------: | ------: |
| High-Risk New High-Spenders |     2,157 |      18.38 |          80.41 |         3.28 | **43%** |
| New Low-Service Customers   |     1,918 |       8.96 |          37.71 |         1.20 | **32%** |
| Loyal High-Value Customers  |     1,938 |      59.83 |          92.09 |         5.06 | **14%** |
| Loyal Low-Cost Customers    |     1,030 |      53.61 |          30.96 |         1.48 |  **5%** |

---

## Segment-Based Retention Actions

### High-Risk New High-Spenders

This is the riskiest segment with a **43% churn rate**.

**Action:** Provide targeted discounts, proactive customer support, and attractive contract offers.

### New Low-Service Customers

These customers have short tenure and use fewer services.

**Action:** Improve onboarding and offer service bundles to increase engagement.

### Loyal High-Value Customers

These customers have long tenure and use many services.

**Action:** Provide loyalty rewards and premium customer support.

### Loyal Low-Cost Customers

These customers have long tenure and the lowest churn rate.

**Action:** Maintain reliable service and offer optional low-cost upgrades.

---

# 6. PCA — Dimensionality Reduction

Principal Component Analysis was used to study feature redundancy and reduce dimensionality.

The original dataset contained:

**30 features**

PCA showed that:

**15 components explain 90% of the variance.**

Therefore, the dimensionality can be reduced from 30 components to 15 while retaining approximately 90% of the information.

---

## PC1 Interpretation

The strongest PC1 loadings were related to:

* InternetService_No
* OnlineSecurity_No internet service
* TechSupport_No internet service
* StreamingTV_No internet service
* DeviceProtection_No internet service
* OnlineBackup_No internet service

This indicates that PC1 mainly represents **internet-service availability**.

---

## PCA Visualization

When customers were projected onto the first two principal components, churned and non-churned customers were mostly mixed together.

This means that the first two principal components do not provide a strong visual separation between churners and customers who stay.

---

# 7. Final Test Evaluation

After selecting the best model using cross-validation, XGBoost was trained on the complete training set.

The test set was then used **only once** for final evaluation.

### Final Model

**XGBoost (Tuned)**

### Test Results

| Metric    |      Score |
| --------- | ---------: |
| ROC AUC   | **0.8483** |
| Recall    |  **0.521** |
| Precision |  **0.659** |

The test AUC of **0.8483** was very close to the cross-validation AUC of **0.8502**.

This indicates that the model's performance on unseen data was consistent with its cross-validation estimate.

---

# 💾 Saved Model

The final trained model was saved as:

```text
churn_model.joblib
```

This file can be loaded later and used for predictions without retraining the model.

Example:

```python
import joblib

model = joblib.load("churn_model.joblib")
```

---

# 🧠 Key Lessons Learned

### Week 1

EDA is important because it helps understand the dataset and discover patterns before building a model.

### Week 2

Different evaluation metrics provide different information. Accuracy alone is not enough for a churn prediction problem.

### Week 3

A single train-test split can produce misleading results. Cross-validation provides a more reliable estimate of model performance.

Hyperparameter tuning can improve model performance, but improvements should be evaluated carefully.

Unsupervised learning can identify customer groups even without using the churn label.

PCA can reduce dimensionality and reveal redundancy between features.

---

# 📈 Final Project Results

* Accuracy range across 20 random splits: **78.0%–82.8%**
* Logistic Regression CV AUC: **0.8464**
* Random Forest CV AUC: **0.8464**
* XGBoost CV AUC: **0.8502**
* Final Test AUC: **0.8483**
* Final Test Recall: **0.521**
* Final Test Precision: **0.659**
* Customer segments: **4**
* Highest-risk segment churn: **43%**
* PCA components for 90% variance: **15 of 30**
* Final model: **Tuned XGBoost**

---

# 🛠️ Technologies Used

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

# 📁 Project Structure

```text
Project-1-ML-Powered-Customer-Analytics/
│
├── We
```
## Week 3: Model Optimization and Unsupervised Learning

* Split-to-split accuracy range across 20 seeds: **78.0% to 82.8%**
* 5-fold CV AUC: **LR 0.846 ± 0.013, RF 0.844 ± 0.011, XGBoost 0.850 ± 0.012**
* Tuning: best RF Grid parameters **max_depth=8, max_features='sqrt', min_samples_leaf=20**; Grid Search took **101s** and Random Search took **109s**
* Test AUC of final model (used once): **0.8483**
* Customer segments (k = 4): **High-Risk New High-Spenders (43% churn), New Low-Service Customers (32% churn), Loyal High-Value Customers (14% churn), Loyal Low-Cost Customers (5% churn)**
* PCA: **15 of 30 components** explain 90% of the variance
* Biggest lesson: **A single train-test split can give an unreliable view of model performance, so cross-validation provides a more trustworthy estimate.**
