# Predictive Churn Risk Classifier

A machine learning project designed to predict customer churn, identify key risk drivers, and stratify customers into actionable risk tiers for proactive retention strategies using the Telco Customer Churn dataset.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription-based services. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project provides an end-to-end machine learning pipeline that:
1. Cleans and preprocesses complex tabular customer data.
2. Identifies key indicators of customer attrition via Exploratory Data Analysis (EDA).
3. Trains and evaluates supervised classification models with a focus on recall and ROC-AUC.
4. Stratifies predictions into **Low**, **Medium**, and **High** risk tiers for targeted business action.

---

## 🛠️ Repository Structure

```text
├── notebooks/
│   ├── 01_Telco_Churn_EDA_and_Data_Preprocessing.ipynb
│   └── 02_Telco_Churn_Model_Training_and_Evaluation.ipynb
├── README.md
└── requirements.txt
