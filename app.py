import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
page_title="Customer Churn Risk Advisor",
page_icon="📊",
layout="wide"
)

RAW_COLS = [
"gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
"PhoneService", "MultipleLines", "InternetService",
"OnlineSecurity", "OnlineBackup", "DeviceProtection",
"TechSupport", "StreamingTV", "StreamingMovies", "Contract",
"PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

SERVICES = [
"OnlineSecurity", "OnlineBackup", "DeviceProtection",
"TechSupport", "StreamingTV", "StreamingMovies"
]

PAYMENTS = [
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]

@st.cache_resource
def load_artifacts():
model = joblib.load("churn_model.joblib")

```
with open("model_meta.json", "r") as file:
    meta = json.load(file)

return model, meta
```

def prepare_input(raw, columns):
missing = [col for col in RAW_COLS if col not in raw.columns]
if missing:
raise ValueError(f"Missing customer columns: {missing}")

```
data = raw[RAW_COLS].copy()
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
).fillna(0)

encoded = pd.get_dummies(data)
return encoded.reindex(columns=columns, fill_value=0).astype(float)
```

@st.cache_data
def load_sample():
if os.path.exists("sample_customers.csv"):
return pd.read_csv("sample_customers.csv")
return None

try:
model, meta = load_artifacts()
COLS = meta["feature_columns"]
except Exception as exc:
st.error(
"Could not load the model files. Check that "
"churn_model.joblib and model_meta.json are in the "
"same GitHub repository as app.py."
)
st.exception(exc)
st.stop()

def score(raw):
features = prepare_input(raw, COLS)
return model.predict_proba(features)[:, 1]

st.title("Customer Churn Risk Advisor")
st.write(
"Predict customer churn risk, explore possible changes, "
"and score multiple customers using a CSV file."
)

st.caption(
f"Model: {meta.get('model_name', 'Saved model')} | "
f"Cross-validation AUC: {meta.get('cv_auc', 0):.4f} "
f"+/- {meta.get('cv_auc_std', 0):.4f} | "
f"Test AUC: {meta.get('test_auc', 0):.4f}"
)

with st.sidebar:
st.header("Customer Profile")

```
tenure = st.slider("Tenure (months)", 0, 72, 4)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

monthly = st.number_input(
    "Monthly charges (USD)",
    min_value=18.0,
    max_value=120.0,
    value=85.0,
    step=0.5
)

internet = st.selectbox(
    "Internet service",
    ["Fiber optic", "DSL", "No"]
)

services = []
if internet != "No":
    services = st.multiselect(
        "Add-on services",
        SERVICES,
        default=["StreamingTV"]
    )

phone = st.radio(
    "Phone service", ["Yes", "No"], horizontal=True
)

if phone == "Yes":
    multi = st.radio(
        "Multiple lines", ["No", "Yes"], horizontal=True
    )
else:
    multi = "No phone service"

payment = st.selectbox("Payment method", PAYMENTS)

paperless = st.radio(
    "Paperless billing", ["Yes", "No"], horizontal=True
)

with st.expander("Demographics"):
    gender = st.radio(
        "Gender", ["Female", "Male"], horizontal=True
    )
    senior = st.checkbox("Senior citizen")
    partner = st.checkbox("Has partner")
    dependents = st.checkbox("Has dependents")

threshold = st.slider(
    "Contact threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(meta.get("threshold", 0.30)),
    step=0.05
)
```

row = {
"gender": gender,
"SeniorCitizen": int(senior),
"Partner": "Yes" if partner else "No",
"Dependents": "Yes" if dependents else "No",
"tenure": tenure,
"PhoneService": phone,
"MultipleLines": multi,
"InternetService": internet,
"Contract": contract,
"PaperlessBilling": paperless,
"PaymentMethod": payment,
"MonthlyCharges": monthly,
"TotalCharges": tenure * monthly
}

for service in SERVICES:
if internet == "No":
row[service] = "No internet service"
else:
row[service] = "Yes" if service in services else "No"

customer = pd.DataFrame([row])

tab1, tab2, tab3 = st.tabs(
["One Customer", "Batch Scoring", "About the Model"]
)

with tab1:
st.subheader("Churn Risk Prediction")

```
try:
    probability = float(score(customer)[0])
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

if probability >= threshold:
    band = "HIGH"
    action = "Contact customer for retention"
elif probability >= threshold / 2:
    band = "WATCH"
    action = "Monitor customer"
else:
    band = "LOW"
    action = "Routine monitoring"

col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{probability:.1%}")
col2.metric("Risk Band", band)
col3.metric("Suggested Action", action)

st.progress(min(max(probability, 0.0), 1.0))

st.caption(
    "The threshold determines the risk band; changing it "
    "does not change the model's predicted probability."
)

st.subheader("What-if Analysis")

changes = [
    ("Contract", value)
    for value in ["One year", "Two year"]
    if value != contract
]

if payment == "Electronic check":
    changes.append(
        ("PaymentMethod", "Credit card (automatic)")
    )

if internet != "No" and "TechSupport" not in services:
    changes.append(("TechSupport", "Yes"))

results = []

for feature, value in changes:
    alternative = customer.copy()
    alternative.loc[:, feature] = value
    new_probability = float(score(alternative)[0])

    results.append({
        "Potential Change": f"{feature}: {value}",
        "New Churn Probability": round(new_probability, 3),
        "Difference from Current": round(
            new_probability - probability, 3
        )
    })

if results:
    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No alternative changes available for this profile.")

st.caption(
    "What-if results show model associations, not guaranteed "
    "causal effects."
)

final_model = (
    model[-1] if hasattr(model, "steps") else model
)

if hasattr(model, "steps") and hasattr(final_model, "coef_"):
    with st.expander("Why this score?"):
        try:
            transformed = model[:-1].transform(
                prepare_input(customer, COLS)
            )[0]

            contributions = pd.Series(
                final_model.coef_[0] * transformed,
                index=COLS
            )

            top = contributions.reindex(
                contributions.abs()
                .sort_values(ascending=False)
                .index
            ).head(8)

            st.bar_chart(top)
            st.caption(
                "Positive contributions push the score toward "
                "churn; negative contributions push it away."
            )
        except Exception:
            st.info(
                "Feature contributions are not available "
                "for this model configuration."
            )
```

with tab2:
st.subheader("Batch Customer Scoring")
st.write(
"Upload a CSV containing the original customer columns. "
"The app will calculate churn probabilities for all rows."
)

```
sample = load_sample()

if sample is not None:
    st.download_button(
        "Download Sample Customer CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        missing = [
            col for col in RAW_COLS if col not in data.columns
        ]

        if missing:
            st.error(f"Required columns are missing: {missing}")
        elif data.empty:
            st.warning("The uploaded CSV contains no customers.")
        else:
            probabilities = score(data)
            data["p_churn"] = np.round(probabilities, 3)
            data["risk_band"] = np.where(
                data["p_churn"] >= threshold,
                "HIGH",
                np.where(
                    data["p_churn"] >= threshold / 2,
                    "WATCH",
                    "LOW"
                )
            )
            data["contact"] = np.where(
                data["p_churn"] >= threshold, "Yes", "No"
            )

            high_risk = int(
                (data["p_churn"] >= threshold).sum()
            )

            st.metric(
                "Customers above contact threshold",
                f"{high_risk} of {len(data)}"
            )

            st.dataframe(
                data.sort_values(
                    "p_churn", ascending=False
                ).head(50),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Scored Customers CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv"
            )

    except Exception as exc:
        st.error(f"Could not score the uploaded file: {exc}")
```

with tab3:
st.subheader("Model Information")

```
visible_meta = {
    key: value
    for key, value in meta.items()
    if key != "feature_columns"
}
st.json(visible_meta)

st.markdown(
    """
    **Limitations**

    - Trained using a US telecom customer churn dataset.
    - Results have not been validated on Pakistani telecom data.
    - Model predictions indicate associations, not causes.
    - A high-risk score is a decision-support signal, not a
      guarantee that a customer will leave.
    """
)
```
