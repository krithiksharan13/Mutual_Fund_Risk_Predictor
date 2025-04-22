
# Mutual Fund Risk Indicator

This project uses machine learning to classify Indian mutual funds into **Low**, **Medium**, or **High** risk categories based on historical NAV (Net Asset Value) performance.

---

## 📊 Dataset

We use the **Indian Mutual Funds NAV History** dataset from Kaggle:

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/balajisr/indian-mutual-funds-dataset-2023)
- **File used:** `Mutual_Funds.csv`
---

## 🧠 Project Workflow

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
```

### 2. Load Dataset
```python
df = pd.read_csv("Mutual_Funds.csv")
```

### 3. Data Preprocessing
- Rename columns
- Parse dates
- Sort NAVs by scheme and date
```python
df.rename(columns={
    'Scheme Code': 'Scheme_Code',
    'Scheme Name': 'Scheme_Name',
    'Net Asset Value': 'NAV',
    'Date': 'Date'
}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.sort_values(by=['Scheme_Code', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)
```

### 4. Feature Engineering
```python
df['Daily_Return'] = df.groupby('Scheme_Code')['NAV'].pct_change()
df['Rolling_Std_30'] = df.groupby('Scheme_Code')['Daily_Return'].rolling(window=30).std().reset_index(level=0, drop=True)
df['Mean_Daily_Return'] = df.groupby('Scheme_Code')['Daily_Return'].transform('mean')
df['Std_Daily_Return'] = df.groupby('Scheme_Code')['Daily_Return'].transform('std')
df['Sharpe_Ratio'] = df['Mean_Daily_Return'] / df['Std_Daily_Return']
```

### 5. Labeling Risk
```python
def classify_risk(std):
    if std < 0.01:
        return 'Low'
    elif std < 0.02:
        return 'Medium'
    else:
        return 'High'

df['Risk_Level'] = df['Std_Daily_Return'].apply(classify_risk)
```

### 6. Train a Machine Learning Model
```python
features = ['Mean_Daily_Return', 'Std_Daily_Return', 'Sharpe_Ratio']
X = df[features]
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 7. Evaluate the Model
```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### 8. Save the Model
```python
joblib.dump(model, 'mutual_fund_risk_model.pkl')
```

### 9. Streamlit App for Deployment
```python
import streamlit as st
import numpy as np
import joblib

model = joblib.load('mutual_fund_risk_model.pkl')

st.title("Mutual Fund Risk Predictor")

mean_return = st.number_input("Mean Daily Return", value=0.0, format="%.5f")
std_return = st.number_input("Standard Deviation of Daily Return", value=0.0, format="%.5f")
sharpe_ratio = st.number_input("Sharpe Ratio", value=0.0, format="%.5f")

if st.button("Predict Risk Level"):
    input_data = np.array([[mean_return, std_return, sharpe_ratio]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Risk Level: {prediction[0]}")
```

```bash
streamlit run streamlit_app.py
```

### 📸 Screenshots

| Medium Risk | High Risk |
|-------------|------------|
| ![Medium Risk](./Screenshot%202025-04-22%20122254.png) | ![High Risk](./Screenshot%202025-04-22%20122430.png) |

---

## ✅ Future Improvements
- Use more advanced models like XGBoost or LightGBM
- Include sector, category, or fund type as additional features
- Add data from multiple years or external economic indicators

---

## 📁 Project Structure
```
├── Mutual_Funds.csv
├── mutual_fund_risk_model.pkl
├── streamlit_app.py
├── Mutual_Fund_Risk_Indicator.ipynb
├── Screenshot 2025-04-22 122254.png
├── Screenshot 2025-04-22 122430.png
└── README.md
```

---

Made with ❤️ for portfolio and educational use.
