import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Data Load
url = "https://archive.ics.uci.edu/static/public/336/data.csv"
df = pd.read_csv(url)
df = df.replace('?', np.nan)

X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# 2. Sahi Mapping - Yahi sabse important hai
# factorize mat use karna, wahi galti thi pehle
mapping = {
    'normal': 0, 'abnormal': 1,
    'notpresent': 0, 'present': 1,
    'no': 0, 'yes': 1,
    'good': 0, 'poor': 1,
    'ckd': 1, 'notckd': 0
}

for col in X.columns:
    X[col] = X[col].astype(str).str.strip().str.lower().map(mapping).fillna(pd.to_numeric(X[col], errors='coerce'))

X = X.fillna(0)
y = y_raw.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

print("CKD = 1, Not CKD = 0")
print(y.value_counts().to_dict())

# 3. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# 4. Save
with open("kidney_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("kidney_model.pkl ban gaya! DONE")