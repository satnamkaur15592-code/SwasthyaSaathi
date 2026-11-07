import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Cleveland Heart Dataset - sahi wala
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv(url, header=None, names=cols)

# ? wali values ko hatana
df = df.replace('?', pd.NA).dropna()
for c in cols:
    df[c] = pd.to_numeric(df[c])

# target ko 0 / 1 me karna - 0 = No disease, 1,2,3,4 = Disease
df['target'] = (df['target'] > 0).astype(int)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Heart model saved!")