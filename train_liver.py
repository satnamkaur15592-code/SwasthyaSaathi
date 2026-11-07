import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Internet se data lo - CSV ki zaroorat nahi
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
df = pd.read_csv(url, header=None)
df.columns = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Proteins','Albumin','Albumin_and_Globulin_Ratio','Dataset']

df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

X = df.drop('Dataset', axis=1)
y = (df['Dataset'] == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

with open("liver_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("liver_model.pkl ban gaya - Sahi wala!")