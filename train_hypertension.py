import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
# 1000 logo ka data banaya hai
# Features: Age, BMI, Smoking, Alcohol, Exercise, Family History, Salt, Stress
X = np.random.rand(1000, 8)
X[:,0] = X[:,0]*70 + 15  # Age 15-85
X[:,1] = X[:,1]*25 + 15  # BMI 15-40
X[:,2] = (X[:,2]>0.5).astype(int) # Smoking 0/1
X[:,3] = (X[:,3]>0.5).astype(int) # Alcohol 0/1
X[:,4] = X[:,4]*10 # Exercise 0-10
X[:,5] = (X[:,5]>0.5).astype(int) # Family History 0/1
X[:,6] = X[:,6]*10 # Salt 0-10
X[:,7] = X[:,7]*10 # Stress 0-10

# Rule: Age zyada + BMI zyada + Smoking + Family History = High BP
y = []
for r in X:
    score = 0
    if r[0] > 50: score += 1
    if r[1] > 28: score += 1
    if r[2] == 1: score += 1
    if r[5] == 1: score += 1
    if r[6] > 7: score += 1
    if r[7] > 7: score += 1
    if r[4] < 2: score += 1
    y.append(1 if score >= 4 else 0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("hypertension_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("hypertension_model.pkl ban gaya! DONE")