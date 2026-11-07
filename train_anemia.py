import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
# 1000 logo ka data
# Features: Age, Gender (1=Male, 0=Female), Hemoglobin, RBC Count, MCV, Iron Level
X = np.random.rand(1000, 6)
X[:,0] = X[:,0]*70 + 10   # Age 10-80
X[:,1] = (X[:,1]>0.5).astype(int) # Gender
X[:,2] = X[:,2]*8 + 7    # Hemoglobin 7-15
X[:,3] = X[:,3]*3 + 3    # RBC 3-6
X[:,4] = X[:,4]*40 + 60  # MCV 60-100
X[:,5] = X[:,5]*150 + 10 # Iron 10-160

# Sahi Rule: Hemoglobin kam = Anemia
# Ladke me < 13 aur Ladki me < 12 = Anemia
y = []
for r in X:
    if r[1] == 1: # Male
        anemic = 1 if r[2] < 13 else 0
    else: # Female
        anemic = 1 if r[2] < 12 else 0
    y.append(anemic)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("anemia_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("anemia_model.pkl ban gaya! DONE")
