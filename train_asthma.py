import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
# 1000 logo ka data
# Features: Age, Gender (1=Male, 0=Female), Smoking (1=Yes, 0=No), Allergies (1=Yes, 0=No), Family History (1=Yes, 0=No), Peak Flow, Wheezing (1=Yes, 0=No), Night Cough (1=Yes, 0=No)
X = np.random.rand(1000, 8)
X[:,0] = X[:,0]*65 + 5    # Age 5-70
X[:,1] = (X[:,1]>0.5).astype(int) # Gender
X[:,2] = (X[:,2]>0.5).astype(int) # Smoking
X[:,3] = (X[:,3]>0.5).astype(int) # Allergies
X[:,4] = (X[:,4]>0.5).astype(int) # Family History
X[:,5] = X[:,5]*400 + 200 # Peak Flow 200-600
X[:,6] = (X[:,6]>0.5).astype(int) # Wheezing
X[:,7] = (X[:,7]>0.5).astype(int) # Night Cough

# Sahi Rule: Peak Flow kam (< 400) aur Wheezing hai = Asthma
y = []
for r in X:
    if r[5] < 400 and r[6] == 1:
        y.append(1)
    else:
        y.append(0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("asthma_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("asthma_model.pkl ban gaya! DONE")