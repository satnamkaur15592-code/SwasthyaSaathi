import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
# 1000 logo ka data
# Features: Age, Gender (1=Male, 0=Female), TSH, T3, T4
X = np.random.rand(1000, 5)
X[:,0] = X[:,0]*60 + 10   # Age 10-70
X[:,1] = (X[:,1]>0.5).astype(int) # Gender
X[:,2] = X[:,2]*10 + 0.1 # TSH 0.1-10.1
X[:,3] = X[:,3]*2 + 0.5  # T3 0.5-2.5
X[:,4] = X[:,4]*10 + 2   # T4 2-12

# Sahi Rule: TSH zyada ( > 5 ) ya T4 kam ( < 5 ) = Thyroid
y = []
for r in X:
    if r[2] > 5 or r[4] < 5:
        y.append(1)
    else:
        y.append(0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("thyroid_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("thyroid_model.pkl ban gaya! DONE")