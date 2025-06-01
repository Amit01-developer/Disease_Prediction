import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

import os
print("Current Directory:", os.getcwd())
print("Files in directory:", os.listdir())

df = pd.read_csv(r"C:\Users\Amit Chaurasiya\Desktop\Task\DiseasePrediction\symptoms_disease.csv")

X = df.drop("disease", axis=1)
y = df["disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
