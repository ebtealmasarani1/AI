import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
np.random.seed(42)

n = 500

income = np.random.normal(30000, 8000, n)
age = np.random.normal(40, 10, n)
credit_score = np.random.normal(600, 100, n)
noise = np.random.normal(0, 2000, n)

loan_approved = (
    (income + noise > 28000) &
    (credit_score > 550)
).astype(int)

data = pd.DataFrame({
    "income": income,
    "age": age,
    "credit_score": credit_score,
    "loan_approved": loan_approved
})

X = data[["income", "age", "credit_score"]]
y = data["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modell
model = LogisticRegression()

model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# Spara modellen
joblib.dump(model, "complement_model.pkl")

print("complement_model.pkl created")
