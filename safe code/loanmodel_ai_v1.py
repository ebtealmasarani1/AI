import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import joblib as bj

np.random.seed(42)

n = 500

income = np.random.normal(30000, 8000, n)
age = np.random.normal(40, 10, n)
credit_score = np.random.normal(600, 100, n)

noise = np.random.normal(0, 2000, n)

loan_approved = (
    (income + noise > 28000) &
    (credit_score > 550) &
    (age > 25)
).astype(int)

data = pd.DataFrame({
    "income": income,
    "age": age,
    "credit_score": credit_score,
    "loan_approved": loan_approved
})

x = data[["income", "age", "credit_score"]]

y = data["loan_approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

for d in [1, 2, 3, 4, 5, 6]:

    model = DecisionTreeClassifier(
        max_depth=d,
        random_state=42
    )

    model.fit(x_train, y_train)

    bj.dump(model, "model.pkl")

    predictions = model.predict(x_test)

    print("\n==============================")
    print("Djup:", d)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    print("Train accuracy:",
          accuracy_score(y_train, train_predictions))

    print("Test accuracy:",
          accuracy_score(y_test, test_predictions))

    print("Accuracy:",
          accuracy_score(y_test, predictions))

    print("Confusion matrix:")

    print(confusion_matrix(y_test, predictions))

    print("Feature importance:")

    for name, importance in zip(
        x.columns,
        model.feature_importances_
    ):

        print(name, ":", importance)

new_people = pd.DataFrame({
    "income": [22000, 32000, 45000],
    "age": [22, 30, 45],
    "credit_score": [500, 580, 700]
})

predictions = model.predict(new_people)

print("\n==============================")
print("Predictions för nya personer:")
print(predictions)

plt.figure(figsize=(14, 8))

tree.plot_tree(
    model,
    feature_names=x.columns,
    class_names=["Denied", "Approved"],
    filled=True
)

plt.show()