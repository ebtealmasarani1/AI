import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

np.random.seed(42) # fast värde för slumpgeneratorn seed() som finns i numpy, ta bort denna för att få lite olika dataset, accuracy, etc varje gång

n = 500 # storlek på dataset

income = np.random.normal(30000, 8000, n)
age = np.random.normal(40, 10, n)
credit_score = np.random.normal(600, 100, n)
noise = np.random.normal(0, 2000, n)
# noise = np.random.normal(0, 9000, n)

loan_approved = (
    (income + noise > 28000) &
    (credit_score > 550) &
    (age > 25)
).astype(int)

# Hårdare bank

#loan_approved = (
#    (income + noise > 34000) &
#    (credit_score > 650)
#).astype(int)

# Snällare bank

#loan_approved = (
#    (income + noise > 24000) &
#    (credit_score > 500)
#).astype(int)


data = pd.DataFrame({
    "income": income,
    "age": age,
    "credit_score": credit_score,
    "loan_approved": loan_approved
})

#print(data)

x = data[["income", "age", "credit_score"]]
#x = data[["age" ]]
y = data["loan_approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

for d in [1, 2, 3, 4, 5, 6]:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(x_train, y_train)

# moment7
    print("Feature importance:")
for name, importance in zip(x.columns, model.feature_importances_):
    print(name, ":", importance)


    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    print("Train accuracy:", accuracy_score(y_train, train_predictions))
    print("Test accuracy:", accuracy_score(y_test, test_predictions))

    predictions = model.predict(x_test)
    print("Djup:", d, "Accuracy:", accuracy_score(y_test, predictions))


print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

plt.figure(figsize = (10, 6))
tree.plot_tree(model,
           feature_names=x.columns,
           class_names=["Denied", "Approved"],
           filled=True)
plt.show()

new_people = pd.DataFrame({
    "income": [22000, 32000, 45000],
    "age": [22, 30, 45],
    "credit_score": [500, 580, 700]
})

predictions = model.predict(new_people)
print(predictions)