# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the student placement dataset and preprocess it.
2.Split the data into training and testing sets.
3.Train the Logistic Regression model using training data.
4.Predict placement status and evaluate the model.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KAVIYA R
RegisterNumber:  212225040179
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"Placement_Data.csv")

df = df.drop(["sl_no", "salary"], axis=1)

df = df.dropna()
df = df.drop_duplicates()


le = LabelEncoder()
categorical_columns = [
    "gender", "ssc_b", "hsc_b", "hsc_s",
    "degree_t", "workex", "specialisation", "status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

print("        LOGISTIC REGRESSION CLASSIFICATION REPORT")
print(report)
print(f"Accuracy of the Model : {accuracy:.4f}")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

sample_input = pd.DataFrame(
    [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]],
    columns=X.columns
)

sample_prediction = model.predict(sample_input)
print("\nSample Prediction:", sample_prediction)

```

## Output:


<img width="953" height="819" alt="Screenshot 2026-04-29 114520" src="https://github.com/user-attachments/assets/1219ee6b-efcd-41ee-ab83-be1da8a687ad" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

