
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_digits



df = pd.read_csv("breast-cancer-wisconsin-data.csv")


df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})


df.drop(columns=["id"], inplace=True)


df.dropna(inplace=True)

print("Dataset Shape:", df.shape)


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


print("\nKNN Model Results")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.colorbar()

plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN")

# Add values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")


X_dig, y_dig = load_digits(return_X_y=True)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dig, y_dig,
    test_size=0.25,
    random_state=23
)

gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_features=5,
    random_state=100
)

gb_model.fit(X_train_d, y_train_d)

y_pred_d = gb_model.predict(X_test_d)

print("\nGradient Boosting Accuracy: {:.2f}%".format(
    accuracy_score(y_test_d, y_pred_d) * 100
))

plt.show()