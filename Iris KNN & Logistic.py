import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Iris (1).csv")

print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns)

print("\nFeature Meaning:")
print("SepalLengthCm : Length of sepal (cm)")
print("SepalWidthCm  : Width of sepal (cm)")
print("PetalLengthCm : Length of petal (cm)")
print("PetalWidthCm  : Width of petal (cm)")
print("Species       : Type of iris flower (target)")

df = df.dropna()

X = df.drop(columns=["Species"])
y = df["Species"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)


log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)


print("\n     KNN Results      ")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))


print("\n      Logistic Regression Results     ")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))


knn_acc = accuracy_score(y_test, y_pred_knn)
log_acc = accuracy_score(y_test, y_pred_log)

print("\n       Model Comparison       ")
print("KNN Accuracy:", knn_acc)
print("Logistic Regression Accuracy:", log_acc)

if knn_acc > log_acc:
    print(" KNN performed better")
else:
    print(" Logistic Regression performed better")


plt.figure(figsize=(6,4))

species_names = le.classes_

for i, species in enumerate(species_names):
    
    subset = df[y == i]
    
    plt.scatter(
        subset["SepalLengthCm"],
        subset["SepalWidthCm"],
        label=species
    )

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Feature Relationship")

plt.legend()

plt.show()


cm = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.colorbar()

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


