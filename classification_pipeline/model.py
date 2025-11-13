from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class KNNFactory:
    @staticmethod
    def create(k: int = 5, **kwargs) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=k, **kwargs)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "report": report, "confusion": cm, "predictions": y_pred}

def plot_confusion(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.title(title, y=1.1)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()