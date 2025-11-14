from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ClassificationModel:
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    

class ClassificationModelFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, model_cls):
        # check instance
        if not issubclass(model_cls, ClassificationModel) and not (hasattr(model_cls, "fit") and hasattr(model_cls, "predict")):
            raise ValueError("Model class must be a subclass of ClassificationModel with fit and predict methods.")
        cls._registry[name] = model_cls
    
    @classmethod
    def create(cls, key: str, **kwargs) -> ClassificationModel:
        if key not in cls._registry:
            raise ValueError(f"Model '{key}' is not registered.")
        model_cls = cls._registry[key]
        return model_cls(**kwargs)
    

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