from pydantic import BaseModel
from typing import Optional
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from .utils import setup_logger
from .data import load_data, replace_zeros_with_nan
from .features import split_and_impute
from .model import ClassificationModelFactory, evaluate_model, plot_confusion


class PipelineConfig(BaseModel):
    n_neighbors: int = 5
    test_size: float = 1/3
    random_state: int = 42


class ClassificationPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.logger = setup_logger()
        self.raw_df: pd.DataFrame = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
        self.eval_results = None

        # register KNN model
        ClassificationModelFactory.register("knn", KNeighborsClassifier)

    def load_and_clean(self):
        self.raw_df = replace_zeros_with_nan(load_data())
        self.logger.info("Missing values after cleaning:\n", self.raw_df.isnull().sum())

    def prepare_features(self):
        self.raw_df["Glucose"].fillna(self.raw_df["Glucose"].mean(), inplace=True)
        self.raw_df["BloodPressure"].fillna(
            self.raw_df["BloodPressure"].mean(), inplace=True
        )
        self.raw_df["SkinThickness"].fillna(
            self.raw_df["SkinThickness"].median(), inplace=True
        )
        self.raw_df["Insulin"].fillna(self.raw_df["Insulin"].median(), inplace=True)
        self.raw_df["BMI"].fillna(self.raw_df["BMI"].median(), inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test, _ = split_and_impute(
            self.raw_df,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify_y=True,
        )

    def fit(self):
        if self.X_train is None:
            self.prepare_features()
        self.model = ClassificationModelFactory.create(
            "knn",
            **{"n_neighbors": self.cfg.n_neighbors},
        )
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"KNN (k={self.cfg.n_neighbors}) trained.")

    def evaluate(self):
        if self.model is None:
            self.fit()
        self.eval_results = evaluate_model(self.model, self.X_test, self.y_test)
        self.logger.info(f"Test accuracy: {self.eval_results['accuracy']:.4f}")

    def plot_confusion(self):
        if self.eval_results is None:
            self.evaluate()
        plot_confusion(self.eval_results["confusion"])
