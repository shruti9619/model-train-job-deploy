from pydantic import BaseModel
from typing import Optional, Union, Literal
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .utils import setup_logger
from .data import load_data, replace_zeros_with_nan
from .features import split_and_impute
from .model import ClassificationModelFactory, evaluate_model, plot_confusion
from .model_params import KNNConfig, DecisionTreeConfig


class PipelineConfig(BaseModel):
    test_size: float = 1 / 3
    random_state: int = 42
    model_name: Literal["knn", "dtree"] = "dtree"  # or other registered models
    model_params: Union[KNNConfig, DecisionTreeConfig] = DecisionTreeConfig()


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
        # register Decision Tree model
        ClassificationModelFactory.register("dtree", DecisionTreeClassifier)

    def load_and_clean(self):
        self.raw_df = replace_zeros_with_nan(load_data())
        self.logger.info(
            f"Missing values after cleaning:\n{self.raw_df.isnull().sum()}"
        )

    def prepare_features(self):
        impute_map = {
            "Glucose": self.raw_df["Glucose"].mean(),
            "BloodPressure": self.raw_df["BloodPressure"].mean(),
            "SkinThickness": self.raw_df["SkinThickness"].median(),
            "Insulin": self.raw_df["Insulin"].median(),
            "BMI": self.raw_df["BMI"].median(),
        }

        # Apply imputation
        self.raw_df.fillna(value=impute_map, inplace=True)

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
            model_name=self.cfg.model_name,
            model_params=self.cfg.model_params,
        )
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(
            f"{self.cfg.model_name} (params={self.cfg.model_params.model_dump()} trained."
        )

    def evaluate(self):
        if self.model is None:
            self.fit()
        self.eval_results = evaluate_model(self.model, self.X_test, self.y_test)
        self.logger.info(f"Test accuracy: {self.eval_results['accuracy']:.4f}")
        self.logger.info(f"Classification Report:\n{self.eval_results['report']}")

    def plot_confusion(self):
        if self.eval_results is None:
            self.evaluate()
        plot_confusion(self.eval_results["confusion"])
