from pydantic import BaseModel
from typing import Optional
import pandas as pd

from .utils import setup_logger
from .data import load_data, replace_zeros_with_nan
from .features import split_and_impute
from .model import KNNFactory, evaluate_model, plot_confusion

class PipelineConfig(BaseModel):
    k: int = 5
    impute_strategy: str = "median"
    test_size: float = 0.2
    random_state: int = 42

class ClassificationPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.logger = setup_logger()
        self.raw_df: pd.DataFrame = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
        self.eval_results = None

    def load_and_clean(self):
        self.raw_df = replace_zeros_with_nan(load_data())
        self.logger.info("Missing values after cleaning:\n", self.raw_df.isnull().sum())

    def prepare_features(self):
        self.X_train, self.X_test, self.y_train, self.y_test, _, _ = split_and_impute(
            self.raw_df,
            impute_strategy=self.cfg.impute_strategy,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

    def fit(self):
        if self.X_train is None:
            self.prepare_features()
        self.model = KNNFactory.create(k=self.cfg.k)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info(f"KNN (k={self.cfg.k}) trained.")

    def evaluate(self):
        if self.model is None:
            self.fit()
        self.eval_results = evaluate_model(self.model, self.X_test, self.y_test)
        self.logger.info(f"Test accuracy: {self.eval_results['accuracy']:.4f}")

    def plot_confusion(self):
        if self.eval_results is None:
            self.evaluate()
        plot_confusion(self.eval_results["confusion"])