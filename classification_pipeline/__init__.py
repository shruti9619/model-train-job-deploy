from .pipeline import ClassificationPipeline
from .model_params import KNNConfig, DecisionTreeConfig  # expose them!

__all__ = ["ClassificationPipeline", "KNNConfig", "DecisionTreeConfig"]
__version__ = "0.1.0"
