from pydantic import BaseModel
from typing import Optional


class KNNConfig(BaseModel):
    n_neighbors: int = 5


class DecisionTreeConfig(BaseModel):
    max_depth: Optional[int] = 6
    min_samples_split: int = 10
    random_state: int = 42
