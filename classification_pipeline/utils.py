"""
Utility functions and shared helpers for the classification pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import json
import yaml
from pydantic import BaseModel, ValidationError


# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
def setup_logger(
    name: str = "classification_pipeline", level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Prevent duplicate handlers in notebooks
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------------- #
# Config Loading (YAML/JSON)
# --------------------------------------------------------------------------- #
def load_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


# --------------------------------------------------------------------------- #
# Pydantic Config Validator (Reusable)
# --------------------------------------------------------------------------- #
def validate_config(config_data: Dict[str, Any], model: type[BaseModel]) -> BaseModel:
    """
    Validate raw config dict against a Pydantic model.
    """
    try:
        return model(**config_data)
    except ValidationError as e:
        logging.getLogger("classification_pipeline").error("Config validation failed:")
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            logging.error(f"  {loc}: {err['msg']} ({err['type']})")
        raise


# --------------------------------------------------------------------------- #
# File/Directory Helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path | str) -> Path:
    """
    Ensure directory exists. Return Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """
    Return the root directory of the project (where pyproject.toml lives).
    """
    return Path(__file__).parent.parent.resolve()
