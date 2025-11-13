import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "diabetes.csv"

def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.copy()

def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[cols] = df[cols].replace(0, pd.NA)
    return df