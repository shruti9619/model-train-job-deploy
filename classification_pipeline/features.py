from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def split_and_impute(
    df: pd.DataFrame,
    target: str = "Outcome",
    test_size: float = 0.2,
    random_state: int = 42,
    impute_strategy: str = "median"
):
    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    imputer = SimpleImputer(strategy=impute_strategy)
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp), columns=X_test_imp.columns, index=X_test_imp.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer