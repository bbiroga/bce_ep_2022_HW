"""This file contains utilities for exploringthe iris dataset"""
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris


def load_iris_dataset() -> pd.DataFrame:
    """load and organize the iris dataset to a usable format"""
    iris_data = load_iris()
    iris_df = pd.DataFrame(data=iris_data["data"], columns=iris_data["feature_names"])
    iris_df["species"] = iris_data["target"]
    iris_df["species"] = iris_df["species"].replace(
        get_iris_species_recode_dict(iris_df, iris_data["target_names"])
    )
    return iris_df


def get_iris_species_recode_dict(
    iris_df: pd.DataFrame, feature_names: List[str]
) -> Dict[int, str]:
    return {
        target_value: target_name
        for target_value, target_name in zip(iris_df["species"].unique(), feature_names)
    }
