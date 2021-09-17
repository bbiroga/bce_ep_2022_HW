import pandas as pd

COLNAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "target",
]


def read_data() -> pd.DataFrame:
    """Read the input data from disk, and calculate binary target variable"""
    adult_data = pd.read_csv("data/uci_adult.data", header=None)
    adult_data.columns = COLNAMES
    adult_data["target_encoded"] = adult_data["target"] != " <=50K"
    return adult_data


def plot_education_against_tv(adult_data: pd.DataFrame) -> None:
    adult_data.groupby("education").agg({"target_encoded": "mean"}).sort_values(
        by="target_encoded"
    ).plot(kind="bar")
