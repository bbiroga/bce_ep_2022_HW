import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def offset_bars_on_secondary_axis(ax, width_scale: float):
    print(type(ax))
    for bar in ax.containers[0]:
        x = bar.get_x()
        w = bar.get_width()
        bar.set_x(x + w * (1 - width_scale))
        bar.set_width(w * width_scale)


def plot_education_against_tv(adult_data: pd.DataFrame) -> None:

    """PLots combined barchart of the pop count and the mean tv in the sample

    Source for combined barchart for sns:
    https://python.tutorialink.com/how-can-i-plot-a-secondary-y-axis-with-seaborns-barplot/
    """
    width_scale = 0.45

    plot_series = adult_data.groupby("education").agg(
        {"target_encoded": ["mean", "count"]}
    )
    plot_series.columns = ["mean", "count"]
    plot_series.sort_values(by="mean", inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.barplot(x=plot_series.index, y="mean", ax=ax, data=plot_series, color="blue")
    for bar in ax.containers[0]:
        bar.set_width(bar.get_width() * width_scale)
    ax2 = ax.twinx()
    sns.barplot(x=plot_series.index, y="count", ax=ax2, data=plot_series, color="grey")
    offset_bars_on_secondary_axis(ax2, width_scale)

    fig.suptitle("Education against Income")
    ax.set_ylabel("Mean tv per group")
    ax.set_xlabel("Education level")
