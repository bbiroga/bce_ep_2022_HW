"""This file contains the scripts and afterwork done for BCE Empiriku pénzügyek Szeminárium 1 by András Kárpáti.

Notice that I also put a docstring in the beginning of the file :) Feel free to check out this file,
I left an abundance of comments to explain why I did what
"""


# Imports come first, there is an Optimize Imports command in PyCharm, feel free to try it out
# The general logis is this: Built-in first, then 3rd party, then your own, and alphabetical order inside the
# categories
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


# module-wide constants are all caps and come after imports
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
    """Read the input data from disk, and calculate binary target variable

    Notice what I did here:

    * Grouped together a unch of related commands, that have a specific output: the data
    * Wrote a docstring
    * Added return type to the function
    * I assert that the encoded target has at least 2 different values
    """
    adult_data = pd.read_csv("data/uci_adult.data", header=None)
    adult_data.columns = COLNAMES
    adult_data["target_encoded"] = adult_data["target"] != " <=50K"
    assert (
        len(adult_data["target_encoded"].unique()) == 2
    )  # useful test to check if I got what I wanted
    return adult_data


def offset_bars_on_double_barplot(ax_1, ax_2, width_scale: float):
    """For double barchart use this to align bars

    Here I am removing a small snippet that is reusable in other scripts to reduce code-complexity
    """
    for bar in ax_1.containers[0]:
        bar.set_width(bar.get_width() * width_scale)

    for bar in ax_2.containers[0]:
        x = bar.get_x()
        w = bar.get_width()
        bar.set_x(x + w * (1 - width_scale))
        bar.set_width(w * width_scale)


def plot_education_against_tv(adult_data: pd.DataFrame) -> None:

    """PLots combined barchart of the pop count and the mean tv in the sample

    Source for combined barchart for sns:
    https://python.tutorialink.com/how-can-i-plot-a-secondary-y-axis-with-seaborns-barplot/

    * Notice how I link the stackoverflow resource for future me, who will change the script somehow and run into the
    exact error above


    Optional Excercise for the reader:
    How could we improve this?
    Well, eg the chart complexity could be reduced by grouping together the categories below HS-grad. This would get rid
    of category-ordering issues on the chart, create a combined category that is relevant in size. Less bars but more
    equal group sizes with a very minimal loss of information is a great tradeoff.

     How does this help us?

    First of all,
    they are very similar groups, so by grouping them together we reduce modeling complexity as well. This is why
    you need good visuals: they help convince the reader and also benefit modeling efforts if you understand whats going
    on.


    """
    width_scale = 0.5

    plot_series = adult_data.groupby("education").agg(
        {"target_encoded": ["mean", "count"]}
    )
    plot_series.columns = ["mean", "count"]
    plot_series.sort_values(by="mean", inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax2 = ax.twinx()

    sns.barplot(
        x=plot_series.index, y="mean", ax=ax, data=plot_series, color="blue", label="TV"
    )
    sns.barplot(
        x=plot_series.index,
        y="count",
        ax=ax2,
        data=plot_series,
        color="grey",
        label="COUNT",
    )
    offset_bars_on_double_barplot(ax, ax2, width_scale)

    fig.suptitle("Education against Income")
    ax.set_ylabel("Mean tv per group")
    ax.set_xlabel("Education level")
    rotate_ax_ticklabels(ax)

    # create legend
    grey_patch = mpatches.Patch(color="grey", label="Population size")
    blue_patch = mpatches.Patch(color="blue", label="Income >50k $")

    plt.legend(handles=[blue_patch, grey_patch], loc=2)


def rotate_ax_ticklabels(ax):
    """Rotate xticklabels by 45 deg

    * No subtask is too small for factoring out from a long function such as the above
    """
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


def create_hours_income_plot(adult_data: pd.DataFrame):
    """Create and label a violin plot from the hours per week and the income"""
    ax = sns.violinplot(x="target", y="hours_per_week", data=adult_data)
    ax.set_title("Hours worked per against income")
    ax.set_ylabel("Hours per week")
    ax.set_xlabel("Target Var")
