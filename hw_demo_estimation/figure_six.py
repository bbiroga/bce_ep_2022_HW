"""Calculations and visualization functions for Figure 6 - Demographic dynamics"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

"""Figure 6 step1: first we create the data for the (a) chart (Age x female users)"""

# F_F connections
def pivot_F_F(edges_w_features):
    """Aim of this function is to create pivot table for FEMALE-FEMALE connections"""

    edges_w_features_F_F = edges_w_features.loc[
        (edges_w_features["gender_x"] == 0.0) & (edges_w_features["gender_y"] == 0.0)
    ]

    df_F_F = edges_w_features_F_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # creating pivot table
    df_F_F_pivot = df_F_F.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    # adding total 2
    df_F_F_pivot["F_total"] = df_F_F_pivot.sum(axis=1)

    return df_F_F_pivot


# F-M relations:
def pivot_F_M(edges_w_features):
    """Aim of this function is to create pivot table for FEMALE-MALE connections"""
    edges_w_features_F_M = edges_w_features.loc[
        (edges_w_features["gender_x"] == 0.0) & (edges_w_features["gender_y"] == 1.0)
    ]

    df_F_M = edges_w_features_F_M.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # creating pivot table
    df_F_M_pivot = df_F_M.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    # adding total 2
    df_F_M_pivot["M_total"] = df_F_M_pivot.sum(axis=1)

    return df_F_M_pivot


# F_F connections to proportions
def pivot_F_F_proportions(df_F_F_pivot, df_F_M_pivot):
    """From the pivot tables, this function calculates the proportion of FEMALE connections by age, from the Age x FEMALE perspective"""
    df_F_F_proportion = df_F_F_pivot
    # to include all connections of Females
    df_F_F_proportion["T_total"] = df_F_F_pivot["F_total"] + df_F_M_pivot["M_total"]

    df_F_F_proportion = df_F_F_proportion.loc[:, 15.0:50.0].div(
        df_F_F_proportion["T_total"], axis=0
    )
    return df_F_F_proportion


# F_F connections to proportions by generational groups
def F_F_gen_proportions(pivot_F_F_proportions):
    """This function calculetes the FEMALE proportions of the three generational categories for FEMALE users"""

    df_pivot_F_F_proportions = pivot_F_F_proportions

    # creating generations, but because of the smaller sample (ages between 15-50 we declared different ranges
    # (same = +- 5 years, older = +10 to +20, younger = -10 to -20 years old)
    same_gen = {}
    older_gen = {}
    younger_gen = {}

    for age in df_pivot_F_F_proportions.index:
        same_gen[age] = sum(
            df_pivot_F_F_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
        )

    for age in df_pivot_F_F_proportions.index:
        if age <= 40:
            older_gen[age] = sum(
                df_pivot_F_F_proportions.loc[age, age + 10 : min(age + 20, 50)]
            )
        # because data is out of the sample
        else:
            older_gen[age] = 0

    for age in df_pivot_F_F_proportions.index:
        if age >= 25:
            younger_gen[age] = sum(
                df_pivot_F_F_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
            )
        # because data is out of the sample
        else:
            younger_gen[age] = 0

    df_pivot_F_F_proportions["F(x-5:x+5)"] = same_gen.values()
    df_pivot_F_F_proportions["F(x+10:x+20)"] = older_gen.values()
    df_pivot_F_F_proportions["F(x-20:x-10)"] = younger_gen.values()
    return df_pivot_F_F_proportions


# F_M connections to proportions
def pivot_F_M_proportions(df_F_F_pivot, df_F_M_pivot):
    """From the pivot tables, this function calculates the proportion of MALE connections by age, from the Age x FEMALE perspective"""
    df_F_M_proportion = df_F_M_pivot
    # to include all relations of Females
    df_F_M_proportion["T_total"] = df_F_F_pivot["F_total"] + df_F_M_pivot["M_total"]

    df_F_M_proportion = df_F_M_proportion.loc[:, 15.0:50.0].div(
        df_F_M_proportion["T_total"], axis=0
    )
    return df_F_M_proportion


# F_M connections to proportions by generational groups
def F_M_gen_proportions(pivot_F_M_proportions):
    """This function calculetes the MALE proportions of the three generational categories for FEMALE users"""
    df_pivot_F_M_proportions = pivot_F_M_proportions

    # creating generations, but because of the smaller sample (ages between 15-50 we declared different ranges
    # (same = +- 5 years, older = +10 to +20, younger = -10 to -20 years old)
    same_gen = {}
    older_gen = {}
    younger_gen = {}

    for age in df_pivot_F_M_proportions.index:
        same_gen[age] = sum(
            df_pivot_F_M_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
        )

    for age in df_pivot_F_M_proportions.index:
        if age <= 40:
            older_gen[age] = sum(
                df_pivot_F_M_proportions.loc[age, age + 10 : min(age + 20, 50)]
            )
        # because data is out of the sample
        else:
            older_gen[age] = 0

    for age in df_pivot_F_M_proportions.index:
        if age >= 25:
            younger_gen[age] = sum(
                df_pivot_F_M_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
            )
        # because data is out of the sample
        else:
            younger_gen[age] = 0

    df_pivot_F_M_proportions["M(x-5:x+5)"] = same_gen.values()
    df_pivot_F_M_proportions["M(x+10:x+20)"] = older_gen.values()
    df_pivot_F_M_proportions["M(x-20:x-10)"] = younger_gen.values()
    return df_pivot_F_M_proportions


"""Figure 6 step2: we create the data for the (b) chart (Age x male users, same logic as in the previous functions)"""

#: M-F relations
def pivot_M_F(edges_w_features):
    """Aim of this function is to create pivot table for MALE-FEMALE connections"""
    edges_w_features_M_F = edges_w_features.loc[
        (edges_w_features["gender_x"] == 1.0) & (edges_w_features["gender_y"] == 0.0)
    ]

    df_M_F = edges_w_features_M_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # creating pivot table
    df_M_F_pivot = df_M_F.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    # adding total 2
    df_M_F_pivot["F_total"] = df_M_F_pivot.sum(axis=1)

    return df_M_F_pivot


#: M-M relations:
def pivot_M_M(edges_w_features):
    """Aim of this function is to create pivot table for FEMALE-MALE connections"""
    edges_w_features_M_M = edges_w_features.loc[
        (edges_w_features["gender_x"] == 1.0) & (edges_w_features["gender_y"] == 1.0)
    ]

    df_M_M = edges_w_features_M_M.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # creating pivot table
    df_M_M_pivot = df_M_M.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    # adding total 2
    df_M_M_pivot["M_total"] = df_M_M_pivot.sum(axis=1)

    return df_M_M_pivot


# Male - Female relations to proportions
def pivot_M_F_proportions(df_M_F_pivot, df_M_M_pivot):
    """From the pivot tables, this function calculates the proportion of FEMALE connections by age, from the Age x MALE perspective"""
    df_M_F_proportion = df_M_F_pivot
    # to include all relations of Males
    df_M_F_proportion["T_total"] = df_M_F_pivot["F_total"] + df_M_M_pivot["M_total"]

    df_M_F_proportion = df_M_F_proportion.loc[:, 15.0:50.0].div(
        df_M_F_proportion["T_total"], axis=0
    )
    return df_M_F_proportion


def M_F_gen_proportions(pivot_M_F_proportion):

    """This function calculetes the FEMALE proportions of the three generational categories for MALE users"""

    df_pivot_M_F_proportions = pivot_M_F_proportion

    # creating generations, but because of the smaller sample (ages between 15-50 we declared different ranges)
    # (same = +- 5 years, older = +10 to +20, younger = -10 to -20 years old)
    same_gen = {}
    older_gen = {}
    younger_gen = {}

    for age in df_pivot_M_F_proportions.index:
        same_gen[age] = sum(
            df_pivot_M_F_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
        )

    for age in df_pivot_M_F_proportions.index:
        if age <= 40:
            older_gen[age] = sum(
                df_pivot_M_F_proportions.loc[age, age + 10 : min(age + 20, 50)]
            )
        # because data is out of the sample
        else:
            older_gen[age] = 0

    for age in df_pivot_M_F_proportions.index:
        if age >= 25:
            younger_gen[age] = sum(
                df_pivot_M_F_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
            )
        # because data is out of the sample
        else:
            younger_gen[age] = 0

    df_pivot_M_F_proportions["F(x-5:x+5)"] = same_gen.values()
    df_pivot_M_F_proportions["F(x+10:x+20)"] = older_gen.values()
    df_pivot_M_F_proportions["F(x-20:x-10)"] = younger_gen.values()
    return df_pivot_M_F_proportions


# Male - Male relations to proportions
def pivot_M_M_proportions(df_M_M_pivot, df_M_F_pivot):
    """From the pivot tables, this function calculates the proportion of MALE connections by age, from the Age x MALE perspective"""

    df_M_M_proportion = df_M_M_pivot
    # to include all relations of Males
    df_M_M_proportion["T_total"] = df_M_F_pivot["F_total"] + df_M_M_pivot["M_total"]

    df_M_M_proportion = df_M_M_proportion.loc[:, 15.0:50.0].div(
        df_M_M_proportion["T_total"], axis=0
    )
    return df_M_M_proportion


def M_M_gen_proportions(pivot_M_M_proportions):
    """This function calculetes the MALE proportions of the three generational categories for MALE users"""

    df_pivot_M_M_proportions = pivot_M_M_proportions

    # creating generations, but because of the smaller sample (ages between 15-50 we declared different ranges)
    # (same = +- 5 years, older = +10 to +20, younger = -10 to -20 years old)
    same_gen = {}
    older_gen = {}
    younger_gen = {}

    for age in df_pivot_M_M_proportions.index:
        same_gen[age] = sum(
            df_pivot_M_M_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
        )

    for age in df_pivot_M_M_proportions.index:
        if age <= 40:
            older_gen[age] = sum(
                df_pivot_M_M_proportions.loc[age, age + 10 : min(age + 20, 50)]
            )
        # because data is out of the sample
        else:
            older_gen[age] = 0

    for age in df_pivot_M_M_proportions.index:
        if age >= 25:
            younger_gen[age] = sum(
                df_pivot_M_M_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
            )
        # because data is out of the sample
        else:
            younger_gen[age] = 0

    df_pivot_M_M_proportions["M(x-5:x+5)"] = same_gen.values()
    df_pivot_M_M_proportions["M(x+10:x+20)"] = older_gen.values()
    df_pivot_M_M_proportions["M(x-20:x-10)"] = younger_gen.values()
    return df_pivot_M_M_proportions


"""Creating the Figure 6 Chart"""


def creating_figure_six(edges_w_features):
    """This functions uses the previously defined functions for creating the 4 input dataframes, then creates the two charts"""
    # creating plot_input_F_F withe the defined functions
    input_1 = pivot_F_F(edges_w_features)
    input_2 = pivot_F_M(edges_w_features)
    input_3 = pivot_F_F_proportions(input_1, input_2)
    plot_input_F_F = F_F_gen_proportions(input_3)

    # creating plot_input_F_M withe the defined functions
    input_1 = pivot_F_F(edges_w_features)
    input_2 = pivot_F_M(edges_w_features)
    input_3 = pivot_F_M_proportions(input_1, input_2)
    plot_input_F_M = F_M_gen_proportions(input_3)

    # creating plot_input_M_F withe the defined functions
    input_1 = pivot_M_F(edges_w_features)
    input_2 = pivot_M_M(edges_w_features)
    input_3 = pivot_M_F_proportions(input_1, input_2)
    plot_input_M_F = M_F_gen_proportions(input_3)

    # creating plot_input_M_M withe the defined functions
    input_1 = pivot_M_M(edges_w_features)
    input_2 = pivot_M_F(edges_w_features)
    input_3 = pivot_M_M_proportions(input_1, input_2)
    plot_input_M_M = M_M_gen_proportions(input_3)

    # creating the two charts with the 4 plot_input
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sns.lineplot(data=plot_input_F_F, x="AGE_x", y="F(x-5:x+5)", color="red", ax=ax1)
    sns.lineplot(
        data=plot_input_F_F, x="AGE_x", y="F(x+10:x+20)", color="green", ax=ax1
    )
    sns.lineplot(data=plot_input_F_F, x="AGE_x", y="F(x-20:x-10)", color="cyan", ax=ax1)
    sns.lineplot(data=plot_input_F_M, x="AGE_x", y="M(x-5:x+5)", color="blue", ax=ax1)
    sns.lineplot(
        data=plot_input_F_M, x="AGE_x", y="M(x+10:x+20)", color="deeppink", ax=ax1
    )
    sns.lineplot(
        data=plot_input_F_M, x="AGE_x", y="M(x-20:x-10)", color="black", ax=ax1
    )
    ax1.set_ylabel("Proportions")
    ax1.set_xlabel("Age x of Female User")
    ax1.set_title("(a) Proportion of Females friends age")
    ax1.legend(
        [
            "F(x-5:x+5)",
            "F(x+10:x+20)",
            "F(x-20:x-10)",
            "M(x-5:x+5)",
            "M(x+10:x+20)",
            "M(x-20:x-10)",
        ]
    )

    sns.lineplot(data=plot_input_M_F, x="AGE_x", y="F(x-5:x+5)", color="red", ax=ax2)
    sns.lineplot(
        data=plot_input_M_F, x="AGE_x", y="F(x+10:x+20)", color="green", ax=ax2
    )
    sns.lineplot(data=plot_input_M_F, x="AGE_x", y="F(x-20:x-10)", color="cyan", ax=ax2)
    sns.lineplot(data=plot_input_M_M, x="AGE_x", y="M(x-5:x+5)", color="blue", ax=ax2)
    sns.lineplot(
        data=plot_input_M_M, x="AGE_x", y="M(x+10:x+20)", color="deeppink", ax=ax2
    )
    sns.lineplot(
        data=plot_input_M_M, x="AGE_x", y="M(x-20:x-10)", color="black", ax=ax2
    )
    ax2.set_ylabel("Proportions")
    ax2.set_xlabel("Age x of Male User")
    ax2.set_title("(b) Proportion of Male’s friends’ age")
    ax2.legend(
        [
            "F(x-5:x+5)",
            "F(x+10:x+20)",
            "F(x-20:x-10)",
            "M(x-5:x+5)",
            "M(x+10:x+20)",
            "M(x-20:x-10)",
        ]
    )

