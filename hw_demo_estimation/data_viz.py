"""Visualization function examples for the homework project"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_degree_distribution(G):
    """Plot a degree distribution of a graph

    TODO: log-log binning! To understand this better, check out networksciencebook.com
    """
    plot_df = (
        pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
    )
    plot_df.columns = ["k", "count"]
    plot_df["log_k"] = np.log(plot_df["k"])
    plot_df["log_count"] = np.log(plot_df["count"])
    fig, ax = plt.subplots()

    ax.scatter(plot_df["k"], plot_df["count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.suptitle("Mutual Degree Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("count_k")


def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[["AGE", "gender"]].copy(deep=True).astype(float)
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.histplot(data=plot_df, x="AGE", hue="gender", bins=np.arange(0, 45, 5) + 15)


"""Functions for Figure 3 charts:
* Degree Centrality
* Neighbor Connectivity
* Triadic closure - local clustering
"""


def plot_node_degree_by_gender(nodes, G):
    """Plot the average of node degree across age and gender"""
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    nodes_w_degree["gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )

    ax = sns.lineplot(
        data=plot_df, x="AGE", y="degree", hue="gender", palette=["red", "blue"]
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Degree")
    ax.set_title("(a) Degree Centrality")


def plot_node_neighbor_conn_by_gender(nodes, G):
    """Plots neihgbor connectivity: the average degree of neighbors of a specific user"""
    nodes_w_neighbor_conn = nodes

    # using the inbuilt nx.average_neighbor_degree function, and mapping it to each node
    nodes_w_neighbor_conn = nodes_w_neighbor_conn.assign(
        neighbor_conn=nodes_w_neighbor_conn.user_id.map(nx.average_neighbor_degree(G))
    )
    nodes_w_neighbor_conn["gender"].replace(
        [0.0, 1.0], ["Female", "Male"], inplace=True
    )

    plot_df = (
        nodes_w_neighbor_conn.groupby(["AGE", "gender"])
        .agg({"neighbor_conn": "mean"})
        .reset_index()
    )
    ax = sns.lineplot(
        data=plot_df, x="AGE", y="neighbor_conn", hue="gender", palette=["red", "blue"]
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Neighbor Connectivity")
    ax.set_title("(b) Neighbor Connectivity")


def plot_node_triadic_clos_by_gender(nodes, G):
    """Plots triadic cluster: the local clustering coefficient  (cc) of each user"""
    nodes_w_triadic_clos = nodes
    nodes_w_triadic_clos = nodes_w_triadic_clos.assign(
        triadic_clos=nodes_w_triadic_clos.user_id.map(nx.clustering(G))
    )
    nodes_w_triadic_clos["gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)

    plot_df = (
        nodes_w_triadic_clos.groupby(["AGE", "gender"])
        .agg({"triadic_clos": "mean"})
        .reset_index()
    )
    ax = sns.lineplot(
        data=plot_df, x="AGE", y="triadic_clos", hue="gender", palette=["red", "blue"]
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("cc")
    ax.set_title("(c) Triadic Closure")


def creating_figure_three(nodes, G):
    """This function merges the previous three functions, making it possible to plot the three charts for Figure 3 together"""
    # (a) Degree Centrality
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    nodes_w_degree["gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)
    plot_input_degree_centrality = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )

    # (b) Neighbor Connectivity
    nodes_w_neighbor_conn = nodes

    # using the inbuil nx.average_neighbor_degree function, and mapping it to each node
    nodes_w_neighbor_conn = nodes_w_neighbor_conn.assign(
        neighbor_conn=nodes_w_neighbor_conn.user_id.map(nx.average_neighbor_degree(G))
    )
    nodes_w_neighbor_conn["gender"].replace(
        [0.0, 1.0], ["Female", "Male"], inplace=True
    )

    plot_input_neighbor_conn = (
        nodes_w_neighbor_conn.groupby(["AGE", "gender"])
        .agg({"neighbor_conn": "mean"})
        .reset_index()
    )

    # (c) Triadic Closure
    nodes_w_triadic_clos = nodes
    nodes_w_triadic_clos = nodes_w_triadic_clos.assign(
        triadic_clos=nodes_w_triadic_clos.user_id.map(nx.clustering(G))
    )
    nodes_w_triadic_clos["gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)

    plot_input_triadic_clos = (
        nodes_w_triadic_clos.groupby(["AGE", "gender"])
        .agg({"triadic_clos": "mean"})
        .reset_index()
    )

    # plotting the three charts
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    sns.lineplot(
        data=plot_input_degree_centrality,
        x="AGE",
        y="degree",
        hue="gender",
        palette=["red", "blue"],
        ax=ax1,
    )
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Age")
    ax1.set_title("(a) Degree Centrality")

    sns.lineplot(
        data=plot_input_neighbor_conn,
        x="AGE",
        y="neighbor_conn",
        hue="gender",
        palette=["red", "blue"],
        ax=ax2,
    )
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Neighbor Connectivity")
    ax2.set_title("(b) Neighbor Connectivity")

    sns.lineplot(
        data=plot_input_triadic_clos,
        x="AGE",
        y="triadic_clos",
        hue="gender",
        palette=["red", "blue"],
        ax=ax3,
    )
    ax3.set_xlabel("Age")
    ax3.set_ylabel("cc")
    ax3.set_title("(c) Triadic Closure")


"""Functions for Figure 5 charts:
Strenght of Social tie between:
* (a) Total population
* (b) M-M pairs
* (c) F-F pairs
* (d) M-F pairs
"""


def plot_age_relations_heatmap(edges_w_features):
    """Plot a heatmap that represents the distribution of edges"""
    # Original version of the heatmap, used as a blueprint for following versions with filtered dataframes by gender
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)

    ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
    ax.invert_yaxis()
    ax.set_xlabel("Age")
    ax.set_ylabel("Age")
    ax.set_title("(a) #connections per pair")


def plot_age_relations_heatmap_M_M(edges_w_features):
    """Plot a heatmap that represents the distribution of edges for Male-Male pairs"""

    # we filter out the input dataframe with only Male-Male pairs
    edges_w_features_M_M = edges_w_features.loc[
        (edges_w_features["gender_x"] == 1.0) & (edges_w_features["gender_y"] == 1.0)
    ]

    plot_df = edges_w_features_M_M.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # plot_df_w_w = plot_df.loc[(0, 0)].reset_index() -> not needed
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
    ax.invert_yaxis()
    ax.set_xlabel("Age (Male)")
    ax.set_ylabel("Age (Male)")
    ax.set_title("(B) #connections per M-M pair")


def plot_age_relations_heatmap_F_F(edges_w_features):
    """Plot a heatmap that represents the distribution of edges for Female-Female pairs"""

    # we filter out the input dataframe with only Female-Female pairs
    edges_w_features_F_F = edges_w_features.loc[
        (edges_w_features["gender_x"] == 0.0) & (edges_w_features["gender_y"] == 0.0)
    ]

    plot_df = edges_w_features_F_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
    ax.invert_yaxis()
    ax.set_xlabel("Age (Female)")
    ax.set_ylabel("Age (Female)")
    ax.set_title("(c) #connections per F-F pair")


def plot_age_relations_heatmap_M_F(edges_w_features):
    """Plot a heatmap that represents the distribution of edges for Male-Female pairs"""

    # we filter out the input dataframe with only the non-mathcing gender pairs
    edges_w_features_M_F = edges_w_features.loc[
        (edges_w_features["gender_x"] != edges_w_features["gender_y"])
    ]

    plot_df = edges_w_features_M_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
    ax.invert_yaxis()
    ax.set_xlabel("Age (Male)")
    ax.set_ylabel("Age (Female)")
    ax.set_title("(d) #connections per M-F pair")

    sns.heatmap(plot_df_heatmap_logged)


# creating figure 5

"""Functions for Figure 5 charts:
Strenght of Social tie between:
* (a) Total population
* (b) M-M pairs
* (c) F-F pairs
* (d) M-F pairs
"""


def creating_figure_five(edges_w_features):
    """This functions merges 4 functions to create the inputs and plot the heatmaps for Figure 5"""

    # (a) #calls per pair
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_input_df_heatmap_logged = np.log(plot_df_heatmap + 1)

    # (b) #calls per M-M pair
    # we filter out the input dataframe with only Male-Male pairs
    edges_w_features_M_M = edges_w_features.loc[
        (edges_w_features["gender_x"] == 1.0) & (edges_w_features["gender_y"] == 1.0)
    ]

    plot_df = edges_w_features_M_M.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    # plot_df_w_w = plot_df.loc[(0, 0)].reset_index() -> not needed
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_input_M_M_df_heatmap_logged = np.log(plot_df_heatmap + 1)

    # (c) #calls per F-F pair
    # we filter out the input dataframe with only Female-Female pairs
    edges_w_features_F_F = edges_w_features.loc[
        (edges_w_features["gender_x"] == 0.0) & (edges_w_features["gender_y"] == 0.0)
    ]

    plot_df = edges_w_features_F_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_input_F_F_df_heatmap_logged = np.log(plot_df_heatmap + 1)

    # (d) #calls per M-F pair
    # we filter out the input dataframe with only the non-mathcing gender pairs
    edges_w_features_M_F = edges_w_features.loc[
        (edges_w_features["gender_x"] != edges_w_features["gender_y"])
    ]

    plot_df = edges_w_features_M_F.groupby(
        ["gender_x", "gender_y", "AGE_x", "AGE_y"]
    ).agg({"smaller_id": "count"})
    plot_df_heatmap = plot_df.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_input_M_F_df_heatmap_logged = np.log(plot_df_heatmap + 1)

    # creating the 4 charts
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(18, 8))
    sns.heatmap(plot_input_df_heatmap_logged, cmap="jet", ax=ax1)
    ax1.invert_yaxis()
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Age")
    ax1.set_title("(a) #connections per pair")

    sns.heatmap(plot_input_M_M_df_heatmap_logged, cmap="jet", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Age (Male)")
    ax2.set_ylabel("Age (Male)")
    ax2.set_title("(B) #connections per M-M pair")

    sns.heatmap(plot_input_F_F_df_heatmap_logged, cmap="jet", ax=ax3)
    ax3.invert_yaxis()
    ax3.set_xlabel("Age (Female)")
    ax3.set_ylabel("Age (Female)")
    ax3.set_title("(c) #connections per F-F pair")

    sns.heatmap(plot_input_M_F_df_heatmap_logged, cmap="jet", ax=ax4)
    ax4.invert_yaxis()
    ax4.set_xlabel("Age (Male)")
    ax4.set_ylabel("Age (Female)")
    ax4.set_title("(d) #connections per M-F pair")

    plt.tight_layout()
