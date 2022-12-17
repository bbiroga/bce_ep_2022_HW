"""Preliminary functions, which were taken out from the data_viz.py file"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

"""Preliminary Functions for Figure 3 charts:
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


"""Preliminary Functions for Figure 5 charts:
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
