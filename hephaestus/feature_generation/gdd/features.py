"""
Create features for nodes of graphs.
"""

import os
from pathlib import Path

import tqdm

import numpy as np
import pandas as pd
import networkx as nx


import deprecation
import torch
from torch_geometric.utils import to_networkx


import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig

FEATURES_DIR = Path(hconfig.FEATURES_DIR)


def gdd(g, size):
    r"""Create a feature vector for each node of `g` using GDD up to `size`.

    Calculate the Graphlet Degree Distribution (GDD) up to `size` for a graph `g`.
    The result will be scaled from 0-1 to ensure that it is comparable across graphs
    with different sizes.

    The `size` parameter should be >= 2.

    Parameters
    ----------
    g : networkx.Graph
        A networkx Graph to work on.
    size : int
        Maximum size of the GDD to be calculated.

    Returns
    -------
    numpy.array : An array of dimensions `size`-1,`g.number_of_nodes()` where each
    row corresponds to the features of each node.
    """
    if size > 2:
        raise ValueError(f"Can only do up to size 2, requested {size}")
    gdd_per_node_vector = []

    max_degree = max(list(g.degree()), key=lambda x: x[1])[1]
    for _ in range(2, size + 1):
        for n in sorted(g.nodes):
            gdd_per_node_vector.append([g.degree(n) / max_degree])
    return np.array(gdd_per_node_vector)


def from_csv_graph(graph_file, type="gdd", **kwargs):
    r"""Calculate features for each node of `graph_file`.

    Given a graph specified by `graph_file`, calculate features for each node of said graph.
    The feature type calculated will be given by `type`.
    Any extra arguments that a type of feature may need should be given as `**kwargs`.

    Features are calculated in increasing order of node id. Hence, the first row corresponds to
    the features of the first node, the second row to the features of the second node and so on.

    Currently only 'gdd' is allowd in `type`.

    Parameters
    ----------
    graph_file : str
        Full expanded path to the graph to be used.
    type : str
        Type of features to be calculated

    Returns
    -------
    numpy.array : An array of dimensions X,`g.number_of_nodes()` where each
        row corresponds to the features of each node and X is the size of the features of each node.
    list(int) : List of node ids.

    Other Parameters
    ----------------
    **kwargs : dict
        Extra arguments based on `type`.

    Raises
    ------
    NotImplementedError
        If using a `type` not supported

    See Also
    --------
    features.gdd : Function that generates features using the GDD.

    Notes
    -----
    The graph used should be stored in a format suitable to be used with Gtrie.
    """
    graph_df = pd.read_csv(
        graph_file, sep=" ", header=None, usecols=[0, 1], names=["source", "target"]
    )
    g = nx.from_pandas_edgelist(
        graph_df.iloc[:, 0:2], source="source", target="target", create_using=nx.Graph()
    )

    if type == "gdd":
        sizes = kwargs["sizes"]
        return gdd(g, sizes), sorted(g.nodes)
    else:
        raise NotImplementedError("Only support for GDD is available")


def from_general_dataset(dataset_name, dataset_dir, type="gdd", **kwargs):
    r"""Create features for all graphs in the `dataset_name` dataset.

    Go to all graphs in the `dataset_dir`, select only the ones that belong to `dataset_name`
    and create features of type `type` for each node of each graph.

    Each graph of the dataset will have its one CSV file with the features of its nodes.
    The results will be saved in `FEATURES_DIR`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to consider.
    dataset_dir : pathlib.Path
        Full expanded directory of where the graphs for `dataset_name` are stored.
    type : str
        Type of the features to generate.

    Other Parameters
    ----------------
    **kwargs : dict
        Extra arguments based on `type`.

    See also
    --------
    features.from_csv_graph : Does one call for each graph to this function to actually calculate stuff.
    Hence, `type` and `**kwargs` are directly passed to this function.

    Notes
    -----
    Ideally, `dataset_name` should be obtained with the functions from `utils.general_utils`
    or by reading directly from `config.ini`.
    """
    dataset_features_dir = FEATURES_DIR / dataset_name
    dataset_features_dir.mkdir(exist_ok=True, parents=True)

    for f in tqdm.tqdm(
        sorted(os.listdir(dataset_dir)), desc="Searching all graphs in given directory"
    ):
        if dataset_name not in f:
            continue

        name = hutils.get_graph_name(f, with_extension=True)
        feature_for_graph, node_names = from_csv_graph(
            os.path.join(dataset_dir, f), type, **kwargs
        )

        df = pd.DataFrame(
            feature_for_graph,
            columns=["G" + str(i) for i in range(feature_for_graph.shape[1])],
        )
        df["NodeName"] = node_names
        df["GraphName"] = name  # Uses pandas auto-repeat
        df.to_csv(
            dataset_features_dir / (dataset_name + "@" + name + "_features.csv"),
            index=False,
        )


@deprecation.deprecated()
def from_pyg_dataset(size, pyg_data_name, dataset_name=None):
    """Give GDD features to nodes of a PyG dataset"""
    if dataset_name is None:
        pyg_dataset = eval(
            "torch_geometric.datasets." + pyg_data_name + "(" + "pygdata_dir" + ")"
        )
        dataset_name = pyg_data_name
    else:
        pyg_dataset = eval(
            "torch_geometric.datasets."
            + pyg_data_name
            + "("
            + "pygdata_dir"
            + ","
            + "dataset_name"
            + ")"
        )

    dataset_features = []
    nodes_per_graph = []
    for graph in pyg_dataset:
        nx_graph = to_networkx(graph, to_undirected=(not graph.is_directed()))
        gdd_vector = gdd(nx_graph, size)
        dataset_features.append(gdd_vector)
        nodes_per_graph.append(nx_graph.number_of_nodes())

    # Columns for node index within the graph and graph index within the dataset
    node_of_graph_index_t = torch.Tensor(0, sum(nodes_per_graph))
    graph_index_t = torch.Tensor(0, sum(nodes_per_graph))
    torch.cat(
        [torch.arange(num_nodes) for num_nodes in nodes_per_graph],
        out=node_of_graph_index_t,
    )
    torch.cat(
        [torch.full((num_nodes,), ii) for ii, num_nodes in enumerate(nodes_per_graph)],
        out=graph_index_t,
    )

    gdd_size = dataset_features[0].shape[1]
    dataset_features = hutils.flatten_nested_list(dataset_features, sort=False)
    df = pd.DataFrame(dataset_features, columns=["G" + str(i) for i in range(gdd_size)])
    df["NodeName"] = node_of_graph_index_t
    df["GraphName"] = graph_index_t
    df.to_csv(str(FEATURES_DIR / dataset_name) + "_features.csv", index=False)
