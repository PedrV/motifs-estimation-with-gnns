"""
Organize and prepare external, most of the times real, graphs to be used internally.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from torch_geometric import datasets as pygdata
from torch_geometric import utils as pygutils

import requests

import hephaestus.utils.load_general_config as hconfig


PROCESSED_DATA_DIR = Path(hconfig.TEST_DIR) / "processed"

SOURCE_WEBPAGE = "https://networkrepository.com/"


def _to_number(n):
    num_to_multiply = 1

    if isinstance(n, (int, float)):
        return n

    if "K" in n:
        num_to_multiply = 1000
        n = n[:-1]
    elif "M" in n:
        num_to_multiply = 1000000
        n = n[:-1]

    return float(n) * num_to_multiply


def _get_stats_from_webpage(network_name):
    webpagefile = SOURCE_WEBPAGE + "/" + network_name + ".php"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0",
    }

    r = requests.get(webpagefile, headers=headers)
    c = r.content
    try:
        cc = pd.read_html(c)[1]
    except Exception:
        return -1, -1

    nodes, edges = cc.iloc[0:2, 1]

    return _to_number(nodes), _to_number(edges)


def transform_network(source_file, destination_file, network_name):
    r"""Transform a real network into the format desired.

    Given a network under `source_file`, transform it to the format
    used in Gtrie and save it to `destination_file`.
    Additionaly, check the number of edges and nodes against the number reported in the source website.

    Parameters
    ----------
    source_file : os.PathLike
        The fully expanded path of the file that contains the network to be used.
    destination_file : os.PathLike
        The fully expanded path of where the transformed network will be saved.
    network_name : str
        The name of the network as presented in the original website

    Notes
    -----
    All graphs will be read and saved as simple undirected graphs (without self-loops).
    """
    nodes, edges = -1, -1
    is_graph_ml = ".graphml" in network_name

    if not is_graph_ml:
        network_name = network_name.rsplit(".")[0]
        nodes, edges = _get_stats_from_webpage(network_name)

    if is_graph_ml:
        g = nx.Graph(
            nx.read_graphml(
                source_file,
                force_multigraph=False,
            ).to_undirected()
        )
    else:
        graph_df = pd.read_csv(
            source_file,
            sep=" ",
            header=None,
            usecols=[0, 1],
            names=["source", "target"],
            comment="%",
        )
        g = nx.from_pandas_edgelist(
            graph_df.iloc[:, 0:2],
            source="source",
            target="target",
            create_using=nx.Graph(),
        )

    g.remove_edges_from(nx.selfloop_edges(g))

    mapping = dict(zip(sorted(list(g.nodes())), range(1, g.number_of_nodes() + 1)))
    g = nx.relabel_nodes(g, mapping)

    bad = False
    if nodes == -1 or edges == -1:
        print("Failed check on remote website!")
        print(f"Local has {g.number_of_nodes()} nodes, and {g.number_of_edges()} edges")
        bad = True

    if not bad and nodes != g.number_of_nodes():
        print(
            f"On website has {nodes} nodes, after transform has {g.number_of_nodes()} nodes"
        )
    if not bad and edges != g.number_of_edges():
        print(
            f"On website has {edges} edges, after transform has {g.number_of_edges()} edges"
        )

    df = nx.to_pandas_edgelist(g)
    df = df.iloc[:,:2]
    df["edge_attrib"] = np.ones((df.shape[0],), dtype=int)
    df.to_csv(
        destination_file,
        sep=" ",
        index=False,
        header=False,
    )


def transform_network_pyg(dataset_name, specific_name, destination_file):
    r"""Transform a real network from the PyG library into the format desired.

    Given a network under `dataset_name` from PyG, transform it to the format
    used in Gtrie and save it to `destination_file`. 
    The `specific_name` is used to select the network from the ones available 
    in `dataset_name`.

    Parameters
    ----------
    dataset_name : str
        Name of a dataset in PyG.
    specific_name : str
        The name of a network inside the selected dataset.
    destination_file : os.PathLike
        The fully expanded path of where the transformed network will be saved.

    Notes
    -----
    All graphs will be read and saved as simple undirected graphs (without self-loops).
    """
    root = PROCESSED_DATA_DIR.parent
    if dataset_name == "citation_full":
        data = pygdata.CitationFull(root, specific_name, to_undirected=True)
    elif dataset_name == "coauthor":
        data = pygdata.Coauthor(root, specific_name)
    else:
        raise NotImplemented("That dataset is not supported!")

    g = nx.from_scipy_sparse_array(
        pygutils.to_scipy_sparse_matrix(data.edge_index),
        create_using=nx.Graph(),
    )

    mapping = dict(zip(sorted(list(g.nodes())), range(1, g.number_of_nodes() + 1)))
    g = nx.relabel_nodes(g, mapping)

    print(
        print(f"Local has {g.number_of_nodes()} nodes, and {g.number_of_edges()} edges")
    )

    df = nx.to_pandas_edgelist(g)
    df = df.iloc[:,:2]
    df["edge_attrib"] = np.ones((df.shape[0],), dtype=int)
    df.to_csv(
        destination_file,
        sep=" ",
        index=False,
        header=False,
    )


def process(categories_to_skip, raw_data_dir):
    r"""Transform all graphs under the specified directory to the Gtrie format.

    All graphs in `raw_data_dir` will be processed to match the format used by Gtrie.
    All graphs of all categories in `categories_to_skip` will not be considered.
    We assume that each category has its own directory under `raw_data_dir` and each graph
    is in its separate folder.

    The result will be saved in `PROCESSED_DATA_DIR`, following the same file structure of the raw data,
    yielding PROCESSED_DATA_DIR/category1/graph1/, PROCESSED_DATA_DIR/category1/graph2/ ...

    Parameters
    ----------
    categories_to_skip : list(str)
        Categories that will be skipped when processing the graphs
    raw_data_dir :  os.PathLike
        Full expanded path of the directory where the graphs are stored.

    See Also
    --------
    transform_network : One call per graph file. This function does the actual transformations.
    """
    raw_data_dir = Path(raw_data_dir)
    for category in sorted(os.listdir(raw_data_dir)):
        path_to_category_raw = raw_data_dir / category
        path_to_category_processed = PROCESSED_DATA_DIR

        if category in categories_to_skip:
            continue

        for network_dir in sorted(os.listdir(path_to_category_raw)):
            if ".zip" in network_dir or ".gz" in network_dir or ".bz2" in network_dir:
                continue

            pyg = False
            if "pyg-ones" in network_dir:
                pyg = True

            for network_name in sorted(os.listdir(path_to_category_raw / network_dir)):
                if (
                    "readme" in network_name.lower()
                    or "info.txt" in network_name.lower()
                    or ".nodes" in network_name
                    or "meta." in network_name
                    or ".json" in network_name
                ):
                    continue

                new_network_name = category + "@" + network_name.rsplit(".")[0] + ".csv"

                source_file = path_to_category_raw / network_dir / network_name
                destination_file = path_to_category_processed / new_network_name

                if pyg:
                    pyg_dataset_name = network_name.split("-")[0]
                    pyg_specific_name = network_name.split("-")[1]
                    transform_network_pyg(
                        pyg_dataset_name, pyg_specific_name, destination_file
                    )
                else:
                    transform_network(source_file, destination_file, network_name)

                print("Done", network_name)
                print()


def prepare_next_stage(final_dir):
    r"""Prepare the next stage of the pipeline.

    Make the necessary steps to allow subsequent stages to run properly.
    Current functionality involves only moving files to `final_dir`.
    Does nothing if files already exist in `final_dir`.

    Parameters
    ----------
    final_dir : os.PathLike
        Full path of directory where the processed graph files need to be for next stages.

    Notes
    -----
    This is a move operation, not a copy operation.
    The synthetic module does not have this functionality explicit. Things are done implicitly.
    """
    final_dir = Path(final_dir)
    for f in sorted(os.listdir(PROCESSED_DATA_DIR)):
        if not os.path.exists(final_dir / f):
            shutil.move(PROCESSED_DATA_DIR / f, final_dir)
