"""
Create a dataset based on PyG InMemoryDataset to train models.
"""

import os
import shutil
from pathlib import Path

import networkx as nx
import pandas as pd

import tqdm

import torch
import torch_geometric.data as pyg_data

from torch_geometric.utils import from_scipy_sparse_matrix

import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig

from hephaestus.dataset_creation import create_splits


complete_dataset_dir = Path(hconfig.COMPLETE_DATASET_DIR)


def clean_my_directories():
    """Reset the directories used by the module"""
    shutil.rmtree(complete_dataset_dir, ignore_errors=True)


def prepare_directories(clean=False):
    """Create the directories used by the module"""
    if clean:
        clean_my_directories()
    Path(complete_dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(complete_dataset_dir / "processed").mkdir(parents=True, exist_ok=True)


class CustomDataset(pyg_data.InMemoryDataset):
    def __init__(
        self,
        root_data,
        datasets_name,
        final_dataset_name="MYDATA",
        has_splits=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._has_splits = has_splits
        self._my_name = final_dataset_name

        self._unified_dataset_name = "unified_data.pt"
        self._processed_files = [
            self._unified_dataset_name,
            "train.pt",
            "validate.pt",
            "test.pt",
        ]

        self._graphs_dir = hconfig.GRAPHS_DIR
        self._labels_dir = hconfig.LABEL_DIR
        self._features_dir = hconfig.FEATURES_DIR

        # Dir where the processed data will be, related with InMemoryDataset
        self._unified_data_dir = os.path.join(root_data, "processed")

        self._datasets = sorted(
            datasets_name
        )

        # Triggers processed() if processed_file_names() returns False
        super().__init__(root_data, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(
            os.path.join(self._unified_data_dir, self._unified_dataset_name)
        )

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    @property
    def number_of_edges(self):
        return self._number_of_edges

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self._has_splits:
            return self._processed_files
        return [self._unified_dataset_name]

    @property
    def original_datasets_names(self):
        return self._datasets

    @property
    def has_splits(self):
        return self._has_splits

    def _get_raw_splits(self):
        r"""Get splits for each dataset in `self._datasets`.

        Split each dataset in `self._datasets` for test, train and validation under the
        constraint of local split.

        See Also
        --------
        create_splits.split_datasets
        """
        required_files = [
            os.path.join(self._labels_dir, d + "_labels.csv") for d in self._datasets
        ]
        if self._my_name == "EVERYTHING":
            create_splits.split_datasets(required_files, full_random=True)
        else:
            create_splits.split_datasets(required_files)

    def get_idx_split(self):
        r"""Get the test, train and validation splits for the current dataset.

        If there is no splits stored in disk, it will call `create_splits.merge_split_idx` in
        order to merge existing local/raw splits into one. It is guaranteed that such local/raw
        splits exist because they are created when `process` is automatically called at the
        initalization of the class instance.

        Returns
        -------
        dict(torch.tensor) : Dictionary with 3 entries named "train", "test", "validate".
        Each entry is a `torch.tensor`.

        See Also
        --------
        create_splits.merge_split_idx
        """
        if (
            not os.path.exists(os.path.join(self._unified_data_dir, "train.pt"))
            or not os.path.exists(os.path.join(self._unified_data_dir, "validate.pt"))
            or not os.path.exists(os.path.join(self._unified_data_dir, "test.pt"))
        ):
            create_splits.merge_split_idx(self._unified_data_dir)

        train = torch.load(os.path.join(self._unified_data_dir, "train.pt"))
        validate = torch.load(os.path.join(self._unified_data_dir, "validate.pt"))
        test = torch.load(os.path.join(self._unified_data_dir, "test.pt"))
        return {"train": train, "validate": validate, "test": test}

    def process(self):
        data_list = []
        total_nodes, total_edges = 0, 0

        if self._has_splits:
            self._get_raw_splits()

        j = -1
        for dataset_name in tqdm.tqdm(self._datasets):
            # One file has labels for all graphs of a dataset
            # Secures condition (2) of create_splits.split_dataset.
            labels = pd.read_csv(
                os.path.join(self._labels_dir, dataset_name + "_labels.csv")
            ).sort_values(by="GraphName", ascending=True)

            if self._has_splits and self._my_name != "EVERYTHING":
                splt_test, splt_train, splt_valid = (
                    create_splits.load_splits_for_dataset(
                        dataset_name,
                    )
                )
            elif self._has_splits and self._my_name == "EVERYTHING":
                splt_test, splt_train, splt_valid = (
                    create_splits.load_splits_for_dataset("EVERYTHING")
                )  # All data is under EVERYTHING

            if self._my_name != "EVERYTHING":
                j = -1  # Reset j if partial splits are used

            # Secures condition (1) of create_splits.split_dataset.
            for f in sorted(os.listdir(self._graphs_dir)):
                if dataset_name not in f:
                    continue

                j += 1
                if self._has_splits and not (
                    j in splt_test or j in splt_train or j in splt_valid
                ):
                    continue

                graph_name = hutils.get_graph_name(f, with_extension=True)
                features_for_graph_file = (
                    dataset_name + "@" + graph_name + "_features.csv"
                )

                # Order does not matter here since we have one file per graph and files come sorted
                features = pd.read_csv(
                    os.path.join(
                        self._features_dir, dataset_name, features_for_graph_file
                    )
                )

                df = pd.read_csv(
                    os.path.join(self._graphs_dir, f),
                    sep=" ",
                    header=None,
                    usecols=[0, 1],
                    names=["source", "target"],
                )
                graph = nx.from_pandas_edgelist(
                    df.iloc[:, 0:2],
                    source="source",
                    target="target",
                    create_using=nx.Graph(),
                )

                # Skip graph name and old target
                y = labels.loc[
                    labels["GraphName"] == graph_name,
                    ~labels.columns.isin(["GraphName", "OldTarget"]),
                ].to_numpy()

                if create_splits.condition_to_skip_graph(y):
                    print("Skipping id {0} from {1}".format(graph_name, dataset_name))
                    continue  # This should make masking in train and eval redundant

                # Skip node names and graph name
                x = features.iloc[:, :-2].to_numpy()

                y = torch.tensor(y, dtype=torch.float32)
                x = torch.tensor(x, dtype=torch.float32)
                edge_index, _ = from_scipy_sparse_matrix(nx.adjacency_matrix(graph))

                if x.size()[0] != graph.number_of_nodes():
                    raise ValueError(
                        f"In {f}\nNX graph has {graph.number_of_nodes()} nodes, and X matrix {x.size()[0]}"
                    )
                if edge_index.size()[1] // 2 != graph.number_of_edges():
                    raise ValueError(
                        f"In {f}\nNX graph has {graph.number_of_edges()} edges, and X matrix {edge_index.size()[1]//2}"
                    )

                total_nodes += x.size()[0]
                total_edges += edge_index.size()[1]

                data_list.append(pyg_data.Data(x=x, edge_index=edge_index, y=y))

        # It DOES maintain the order from data_list
        data, slices = self.collate(data_list)
        torch.save(
            (data, slices),
            os.path.join(self._unified_data_dir, self._unified_dataset_name),
        )

        del data, slices

        self.data, self.slices = torch.load(
            os.path.join(self._unified_data_dir, self._unified_dataset_name)
        )

        if self._has_splits:
            self.get_idx_split()

    def __str__(self):
        return "{1}({0})".format(len(self), self._my_name)

    def __repr__(self):
        return "{1}({0})".format(len(self), self._my_name)
