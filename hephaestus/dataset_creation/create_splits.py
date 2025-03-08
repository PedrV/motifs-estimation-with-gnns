"""
Create train-test-validate splits to be used with CustomDataset.
"""

import os
import shutil

import numpy as np
import pandas as pd

import torch
from torch_geometric.seed import seed_everything

import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig


seed_everything(42)
rng = np.random.default_rng(42)

TRAIN_PERCENTAGE = 0.7
VALIDATE_PERCENTAGE = 0.2

CAP = 3490 
DATASET_SIZE = 40000

def clean_my_directories():
    """Reset the directories used by the module"""
    shutil.rmtree(hconfig.RAW_SPLITS_DIR, ignore_errors=True)


def prepare_directories(clean=False):
    """Create the directories used by the module"""
    if clean:
        clean_my_directories()
    os.makedirs(hconfig.RAW_SPLITS_DIR, exist_ok=True)


def load_splits_for_dataset(dataset_name):
    r"""Load a splits of test, train and validation for `dataset_name`.

    Parameters
    ----------
    dataset_name : os.PathLike
        Name of the dataset to operate on. Should follow the conventions specified in conventions.txt.

    Returns
    -------
    torch.tensor
        The test indexes.
    torch.tensor
        The train indexes.
    torch.tensor
        The validation indexes.
    """
    p = os.path.join(hconfig.RAW_SPLITS_DIR, dataset_name)
    test_file = os.path.join(p, "test" + "_" + dataset_name + ".pt")
    train_file = os.path.join(p, "train" + "_" + dataset_name + ".pt")
    validate_file = os.path.join(p, "validate" + "_" + dataset_name + ".pt")
    return torch.load(test_file), torch.load(train_file), torch.load(validate_file)


def condition_to_skip_graph(zscores):
    """
    Verify if the `zscores` for a graph verify the condition for the graph to no be valid.

    (1) Have all its `zscores` equal to 0, e.g. a complete graph, a star-graph etc.
    
    (2) Have at least a NaN in its `zscores`, e.g. graph is to small for the wanted patterns.

    :param `np.array(np.float)` `zscores`: List of the z-scores of a graph.
    """
    complete_graph = np.all(list(map(lambda x: x == 0.0, zscores)))
    too_small = np.any(list(map(np.isnan, zscores)))
    return complete_graph or too_small


def _get_splits(valid_graphs_idx, use_cap = True):
    """
    Given a list of valid graph IDs, randomly select a part of them for train, validation and test.

    :param `list(int)` `valid_graphs_idx`: List of valid graph IDs to split.
    """
    if use_cap:
        data_size = min(
            len(valid_graphs_idx), CAP
        )  # equal data size for d vs nd
    else:
        data_size = len(valid_graphs_idx)

    index_train = rng.choice(
        valid_graphs_idx, size=int(np.round(data_size * TRAIN_PERCENTAGE)), replace=False
    )

    possible_indexes_validate = [
        i for i in valid_graphs_idx if i not in index_train
    ]
    index_validate = rng.choice(
        possible_indexes_validate,
        size=int(np.round(data_size * VALIDATE_PERCENTAGE)),
        replace=False,
    )

    possible_indexes_test = [
        i for i in possible_indexes_validate if i not in index_validate
    ]
    index_test = rng.choice(
        possible_indexes_test,
        size=int(np.round(data_size * (1 - (TRAIN_PERCENTAGE + VALIDATE_PERCENTAGE)))),
        replace=False,
    )

    used = len(index_train) + len(index_validate) + len(index_test)
    print(
        f"[create_splits] {len(valid_graphs_idx)-used} graphs left, check if this result makes sense!"
    )
    return index_train, index_validate, index_test


def _get_valid_graphs(file_path):
    """
    Discover the valid graphs to be considered according to `condition_to_skip_graph`.
    The z-scores should be given in a CSV named 'DATASETNAME_labels.csv' where DATASETNAME is the
    dataset name that will have its graphs tested.

    :param `str`|`pathlib.Path` `file_path`: File name (and path) that hold the z-scores of graphs to be tested.
    """
    # Make sure order is the same as reading in custom_dataset.py
    df = pd.read_csv(file_path)
    df = df.sort_values(by="GraphName", ascending=True)

    cnt = 0
    valid_graphs_idx = []
    for i in range(df.shape[0]):
        if condition_to_skip_graph(df.iloc[i, 2:]):
            cnt += 1
            continue
        valid_graphs_idx.append(i)

    return sorted(valid_graphs_idx), cnt, df.shape[0]


def _merge_split_idx(dataset_name, lower):
    r"""Normalize a dataset split indexes (test,train,validation) to start from `lower`.

    Normalize possible discontinous indexes for test, train and validation into a continous
    sequence starting from `lower`.

    Parameters
    ----------
    dataset_name : os.PathLike
        Name of the dataset to operate on. Should follow the conventions specified in conventions.txt.
    lower : int
        The number from which the splits should start from.

    Returns
    -------
    norm_train_idxs : list[int]
    norm_valid_idxs : list[int]
    norm_test_idxs : list[int]

    Raises
    ------
    ValueError
        Because not all indexes were assigned to on split: test, train or validation.

    See Also
    --------
    create_splits.merge_split_idx   
    """
    test_idxs, train_idxs, validate_idxs = load_splits_for_dataset(dataset_name)

    used = len(train_idxs) + len(validate_idxs) + len(test_idxs)
    upper = lower + used
    norm_train_idxs, norm_valid_idxs, norm_test_idxs = [], [], []

    tagged_idx = []
    for i in train_idxs:
        tagged_idx.append((i, "train"))
    for i in validate_idxs:
        tagged_idx.append((i, "valid"))
    for i in test_idxs:
        tagged_idx.append((i, "test"))

    cur = lower
    tagged_idx = sorted(tagged_idx, key=lambda x: x[0])
    for i, tag in tagged_idx:
        if tag == "train":
            norm_train_idxs.append(cur)
        if tag == "valid":
            norm_valid_idxs.append(cur)
        if tag == "test":
            norm_test_idxs.append(cur)
        cur += 1

    if cur != upper:
        raise ValueError(f"Problem with merging splits {cur} != {upper}")

    return norm_train_idxs, norm_valid_idxs, norm_test_idxs


def merge_split_idx(path_to_save):
    r"""Merge the splits created with `split_datasets` under a single dataset.

    Read all the splits created with `split_datasets` and transforms them into 
    splits of an unified dataset, meaning a single index from 0 to the total size
    of all the datasets combined.

    Note: The indexes of graphs will be determined by:

    (1) The increasing lexicographical order of the dataset name they belong to;

    (2) The increasing lexicographical order of the graph name;

    e.g. given 2 datasets DATA1, DATA2, with graphs DATA1:[graph33, graph42], DATA2:[graph1, graph2],
        the ID 0 will belong to graph33 of DATA1, ID 1 to graph42 of DATA1, ID 2 to graph1 of DATA2 ...
        An example of a split would be train:[0,2], validate:[1], test:[3].

    *WARNING*: It is assumed that the order specified earlier is the same as the order in the creation 
    of the dataset in `custom_dataset.CustomDataset`.
    
    :param `os.PathLike` `path_to_save`: Path to save the splits for the complete dataset.
    """
    lower = 0
    full_train, full_validate, full_test = [], [], []
    for d in sorted(os.listdir(hconfig.RAW_SPLITS_DIR)):
        train, validate, test = _merge_split_idx(d, lower)
        lower += len(train) + len(validate) + len(test)

        full_test.append(test)
        full_train.append(train)
        full_validate.append(validate)

    full_test = hutils.flatten_nested_list(full_test)
    full_train = hutils.flatten_nested_list(full_train)
    full_validate = hutils.flatten_nested_list(full_validate)

    torch.save(torch.tensor(full_test), os.path.join(path_to_save, "test.pt"))
    torch.save(torch.tensor(full_train), os.path.join(path_to_save, "train.pt"))
    torch.save(torch.tensor(full_validate), os.path.join(path_to_save, "validate.pt"))


def split_datasets(datasets_label_filepath, full_random=False):
    r"""Split each given dataset into test, train and validation.

    Mode 1:
    Given a list of dataset names of the form `DATASETNAME_labels.csv`, for each dataset,
    go through all their graphs, remove the ones that do not pass the `condition_to_skip_graph` 
    and create splits for train, validation and test for each dataset.

    The splits for each dataset will be stored in `hconfig.RAW_SPLITS_DIR` under a directory
    given by the result of calling `hutils.get_dataset_name` with each dataset name.
    Each directory has 3 files `test_datasetname.pt`, `train_datasetname.pt` and `validation_datasetname.pt`.
    Each dataset has indexes starting from 0 to the size of the dataset.
    
    Mode 2:
    If `full_random` is used, all datasets given by `datasets_label_filepath` will be "merged" and
    will be considered just one big dataset. The splits are done uniformely at random given that
    big dataset.

    The splits are stored like mode 1 but under the name `test_EVERYTHING.pt`, `train_EVERYTHING.pt` and `validation_EVERYTHING.pt`.

    Parameters
    ----------
    datasets_label_filepath : list(str | pathlib.Path)
        File names (with full path) of the datasets to be considered.
    full_random : Bool
        `True` to use the function in Mode 2.

    Notes
    -----
    The order of the splits is given according to point (2) o `merge_split_idx`.
    """
    print("[create_splits] Creating splits ...")
    total_outed_graphs = 0

    if full_random:
        nd_graphs = []
        d_graphs = []
        seen = 0
        for dataset_filepath in sorted(datasets_label_filepath):
            good_to_go, outed_graphs_cnt, tot = _get_valid_graphs(dataset_filepath)
            total_outed_graphs += outed_graphs_cnt

            dataset_name = hutils.get_dataset_name(dataset_filepath, with_extension=True, has_file_identifier=True)
            if dataset_name.startswith("nd"):
                nd_graphs.append(np.array(good_to_go)+seen)
            else:
                d_graphs.append(np.array(good_to_go)+seen)
            seen += tot 
        
        nd_graphs = hutils.flatten_nested_list(nd_graphs, sort=False)
        d_graphs = hutils.flatten_nested_list(d_graphs, sort=False)
        
        picked_nd_train = rng.choice(nd_graphs[:50000], size=int(DATASET_SIZE*0.92), replace=False)
        picked_d_train = rng.choice(d_graphs[:10000], size=int(DATASET_SIZE*0.08), replace=False)
        available_train_graphs = np.hstack([picked_nd_train, picked_d_train])
        train, _, _ = _get_splits(available_train_graphs, use_cap=False)

        picked_nd_val = rng.choice(nd_graphs[50000:], size=int(DATASET_SIZE*0.92), replace=False)
        picked_d_val = rng.choice(d_graphs[10000:], size=int(DATASET_SIZE*0.08), replace=False)
        available_val_graphs = np.hstack([picked_nd_val, picked_d_val])
        _, validate, test = _get_splits(available_val_graphs, use_cap=False)

        print(f"[create_splits] Splits train {train.shape} graphs.")
        print(f"[create_splits] Splits test {test.shape} graphs.")
        print(f"[create_splits] Splits validate {validate.shape} graphs.")

        p = os.path.join(hconfig.RAW_SPLITS_DIR, "EVERYTHING")
        os.makedirs(p, exist_ok=True)
        torch.save(torch.tensor(test), os.path.join(p, "test_EVERYTHING.pt"))
        torch.save(torch.tensor(train), os.path.join(p, "train_EVERYTHING.pt"))
        torch.save(torch.tensor(validate), os.path.join(p, "validate_EVERYTHING.pt"))

    else:
        for dataset_filepath in sorted(datasets_label_filepath):
            good_to_go, outed_graphs_cnt, _ = _get_valid_graphs(dataset_filepath)
            train, validate, test = _get_splits(good_to_go)
            total_outed_graphs += outed_graphs_cnt

            dataset_name = hutils.get_dataset_name(dataset_filepath, with_extension=True, has_file_identifier=True)

            p = os.path.join(hconfig.RAW_SPLITS_DIR, dataset_name)
            os.makedirs(p, exist_ok=True)
            torch.save(torch.from_numpy(test), os.path.join(p, "test" + "_" + dataset_name + ".pt"))
            torch.save(torch.from_numpy(train), os.path.join(p, "train" + "_" + dataset_name + ".pt"))
            torch.save(torch.from_numpy(validate), os.path.join(p, "validate" + "_" + dataset_name + ".pt"))

    print(f"[create_splits] Skipped {total_outed_graphs} graphs.")
