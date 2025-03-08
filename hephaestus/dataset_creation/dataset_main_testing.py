"""Stand-alone script to test dataset and split creation tests"""

import os
import sys
from pathlib import Path

import time

sys.path.insert(0, os.getcwd())

import hephaestus.dataset_creation.custom_dataset as pgeometric_dataset
import hephaestus.dataset_creation.create_splits as splits

# import hephaestus.dataset_creation.create_splits as splits
import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils

SEED = 42

complete_dataset_dir = Path(hconfig.COMPLETE_DATASET_DIR)

splits.prepare_directories(clean=True)

# Synthetic dataset creation
dataset = pgeometric_dataset.CustomDataset(
    str(complete_dataset_dir)+"_nd",
    hutils.flatten_nested_list(
        [
            list(hconfig.NDETERMINISTIC_DATA.values()),
        ],
        sort=True,
    ),
    final_dataset_name="ND",
    has_splits=True,
)
print(len(dataset))
print(dataset.get_idx_split())

time.sleep(1)

###############################################################

splits.prepare_directories(clean=True)

dataset2 = pgeometric_dataset.CustomDataset(
    str(complete_dataset_dir)+"_d",
    hutils.flatten_nested_list(
        [
            list(hconfig.DETERMINISTIC_DATA.values()),
        ],
        sort=True,
    ),
    final_dataset_name="D",
    has_splits=True,
)
print(len(dataset2))
print(dataset2.get_idx_split())
