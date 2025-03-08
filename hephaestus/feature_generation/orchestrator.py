"""
Orchestrate feature creation.
"""

import shutil
from pathlib import Path

import tqdm

import hephaestus.feature_generation.gdd.features as features
import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils


def clean_my_directories():
    r"""Reset the directories used by the module"""
    shutil.rmtree(features.FEATURES_DIR, ignore_errors=True)


def prepare_directories(clean=False):
    r"""Create the directories used by the module
    
    Parameters
    ----------
    clean : bool
        Clean all directories before doing anything
    """
    if clean:
        clean_my_directories()
    Path(features.FEATURES_DIR).mkdir(parents=True, exist_ok=True)


def generate_features(type, feature_type):
    r"""Generate features for nodes of graphs in `hconfig.GRAPHS_DIR`.

    Entry point for the functions that generate features for each node of each graph
    in the directory `hconfig.GRAPHS_DIR`.

    Parameters
    ----------
    type : str
        Type of graphs to use.
    feature_type : str
        Type of features to generate

    See Also
    --------
    features.from_general_dataset

    Notes
    -----
    Not exposing the path to where the graphs are stored is done on purpose to not introduce
    more entropy into the file handling. Any change should be done in config.ini.
    """
    if type == hutils.NDETERMINISTIC_INDICATOR:
        datasets = sorted(list(hconfig.NDETERMINISTIC_DATA.values()))
    elif type == hutils.DETERMINISTIC_INDICATOR:
        datasets = sorted(list(hconfig.DETERMINISTIC_DATA.values()))
    elif type == hutils.SMALL_REAL_INDICATOR:
        datasets = sorted(list(hconfig.SREAL_DATA.values()))
    elif type == hutils.MEDIUMLARGE_REAL_INDICATOR:
        datasets = sorted(list(hconfig.MLREAL_DATA.values()))
    else:
        raise NotImplementedError(f"Implementation for {type} data not available!")

    for dataset in tqdm.tqdm(datasets, desc="Generating Features"):
        if feature_type == "gdd2":
            features.from_general_dataset(
                dataset, hconfig.GRAPHS_DIR, type="gdd", **{"sizes": 2}
            )
