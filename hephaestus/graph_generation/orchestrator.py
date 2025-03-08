"""
Orchestrate graph creation.
"""

from pathlib import Path
import shutil

import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig
import hephaestus.graph_generation.real.organize_test_set as prepare_test_set

from hephaestus.graph_generation.synthetic.graph_by_generator import Generators


def clean_my_directories():
    r"""Reset the directories used by the module"""
    shutil.rmtree(hconfig.GRAPHS_DIR, ignore_errors=True)
    shutil.rmtree(hconfig.DATASETS_STATS_DIR, ignore_errors=True)
    shutil.rmtree(prepare_test_set.PROCESSED_DATA_DIR, ignore_errors=True)


def prepare_directories(clean=False):
    r"""Create the directories used by the module
    
    Parameters
    ----------
    clean : bool
        Clean all directories before doing anything
    """
    if clean:
        clean_my_directories()
    # GRAPHS_DIR and DATASETS_STATS_DIR belong to the class Generators
    Path(hconfig.GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
    Path(hconfig.DATASETS_STATS_DIR).mkdir(parents=True, exist_ok=True)
    Path(prepare_test_set.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)


def generate_graphs(type):
    r"""Generate graphs.

    Entry point for the functions that generate/prepare graphs to be used in other modules.

    Parameters
    ----------
    type : str
        Type of graphs to use.

    See Also
    --------
    graph_by_generator.Generators
    organize_test_set.prepare_test_set

    Notes
    -----
    Not exposing any path is done on purpose to not introduce
    more entropy into the file handling. Any change should be done in config.ini.
    """
    if type == hutils.NDETERMINISTIC_INDICATOR:
        dg = Generators(
            hconfig.GRAPHS_DIR,
            hconfig.DATASETS_STATS_DIR,
            Generators.ND_TYPE,
        )
        dg.generate()
    elif type == hutils.DETERMINISTIC_INDICATOR:
        dg = Generators(
            hconfig.GRAPHS_DIR,
            hconfig.DATASETS_STATS_DIR,
            Generators.D_TYPE,
        )
        dg.generate()
    elif type == hutils.SMALL_REAL_INDICATOR:
        raw_data_dir = Path(hconfig.TEST_DIR) / (hutils.SMALL_REAL_INDICATOR + "_raw")
        prepare_test_set.process([], raw_data_dir)
        prepare_test_set.prepare_next_stage(hconfig.GRAPHS_DIR)
    elif type == hutils.MEDIUMLARGE_REAL_INDICATOR:
        raw_data_dir = Path(hconfig.TEST_DIR) / (hutils.MEDIUMLARGE_REAL_INDICATOR + "_raw")
        prepare_test_set.process([], raw_data_dir)
        prepare_test_set.prepare_next_stage(hconfig.GRAPHS_DIR)
