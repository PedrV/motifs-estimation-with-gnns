"""
Connection between other modules and config.ini
"""

import ast
import configparser
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(os.path.join(ROOT_DIR, "config.ini"))

DETERMINISTIC_DATA = config["DGEN_TO_DATASET"]
NDETERMINISTIC_DATA = config["NDGEN_TO_DATASET"]
SREAL_DATA = config["SMALL_REAL_TO_DATASET"]
MLREAL_DATA = config["MEDIUM_LARGE_REAL_TO_DATASET"]

MARGINS_ZSCORE = ast.literal_eval(config.get("SUBGRAPH_INFO", "MARGINS_ZSCORE"))
NUM_SUBGRAPHS = config.getint("SUBGRAPH_INFO", "NUM_SUBGRAPHS")
SUBGRAPH_SIZE = ast.literal_eval(config.get("SUBGRAPH_INFO", "SUBGRAPH_SIZE"))
RANDS = config.getint("GTRIE", "RANDS")

MODULES_CWD = config.get("COMMON_PATHS", "modules_cwd")

INFO_PATH = config.get("INFO_PATHS", "info_path")
LOGGER_PATH = config.get("INFO_PATHS", "logger_path")

TEMP_DIR = config.get("TEMPDIR_PATHS", "temp_dir")
GRAPHS_DIR = config.get("DATA_PATHS", "graphs_dir")
DATASETS_STATS_DIR = config.get("DATA_PATHS", "datasets_stats_dir")
COMPLETE_DATASET_DIR = config.get("DATA_PATHS", "complete_dataset_dir")
FEATURES_DIR = config.get("DATA_PATHS", "features_dir")
SCORE_DIR = config.get("DATA_PATHS", "score_dir")
LABEL_DIR = config.get("DATA_PATHS", "labels_dir")
RAW_SPLITS_DIR = config.get("DATA_PATHS", "raw_splits_dir")
TEST_DIR = config.get("DATA_PATHS", "test_dir")

COUNT_SCRIPT_PATH = config.get("BASH_SCRIPT_PATHS", "count_script_path")
APPEND_SCRIPT_PATH = config.get("BASH_SCRIPT_PATHS", "append_script_path")

CLASSIFICATION_ENGINE_V1_PARAM_PATH = config.get(
    "PARAMETERS_PATHS", "classification_engine_v1"
)
CLASSIFICATION_ENGINE_V1_OPTIMIZATION_PARAM_PATH = config.get(
    "PARAMETERS_PATHS", "classification_engine_v1_optimization"
)

DETERM_GEN_PARAM_PATH = config.get("PARAMETERS_PATHS", "deterministic_gen_params")
NDETERM_GEN_PARAM_PATH = config.get("PARAMETERS_PATHS", "ndeterministic_gen_params")
RESOURCES_PATH = config.get("PARAMETERS_PATHS", "resources_dir")

UNIT_TESTS_DATA_DIR = config.get("UNIT_TESTS_PATH", "unity_tests_data_dir")
