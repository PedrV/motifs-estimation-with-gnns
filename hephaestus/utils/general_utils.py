"""
Utility functions used in other modules
"""

import os
from pathlib import Path
import importlib
import logging

import numpy as np

import hephaestus.utils.load_general_config as hconf

# Must match the initial part of the dataset names in config.ini
NDETERMINISTIC_INDICATOR = "nd"
DETERMINISTIC_INDICATOR = "d"
SMALL_REAL_INDICATOR = "sreal"
MEDIUMLARGE_REAL_INDICATOR = "mlreal"


def get_logging_function(logger_name):
    """
    Given a logger name returns a logger object logging with `INFO` level.
    Messages are saved according to `config.ini` under a file named `logger_name`.log.

    :param `str` `logger_name`: Name of the logger object.
    """
    log_path = Path(hconf.LOGGER_PATH)
    log_path.mkdir(exist_ok=True, parents=True)

    logger_object = logging.getLogger(logger_name)
    # Receive anything from level info and up
    logger_object.setLevel(logging.INFO)

    logger_object_fh = logging.FileHandler(
        filename=log_path / (logger_name + ".log"), mode="w", encoding="utf-8"
    )
    logger_object_fh.setLevel(logging.INFO)
    logger_object_fh.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(name)s-%(process)d-%(asctime)s-%(message)s"
        )
    )

    logger_object.addHandler(logger_object_fh)

    return logger_object


def flatten_nested_list(seq, sort=True):
    """
    Flat a 2d list with option to sort the resultant 1d list.

    :param `list` `seq`: List to be sorted
    :param `bool` `sort`: `True` if the list should be sorted. Default `True`
    """
    seq_flat = []
    for arr in seq:
        for elem in arr:
            seq_flat.append(elem)
    if sort:
        return np.array(sorted(seq_flat))

    return np.array(seq_flat)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_graph_name(raw_graph_file, with_extension=False, has_file_identifier=False):
    """
    Given a filename or filepath of a graph in the format specified in conventions.txt,
    e.g. DATASETNAME@GRAPHNAME.score-sizeSIZE, DATASETNAME@GRAPHNAME.csv,
    return the name specified graph.
    :param bool with_extension: True if the raw_graph_file as an extension.
    :param bool has_file_identifier: True if it has an identifier like _labels, _features before the extension
    """
    graph_name = None
    without_possible_path = os.path.basename(os.path.normpath(raw_graph_file))
    if with_extension:
        without_possible_path = os.path.splitext(without_possible_path)[0]

    # Looks like there isn't a dataset name before. b9339ff9-49c0-4e0b-b0d6-e663a9c3e2e6 should fix it.
    try:
        pseudo_graph_name = without_possible_path.split("@")[1]
    except IndexError:
        pseudo_graph_name = without_possible_path.split("@")[0]

    if has_file_identifier:
        graph_name = pseudo_graph_name.removesuffix(
            "_" + pseudo_graph_name.split("_")[-1]
        )
    else:
        graph_name = pseudo_graph_name
    return graph_name


def get_dataset_name(raw_graph_file, with_extension=False, has_file_identifier=False):
    """
    Given a filename or filepath of a graph in the format specified in conventions.txt,
    e.g. DATASETNAME@GRAPHNAME.score-sizeSIZE, DATASETNAME@GRAPHNAME.csv,
    return the name of the dataset containing the specified graph.
    :param bool with_extension: True if the raw_graph_file as an extension.
    :param bool has_file_identifier: True if it has an identifier like _labels, _features before the extension
    """
    dataset_name = None
    without_possible_path = os.path.basename(os.path.normpath(raw_graph_file))
    if with_extension:
        without_possible_path = os.path.splitext(without_possible_path)[0]

    pseudo_dataset_name = without_possible_path.split("@")[0]
    if has_file_identifier:
        dataset_name = pseudo_dataset_name.removesuffix(
            "_" + pseudo_dataset_name.split("_")[-1]
        )
    else:
        dataset_name = pseudo_dataset_name
    return dataset_name


def get_pretty_graph_name(
    raw_graph_file, with_extension=False, has_file_identifier=False
):
    """
    Returns the graph name of a graph file in a pretty format (to be used in text like plot titles).
    This format removes any identifiers signaled by "+" (as per conventions.txt).
    :param bool with_extension: True if the raw_graph_file as an extension.
    :param bool has_file_identifier: True if it has an identifier like _labels, _features before the extension
    """
    graph_name = get_graph_name(raw_graph_file, with_extension, has_file_identifier)
    no_modules = graph_name.split(".")[-1]
    no_symbols = no_modules.split("+")[0]
    return no_symbols.replace("_", " ").title()
