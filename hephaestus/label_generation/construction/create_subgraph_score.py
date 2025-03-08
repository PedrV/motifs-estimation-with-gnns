"""
Creates the raw motif scores for graphs using GTrie
"""

import os
import subprocess
from pathlib import Path

import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils
from hephaestus.label_generation.utils.label_utils import PREAMBLE

SCORE_DIR = Path(hconfig.SCORE_DIR)


def pattern_calculation(network_file_name, direction, result_file_name, subgraph_size):
    """
    Count subgraphs for a graph given by `network_file_name` using Gtrie.
    This will give use `compute_pattern.sh` to handle all computation.

    Note: `result_file_name` will not be created if Gtrie exists with error.

    :param `str` `network_file_name`: The graph file name with full expanded path of the graph to be calculated.
    :param `str` `direction`: Direction of the graph, undirected or directed.
    :param `str` `result_file_name`: The file name with full expanded path where the result will be stored.
    :param `int` `subgraph_size`: The size of the subgraphs whose occurrences Gtrie will compute to the given network.
    """
    with open(os.devnull, mode="w", encoding="utf-8") as devnull:
        success = subprocess.call(
            [
                "bash",
                hconfig.COUNT_SCRIPT_PATH,
                network_file_name,
                direction,
                str(subgraph_size),
                str(hconfig.RANDS),
                result_file_name,
            ],
            stdout=devnull,
            stderr=devnull,
        )
    if success != 0:
        print(
            f"[create_subgraph_score.py] Failed call for {network_file_name}, size {subgraph_size}"
        )
        return False

    return True


def merge_files(network_file_names, composed_graph_name):
    """
    Merge the output of `pattern_calculation` into a single file for a given graph.
    For example, for size 3 and 4 we would have DATASETNAME@GRAPHNAME.score-size3 and DATASETNAME@GRAPHNAME.score-size4.
    This will put the result into a file named DATASETNAME@GRAPHNAME.score.
    Uses `append_files.sh` to join files together.

    Note: It is recommended that `composed_graph_name` is built using the functions from `utils.general_utils`.

    :param `list` `network_file_names`: List with the files (with full path) with scores for a given graph.
    :param `str` `composed_graph_name`: Name of the graph that owns the scores in `network_file_names`.
    """
    destination = Path(os.path.join(SCORE_DIR, composed_graph_name + ".score"))
    destination.touch()

    with open(destination, mode="w", encoding="utf-8") as text_file:
        text_file.write(PREAMBLE)

    for file in network_file_names:
        subprocess.call(
            [
                "bash",
                hconfig.APPEND_SCRIPT_PATH,
                os.path.join(SCORE_DIR, file),
                str(destination),
            ]
        )


def extract_pattern(full_name_with_path, direction):
    """
    Calculates patterns of a given size for the graph given by `full_name_with_path`.

    :param `str` `full_name_with_path`: Full path with graph name of the graph that will be processed to find subgraph occurences
    :param `str` `direction`: Direction of the graph in `full_name_with_path`. Either directerd or undirected.
    """

    file_names = []
    failed_files = []

    pathless_graph_name = hutils.get_graph_name(
        full_name_with_path, with_extension=True
    )
    pathless_dataset_name = hutils.get_dataset_name(
        full_name_with_path, with_extension=True
    )
    composed_graph_name = pathless_dataset_name + "@" + pathless_graph_name

    for size in hconfig.SUBGRAPH_SIZE:
        result_file_name = os.path.join(
            SCORE_DIR, composed_graph_name + ".score-size" + str(size)
        )

        file_names.append(result_file_name)

        success = pattern_calculation(
            full_name_with_path, direction, result_file_name, str(size)
        )
        if not success:
            failed_files.append(result_file_name)

    merge_files(network_file_names=file_names, composed_graph_name=composed_graph_name)

    return failed_files
