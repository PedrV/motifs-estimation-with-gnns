"""
Orchestrate creation of the labels for each graph.
"""

import os
import sys
import time
import shutil
from pathlib import Path
import yaml

import numpy as np
import multiprocessing

sys.path.insert(0, os.getcwd())

import hephaestus.label_generation.construction.create_subgraph_score as subgraph_score
import hephaestus.label_generation.construction.build_label_csv as label_build
import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils


with open(Path(hconfig.RESOURCES_PATH) / "resources.yaml") as f:
    TOTAL_CPU_FRACTION_TO_USE = yaml.safe_load(f)["label_generation"][
        "TOTAL_CPU_FRACTION_TO_USE"
    ]


def clean_my_directories():
    """Reset the directories used by the module"""
    shutil.rmtree(label_build.LABEL_DIR, ignore_errors=True)
    shutil.rmtree(subgraph_score.SCORE_DIR, ignore_errors=True)


def prepare_directories(clean=False):
    r"""Create the directories used by the module

    Parameters
    ----------
    clean : bool
    """
    if clean:
        clean_my_directories()
    Path(label_build.LABEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(subgraph_score.SCORE_DIR).mkdir(parents=True, exist_ok=True)


def generate_labels(type, my_id=0, number_workers=1):
    r"""Generate labels for the graphs according to the given type.

    Entry point for the functions that generate/prepare graphs to be used in other modules.
    This function is prepared to be used in a multicore scenario where its id is given by
    `my_id` and the chunk size is calculated with `number_workers` and `my_id`.

    Parameters
    ----------
    type : str
        Type of graphs to use.
    label_type : str
        Type of label to use for the label of the graphs.
    my_id : int
        Id given to the process running the function.
    number_workers : int
        Number of parallel instances running this function.

    See Also
    --------
    build_label_csv.build_missing_file
    build_label_csv.create_csv_and_image
    create_subgraph_score.extract_pattern

    Notes
    -----
    Not exposing any path is done on purpose to not introduce
    more entropy into the file handling. Any change should be done in config.ini.
    """
    if type == hutils.DETERMINISTIC_INDICATOR:
        generators = hconfig.DETERMINISTIC_DATA
    elif type == hutils.NDETERMINISTIC_INDICATOR:
        generators = hconfig.NDETERMINISTIC_DATA
    elif type == hutils.SMALL_REAL_INDICATOR:
        generators = hconfig.SREAL_DATA
    elif type == hutils.MEDIUMLARGE_REAL_INDICATOR:
        generators = hconfig.MLREAL_DATA

    CL_LOGGER = hutils.get_logging_function("ConstructLabels" + str(my_id))

    generators_chunk = int(np.round(len(generators) / number_workers))
    lower_part = int(my_id * generators_chunk)
    upper_part = int((my_id + 1) * generators_chunk)
    generators_keys = list(generators.keys())[lower_part:upper_part]
    if my_id + 1 == number_workers:
        generators_keys = list(generators.keys())[lower_part:]
    else:
        generators_keys = list(generators.keys())[lower_part:upper_part]

    for generator_name in generators_keys:
        ### Get Graph Details ###
        dataset_files = []
        dataset_name = generators[generator_name]
        for f in sorted(os.listdir(hconfig.GRAPHS_DIR)):
            if dataset_name in f:
                dataset_files.append(f)

        # This allows to run the code without commeting generator names in the config
        if len(dataset_files) == 0:
            CL_LOGGER.info(
                f"Skipped {generator_name}, this, most likely, a partial run."
            )
            continue

        dataset_size = len(dataset_files)
        network_file_names = [
            os.path.join(hconfig.GRAPHS_DIR, f) for f in dataset_files
        ]
        direction_of_graphs = ["undirected" for _ in range(dataset_size)]
        dataset_previous_label = np.array([])
        #####################

        #### Create raw labels ####
        cpu_count = round(multiprocessing.cpu_count() * TOTAL_CPU_FRACTION_TO_USE)
        chunk_size = int(np.ceil(dataset_size / cpu_count))

        CL_LOGGER.info(f"Starting label generation for {dataset_name} ....")
        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = pool.starmap(
                subgraph_score.extract_pattern,
                zip(network_file_names, direction_of_graphs),
                chunksize=chunk_size,
            )
        CL_LOGGER.info(f"Ended label generation for {dataset_name}!")
        #####################

        time.sleep(10)

        ### Process raw labels ###
        CL_LOGGER.info(f"Starting label refinement for {dataset_name} ....")
        blc_logger = hutils.get_logging_function("BuildLabel" + str(my_id))
        for res in results:
            if len(res) == 0:
                continue
            label_build.build_missing_file(res, dataset_name, blc_logger)
        label_build.create_csv_and_image(
            dataset_name, dataset_previous_label, blc_logger
        )
        CL_LOGGER.info(f"Ended label refinement for {dataset_name}!")

        Path(os.path.join(hconfig.LOGGER_PATH, "incomplete")).mkdir(exist_ok=True)
        with open(
            file=os.path.join(
                hconfig.LOGGER_PATH,
                "incomplete",
                dataset_name + "_incomplete_graphs.txt",
            ),
            mode="w",
            encoding="utf-8",
        ) as f:
            for line in hutils.flatten_nested_list(results):
                f.write(f"{line}\n")
        #####################
