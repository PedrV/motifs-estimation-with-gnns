import os
import sys
import time
import argparse

import yaml

from multiprocessing import Process

sys.path.insert(0, os.getcwd())

import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig
import hephaestus.graph_generation.orchestrator as graph_gen_orchestrator
import hephaestus.label_generation.orchestrator as label_gen_orchestrator
import hephaestus.feature_generation.orchestrator as feature_gen_orchestrator

with open(os.path.join(hconfig.RESOURCES_PATH, "resources.yaml")) as f:
    NUMBER_OF_WORKERS = yaml.safe_load(f)["label_generation"]["NUMBER_OF_WORKERS"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--operation",
        type=str,
        choices=["make_labels", "make_graphs", "make_features", "make_all"],
        dest="operation",
        default="all",
        required=True,
        help="What type of operation to make.\n (1) Make labels for the graphs in the graph directory;\n (2) Make graphs given a generation type;\n (3) Generate features for the graphs in the graph directory;\n (4) Do all of the above",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            hutils.NDETERMINISTIC_INDICATOR,
            hutils.DETERMINISTIC_INDICATOR,
            hutils.SMALL_REAL_INDICATOR,
            hutils.MEDIUMLARGE_REAL_INDICATOR,
        ],
        dest="type",
        default=hutils.NDETERMINISTIC_INDICATOR,
        required=True,
        nargs="+",
        help="Type of graphs that will be used for the task specified in the operation argument.\n Non-deterministic (nd) and deterministic (d) are sythetic graphs.\n Small (s) and large (l) are real graphs.",
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["gdd2"],
        dest="feature_type",
        default="gdd2",
        required=False,
        help="Type of feature to generate for each node of each graph in the graph directory.",
    )
    parser.add_argument(
        "--clean",
        choices=[
            "None",
            "All",
            "make_graphs",
            "make_labels",
            "make_features",
        ],
        dest="clean",
        default="None",
        required=True,
        nargs="+",
        help="Should all directories related with the operation given by the operation argument be cleaned before starting the procedure?",
    )

    args, _ = parser.parse_known_args()

    clean_graph_gen = "All" in args.clean or "make_graphs" in args.clean
    clean_feature_gen = "All" in args.clean or "make_features" in args.clean
    clean_labels_gen = "All" in args.clean or "make_labels" in args.clean

    if args.operation == "make_graphs" or args.operation == "make_all":
        graph_gen_orchestrator.prepare_directories(clean=clean_graph_gen)
        for t in args.type:
            graph_gen_orchestrator.generate_graphs(t)

    time.sleep(10)

    if args.operation == "make_labels" or args.operation == "make_all":
        label_gen_orchestrator.prepare_directories(clean=clean_labels_gen)
        for t in args.type:
            processes = []
            for i in range(0, NUMBER_OF_WORKERS):
                p = Process(
                    target=label_gen_orchestrator.generate_labels,
                    args=(
                        t,
                        i,
                        NUMBER_OF_WORKERS,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            time.sleep(2)

    time.sleep(10)

    if args.operation == "make_features" or args.operation == "make_all":
        feature_gen_orchestrator.prepare_directories(clean=clean_feature_gen)
        for t in args.type:
            feature_gen_orchestrator.generate_features(t, args.feature_type)
