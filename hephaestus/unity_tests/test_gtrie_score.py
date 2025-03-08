"""
Tests related with the output of GTrie e.g. merge multiple files and create files
not populated by GTrie due to an error when running GTrie.
"""

import filecmp
import sys
import os
import shutil
from pathlib import Path

import unittest

import numpy as np
import networkx as nx

os.chdir(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
sys.path.insert(0, os.getcwd())

import hephaestus.label_generation.construction.create_subgraph_score as create_subgraph_score
import hephaestus.label_generation.construction.build_label_csv as build_label_csv
import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils

TEST_DIR = hconfig.UNIT_TESTS_DATA_DIR


class TestGTrieScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.setUpClassInner()
        except Exception as e:
            cls.tearDownClass()
            raise e

    @classmethod
    def setUpClassInner(cls):
        cls.append_raw = os.path.join(TEST_DIR, "storage", "gtrie_score", "append_raw")
        cls.append_true = os.path.join(
            TEST_DIR, "storage", "gtrie_score", "append_truth"
        )
        cls.gen_true = os.path.join(TEST_DIR, "storage", "gtrie_score", "gen_truth")

        cls.graph_gen_dir = os.path.join(TEST_DIR, "tests", "graph_gen")
        cls.gtrie_score_dir_app = os.path.join(
            TEST_DIR, "tests", "gtrie_scores", "append"
        )
        cls.gtrie_score_dir_gen = os.path.join(TEST_DIR, "tests", "gtrie_scores", "gen")

        os.makedirs(os.path.join(TEST_DIR, "tests", "gtrie_scores"), exist_ok=True)
        os.mkdir(cls.gtrie_score_dir_app)
        os.mkdir(cls.gtrie_score_dir_gen)
        os.mkdir(cls.graph_gen_dir)
        create_subgraph_score.SCORE_DIR = cls.gtrie_score_dir_app

        for f in os.listdir(cls.append_raw):
            shutil.copy(
                os.path.join(cls.append_raw, f),
                os.path.join(cls.gtrie_score_dir_app, f),
            )

    def test_merge_files(self):
        files = os.listdir(self.gtrie_score_dir_app)
        dataset_name = hutils.get_dataset_name(files[0], with_extension=True)
        graph_name = hutils.get_graph_name(files[0], with_extension=True)
        composed_graph_name = dataset_name + "@" + graph_name

        create_subgraph_score.merge_files(files, composed_graph_name)

        for pred, true in zip(
            sorted(os.listdir(self.gtrie_score_dir_app)),
            sorted(os.listdir(self.append_true)),
        ):
            eq = filecmp.cmp(
                os.path.join(self.gtrie_score_dir_app, pred),
                os.path.join(self.append_true, true),
            )
            self.assertTrue(eq, f"Merge failed: {pred} != {true}")

    def test_producing_missing_files(self):
        f = open(os.devnull, "w")
        sys.stdout = f

        gs = [
            nx.from_edgelist([(1, 2), (2, 3), (3, 1), (4, 5), (4, 6)]),
            nx.from_edgelist([(1, 2), (4, 3)]),
        ]

        for i, g in enumerate(gs):
            g_name = os.path.join(self.graph_gen_dir, "UNIT@graph" + str(i) + ".csv")

            # Directly from graph_by_generator.py
            df = nx.to_pandas_edgelist(g)
            df["edge_attrib"] = np.ones((df.shape[0],), dtype=int)
            df.to_csv(g_name, sep=" ", index=False, header=False)

            # Directly from construct_labels.py
            # wrapped in [] because it does not a starmap call that produces list of many results
            results = [create_subgraph_score.extract_pattern(g_name, "undirected")]
            self.assertEqual(
                len(results[0]), i + 1, "Wrong amount of invalid subgraph sizes."
            )
            for res in results:
                if len(res) == 0:
                    continue
                build_label_csv.build_missing_file(
                    res, "UNIT", blc_logger=hutils.get_logging_function("unit")
                )

        for pred, true in zip(
            sorted(os.listdir(self.gtrie_score_dir_gen)),
            sorted(os.listdir(self.gen_true)),
        ):
            eq = filecmp.cmp(
                os.path.join(self.gtrie_score_dir_gen, pred),
                os.path.join(self.gen_true, true),
            )
            self.assertTrue(eq, f"Merge failed: {pred} != {true}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.gtrie_score_dir_app)
        shutil.rmtree(cls.graph_gen_dir)
        shutil.rmtree(os.path.join(TEST_DIR, "tests", "gtrie_scores"))
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main(verbosity=2)
