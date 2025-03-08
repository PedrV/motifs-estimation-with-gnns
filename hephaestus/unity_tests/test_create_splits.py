"""
Tests related with the splits of the dataset e.g. is the condition_to_skip being held.
"""

import sys
import os
import shutil
from pathlib import Path

import unittest

import numpy as np

os.chdir(Path(os.path.dirname(os.path.realpath(__file__))).parent.parent)
sys.path.insert(0, os.getcwd())

import hephaestus.dataset_creation.create_splits as create_splits
import hephaestus.label_generation.construction.build_label_csv as build_label_csv
import hephaestus.utils.load_general_config as hconfig
import hephaestus.utils.general_utils as hutils

TEST_DIR = hconfig.UNIT_TESTS_DATA_DIR


class TestSplits(unittest.TestCase):
    def setUp(self):
        try:
            self.setUpClassInner()
        except Exception as e:
            self.tearDown()
            raise e

    def setUpClassInner(self):
        self.label_dir = os.path.join(TEST_DIR, "labels")
        self.score_dir = os.path.join(TEST_DIR, "raw_scores")

        self.splits_dir = os.path.join(
            TEST_DIR, "storage", "create_splits", "processed"
        )

        os.mkdir(self.label_dir)
        os.mkdir(self.score_dir)

        for f in os.listdir(self.splits_dir):
            shutil.copy(
                os.path.join(self.splits_dir, f),
                os.path.join(self.score_dir, f),
            )

        build_label_csv.LABEL_DIR = self.label_dir
        build_label_csv.SCORE_DIR = self.score_dir
        build_label_csv.create_csv_and_image(
            "UNIT", np.array([]), blc_logger=hutils.get_logging_function("unit")
        )

    def test_condition_to_skip(self):
        r, cnt, _ = create_splits._get_valid_graphs(
            os.path.join(TEST_DIR, "labels", "UNIT_labels.csv")
        )

        self.assertEqual(
            r, [1, 2], "Identified the wrong graphs as having NaNs or 0 score"
        )
        self.assertEqual(
            cnt, 2, "Identified wrong amount of graphs as having NaNs or 0 score"
        )

    def tearDown(self):
        shutil.rmtree(self.label_dir)
        shutil.rmtree(self.score_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
