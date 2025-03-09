import os
import sys
from pathlib import Path
import hashlib

import json

import tqdm
from time import sleep

import torch
import torch_geometric

import numpy as np
import pandas as pd

import random

from matplotlib import pyplot as plt

sys.path.insert(0, "path/to/where/hephaestus/is/placed")

from hephaestus.models.classfication_engine import ClassificationEngineV1
from hephaestus.dataset_creation.custom_dataset import CustomDataset

import hephaestus.utils.general_utils as hutils
import hephaestus.utils.load_general_config as hconfig

import seaborn as sns

########################## SEABORN STUFF ##########################
sns.set_context("paper", font_scale=1)
sns.set_style("whitegrid")

p = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#FB6467FF",
    "#808282",
    "#F0E442",
    "#440154FF",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#C2CD23",
    "#918BC3",
    "#FFFFFF",
]

sns.color_palette(p)
#####################################################################

########################## CHANGES GO HERE ###########################
EXTRA_TYPE = "REPLICATON"  # e.g. dropout
DECODER_VERSION = "simple-decoder_V1"
CSV_SAVE_PREDICTIONS = True

REPEATED_NAMES = False

MODEL_TYPE = "gin"
DATASET_TYPE = "d"
PER_GENERATOR_ELEMENTS = 320 if DATASET_TYPE == "d" else 349  # Graphs per generator

TRIALS = [
    "TorchTrainer_e9bc79c6"
    # sage. "TorchTrainer_995a1ad7", d-gin. , nd-gin. "TorchTrainer_d9c8c887"
]
EXPERIMENTS = [
    "TALOS_20240121-203233",
]  # sage. "TALOS_20240216-185022" , d-gin.  nd-gin. "TALOS_20240201-180351"
#####################################################################

########################## SET UP DIRECTORIES #######################
BASE_EXPERIMENT_DIRECTORY = Path("path/to/where/model_storage/models/is/located")
MODELS_PATH = BASE_EXPERIMENT_DIRECTORY

BASE_PATH = Path("some/base/path/where/stuff/is/saved")
SAVE_PATH = (
    BASE_PATH / "plots" / "evaluate_models" / EXTRA_TYPE / DATASET_TYPE / MODEL_TYPE
)
os.makedirs(SAVE_PATH, exist_ok=True)
#####################################################################

####################### SET UP SEGMENTS ###########################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGMENTS = [DATASET_TYPE]

if "nd" in DATASET_TYPE:
    GENERATORS = hutils.flatten_nested_list(
        [
            list(hconfig.NDETERMINISTIC_DATA.values()),
        ],
        sort=True,
    )
else:
    GENERATORS = hutils.flatten_nested_list(
        [
            list(hconfig.DETERMINISTIC_DATA.values()),
        ],
        sort=True,
    )
#####################################################################

datasets_names = []
individual_dataset_name = []

SEED = 42
rng = np.random.default_rng(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def return_trial_details(query_trial):
    best_value = np.inf
    best_value_idx = -1
    experiment = ""

    for exp in EXPERIMENTS:
        experiment_path = MODELS_PATH / exp

        for f in os.listdir(experiment_path):
            if query_trial not in f:
                continue

            progress = pd.read_csv(experiment_path / f / "progress.csv")

            best_value = progress["loss"].min()
            best_value_idx = progress["loss"].idxmin()
            best_trial = f
            experiment = exp
            break

    best_trial_pretty = best_trial.split("_")[0] + "_" + best_trial.split("_")[1]
    print(
        f"Best Trial {best_trial_pretty}, with {best_value} at epoch {best_value_idx}!"
    )

    return best_trial, best_value_idx, experiment, best_value


def get_metrics(model, silent=False):
    complete_dataset_dir = Path(hconfig.COMPLETE_DATASET_DIR).parent

    global datasets_names, individual_dataset_name
    datasets_names = []
    individual_dataset_name = []

    all_preds = []
    all_truths = []

    g = torch.Generator()
    g.manual_seed(SEED)

    tot_seen = 0
    for j, dataset_name in enumerate(SEGMENTS):

        cdir = complete_dataset_dir / f"complete_dataset_{dataset_name}"
        print(cdir)

        dataset = CustomDataset(
            cdir,
            GENERATORS,
            final_dataset_name=dataset_name,
            has_splits=True,
        )
        splts = dataset.get_idx_split()
        dataset = dataset[splts["test"]]

        tot_seen += len(dataset)
        if not silent:
            print(dataset_name, len(dataset))

        for i in range(len(GENERATORS)):
            if "dSTARGRAPH" in GENERATORS[i]:
                datasets_names.append([GENERATORS[i]] * (159 - 1))
                individual_dataset_name.append(
                    [GENERATORS[i] + "_" + str(j) for j in range(159 - 1)]
                )
            else:
                datasets_names.append([GENERATORS[i]] * PER_GENERATOR_ELEMENTS)
                individual_dataset_name.append(
                    [
                        GENERATORS[i] + "_" + str(j)
                        for j in range(PER_GENERATOR_ELEMENTS)
                    ]
                )

        model.eval()
        assert not model.training

        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        loss_fn = torch.nn.MSELoss()

        preds = []
        truths = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):

                batch.to(DEVICE)  # Does not need to reassign

                y_pred = model(batch.edge_index, batch.x, batch.batch)

                loss_fn(y_pred, batch.y).cpu()
                preds.append(y_pred.cpu())
                truths.append(batch.y.cpu())

            y_true = torch.cat(truths, dim=0).numpy()
            y_pred = torch.cat(preds, dim=0).numpy()

        all_truths.append(y_true)
        all_preds.append(y_pred)

    return all_preds, all_truths


def do_everything(trial):
    best_trial_pretty = trial
    best_trial_raw, best_value_idx, experiment, best_value = return_trial_details(
        best_trial_pretty
    )

    splited_name = best_trial_raw.split("_")[0] + "_" + best_trial_raw.split("_")[1]
    if REPEATED_NAMES:
        splited_name += (
            "_" + best_trial_raw.split("_")[2] + "_" + best_trial_raw.split("_")[3]
        )

    print(splited_name)
    EXPERIMENT_PATH = MODELS_PATH / experiment
    assert splited_name == best_trial_pretty

    if best_value_idx < 10:
        checkpoint_string = "checkpoint_00000" + str(best_value_idx)
    elif best_value_idx < 100:
        checkpoint_string = "checkpoint_0000" + str(best_value_idx)
    else:
        checkpoint_string = "checkpoint_000" + str(best_value_idx)

    restored_model = torch.load(
        EXPERIMENT_PATH / best_trial_raw / checkpoint_string / "model_checkpoint.pt"
    )
    extra_details = torch.load(
        EXPERIMENT_PATH
        / best_trial_raw
        / checkpoint_string
        / "extra_state_checkpoint.pt"
    )

    with open(EXPERIMENT_PATH / best_trial_raw / "params.json") as json_data:
        d = json.load(json_data)
        json_data.close()

    CORRECT_SAVE_PATH = SAVE_PATH / (experiment + "-" + best_trial_pretty)
    os.makedirs(CORRECT_SAVE_PATH, exist_ok=True)

    if MODEL_TYPE == "sage":
        model = ClassificationEngineV1(
            decoder_depth=d["train_loop_config"]["decoder_depth"],
            decoder_dropout=d["train_loop_config"]["decoder_dropout"],
            decoder_hidden_dim=d["train_loop_config"]["decoder_hidden_dim"],
            decoder="hephaestus.models.simple_decoder.SimpleDecoder",
            mpgnn="torch_geometric.nn.GraphSAGE",
            mpgnn_depth=d["train_loop_config"]["mpgnn_depth"],
            mpgnn_dropout=d["train_loop_config"]["mpgnn_dropout"],
            mpgnn_hidden_dim=d["train_loop_config"]["mpgnn_hidden_dim"],
            mpgnn_jk=d["train_loop_config"]["mpgnn_jk"],
            mpgnn_pool=d["train_loop_config"]["pooling"],
            input_dim=1,
            output_dim=8,
            **{"my_decoder_version": DECODER_VERSION},
        )
    elif MODEL_TYPE == "gin":
        model = ClassificationEngineV1(
            decoder_depth=d["train_loop_config"]["decoder_depth"],
            decoder_dropout=d["train_loop_config"]["decoder_dropout"],
            decoder_hidden_dim=d["train_loop_config"]["decoder_hidden_dim"],
            decoder="hephaestus.models.simple_decoder.SimpleDecoder",
            mpgnn="torch_geometric.nn.GIN",
            mpgnn_depth=d["train_loop_config"]["mpgnn_depth"],
            mpgnn_dropout=d["train_loop_config"]["mpgnn_dropout"],
            mpgnn_hidden_dim=d["train_loop_config"]["mpgnn_hidden_dim"],
            mpgnn_jk=d["train_loop_config"]["mpgnn_jk"],
            mpgnn_pool=d["train_loop_config"]["pooling"],
            input_dim=1,
            output_dim=8,
            **{"my_decoder_version": DECODER_VERSION},
        )

    model.to(DEVICE)
    try:
        for mv, sv in zip(model.my_versions, extra_details["versions"]):
            if mv != sv:
                raise ValueError(
                    f"Incompatible model version! {mv} on local model class, {sv} in stored model details."
                )
    except KeyError:
        print(f"Version did not use model version info. Using {model.my_versions}")

    print("Model Versions Verified, Everything Matches!")
    model.load_state_dict(restored_model)

    # model.reset_parameters()
    # for name, param in model.named_parameters():
    #     print(name, param)

    ################## Calculate metrics for relevant models ######################
    global datasets_names, individual_dataset_name

    all_preds, all_truths = get_metrics(model)

    sleep(1)
    ##############################################################################

    ###################### Prepare all the dataframes ######################
    df_pred = pd.DataFrame(
        np.vstack(all_preds),
        columns=["G" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_pred["DatasetName"] = hutils.flatten_nested_list(datasets_names)
    df_pred["GraphName"] = hutils.flatten_nested_list(individual_dataset_name)
    df_pred["Type"] = ["Pred"] * df_pred.shape[0]

    df_truth = pd.DataFrame(
        np.vstack(all_truths),
        columns=["G" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_truth["DatasetName"] = hutils.flatten_nested_list(datasets_names)
    df_truth["GraphName"] = hutils.flatten_nested_list(individual_dataset_name)
    df_truth["Type"] = ["True"] * df_truth.shape[0]

    df_patterns = pd.concat([df_pred, df_truth])

    if CSV_SAVE_PREDICTIONS:
        _save_p = Path(
            os.path.join(
                SAVE_PATH / (experiment + "+" + trial + "+" + DATASET_TYPE + ".csv")
            )
        )
        df_patterns.to_csv(_save_p)

        with open(_save_p, "rb") as f:
            file_data = f.read()

        # Create SHA-256 checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        with open(SAVE_PATH / (_save_p.stem + ".sha256"), "w") as f:
            f.write(checksum)
    ##############################################################################


if __name__ == "__main__":
    for trial in tqdm.tqdm(TRIALS):
        do_everything(trial)
        sleep(1)
