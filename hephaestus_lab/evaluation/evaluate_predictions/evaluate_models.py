import os
import sys
import shutil
from pathlib import Path
import hashlib

import json

import tqdm
from time import sleep

import torch
import torch_geometric

import numpy as np
import pandas as pd
import sklearn.metrics as skm

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
EXTRA_TYPE = "REPLICATION"
DECODER_VERSION = "simple-decoder_V1"
CSV_SAVE_PREDICTIONS = True

REPEATED_NAMES = False

RECUPERATE_TEST_SET_ERROR = True
TEST_SET_DATASET = "d"
TEST_SET_LOCATION = f"path/to/folder/_excluded/complete_dataset_{TEST_SET_DATASET}/"

MODEL_TYPE = "gin"

TRIALS = [
    "TorchTrainer_e9bc79c6"
    # sage. "TorchTrainer_995a1ad7" , d-gin., nd-gin. "TorchTrainer_d9c8c887"
]

EXPERIMENTS = [
    "TALOS_20240121-203233"
]  # sage. "TALOS_20240216-185022", d-gin. , nd-gin. "TALOS_20240201-180351"

DATASET_TYPE = "sreal"

########################## SET UP DIRECTORIES #######################
TRUE_LABELS_PATH = Path("path/to/where/the/labels/of/the/real-world/data/are/saved")

BASE_EXPERIMENT_DIRECTORY = Path("path/to/where/model_storage/models/is/located")
MODELS_PATH = BASE_EXPERIMENT_DIRECTORY / "model_storage" / "models"

BASE_PATH = Path("some/base/path/where/stuff/is/saved")
SAVE_PATH = (
    BASE_PATH / "plots" / "evaluate_models" / EXTRA_TYPE / DATASET_TYPE / MODEL_TYPE
)
os.makedirs(SAVE_PATH, exist_ok=True)
#####################################################################
#####################################################################

####################### SET UP GENERATORS ###########################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENERATORS = (
    sorted(list(hconfig.MLREAL_DATA.values()))
    if DATASET_TYPE == "mlreal"
    else sorted(list(hconfig.SREAL_DATA.values()))
)
#####################################################################

datasets_names = []
quartiles_dataset_names = []
individual_dataset_name = []

rng = np.random.default_rng(42)


def random_uniform_model(pattern):
    r"""Predict a pattern based on a random uniform model.

    Predict a pattern given by `pattern` using the expected response for a random uniform model.

    Parameters
    ----------
    pattern : torch.tensor
        The pattern for the model to predict
    
    Returns
    -------
    expected_loss : float
        The expected loss for this model
    
    expected_errors : np.array
        The expected error obtained in each part of the pattern

    torch.tensor
        The mean predicted pattern according to the model.

    Notes
    -----
    Surprisingly it is better than allowing only 0.7017, -0.7017 and 1 as choices for the fist 2. 
    The mean predicted pattern has indeterminacy on the signal of the predicted values.
    It has a bias towards the positive prediction by construction e.g. with error 0.4 and true value 0.1,
    it will return a value of 0.5 instead of -0.3.

    Derivation of the error:    
    
    .. math::
    \begin{align}
    & \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}\big( (y_i - \hat{Y}_i)^2 \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}\big( y_i^2 - y_i\hat{Y}_i + \hat{Y}^2_i \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(\mathbb{E}(y_i^2) - \mathbb{E}(y_i\hat{Y}_i + \hat{Y}^2_i) \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i - \mathbb{E}(y_i\hat{Y}_i) + \mathbb{E}(\hat{Y}^2_i) - \mathbb{E}(\hat{Y}_i)^2 \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i - y_i\mathbb{E}(\hat{Y}_i) + Var(\hat{Y}_i) \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i + Var(\hat{Y}_i) \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} (y^2_i + 1/3) \\
    & \equiv \frac{1}{n} \bigg(n/3 + \sum_{i=1}^{n} y^2_i \bigg) \\
    & \equiv 1/3 + 2/n
    \end{align}

    Note that since there are 2 patterns involved, size 3 and size 4, and they are normalized according to [1],
    the we sum 2 to n/3.

    References
    ----------
    .. [1] Ron Milo et al., "Superfamilies of Evolved and Designed Networks."
       Science303,1538-1542(2004).DOI:10.1126/science.1089167 
    """

    expected_errors = np.array(pattern) ** 2 + 1 / 3
    mean_predicted_pattern = []
    for error, true in zip(expected_errors, pattern):
        true = true.item()
        error = np.sqrt(error)
        if true + error < 1:
            mean_predicted_pattern.append(true + error)
        else:
            mean_predicted_pattern.append(true - error)

    expected_loss = np.mean(expected_errors)

    return expected_loss, expected_errors, torch.tensor(mean_predicted_pattern)


def predict_res(dim):
    coords = rng.normal(0, 1, size=dim)
    norm = np.linalg.norm(coords)
    coords /= norm
    return coords


def predict_rand_res(pattern):
    r"""
    Randomly predicting the score of a pattern having into account the restrictions of the problem
    as formulated. That being the sum of the squares of the Z-scores of graphs of the same size is 1.

    Parameters
    ----------
    pattern : torch.tensor
        The pattern for the model to predict
    
    Returns
    -------
    expected_loss : float
        The expected loss for this model
    
    expected_errors : np.array
        The expected error obtained in each part of the pattern

    torch.tensor
        The mean predicted pattern according to the model.

    Notes
    -----
    This give the same loss as predicting always -0.707106 and 0.707106
    uniformly at random in a symmetric way. That is, even though we do not use
    \hat{Y}_i as a random uniform prediction of -0.707106 or 0.707106 as it should be, the
    used method ends up with a very close variance and mean to this distribution. 
    The method used is supposed to simulate a random uniform distribution inscribed 
    in a unitary space of dim dimensions. For a dim=2 it should generate points uniformly 
    in the unity circle. Due to some perturbations on getting a true uniform distribution 
    with the method used, for 2 dimensions, the result has characteristics (mean and variance) 
    similar to the random uniform prediction of -0.707106 or 0.707106.
    For \hat{Y}_j it is supposed to be uniformely distributed in a unitary hypersphere of dim=6.
    The calculation follows the normal procedures. \hat{Y}_j has in the denominator a Chi distribution
    with 6 degrees of freedom. In the numerator it has a standard normal. This yields random coordinates
    in the 6-dimensional unit space. The mean is 0 since the unit vector is equally likely to point in
    any direction. The variance is 1/6 and can be obtained by solving the E[\hat{Y}_j]^2 - E[\hat{Y}_j^2]
    with \hat{Y}_j^2 following a Beta distribution with parameters 1/2 and 5/2.

    Usage of standard normal based on improbable generation of uniform points in
    hihg-dimensional spaces [1,2].

    Derivation of the error:    
    
    .. math::
    \begin{align}
    & \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}\big( (y_i - \hat{Y}_i)^2 \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}\big( y_i^2 - y_i\hat{Y}_i + \hat{Y}^2_i \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(\mathbb{E}(y_i^2) - \mathbb{E}(y_i\hat{Y}_i + \hat{Y}^2_i) \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i - \mathbb{E}(y_i\hat{Y}_i) + \mathbb{E}(\hat{Y}^2_i) - \mathbb{E}(\hat{Y}_i)^2 \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i - y_i\mathbb{E}(\hat{Y}_i) + Var(\hat{Y}_i) \big) \\
    & \equiv \frac{1}{n} \sum_{i=1}^{n} \big(y^2_i + Var(\hat{Y}_i) \big) \\
    & \equiv \frac{1}{8} \bigg(\sum_{i=1}^{2} (y^2_i + Var(\hat{Y}_i)) + \sum_{j=3}^{8}  (y^2_j + Var(\hat{Y}_j))\bigg) \\
    & \equiv \frac{1}{8} \bigg((1 + 2\cdot0.5) + (1+6\cdot0.16) \bigg) \\
    & \equiv 0.495
    \end{align}

    References
    ----------
    .. [1] Giannopoulos, A. A., & Milman, V. D. (2000). Concentration Property on Probability Spaces. 
    In Advances in Mathematics (Vol. 156, Issue 1, pp. 77\-106). 
    Elsevier BV. https://doi.org/10.1006/aima.2000.1949 
    .. [2] Pisier, G. (1989). The Volume of Convex Bodies and Banach Space Geometry. 
    Cambridge University Press. https://doi.org/10.1017/cbo9780511662454 
    """
    EXPECTED_LOSS = 0.495
    s3 = predict_res(2)
    s4 = predict_res(6)

    preds = np.hstack([s3, s4])
    errs = (pattern.numpy() - preds) ** 2

    return EXPECTED_LOSS, errs, preds


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


def get_metrics(model, is_torch_model=True, silent=True):
    complete_dataset_dir = Path(hconfig.COMPLETE_DATASET_DIR)

    global datasets_names, individual_dataset_name, quartiles_dataset_names
    datasets_names = []
    quartiles_dataset_names = []
    individual_dataset_name = []

    losses = []
    metrics_graph = []
    metrics_pattern = []

    all_preds = []
    all_truths = []

    tot_seen = 0
    for j, dataset_name in enumerate(GENERATORS):

        cdir = complete_dataset_dir / dataset_name
        dataset = CustomDataset(
            cdir,
            [dataset_name],
            final_dataset_name=dataset_name,
            has_splits=False,
        )

        tot_seen += len(dataset)
        if not silent:
            print(dataset_name, len(dataset))

        datasets_names.append([dataset_name] * len(dataset))

        if is_torch_model:
            model.eval()
            assert not model.training

        dataloader = torch_geometric.loader.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        loss = []
        loss_fn = torch.nn.MSELoss()

        preds = []
        truths = []
        errs = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                if is_torch_model:
                    batch.to(DEVICE)  # Does not need to reassign

                    y_pred = model(batch.edge_index, batch.x, batch.batch)

                    loss.append(loss_fn(y_pred, batch.y).cpu())
                    preds.append(y_pred.cpu())
                    truths.append(batch.y.cpu())
                else:
                    l, e, _ = model(batch.y[0])
                    loss.append(l)
                    errs.append(e)

        if is_torch_model:
            y_true = torch.cat(truths, dim=0).numpy()
            y_pred = torch.cat(preds, dim=0).numpy()
            errs = (y_true - y_pred) ** 2
        else:
            errs = np.vstack(errs)

        ##
        if is_torch_model:
            all_truths.append(y_true)
            all_preds.append(y_pred)

        ##
        losses.append(loss)
        individual_dataset_name.append(
            pd.read_csv(TRUE_LABELS_PATH / (dataset_name + "_labels.csv"))[
                "GraphName"
            ].to_numpy()
        )

        ##
        if is_torch_model:
            median_absolute_error = np.median(
                skm.median_absolute_error(y_true, y_pred, multioutput="raw_values")
            )
        else:
            median_absolute_error = -1

        whole_pattern_errs = np.sum(errs, axis=1)
        max_squared_err_pattern_idx = np.argmax(whole_pattern_errs)
        max_squared_err_pattern = whole_pattern_errs[max_squared_err_pattern_idx]
        min_squared_err_pattern_idx = np.argmin(whole_pattern_errs)
        min_squared_err_pattern = whole_pattern_errs[min_squared_err_pattern_idx]
        mean_squared_err_pattern = np.mean(whole_pattern_errs)
        metrics_pattern.append(
            [
                median_absolute_error,
                min_squared_err_pattern,
                mean_squared_err_pattern,
                max_squared_err_pattern,
                dataset_name,
            ]
        )

        ##
        q_per_graph_errs = np.quantile(errs, [0.25, 0.5, 0.75, 0.95, 1], axis=0)
        quartiles_dataset_names.append([dataset_name] * q_per_graph_errs.shape[0])
        metrics_graph.append(q_per_graph_errs)

    if "sreal" in str(SAVE_PATH):
        assert tot_seen == 56, f"Not all graphs were seen! {tot_seen} != 56"
    elif "mlreal" in str(SAVE_PATH):  # It is 58 because anybeat as a problem!
        assert tot_seen == 59 - 1, "Not all graphs were seen! {tot_seen} != 58"

    return metrics_pattern, metrics_graph, losses, all_preds, all_truths


def recuperate_test_error(model, is_torch_model=True):
    test_set_dir = Path(TEST_SET_LOCATION)

    dataset = CustomDataset(
        test_set_dir,
        [TEST_SET_DATASET],
        final_dataset_name="TEST_RECUP",
        has_splits=True,
    )

    dataset = dataset[dataset.get_idx_split()["test"]]
    print(f"Using test set with {len(dataset)} examples")

    if is_torch_model:
        model.eval()
        assert not model.training

    dataloader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    loss = []
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            if is_torch_model:
                batch.to(DEVICE)  # Does not need to reassign

                y_pred = model(batch.edge_index, batch.x, batch.batch)

                loss.append(loss_fn(y_pred, batch.y).cpu())
            else:
                raise NotImplementedError("Cannot get test error of non-torch model")

    return np.mean(loss)


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

    if "sreal" in SAVE_PATH.__str__():
        shutil.copy2(
            EXPERIMENT_PATH / best_trial_raw / "params.json",
            CORRECT_SAVE_PATH,
        )
    else:
        shutil.copy2(
            EXPERIMENT_PATH / best_trial_raw / "params.json",
            CORRECT_SAVE_PATH,
        )

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
    synthetic_test_error = -1
    if RECUPERATE_TEST_SET_ERROR:
        synthetic_test_error = recuperate_test_error(model, is_torch_model=True)

    global datasets_names, individual_dataset_name, quartiles_dataset_names

    metrics_pattern, metrics_graph, losses, all_preds, all_truths = get_metrics(
        model, is_torch_model=True
    )

    _, metrics_graph_rand, losses_rand, _, _ = get_metrics(
        random_uniform_model, is_torch_model=False
    )

    loss_rand_params_list = []
    for _ in range(100):
        model.reset_parameters()
        _, _, losses1, _, _ = get_metrics(model, is_torch_model=True, silent=True)
        loss_rand_params_list.append(np.mean(np.hstack(losses1)))
    loss_rand_params = np.mean(loss_rand_params_list)

    loss_rand_res_params_list = []
    for _ in range(100):
        _, _, losses2, _, _ = get_metrics(
            predict_rand_res, is_torch_model=False, silent=True
        )
        loss_rand_res_params_list.append(np.mean(np.hstack(losses2)))
    loss_rand_res_params = np.mean(loss_rand_res_params_list)

    loss_rand = losses_rand[0][0]
    sleep(1)
    ##############################################################################

    # Remove dataset with errors in SP
    for i in range(len(individual_dataset_name)):
        _a = np.delete(
            individual_dataset_name[i],
            np.where(individual_dataset_name[i] == "soc-anybeat"),
        )
        individual_dataset_name[i] = _a

    ###################### Prepare all the dataframes ######################
    df_metrics_pattern = pd.DataFrame(
        metrics_pattern,
        columns=[
            "Median ABS Error",
            "Min SQRD Error",
            "Mean SQRD Error",
            "Max SQRD Error",
            "DatasetName",
        ],
    )
    df_metrics_graph = pd.DataFrame(
        np.vstack(metrics_graph),
        columns=["Subgraph" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_metrics_graph["DatasetName"] = hutils.flatten_nested_list(
        quartiles_dataset_names
    )
    df_metrics_graph["Quartile"] = hutils.flatten_nested_list(
        np.repeat(
            [["0.25", "0.5", "0.75", "0.95", "1"]], len(df_metrics_graph) / 5, axis=0
        ),
        sort=False,
    )

    df_metrics_graph_rand = pd.DataFrame(
        np.vstack(metrics_graph_rand),
        columns=["Subgraph" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_metrics_graph_rand["DatasetName"] = hutils.flatten_nested_list(
        quartiles_dataset_names
    )
    df_metrics_graph_rand["Quartile"] = hutils.flatten_nested_list(
        np.repeat(
            [["0.25", "0.5", "0.75", "0.95", "1"]],
            len(df_metrics_graph_rand) / 5,
            axis=0,
        ),
        sort=False,
    )

    df_metrics_individual = pd.DataFrame(
        np.hstack(losses),
        columns=[
            "Loss",
        ],
    )
    df_metrics_individual["DatasetName"] = hutils.flatten_nested_list(datasets_names)
    _t = hutils.flatten_nested_list(individual_dataset_name, sort=False)
    print(sum((np.unique(_t, return_counts=True)[1])))
    df_metrics_individual["GraphName"] = hutils.flatten_nested_list(
        individual_dataset_name, sort=False
    )
    df_metrics_individual["SQRD Error"] = df_metrics_individual["Loss"].apply(
        lambda x: np.round(x, decimals=3)
    )
    df_metrics_individual["ABS Error"] = df_metrics_individual["Loss"].apply(
        lambda x: np.round(np.sqrt(x), decimals=3)
    )

    df_pred = pd.DataFrame(
        np.vstack(all_preds),
        columns=["G" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_pred["DatasetName"] = hutils.flatten_nested_list(datasets_names)
    df_pred["GraphName"] = hutils.flatten_nested_list(
        individual_dataset_name, sort=False
    )
    df_pred["Type"] = ["Pred"] * df_pred.shape[0]

    df_truth = pd.DataFrame(
        np.vstack(all_truths),
        columns=["G" + str(i) for i in range(hconfig.NUM_SUBGRAPHS)],
    )
    df_truth["DatasetName"] = hutils.flatten_nested_list(datasets_names)
    df_truth["GraphName"] = hutils.flatten_nested_list(
        individual_dataset_name, sort=False
    )
    df_truth["Type"] = ["True"] * df_truth.shape[0]

    df_patterns = pd.concat([df_pred, df_truth])
    if CSV_SAVE_PREDICTIONS:
        _save_p = Path(
            os.path.join(SAVE_PATH / (experiment + "+" + trial + "+" + DATASET_TYPE + ".csv"))
        )
        df_patterns.to_csv(_save_p)

        with open(_save_p, "rb") as f:
            file_data = f.read()

        # Create SHA-256 checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        with open(SAVE_PATH / (_save_p.stem + ".sha256"), "w") as f:
            f.write(checksum)

    ##############################################################################

    ############################## Individual Preds ##############################
    for datasetname in sorted(GENERATORS):
        df_patterns_sub = df_patterns[df_patterns["DatasetName"] == datasetname].drop(
            ["DatasetName"], axis=1
        )
        df_patterns_sub_melt = df_patterns_sub.melt(
            id_vars=["GraphName", "Type"], var_name="SubPattern", value_name="ZScore"
        )

        sns.set_context("paper", font_scale=2)
        plt.figure(figsize=(22, 14))

        STYLE = {
            "marker": ["o", "v"],
            "ls": ["-", "--"],
            "color": [p[9], p[11]],
        }

        g = sns.FacetGrid(
            df_patterns_sub_melt,
            col="GraphName",
            col_wrap=3,
            hue="Type",
            sharey=True,
            sharex=True,
            aspect=16 / 9,
            hue_kws=STYLE,
        )

        g.map_dataframe(
            sns.lineplot,
            x="SubPattern",
            y="ZScore",
            alpha=0.85,
            markeredgecolor=p[0],
        )

        if (
            "INTERACTION" in datasetname
            or ("sreal" in datasetname and "COLLABORATIONCITATION" in datasetname)
            or ("sreal" in datasetname and "INFRASTRUCTURE" in datasetname)
        ):
            g.set(ylim=(-1, 1.15), yticks=[-1, 0, 1])
            g.fig.subplots_adjust(top=0.85)
        elif "sreal" in datasetname and "SOCIALCOMMUNICATION" in datasetname:
            g.set(ylim=(-1, 1.15), yticks=[-1, 0, 1])
            g.fig.subplots_adjust(top=0.75)
        else:
            g.set(
                ylim=(-1, 1.15),
                yticks=np.arange(-1, 1.25, step=0.25),
            )
            g.fig.subplots_adjust(top=0.90)

        pretty_dname = None
        if "mlreal" in datasetname:
            pretty_dname = "Medium-Large - "
            pretty_dname += datasetname.split("mlreal")[1]
        else:
            pretty_dname = "Small - "
            pretty_dname += datasetname.split("sreal")[1]
        # print(pretty_dname)

        g.fig.suptitle(pretty_dname)
        g.set_titles(col_template="{col_name}")
        g.despine(top=True, left=True, right=True)
        g.add_legend()

        plt.savefig(
            CORRECT_SAVE_PATH / (datasetname + "_predictions.pdf"),
            dpi=1200,
            bbox_inches="tight",
        )
        plt.close()
    sleep(1)
    ##############################################################################


if __name__ == "__main__":
    for trial in tqdm.tqdm(TRIALS):
        do_everything(trial)
        sleep(1)
