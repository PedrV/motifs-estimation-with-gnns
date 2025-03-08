"""
Script to train the classification using Ray, Optuna and Wandb.
"""

import os
import sys
import tempfile
from pathlib import Path
import argparse
import datetime

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.getcwd())

import wandb as wb

import random
import numpy as np
import sklearn.metrics as skm

import torch
import torch.utils.data as torch_data
import torch_geometric

from ray import tune, train
from ray.train.torch import TorchTrainer

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import setup_wandb

from hephaestus.training.training_utils import Unloader, EarlyStopper
from hephaestus.models.classfication_engine import ClassificationEngineV1

import hephaestus.utils.load_general_config as hconfig
import hephaestus.dataset_creation.custom_dataset as hdataset

SEED = 42
NAME = "EXPNAME" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
STORAGE_PATH = os.path.abspath(os.path.expanduser("~/ray_results/"))
PROJECT_NAME = "PROJNAME"

COLOR_PRED = "deeppink"
COLOR_TRUE = "darkturquoise"
STYLE_PRED = {"marker": "o", "linestyle": "solid", "color": COLOR_PRED}
STYLE_TRUE = {"marker": "v", "linestyle": "dashed", "color": COLOR_TRUE}

MPGNN = None
DECODER = None
EPOCHS = None


with open(
    os.path.join(
        hconfig.CLASSIFICATION_ENGINE_V1_OPTIMIZATION_PARAM_PATH,
        "optimization_parameters_t.yaml",
    ),
    mode="r",
    encoding="utf-8",
) as f:
    params = yaml.safe_load(f)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_func(config):
    # print(os.environ["WORLD_SIZE"])
    train.torch.enable_reproducibility(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    if TURN_ON_WANDB:
        wandb = setup_wandb(config, project=PROJECT_NAME)

    global EPOCHS, MPGNN, DECODER
    epochs = EPOCHS
    mpgnn = MPGNN
    decoder = DECODER

    batch_size = config["batch_size"]
    lr = config["lr"]

    # Get dataloaders inside worker training function
    dataset = hdataset.CustomDataset(
        Path(hconfig.COMPLETE_DATASET_DIR), list(hconfig.NDETERMINISTIC_DATA.values())
    )
    split_idx = dataset.get_idx_split()

    train_dataloader = torch_geometric.loader.DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_dataloader = torch_geometric.loader.DataLoader(
        dataset[split_idx["validate"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    un_train_dataloader = torch_data.DataLoader(
        Unloader(train_dataloader),
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    un_valid_dataloader = torch_data.DataLoader(
        Unloader(valid_dataloader),
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    train_dataloader_geom = train.torch.prepare_data_loader(un_train_dataloader)
    valid_dataloader_geom = train.torch.prepare_data_loader(un_valid_dataloader)
    
    del train_dataloader, valid_dataloader
    del un_train_dataloader, un_valid_dataloader

    model = ClassificationEngineV1(
        mpgnn=mpgnn,
        mpgnn_depth=config["mpgnn_depth"],
        mpgnn_hidden_dim=config["mpgnn_hidden_dim"],
        mpgnn_pool=config["pooling"],
        mpgnn_dropout=config["mpgnn_dropout"],
        mpgnn_jk=config["mpgnn_jk"],
        decoder=decoder,
        decoder_hidden_dim=config["decoder_hidden_dim"],
        decoder_dropout=config["decoder_dropout"],
        decoder_depth=config["decoder_depth"],
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        **{},
    )
    model = train.torch.prepare_model(model)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch, crash_wandb = 1, False
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "model_checkpoint.pt")
            )
            optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "optimizer_checkpoint.pt")
            )
            start_epoch = (
                torch.load(
                    os.path.join(loaded_checkpoint_dir, "extra_state_checkpoint.pt")
                )["epoch"]
                + 1
            )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Model training loop
    early_stopper = EarlyStopper(grace_period=25, global_patience=15)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0
        for edge_index, x, y, batch in train_dataloader_geom:
            # Remove batch_dim, data is shared across tensors
            edge_index = torch.squeeze(edge_index, dim=0)
            x = torch.squeeze(x, dim=0)
            y = torch.squeeze(y, dim=0)
            batch = torch.squeeze(batch, dim=0)

            if x.shape[0] == 1 or batch[-1] == 0:  # Avoid single graph batching
                pass
            else:
                out = model(edge_index, x, batch)
                loss = loss_fn(out, y.float())
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_loss /= len(train_dataloader_geom)

        model.eval()
        valid_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for edge_index, x, y, batch in valid_dataloader_geom:
                # Remove batch_dim, data is shared across tensors
                edge_index = torch.squeeze(edge_index, dim=0)
                x = torch.squeeze(x, dim=0)
                y = torch.squeeze(y, dim=0)
                batch = torch.squeeze(batch, dim=0)

                if x.shape[0] == 1 or batch[-1] == 0:  # Avoid single graph batching
                    pass
                else:
                    pred = model(edge_index, x, batch)
                    loss = loss_fn(pred, y.float())

                    valid_loss += loss.item()
                    y_true.append(y.detach().cpu())
                    y_pred.append(pred.detach().cpu())

        valid_loss /= len(valid_dataloader_geom)
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        median_absolute_error = np.median(
            skm.median_absolute_error(y_true, y_pred, multioutput="raw_values")
        )

        whole_pattern_errs = np.sum((y_true - y_pred) ** 2, axis=1)
        max_squared_err_pattern_idx = np.argmax(whole_pattern_errs)
        max_squared_err_pattern = whole_pattern_errs[max_squared_err_pattern_idx]

        min_squared_err_pattern_idx = np.argmin(whole_pattern_errs)
        min_squared_err_pattern = whole_pattern_errs[min_squared_err_pattern_idx]

        mean_squared_err_pattern = np.mean(whole_pattern_errs)

        per_graph_errs = np.sum((y_true - y_pred) ** 2, axis=0)
        mean_worse_squared_errs_graph_idx = np.argmax(per_graph_errs)

        q_per_graph_errs = np.quantile(
            (y_true - y_pred) ** 2, [0.25, 0.5, 0.75, 0.95, 1], axis=0
        )

        ########################################
        # LEGACY METRICS TO BE REMOVED
        max_abs_errs_pattern = np.sum(abs(y_true - y_pred), axis=1)
        max_abs_err_pattern_idx = np.argmax(max_abs_errs_pattern)
        max_abs_err_pattern = max_abs_errs_pattern[max_abs_err_pattern_idx]

        mean_worse_abs_errs_graph = np.sum(abs(y_true - y_pred), axis=0)
        mean_worse_abs_errs_graph_idx = np.argmax(mean_worse_abs_errs_graph)
        mean_worse_abs_errs_graph = mean_worse_abs_errs_graph[
            mean_worse_abs_errs_graph_idx
        ] / (len(split_idx["validate"]))
        ########################################

        should_checkpoint = epoch % 1 == 0 or epoch + 1 == epochs
        should_log_media = epoch % 25 == 0 or epoch + 1 == epochs
        if TURN_ON_WANDB and should_log_media:
            worst_pattern_pred = y_pred[max_abs_err_pattern_idx, :]
            worst_pattern_true = y_true[max_abs_err_pattern_idx, :]
            plt.plot(range(len(worst_pattern_true)), worst_pattern_pred, **STYLE_PRED)
            plt.plot(range(len(worst_pattern_true)), worst_pattern_true, **STYLE_TRUE)
            wandb.log({"plot": wb.Image(plt)})
            plt.close()

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if train.get_context().get_world_rank() == 0 and should_checkpoint:
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model_checkpoint.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(temp_checkpoint_dir, "optimizer_checkpoint.pt"),
                )
                torch.save(
                    {"epoch": epoch, "versions": model.my_versions},
                    os.path.join(temp_checkpoint_dir, "extra_state_checkpoint.pt"),
                )
                checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)

            if TURN_ON_WANDB:
                wandb.log(
                    {
                        "loss": valid_loss,
                        "train_loss": train_loss,
                        "med_abs_error": median_absolute_error,
                        "max_squared_err_pattern_idx": max_squared_err_pattern_idx,
                        "max_squared_err_pattern": max_squared_err_pattern,
                        "min_squared_err_pattern_idx": min_squared_err_pattern_idx,
                        "min_squared_err_pattern": min_squared_err_pattern,
                        "mean_squared_err_pattern": mean_squared_err_pattern,
                        "mean_worse_squared_errs_graph_idx": mean_worse_squared_errs_graph_idx,
                        "q11": q_per_graph_errs[0][0],
                        "q12": q_per_graph_errs[1][0],
                        "q13": q_per_graph_errs[2][0],
                        "q14": q_per_graph_errs[3][0],
                        "q15": q_per_graph_errs[4][0],

                        "q21": q_per_graph_errs[0][1],
                        "q22": q_per_graph_errs[1][1],
                        "q23": q_per_graph_errs[2][1],
                        "q24": q_per_graph_errs[3][1],
                        "q25": q_per_graph_errs[4][1],

                        "q31": q_per_graph_errs[0][2],
                        "q32": q_per_graph_errs[1][2], 
                        "q33": q_per_graph_errs[2][2],
                        "q34": q_per_graph_errs[3][2],
                        "q35": q_per_graph_errs[4][2],

                        "q41": q_per_graph_errs[0][3],
                        "q42": q_per_graph_errs[1][3],
                        "q43": q_per_graph_errs[2][3],
                        "q44": q_per_graph_errs[3][3],
                        "q45": q_per_graph_errs[4][3],

                        "q51": q_per_graph_errs[0][4],
                        "q52": q_per_graph_errs[1][4],
                        "q53": q_per_graph_errs[2][4],
                        "q54": q_per_graph_errs[3][4],
                        "q55": q_per_graph_errs[4][4],

                        "q61": q_per_graph_errs[0][5],
                        "q62": q_per_graph_errs[1][5],
                        "q63": q_per_graph_errs[2][5],
                        "q64": q_per_graph_errs[3][5],
                        "q65": q_per_graph_errs[4][5],

                        "q71": q_per_graph_errs[0][6],
                        "q72": q_per_graph_errs[1][6],
                        "q73": q_per_graph_errs[2][6],
                        "q74": q_per_graph_errs[3][6],
                        "q75": q_per_graph_errs[4][6],

                        "q81": q_per_graph_errs[0][7],
                        "q82": q_per_graph_errs[1][7],
                        "q83": q_per_graph_errs[2][7],
                        "q84": q_per_graph_errs[3][7],
                        "q85": q_per_graph_errs[4][7],

                        "max_abs_err_pattern_idx": max_abs_err_pattern_idx,
                        "max_abs_err_pattern": max_abs_err_pattern,
                        "mean_worse_abs_errs_graph_idx": mean_worse_abs_errs_graph_idx,
                        "mean_worse_abs_errs_graph": mean_worse_abs_errs_graph,
                    }
                )
            train.report(
                {
                    "loss": valid_loss,
                    "train_loss": train_loss,
                    "med_abs_error": median_absolute_error,
                    "max_squared_err_pattern_idx": max_squared_err_pattern_idx,
                    "max_squared_err_pattern": max_squared_err_pattern,
                    "min_squared_err_pattern_idx": min_squared_err_pattern_idx,
                    "min_squared_err_pattern": min_squared_err_pattern,
                    "mean_squared_err_pattern": mean_squared_err_pattern,
                    "mean_worse_squared_errs_graph_idx": mean_worse_squared_errs_graph_idx,
                        "q11": q_per_graph_errs[0][0],
                        "q12": q_per_graph_errs[1][0],
                        "q13": q_per_graph_errs[2][0],
                        "q14": q_per_graph_errs[3][0],
                        "q15": q_per_graph_errs[4][0],

                        "q21": q_per_graph_errs[0][1],
                        "q22": q_per_graph_errs[1][1],
                        "q23": q_per_graph_errs[2][1],
                        "q24": q_per_graph_errs[3][1],
                        "q25": q_per_graph_errs[4][1],

                        "q31": q_per_graph_errs[0][2],
                        "q32": q_per_graph_errs[1][2], 
                        "q33": q_per_graph_errs[2][2],
                        "q34": q_per_graph_errs[3][2],
                        "q35": q_per_graph_errs[4][2],

                        "q41": q_per_graph_errs[0][3],
                        "q42": q_per_graph_errs[1][3],
                        "q43": q_per_graph_errs[2][3],
                        "q44": q_per_graph_errs[3][3],
                        "q45": q_per_graph_errs[4][3],

                        "q51": q_per_graph_errs[0][4],
                        "q52": q_per_graph_errs[1][4],
                        "q53": q_per_graph_errs[2][4],
                        "q54": q_per_graph_errs[3][4],
                        "q55": q_per_graph_errs[4][4],

                        "q61": q_per_graph_errs[0][5],
                        "q62": q_per_graph_errs[1][5],
                        "q63": q_per_graph_errs[2][5],
                        "q64": q_per_graph_errs[3][5],
                        "q65": q_per_graph_errs[4][5],

                        "q71": q_per_graph_errs[0][6],
                        "q72": q_per_graph_errs[1][6],
                        "q73": q_per_graph_errs[2][6],
                        "q74": q_per_graph_errs[3][6],
                        "q75": q_per_graph_errs[4][6],

                        "q81": q_per_graph_errs[0][7],
                        "q82": q_per_graph_errs[1][7],
                        "q83": q_per_graph_errs[2][7],
                        "q84": q_per_graph_errs[3][7],
                        "q85": q_per_graph_errs[4][7],

                    "max_abs_err_pattern_idx": max_abs_err_pattern_idx,
                    "max_abs_err_pattern": max_abs_err_pattern,
                    "mean_worse_abs_errs_graph_idx": mean_worse_abs_errs_graph_idx,
                    "mean_worse_abs_errs_graph": mean_worse_abs_errs_graph,
                },
                checkpoint=checkpoint,
            )

        if early_stopper.early_stop(valid_loss, epoch):
            crash_wandb = True
            break

    if TURN_ON_WANDB and not crash_wandb:
        wandb.finish()  # Keep pattern of "crashing" on early-stop, sry wandb :(


def train_engine(
    mpgnn,
    decoder,
    smoke_test=False,
    restore_name=None,
    unfinished_trials=False,
    continue_optimization=False,
):
    decoder_general_params = params["Decoder_general_params"]
    mpgnns_general_params = params["MPGNNs_general_params"]
    general_params = params["General_Params"]

    if unfinished_trials and continue_optimization:
        raise ValueError(f"Either resume experiment or optimization!")

    trials = 450
    if smoke_test:
        print("Peforming a 'smoke-test'... ")
        trials = 1
        general_params["epochs"] = 5

    name, name_for_opt = NAME, None
    if unfinished_trials:
        if restore_name is None:
            raise ValueError("Name of the experiment to restore cannot be None!")
        name = restore_name
    elif continue_optimization:
        if restore_name is None:
            raise ValueError("Name of the experiment to restore cannot be None!")
        name_for_opt = restore_name

    random.seed(SEED)
    np.random.seed(SEED)

    global EPOCHS, MPGNN, DECODER
    EPOCHS = general_params["epochs"]
    MPGNN = mpgnn
    DECODER = decoder

    config = {
        "mpgnn_jk": tune.choice(mpgnns_general_params["jk"]),
        "mpgnn_depth": tune.randint(
            mpgnns_general_params["depth"][0], mpgnns_general_params["depth"][1] + 1
        ),
        "mpgnn_hidden_dim": tune.randint(
            mpgnns_general_params["hidden_dim"][0],
            mpgnns_general_params["hidden_dim"][1] + 1,
        ),
        "mpgnn_dropout": tune.uniform(
            mpgnns_general_params["dropout"][0], mpgnns_general_params["dropout"][1]
        ),
        "decoder_depth": tune.randint(
            decoder_general_params["depth"][0], decoder_general_params["depth"][1] + 1
        ),
        "decoder_hidden_dim": tune.randint(
            decoder_general_params["hidden_dim"][0],
            decoder_general_params["hidden_dim"][1] + 1,
        ),
        "decoder_dropout": tune.uniform(
            decoder_general_params["dropout"][0], decoder_general_params["dropout"][1]
        ),
        "pooling": tune.choice(params["Pooling"]),
        "lr": tune.loguniform(
            general_params["learning_rate"][0], general_params["learning_rate"][1]
        ),
        "batch_size": tune.choice(general_params["batch_size"]),
    }

    resource_scaling_config = train.ScalingConfig(
        trainer_resources={"CPU": 1},
        use_gpu=True,
        num_workers=1,
        resources_per_worker={"GPU": 1},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=resource_scaling_config,
        torch_config=train.torch.TorchConfig(backend="nccl"),
    )

    search_alg = OptunaSearch(seed=SEED)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)

    if continue_optimization and os.path.exists(
        os.path.join(STORAGE_PATH, name_for_opt)
    ):
        print("RESTORING OPTIMIZATION STATE ...")
        search_alg.restore_from_dir(os.path.join(STORAGE_PATH, name_for_opt))

    median_scheduler = tune.schedulers.MedianStoppingRule(
        time_attr="training_iteration",
        min_samples_required=25,
        grace_period=25,
    )

    tuner_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=median_scheduler,
        search_alg=search_alg,
        max_concurrent_trials=None,  # Third-party searcher, leave concurrency logic to them (ConcurrencyLimiter)
        num_samples=trials,  # Do 10 trials, each grid 10 times
    )

    run_config = train.RunConfig(
        name=name,
        storage_path=STORAGE_PATH,
        log_to_file=False,
        failure_config=train.FailureConfig(fail_fast=True),
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    )

    path = os.path.join(STORAGE_PATH, name)
    if unfinished_trials and tune.Tuner.can_restore(path):
        print("RESTORING PREVIOUS EXPERIMENT ...")
        tuner = tune.Tuner.restore(
            path, trainable=trainer, resume_unfinished=True, restart_errored=True
        )
    elif continue_optimization:
        tuner = tune.Tuner(
            trainable=trainer,
            tune_config=tuner_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner(
            trainable=trainer,
            tune_config=tuner_config,
            run_config=run_config,
            param_space={"train_loop_config": config},
        )

    tuner.fit()


if __name__ == "__main__":
    available_mpgnns = params["MPGNN"]
    available_decoders = params["Decoder"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        dest="smoke_test",
        help="Finish quickly for testing",
    )
    parser.add_argument(
        "--mpgnn",
        dest="mpgnn",
        type=int,
        choices=range(len(available_mpgnns)),
        default=0,
        help="Select MPGNN through its index in parameters.yaml",
    )
    parser.add_argument(
        "--decoder",
        dest="decoder",
        type=int,
        choices=range(len(available_decoders)),
        default=0,
        help="Select 'decoder' through its index in parameters.yaml",
    )
    parser.add_argument(
        "--restore-experiment",
        dest="restore_name",
        type=str,
        help="Name of the experminet to restore",
    )
    parser.add_argument(
        "--unfinished-trials",
        action="store_true",
        dest="unfinished_trials",
        help="Did any trial exited with error or was left unifinished?\n This option is only relevant when used with --restore-experiment",
    )
    parser.add_argument(
        "--continue-optimization",
        action="store_true",
        dest="cont_opt",
        help="Will restore the Searcher state (assuming the Searcher supports it) and create a new experiment from it\n This option is only relevant when used with --restore-experiment as it will be used to know where to get experiment from.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        dest="use_wandb",
        help="If active, Wandb will be used to log the results.",
    )
    args, _ = parser.parse_known_args()

    global TURN_ON_WANDB
    TURN_ON_WANDB = args.use_wandb

    hdataset.prepare_directories(clean=False)
    a = hdataset.CustomDataset(
        Path(hconfig.COMPLETE_DATASET_DIR), list(hconfig.NDETERMINISTIC_DATA.values())
    )  
    # make first call to build the dataset and save time, if the dataset exists, the second argument is not important
    # If the dataset exists, this is as no effect 
    del a
    
    train_engine(
        available_mpgnns[args.mpgnn],
        available_decoders[args.decoder],
        args.smoke_test,
        args.restore_name,
        args.unfinished_trials,
        args.cont_opt,
    )
