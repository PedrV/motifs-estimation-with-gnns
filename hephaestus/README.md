[< Back](../README.md)


**WARNING**: If at any point a password is prompted when unzipping files, the password is `hephaestus`. Furthermore, if using the command line to unzip, a password protected file might not ask for the password and extract just empty folders.

## Setting up

1. Create a python virtual environment. Preferably with `venv`. Versions `3.11.X` and `3.10.X` should work. Tested with `3.11.10` and `3.10.12`.
2. Run the script `install_reqs.sh` providing the path to the created environment as the first argument.
    * This script will install the necessary packages with the versions used in the original experiments. Be aware that using a different graphics card on a different CUDA driver might result in slight numerical differences.
3. Change the `home_dir` of [hephaestus/config.ini](./config.ini) to a suitable one.

---

## 1 Generating the synthetic and real graphs and their labels

1. Navigate to [hephaestus/_excluded/gtrieScanner_src_01](./_excluded/gtrieScanner_src_01) and execute the `make` command [^1][^2].
2. Change the `/path/to/env` in [run_gen_and_label.sh](../run_gen_and_label.sh).
3. <u>(synthetic data)</u>: Run [run_gen_and_label.sh](../run_gen_and_label.sh) with the default parameters.
    * With the default parameters, this script will generate all graphs and their labels for the deterministic and non-deterministic segments. The resources used can be controlled by changing [resources.yaml](./_configs/resources/resources.yaml). A high number of workers using less cores per worker is preferable when some generators bottleneck the process. This is more relevant for the non-deterministic generator. Nonetheless, the default values are usable.
4. <u>(real-world data)</u>: Download the raw data using this [link](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/real_data_raw.zip") (`raw_data/real_data_raw.zip`).
5. <u>(real-world data)</u>: Place the folders `sreal_raw` and `mlreal_raw` in the folder `_excluded/test_data`.
6. <u>(real-world data)</u>: Change the `--type` argument in [run_gen_and_label.sh](../run_gen_and_label.sh) to `sreal` and `mlreal` and run the script.
    * If for some networks the message `Failed check on remote website!` appears it only means that the script could not reach the source website to confirm the data. This is not problematic.
    * The networks will be placed in the folder `_excluded/graphs`.

**Warning:** It will take easily more than 1 week to generate all graphs and labels! Alternatives:
- Generate a smaller number of graphs by changing the YAML files for each segment.
- Using only some generators by changing the `nx_available_generators_*` variables in [generation_utils.py](./graph_generation/synthetic/generation_utils.py).
- Download the data used.
    * Non-deterministic segment [here for graphs](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/nd_v1/graphs.tar.gz") and [here for the rest](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/nd_v1/v1_missing_parts.tar.xz") (`raw_data/nd_v1/graphs.tar.gz` and `raw_data/nd_v1/v1_missing_parts.tar.xz`).
    * Deterministic segment [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/d_v2/v2.tar.xz") (`raw_data/d_v2/v2.tar.xz`).
    * Small real-world segment [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/real_data/sreal/sreal-dataset_all.zip") (`raw_data/real_data/sreal/sreal-dataset_all.zip`).
    * Medium-Large real-world segment [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: raw_data/real_data/mlreal/mlreal-dataset_all.zip") (`raw_data/real_data/mlreal/mlreal-dataset_all.zip`).

---

## 2 Generating PyG datasets

1. <u>(synthetic data)</u>: Execute [dataset_main_testing.py](./dataset_creation/dataset_main_testing.py).
    * This script will search the folders `_excluded/graphs`, `_excluded/features` and `_excluded/labels` according to `config.ini` to know what graphs it should use and their labels and features. Hence, if the data was downloaded and not generated, put it in the correct folders.
    * This script will generate two folders `_excluded/complete_dataset_d` and `_excluded/complete_dataset_nd`. 
2. <u>(real-world data)</u>: The real-world datasets are created during the evaluation of the trained models. The code is similar to the one used for the synthetic data. It can be seen in the file [evaluate_models.py](../hephaestus_lab/evaluation/evaluate_predictions/evaluate_models.py) in the function `get_metrics()` and it is taken care of in section [3 Evaluate and Analyse the Predictions](../hephaestus_lab/README.md#3-evaluate-and-analyse-the-predictions).


An alternative to generate the PyG datasets would be to download datasets directly.
- Non-deterministic segment [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: transformed_data/complete_dataset_nd.tar.gz") (`transformed_data/complete_dataset_nd.tar.gz`).
- Deterministic segment [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: transformed_data/complete_dataset_d.zip") (`transformed_data/complete_dataset_d.zip`).
- Small Real [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: transformed_data/complete_dataset_sreal.zip"), (`transformed_data/complete_dataset_sreal.zip`).
- Medium-Large Real [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: transformed_data/complete_dataset_mlreal.zip") (`transformed_data/complete_dataset_mlreal.zip`).

---

## 3 Training the models

1. Make sure the data under `_excluded/complete_dataset` is the desired one. For example, if you want to train with the deterministic data remove the indicator `_d` from the folder name.
2. Change the `/path/to/env` in [train_model.sh](../train_model.sh) and run the script.
    * Running `train_model.sh 0` will use GIN M1, Using `1` will use GCN, `2` GAT and `3` SAGE. The order is given by the order they are present in the array in the config file [optimization_parameters_t.yaml](./_configs/classification_engine_v1/optimization/optimization_parameters_t.yaml).
    * To understand what metrics are reported and how to further configure Ray or turn on Wandb explore the script [train_model.py](./training/train_model.py).

The results of the training procedure can be seen following this [link](Not in the anonymous version). To get the actual results, download them from this [link](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: model_trainning_results/relevant_experiments.tar.xz") (`model_trainning_results/relevant_experiments.tar.xz`). The organisation is as follows:
* GCN with deterministic segment: "TALOS_20240110-163351" and "TAUROI_KHALKEOI_20231231-004707" (2 parts)
* GAT with deterministic segment: "TALOS_20240113-155714" and "TAUROI_KHALKEOI_20240101-170302" (2 parts)
* SAGE with deterministic segment: "TALOS_20240119-150636" and "TAUROI_KHALKEOI_20240108-205055" (2 parts)
* GIN with deterministic segment: "TALOS_20240121-203233" and "TAUROI_KHALKEOI_20240106-193249" (2 parts)
* GIN with non-deterministic segment: "TALOS_20240129-182636" and "TALOS_20240201-180351" (2 parts)
* GAT with non-deterministic segment: "TALOS_20240203-234629" and "TALOS_20240209-190133" (2 parts)
* GCN with non-deterministic segment: "TALOS_20240212-124657" (1 part)
* SAGE with non-deterministic segment: "TALOS_20240216-185022" (1 part)

---

#### Extra Information - What can appear inside `_excluded`?

1. `complete_dataset_*`: Folders with this name correspond have a PyG dataset inside. The characters represented by `*` are the name of the dataset the folder holds. For example, `complete_dataset_ND` holds the non-deterministic segment.
2. `complete_dataset`: Represents exactly the same as point 1., but when the folder name does not have an indicator, it generally means it will be used as train data by the models. This folder is typically created when
3. `datasets_stats`: Folder with CSVs with number of edges and nodes for each graph generated.
4. `features`: Generally speaking, this folder contains the features of the dataset the models use as train data. 
5. `graphs`: Generally speaking, this folder contains the graphs that will compose the dataset the models use as train data.
6. `labels`: Generally speaking, this folder contains the labels of the graphs of the dataset the models use as train data.
7. `raw_scores`: Generally speaking, this folder contains the output of the Gtries for each graph that was in the folder `graphs` when Gtrie was executed.
8. `raw_splits_dir`: Should contain the indexes point to each segment: train, test or validation, of each graph that was in the folder `graphs` when the scripts to create a dataset were executed.
9. `test_data`: Directory where typically the real-world raw data should be placed.
10. `gtrieScanner_src_01`: The used tool to extract the ground truth for the generated graphs.


[^1]: Original website for [Gtries](https://www.dcc.fc.up.pt/gtries/). The version used has slight modifications.
[^2]: *Optionally*, at this point, you can run the unit tests in [hephaestus/unity_tests](./unity_tests) to make sure everything is ok.
