[< Back](../README.md)

The folder [hephaestus_lab](.) has all the files related with the analysis of the results obtained by the trained models [^1]. Furthermore, for some critical data, a SHA file is generated. This file serves as a simple way to guarantee that the data was not altered by mistake. All the results from the scripts in this folder are available for download [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_27-09-2024.zip") (`experiment_results/plots_27-09-2024.zip`). If you just want to replicate this portion of the experiments, putting the path to the unziped `experiment_results/plots_27-09-2024.zip` where there are mentions to "results from evaluate" should "link" all data necessary.

It is recommended that the module `hephaestus` and the folder `hephaestus_lab` to be in the same parent directory.

## 1 Dataset Stats

### [stats_per_dataset_real.ipynb](dataset_stats/stats_per_dataset_real.ipynb)

This notebook will generate every statistic related to the real-world datasets. To execute the notebook do the following:

1. Create a folder with any name e.g. `test_folder` having inside a folder named `graphs`.
2. In the folder `graphs`, put all real-world graphs.
    * The graphs can either be the generated ones or the downloaded ones according to [1 Generating the synthetic and real graphs and their labels](../hephaestus/README.md#1-generating-the-synthetic-and-real-graphs-and-their-labels).
3. Change the paths in the notebook accordingly and run all cells of the notebook. 
    * Feel free to change any path to where the documents are saved.

### [stats_per_dataset.ipynb](dataset_stats/stats_per_dataset.ipynb)

This notebook will generate every statistic related with the synthetic datasets. To execute the notebook do the following:

1. Change the `BASE_MODULO_PATH` to the path where `hephaestus` is located.
2. Change the `DETERMINISTIC_DATA_PATH` and `NON_DETERMINISTIC_DATA_PATH` to the path where `v2` and `v1_missing_parts` were extracted and run all cells of the notebook.
    * Feel free to change any path to where the documents are saved.

### [test_size_three.ipynb](dataset_stats/test_size_three.ipynb)

This notebook will generate the figure related to the separation of the Z-Scores of size 4 given the Z-Scores of size 3. To execute the notebook do the following:

1. Create a folder with any name e.g. `test_folder` having inside a folder named `labels`.
2. In the folder `labels`, put all the labels of the synthetic graphs.
3. Change the paths in the notebook accordingly and run all cells of the notebook. 
    * Feel free to change any path to where the documents are saved.

---

## 2 Compare the Training Results

### [compare_models.ipynb](evaluation/compare_training_progress/compare_models.ipynb)

This notebook will generate the images related with the metrics used to monitor the training of the models. To execute the notebook do the following:

1. Put all the results from the experiments in a folder `model_storage/models`.
    * The experiments could either be the result from training everything from scratch or be the files downloaded according to the section [3 Training the models](../hephaestus/README.md#3-training-the-models).
2. Change the `BASE_PATH` to the path where `hephaestus` is located.
3. Change the `BASE_EXPERIMENT_DIRECTORY` to the folder where `model_storage/models` is located and run all cells.
    * Feel free to change any path to where the documents are saved.

---

## 3 Evaluate and Analyse the Predictions

### [evaluate_models.py](evaluation/evaluate_predictions/evaluate_models.py)

This script generates the predictions and statistics of the predictions for the real-world data, together with some statistics for the synthetic test data. Furthermore, it creates the data for the blue horizontal line, red horizontal line and the predictions with random weights. To execute the script do the following:

1. Change the second argument of `sys.path.insert` to the folder where `hephaestus` is placed. 
2. Define a model of interest. That is, the model you want to test.
3. Change `TRIALS`, `EXPERIMENTS` and `MODEL_TYPE` accordingly.
    * Follow the comments in the file to understand what trial and experiment correspond to each model. [link](Not in the anonymous version) also has this information.
    * For example, to evaluate the GIN model trained with a deterministic segment, select `TorchTrainer_e9bc79c6`, `TALOS_20240121-203233` and `gin`.
4. Following the section [2 Generating the PyG datasets](../hephaestus/README.md#2-generating-pyg-datasets), you should have two folders `_excluded/complete_dataset_d` and `_excluded/complete_dataset_nd`. If not, create them with the data downloaded.
5. Change `TEST_SET_DATASET` to the indicator (d or nd) of the segment the model of interest was trained on.
6. Define a real-world dataset of interest. That is, the dataset the chosen model will be evaluated on.
7. Change `DATASET_TYPE` accordingly. That is, either to `mlreal` or `sreal`.
8. If you downloaded the real-world datasets in the PyG format available in section [2 Generating the PyG datasets](../hephaestus/README.md#2-generating-pyg-datasets) go to step 9 (and skip 10) else go to step 10.
9. Depending on the chosen dataset, place the extracted folder from either `complete_dataset_sreal.zip` or `complete_dataset_mlreal.zip` in the `_excluded` folder without the identifier (the underscore and everything after it).
10. Depending on the chosen dataset, place the folders `graphs`, `features` and `labels` downloaded from the links in section [1 Generating the synthetic and real graphs and their labels](../hephaestus/README.md#1-generating-the-synthetic-and-real-graphs-and-their-labels) in the folder `_excluded`.
11. Change `BASE_EXPERIMENT_DIRECTORY` to the same value used in the previous section.
    * Feel free to change any path to where the documents are saved.
12. Change `TRUE_LABELS_PATH` to the path where the labels of the real-world datasets are stored.
    * The folder could either have just the labels for the `sreal`/`mlreal` and you change the content based on the type of dataset of interest or the folder could have all the labels.
13. Run the script.

### [evaluate_models_reduced.py](evaluation/evaluate_predictions/evaluate_models_reduced.py)

This script complements the one in the previous section. It generates the predictions for the synthetic test data. To execute the script do the following:

1. Change the second argument of `sys.path.insert` to the folder where `hephaestus` is placed. 
2. Define a model of interest. That is, the model you want to test.
3. Change `TRIALS`, `EXPERIMENTS` and `MODEL_TYPE` accordingly.
    * Follow the comments in the file to understand what trial and experiment correspond to each model. [link](Not in the anonymous version) also has this information.
    * For example, to evaluate the GIN model trained with a deterministic segment, select `TorchTrainer_e9bc79c6`, `TALOS_20240121-203233` and `gin`.
4. Following the section [2 Generating the PyG datasets](../hephaestus/README.md#2-generating-pyg-datasets), you should have two folders `_excluded/complete_dataset_d` and `_excluded/complete_dataset_nd`. If not, create them with the data downloaded.
5. Change `DATASET_TYPE` accordingly. That is, either to `nd` or `d`.
6. Change `BASE_EXPERIMENT_DIRECTORY` to the same value used in the previous section.
    * Feel free to change any path to where the documents are saved.
7. Run the script.

### [barplot_models.ipynb](evaluation/evaluate_predictions/barplot_models.ipynb) [TO BE REMOVED]

This notebook generates the barplot with the summary of the results from the two previous sections. To execute the notebook do the following:

1. You must have executed the code for the last two sections with all combinations for the barplot to have the correct format. That is evaluated each of the 3 models in each of the 2 real-world datasets for [evaluate_models.py](evaluation/evaluate_predictions/evaluate_models.py) and executed for the 3 models for [evaluate_models_reduced.py](evaluation/evaluate_predictions/evaluate_models_reduced.py).
    * Alternatively, you can download the results [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_27-09-2024.zip") (`experiment_results/plots_27-09-2024.zip`).
2. Change `BASE_PATH` and `RESULTS_FROM_EVALUATE_MODELS` to the folder where the results from the previous two sections are stored and run all cells.

### [model_preds_synthetic-real.ipynb](evaluation/evaluate_predictions/model_preds_synthetic-real.ipynb) [TO BE MODIFIED]

This notebook generates the lineplot with the comparison of the predictions and the real SPs and the number of "sufficiently correct" predictions per generator. To execute the notebook do the following:

1. You must have executed the code for the last two sections with all combinations for the barplot to have the correct format. That is evaluated each of the 3 models in each of the 2 real-world datasets for [evaluate_models.py](evaluation/compare_training_progress/mini_compare_models.py) and executed for the 3 models for [evaluate_models_reduced.py](evaluation/compare_training_progress/evaluate_models_reduced.py).
    * Alternatively, you can download the results [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_27-09-2024.zip") (`experiment_results/plots_27-09-2024.zip`).
2. Change the argument of `os.chdir` to the folder where `hephaestus` and `hephaestus_lab` are placed.
3. Change `DATA_DIR` to the folder where the results from the two sections regarding the `evaluate_*` scripts are stored.
4. Follow the indications in section [Model Predictions in the Synthetic Dataset](./evaluation/evaluate_predictions/model_preds_synthetic-real.ipynb#model-predictions-in-the-synthethic-dataset) of [model_preds_synthetic-real.ipynb](./evaluation/evaluate_predictions/model_preds_synthetic-real.ipynb) to select the desired dataset and run all cells.

---

## 4 Persistent Patterns [TO BE REMOVED]

This folder has everything related with the experiments regarding the concept of persistent pattern.
To run the scripts do the following:

1. Change the `SAVE_PATH` from [run_persistent_patterns.py](./evaluation/persistent_patterns/run_persistent_patterns.py) and [persistent_patterns.py](./evaluation/persistent_patterns/persistent_patterns.py) to a folder named `persistent_patterns` in the folder where the results from the two sections regarding the `evaluate_*` scripts are stored.
2. Change `DATA_DIR` to the folder where the results from the two sections regarding the `evaluate_*` scripts are stored and run the script [run_persistent_patterns.py](./evaluation/persistent_patterns/run_persistent_patterns.py).

---

## 5 Dropout Experiments [TO BE REMOVED]

The analysis of these experiments is done with [evaluate_models.py](evaluation/evaluate_predictions/evaluate_models.py)

1. Change the second argument of `sys.path.insert` to the folder where `hephaestus` is placed. 
1. Put all the results from the experiments in a folder `model_storage/models`.
    * The experiments could either be the result from training everything from scratch or be the files downloaded from [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/dropout.zip") (`experiment_results/dropout.zip`).
2. Change the variable `REPEATED_NAMES` to `True`, `CSV_SAVE_PREDICTIONS` to `False` and `RECUPERATE_TEST_SET_ERROR` to `False`.
3. Change `MODEL_TYPE` accordingly to the model desired.
    * GIN corresponds to `KUONES_KHRYSEOS&ARGYREOS_20240324-19241` the other to `SAGE`.
4. Change `TRIALS` to the list of experiments e.g. `[TorchTrainer_1ac68_00000_0, TorchTrainer_1ac68_00001_1, TorchTrainer_1ac68_00002_2, ..]`
5. Change the `EXPERIMENTS` to the one corresponding to the desired model e.g. `KUONES_KHRYSEOS&ARGYREOS_20240324-19241` for GIN.
6. Change `DATASET_TYPE` to the one desired.
   * We used `sreal` with GIN and `mlreal` with SAGE.
7. Follow steps 8 through 13 of section caretaking to the script [evaluate_models.py](evaluation/evaluate_predictions/evaluate_models.py).


## 6 Validation of the Assumptions Made

### [validation.ipynb](evaluation/validation/validation.ipynb)

1. For variables `PATH`, `PATH2` and `PATH3` either all steps until now where executed correctly and you have the files, or you may use the ones available for download [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_27-09-2024.zip") (`experiment_results/plots_27-09-2024.zip`) and [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/sreal/raw_scores.zip") (`experiment_results/sreal/raw_scores.zip`).
2. For the final cell, the file must be downloaded from [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/multi-vs-islated.csv") (`experiment_results/multi-vs-islated.csv`)
3. Once the files are present, run all cells.


[^1]: Handling these scripts might be a little complicated. Initially I did not design them with the intention of them being "easy to run by another person". Apologies for that.
