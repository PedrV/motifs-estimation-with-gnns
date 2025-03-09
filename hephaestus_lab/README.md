[< Back](../README.md)

The folder [hephaestus_lab](.) has all the files related with the analysis of the results obtained by the trained models [^1]. Furthermore, for some critical data, a SHA file is generated. This file serves as a simple way to guarantee that the data was not altered by mistake. All the results from the scripts in this folder are available for download [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_7-03-2025.zip") (`experiment_results/plots_7-03-2025.zip`).

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

1. Put all the results from the experiments in a folder.
    * The experiments could either be the result from training everything from scratch or be the files downloaded according to the section [3 Training the models](../hephaestus/README.md#3-training-the-models).
2. Change the `BASE_PATH` to the path where `hephaestus` is located.
3. Change the `BASE_EXPERIMENT_DIRECTORY` to the folder where the models are located and run all cells.
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
    * You may see an empty directory besides the files with predictions. That is ok!


### [inter-intra-predictions.ipynb](evaluation/evaluate_predictions/inter-intra-predictions.ipynb)

This notebook generates the heatmaps and the base (no customization) for the tables used in the paper. It also generates the plots for a more detailed comparison of how many predictions were considered correct for multiple threshold (two figures in the additional materials).

1. You must have executed the code for the last two sections with all combinations. That is evaluated each of the 3 models in each of the 2 real-world datasets for [evaluate_models.py](evaluation/evaluate_predictions/evaluate_models.py) and executed for the 3 models for [evaluate_models_reduced.py](evaluation/evaluate_predictions/evaluate_models_reduced.py).
    * Alternatively, you can download the results [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_7-03-2025.zip") (`experiment_results/plots_7-03-2025.zip`).
    * If you download the files, all the predictions are already inside a single folder `..../experiment_results/plots_7-03-2025/plots/evaluate_models/CORRECTIONS/preds`. If you generated the predictions using the previous sections, similarly to the downloaded content, put all of the predictions in a single folder.
2. Change the argument of `os.chdir` to the folder where `hephaestus` and `hephaestus_lab` are placed.
3. Change `DATA_DIR` to the folder where all the predictions are stored.
4. The cell with the function `def most_similar()` takes around 6h to run. If you do not want to do this, the folder `..../experiment_results/plots_7-03-2025/plots/evaluate_models/CORRECTIONS/inter-generator/similarities` has the precomputed values. Replace the path at the start of each of the three following cells with the one pointing to the correct data.
5. The cells past "Detailed Comparison" will require rerunning the second cell after "Correct Predictions" (search for `# np.arange(0.05, 0.5, 0.01)`). After doing this, you can run the rest of the cells.

### [inter-predictions-shape.ipynb](evaluation/evaluate_predictions/inter-predictions-shape.ipynb)

This notebook has the comparison per generator of the average predicted significance-profile against the true one. This images are present only in the additional material.   

1. Follow exactly the procedure from points 1 to 3 from the previous section.
2. Follow the indications in the markdown cell named "Inter-generator Predictions by Shape".

---


## 4 Validation of the Assumptions Made

### [validation.ipynb](evaluation/validation/validation.ipynb)

This notebook generates the tables (no formatting) that compare the multi-target regression against single-target and direct SP prediction to subgraph estimation.

1. For variables `PATH`, `PATH2` and `PATH3` either all steps until now where executed correctly and you have the files, or you may use the ones available for download [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/plots_7-03-2025.zip") (`experiment_results/plots_7-03-2025.zip`) and [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/sreal/raw_scores.zip") (`experiment_results/sreal/raw_scores.zip`).
   * The correct path for the `PATH` variables correspond to one that leads to the files with the predictions made by the models and the raw count as estimated by GTrie. Concretely, `PATH` wants the ND predictions, `PATH2` the Small Real predictions and `PATH3` the raw score estimated by Gtrie for the Small Real dataset.
2. For the final cell, the file must be downloaded from [here](https://figshare.com/s/794d3e3dc66ee09c0e86 "Figshare: experiment_results/multi-vs-isolated.csv") (`experiment_results/multi-vs-isolated.csv`)
3. Once the files are present, run all cells.


[^1]: Handling these scripts might be a little complicated. Initially I did not design them with the intention of them being "easy to run by another person". Apologies for that.
