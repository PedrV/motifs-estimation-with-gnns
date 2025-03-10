# Studying and Improving Graph Neural Network-based Motif Estimation


**Authors:** -.


*Abstract*

Graph Neural Networks (GNNs) are a predominant method for graph representation learning. However, beyond subgraph frequency estimation, their application to network motif significance-profile (SP) prediction remains under-explored, with no established benchmarks in the literature. We propose to address this problem, framing SP estimation as task independent of subgraph frequency estimation. Our approach shifts from frequency counting to direct SP estimation and modulates the problem as multitarget regression. The reformulation is optimised for interpretability, stability and scalability on large graphs. We validate our method using a large synthetic dataset and further test it on real-world graphs. Our experiments reveal that 1-WL limited models struggle to make precise estimations of SPs. However, they can generalise to approximate the graph generation processes of networks by comparing their predicted SP with the ones originating from synthetic generators. This first study on GNN-based motif estimation also hints at how using direct SP estimation can help go past the theoretical limitations that motif estimation faces when performed through subgraph counting.

---

## [Hephaestus](hephaestus/README.md) - Code for Experiments

The folder [hephaestus](hephaestus) has all the code used to generate the graphs (synthetic and real) and their labels and features. Furthermore, it has all the code to define the models and train them. Follow the README in the said folder to understand how to reproduce each step of the experiments.

## [Hephaestus Lab](hephaestus_lab/README.md) - Code for Analysis

The folder [hephaestus_lab](hephaestus_lab) has all the code used to analyse the results from training the multiple models used. Follow the README in the said folder to understand how to reproduce each step of the analysis made.

## [Real World Data](real_world_data/README.md) - Data

The folder [real_world_data](real_world_data) has some images of the outputs of the various models together with source for the real-world data and indications on how to download the real-world data and the synthetic data.


---

<sub><sup>Why are the folders named hephaestus? I needed a name to start the project and I like greek myths and legends :).</sup></sub>
