"""
Holds the class that generates synthethic graphs.
"""

import os
from pathlib import Path

import yaml

import multiprocessing

import igraph
import networkx as nx

import random
import numpy as np
import pandas as pd

from datetime import datetime

from hephaestus.utils import load_general_config as hconfig
from hephaestus.utils import general_utils as hutils
from hephaestus.graph_generation.synthetic import generation_utils as gen_utils


class Generators:
    _nd_config_path = Path(hconfig.NDETERM_GEN_PARAM_PATH)
    _d_config_path = Path(hconfig.DETERM_GEN_PARAM_PATH)
    logger = None

    ND_TYPE = 1
    D_TYPE = 0

    TOTAL_CPU_FRACTION_TO_USE = None

    def __init__(
        self,
        graph_dir,
        stats_dir,
        gen_type,
        total_number_of_graphs=-1,
        proportion_for_generator=None,
        proportion_per_cycle=None,
    ):
        super().__init__()

        self.gen_type = gen_type
        self.graph_dir = graph_dir
        self.stats_dir = stats_dir
        self.proportion_per_cycle = proportion_per_cycle

        if self.gen_type == Generators.D_TYPE:
            self.generators = gen_utils.nx_available_generators_d
            file_path = Generators._d_config_path / "deterministic_gen_params.yaml"
            self.logger_name = "DeterministicGen"
        else:
            self.generators = gen_utils.nx_available_generators_nd
            file_path = Generators._nd_config_path / "ndeterministic_gen_params.yaml"
            self.logger_name = "NDeterministicGen"

        with open(file_path) as f:
            self.config = yaml.safe_load(f)

        if proportion_for_generator is None:
            self.proportion_for_generator = dict(
                [(gen, 1 / len(self.generators)) for gen in self.generators]
            )
        else:
            self.proportion_for_generator = proportion_for_generator

        if np.abs(sum(self.proportion_for_generator.values()) - 1) > 0.01:
            raise ValueError(
                "Proportions must sum to 1, got ",
                sum(self.proportion_for_generator.values()),
            )

        self.number_of_generators = len(self.generators)
        if total_number_of_graphs == -1:
            _cyc = len(list(self.config["Cycles"].values())[0])
            self.total_number_of_graphs = self.number_of_generators * self.config["AmountOfGraphs"] * _cyc
        else:
            self.total_number_of_graphs = total_number_of_graphs

        with open(Path(hconfig.RESOURCES_PATH) / "resources.yaml") as f:
            Generators.TOTAL_CPU_FRACTION_TO_USE = yaml.safe_load(f)[
                "graph_generation"
            ]["TOTAL_CPU_FRACTION_TO_USE"]

        Generators.logger = hutils.get_logging_function(self.logger_name)

    def generate(self):
        cpu_count = min(
            round(multiprocessing.cpu_count() * Generators.TOTAL_CPU_FRACTION_TO_USE),
            self.number_of_generators,
        )
        chunk_size = int(np.ceil(len(self.generators) / cpu_count))

        generators_params = []
        cycle_params = []
        total_graphs_for_gen = []
        graphs_per_param = []

        for i, gen in enumerate(self.generators):
            cycle_params.append(self.config["Cycles"][gen])

            # Repeate the params for every cycle. Every cycle uses the same parameters.
            generators_params.append(
                [self.config["Params"][gen] for _ in range(len(cycle_params[i]))]
            )

            # Define the number of graphs a generator is allowed to generate.
            n = np.round(
                self.total_number_of_graphs * self.proportion_for_generator[gen]
            )
            total_graphs_for_gen.append(n)

            param_length = len(self.config["Params"][gen])

            proportion_per_cycle = self.config["ProportionPerCycles"][gen]
            if np.abs(sum(proportion_per_cycle) - 1) > 0.01:
                raise ValueError(
                    "Proportions per cycle must sum to 1, got ",
                    sum(proportion_per_cycle),
                )

            # Defines the proportion of graphs (n) that each cycle has.
            available_graphs_per_cycle = [
                ppc * total_graphs_for_gen[i] for ppc in proportion_per_cycle
            ]
            # Defines the amount of graphs each parameter of each cycle has.
            # Assumes each parameter will have an equal amount of graphs within a given cycle.
            step_size_per_cycle = [
                np.round(agpc / param_length) for agpc in available_graphs_per_cycle
            ]

            number_graphs_to_gen = []
            for cycle_id in range(len(step_size_per_cycle)):
                _temp_graphs_to_gen = []
                for j in range(param_length):
                    n = (
                        step_size_per_cycle[cycle_id]
                        if available_graphs_per_cycle[cycle_id] > 0
                        else 0
                    )

                    if (
                        available_graphs_per_cycle[cycle_id]
                        - step_size_per_cycle[cycle_id]
                        < 0
                        or j + 1 == param_length
                    ):
                        n = max(available_graphs_per_cycle[cycle_id], 0)

                    available_graphs_per_cycle[cycle_id] -= n
                    _temp_graphs_to_gen.append(round(n))  # TODO: Remove round

                number_graphs_to_gen.append(_temp_graphs_to_gen)
            graphs_per_param.append(number_graphs_to_gen)

        graph_dir = [self.graph_dir] * self.number_of_generators
        seeds = [42] * self.number_of_generators

        Generators.logger.info("Starting Parameter Verification ...")
        for g, gp, gpp, tgg, cp, gd, s in zip(
            self.generators,
            generators_params,
            graphs_per_param,
            total_graphs_for_gen,
            cycle_params,
            graph_dir,
            seeds,
        ):
            _verify_params(g, gp, gpp, tgg, cp, gd, s)
        del g, gp, gpp, tgg, cp, gd, s

        Generators.logger.info("Starting Generators ...")
        s = datetime.now()
        with multiprocessing.Pool(processes=cpu_count) as pool:
            results = pool.starmap(
                _generate,
                zip(
                    self.generators,
                    generators_params,
                    graphs_per_param,
                    total_graphs_for_gen,
                    cycle_params,
                    graph_dir,
                    seeds,
                ),
                chunksize=chunk_size,
            )

        stats = []
        for d, gen, e in results:
            stats.append(d)
            Generators.logger.info(f"{gen} completed in {e - s}.")

        df_stats = pd.concat(stats)
        df_stats.to_csv(
            os.path.join(self.stats_dir, self.logger_name + "_stats.csv"), index=False
        )

        Generators.logger.info(f"{self.total_number_of_graphs} graphs generated!")


def _verify_params(
    generator,
    generator_params,
    graphs_per_param,
    total_graphs_for_gen,
    cycle_params,
    graph_dir,
    seed,
):
    if len(cycle_params) != len(generator_params):
        raise ValueError(
            f"All cycles must have parameters."
            f"Got {len(cycle_params)} cycles and {len(generator_params)} params."
        )

    for i in range(len(generator_params)):
        for j in range(len(generator_params)):
            if generator_params[i] != generator_params[j]:
                raise ValueError(
                    f"Supports only equal parameters across cycles."
                    f"Got {generator_params[i]} and {generator_params[j]}."
                )

    if np.sum(graphs_per_param) != total_graphs_for_gen:
        raise ValueError(
            f"Sum of graphs p/ param for all cycles must equal the total graphs. "
            f"Got {np.sum(graphs_per_param)} and {total_graphs_for_gen}."
        )

    if len(generator_params[0]) != len(graphs_per_param[0]):
        raise ValueError(
            f"Each param has to have the correct quantity of graphs to generate."
            f"Got {len(generator_params)} params and {len(graphs_per_param)} graphs p/ param."
        )

    if (
        generator not in gen_utils.nx_available_generators_d
        and generator not in gen_utils.nx_available_generators_nd
    ):
        raise ValueError(f"{generator} is not one of the available generators.")

    if not isinstance(seed, int):
        raise ValueError(f"Seed must be int, got {type(seed)}.")

    if not os.path.exists(graph_dir):
        raise ValueError(f"Directory {graph_dir} does not exist.")


def _random_edge(rng, graph, num_edges, percentage_of_edges=-1.0, del_orig=True):
    # nonedges = list(nx.non_edges(graph))
    edges = list(graph.edges)
    num_edges = int(min(num_edges, len(edges)))
    if percentage_of_edges >= 0 and percentage_of_edges <= 1:
        num_edges = int(np.round(percentage_of_edges * len(edges)))

    for _ in range(num_edges):
        chosen_edge = tuple(rng.choice(edges))
        edges.remove(chosen_edge)

        end_point1 = rng.choice(np.array(chosen_edge))
        end_point2 = end_point1
        while end_point2 == chosen_edge[0] or end_point2 == chosen_edge[1]:
            end_point2 = rng.integers(1, graph.number_of_nodes() + 1)

        graph.add_edge(end_point1, end_point2)
        if del_orig:
            graph.remove_edge(chosen_edge[0], chosen_edge[1])

    return graph


def _generate(
    generator,
    generator_params,
    graphs_per_param,
    total_graphs_for_gen,
    cycle_params,
    graph_dir,
    seed,
):
    print(f"Starting {generator} ... ")
    total_graphs_generated = 0
    cycles = 0

    if generator in gen_utils.nx_available_generators_nd:
        dataset_name = hconfig.NDETERMINISTIC_DATA[generator]
    else:
        dataset_name = hconfig.DETERMINISTIC_DATA[generator]

    number_of_nodes = []
    number_of_edges = []
    graphs_name = []

    epsilon = 0.001
    random.seed(seed)
    np.random.seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    _special_tmp_eba_params = []
    while total_graphs_generated < total_graphs_for_gen:
        swaps = cycle_params[cycles]

        for i, (gen_param, number_graphs_to_gen) in enumerate(
            zip(generator_params[cycles], graphs_per_param[cycles])
        ):
            graphs_generated = 0

            while graphs_generated < number_graphs_to_gen:
                graph_name = dataset_name + "@" + generator

                if generator == "networkx.balanced_tree":
                    g = nx.balanced_tree(gen_param[0], gen_param[1])
                elif generator == "networkx.barbell_graph":
                    m1 = rng.integers(gen_param[0][0], gen_param[0][1])
                    m2 = rng.integers(gen_param[1][0], gen_param[1][1])
                    g = nx.barbell_graph(m1, m2)
                elif generator == "networkx.binomial_tree":
                    g = nx.binomial_tree(gen_param[0])
                elif generator == "networkx.circular_ladder_graph":
                    n = rng.integers(gen_param[0], gen_param[1])
                    g = nx.circular_ladder_graph(n)
                elif generator == "networkx.dorogovtsev_goltsev_mendes_graph":
                    g = nx.dorogovtsev_goltsev_mendes_graph(gen_param[0])
                elif generator == "networkx.full_rary_tree":
                    r = rng.integers(gen_param[0][0], gen_param[0][1])
                    n = rng.integers(gen_param[1][0], gen_param[1][1])
                    g = nx.full_rary_tree(r, n)
                elif generator == "networkx.lollipop_graph":
                    m = rng.integers(gen_param[0][0], gen_param[0][1])
                    n = rng.integers(gen_param[1][0], gen_param[1][1])
                    g = nx.lollipop_graph(m, n)
                elif generator == "networkx.star_graph":
                    n = rng.integers(gen_param[0], gen_param[1])
                    g = nx.star_graph(n)
                elif generator == "networkx.chordal_cycle_graph":
                    n = rng.integers(gen_param[0], gen_param[1])
                    g = nx.chordal_cycle_graph(int(n))
                elif generator == "networkx.grid_graph":
                    x = rng.integers(gen_param[0][0], gen_param[0][1])
                    y = rng.integers(gen_param[1][0], gen_param[1][1])
                    z = rng.integers(gen_param[2][0], gen_param[2][1])
                    g = nx.grid_graph(dim=(x, y, z))
                elif generator == "networkx.hexagonal_lattice_graph":
                    x = rng.integers(gen_param[0][0], gen_param[0][1])
                    y = rng.integers(gen_param[1][0], gen_param[1][1])
                    periodic = rng.integers(gen_param[2][0], gen_param[2][1])
                    g = nx.hexagonal_lattice_graph(x, y, bool(periodic))
                elif generator == "networkx.hypercube_graph":
                    g = nx.hypercube_graph(gen_param[0])
                elif generator == "networkx.triangular_lattice_graph":
                    x = rng.integers(gen_param[0][0], gen_param[0][1])
                    y = rng.integers(gen_param[1][0], gen_param[1][1])
                    periodic = rng.integers(gen_param[2][0], gen_param[2][1])
                    g = nx.triangular_lattice_graph(x, y, bool(periodic))
                elif generator == "networkx.fast_gnp_random_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    if gen_param[1] == "critical":
                        p = 1 / n
                    elif gen_param[1] == "supercritical":
                        p = rng.uniform(1 / n, np.log(n) / (1.5 * n))
                    elif gen_param[1] == "connected":
                        p = rng.uniform(np.log(n) / n, 1.2 * (np.log(n) / n))
                    else:
                        raise ValueError(
                            f"[graph_by_generator] Unknown ER type {gen_param[1]}]"
                        )
                    g = nx.fast_gnp_random_graph(n, p, seed=seed)
                elif generator == "networkx.watts_strogatz_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    k = rng.integers(gen_param[1][0], gen_param[1][1])
                    p = rng.uniform(gen_param[2][0], gen_param[2][1])
                    g = nx.watts_strogatz_graph(n, k, p, seed=seed)
                elif generator == "networkx.random_regular_graph":
                    n = rng.integers(gen_param[1][0], gen_param[1][1])
                    d = rng.integers(gen_param[0][0], gen_param[0][1])
                    while d * n % 2 != 0:
                        n = rng.integers(gen_param[1][0], gen_param[1][1])
                        d = rng.integers(gen_param[0][0], gen_param[0][1])
                    g = nx.random_regular_graph(d, n, seed=seed)
                elif generator == "networkx.extended_barabasi_albert_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    m = rng.integers(gen_param[1][0], gen_param[1][1])
                    if gen_param[2] == "classicalBA":
                        p, q = 0, 0
                    elif gen_param[2] == "exponential":
                        p = rng.uniform(epsilon, 0.35)
                        q_max = min(1 - p, (1 - p + m) / (1 + 2 * m))
                        q = rng.uniform(q_max + epsilon, 1 - p - epsilon)
                    elif gen_param[2] == "powerlaw":
                        p = rng.uniform(epsilon, 0.35)
                        q_max = min(1 - p, (1 - p + m) / (1 + 2 * m))
                        q = rng.uniform(0, q_max)
                    else:
                        raise ValueError(
                            f"[graph_by_generator] Unknown BA type {gen_param[2]}"
                        )
                    _special_tmp_eba_params.append([gen_param[2], m, n, p, q])
                    g = nx.extended_barabasi_albert_graph(n, m, p, q, seed=seed)
                elif generator == "networkx.powerlaw_cluster_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    m = gen_param[1][0]
                    p = gen_param[1][1] / (m - 1)
                    g = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
                elif generator == "networkx.duplication_divergence_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    if gen_param[1] == "self-averaging1":
                        p = rng.uniform(0.1, 1 / np.e)
                    elif gen_param[1] == "self-averaging2":
                        p = rng.uniform(1 / np.e, 0.5)
                    elif gen_param[1] == "not self-averaging":
                        if n > 150 and n < 250:
                            p = rng.uniform(0.5 + epsilon, 0.85)
                        elif n > 250 and n < 500:
                            p = rng.uniform(0.5 + epsilon, 0.75)
                        elif n > 500 and n < 750:
                            p = rng.uniform(0.5 + epsilon, 0.65)
                        else:
                            p = rng.uniform(0.5 + epsilon, 0.55)
                    else:
                        raise ValueError(
                            f"[graph_by_generator] Unknown DD type {gen_param[1]}"
                        )
                    g = nx.duplication_divergence_graph(n, p, seed=seed)
                elif generator == "networkx.gaussian_random_partition_graph":
                    v = 10
                    max_overattractivness = 5
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    s = rng.integers(gen_param[1][0], gen_param[1][1])
                    max_edges = gen_param[2][0]
                    p_in, p_out = gen_utils.suggest_probability(
                        n, s, max_overattractivness, max_edges, rng
                    )
                    g = nx.gaussian_random_partition_graph(
                        n, s, v, p_in, p_out, directed=False, seed=seed
                    )
                elif generator == "igraph.graph.forest_fire":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    fw_prob = rng.uniform(gen_param[1][0], gen_param[1][1] + epsilon)
                    bw_prob = rng.uniform(gen_param[2][0], gen_param[2][1] + epsilon)
                    g = igraph.Graph.Forest_Fire(
                        n=n, fw_prob=fw_prob, bw_factor=bw_prob, directed=False
                    ).to_networkx()
                elif (
                    generator
                    == "hephaestus.graph_generation.generation_utils.random_limited_geometric_graph"
                ):
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    dim = rng.integers(gen_param[1][0], gen_param[1][1])
                    max_edges = rng.integers(gen_param[2][0], gen_param[2][1])
                    g = gen_utils.random_limited_geometric_graph(
                        n, max_edges, dim, seed=seed
                    )
                elif (
                    generator
                    == "hephaestus.graph_generation.generation_utils.random_limited_3dgeo_dd_graph"
                ):
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    max_edges = rng.integers(gen_param[1][0], gen_param[1][1])
                    g = gen_utils.random_limited_3dgeo_dd_graph(n, max_edges, seed=seed)
                elif generator == "networkx.newman_watts_strogatz_graph":
                    n = rng.integers(gen_param[0][0], gen_param[0][1])
                    k = rng.integers(gen_param[1][0], gen_param[1][1])
                    p = rng.uniform(gen_param[2][0], gen_param[2][1])
                    g = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
                else:
                    print(f"[graph_by_generator.py] passed {generator}")
                    g = nx.Graph()

                g = nx.Graph(g)
                g.remove_edges_from(nx.selfloop_edges(g))

                mapping = dict(
                    zip(sorted(list(g.nodes())), range(1, g.number_of_nodes() + 1))
                )
                g = nx.relabel_nodes(g, mapping)

                if isinstance(swaps, float) and swaps >= 0 and swaps <= 1:
                    g = _random_edge(rng, g, swaps, percentage_of_edges=swaps)
                else:
                    g = _random_edge(rng, g, swaps)

                # e.g. hypercube_graph+graph+1+param+2+cycle+3
                graph_name += (
                    "+graph+"
                    + str(graphs_generated)
                    + "+param+"
                    + str(i)
                    + "+cycle+"
                    + str(cycles + 1)
                )
                df = nx.to_pandas_edgelist(g)
                # Gtrie limitation, an edge weight must be included
                df["edge_attrib"] = np.ones((df.shape[0],), dtype=int)
                df.to_csv(
                    os.path.join(graph_dir, graph_name + ".csv"),
                    sep=" ",
                    index=False,
                    header=False,
                )

                number_of_nodes.append(g.number_of_nodes())
                number_of_edges.append(g.number_of_edges())
                graphs_name.append(graph_name)

                graphs_generated += 1

            total_graphs_generated += graphs_generated
        cycles += 1

    df_stats = pd.DataFrame(
        {
            "NumberNodes": number_of_nodes,
            "NumberOfEdges": number_of_edges,
            "GraphNames": graphs_name,
        }
    )

    return df_stats, generator, datetime.now()
