"""
Every utility used by the synthetic generation procedure.
"""

import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist

# Governor array for non-deterministic
# It does not matter what is in config.ini.
# If it is note here it will not be generated.
# However, if something is here and not in cofig.ini it will have undefined behaviour
# Most likely will crash on the first "if" of graph_by_generator._generate
nx_available_generators_nd = [
    "hephaestus.graph_generation.generation_utils.random_limited_geometric_graph",
    "hephaestus.graph_generation.generation_utils.random_limited_3dgeo_dd_graph",
    "networkx.watts_strogatz_graph",
    "networkx.extended_barabasi_albert_graph",
    "networkx.fast_gnp_random_graph",
    "networkx.powerlaw_cluster_graph",
    "networkx.duplication_divergence_graph",
    "networkx.random_regular_graph",
    "networkx.gaussian_random_partition_graph",
    "networkx.newman_watts_strogatz_graph",
    "igraph.graph.forest_fire",
]

# Governor array for deterministic
# It does not matter what is in config.ini.
# If it is note here it will not be generated.
# However, if something is here and not in cofig.ini it will have undefined behaviour
# Most likely will crash on the first "if" of graph_by_generator._generate
nx_available_generators_d = [
    "networkx.balanced_tree",
    "networkx.barbell_graph",
    "networkx.binomial_tree",
    "networkx.circular_ladder_graph",
    "networkx.dorogovtsev_goltsev_mendes_graph",
    "networkx.full_rary_tree",
    "networkx.lollipop_graph",
    "networkx.star_graph",
    "networkx.chordal_cycle_graph",
    "networkx.grid_graph",
    "networkx.hexagonal_lattice_graph",
    "networkx.triangular_lattice_graph",
]


def suggest_probability(
    n, sigma, max_over_attractiveness, max_edges, rng, print_solution=False
):
    """
    The calculations for the limit of maximum edges were made assuming p_in = max_over_attractiveness*p_out.
    Hence, anything bellow that value for p_in is guaranteed to follow the maximum limit of edges.
    However, we lose some control over the result by allowing p_in to vary between 0 and the other bound.
    Note: For some reason the bound seems to be overpessimistic by a factor of sigma in the denominator.
            This means it gives a probability that is lower than what it could have been to achieve the desired edges.
            Hence, we got np.power(sigma, 1.2) instead of np.power(sigma, 2).

    :param `int` `n`: Number of nodes that the graph will have.
    :param `float` `sigma`: Mean amount of nodes in a group of nodes.
    :param `float` `max_over_attractiveness`: How much (at max) p_in should be more than p_out.
    :param `int` `max_edges`: Limit of edges to bound probabilities.
    :param `numpy.random.default_rng` `rng`: For reproducibility.
    """
    max_out_prob = (2 * max_edges) / (
        np.power(n, 2)
        + n
        * (
            max_over_attractiveness * np.power(sigma, 1.2)
            - (max_over_attractiveness + 1) * sigma
        )
    )

    if print_solution:
        print(f"Raw max prob: {max_out_prob}")

    max_out_prob = min(1.0, max_out_prob)
    p_out = rng.uniform(max_out_prob / 100, max_out_prob)

    # Should perhaps put lower bound always >= p_out to express p_in being always more attractive than p_out
    # As it stands that may not happen.
    max_in_prob = min(1.0, p_out * max_over_attractiveness)
    p_in = rng.uniform(max_in_prob / 100, max_in_prob)

    return p_in, p_out


# Adapted from https://github.com/migueleps/dir-netemd-netsgen/blob/main/gen_geomgd.py
def normalize(x):
    return x / np.linalg.norm(x)


def random_limited_geometric_graph(N, E, dim=2, pos=None, p=2, seed=None):
    """
    Brute-force searching for the distance that produces E edges, hence bound to Omega(N^2)
    Manageable because we are not interested in very large graphs.

    `dim`, `p`, `pos`, `seed` like `networkx.random_geometric_graph`.

    :param `int` `N`: Number of nodes that the graph will have.
    :param `int` `E`: Number of edges that the graph will have.
    """
    rng = np.random.default_rng(seed)
    pos = _initialize_points(N, dim, rng)

    E = min(E, (N**2 - N) / 2)
    D = pdist(np.array(list(pos.values())), metric="minkowski", **{"p": p})

    # find radius at which the geometric graph has E edges
    r = np.sort(D, axis=None)[E - 1]
    return nx.random_geometric_graph(N, r, dim=dim, pos=pos, seed=seed)


def random_limited_3dgeo_dd_graph(N, E, seed=None):
    rng = np.random.default_rng(seed)
    points = _initialize_points(N, 3, rng, duplication_divergence=True)

    E = min(E, (N**2 - N) / 2)
    D = pdist(np.array(list(points.values())), metric="euclidean")

    # find radius at which the geometric graph has E edges
    r = np.sort(D, axis=None)[E - 1]
    return nx.random_geometric_graph(N, r, dim=3, pos=points, seed=seed)


def _initialize_points(N, dim, rng, duplication_divergence=False):
    initial_points = N
    if duplication_divergence:
        initial_points = 5

    p = {}
    for i in range(initial_points):
        p[i] = rng.uniform(0, 1 + 0.001, (dim,))

    if duplication_divergence:
        for i in range(initial_points, N):  # create new nodes by duplication divergence
            k = rng.integers(0, i)
            p[i] = p[k] + 2 * rng.uniform(0, 1 + 0.001) * normalize(
                rng.uniform(0, 1 + 0.001, (dim,)) - 0.5
            )
            # position of new node is the position of the parrent
            # plus a vector with random direction and random length between O and 2

    return p
