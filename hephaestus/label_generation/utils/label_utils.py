"""
Utilities used by label_generation/ module.
"""
import hephaestus.utils.general_utils as hutils

PREAMBLE = "adjmatrix, occ_original, z_score, avg_random, stdev_random\n"
SIZE3_GRAPHS = [
    '"011100100"', 
    '"011101110"'
]
SIZE4_GRAPHS = [
    '"0110100110000100"',
    '"0110100110010110"',
    '"0111100010001000"',
    '"0111101011001000"',
    '"0111101111001100"',
    '"0111101111011110"',
]


def build_size_4_string():
    """Build string for dummy NaN file of size 4"""
    s = ""
    number_of_nans = len(PREAMBLE.split(", ")) - 1
    for graph in SIZE4_GRAPHS:
        s += "".join(
            hutils.flatten_nested_list(
                [
                    [graph],
                    [", "],
                    [
                        "nan, " if i != number_of_nans - 1 else "nan\n"
                        for i in range(number_of_nans)
                    ],
                ],
                sort=False,
            )
        )
    return s


def build_size_3_string():
    """Build string for dummy NaN file of size 3"""
    s = ""
    number_of_nans = len(PREAMBLE.split(", ")) - 1
    for graph in SIZE3_GRAPHS:
        s += "".join(
            hutils.flatten_nested_list(
                [
                    [graph],
                    [", "],
                    [
                        "nan, " if i != number_of_nans - 1 else "nan\n"
                        for i in range(number_of_nans)
                    ],
                ],
                sort=False,
            )
        )
    return s
