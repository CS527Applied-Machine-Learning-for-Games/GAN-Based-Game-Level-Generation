import dataclasses
import os
import pickle
from collections import defaultdict
from math import log
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

symbol_identity_map = dict(
    {
        "X": 0,
        "S": 1,
        "-": 2,
        "?": 3,
        "Q": 4,
        "E": 5,
        "<": 6,
        ">": 7,
        "[": 8,
        "]": 9,
        "o": 10,
    }
)
tile_space_size = len(symbol_identity_map)


def load_lvl(pkl_dir, pkl_filename):
    dbfile = open(os.path.join(pkl_dir, pkl_filename + ".pkl"), "rb")
    lvl_str = pickle.load(dbfile)
    return lvl_str


def read_txt_level(file_path, filename):
    lvl_str = []
    with open(os.path.join(file_path, filename + ".txt"), "rt") as file:
        for line in file:
            row_arr = []
            for i in range(0, len(line)):
                if line[i] != "\n":
                    row_arr.append(symbol_identity_map[line[i]])
            lvl_str.append(row_arr)
    return np.array(lvl_str)


@dataclasses.dataclass
class LevelInfo:
    def __init__(self, fmt, name, path, lvl_array=None):
        self.fmt = fmt
        self.name = name
        self.path = path
        self.lvl_array = lvl_array if lvl_array is not None else self.load_level()

    def load_level(self):
        if self.fmt == "pkl":
            return load_lvl(self.path, self.name)

        elif self.fmt == "txt":
            return read_txt_level(self.path, self.name)


class KLTileDistribution:
    epsilon = 0.01

    def __init__(self, l1, l2, window_size=(3, 3)):
        self.window_size = (
            window_size
            if isinstance(window_size, tuple)
            else (window_size, window_size)
        )
        self.l1_str = l1
        self.l2_str = l2

    def get_tile_distribution(self, lvl_str) -> Dict[str, float]:
        tile_counts = defaultdict(int)
        num_rows, num_colums = lvl_str.shape
        row_slides, column_slides = (
            num_rows - self.window_size[0] + 1,
            num_colums - self.window_size[1] + 1,
        )
        total_windows = row_slides * column_slides
        for i in range(row_slides):
            for j in range(column_slides):
                tile = np.str(
                    lvl_str[
                        i : i + self.window_size[0], j : j + self.window_size[1]
                    ].flatten()
                )
                tile_counts[tile] += 1
        tile_distributions = dict()
        normalizer = (total_windows + self.epsilon) * (1 + self.epsilon)
        for tile, count in tile_counts.items():
            tile_distributions[tile] = (count + self.epsilon) / normalizer
        return tile_distributions, total_windows

    def display_tile_distribution(self, window_counts):
        # window_counts = list(filter((0).__ne__, window_counts))
        plt.bar(list(range(len(window_counts))), window_counts)
        plt.show()

    # calculate the kl divergence
    def kl_divergence(self, p, q, q_total_windows):
        default_freq = self.epsilon / (
            (q_total_windows + self.epsilon) * (1 + self.epsilon)
        )
        return sum((p[tile] * log(p[tile] / q.get(tile, default_freq)) for tile in p))

    def compute_kl_loss(self, w=0.5):
        l1, l1_total_windows = self.get_tile_distribution(self.l1_str)
        l2, l2_total_windows = self.get_tile_distribution(self.l2_str)
        kl_l1_l2 = self.kl_divergence(l1, l2, l2_total_windows)
        kl_l2_l1 = self.kl_divergence(l2, l1, l1_total_windows)

        return -(w * kl_l1_l2 + (1 - w) * kl_l2_l1)


# #display_tile_distribution(get_tile_distribution("originallvl1.txt", 2))
# ori_lvl = LevelInfo(fmt='txt', name='sky_level', path='')
# gen_lvl = LevelInfo(fmt='pkl', name='num_enemies_80000', path='gen_levels_30')
# kl_generated_original = KLTileDistribution(2, gen_lvl, ori_lvl).compute_kl()
# print (kl_generated_original)
