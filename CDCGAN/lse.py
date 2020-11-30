import csv
import os

import numpy as np
import torch
from image_gen.asset_map import get_asset_map
from image_gen.fixer import PipeFixer
from image_gen.image_gen import GameImageGenerator
from tqdm import tqdm

from cmaes import SimpleCMAES as cma
from data_loader import MarioDataset
from get_level import GetLevel as getLevel
from level_kl import KLTileDistribution, LevelInfo
from models.custom import Generator


def init_write(path=None, file_name="lse"):
    if os.path.exists(path + file_name + ".csv"):
        os.remove(file_name + ".csv")
    with open("lse.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["avg. fitnesses", "max fitness", "best solution noise", "full level"]
        )
        # np.savetxt(f, to_add, delimiter=",", fmt="%f")
        f.close()


def write_row(to_add):
    """Append a row to csv file"""
    with open("lse.csv", "a+") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(to_add)
        # np.savetxt(f, to_add, delimiter=",", fmt="%f")
        f.close()


def read_row(file_name="lse"):
    with open(file_name + ".csv", "r") as f:
        # Read all the data from the csv file
        allRows = np.loadtxt(f, delimiter=" ")
    print(allRows, allRows.shape)

    return allRows


def normalize_fitness(x, lower_bound=-1.0, upper_bound=1.0):
    if x < -100:
        x = -100
    elif x > 100:
        x = 100

    if x < 0:
        return (0 - lower_bound) * (x / 100.0)
    else:
        return (upper_bound - 0) * (x / 100.0)


def matrix_eval_simple(level):
    sky_region, ground_region = level[:10], level[13]
    num_ground_tiles = np.sum(ground_region == 0)
    num_sky_tiles = np.sum(np.logical_and(sky_region != 2, sky_region != 5))
    raw_fitness = num_sky_tiles - (num_ground_tiles * 2)
    fitness = normalize_fitness(raw_fitness)
    return fitness


def matrix_eval(level):

    sky_region, upper_half = level[:10], level[:7]
    ground_tiles = np.logical_or(level == 0, level == 1)

    num_enemies = np.sum(np.logical_and(level[:-1] == 5, ground_tiles[1:]))
    num_sky_enemies = np.sum(level == 5) - num_enemies
    num_sky_tiles = np.sum(np.logical_and(sky_region != 2, sky_region != 5))
    num_sky_tiles_at_half = np.sum(np.logical_and(upper_half != 2, upper_half != 5))
    num_ground_tiles = np.sum(level == 0)
    num_pipes = np.sum(level == 6)
    # return (
    #     (num_ground_tiles / 3 - num_enemies) * 0.7
    #     + (num_sky_tiles_at_half - num_sky_tiles) * 0.3
    #     - num_sky_enemies * 0.5
    # )
    return (
        (num_sky_tiles_at_half + num_sky_tiles)
        - num_ground_tiles
        + 10 * np.sum(np.logical_or(level == 5, level == 11, level == 12))
    )
    # return num_enemies + num_sky_enemies


def evaluate_kl(ori_lvl, gen_lvl):
    return KLTileDistribution(
        l1=gen_lvl, l2=ori_lvl, window_size=(3, 3)
    ).compute_kl_loss()


if __name__ == "__main__":
    samples_per_member = 5
    population_size = 1000
    conditional_channels = [
        0,
        1,
        6,
        7,
    ]  # channels on which generator is conditioned on
    dataset = MarioDataset()
    netG = Generator(
        latent_size=(len(conditional_channels) + 1, 14, 14), out_size=(13, 32, 32)
    )
    netG.load_state_dict(torch.load("./trained_models/netG_epoch_300000_0_32.pth"))
    # 300000
    mario_map = get_asset_map(game="mario")
    gen = GameImageGenerator(asset_map=mario_map)
    prev_frame, curr_frame = dataset[[120]]  # 51
    fixer = PipeFixer()

    # mean = np.load("best_sky_tile_member.npy")

    lse = cma(
        standard_deviation=0.5, population_size=population_size, noise_size=(14, 14),
    )  # , mean=mean.flatten())
    level_gen = getLevel(netG, gen, fixer, prev_frame, curr_frame, conditional_channels)
    count = 0
    overall_max_fitness = None
    dir_path = "enemy_lvl/"
    init_write(path=dir_path)
    best_member = None

    ori_lvl = LevelInfo(fmt="txt", name="sky_level", path="")
    while count <= 1000:
        populated_noises = lse.get_pop_noise()
        fitnesses = []
        for noise in tqdm(populated_noises, desc=f"{count}: evaluating fitness"):
            fitness = []
            levels = []
            for _ in range(samples_per_member):
                level_gen.reset(prev_frame=prev_frame, curr_frame=curr_frame)
                level = level_gen.generate_frames(noise, var=0.07, frame_count=3)

                fitness.append(matrix_eval(level))
                # fitness.append(evaluate_kl(ori_lvl.lvl_array, level))
                prev_frame, curr_frame = dataset[[torch.randint(len(dataset), (1,))]]
                # gen_lvl = LevelInfo(fmt=None, name=None, path=None, lvl_str=full_level)

            fitnesses.append(np.mean(fitness))
        # print(fitnesses)
        lse.update_cmaes(fitnesses)
        s = lse.get_best_solution()
        current_max_fitness = max(fitnesses)
        print(f"overall_max:{overall_max_fitness}, current_max: {current_max_fitness}")
        if overall_max_fitness is None or current_max_fitness > overall_max_fitness:
            best_member = s
            overall_max_fitness = current_max_fitness
        elif (
            (current_max_fitness - overall_max_fitness) / overall_max_fitness
        ) <= -0.30:
            print("Resetting!!")
            lse.reset(mean=s.flatten())

        # save every 5 steps
        if count % 1 == 0:
            level_gen.reset()
            full_level = level_gen.generate_frames(s.flatten(), var=0.07)

            image_name = dir_path + "normal_eval" + "_" + str(count)

            level_gen.save_full_level(file_name=image_name)
            level_gen.gen.save_gen_level(img_name=image_name)
            write_row([s])

        if count % 10 == 0:
            lse.plot_cma(dir_path=dir_path, count=count)

        count = count + 1
        print(
            f"iter: {count} avg fitness: {np.mean(fitnesses)}, max: {np.max(fitnesses)}, median: {np.median(fitnesses)}"
        )
        # print("Current best solution found:", s.reshape(14, 14))
        write_row([np.mean(fitnesses), np.max(fitnesses), s, full_level.tolist()])

    np.save("best_member", best_member)
