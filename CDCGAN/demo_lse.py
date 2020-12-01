import numpy as np
import torch

torch.manual_seed(75)

from cursesmenu import CursesMenu, SelectionMenu
from image_gen.asset_map import get_asset_map
from image_gen.fixer import PipeFixer
from image_gen.image_gen import GameImageGenerator

from data_loader import MarioDataset
from get_level import GetLevel as getLevel
from models.custom import Generator


if __name__ == "__main__":
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
    netG.load_state_dict(torch.load("trained_models/netG_epoch_300000_0_32.pth"))
    mario_map = get_asset_map(game="mario")
    gen = GameImageGenerator(asset_map=mario_map)
    prev_frame, curr_frame = dataset[[120]]  # 51
    fixer = PipeFixer()
    level_gen = getLevel(netG, gen, fixer, prev_frame, curr_frame, conditional_channels)

    noise_params = (
        np.load("underground_params.npy"),
        np.load("best_sky_tile_member.npy"),
        np.zeros((196,)),
    )
    features = ["Underground Level", "More Sky Tiles", "Random Noise"]

    while True:
        selection = SelectionMenu.get_selection(
            features, title="Select the features you would like to generate:",
        )
        var = 0.07 if selection != 2 else 1.0
        try:
            noise = noise_params[selection]
            level = level_gen.generate_frames(noise, var=var, frame_count=6)
            level_gen.gen.save_gen_level(img_name="lse_demo")
        except IndexError:
            break

