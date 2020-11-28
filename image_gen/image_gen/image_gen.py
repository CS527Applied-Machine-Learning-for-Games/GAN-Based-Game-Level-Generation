import pickle
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


class GameImageGenerator:
    def __init__(self, asset_map: Dict[int, Image.Image]):
        """
        Class to facilitate the rendering of a game level from given feature map.
        
        Args:
            asset_map (Dict[int, Image]): Used to map feature->image_asset.
        """
        self.asset_map = asset_map
        self.level = None

    def render(
        self,
        image_array: np.array,
        sprite_dims: Tuple,
        bg_color="blue",
        path="trial_image.png",
    ):
        """
        Renders the image given an image array.
        Assumes that array is 2D i.e only one channel,
        with an integer associated to a certain asset/sprite.
        """

        level_dims = (
            image_array.shape[1] * sprite_dims[1],
            image_array.shape[0] * sprite_dims[0],
        )  # x,y
        self.level = Image.new(mode="RGBA", size=level_dims, color="#00b3ff")
        for i, row in enumerate(image_array):
            for j, asset_id in enumerate(row):
                if asset_id not in self.asset_map:
                    continue
                try:
                    self.level.paste(
                        self.asset_map[asset_id],
                        (sprite_dims[0] * j, sprite_dims[0] * i),
                        self.asset_map[asset_id],
                    )
                except ValueError:
                    self.level.paste(
                        self.asset_map[asset_id],
                        (sprite_dims[0] * j, sprite_dims[0] * i),
                    )
        # level.show()
        # self.level.save(path)

    def save_gen_level(self, img_name="trial_image"):
        self.level.save(img_name + ".png")

if __name__ == "__main__":
    base_path = "./data/mario_assets/encoding_{}.png"
    asset_map = {}
    for i in range(11):
        try:
            asset_map[i] = Image.open(base_path.format(i))
        except FileNotFoundError:
            continue
    image_gen = GameImageGenerator(asset_map=asset_map)
    with open("test_frame.pkl", "rb") as fp:
        image_array = pickle.load(fp)
    image_gen.render(image_array=image_array[0], sprite_dims=(16, 16))

