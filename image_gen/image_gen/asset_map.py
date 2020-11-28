import os
import pickle

import PIL


def get_asset_map(game="mario"):
    base_path = os.path.dirname(__file__)
    asset_path = f"data/asset_maps/{game}.pkl"
    with open(os.path.join(base_path, asset_path), "rb") as fp:
        asset_map = pickle.load(fp)

    return asset_map

