import numpy as np


class PipeFixer:
    PIPE = {
        "top-left": 6,
        "top-right": 7,
        "body-left": 8,
        "body-right": 9,
    }
    GROUND = {0, 1}
    SKY = 2

    def __init__(self):
        pass

    def fix(self, image_array):
        l_no_top, r_no_top = self.find_no_top_pipes(image_array)
        self.add_tops(l_no_top, "left", image_array)
        self.add_tops(r_no_top, "right", image_array)

        top_left_idx, top_right_idx = self.find_top_pipes(image_array)

        self.build_pipe_from_top(top_left_idx, top="left", image_array=image_array)
        self.build_pipe_from_top(top_right_idx, top="right", image_array=image_array)

        self.remove_ground_from_pipe_top(top_left_idx, image_array=image_array)
        self.remove_ground_from_pipe_top(top_right_idx, image_array=image_array)
        return image_array

    def remove_ground_from_pipe_top(self, idx, image_array):
        for i in idx:
            image_array[(max(0, i[0]-1), i[1])] = self.SKY

    def build_pipe_from_top(self, idx, top, image_array):
        if top == "left":
            move = {
                "offset": (0, 1),
                "top_asset": self.PIPE["top-right"],
                "down_asset": self.PIPE["body-left"],
                "down_offset_asset": self.PIPE["body-right"],
            }

        else:
            move = {
                "offset": (0, -1),
                "top_asset": self.PIPE["top-left"],
                "down_asset": self.PIPE["body-right"],
                "down_offset_asset": self.PIPE["body-left"],
            }
        for i in idx:
            image_array[(i[0] + move["offset"][0], i[1] + move["offset"][1])] = move[
                "top_asset"
            ]
            i_d = (i[0] + 1, i[1])
            while i_d[0] < image_array.shape[0] and image_array[i_d] not in self.GROUND:
                image_array[i_d] = move["down_asset"]
                image_array[
                    (i_d[0] + move["offset"][0], i_d[1] + move["offset"][1])
                ] = move["down_offset_asset"]
                i_d = (i_d[0] + 1, i_d[1])

    def find_no_top_pipes(self, image_array):
        left_potent_idx = np.where(image_array == self.PIPE["body-left"])
        left_potent_idx = [*zip(left_potent_idx[0], left_potent_idx[1])]
        right_potent_idx = np.where(image_array == self.PIPE["body-right"])
        right_potent_idx = [*zip(right_potent_idx[0], right_potent_idx[1])]

        left_no_top_pipe_idx = []
        right_no_top_pipe_idx = []
        pipe_parts = self.PIPE.values()
        for (l, r) in zip(left_potent_idx, right_potent_idx):
            if (image_array[(l[0] - 1, l[1])] not in pipe_parts) and (
                image_array[(l[0] - 2, l[1])] not in pipe_parts
            ):
                left_no_top_pipe_idx.append(l)
            if (image_array[(r[0] - 1, r[1])] not in pipe_parts) and (
                image_array[(r[0] - 2, r[1])] not in pipe_parts
            ):
                right_no_top_pipe_idx.append(r)
        return left_no_top_pipe_idx, right_no_top_pipe_idx

    def find_top_pipes(self, image_array):
        # find pipes with tops.
        matches = np.where(image_array == self.PIPE["top-left"])
        top_left_idx = [*zip(matches[0], matches[1])]
        matches = np.where(image_array == self.PIPE["top-right"])
        top_right_idx = [*zip(matches[0], matches[1])]

        return top_left_idx, top_right_idx

    def add_tops(self, idx, side, image_array):
        for i in idx:
            image_array[(i[0] - 1, i[1])] = self.PIPE[f"top-{side}"]
