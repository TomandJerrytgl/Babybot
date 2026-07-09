import numpy as np


# 在这部分希望实现 region = Attention(observation)
# 返回一个 region 进行后续处理

class Attention:

    def __init__(self, observation, object=None):

        self.observation = observation
        self.object = object

        if object is None:
            self.focus = self.default_focus()

    def default_focus(self):

        left = self.observation.left

        return find_attention_window(left)


def find_attention_window(
        image,
        window_size=256,
        step=16,
        vivid_weight=1.0,
        center_weight=20.0):

    height, width = image.shape[:2]

    center_x = width / 2
    center_y = height / 2

    max_distance = np.sqrt(center_x**2 + center_y**2)

    best_score = -1e9
    best_window = None

    for y in range(0, height - window_size + 1, step):

        for x in range(0, width - window_size + 1, step):

            window = image[
                y:y + window_size,
                x:x + window_size
            ]

            r = window[:, :, 0]
            g = window[:, :, 1]
            b = window[:, :, 2]

            vividness = np.mean(
                np.maximum.reduce([r, g, b]) -
                np.minimum.reduce([r, g, b])
            )

            wx = x + window_size / 2
            wy = y + window_size / 2

            distance = np.sqrt(
                (wx - center_x) ** 2 +
                (wy - center_y) ** 2
            )

            center_score = 1 - distance / max_distance

            score = (
                vivid_weight * vividness +
                center_weight * center_score
            )

            if score > best_score:

                best_score = score

                best_window = (
                    x,
                    y,
                    window_size,
                    window_size
                )

    return best_window