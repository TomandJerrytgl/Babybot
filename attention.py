import numpy as np


class Attention:

    def __init__(self, observation, object=None):

        self.observation = observation
        self.object = object

        if object is None:
            self.focus = self.default_focus()

    ##############################################################

    def default_focus(self):

        left = self.observation.left

        return self.find_attention_window(left)

    ##############################################################

    def brightness_preference(self, brightness):

        ideal = 170
        sigma = 60

        score = np.exp(
            -((brightness - ideal) ** 2) /
            (2 * sigma * sigma)
        )

        return score

    ##############################################################

    def visual_saliency(
            self,
            image,
            x,
            y,
            window_size):

        ##########################################################
        # Current Window
        ##########################################################

        window = image[
            y:y+window_size,
            x:x+window_size
        ]

        ##########################################################
        # Brightness
        ##########################################################

        r = window[:, :, 0].astype(np.float32)
        g = window[:, :, 1].astype(np.float32)
        b = window[:, :, 2].astype(np.float32)

        brightness = np.mean(
            0.299 * r +
            0.587 * g +
            0.114 * b
        )

        brightness_score = self.brightness_preference(
            brightness
        )

        ##########################################################
        # Local Contrast
        ##########################################################

        h, w = image.shape[:2]

        background_size = window_size * 4

        x1 = max(0, x - background_size // 2)
        y1 = max(0, y - background_size // 2)

        x2 = min(
            w,
            x + window_size + background_size // 2
        )

        y2 = min(
            h,
            y + window_size + background_size // 2
        )

        background = image[
            y1:y2,
            x1:x2
        ]

        window_mean = np.mean(
            window.reshape(-1, 3),
            axis=0
        )

        background_mean = np.mean(
            background.reshape(-1, 3),
            axis=0
        )

        contrast_score = np.linalg.norm(
            window_mean - background_mean
        )

        ##########################################################
        # Final Score
        ##########################################################

        score = (
            0.4 * brightness_score +
            0.6 * contrast_score
        )

        return score

    ##############################################################

    def find_attention_window(
            self,
            image,
            window_size=256,
            step=64):

        height, width = image.shape[:2]

        best_score = -1

        best_window = None

        for y in range(
                0,
                height-window_size+1,
                step):

            for x in range(
                    0,
                    width-window_size+1,
                    step):

                score = self.visual_saliency(
                    image,
                    x,
                    y,
                    window_size
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