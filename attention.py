import numpy as np


class Attention:

    def __init__(self, observation, object=None):

        self.observation = observation
        self.object = object

        if object is None:
            self.focus = self.default_focus()

    ##############################################################
    # Default Attention
    ##############################################################

    def default_focus(self):

        left = self.observation.left

        return self.find_attention_window(left)

    ##############################################################
    # Brightness Preference
    ##############################################################

    def brightness_preference(self, brightness):

        ideal = 170
        sigma = 60

        score = np.exp(
            -((brightness - ideal) ** 2) /
            (2 * sigma * sigma)
        )

        return float(score)

    ##############################################################
    # Center Preference
    ##############################################################

    def center_preference(
            self,
            x,
            y,
            width,
            height):

        center_x = width / 2
        center_y = height / 2

        distance = np.sqrt(
            (x - center_x) ** 2 +
            (y - center_y) ** 2
        )

        max_distance = np.sqrt(
            center_x ** 2 +
            center_y ** 2
        )

        score = 1 - distance / max_distance

        score = np.clip(score, 0.0, 1.0)

        return float(score)

    ##############################################################
    # Contrast
    ##############################################################

    def contrast_score(
            self,
            image,
            x,
            y,
            window_size):

        h, w = image.shape[:2]

        window = image[
            y:y + window_size,
            x:x + window_size
        ]

        background_size = window_size * 2

        x1 = max(
            0,
            x - background_size // 2
        )

        y1 = max(
            0,
            y - background_size // 2
        )

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

        contrast = np.linalg.norm(
            window_mean -
            background_mean
        )

        contrast /= 255.0

        contrast = np.clip(
            contrast,
            0.0,
            1.0
        )

        return float(contrast)

    ##############################################################
    # Color Preference
    ##############################################################

    def color_preference(self, window):

        r = window[:, :, 0].astype(np.float32)
        g = window[:, :, 1].astype(np.float32)
        b = window[:, :, 2].astype(np.float32)

        score = np.mean(
            np.maximum.reduce([r, g, b]) -
            np.minimum.reduce([r, g, b])
        )

        score /= 255.0

        score = np.clip(
            score,
            0.0,
            1.0
        )

        return float(score)

    ##############################################################
    # Visual Saliency
    ##############################################################

    def visual_saliency(
            self,
            image,
            x,
            y,
            window_size):

        window = image[
            y:y + window_size,
            x:x + window_size
        ]

        r = window[:, :, 0].astype(np.float32)
        g = window[:, :, 1].astype(np.float32)
        b = window[:, :, 2].astype(np.float32)

        ##########################################################
        # Brightness
        ##########################################################

        brightness = np.mean(
            0.299 * r +
            0.587 * g +
            0.114 * b
        )

        brightness_score = self.brightness_preference(
            brightness
        )

        ##########################################################
        # Contrast
        ##########################################################

        contrast_score = self.contrast_score(
            image,
            x,
            y,
            window_size
        )

        ##########################################################
        # Color Preference
        ##########################################################

        color_score = self.color_preference(
            window
        )

        ##########################################################
        # Visual Score
        ##########################################################

        visual_score = (
            0.3 * brightness_score +
            0.5 * contrast_score +
            0.8 * color_score
        )

        return (
            brightness_score,
            contrast_score,
            color_score,
            visual_score
        )
        ##############################################################
    # Find Attention Window
    ##############################################################

    def find_attention_window(
            self,
            image,
            window_size=128,
            step=8):

        height, width = image.shape[:2]

        best_score = -1

        best_window = None

        best_result = None

        for y in range(
                0,
                height - window_size + 1,
                step):

            for x in range(
                    0,
                    width - window_size + 1,
                    step):

                (
                    brightness_score,
                    contrast_score,
                    color_score,
                    visual_score
                ) = self.visual_saliency(
                    image,
                    x,
                    y,
                    window_size
                )

                center_score = self.center_preference(
                    x + window_size / 2,
                    y + window_size / 2,
                    width,
                    height
                )

                ##################################################
                # Final Attention Score
                ##################################################

                final_score = (
                    0.8 * visual_score +
                    0.2 * center_score
                )

                if final_score > best_score:

                    best_score = final_score

                    best_window = (
                        x,
                        y,
                        window_size,
                        window_size
                    )

                    best_result = (
                        brightness_score,
                        contrast_score,
                        color_score,
                        visual_score,
                        center_score,
                        final_score
                    )

        print("---------------------------------------")

        print(f"Brightness   : {best_result[0]:.3f}")
        print(f"Contrast     : {best_result[1]:.3f}")
        print(f"Color        : {best_result[2]:.3f}")
        print(f"Visual Score : {best_result[3]:.3f}")
        print(f"Center Score : {best_result[4]:.3f}")
        print(f"Final Score  : {best_result[5]:.3f}")

        print("Window :", best_window)

        print("---------------------------------------")

        return best_window