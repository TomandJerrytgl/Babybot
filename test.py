import cv2
from attention import Attention


class Observation:

    def __init__(self, left):
        self.left = left


def main():

    image = cv2.imread("test/test1.png")

    if image is None:
        print("Image load failed.")
        return

    observation = Observation(image)

    attention = Attention(observation)

    x, y, w, h = attention.focus

    print("Attention Window:", attention.focus)

    result = image.copy()

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.circle(
        result,
        (x + w // 2, y + h // 2),
        3,
        (0, 0, 255),
        -1
    )

    cv2.imshow("Attention Test", result)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()