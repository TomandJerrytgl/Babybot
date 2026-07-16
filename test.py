import cv2
from attention import Attention


class Observation:

    def __init__(self, left):
        self.left = left


def main():

    print("1. Loading image...")

    image = cv2.imread("test/test1.png")
    scale = 0.5

    image = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale)

    if image is None:
        print("Image load failed.")
        return

    print("2. Image loaded.")

    print("Image Shape :", image.shape)

    observation = Observation(image)

    print("3. Creating Attention...")

    attention = Attention(observation)

    print("4. Attention finished.")

    x, y, w, h = attention.focus

    print("Attention Window :", attention.focus)

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
        4,
        (0, 0, 255),
        -1
    )

    print("5. Display image.")

    cv2.imshow("Attention Test", result)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print("6. Program Finished.")


if __name__ == "__main__":
    main()