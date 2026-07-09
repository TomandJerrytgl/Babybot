import cv2
from attention import find_attention_window


def main():

    image = cv2.imread("test/test3.png")

    if image is None:
        print("Image load failed.")
        return

    attention = find_attention_window(image)

    print("Attention Window:", attention)

    x, y, w, h = attention

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