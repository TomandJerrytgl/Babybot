import cv2
import os

from attention import Attention


class Observation:

    def __init__(self, left):
        self.left = left


def draw_attention_candidates(
        image,
        candidates,
        maximum_candidates=5):

    result = image.copy()

    # OpenCV 使用 BGR 颜色顺序。
    colors = [
        (0, 0, 255),       # Rank 1: Red
        (0, 165, 255),     # Rank 2: Orange
        (0, 255, 255),     # Rank 3: Yellow
        (0, 255, 0),       # Rank 4: Green
        (255, 0, 0)        # Rank 5: Blue
    ]

    displayed_candidates = candidates[
        :maximum_candidates
    ]

    for rank, candidate in enumerate(
            displayed_candidates,
            start=1):

        x, y, width, height = candidate["window"]

        score = candidate["score"]

        color = colors[
            (rank - 1) % len(colors)
        ]

        # 第一名使用更粗的框。
        thickness = 4 if rank == 1 else 2

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            color,
            thickness
        )

        center_x = x + width // 2
        center_y = y + height // 2

        cv2.circle(
            result,
            (center_x, center_y),
            5,
            color,
            -1
        )

        label = (
            f"Rank {rank}  "
            f"Score {score:.3f}  "
            f"Size {width}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )

        label_x = x
        label_y = y - 10

        # 如果框太靠近图片顶部，将文字放到框内。
        if label_y - text_height < 0:
            label_y = y + text_height + 10

        background_x1 = label_x
        background_y1 = label_y - text_height - 6
        background_x2 = label_x + text_width + 8
        background_y2 = label_y + baseline + 4

        cv2.rectangle(
            result,
            (background_x1, background_y1),
            (background_x2, background_y2),
            color,
            -1
        )

        cv2.putText(
            result,
            label,
            (label_x + 4, label_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

    return result


def print_candidate_details(candidates):

    if not candidates:
        print("No attention candidates found.")
        return

    print()
    print("=" * 55)
    print("Top Attention Candidates")
    print("=" * 55)

    for rank, candidate in enumerate(
            candidates[:5],
            start=1):

        print(f"Rank {rank}")
        print(f"Window       : {candidate['window']}")
        print(f"Brightness   : {candidate['brightness']:.3f}")
        print(f"Contrast     : {candidate['contrast']:.3f}")
        print(f"Color        : {candidate['color']:.3f}")
        print(f"Visual Score : {candidate['visual']:.3f}")
        print(f"Center Score : {candidate['center']:.3f}")
        print(f"Final Score  : {candidate['score']:.3f}")
        print("-" * 55)


def resize_for_display(
        image,
        maximum_width=1400,
        maximum_height=900):

    height, width = image.shape[:2]

    scale = min(
        1.0,
        maximum_width / width,
        maximum_height / height
    )

    if scale >= 1.0:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def main():

    image_path = "test/videoresult/left_test.jpg"

    print("1. Loading image...")

    image = cv2.imread(image_path)

    if image is None:
        print(f"Image load failed: {image_path}")
        return

    print("2. Image loaded.")
    print("Image shape:", image.shape)

    observation = Observation(
        left=image
    )

    print("3. Running pyramid attention...")

    attention = Attention(observation)

    print("4. Attention finished.")
    print(
        f"Processing time: "
        f"{attention.elapsed_time * 1000:.2f} ms"
    )

    if attention.elapsed_time > 0:

        print(
            f"Estimated FPS: "
            f"{1.0 / attention.elapsed_time:.2f}"
        )

    candidates = attention.candidates

    print_candidate_details(candidates)

    if not candidates:
        print("No candidate can be displayed.")
        return

    result = draw_attention_candidates(
        image=image,
        candidates=candidates,
        maximum_candidates=5
    )

    display_result = resize_for_display(
        result
    )

    output_dir = "test/attentionresult"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        "attention_result.jpg"
    )

    cv2.imwrite(output_path, display_result)

    print(f"Attention result saved to: {output_path}")

    print()
    print("Press any key in the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()