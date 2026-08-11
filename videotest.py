import cv2
import time
import os

OUTPUT_DIR = "test/videoresult"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


##############################################################
# Camera Settings
##############################################################

LEFT_CAMERA = 0
RIGHT_CAMERA = 2

WIDTH = 1280
HEIGHT = 800
FPS = 60

WARMUP_SECONDS = 3.0


##############################################################
# Open Camera
##############################################################

def open_camera(index):

    cap = cv2.VideoCapture(
        index,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS
    )

    return cap


##############################################################
# Main
##############################################################

def main():

    print("Opening cameras...")

    left = open_camera(
        LEFT_CAMERA
    )

    right = open_camera(
        RIGHT_CAMERA
    )

    ##########################################################
    # Check Camera
    ##########################################################

    print(
        "Left opened :",
        left.isOpened()
    )

    print(
        "Right opened:",
        right.isOpened()
    )

    if not left.isOpened():

        print("Left camera failed.")

        right.release()

        return

    if not right.isOpened():

        print("Right camera failed.")

        left.release()

        return

    ##########################################################
    # Actual Camera Settings
    ##########################################################

    print()

    print(
        "Left:",
        int(
            left.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),
        "x",
        int(
            left.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),
        "@",
        left.get(
            cv2.CAP_PROP_FPS
        )
    )

    print(
        "Right:",
        int(
            right.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),
        "x",
        int(
            right.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),
        "@",
        right.get(
            cv2.CAP_PROP_FPS
        )
    )

    ##########################################################
    # Camera Warm-up
    ##########################################################

    print()

    print(
        f"Warming up cameras for "
        f"{WARMUP_SECONDS:.1f} seconds..."
    )

    warmup_start = time.perf_counter()

    warmup_frames = 0

    while (
        time.perf_counter()
        - warmup_start
        < WARMUP_SECONDS
    ):

        left_grab = left.grab()
        right_grab = right.grab()

        if not left_grab:
            print(
                "Warning: left camera "
                "grab failed during warm-up."
            )

        if not right_grab:
            print(
                "Warning: right camera "
                "grab failed during warm-up."
            )

        left.retrieve()
        right.retrieve()

        warmup_frames += 1

    print(
        "Camera warm-up finished."
    )

    print(
        "Warm-up frame pairs:",
        warmup_frames
    )

    ##########################################################
    # Capture One Stereo Pair
    ##########################################################

    print()
    print("Capturing stereo pair...")

    start_time = time.perf_counter()

    left_grab = left.grab()
    right_grab = right.grab()

    left_ok, left_frame = (
        left.retrieve()
    )

    right_ok, right_frame = (
        right.retrieve()
    )

    elapsed_time = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    ##########################################################
    # Capture Result
    ##########################################################

    print()

    print(
        "Left grab   :",
        left_grab
    )

    print(
        "Right grab  :",
        right_grab
    )

    print(
        "Left frame  :",
        left_ok,
        None
        if left_frame is None
        else left_frame.shape
    )

    print(
        "Right frame :",
        right_ok,
        None
        if right_frame is None
        else right_frame.shape
    )

    print(
        f"Capture pair time: "
        f"{elapsed_time:.2f} ms"
    )

    ##########################################################
    # Brightness Debug
    ##########################################################

    if left_ok:

        left_gray = cv2.cvtColor(
            left_frame,
            cv2.COLOR_BGR2GRAY
        )

        print(
            "Left mean brightness :",
            f"{left_gray.mean():.2f}"
        )

    if right_ok:

        right_gray = cv2.cvtColor(
            right_frame,
            cv2.COLOR_BGR2GRAY
        )

        print(
            "Right mean brightness:",
            f"{right_gray.mean():.2f}"
        )

    ##########################################################
    # Save Images
    ##########################################################

    print()

    if left_ok:

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "left_test.jpg"),
            left_frame
        )

        print(
            "Saved left_test.jpg"
        )

    else:

        print(
            "Left image was not saved."
        )

    if right_ok:

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "right_test.jpg"),
            right_frame
        )

        print(
            "Saved right_test.jpg"
        )

    else:

        print(
            "Right image was not saved."
        )

    ##########################################################
    # Release Cameras
    ##########################################################

    left.release()
    right.release()

    print()
    print("Cameras released.")
    print("Video test finished.")


##############################################################
# Entry
##############################################################

if __name__ == "__main__":
    main()