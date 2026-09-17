# Image acquisition


import cv2


def start_camera():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Camera could not be opened.")

        return

    print("Camera started.")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera frame.")

            break

        cv2.imshow(
            "HMER Camera",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    start_camera()
    