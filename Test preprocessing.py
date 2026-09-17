import cv2

def preprocess_image(image_path):

    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    normalized = cv2.resize(
        binary,
        (224, 224)
    )

    return (
        original,
        gray,
        binary,
        normalized
    )