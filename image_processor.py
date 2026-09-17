# Image preprocessing

# Now we move into the most important ECE part: digital image processing.




import cv2
import numpy as np

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)


def read_image(image_path):

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    return image


def convert_to_grayscale(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def remove_noise(gray):

    return cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


def threshold_image(image):

    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    return binary


def resize_image(image):

    return cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def normalize_image(image):

    image = image.astype(
        np.float32
    )

    image = image / 255.0

    return image


def preprocess_image(image_path):

    original = read_image(
        image_path
    )

    gray = convert_to_grayscale(
        original
    )

    filtered = remove_noise(
        gray
    )

    binary = threshold_image(
        filtered
    )

    resized = resize_image(
        binary
    )

    normalized = normalize_image(
        resized
    )

    normalized = np.expand_dims(
        normalized,
        axis=-1
    )

    return (
        original,
        gray,
        binary,
        normalized
    )