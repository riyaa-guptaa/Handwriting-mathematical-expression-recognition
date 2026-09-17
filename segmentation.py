import cv2


def find_components(binary_image):

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    components = []

    for contour in contours:

        x, y, width, height = (
            cv2.boundingRect(contour)
        )

        if width < 3 or height < 3:

            continue

        components.append(
            (x, y, width, height)
        )

    components.sort(
        key=lambda item: item[0]
    )

    return components