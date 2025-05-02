import cv2
import numpy as np


def blur_box(
    image: np.ndarray, box: tuple[int, int, int, int], size: int = 10
) -> np.ndarray:
    """Blurred rain on image."""
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = cv2.GaussianBlur(
        image[y1:y2, x1:x2], (size, size), cv2.BORDER_ISOLATED
    )
    return image


def pixelate(roi: np.ndarray, pixel_size: int = 8) -> np.ndarray:
    """
    Pixelate whole image.

    Parameters:
    ----------------
        roi (np.ndarray): Image to pixelate.
        pixel_size (int): Pixel size.

    Returns:
    ----------------
        np.ndarray: Pixelated image.
    """
    temp = cv2.resize(
        roi,
        (roi.shape[1] // pixel_size, roi.shape[0] // pixel_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.resize(
        temp, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def pixelate_box(
    image: np.ndarray, box: tuple[int, int, int, int], pixel_size: int = 8
) -> np.ndarray:
    """
    Pixelate whole image.

    Parameters:
    ----------------
        roi (np.ndarray): Image to pixelate.
        box : tuple[int,int,int,int]: Box to pixelate.
        pixel_size (int): Pixel size.

    Returns:
    ----------------
        np.ndarray: Pixelated image.
    """
    roi = image[box[1] : box[3], box[0] : box[2]]
    downscaled_width = max(1, roi.shape[1] // pixel_size)
    downscaled_height = max(1, roi.shape[0] // pixel_size)
    box_resized = cv2.resize(
        roi, (downscaled_width, downscaled_height), interpolation=cv2.INTER_NEAREST
    )
    image[box[1] : box[3], box[0] : box[2]] = cv2.resize(
        box_resized, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return image
