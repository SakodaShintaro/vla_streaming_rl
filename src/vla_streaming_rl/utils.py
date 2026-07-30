# SPDX-License-Identifier: MIT
import cv2
import numpy as np


def concat_images(image_list: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate images from image_list side by side.
    Args:
        image_list (list[np.ndarray]): The list of rgb images to concatenate.
    Returns:
        np.ndarray: The concatenated bgr image (np.uint8).
    """
    max_height = max(img.shape[0] for img in image_list)
    padded_images = []
    for img in image_list:
        padding = np.zeros((max_height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        padded_img = np.vstack((img, padding))
        padded_images.append(padded_img)
    concat = cv2.hconcat(padded_images)
    bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    return bgr_array


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 format.
    Args:
        image (np.ndarray): The original image
    Returns:
        np.ndarray: The image in uint8 format
    """
    if image.dtype == np.float32:
        img_converted = image * 255.0
        img_converted = np.clip(img_converted, 0, 255).astype(np.uint8)
        return img_converted
    else:
        return image.copy()


def add_text_label_on_top(image: np.ndarray, text: str) -> np.ndarray:
    """
    Add a text label on top of the image.
    Args:
        image (np.ndarray): The original image (RGB format, uint8 expected)
        text (str): The text to add (e.g., variable name)
    Returns:
        np.ndarray: The image with text label added on top
    """
    # Parameters for text drawing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (50, 50, 50)

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Height of text area (including margin)
    # FFMPEG libx264 encoder requires height to be divisible by 2, so adjust to even number
    label_height = text_height + baseline + 10
    if label_height % 2 != 0:
        label_height += 1

    # Create image area for text (same width as the image)
    label_img = np.full((label_height, image.shape[1], 3), bg_color, dtype=np.uint8)

    # Position text at the left
    text_x = 5
    text_y = text_height + 5
    cv2.putText(label_img, text, (text_x, text_y), font, font_scale, text_color, thickness)

    # Concatenate text area and original image vertically
    result = np.vstack((label_img, image))

    return result


_CAPTION_LINE_BUDGET = 5


def overlay_caption(image: np.ndarray, text: str) -> np.ndarray:
    """Append a word-wrapped text caption on a dark band below ``image``.

    Used to annotate a panel with free-form text (e.g. the LIBERO task prompt)
    that is too long to fit on a single-line panel label. The band always
    reserves ``_CAPTION_LINE_BUDGET`` lines of height so the returned frame size
    stays constant across a run even when the text length changes step to step
    (the video writer requires identical frame sizes); text beyond the budget is
    dropped and a shorter text simply leaves blank lines.
    """
    image = convert_to_uint8(image)

    width = image.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    # Greedy word-wrap to the panel width.
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        (trial_width, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if trial_width <= width - 10 or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    lines = lines[:_CAPTION_LINE_BUDGET]

    (_, text_height), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_height = text_height + baseline + 4
    band_height = line_height * _CAPTION_LINE_BUDGET + 6
    if band_height % 2 != 0:  # keep even for the libx264 video writer
        band_height += 1

    band = np.full((band_height, width, 3), (50, 50, 50), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            band, line, (5, (i + 1) * line_height), font, font_scale, (255, 255, 255), thickness
        )
    return np.vstack((image, band))


def concat_labeled_images(panels: dict[str, np.ndarray]) -> np.ndarray:
    """Lay out named image panels side by side as a single RGB strip.

    ``panels`` maps a label to an RGB image (uint8 or float32). Panels are
    drawn left-to-right in insertion order, each captioned with its label,
    converted to uint8, and padded to a common height. Agents and envs can
    contribute arbitrary extra panels (e.g. a bird's-eye trajectory view)
    without this function knowing the panel set in advance.
    """
    labeled_images = [
        add_text_label_on_top(convert_to_uint8(img), label) for label, img in panels.items()
    ]
    final_image_bgr = concat_images(labeled_images)
    final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
    return final_image_rgb
