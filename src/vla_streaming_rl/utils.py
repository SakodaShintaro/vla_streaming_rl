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


def create_reward_image(pred_reward: float | None, actual_reward: float) -> np.ndarray:
    """
    Visualize the reward (simple text display). The panel is a fixed 200x200
    regardless of inputs. ``pred_reward=None`` means the agent does not predict
    the reward (e.g. SimLingo): only the actual reward is shown, with the Pred
    / Error lines omitted.
    """
    height, width = (200, 200)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    if pred_reward is not None:
        pred_text = f"Pred: {pred_reward:.3f}"
        cv2.putText(img, pred_text, (10, 40), font, font_scale, (0, 128, 255), thickness)

    # Actual reward
    actual_text = f"Actual: {actual_reward:.3f}"
    cv2.putText(img, actual_text, (10, 80), font, font_scale, (255, 0, 0), thickness)

    if pred_reward is not None:
        error = abs(pred_reward - actual_reward)
        error_text = f"Error: {error:.3f}"
        cv2.putText(img, error_text, (10, 120), font, font_scale, (255, 255, 255), thickness)

    return img


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
