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


# The AnimalAI task prompt plus its live scalars wraps to 9 lines at the render
# width, so anything less cuts the caption off mid-sentence.
_CAPTION_LINE_BUDGET = 20
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# What a VLM writes is not ASCII -- curly quotes, dashes, ellipses -- and the
# Hershey fonts OpenCV draws with have no glyph for any of it, so every one of
# them lands in a panel as "???". Folded to their ASCII shapes on the way in.
_DRAWABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u2022": "*",
        "\u00a0": " ",
    }
)


def wrap_text(text: str, width: int, font_scale: float, thickness: int) -> list[str]:
    """Greedily word-wrap ``text`` to a pixel ``width``."""
    lines: list[str] = []
    current = ""
    for word in text.translate(_DRAWABLE).split():
        trial = f"{current} {word}".strip()
        (trial_width, _), _ = cv2.getTextSize(trial, _FONT, font_scale, thickness)
        if trial_width <= width - 10 or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_text_panel(text: str, width: int, height: int) -> np.ndarray:
    """Free-form text word-wrapped onto a dark panel of exactly ``width`` x
    ``height``.

    A panel rather than a caption band, so free-form text can stand as its own
    entry in the render strip. The size is fixed by the arguments and not by the
    text, which is what the stable-panel contract needs; text past the last line
    that fits is dropped.
    """
    font_scale = 0.45
    thickness = 1
    (_, text_height), baseline = cv2.getTextSize("Ag", _FONT, font_scale, thickness)
    line_height = text_height + baseline + 4

    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    lines = wrap_text(text, width, font_scale, thickness)[: height // line_height]
    for i, line in enumerate(lines):
        cv2.putText(
            panel, line, (5, (i + 1) * line_height), _FONT, font_scale, (235, 235, 235), thickness
        )
    return panel


# The chain's conversation, drawn the way a chat log reads: what the agent was
# shown on one side, what it wrote on the other.
_USER_BUBBLE = (78, 62, 48)
_ASSISTANT_BUBBLE = (52, 84, 52)
_BUBBLE_TEXT = (238, 238, 238)
_BUBBLE_FRACTION = 0.78


def render_conversation_panel(
    turns: list[dict], status: str, width: int, height: int
) -> np.ndarray:
    """A conversation drawn as chat bubbles on a panel of exactly ``width`` x
    ``height``, under a line of ``status``.

    The standing task is not drawn: it is the same every tick and the trainer
    already captions the environment panel with it. What is drawn is what
    changes -- the turn the agent was shown this tick and what the chain wrote
    about the ones before it.

    The newest turn sits at the bottom and older ones are laid above it until the
    panel is full, so the latest exchange is always visible however long the
    conversation has grown. User turns are left-aligned, the chain's own replies
    right-aligned, each on its own colour. ``status`` is drawn as a strip along
    the top, where it stays put as the turns scroll under it.
    """
    font_scale = 0.45
    thickness = 1
    (_, text_height), baseline = cv2.getTextSize("Ag", _FONT, font_scale, thickness)
    line_height = text_height + baseline + 4
    padding = 6
    gap = 6
    bubble_width = int(width * _BUBBLE_FRACTION)

    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    status_height = line_height + padding * 2
    cv2.rectangle(panel, (0, 0), (width, status_height), (48, 48, 48), cv2.FILLED)
    cv2.putText(
        panel,
        status,
        (gap, padding + line_height - baseline),
        _FONT,
        font_scale,
        (190, 190, 190),
        thickness,
    )

    bottom = height - gap
    said = [turn for turn in turns if turn and turn["role"] in ("user", "assistant")]
    for turn in reversed(said):
        text = " ".join(
            part["text"] for part in turn["content"] if part["type"] == "text" and part["text"]
        )
        lines = wrap_text(text if text else "(frame only)", bubble_width, font_scale, thickness)
        bubble_height = line_height * len(lines) + padding * 2
        top = bottom - bubble_height
        if top < status_height + gap:
            break
        assistant = turn["role"] == "assistant"
        left = width - gap - bubble_width if assistant else gap
        cv2.rectangle(
            panel,
            (left, top),
            (left + bubble_width, bottom),
            _ASSISTANT_BUBBLE if assistant else _USER_BUBBLE,
            cv2.FILLED,
        )
        for index, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (left + padding, top + padding + (index + 1) * line_height - baseline),
                _FONT,
                font_scale,
                _BUBBLE_TEXT,
                thickness,
            )
        bottom = top - gap
    return panel


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
    font = _FONT
    font_scale = 0.4
    thickness = 1

    lines = wrap_text(text, width, font_scale, thickness)[:_CAPTION_LINE_BUDGET]

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
