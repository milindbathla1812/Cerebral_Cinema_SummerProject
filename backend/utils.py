import re
import numpy as np
from pathlib import Path


def load_npy(path):
    """
    Load numpy file.
    """
    return np.load(path, allow_pickle=False)


def episode_name(path):
    """
    Extract episode identifier such as s01e03
    from a filename.
    """

    match = re.search(
        r"s\d{2}e\d{2}[a-z]?",
        Path(path).stem
    )

    if match is None:
        raise ValueError(f"Cannot parse episode from {path}")

    return match.group(0)


def get_fmri_file(folder, episode):
    """
    Find corresponding fMRI file.
    """

    files = list(
        Path(folder).glob(f"*{episode}.npy")
    )

    if len(files) == 0:
        return None

    return files[0]


def build_text_windows(text):
    """
    Convert transcript embeddings into
    overlapping windows of length 8.
    """

    windows = []

    for i in range(len(text) - 7):
        windows.append(text[i:i + 8])

    return np.stack(windows)


def fuse_modalities(text_window, video_window):
    """
    Concatenate text and video embeddings.
    """

    print("Text window :", text_window.shape)
    print("Video window:", video_window.shape)

    text_window = np.asarray(text_window, dtype=np.float32)
    video_window = np.asarray(video_window, dtype=np.float32)

    # --------------------------
    # Remove unnecessary singleton dimensions
    # --------------------------

    text_window = np.squeeze(text_window)
    video_window = np.squeeze(video_window)

    # --------------------------
    # Text should be (8,2048)
    # --------------------------

    if text_window.ndim == 1:
        text_window = text_window[None, :]

    # --------------------------
    # Video should become (8,768)
    # --------------------------

    if video_window.ndim == 1:

        video_window = np.repeat(
            video_window[None, :],
            text_window.shape[0],
            axis=0
        )

    elif video_window.ndim == 2:

        if video_window.shape[0] == 1:

            video_window = np.repeat(
                video_window,
                text_window.shape[0],
                axis=0
            )

        elif video_window.shape[0] != text_window.shape[0]:

            video_window = np.repeat(
                video_window[0:1],
                text_window.shape[0],
                axis=0
            )

    print("Text final :", text_window.shape)
    print("Video final:", video_window.shape)

    fused = np.concatenate(
        [text_window, video_window],
        axis=-1
    )

    print("Fused shape :", fused.shape)

    return fused


def prepare_sample(text, video, tr):
    """
    Prepare one Transformer input.
    """

    text_window = np.asarray(
        text[tr:tr + 8],
        dtype=np.float32
    )

    video_window = np.asarray(
        video[tr],
        dtype=np.float32
    )

    fused = fuse_modalities(
        text_window,
        video_window
    )

    return fused


def pearson_score(prediction, target):
    """
    Mean Pearson correlation.
    """

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    prediction = prediction - prediction.mean()
    target = target - target.mean()

    numerator = np.sum(prediction * target)

    denominator = np.sqrt(
        np.sum(prediction ** 2)
        * np.sum(target ** 2)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)