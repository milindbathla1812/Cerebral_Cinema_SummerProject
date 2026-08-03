import re
import numpy as np
from pathlib import Path

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
    Find corresponding fMRI file
    for an episode.
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

        windows.append(
            text[i:i+8]
        )

    return np.stack(windows)

def fuse_modalities(text_window, video_window):
    """
    Concatenate text and video features.
    """

    return np.concatenate(
        [
            text_window,
            video_window
        ],
        axis=-1
    )

def load_npy(path):
    """
    Load a numpy array.
    """

    return np.load(
        path,
        mmap_mode="r"
    )

def prepare_sample(text, video, tr):
    """
    Prepare one Transformer input.
    """

    text_window = np.asarray(
        text[tr:tr+8],
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
    Compute mean Pearson correlation
    across brain parcels.
    """

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)

    prediction = prediction - prediction.mean()
    target = target - target.mean()

    numerator = np.sum(
        prediction * target
    )

    denominator = np.sqrt(
        np.sum(prediction**2)
        *
        np.sum(target**2)
    )

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )