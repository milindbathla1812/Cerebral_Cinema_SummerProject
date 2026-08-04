import torch
import numpy as np

from model import load_model
from utils import (
    load_npy,
    prepare_sample,
    pearson_score
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(
    "models/best_tribe_lite_epoch20.pth",
    DEVICE
)


def predict(text_file, video_file, fmri_file):
    """
    Predict brain activity for one sample.
    """

    print("Loading files...")

    text = load_npy(text_file)
    video = load_npy(video_file)
    fmri = load_npy(fmri_file)

    # Ensure enough frames are available
    if len(text) < 8:
        raise ValueError("Text embedding file must contain at least 8 time steps.")

    if len(video) < 1:
        raise ValueError("Video embedding file is empty.")

    if len(fmri) < 6:
        raise ValueError("fMRI file must contain at least 6 TRs.")

    tr = 0

    fused = prepare_sample(
        text,
        video,
        tr
    )

    fused = torch.from_numpy(fused).unsqueeze(0).to(DEVICE)

    print("Running model...")

    with torch.no_grad():
        prediction = model(fused)

    prediction = prediction.squeeze(0).cpu().numpy()

    target = np.asarray(
        fmri[5],
        dtype=np.float32
    )

    score = pearson_score(
        prediction,
        target
    )

    print(f"Pearson Score : {score:.4f}")
    print("Inference completed.")

    return {
        "pearson": round(float(score), 4),
        "prediction": prediction.tolist(),
        "target": target.tolist(),
        "graph": None
    }