import torch
import numpy as np
import matplotlib.pyplot as plt

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
    Predict brain activity and compare with
    ground truth fMRI.
    """

    text = load_npy(text_file)

    video = load_npy(video_file)

    fmri = load_npy(fmri_file)

    usable = min(
        len(text)-7,
        len(video),
        len(fmri)-5
    )
    usable = min(
        len(text)-7,
        len(video),
        len(fmri)-5
    )
    predictions = []
    targets = []

    with torch.no_grad():

        for tr in range(usable):

            fused = prepare_sample(
                text,
                video,
                tr
            )

            fused = torch.from_numpy(
                fused
            ).unsqueeze(0).to(DEVICE)

            prediction = model(fused)

            prediction = (
                prediction
                .cpu()
                .numpy()[0]
            )

            target = fmri[tr+5]

            predictions.append(prediction)

            targets.append(target)

    predictions = np.stack(predictions)

    targets = np.stack(targets)

    scores = []

    for i in range(len(predictions)):

        scores.append(

            pearson_score(
                predictions[i],
                targets[i]
            )

        )

    mean_score = float(
        np.mean(scores)
    )

    plt.figure(figsize=(10,5))

    plt.plot(
        targets[:,0],
        label="Actual"
    )

    plt.plot(
        predictions[:,0],
        label="Predicted"
    )

    plt.xlabel("TR")

    plt.ylabel("BOLD Signal")

    plt.title(
        "Predicted vs Actual fMRI"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "prediction.png"
    )

    plt.close()

    return {

        "pearson": mean_score,

        "prediction": predictions,

        "target": targets,

        "graph": "prediction.png"

    }