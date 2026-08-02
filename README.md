# 🧠 Cerebral Cinema
<div align="center">

<table>
<tr>
<td align="left" width="35%">
<img src="images/logo.jpg.jpeg" alt="BCS IIT Kanpur" width="220">
</td>

<td align="center" width="65%">
<img src="images/cerebral.png" alt="Cerebral Cinema" width="500">
</td>
</tr>
</table>

</div>

This GitHub repository contains data and code for the Cerebral Cinema Project with has close resemblence with the Algonauts Competition 2025. 
<div align="center">
  
### Predicting Human Brain Activity from Multimodal Movie Stimuli using Transformers



[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

</div>

---

## 📖 Overview

**Cerebral Cinema** is a multimodal brain encoding framework that predicts fMRI responses while participants watch episodes of the TV series *Friends*.

The project combines **semantic information from dialogue transcripts** and **visual representations extracted from video clips**, fuses them through a **Transformer encoder**, and predicts whole-brain activity represented using the **1000-parcel Schaefer atlas**.

The framework is inspired by recent advances in multimodal brain decoding, including the **TRIBE** architecture and the **Algonauts Challenge**, while extending them to movie-based naturalistic stimuli.

---

## 🎯 Objectives

- Predict cortical brain activity from natural movie stimuli.
- Learn joint representations of language and vision.
- Investigate multimodal fusion for neuroscience.
- Build an interactive platform for visualizing predicted brain activity.

---
# 🧠 Understanding BOLD Signals

Functional Magnetic Resonance Imaging (**fMRI**) measures brain activity indirectly through the **Blood Oxygenation Level-Dependent (BOLD)** signal. Rather than recording neuronal firing directly, the BOLD signal captures changes in blood oxygenation that occur following neural activation.

When a group of neurons becomes active, it consumes more oxygen. In response, the brain increases local cerebral blood flow, delivering oxygen-rich blood that exceeds the amount actually consumed. This change alters the ratio of oxygenated to deoxygenated hemoglobin, producing measurable variations in the MRI signal known as the **BOLD response**.

Since this vascular response is delayed relative to neural activity, the BOLD signal typically peaks **4–6 seconds after a stimulus**, corresponding to approximately **5 TRs** in our dataset. To account for this physiological delay, our model aligns multimodal movie features with fMRI recordings using a **5-TR temporal shift**, ensuring that each predicted brain response corresponds to the appropriate neural event.

### Key Characteristics

- 🧠 Measures **hemodynamic response**, not electrical activity.
- ⏳ Peaks approximately **4–6 seconds** after neural activation.
- 📈 Represents changes in blood oxygenation caused by neuronal activity.
- 🎯 Used as the target signal for predicting cortical responses across **1000 Schaefer brain parcels**.
- 🔄 A **5-TR delay** is incorporated to synchronize movie stimuli with the corresponding BOLD response.

<p align="center">
<img src="images/BOLD.png" width="700">
</p>

# 🏗 Project Pipeline

<p align="center">
<img src="https://github.com/milindbathla1812/Cerebral_Cinema_SummerProject/blob/main/images/pipeline.png" width="900">
</p>

The overall pipeline consists of:

1. Movie & transcript preprocessing
2. fMRI preprocessing
3. Feature Extraction
4. Multimodal fusion
5. Transformer-based brain encoding
6. fMRI prediction
7. Brain activity visualization

---

# 📂 Dataset

The project uses the **Friends** naturalistic movie dataset collected from **four participants**.

Each participant watched episodes from **Friends Seasons 1–6** while undergoing functional MRI scans.

The dataset contains:

- 🎬 Video clips
- 💬 Dialogue transcripts
- 🧠 fMRI recordings
- ⏱ Time-aligned stimulus-response pairs

### Dataset Statistics

| Modality | Description |
|-----------|-------------|
| Video | Friends Seasons 1–6 |
| Text | Episode transcripts |
| Subjects | 4 |
| Brain Regions | 1000 Schaefer parcels |
| Imaging | Functional MRI (BOLD) |

---

# ⚙️ Feature Extraction

### 📝 Text Features

Dialogue transcripts are encoded using **TinyLlama**, producing **2048-dimensional semantic embeddings**.

Temporal context is preserved using **8-TR sliding windows**.

---

### 🎥 Visual Features

Video clips are processed using **VideoMAE**, generating

**8 × 768-dimensional temporal visual embeddings**.

These embeddings capture both spatial and temporal visual information.

---

### 🔗 Multimodal Fusion

For every TR:

- Text window
- Video embedding

are concatenated to create a unified multimodal representation before being fed into the Transformer.

---

# 🧠 fMRI Preprocessing

Functional MRI recordings were preprocessed using **fMRIPrep**.

The preprocessing pipeline includes:

- Motion correction
- Slice timing correction
- Spatial normalization
- Registration to MNI space
- Schaefer-1000 cortical parcellation
- Session-wise z-score normalization

A **5-TR hemodynamic delay** is incorporated to align neural activity with presented stimuli.

---
<p align="center">
<img src="images/fMRI_prep.png" width="700">
</p>

# 🤖 Model Architecture

<p align="center">
<img src="images/Architecture.png" width="850">
</p>

Our model consists of:

- Modality Dropout
- Linear Projection
- Positional Encoding
- CLS Token
- Transformer Encoder
- Prediction Head

### Transformer Configuration

| Component | Value |
|------------|------|
| Encoder Layers | 4 |
| Attention Heads | 8 |
| Hidden Dimension | 384 |
| Feed Forward | 1024 |
| Activation | GELU |
| Output | 1000 Brain Parcels |

---

# 🚀 Training Strategy

Training incorporates several modern optimization techniques:

- ✅ AdamW Optimizer – Optimizes model parameters using adaptive learning rates while decoupling weight decay for better regularization.
- ✅ Cosine Annealing Learning Rate – Gradually decreases the learning rate following a cosine schedule to improve convergence and stability.
- ✅ Early Stopping – Stops training automatically when the validation performance no longer improves, preventing overfitting.
- ✅ Stochastic Weight Averaging (SWA) – Averages weights from multiple training checkpoints to obtain a flatter optimum and better generalization.
- ✅ Modality Dropout – Randomly masks one or more input modalities during training to improve robustness against missing or noisy data.

Training:

- Seasons **1–5**

Testing:

- Season **6**

---

# 📈 Results

Example prediction from one cortical parcel:

<p align="center">
<img src="assets/prediction.png" width="800">
</p>

Performance metric:

- Pearson Correlation

| Split | Pearson |
|---------|----------|
| Validation | xx.xxx |
| Test | xx.xxx |

---

# 🌐 Interactive Website

A web interface is being developed to explore multimodal brain decoding interactively.

### Planned Features

- 🎬 Upload movie clips
- 💬 View transcript embeddings
- 🧠 Predict brain activity
- 📈 Compare predicted vs actual fMRI
- 🌍 Interactive cortical brain visualization
- 📊 Attention map visualization

---

# 📁 Repository Structure

```text
Cerebral-Cinema/
│
├── assets/
│   ├── logo.png
│   ├── pipeline.png
│   ├── model_architecture.png
│   └── prediction.png
│
├── notebooks/
│
├── models/
│
├── transcripts/
│
├── video_windows/
│
├── fmri/
│
├── website/
│
├── utils/
│
├── train.py
├── evaluate.py
├── inference.py
│
└── README.md
```

---

# 🛠 Installation

```bash
git clone https://github.com/yourusername/Cerebral-Cinema.git

cd Cerebral-Cinema

pip install -r requirements.txt
```

---

# ▶️ Training

```bash
python train.py
```

---

# 📊 Evaluation

```bash
python evaluate.py
```

---

# 📜 Citation

If you use this work, please cite:

```bibtex
@misc{cerebralcinema2026,
  title={Cerebral Cinema: Predicting Human Brain Activity from Multimodal Movie Stimuli},
  author={Milind Bathla},
  year={2026}
}
```

---

# 🙏 Acknowledgements

- Meta AI (VideoMAE)
- TinyLlama
- fMRIPrep
- CNeuroMod
- Schaefer Atlas
- PyTorch
- Algonauts Challenge
- TRIBE
