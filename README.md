# 🌌 SpectralCA: Spectral Cross-Attention for Hyperspectral Image Classification

This repository implements a hybrid model architecture for hyperspectral image classification, combining **3D convolutions** with a custom-designed **Spectral Cross-Attention (SpectralCA)** mechanism. The method fuses spatial and spectral representations symmetrically and efficiently, achieving a balance between accuracy and inference speed.

The model improves on the MDvT architecture by replacing MobileViTBlock with the custom SpectralCA block.

---

## 📚 Table of Contents

- [🧠 Research Summary](#-research-summary)
- [🛠 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📈 SpectralCA Architecture](#-spectralca-architecture)
- [📊 Results](#-results)
- [🧪 Dataset](#-dataset)
- [💡 Semi-Supervised Learning](#-semi-supervised-learning)
- [📄 License & Credits](#-license--credits)

---

## 🧠 Research Summary

The SpectralCA block is designed for **hyperspectral image (HSI)** classification, with a special focus on:

- Preserving **geometric structure** of scenes
- Enhancing **feature fusion** via dual-directional **cross-attention**
- Reducing model **size and inference time**

This method was evaluated on the WHU-Hi-HongHu dataset and compared to MobileViTBlock within the MDvT architecture.

---

## 🛠 Project Structure

```
spectral-cross-attention/
├── spectral_cross_attention_with_ssl.ipynb   # Core notebook
├── README.md                                # This file
```

---

## 🚀 Getting Started

Install dependencies:

```bash
pip install torch numpy matplotlib scikit-learn
```

Run the notebook:

```bash
jupyter notebook spectral_cross_attention_with_ssl.ipynb
```

---

## 📈 SpectralCA Architecture

The architecture consists of two branches:

- A **2D convolutional stream** for spatial features
- A **3D convolutional stream** for spectral features
- Two **cross-attention heads**:
  - Spatial → Spectral
  - Spectral → Spatial

Final feature maps are concatenated and projected using 3D convolutions.

📌 *Diagram of the SpectralCA block:*

<img width="468" alt="image" src="https://github.com/user-attachments/assets/18b3beb5-64cb-4f46-ac96-f0392364eec4" />


---

## 📊 Results

| Model              | OA     | AA     | Kappa  | Train Time (s) | Inference (s) | Params (M) |
|-------------------|--------|--------|--------|----------------|----------------|-------------|
| MobileViTBlock     | 0.974  | 0.9739 | 0.9671 | 545.64         | 94             | 7.746       |
| **SpectralCA**     | 0.9334 | 0.9308 | 0.9162 | 274.38         | 50             | 6.628       |

✅ **2× faster** training and inference  
✅ **~1.1M fewer** parameters  
✅ Accuracy remains above **93%**

---

## 💡 Semi-Supervised Learning

The model supports **SSL via proxy labeling** (self-training).  
The process:

1. Train on a small labeled subset
2. Predict pseudo-labels on unlabeled data
3. Retrain on confident predictions
4. Iterate until convergence

SSL enables effective training in scenarios with limited labeled data — highly relevant for agricultural and military HSI applications.
