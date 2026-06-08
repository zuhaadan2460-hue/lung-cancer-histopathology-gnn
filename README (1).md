# 🧠 GNN Framework for Representation Learning in Whole-Slide Histopathology Images

> **Final Year Project | University of Agriculture, Faisalabad**
> Department of Computer Science — BINFO-606 Final Year Project-I
> **Author:** Noor ul Ain Fatima (2022-ag-7673) | **Supervisor:** Dr.Muhmmad Zeeshan Asaf

---

## 📌 Overview

This project proposes a **Graph Neural Network (GNN)-based representation learning framework** for automated classification of lung cancer subtypes — **Lung Adenocarcinoma (LUAD)** and **Lung Squamous Cell Carcinoma (LUSC)** — from Whole Slide Histopathology Images (WSIs).

The framework addresses key challenges in computational pathology: gigapixel image sizes, stain variability, lack of annotated data, and the loss of spatial tissue relationships in conventional patch-based methods.

---

## 🎯 Objectives

- **Objective 1 — Informative Patch Selection:** Extract diagnostically relevant tissue patches from WSIs using tissue masking, variance filtering, and balanced patch selection.
- **Objective 2 — Stain Normalization:** Apply LAB-based gentle stain normalization to reduce inter-slide color variability while preserving tissue morphology.
- **Objective 3 — Graph-Based Representation Learning:** Construct tissue graphs and train a self-supervised Graph Autoencoder + GCN classifier for LUAD/LUSC classification.

---

## 🗂️ Dataset

| Source | Description |
|--------|-------------|
| [TCGA-LUAD](https://portal.gdc.cancer.gov/) | Whole slide images of Lung Adenocarcinoma |
| [TCGA-LUSC](https://portal.gdc.cancer.gov/) | Whole slide images of Lung Squamous Cell Carcinoma |

- **Total WSIs used:** 42 (SVS format)
- **Patches extracted:** ~2,770 tissue-rich patches (256×256 pixels)
- **Patches per slide:** 35 (LUAD), 100 (LUSC)

---

## 🔧 Methodology

```
WSI (SVS) ──► Thumbnail Generation ──► Tissue Mask Generation
    ──► Padding ──► Patch Extraction ──► Tissue Filtering
    ──► LAB Stain Normalization ──► Coordinate-Based Reconstruction
    ──► ResNet50 Feature Extraction ──► KNN Graph Construction
    ──► Graph Autoencoder Pretraining (Self-Supervised)
    ──► GCN Fine-Tuning ──► LUAD / LUSC Classification
```

### Key Components

| Component | Details |
|-----------|---------|
| **Slide Reading** | OpenSlide (SVS format) |
| **Tissue Masking** | Grayscale thresholding + morphological operations |
| **Patch Size** | 256×256 px, stride 256 |
| **Filtering Thresholds** | Tissue ratio > 0.2, Variance > 300 |
| **Stain Normalization** | LAB-based gentle normalization (H&E) |
| **Feature Extractor** | Pretrained ResNet50 → 2048-dim feature vectors |
| **Graph Construction** | KNN connectivity (k=5) |
| **Pretraining** | Graph Autoencoder (GAE) — self-supervised, 30 epochs |
| **Classifier** | Graph Convolutional Network (GCN) — fine-tuned, 50 epochs |
| **Visualization** | PCA, t-SNE, confusion matrix |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **1.0** |
| Precision | **1.0** |
| Recall | **1.0** |
| F1-Score | **1.0** |
| AUC-ROC | **1.0** |

The model perfectly classified all test samples with no false positives or false negatives. PCA and t-SNE visualizations confirmed clear separation between LUAD and LUSC graph embeddings.

---

## 🧪 Requirements

```bash
pip install openslide-python opencv-python torch torch-geometric
pip install scikit-learn matplotlib seaborn numpy torchvision
```

> **Note:** OpenSlide requires system-level installation. On Ubuntu: `sudo apt-get install openslide-tools`

### Main Libraries

- `Python 3.8+`
- `PyTorch` + `PyTorch Geometric` (GNN)
- `OpenSlide` (WSI reading)
- `OpenCV` (image processing)
- `torchvision` (ResNet50)
- `scikit-learn` (evaluation metrics, PCA, t-SNE)
- `matplotlib` / `seaborn` (visualization)

---

## 📁 Project Structure

```
├── data/
│   ├── LUAD/               # TCGA-LUAD SVS slides
│   └── LUSC/               # TCGA-LUSC SVS slides
├── preprocessing/
│   ├── thumbnail.py        # Thumbnail generation
│   ├── tissue_mask.py      # Tissue mask generation
│   ├── patch_extraction.py # Patch extraction & filtering
│   └── stain_norm.py       # LAB-based stain normalization
├── graph/
│   ├── feature_extract.py  # ResNet50 feature extraction
│   ├── graph_build.py      # KNN graph construction
│   └── visualize_graph.py  # Graph visualization
├── models/
│   ├── graph_autoencoder.py # Self-supervised GAE pretraining
│   └── gcn_classifier.py    # GCN fine-tuning & classification
├── evaluation/
│   ├── metrics.py           # Accuracy, Precision, Recall, F1, AUC
│   └── visualize.py         # PCA, t-SNE, confusion matrix
├── notebooks/
│   └── pipeline.ipynb       # End-to-end Jupyter notebook
├── results/                 # Output figures and metrics
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/gnn-histopathology-wsi.git
   cd gnn-histopathology-wsi
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from [TCGA GDC Portal](https://portal.gdc.cancer.gov/) and place SVS files in `data/LUAD/` and `data/LUSC/`

4. **Run preprocessing**
   ```bash
   python preprocessing/patch_extraction.py
   python preprocessing/stain_norm.py
   ```

5. **Build graphs and extract features**
   ```bash
   python graph/feature_extract.py
   python graph/graph_build.py
   ```

6. **Train the model**
   ```bash
   python models/graph_autoencoder.py   # Pretraining
   python models/gcn_classifier.py      # Fine-tuning
   ```

7. **Evaluate**
   ```bash
   python evaluation/metrics.py
   python evaluation/visualize.py
   ```

---

## 📈 Visual Results

| Visualization | Description |
|---------------|-------------|
| Tissue Mask | Separates tissue from background |
| Before/After Stain Norm | Improved color consistency |
| LUAD Graph (24 nodes, 120 edges) | KNN tissue patch graph |
| LUSC Graph (100 nodes, 500 edges) | KNN tissue patch graph |
| Self-Supervised Loss Curve | Converges over 30 epochs |
| GCN Training Curve | Reaches 1.0 accuracy |
| Confusion Matrix | Perfect diagonal (0 misclassifications) |
| PCA / t-SNE | Clear LUAD vs LUSC cluster separation |

---

## 🔬 Key Contributions

- Multi-stage preprocessing pipeline optimized for WSI analysis
- LAB-based gentle stain normalization preserving morphological details
- Graph-based tissue representation capturing spatial relationships between patches
- Self-supervised pretraining reducing dependency on labeled annotations
- Perfect classification performance on LUAD/LUSC lung cancer subtypes

---

## 📚 References

Key references used in this work:

- Adnan et al. (2020). Representation learning of histopathology images using GNNs. *CVPR Workshops.*
- Chen et al. (2022). Fast and scalable search of WSIs via self-supervised deep learning. *Nature Biomedical Engineering.*
- Brussee et al. (2024). Graph neural networks in histopathology: emerging trends. *arXiv.*
- Ilse et al. (2018). Attention-based deep multiple instance learning. *ICML.*

Full reference list available in the [project report](./report/).

---

## 🏫 Institution

**University of Agriculture, Faisalabad (UAF)**
Department of Computer Science, Faculty of Sciences
Course: BI-606 Final Year Project-I 6(0-6) | Year: 2026

---

## 📄 License

This project is submitted as an academic final year project at UAF. For reuse or citation, please contact the author.
