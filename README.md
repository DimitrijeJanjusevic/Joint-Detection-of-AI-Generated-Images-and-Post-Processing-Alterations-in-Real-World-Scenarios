# Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios

**Sapienza University of Rome — Computer Vision Course (Spring 2026)**  
**Prof. Irene Amerini**  
**Student: Dimitrije Janjušević — ID: 2285729**


## Project Overview

This project uses a unified multi-task deep learning framework that simultaneously addresses two questions about any input image:

1. **Is the image real or AI-generated?** (binary classification)
2. **What post-processing transformation was applied?** (3-class: original / internet-transmitted / re-digitized)

A shared ResNet-50 backbone extracts visual features, which are then passed to two independent classification heads trained jointly with a weighted combined loss function. The project evaluates whether joint multi-task training improves, degrades, or leaves unchanged performance compared to single-task baselines.


## Dataset

**RRDataset** — a real-world robustness benchmark containing real photographs and AI-generated images across three post-processing splits.

Download: [https://zenodo.org/records/14963880](https://zenodo.org/records/14963880)

**Subset used:** 2000 training + 500 validation images per bucket (6 buckets), totaling 12,000 training and 3,000 validation images. All classes are perfectly balanced.

---

## Architecture

- **Backbone:** ResNet-50 pretrained on ImageNet (transfer learning)
- **Head 1:** Linear(2048 → 2) — real/fake classification
- **Head 2:** Linear(2048 → 3) — transformation type classification
- **Loss:** `L = α × CrossEntropy(real/fake) + β × CrossEntropy(transform)`
- **Optimizer:** Adam, lr=1e-4
- **Scheduler:** StepLR (halve lr every 3 epochs)
- **Epochs:** 5

---

## Results

### Baseline vs Joint Model

| Model | Val RF Accuracy | Val Transform Accuracy |
|---|---|---|
| Baseline RF only | 95.6% | — |
| Baseline T only | — | 91.3% |
| **Joint (α=0.5, β=0.5)** | **96.0%** | **92.5%** |

### Per-Transformation Accuracy (Joint Model)

| Transformation | AI Accuracy | Real Accuracy | Total |
|---|---|---|---|
| Original | 95.0% | 96.0% | 95.5% |
| Transfer | 97.6% | 97.2% | 97.4% |
| Redigital | 93.8% | 91.6% | 92.7% |

### Ablation Study

| α | β | Val RF | Val T |
|---|---|---|---|
| 0.8 | 0.2 | 94.7% | 89.2% |
| **0.5** | **0.5** | **96.0%** | **92.5%** |
| 0.2 | 0.8 | 93.8% | 91.4% |

---

## How to Run

### Requirements
```bash
pip install torch torchvision pillow scikit-learn matplotlib seaborn numpy
```

### Steps

1. Open `CV_Project2_2285729.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells top to bottom
4. The dataset downloads automatically in Step 2 (~20 minutes)
5. To skip retraining, run the load cell after Step 5 and upload the `.pth` files from the `/models` folder

### Pretrained Models

Due to file size constraints, pretrained model weights are not included in this repository.
To use pretrained weights, run the notebook top to bottom — training takes approximately
90 minutes on a T4 GPU in Google Colab.

Alternatively, please contact me to send the models 
