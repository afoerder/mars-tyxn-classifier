# Mars TYXN Junction Classifier

Automated detection and classification of T-, Y-, X-, and N-type fracture junctions in martian thermal contraction polygon (TCP) networks from HiRISE imagery.

**Associated paper:** Foerder, A. B., Magocs, B. L., Thomson, B. J., & Bhidya, H. (2026). Toward Paleoclimate Mapping on Mars by Detecting and Classifying Fracture Junctions with Machine Learning. *Journal of Geophysical Research: Machine Learning and Computation*.

## Overview

Thermal contraction polygons on Mars record the planet's climate history through their fracture network geometry. The relative proportions of T-junctions (hierarchical sequential cracking), Y-junctions (cyclic freeze-thaw via junction twisting), and X-junctions (crack-healing by ice) encode information about the formation and modification history of the fracture network (Silver et al., 2025).

This repository provides a complete two-stage pipeline for automated junction detection and classification from HiRISE orbital imagery:

**Stage 1 (Detection):** A training-free topological method that locates candidate junctions from skeleton images via degree-based proposals, crossing-number filtering, and non-maximum suppression.

**Stage 2 (Classification):** Eight methods are benchmarked, including a stacking meta-classifier (the recommended production model) that combines Random Forest, XGBoost, and a Gaussian-masked CNN.

### Key Results

**Junction Detection (Table 1):**

| Dataset | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Martian (667 junctions) | 0.739 | 0.875 | 0.801 |
| Silver et al. (2025) | 0.957 | 0.849 | 0.900 |

**Junction Classification (Table 2, martian evaluation):**

| Method | T F1 | Y F1 | X F1 | N F1 | Macro F1 |
|--------|------|------|------|------|----------|
| Stacking | 0.62 | 0.91 | 0.74 | 0.81 | 0.77 |
| CNN (Gaussian mask) | 0.56 | 0.90 | 0.78 | 0.82 | 0.77 |
| XGB | 0.51 | 0.85 | 0.59 | 0.61 | 0.64 |
| RF | 0.48 | 0.86 | 0.59 | 0.49 | 0.61 |
| MLP | 0.46 | 0.86 | 0.54 | 0.43 | 0.57 |
| SVM | 0.45 | 0.78 | 0.48 | 0.41 | 0.53 |
| Template* | 0.39 | 0.84 | 0.06 | 0.66 | 0.49 |
| Geometric* | 0.40 | 0.82 | 0.06 | 0.66 | 0.49 |

\* Training-free methods (no labeled data required).

**Regional Demonstration:** Applied to 19 HiRISE images in western Utopia Planitia (40-50 N), the T fraction (fT) decreases with poleward latitude (Spearman rho = -0.77, p = 0.009), consistent with the independently established LCP/HCP gradient of Soare et al. (2021).

---

## Repository Structure

```
mars-tyxn-junction-classifier/
├── README.md
├── LICENSE                            # MIT
├── requirements.txt
│
├── src/                               # All Python source
│   ├── predict_unet.py                # U-Net inference (supports SegFormer mit_b3)
│   ├── tile_hirise_for_pipeline.py    # Tile HiRISE JP2 images for pipeline input
│   ├── extract_inference_patches.py   # Junction detection + patch extraction
│   ├── junction_proposals.py          # Virtual bridge proposal generation
│   ├── junction_geometry.py           # Local skeleton geometry analysis
│   ├── classical_feature_builder.py   # 31-dim geometry feature extraction
│   ├── train_cnn.py                   # CNN training (DeeperCNN_GAP_v2, Gaussian mask)
│   ├── train_rf.py                    # Random Forest training
│   ├── train_xgb.py                   # XGBoost training
│   ├── train_svm.py                   # SVM training
│   ├── train_mlp.py                   # MLP training
│   ├── predict_ensemble.py            # Multi-model ensemble inference
│   ├── calibration_analysis.py        # Monte Carlo calibration analysis
│   ├── run_glyph_benchmark.py         # Geometric classifier (training-free)
│   ├── run_geometric_on_martian.py    # Geometric classifier evaluation
│   ├── template_matcher.py            # Template matching classifier (training-free)
│   ├── template_generator.py          # Template generation utilities
│   ├── hog_transformer.py            # HOG feature transformer (legacy)
│   ├── meta_features.py              # Meta-classifier feature computation
│   ├── evaluate_ground_truth.py      # Evaluation against ground truth
│   ├── unet.py                       # U-Net architecture (v2, smp encoders)
│   └── unet_v1.py                    # U-Net architecture (v1, preserved)
│
├── models/
│   ├── unet/
│   │   ├── mit_b3_skelrecall.pth             # Production U-Net (SegFormer mit_b3, 45M params)
│   │   └── mit_b3_skelrecall_metrics.json    # Training metrics and config
│   └── classifiers/
│       ├── stacking_gauss40_best.pkl         # Stacking meta-classifier (recommended)
│       ├── CNN_ft_gauss40.pt                 # CNN with Gaussian center mask (sigma=40)
│       ├── RF_32.joblib                      # Random Forest (31-dim geometry features)
│       ├── XGB_32_d6.joblib                  # XGBoost (31-dim geometry features)
│       ├── SVM_32.joblib                     # SVM (31-dim geometry features)
│       └── MLP_32.joblib                     # MLP (31-dim geometry features)
│
├── data/
│   ├── training/                      # 4,000 matched-domain patches (192x192)
│   │   ├── manifest.csv
│   │   └── image_patches/
│   ├── pretraining/                   # 24,447 legacy patches (96x96, for CNN pre-training)
│   │   ├── manifest.csv
│   │   └── image_patches/
│   ├── evaluation_martian/            # 667 expert-annotated junctions (192x192)
│   │   ├── manifest.csv
│   │   └── image_patches/
│   ├── evaluation_silver/             # 329 junctions from Silver et al. (2025)
│   │   ├── images/
│   │   └── labels/
│   └── templates/                     # Synthetic junction templates for NCC matching
│
├── results/
│   ├── benchmark/
│   │   └── bootstrap_f1_ci.json       # Bootstrap 95% CIs for all methods
│   └── utopia_planitia/
│       ├── utopia_junction_summary.csv  # Per-image fT, T/Y/X counts, lat/lon
│       ├── image_summary.csv            # HiRISE image metadata
│       └── master_tile_metadata.csv     # Per-tile metadata with coordinates
│
├── figures/                           # Publication figures (Figures 1-10)
│
├── splits/                            # Train/val/test split definitions
│   ├── marscracks_258_holdout.json
│   └── marscracks_258_trainval.json
│
└── notebooks/
    └── MarsCracks_v7_classifier_pipeline.ipynb  # Full v7 training pipeline (Colab)
```

---

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended for U-Net and CNN inference (CPU also works)

### Setup

```bash
conda create -n tyxn python=3.10
conda activate tyxn
pip install -r requirements.txt
```

---

## Usage

All scripts are in `src/`. Run them from the `src/` directory so imports resolve correctly:

```bash
cd src
```

### Quick Start: Reproduce Paper Results (Table 2)

Evaluate the stacking ensemble on the martian evaluation set:

```python
import pickle, csv, cv2, torch, numpy as np, sys
sys.path.insert(0, '.')
from classical_feature_builder import extract_geometry_feature_vector
from train_cnn import DeeperCNN_GAP_v2
from scipy.ndimage import binary_dilation
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load stacking bundle
with open('../models/classifiers/stacking_gauss40_best.pkl', 'rb') as f:
    bundle = pickle.load(f)

rf_model = bundle['rf_model']
xgb_model = bundle['xgb_model']
meta_clf = bundle['meta_classifier']
SIGMA = bundle['gaussian_sigma']  # 40
PATCH_SIZE = bundle['patch_size']  # 192
IDX_TO_LABEL = bundle['idx_to_label']  # ['N', 'T', 'X', 'Y']

# Reconstruct CNN
cnn = DeeperCNN_GAP_v2(num_classes=4, in_channels=2, dropout=0.3)
cnn.load_state_dict(bundle['cnn_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn.to(device).eval()

# Gaussian mask
h = w = PATCH_SIZE
y_grid, x_grid = np.mgrid[0:h, 0:w]
gauss = np.exp(-((x_grid - h//2)**2 + (y_grid - w//2)**2) / (2 * SIGMA**2)).astype(np.float32)

# Load evaluation data
EVAL = '../data/evaluation_martian'
rows = list(csv.DictReader(open(f'{EVAL}/manifest.csv')))
patches, geom_features, true_labels = [], [], []
for r in rows:
    p = cv2.imread(f'{EVAL}/{r["relpath"]}', cv2.IMREAD_GRAYSCALE)
    patches.append(p)
    geom_features.append(extract_geometry_feature_vector(
        (p > 0).astype(np.float32), trace_len=40))
    true_labels.append(r['label'])

X_geom = np.array(geom_features, dtype=np.float32)

# Base model probabilities
rf_proba = rf_model.predict_proba(X_geom)
xgb_proba = xgb_model.predict_proba(X_geom)

cnn_proba = []
with torch.no_grad():
    for p in patches:
        skel = p.astype(np.float32) / 255.0
        mask = binary_dilation(skel > 0.5, structure=np.ones((3,3), dtype=bool)).astype(np.float32)
        inp = torch.tensor(np.stack([skel * gauss, mask * gauss]),
                           dtype=torch.float32).unsqueeze(0).to(device)
        cnn_proba.append(torch.softmax(cnn(inp), dim=1).cpu().numpy()[0])
cnn_proba = np.array(cnn_proba)

# Stacking meta-classifier
X_meta = np.hstack([rf_proba, xgb_proba, cnn_proba, X_geom])
preds = [IDX_TO_LABEL[i] for i in meta_clf.predict(X_meta)]

print(classification_report(true_labels, preds, labels=['T','Y','X','N'], digits=3))
```

### Full Pipeline: Process New HiRISE Imagery

#### Step 1: Tile HiRISE Images

```bash
python tile_hirise_for_pipeline.py \
    --input /path/to/ESP_XXXXXX_XXXX_RED.JP2 \
    --output-dir /path/to/output \
    --tile-size 768
```

#### Step 2: U-Net Segmentation

```bash
python predict_unet.py \
    --model-path ../models/unet/mit_b3_skelrecall.pth \
    --input /path/to/tiles \
    --output-dir /path/to/skeletons
```

The model config is auto-detected from the companion `_metrics.json` file.

#### Step 3: Junction Detection and Classification

Detection and classification are performed together. See `notebooks/MarsCracks_v7_classifier_pipeline.ipynb` for the complete inference pipeline used in the paper, including:
- Spur pruning (12-pixel iterative endpoint erosion)
- Junction detection via crossing-number filtering and NMS
- Patch extraction (192x192 around each junction)
- Stacking ensemble classification with Gaussian center mask

### Training New Models

#### CNN (with Gaussian mask pre-training + fine-tuning)

```bash
# Pre-train on 24,447 legacy patches
python train_cnn.py \
    --manifest ../data/pretraining/manifest.csv \
    --data-root ../data/pretraining \
    --output-model ../models/classifiers/CNN_pretrained.pt \
    --arch deeper_v2 --in-channels 2 --patch-size 192 \
    --gaussian-sigma 40 --epochs 50 --batch-size 32

# Fine-tune on 4,000 matched-domain patches
python train_cnn.py \
    --manifest ../data/training/manifest.csv \
    --data-root ../data/training \
    --output-model ../models/classifiers/CNN_finetuned.pt \
    --init-weights ../models/classifiers/CNN_pretrained.pt \
    --arch deeper_v2 --in-channels 2 --patch-size 192 \
    --gaussian-sigma 40 --epochs 80 --batch-size 32
```

#### Classical Models (31-dim geometry features)

```bash
# Random Forest
python train_rf.py \
    --manifest ../data/training/manifest.csv \
    --data-root ../data/training \
    --output-model ../models/classifiers/RF.joblib \
    --patch-size 192 --feature-regime geom_only \
    --class-weight-mode balanced --geometry-trace-len 40

# XGBoost
python train_xgb.py \
    --manifest ../data/training/manifest.csv \
    --data-root ../data/training \
    --output-model ../models/classifiers/XGB.joblib \
    --patch-size 192 --feature-regime geom_only \
    --class-weight-mode balanced --geometry-trace-len 40

# SVM
python train_svm.py \
    --manifest ../data/training/manifest.csv \
    --data-root ../data/training \
    --output-model ../models/classifiers/SVM.joblib \
    --patch-size 192 --feature-regime geom_only \
    --class-weight-mode balanced --geometry-trace-len 40

# MLP
python train_mlp.py \
    --manifest ../data/training/manifest.csv \
    --data-root ../data/training \
    --output-model ../models/classifiers/MLP.joblib \
    --patch-size 192 --feature-regime geom_only \
    --class-weight-mode balanced --geometry-trace-len 40
```

---

## Data

### Training Set (`data/training/`)

4,000 labeled junction patches (192x192 pixels) extracted from skeletons produced by the production mit_b3 U-Net. Labels were assigned by manual annotation on matched-domain skeletons.

| Class | Count | Percentage |
|-------|------:|------------|
| Y | 2,587 | 64.7% |
| T | 665 | 16.6% |
| N | 528 | 13.2% |
| X | 220 | 5.5% |

Data are split by source image (train/val/test: 3,236/344/420) so that no HiRISE image contributes patches to more than one split.

### Pre-training Set (`data/pretraining/`)

24,447 legacy patches (96x96 pixels) from an earlier U-Net architecture, used for CNN pre-training via transfer learning before fine-tuning on the matched-domain set.

### Evaluation Sets

**Martian evaluation** (`data/evaluation_martian/`): 667 expert-annotated junctions (451 Y, 66 T, 35 X, 115 N) from four 768x768 skeleton tiles from two independent HiRISE scenes.

**Silver evaluation** (`data/evaluation_silver/`): 329 junctions (191 T, 119 Y, 19 X) from a hand-traced fracture network by Silver et al. (2025). Not produced by the U-Net pipeline; provides a cross-domain generalization test.

---

## Key Design Choices

**Gaussian center mask (sigma = 40 pixels):** Applied to the 192x192 CNN input during both training and inference. In dense TCP networks, neighboring junctions fall within the CNN's receptive field, contaminating the local signal. The Gaussian mask attenuates peripheral structure and focuses attention on the target junction. This single change improved CNN macro F1 from 0.54 to 0.77.

**Matched-domain training:** Classifiers are trained on junctions extracted from the same mit_b3 U-Net skeletons used in deployment. Early experiments showed that models trained on one skeleton style transferred poorly to another.

**Stacking meta-classifier:** Combines RF (4-class probabilities), XGB (4-class probabilities), and Gaussian-masked CNN (4-class probabilities) with the 31-dimensional geometry feature vector into a 43-dimensional meta-input for a second-stage XGBoost model. The base models are trained on 3,236 junctions; the meta-classifier is trained on a separate 420-junction set using the base models' out-of-sample predictions.

**31-dimensional geometry features:** Computed on the binarized skeleton with trace_len = 40. Include collinearity deviation at seven radii (r = 12 to 80 pixels), branch count, angular gap statistics, branch length ratios, template-likeness scores, and derived consistency measures.

---

## Utopia Planitia Regional Demonstration

The pipeline was applied to 19 HiRISE images spanning 40-50 N, 82-95 E in western Utopia Planitia. Results are in `results/utopia_planitia/`.

- `utopia_junction_summary.csv`: Per-image fT, T/Y/X/N counts, latitude, longitude
- `image_summary.csv`: HiRISE image metadata (dimensions, tile counts, geolocation)
- `master_tile_metadata.csv`: Per-tile metadata with center coordinates

The T fraction (fT = T / (T + Y + X)) decreases monotonically with poleward latitude (Spearman rho = -0.77, p = 0.009), consistent with the latitude-dependent ice-stability gradient reported by Soare et al. (2021).

---

## Citation

If you use this code or data, please cite:

```
Foerder, A. B., Magocs, B. L., Thomson, B. J., & Bhidya, H. (2026).
Toward Paleoclimate Mapping on Mars by Detecting and Classifying Fracture
Junctions with Machine Learning. Journal of Geophysical Research: Machine
Learning and Computation.
```

## License

MIT License. See [LICENSE](LICENSE).
