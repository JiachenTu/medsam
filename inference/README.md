# MedSAM3 Inference and Evaluation

This directory contains scripts for evaluating SAM3 and MedSAM3 models on medical image segmentation datasets.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Evaluation Results](#evaluation-results)
- [Usage](#usage)
- [File Structure](#file-structure)

---

## Overview

This evaluation compares two models:

| Model | Description | Source |
|-------|-------------|--------|
| **Base SAM3** | Zero-shot Segment Anything 3 | HuggingFace (default) |
| **MedSAM3** | Fine-tuned on medical images with text supervision | [Potestates/medsam3_stage1_text_full](https://huggingface.co/Potestates/medsam3_stage1_text_full) |

Both models are evaluated using:
- **Box Prompts**: Bounding boxes derived from ground truth masks
- **Text Prompts**: Natural language descriptions (zero-shot)

---

## Environment Setup

```bash
# Activate the conda environment
conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3

# Required packages: torch, numpy, pandas, pillow, tqdm, scikit-image
```

---

## Datasets

Five medical imaging datasets are used for evaluation:

| Dataset | Modality | Target | Samples | Text Prompt |
|---------|----------|--------|---------|-------------|
| CHASE_DB1 | Fundus | Retinal Vessels | 28 | "Retinal Blood Vessel" |
| STARE | Fundus | Retinal Vessels | 20 | "Retinal Blood Vessel" |
| CVC-ClinicDB | Endoscopy | Polyps | 612 | "Polyp" |
| ETIS-Larib | Endoscopy | Polyps | 196 | "Polyp" |
| PH2 | Dermoscopy | Skin Lesions | 200 | "Skin Lesion" |

**Data Location**: `/srv/local/shared/medsam_data/`

---

## Evaluation Results

### Box Prompt Performance (Dice Score)

| Dataset | Base SAM3 | MedSAM3 | Change |
|---------|-----------|---------|--------|
| CHASE_DB1 | 8.2% | 32.7% | **+24.5%** |
| STARE | 19.5% | 58.9% | **+39.4%** |
| CVC-ClinicDB | 93.6% | 91.5% | -2.1% |
| ETIS-Larib | 94.4% | 91.9% | -2.5% |
| PH2 | 93.9% | 93.6% | -0.3% |
| **Average** | **61.9%** | **73.6%** | **+11.7%** |

### Text Prompt Performance (Dice Score)

| Dataset | Base SAM3 | MedSAM3 | Change |
|---------|-----------|---------|--------|
| CHASE_DB1 | 17.9% | 62.6% | **+44.7%** |
| STARE | 18.6% | 54.4% | **+35.8%** |
| CVC-ClinicDB | 0.0% | 87.9% | **+87.9%** |
| ETIS-Larib | 0.0% | 86.1% | **+86.1%** |
| PH2 | 18.4% | 92.7% | **+74.3%** |
| **Average** | **11.0%** | **76.7%** | **+65.7%** |

### Key Findings

1. **Text Prompt Improvement**: MedSAM3 improves text-prompted segmentation by **+65.7%** average Dice score.

2. **Medical Terminology**: Base SAM3 completely fails on medical terms like "Polyp" (0% Dice), while MedSAM3 achieves 86-88%.

3. **Box Prompt Trade-off**: Minor decrease (-2%) on polyp/lesion datasets, but significant improvement (+24-39%) on challenging retinal vessel datasets.

4. **Zero-Shot Capability**: MedSAM3 enables true zero-shot medical image segmentation using natural language descriptions.

---

## Usage

### Evaluate Base SAM3

```bash
conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3

python run_evaluation.py
```

### Evaluate MedSAM3 with Custom Checkpoint

```bash
conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3

python run_medsam3_evaluation.py \
    --checkpoint /srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt \
    --model-name medsam3_ckpt0
```

### Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to MedSAM3 checkpoint file | Required |
| `--model-name` | Name for output files | `medsam3` |
| `--max-samples` | Limit samples per dataset (for testing) | All |
| `--datasets` | Comma-separated list of datasets | All 5 |

### Example: Evaluate Specific Datasets

```bash
python run_medsam3_evaluation.py \
    --checkpoint /srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt \
    --model-name medsam3_polyp_only \
    --datasets "CVC-ClinicDB,ETIS-Larib"
```

### Example: Quick Test with Limited Samples

```bash
python run_medsam3_evaluation.py \
    --checkpoint /srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt \
    --model-name medsam3_test \
    --max-samples 5
```

---

## File Structure

```
inference/
├── README.md                    # This documentation
├── dataset_loaders.py           # Dataset loading utilities
├── sam3_inference.py            # SAM3/MedSAM3 inference wrapper
├── metrics.py                   # Dice, IoU, PSNR, SSIM metrics
├── run_evaluation.py            # Base SAM3 evaluation script
├── run_medsam3_evaluation.py    # MedSAM3 evaluation script
├── sam3_visualization.ipynb     # Visualization notebook
└── results/
    ├── sam3_evaluation_detailed.json       # Base SAM3 results
    ├── sam3_evaluation_report.md           # Base SAM3 report
    ├── medsam3_ckpt0_evaluation_detailed.json  # MedSAM3 results
    ├── medsam3_ckpt0_evaluation_report.md      # MedSAM3 report
    └── medsam3_comparison_report.md        # Comparison report
```

### MedSAM3 Checkpoints

Located at: `/srv/local/shared/temp/tmp1/jtu9/medsam3_weights/`

| Checkpoint | Size | Description |
|------------|------|-------------|
| checkpoint.pt | ~9.4 GB | Main checkpoint (evaluated) |
| checkpoint_1.pt | ~9.4 GB | Training checkpoint 1 |
| checkpoint_2.pt | ~9.4 GB | Training checkpoint 2 |
| checkpoint_3.pt | ~9.4 GB | Training checkpoint 3 |

---

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Dice** | F1 score, harmonic mean of precision and recall | 0-1 (higher is better) |
| **IoU** | Intersection over Union (Jaccard index) | 0-1 (higher is better) |
| **PSNR** | Peak Signal-to-Noise Ratio | dB (higher is better) |
| **SSIM** | Structural Similarity Index | -1 to 1 (higher is better) |
| **Precision** | True positives / Predicted positives | 0-1 (higher is better) |
| **Recall** | True positives / Actual positives | 0-1 (higher is better) |

---

## Technical Details

- **Inference Precision**: bfloat16
- **Confidence Threshold**: 0.1
- **Device**: CUDA (GPU)
- **SAM3 Root**: `/home/jtu9/SAM/sam3`

---

## References

- [SAM3 Repository](https://github.com/anthropics/sam3)
- [MedSAM3 HuggingFace](https://huggingface.co/Potestates/medsam3_stage1_text_full)
- [Original SAM Paper](https://arxiv.org/abs/2304.02643)
