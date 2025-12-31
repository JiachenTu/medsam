# SAM3 vs MedSAM3 Medical Image Segmentation Comparison

**Generated**: 2025-12-31

## Overview

This report compares the performance of:
- **Base SAM3**: Zero-shot model from HuggingFace (Segment Anything 3)
- **MedSAM3**: Fine-tuned model from `Potestates/medsam3_stage1_text_full` (checkpoint.pt)

Both models were evaluated on 5 medical imaging datasets using:
- **Box Prompts**: Bounding boxes derived from ground truth masks
- **Text Prompts**: Natural language descriptions (zero-shot text-based segmentation)

---

## Summary Results

### Dice Score Comparison (Higher is Better)

| Dataset | Modality | Base SAM3 Box | Base SAM3 Text | MedSAM3 Box | MedSAM3 Text | Text Improvement |
|---------|----------|---------------|----------------|-------------|--------------|------------------|
| CHASE_DB1 | Fundus | 8.2% | 17.9% | 32.7% | **62.6%** | +44.7% |
| STARE | Fundus | 19.5% | 18.6% | 58.9% | **54.4%** | +35.8% |
| CVC-ClinicDB | Endoscopy | 93.6% | 0.0% | 91.5% | **87.9%** | +87.9% |
| ETIS-Larib | Endoscopy | 94.4% | 0.0% | 91.9% | **86.1%** | +86.1% |
| PH2 | Dermoscopy | 93.9% | 18.4% | 93.6% | **92.7%** | +74.3% |

### Overall Averages

| Metric | Base SAM3 | MedSAM3 | Change |
|--------|-----------|---------|--------|
| Box Prompt Dice | 61.9% | 73.6% | **+11.7%** |
| Text Prompt Dice | 11.0% | 76.7% | **+65.7%** |
| Box Prompt IoU | 56.7% | 64.8% | **+8.1%** |
| Text Prompt IoU | 7.0% | 66.3% | **+59.3%** |

---

## Detailed Analysis

### 1. Text Prompt Performance (Major Improvement)

The most significant improvement is in **text-prompted segmentation**:

| Dataset | Text Prompt Used | Base SAM3 | MedSAM3 | Absolute Gain |
|---------|------------------|-----------|---------|---------------|
| CHASE_DB1 | "Retinal Blood Vessel" | 17.9% | 62.6% | +44.7% |
| STARE | "Retinal Blood Vessel" | 18.6% | 54.4% | +35.8% |
| CVC-ClinicDB | "Polyp" | 0.0% | 87.9% | +87.9% |
| ETIS-Larib | "Polyp" | 0.0% | 86.1% | +86.1% |
| PH2 | "Skin Lesion" | 18.4% | 92.7% | +74.3% |

**Key Finding**: MedSAM3's fine-tuning on medical text enables it to understand medical terminology that base SAM3 completely fails on (e.g., "Polyp" yielded 0% Dice on base SAM3).

### 2. Box Prompt Performance

| Dataset | Base SAM3 | MedSAM3 | Change |
|---------|-----------|---------|--------|
| CHASE_DB1 | 8.2% | 32.7% | +24.5% |
| STARE | 19.5% | 58.9% | +39.4% |
| CVC-ClinicDB | 93.6% | 91.5% | -2.1% |
| ETIS-Larib | 94.4% | 91.9% | -2.5% |
| PH2 | 93.9% | 93.6% | -0.3% |

**Key Finding**: Box prompt performance improved significantly for retinal vessel datasets (challenging fine structures) but slightly decreased for polyp/lesion datasets (simpler blob-like structures). This suggests the fine-tuning prioritized text understanding over box-guided segmentation.

### 3. Dataset-Specific Insights

#### Retinal Vessel Datasets (CHASE_DB1, STARE)
- Both box and text prompts improved substantially
- Fine vessels are challenging for general-purpose models
- MedSAM3 shows better understanding of vessel anatomy

#### Polyp Datasets (CVC-ClinicDB, ETIS-Larib)
- Text prompts: **Critical improvement** (0% → 86-88%)
- Box prompts: Slight decrease (~2.5%)
- Base SAM3 completely failed to understand "Polyp" as a text prompt

#### Dermoscopy Dataset (PH2)
- Both prompts perform excellently (>92%)
- Text prompt improved from 18.4% to 92.7%
- Shows strong generalization to skin lesion terminology

---

## SSIM (Structural Similarity) Comparison

| Dataset | Base SAM3 Text | MedSAM3 Text | Improvement |
|---------|----------------|--------------|-------------|
| CHASE_DB1 | 0.037 | 0.548 | +0.511 |
| STARE | 0.031 | 0.422 | +0.391 |
| CVC-ClinicDB | 0.000 | 0.851 | +0.851 |
| ETIS-Larib | 0.000 | 0.850 | +0.850 |
| PH2 | -0.164 | 0.874 | +1.038 |

---

## Conclusions

1. **MedSAM3 dramatically improves text-prompted segmentation** for medical images, with an average improvement of **+65.7%** Dice score.

2. **Medical terminology understanding is key**: Base SAM3 completely fails on domain-specific terms like "Polyp", while MedSAM3 achieves 86-88% Dice.

3. **Fine-tuning enables zero-shot medical image segmentation**: Users can describe what they want segmented in natural language without providing bounding boxes.

4. **Box prompt performance trade-off**: A slight decrease (-2%) in polyp/lesion datasets suggests the model optimized more for text understanding, though performance remains excellent (>91%).

5. **Best use case**: MedSAM3 is ideal for scenarios where:
   - Users want text-based segmentation without manual annotation
   - Medical terminology needs to be understood (polyp, lesion, vessel)
   - Zero-shot generalization across medical imaging modalities is needed

---

## Technical Details

- **Base SAM3**: HuggingFace default weights
- **MedSAM3**: `Potestates/medsam3_stage1_text_full/checkpoint.pt`
- **Confidence Threshold**: 0.1
- **Inference**: bfloat16 precision on CUDA
- **Evaluation Metrics**: Dice, IoU, PSNR, SSIM, Precision, Recall

---

## Files

- Base SAM3 results: `results/sam3_evaluation_detailed.json`
- MedSAM3 results: `results/medsam3_ckpt0_evaluation_detailed.json`
- Checkpoints: `/srv/local/shared/temp/tmp1/jtu9/medsam3_weights/`
