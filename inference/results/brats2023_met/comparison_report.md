# SAM3 vs MedSAM3 Comparison on BraTS2023_MET

**Generated**: 2026-01-14

**Dataset**: BraTS2023_MET (Brain Tumor Segmentation - Metastases)

**Cases Evaluated**: 48 (~20% sample)

**Evaluation**: Middle slice of each 3D MRI volume

---

## Executive Summary

MedSAM3 (fine-tuned) significantly outperforms SAM3 (base) on **text-prompted** brain tumor segmentation, with improvements of up to **29.6 percentage points** on the best text prompts. For **box prompts**, SAM3 slightly edges out MedSAM3 (63.3% vs 60.5% average Dice), suggesting box prompts are less dependent on domain-specific training.

### Key Findings

| Metric | SAM3 | MedSAM3 | Winner |
|--------|------|---------|--------|
| Best Box Dice | 70.6% (T2F) | 67.6% (T2F) | SAM3 |
| Avg Box Dice | 63.3% | 60.5% | SAM3 |
| Best Text Dice | 21.0% (ET, T2F) | 38.0% (ET, T2F) | **MedSAM3** |
| "Brain Metastasis" Dice | 1.5% (T2F) | 31.1% (T2F) | **MedSAM3** |
| "Brain Tumor" Dice | 0.0% | 18.2% (T2F) | **MedSAM3** |

---

## Results by MRI Contrast

### Box Prompt Performance (Dice %)

| Contrast | SAM3 | MedSAM3 | Difference |
|----------|------|---------|------------|
| T1N | 59.8% | 60.1% | +0.3% |
| T1C | 58.2% | 55.2% | -3.0% |
| T2W | 64.7% | 59.2% | -5.5% |
| T2F | **70.6%** | **67.6%** | -3.0% |
| **Average** | **63.3%** | **60.5%** | -2.8% |

**Observation**: SAM3 performs slightly better on box prompts. Both models show T2F (FLAIR) as the best contrast for whole tumor segmentation.

### Text Prompt Performance (Dice %)

#### "Brain Tumor" Prompt

| Contrast | SAM3 | MedSAM3 | Improvement |
|----------|------|---------|-------------|
| T1N | 0.0% | 7.8% | **+7.8%** |
| T1C | 0.0% | 11.2% | **+11.2%** |
| T2W | 0.0% | 7.2% | **+7.2%** |
| T2F | 0.0% | **18.2%** | **+18.2%** |

#### "Brain Metastasis" Prompt

| Contrast | SAM3 | MedSAM3 | Improvement |
|----------|------|---------|-------------|
| T1N | 0.6% | 13.1% | **+12.5%** |
| T1C | 0.0% | 18.1% | **+18.1%** |
| T2W | 0.0% | 17.3% | **+17.3%** |
| T2F | 1.5% | **31.1%** | **+29.6%** |

#### "Enhancing Tumor" Prompt

| Contrast | SAM3 | MedSAM3 | Improvement |
|----------|------|---------|-------------|
| T1N | 20.2% | 22.5% | +2.3% |
| T1C | 20.3% | 29.2% | **+8.9%** |
| T2W | 21.0% | 32.8% | **+11.8%** |
| T2F | 21.0% | **38.0%** | **+17.0%** |

#### "Tumor" Prompt

| Contrast | SAM3 | MedSAM3 | Improvement |
|----------|------|---------|-------------|
| T1N | 6.9% | 18.2% | **+11.3%** |
| T1C | 2.4% | 23.8% | **+21.4%** |
| T2W | 5.3% | 25.8% | **+20.5%** |
| T2F | 11.1% | **33.4%** | **+22.3%** |

---

## Results by Tumor Label

### NETC (Non-Enhancing Tumor Core)

| Model | Best Contrast | Box Dice | Best Text Prompt | Text Dice |
|-------|---------------|----------|------------------|-----------|
| SAM3 | T1C | 61.4% | Enhancing Tumor | 20.7% |
| MedSAM3 | T1C | 58.6% | Enhancing Tumor | 25.3% |

**Text improvement: +4.6%**

### SNFH (Edema / Surrounding Non-enhancing FLAIR Hyperintensity)

| Model | Best Contrast | Box Dice | Best Text Prompt | Text Dice |
|-------|---------------|----------|------------------|-----------|
| SAM3 | T2F | 67.5% | Enhancing Tumor | 14.0% |
| MedSAM3 | T2F | 60.8% | Enhancing Tumor | 27.1% |

**Text improvement: +13.1%**

### ET (Enhancing Tumor)

| Model | Best Contrast | Box Dice | Best Text Prompt | Text Dice |
|-------|---------------|----------|------------------|-----------|
| SAM3 | T1C | 57.7% | Enhancing Tumor | 40.6% |
| MedSAM3 | T1C | 55.3% | Enhancing Tumor | 31.4% |

**Text change: -9.2%** (SAM3 better for ET with text prompt)

---

## Recommendations

### Best Configurations

| Scenario | Model | Contrast | Prompt | Expected Dice |
|----------|-------|----------|--------|---------------|
| Box-prompted whole tumor | SAM3 | T2F | Bounding box | ~70% |
| Text-prompted whole tumor | MedSAM3 | T2F | "Enhancing Tumor" | ~38% |
| Text-prompted metastasis | MedSAM3 | T2F | "Brain Metastasis" | ~31% |
| Enhancing tumor (label 3) | SAM3/MedSAM3 | T1C | Box prompt | ~55-58% |
| Edema segmentation | SAM3 | T2F | Box prompt | ~67% |

### Clinical Implications

1. **For automated pipelines**: Use SAM3 with box prompts derived from detection models
2. **For zero-shot text prompts**: Use MedSAM3 with "Enhancing Tumor" prompt on T2F images
3. **For contrast selection**:
   - T2F (FLAIR): Best for whole tumor and edema
   - T1C (contrast-enhanced): Best for enhancing tumor components

---

## Visualizations

Sample visualizations are available in:
```
/home/jtu9/SAM/medsam/inference/visualization/BraTS2023_MET/
```

Each case contains:
- Individual contrast directories (t1n, t1c, t2w, t2f)
- `all_contrasts_comparison.png` - 4x4 grid of all contrasts
- Per-contrast comparison images for each text prompt

---

## Methodology

- **Sampling**: Random 20% of 238 cases (n=48)
- **Slice selection**: Middle axial slice of each 3D volume
- **Ground truth**: Combined tumor mask (labels 1-3)
- **Box prompts**: Derived from ground truth bounding boxes
- **Text prompts**: Zero-shot evaluation (no fine-tuning on prompts)
- **Metrics**: Dice coefficient and IoU

---

## Files

| File | Description |
|------|-------------|
| `sam3_brats_summary.csv` | SAM3 per-contrast summary |
| `sam3_brats_detailed.json` | SAM3 detailed results |
| `medsam3_brats_summary.csv` | MedSAM3 per-contrast summary |
| `medsam3_brats_detailed.json` | MedSAM3 detailed results |
| `*_brats_by_label.csv` | Per-label breakdown |

---

*Report generated by automated evaluation pipeline*
