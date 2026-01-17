# SAM3 Evaluation on BraTS2018 (High-Grade Glioma)

**Generated**: 2026-01-14T23:35:27.446329

**Dataset**: BraTS2018 (High-Grade Glioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 0.0% | 0.0% | 36.6% | 21.0% |
| T1CE | 0.0% | 0.0% | 33.4% | 14.8% |
| T2 | 0.0% | 0.0% | 34.4% | 20.0% |
| FLAIR | 0.0% | 0.0% | 26.6% | 20.8% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NCR | T1CE | Non-Enhancing Tumor Core | 55.6% | 45.9% |
| ED | T2 | Peritumoral Edema | 73.5% | 60.3% |
| ET | T1CE | Contrast-Enhancing Tumor | 83.9% | 73.2% |

## Label Definitions

- **NCR**: Necrotic/Non-Enhancing Tumor Core (label 1, 100% of cases)
- **ED**: Peritumoral Edema (label 2, ~99.6% of cases)
- **ET**: Enhancing Tumor (label 4, ~90.5% of cases) - NOTE: label 4, not 3!

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2018: High-grade glioma brain tumors (HGG only)
- Contrasts: T1, T1CE, T2, FLAIR
- File format: .nii (uncompressed)
- 190 total cases in dataset
