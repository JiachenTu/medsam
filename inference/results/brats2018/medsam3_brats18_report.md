# MEDSAM3 Evaluation on BraTS2018 (High-Grade Glioma)

**Generated**: 2026-01-14T23:38:14.106936

**Dataset**: BraTS2018 (High-Grade Glioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 35.2% | 27.5% | 44.2% | 42.2% |
| T1CE | 31.4% | 23.1% | 35.5% | 34.6% |
| T2 | 27.3% | 44.5% | 41.3% | 41.7% |
| FLAIR | 45.9% | 50.2% | 52.2% | 53.3% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NCR | T1CE | Non-Enhancing Tumor Core | 41.3% | 32.4% |
| ED | FLAIR | Edema | 48.3% | 39.1% |
| ET | T1CE | Enhancing Tumor | 59.2% | 47.0% |

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
