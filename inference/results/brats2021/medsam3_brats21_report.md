# MEDSAM3 Evaluation on BraTS2021 (Adult Glioma)

**Generated**: 2026-01-14T23:01:44.035952

**Dataset**: BraTS2021 (Adult Glioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 39.8% | 35.4% | 38.6% | 42.1% |
| T1CE | 38.6% | 33.8% | 38.0% | 47.5% |
| T2 | 27.3% | 53.5% | 48.0% | 45.9% |
| FLAIR | 40.5% | 52.5% | 55.5% | 54.2% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NCR | T1CE | Non-Enhancing Tumor Core | 50.4% | 43.7% |
| ED | T2 | Peritumoral Edema | 56.0% | 41.2% |
| ET | T1CE | Contrast-Enhancing Tumor | 61.6% | 51.2% |

## Label Definitions

- **NCR**: Necrotic/Non-Enhancing Tumor Core (label 1, ~97.4% of cases)
- **ED**: Peritumoral Edema (label 2, ~96.6% of cases)
- **ET**: Enhancing Tumor (label 4, ~99.9% of cases) - NOTE: label 4, not 3!

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2021: Adult glioma brain tumors
- Contrasts: T1, T1CE, T2, FLAIR (different from BraTS2023)
- 1,251 total cases in dataset
