# SAM3 Evaluation on BraTS2021 (Adult Glioma)

**Generated**: 2026-01-14T22:58:52.680306

**Dataset**: BraTS2021 (Adult Glioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 0.0% | 0.0% | 41.6% | 16.5% |
| T1CE | 0.0% | 0.0% | 31.8% | 16.7% |
| T2 | 0.0% | 0.0% | 35.0% | 11.9% |
| FLAIR | 0.0% | 0.0% | 36.3% | 24.2% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NCR | T1CE | Non-Enhancing Tumor Core | 51.2% | 44.9% |
| ED | FLAIR | Brain Edema | 76.4% | 63.2% |
| ET | T1CE | Enhancing Tumor | 67.5% | 58.9% |

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
