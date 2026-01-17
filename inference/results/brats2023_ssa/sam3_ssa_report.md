# SAM3 Evaluation on BraTS2023_SSA (Sub-Saharan Africa)

**Generated**: 2026-01-14T22:10:00.915043

**Dataset**: BraTS2023_SSA (Sub-Saharan Africa Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 0.0% | 0.0% | 24.0% | 19.1% |
| T1C | 0.0% | 0.0% | 26.5% | 25.6% |
| T2W | 0.0% | 0.0% | 32.5% | 26.8% |
| T2F | 0.0% | 4.5% | 33.8% | 35.7% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NC | T1C | Non-Enhancing Tumor Core | 82.3% | 74.0% |
| ED | T2F | Edema | 69.0% | 56.0% |
| ET | T1C | Enhancing Core | 66.7% | 56.6% |

## Label Definitions

- **NC**: Non-Enhancing Tumor Core (~91.7% of cases)
- **ED**: Edema/Oedema (100% of cases)
- **ET**: Enhancing Tumor (100% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_SSA: Sub-Saharan Africa gliomas
- Lower-quality MRI technology compared to standard BraTS
