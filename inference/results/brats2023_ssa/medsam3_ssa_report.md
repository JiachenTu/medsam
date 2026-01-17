# MEDSAM3 Evaluation on BraTS2023_SSA (Sub-Saharan Africa)

**Generated**: 2026-01-14T22:12:57.384616

**Dataset**: BraTS2023_SSA (Sub-Saharan Africa Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 24.8% | 21.3% | 38.5% | 34.6% |
| T1C | 33.6% | 27.7% | 34.1% | 35.0% |
| T2W | 35.7% | 54.4% | 54.5% | 50.7% |
| T2F | 42.9% | 50.1% | 49.2% | 49.4% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NC | T1C | Non-Enhancing Tumor Core | 69.2% | 63.4% |
| ED | T2W | Edema | 36.2% | 26.6% |
| ET | T1C | Enhancing Tumor | 58.5% | 47.6% |

## Label Definitions

- **NC**: Non-Enhancing Tumor Core (~91.7% of cases)
- **ED**: Edema/Oedema (100% of cases)
- **ET**: Enhancing Tumor (100% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_SSA: Sub-Saharan Africa gliomas
- Lower-quality MRI technology compared to standard BraTS
