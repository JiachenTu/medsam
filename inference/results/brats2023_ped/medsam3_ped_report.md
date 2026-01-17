# MEDSAM3 Evaluation on BraTS2023_PED (Pediatric)

**Generated**: 2026-01-14T21:43:55.168701

**Dataset**: BraTS2023_PED (Pediatric Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Pediatric Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|----------|
| T1N | 42.9% | 51.8% | 43.7% | 53.8% | 60.3% |
| T1C | 44.3% | 62.8% | 40.2% | 50.2% | 61.2% |
| T2W | 44.7% | 64.3% | 67.5% | 61.6% | 63.6% |
| T2F | 53.4% | 67.5% | 65.5% | 67.9% | 75.2% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NC | T2W | Tumor Core | 68.0% | 60.1% |
| ED | T2F | Edema | 71.0% | 57.7% |
| ET | T1C | Contrast-Enhancing Tumor | 35.0% | 23.3% |

## Label Definitions

- **NC**: Non-Enhancing Core (present in 100% of cases)
- **ED**: Edema (present in ~57.6% of cases)
- **ET**: Enhancing Tumor (present in ~88.9% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_PED contains pediatric gliomas
- Different from BraTS2023_MET (adult metastases)
