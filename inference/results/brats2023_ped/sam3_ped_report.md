# SAM3 Evaluation on BraTS2023_PED (Pediatric)

**Generated**: 2026-01-14T21:32:52.988583

**Dataset**: BraTS2023_PED (Pediatric Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Pediatric Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|----------|
| T1N | 0.0% | 14.2% | 2.3% | 29.1% | 31.2% |
| T1C | 2.2% | 2.2% | 0.0% | 20.3% | 24.6% |
| T2W | 0.0% | 2.0% | 0.0% | 35.9% | 18.7% |
| T2F | 0.0% | 4.0% | 0.0% | 44.8% | 37.8% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NC | T1C | Non-Enhancing Tumor Core | 50.6% | 40.2% |
| ED | T2F | Peritumoral Edema | 78.0% | 65.7% |
| ET | T1C | Contrast-Enhancing Tumor | 46.8% | 35.6% |

## Label Definitions

- **NC**: Non-Enhancing Core (present in 100% of cases)
- **ED**: Edema (present in ~57.6% of cases)
- **ET**: Enhancing Tumor (present in ~88.9% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_PED contains pediatric gliomas
- Different from BraTS2023_MET (adult metastases)
