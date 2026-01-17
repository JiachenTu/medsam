# MEDSAM3 Evaluation on BraTS2023_MEN (Meningioma)

**Generated**: 2026-01-14T22:36:26.405671

**Dataset**: BraTS2023_MEN (Meningioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Meningioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 40.0% | 26.6% | 39.9% | 36.8% |
| T1C | 41.8% | 34.2% | 42.9% | 42.7% |
| T2W | 49.9% | 35.7% | 38.9% | 51.1% |
| T2F | 32.1% | 34.8% | 25.1% | 46.4% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NETC | T1N | Tumor Core | 91.0% | 83.4% |
| SNFH | T2F | Edema | 48.1% | 39.6% |
| ET | T1C | Enhancing Tumor | 94.5% | 89.8% |

## Label Definitions

- **NETC**: Non-Enhancing Tumor Core (~33.9% of cases)
- **SNFH**: Surrounding Non-Enhancing FLAIR Hyperintense (~53.6% of cases)
- **ET**: Enhancing Tumor (~99.9% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_MEN: Meningioma brain tumors
- Large dataset (1,139 cases)
