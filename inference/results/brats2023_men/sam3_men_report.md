# SAM3 Evaluation on BraTS2023_MEN (Meningioma)

**Generated**: 2026-01-14T22:34:44.825083

**Dataset**: BraTS2023_MEN (Meningioma Brain Tumors)

**Cases evaluated**: 20

**Evaluation type**: Middle slice, text prompts only

## Combined Tumor Mask - Text Prompt Performance

| Contrast | Brain Tumor | Meningioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 0.0% | 6.4% | 21.8% | 40.6% |
| T1C | 0.0% | 0.0% | 43.5% | 2.1% |
| T2W | 0.0% | 0.0% | 37.7% | 38.8% |
| T2F | 0.0% | 0.0% | 34.1% | 39.5% |

## Per-Label Best Performance

| Label | Contrast | Best Text Prompt | Dice | IoU |
|-------|----------|------------------|------|-----|
| NETC | T1N | Necrotic Core | 95.6% | 91.6% |
| SNFH | T2W | Peritumoral Edema | 82.2% | 71.8% |
| ET | T1C | Enhancing Core | 95.8% | 92.0% |

## Label Definitions

- **NETC**: Non-Enhancing Tumor Core (~33.9% of cases)
- **SNFH**: Surrounding Non-Enhancing FLAIR Hyperintense (~53.6% of cases)
- **ET**: Enhancing Tumor (~99.9% of cases)

## Notes

- Evaluated on middle slices of sampled cases
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_MEN: Meningioma brain tumors
- Large dataset (1,139 cases)
