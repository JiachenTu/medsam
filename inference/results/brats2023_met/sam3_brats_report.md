# sam3 Evaluation on BraTS2023_MET

**Generated**: 2026-01-14T14:33:52.230249

**Cases evaluated**: 48

## Results by MRI Contrast (Combined Tumor Mask)

| Contrast | Box Dice | Box IoU | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |
|----------|----------|---------|-------------|------------------|-----------------|-------|
| T1N | 59.8% | 47.0% | 0.0% | 0.6% | 20.2% | 6.9% |
| T1C | 58.2% | 45.0% | 0.0% | 0.0% | 20.3% | 2.4% |
| T2W | 64.7% | 51.8% | 0.0% | 0.0% | 21.0% | 5.3% |
| T2F | 70.6% | 58.8% | 0.0% | 1.5% | 21.0% | 11.1% |

**Average Box Prompt Dice**: 63.3%

## Results by Tumor Label

| Label | Best Contrast | Box Dice | Best Text Prompt | Text Dice |
|-------|---------------|----------|------------------|----------|
| NETC | T1C | 61.4% | Enhancing Tumor | 20.7% |
| SNFH | T2F | 67.5% | Enhancing Tumor | 14.0% |
| ET | T1C | 57.7% | Enhancing Tumor | 40.6% |

## Notes

- Evaluated on middle slices of sampled cases (~20%)
- Box prompts derived from ground truth bounding boxes
- Text prompts evaluated zero-shot
- **NETC**: Non-Enhancing Tumor Core
- **SNFH**: Surrounding Non-enhancing FLAIR Hyperintensity (Edema)
- **ET**: Enhancing Tumor
