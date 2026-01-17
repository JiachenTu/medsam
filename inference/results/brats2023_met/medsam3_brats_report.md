# medsam3 Evaluation on BraTS2023_MET

**Generated**: 2026-01-14T14:34:08.968443

**Cases evaluated**: 48

## Results by MRI Contrast (Combined Tumor Mask)

| Contrast | Box Dice | Box IoU | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |
|----------|----------|---------|-------------|------------------|-----------------|-------|
| T1N | 60.1% | 49.5% | 7.8% | 13.1% | 22.5% | 18.2% |
| T1C | 55.2% | 42.7% | 11.2% | 18.1% | 29.2% | 23.8% |
| T2W | 59.2% | 48.1% | 7.2% | 17.3% | 32.8% | 25.8% |
| T2F | 67.6% | 57.0% | 18.2% | 31.1% | 38.0% | 33.4% |

**Average Box Prompt Dice**: 60.5%

## Results by Tumor Label

| Label | Best Contrast | Box Dice | Best Text Prompt | Text Dice |
|-------|---------------|----------|------------------|----------|
| NETC | T1C | 58.6% | Enhancing Tumor | 25.3% |
| SNFH | T2F | 60.8% | Enhancing Tumor | 27.1% |
| ET | T1C | 55.3% | Enhancing Tumor | 31.4% |

## Notes

- Evaluated on middle slices of sampled cases (~20%)
- Box prompts derived from ground truth bounding boxes
- Text prompts evaluated zero-shot
- **NETC**: Non-Enhancing Tumor Core
- **SNFH**: Surrounding Non-enhancing FLAIR Hyperintensity (Edema)
- **ET**: Enhancing Tumor
