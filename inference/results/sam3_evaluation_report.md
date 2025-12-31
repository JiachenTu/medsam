# SAM3 Zero-Shot Medical Image Segmentation Evaluation

**Generated**: 2025-12-30 23:00:44

## Summary

| Dataset | Text Prompt | Box Dice | Box IoU | Text Dice | Text IoU | Samples |
|---------|-------------|----------|---------|-----------|----------|--------|
| CHASE_DB1 | Retinal Blood Vessel | 8.2% | 4.5% | 17.9% | 9.8% | 28 |
| STARE | Retinal Blood Vessel | 19.5% | 10.9% | 18.6% | 10.3% | 20 |
| CVC-ClinicDB | Polyp | 93.6% | 88.6% | 0.0% | 0.0% | 612 |
| ETIS-Larib | Polyp | 94.4% | 89.7% | 0.0% | 0.0% | 196 |
| PH2 | Skin Lesion | 93.9% | 89.7% | 18.4% | 14.9% | 200 |

## Overall Averages

- **Box Prompt Average**: Dice=61.9%, IoU=56.7%
- **Text Prompt Average**: Dice=11.0%, IoU=7.0%

## Notes

- Box prompts use bounding boxes derived from ground truth masks
- Text prompts use natural language descriptions (zero-shot)
- Metrics: Dice (F1), IoU (Jaccard), PSNR, SSIM
