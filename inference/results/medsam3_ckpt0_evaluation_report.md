# medsam3_ckpt0 Medical Image Segmentation Evaluation

**Generated**: 2025-12-31 02:04:39

## Summary

| Dataset | Text Prompt | Box Dice | Box IoU | Text Dice | Text IoU | Samples |
|---------|-------------|----------|---------|-----------|----------|--------|
| CHASE_DB1 | Retinal Blood Vessel | 32.7% | 21.0% | 62.6% | 45.7% | 28 |
| STARE | Retinal Blood Vessel | 59.0% | 42.6% | 54.4% | 37.8% | 20 |
| CVC-ClinicDB | Polyp | 91.5% | 85.6% | 87.9% | 81.2% | 612 |
| ETIS-Larib | Polyp | 91.9% | 86.3% | 86.1% | 79.3% | 196 |
| PH2 | Skin Lesion | 93.6% | 88.6% | 92.7% | 87.5% | 200 |

## Overall Averages

- **Box Prompt Average**: Dice=73.7%, IoU=64.8%
- **Text Prompt Average**: Dice=76.8%, IoU=66.3%

## Notes

- Box prompts use bounding boxes derived from ground truth masks
- Text prompts use natural language descriptions (zero-shot)
- Metrics: Dice (F1), IoU (Jaccard), PSNR, SSIM
