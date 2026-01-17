# BraTS2023_PED Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T21:47:36.856385

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 20

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Pediatric Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|----------|
| T1N | 0.00% | 14.16% | 2.30% | 29.10% | 31.22% |
| T1C | 2.19% | 2.21% | 0.00% | 20.27% | 24.65% |
| T2W | 0.00% | 1.97% | 0.00% | 35.86% | 18.69% |
| T2F | 0.00% | 4.03% | 0.00% | 44.76% | 37.83% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Pediatric Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|----------|
| T1N | 42.94% | 51.80% | 43.65% | 53.82% | 60.28% |
| T1C | 44.32% | 62.83% | 40.20% | 50.16% | 61.17% |
| T2W | 44.68% | 64.29% | 67.50% | 61.61% | 63.59% |
| T2F | 53.35% | 67.52% | 65.50% | 67.89% | 75.19% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1N | Brain Tumor | 0.0% | 42.94% | **+42.94%** |
| T1N | Pediatric Brain Tumor | 14.16% | 51.8% | **+37.64%** |
| T1N | Glioma | 2.3% | 43.65% | **+41.36%** |
| T1N | Enhancing Tumor | 29.1% | 53.82% | **+24.72%** |
| T1N | Tumor | 31.22% | 60.28% | **+29.06%** |
| T1C | Brain Tumor | 2.19% | 44.32% | **+42.12%** |
| T1C | Pediatric Brain Tumor | 2.21% | 62.83% | **+60.62%** |
| T1C | Glioma | 0.0% | 40.2% | **+40.2%** |
| T1C | Enhancing Tumor | 20.27% | 50.16% | **+29.89%** |
| T1C | Tumor | 24.65% | 61.17% | **+36.53%** |
| T2W | Brain Tumor | 0.0% | 44.68% | **+44.68%** |
| T2W | Pediatric Brain Tumor | 1.97% | 64.29% | **+62.33%** |
| T2W | Glioma | 0.0% | 67.5% | **+67.5%** |
| T2W | Enhancing Tumor | 35.86% | 61.61% | **+25.75%** |
| T2W | Tumor | 18.69% | 63.59% | **+44.9%** |
| T2F | Brain Tumor | 0.0% | 53.35% | **+53.35%** |
| T2F | Pediatric Brain Tumor | 4.03% | 67.52% | **+63.49%** |
| T2F | Glioma | 0.0% | 65.5% | **+65.5%** |
| T2F | Enhancing Tumor | 44.76% | 67.89% | **+23.13%** |
| T2F | Tumor | 37.83% | 75.19% | **+37.36%** |

## Key Findings

- **Best MedSAM3 configuration**: Tumor on T2F = 75.19% Dice
- **Average improvement over SAM3**: +43.7% Dice
- **Largest improvement**: Glioma on T2W = 67.5% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1-3)
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_PED: Pediatric brain gliomas
