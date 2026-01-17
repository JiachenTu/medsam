# BraTS2018 Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T23:40:42.077690

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 20

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 0.00% | 0.00% | 36.59% | 21.02% |
| T1CE | 0.00% | 0.00% | 33.37% | 14.79% |
| T2 | 0.00% | 0.00% | 34.40% | 20.03% |
| FLAIR | 0.00% | 0.00% | 26.60% | 20.81% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 35.22% | 27.53% | 44.22% | 42.16% |
| T1CE | 31.41% | 23.14% | 35.52% | 34.62% |
| T2 | 27.31% | 44.46% | 41.30% | 41.67% |
| FLAIR | 45.85% | 50.15% | 52.16% | 53.27% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1 | Brain Tumor | 0.0% | 35.22% | **+35.22%** |
| T1 | Glioma | 0.0% | 27.53% | **+27.53%** |
| T1 | Enhancing Tumor | 36.59% | 44.22% | **+7.63%** |
| T1 | Tumor | 21.02% | 42.16% | **+21.14%** |
| T1CE | Brain Tumor | 0.0% | 31.41% | **+31.41%** |
| T1CE | Glioma | 0.0% | 23.14% | **+23.14%** |
| T1CE | Enhancing Tumor | 33.37% | 35.52% | **+2.15%** |
| T1CE | Tumor | 14.79% | 34.62% | **+19.83%** |
| T2 | Brain Tumor | 0.0% | 27.31% | **+27.31%** |
| T2 | Glioma | 0.0% | 44.46% | **+44.46%** |
| T2 | Enhancing Tumor | 34.4% | 41.3% | **+6.9%** |
| T2 | Tumor | 20.03% | 41.67% | **+21.63%** |
| FLAIR | Brain Tumor | 0.0% | 45.85% | **+45.85%** |
| FLAIR | Glioma | 0.0% | 50.15% | **+50.15%** |
| FLAIR | Enhancing Tumor | 26.6% | 52.16% | **+25.56%** |
| FLAIR | Tumor | 20.81% | 53.27% | **+32.46%** |

## Key Findings

- **Best MedSAM3 configuration**: Tumor on FLAIR = 53.27% Dice
- **Average improvement over SAM3**: +26.4% Dice
- **Largest improvement**: Glioma on FLAIR = 50.15% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1, 2, 4)
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2018: High-grade glioma brain tumors (HGG only)
- Labels: NCR (1), ED (2), ET (4) - note ET is label 4
- File format: .nii (uncompressed)
- 190 total cases in dataset
