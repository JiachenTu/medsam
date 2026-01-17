# BraTS2021 Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T23:04:30.269112

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 20

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 0.00% | 0.00% | 41.63% | 16.54% |
| T1CE | 0.00% | 0.00% | 31.81% | 16.67% |
| T2 | 0.00% | 0.00% | 34.99% | 11.91% |
| FLAIR | 0.00% | 0.00% | 36.26% | 24.16% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1 | 39.80% | 35.41% | 38.62% | 42.07% |
| T1CE | 38.62% | 33.76% | 37.97% | 47.47% |
| T2 | 27.31% | 53.54% | 47.96% | 45.94% |
| FLAIR | 40.46% | 52.53% | 55.53% | 54.22% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1 | Brain Tumor | 0.0% | 39.8% | **+39.8%** |
| T1 | Glioma | 0.0% | 35.41% | **+35.41%** |
| T1 | Enhancing Tumor | 41.63% | 38.62% | **-3.01%** |
| T1 | Tumor | 16.54% | 42.07% | **+25.53%** |
| T1CE | Brain Tumor | 0.0% | 38.62% | **+38.62%** |
| T1CE | Glioma | 0.0% | 33.76% | **+33.76%** |
| T1CE | Enhancing Tumor | 31.81% | 37.97% | **+6.16%** |
| T1CE | Tumor | 16.67% | 47.47% | **+30.8%** |
| T2 | Brain Tumor | 0.0% | 27.31% | **+27.31%** |
| T2 | Glioma | 0.0% | 53.54% | **+53.54%** |
| T2 | Enhancing Tumor | 34.99% | 47.96% | **+12.97%** |
| T2 | Tumor | 11.91% | 45.94% | **+34.03%** |
| FLAIR | Brain Tumor | 0.0% | 40.46% | **+40.46%** |
| FLAIR | Glioma | 0.0% | 52.53% | **+52.53%** |
| FLAIR | Enhancing Tumor | 36.26% | 55.53% | **+19.28%** |
| FLAIR | Tumor | 24.16% | 54.22% | **+30.06%** |

## Key Findings

- **Best MedSAM3 configuration**: Enhancing Tumor on FLAIR = 55.53% Dice
- **Average improvement over SAM3**: +29.8% Dice
- **Largest improvement**: Glioma on T2 = 53.54% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1, 2, 4)
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2021: Adult glioma brain tumors
- Labels: NCR (1), ED (2), ET (4) - note ET is label 4
- 1,251 total cases in dataset
