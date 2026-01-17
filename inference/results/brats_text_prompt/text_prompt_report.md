# BraTS2023_MET Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T19:17:15.190112

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 48 (~20% of dataset)

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |
|----------|-------------|------------------|-----------------|-------|
| T1N | 0.0% | 0.6% | 20.25% | 6.87% |
| T1C | 0.0% | 0.0% | 20.28% | 2.38% |
| T2W | 0.0% | 0.0% | 20.96% | 5.31% |
| T2F | 0.0% | 1.49% | 21.03% | 11.14% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |
|----------|-------------|------------------|-----------------|-------|
| T1N | 7.82% | 13.12% | 22.49% | 18.15% |
| T1C | 11.15% | 18.14% | 29.23% | 23.82% |
| T2W | 7.16% | 17.3% | 32.77% | 25.81% |
| T2F | 18.17% | 31.14% | 37.99% | 33.39% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1N | Brain Tumor | 0.0% | 7.82% | **+7.82%** |
| T1N | Brain Metastasis | 0.6% | 13.12% | **+12.52%** |
| T1N | Enhancing Tumor | 20.25% | 22.49% | **+2.24%** |
| T1N | Tumor | 6.87% | 18.15% | **+11.28%** |
| T1C | Brain Tumor | 0.0% | 11.15% | **+11.15%** |
| T1C | Brain Metastasis | 0.0% | 18.14% | **+18.14%** |
| T1C | Enhancing Tumor | 20.28% | 29.23% | **+8.95%** |
| T1C | Tumor | 2.38% | 23.82% | **+21.44%** |
| T2W | Brain Tumor | 0.0% | 7.16% | **+7.16%** |
| T2W | Brain Metastasis | 0.0% | 17.3% | **+17.3%** |
| T2W | Enhancing Tumor | 20.96% | 32.77% | **+11.81%** |
| T2W | Tumor | 5.31% | 25.81% | **+20.5%** |
| T2F | Brain Tumor | 0.0% | 18.17% | **+18.17%** |
| T2F | Brain Metastasis | 1.49% | 31.14% | **+29.65%** |
| T2F | Enhancing Tumor | 21.03% | 37.99% | **+16.96%** |
| T2F | Tumor | 11.14% | 33.39% | **+22.25%** |

## Key Findings

- **Best MedSAM3 configuration**: Enhancing Tumor on T2F = 37.99% Dice
- **Average improvement over SAM3**: +14.8% Dice
- **Largest improvement**: Brain Metastasis on T2F = 29.65% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1-3)
- Text prompts evaluated zero-shot (no prompt tuning)
- MedSAM3 significantly improves text prompt understanding over SAM3 base model
