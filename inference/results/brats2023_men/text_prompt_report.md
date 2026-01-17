# BraTS2023_MEN Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T22:37:49.157659

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 20

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Meningioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 0.00% | 6.41% | 21.85% | 40.57% |
| T1C | 0.00% | 0.00% | 43.49% | 2.08% |
| T2W | 0.00% | 0.00% | 37.71% | 38.81% |
| T2F | 0.00% | 0.00% | 34.07% | 39.48% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Meningioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 40.04% | 26.61% | 39.95% | 36.83% |
| T1C | 41.82% | 34.24% | 42.87% | 42.74% |
| T2W | 49.90% | 35.67% | 38.87% | 51.12% |
| T2F | 32.12% | 34.79% | 25.14% | 46.41% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1N | Brain Tumor | 0.0% | 40.04% | **+40.04%** |
| T1N | Meningioma | 6.41% | 26.61% | **+20.2%** |
| T1N | Enhancing Tumor | 21.85% | 39.95% | **+18.1%** |
| T1N | Tumor | 40.57% | 36.83% | **-3.74%** |
| T1C | Brain Tumor | 0.0% | 41.82% | **+41.82%** |
| T1C | Meningioma | 0.0% | 34.24% | **+34.24%** |
| T1C | Enhancing Tumor | 43.49% | 42.87% | **-0.62%** |
| T1C | Tumor | 2.08% | 42.74% | **+40.66%** |
| T2W | Brain Tumor | 0.0% | 49.9% | **+49.9%** |
| T2W | Meningioma | 0.0% | 35.67% | **+35.67%** |
| T2W | Enhancing Tumor | 37.71% | 38.87% | **+1.16%** |
| T2W | Tumor | 38.81% | 51.12% | **+12.3%** |
| T2F | Brain Tumor | 0.0% | 32.12% | **+32.12%** |
| T2F | Meningioma | 0.0% | 34.79% | **+34.79%** |
| T2F | Enhancing Tumor | 34.07% | 25.14% | **-8.93%** |
| T2F | Tumor | 39.48% | 46.41% | **+6.93%** |

## Key Findings

- **Best MedSAM3 configuration**: Tumor on T2W = 51.12% Dice
- **Average improvement over SAM3**: +22.2% Dice
- **Largest improvement**: Brain Tumor on T2W = 49.9% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1-3)
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_MEN: Meningioma brain tumors
- Large dataset (1,139 cases)
