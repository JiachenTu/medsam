# BraTS2023_SSA Text Prompt-Only Evaluation Report

**Generated**: 2026-01-14T22:15:38.109279

**Evaluation Type**: Middle slice, text prompts only

**Cases**: 20

## SAM3 (Base Model) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 0.00% | 0.00% | 24.02% | 19.06% |
| T1C | 0.00% | 0.00% | 26.50% | 25.64% |
| T2W | 0.00% | 0.00% | 32.50% | 26.79% |
| T2F | 0.00% | 4.52% | 33.84% | 35.70% |

## MedSAM3 (Fine-tuned) - Text Prompt Performance

| Contrast | Brain Tumor | Glioma | Enhancing Tumor | Tumor |
|----------|----------|----------|----------|----------|
| T1N | 24.77% | 21.25% | 38.46% | 34.57% |
| T1C | 33.65% | 27.72% | 34.07% | 35.04% |
| T2W | 35.73% | 54.37% | 54.53% | 50.65% |
| T2F | 42.92% | 50.11% | 49.16% | 49.43% |

## MedSAM3 Improvement over SAM3

| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |
|----------|-------------|-----------|--------------|-------------|
| T1N | Brain Tumor | 0.0% | 24.77% | **+24.77%** |
| T1N | Glioma | 0.0% | 21.25% | **+21.25%** |
| T1N | Enhancing Tumor | 24.02% | 38.46% | **+14.44%** |
| T1N | Tumor | 19.06% | 34.57% | **+15.51%** |
| T1C | Brain Tumor | 0.0% | 33.65% | **+33.65%** |
| T1C | Glioma | 0.0% | 27.72% | **+27.72%** |
| T1C | Enhancing Tumor | 26.5% | 34.07% | **+7.57%** |
| T1C | Tumor | 25.64% | 35.04% | **+9.4%** |
| T2W | Brain Tumor | 0.0% | 35.73% | **+35.73%** |
| T2W | Glioma | 0.0% | 54.37% | **+54.37%** |
| T2W | Enhancing Tumor | 32.5% | 54.53% | **+22.04%** |
| T2W | Tumor | 26.79% | 50.65% | **+23.87%** |
| T2F | Brain Tumor | 0.0% | 42.92% | **+42.92%** |
| T2F | Glioma | 4.52% | 50.11% | **+45.59%** |
| T2F | Enhancing Tumor | 33.84% | 49.16% | **+15.32%** |
| T2F | Tumor | 35.7% | 49.43% | **+13.73%** |

## Key Findings

- **Best MedSAM3 configuration**: Enhancing Tumor on T2W = 54.53% Dice
- **Average improvement over SAM3**: +25.5% Dice
- **Largest improvement**: Glioma on T2W = 54.37% Dice improvement

## Notes

- Results from middle slice of each 3D MRI volume
- Combined tumor mask (all tumor labels 1-3)
- Text prompts evaluated zero-shot (no prompt tuning)
- BraTS2023_SSA: Sub-Saharan Africa gliomas
- Lower-quality MRI technology
