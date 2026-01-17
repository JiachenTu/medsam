# BraTS2023_MET All-Slices Evaluation Report

**Generated**: 2026-01-14T18:18:53.663356

**Total evaluations**: 13904

**Cases**: 5

**Total slices evaluated**: 409

## Box vs Text Prompt Performance (Overall)

| Model | Prompt | Dice (mean) | IoU (mean) | N |
|-------|--------|-------------|------------|---|
| sam3 | box | 65.5% | 54.0% | 3476 |
| sam3 | text | 53.2% | 43.1% | 3476 |
| medsam3 | box | 60.1% | 48.1% | 3476 |
| medsam3 | text | 38.0% | 29.8% | 3476 |

## Performance by Tumor Label

| Label | Model | Box Dice | Text Dice | Box-Text Gap |
|-------|-------|----------|-----------|-------------|
| NETC | sam3 | 62.7% | 45.9% | +16.8% |
| NETC | medsam3 | 61.4% | 38.8% | +22.6% |
| SNFH | sam3 | 73.1% | 61.1% | +12.0% |
| SNFH | medsam3 | 64.4% | 34.4% | +30.0% |
| ET | sam3 | 56.7% | 46.3% | +10.3% |
| ET | medsam3 | 53.8% | 42.2% | +11.6% |

## Performance by MRI Contrast

| Contrast | Model | Box Dice | Text Dice |
|----------|-------|----------|----------|
| T1N | sam3 | 60.6% | 46.6% |
| T1N | medsam3 | 57.3% | 30.1% |
| T1C | sam3 | 66.0% | 57.9% |
| T1C | medsam3 | 61.3% | 42.3% |
| T2W | sam3 | 67.1% | 56.1% |
| T2W | medsam3 | 59.9% | 38.0% |
| T2F | sam3 | 68.1% | 52.2% |
| T2F | medsam3 | 61.9% | 41.4% |

## Label-Specific Text Prompts

| Label | Text Prompt Used |
|-------|------------------|
| NETC | "Non-Enhancing Tumor Core" |
| SNFH | "Peritumoral Edema" |
| ET | "Enhancing Tumor" |

## Key Findings

- **Best box prompt**: sam3 on SNFH (T2F) = 80.2% Dice
- **Best text prompt**: sam3 on SNFH (T2F) = 69.1% Dice
- **MedSAM3 text improvement over SAM3**: -15.2% Dice
