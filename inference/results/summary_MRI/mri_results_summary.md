# MRI Brain Tumor Segmentation Results Summary

**Generated**: 2026-01-15

This document summarizes SAM3 vs MedSAM3 performance across six BraTS brain tumor datasets using text prompts only.

## Dataset Overview

| Dataset | Description | Labels | Contrasts | Cases |
|---------|-------------|--------|-----------|-------|
| BraTS2018 | High-Grade Glioma (HGG) | NCR, ED, ET | T1, T1CE, T2, FLAIR | 20 |
| BraTS2021 | Adult Glioma | NCR, ED, ET | T1, T1CE, T2, FLAIR | 20 |
| BraTS2023-MET | Brain Metastases | NETC, SNFH, ET | T1N, T1C, T2W, T2F | 20 |
| BraTS2023-PED | Pediatric Glioma | NC, ED, ET | T1N, T1C, T2W, T2F | 20 |
| BraTS2023-SSA | Sub-Saharan Africa Glioma | NC, ED, ET | T1N, T1C, T2W, T2F | 20 |
| BraTS2023-MEN | Meningioma | NETC, SNFH, ET | T1N, T1C, T2W, T2F | 20 |

## Label Definitions

| Abbreviation | Full Name | Datasets |
|--------------|-----------|----------|
| NCR | Necrotic/Non-Enhancing Tumor Core | BraTS2018, BraTS2021 |
| NC | Necrotic Core | BraTS2023-PED, BraTS2023-SSA |
| NETC | Non-Enhancing Tumor Core | BraTS2023-MET, BraTS2023-MEN |
| ED | Peritumoral Edema | BraTS2018, BraTS2021, BraTS2023-PED, BraTS2023-SSA |
| SNFH | Surrounding Non-Enhancing FLAIR Hyperintensity | BraTS2023-MET, BraTS2023-MEN |
| ET | Enhancing Tumor | All datasets |

## Contrast Mapping

| BraTS2018/2021 | BraTS2023 | Description |
|----------------|-----------|-------------|
| T1 | T1N | T1-weighted native |
| T1CE | T1C | T1 contrast-enhanced |
| T2 | T2W | T2-weighted |
| FLAIR | T2F | T2-FLAIR |

## Evaluation Protocol

- **Slice selection**: Middle slice from each 3D MRI volume
- **Prompt type**: Text prompts only (zero-shot, no prompt tuning)
- **Metrics**: Dice coefficient (%) and Intersection over Union (IoU, %)
- **Best prompt selection**: For each label-contrast combination, we report the best-performing text prompt from a candidate set

---

## BraTS2018 Results (High-Grade Glioma)

| Label | Contrast | SAM3 Dice | SAM3 IoU | MedSAM3 Dice | MedSAM3 IoU | N |
|-------|----------|-----------|----------|--------------|-------------|---|
| NCR | T1 | 49.8% | 38.1% | 40.7% | 28.5% | 15 |
| NCR | T1CE | 55.6% | 45.9% | 41.3% | 32.4% | 15 |
| NCR | T2 | 39.4% | 30.3% | 27.9% | 18.8% | 14 |
| NCR | FLAIR | 40.8% | 30.3% | 36.3% | 25.1% | 15 |
| ED | T1 | 58.2% | 44.0% | 37.7% | 27.3% | 20 |
| ED | T1CE | 64.3% | 50.8% | 31.1% | 24.2% | 20 |
| ED | T2 | 73.5% | 60.3% | 45.9% | 34.3% | 19 |
| ED | FLAIR | 73.5% | 60.2% | 48.3% | 39.1% | 20 |
| ET | T1 | 54.5% | 40.3% | 45.2% | 32.4% | 14 |
| ET | T1CE | **83.9%** | **73.2%** | 59.2% | 47.0% | 14 |
| ET | T2 | 63.7% | 49.5% | 48.8% | 34.9% | 13 |
| ET | FLAIR | 53.9% | 41.0% | 48.6% | 35.8% | 14 |

**Best SAM3**: ET on T1CE = 83.9% Dice
**Best MedSAM3**: ET on T1CE = 59.2% Dice

---

## BraTS2021 Results (Adult Glioma)

| Label | Contrast | SAM3 Dice | SAM3 IoU | MedSAM3 Dice | MedSAM3 IoU | N |
|-------|----------|-----------|----------|--------------|-------------|---|
| NCR | T1 | 36.0% | 28.3% | 28.9% | 23.3% | 12 |
| NCR | T1CE | 51.2% | 44.9% | 50.4% | 43.7% | 12 |
| NCR | T2 | 44.7% | 34.8% | 40.1% | 30.4% | 12 |
| NCR | FLAIR | 42.6% | 33.5% | 31.7% | 24.3% | 12 |
| ED | T1 | 55.5% | 41.7% | 38.3% | 27.4% | 18 |
| ED | T1CE | 61.9% | 47.9% | 47.9% | 36.1% | 18 |
| ED | T2 | 74.5% | 61.4% | 56.0% | 41.2% | 18 |
| ED | FLAIR | **76.4%** | **63.2%** | 55.2% | 41.8% | 18 |
| ET | T1 | 52.0% | 38.9% | 33.8% | 24.3% | 15 |
| ET | T1CE | 67.5% | 58.9% | 61.6% | 51.2% | 15 |
| ET | T2 | 53.7% | 40.1% | 50.4% | 37.8% | 15 |
| ET | FLAIR | 51.2% | 39.0% | 41.9% | 28.9% | 15 |

**Best SAM3**: ED on FLAIR = 76.4% Dice
**Best MedSAM3**: ET on T1CE = 61.6% Dice

---

## BraTS2023-MET Results (Brain Metastases)

*Note: Only Dice scores available for text prompts in this dataset.*

| Label | Contrast | SAM3 Dice | MedSAM3 Dice | N |
|-------|----------|-----------|--------------|---|
| NETC | T1N | 22.4% | 25.1% | 13 |
| NETC | T1C | 20.7% | 25.3% | 13 |
| NETC | T2W | 26.0% | 24.8% | 13 |
| NETC | T2F | 19.0% | 18.9% | 13 |
| SNFH | T1N | 10.4% | 13.4% | 32 |
| SNFH | T1C | 4.5% | 19.1% | 32 |
| SNFH | T2W | 8.8% | 20.0% | 32 |
| SNFH | T2F | 14.0% | **27.1%** | 32 |
| ET | T1N | 23.0% | 18.6% | 26 |
| ET | T1C | **40.6%** | 31.4% | 26 |
| ET | T2W | 32.0% | 27.1% | 26 |
| ET | T2F | 20.8% | 25.1% | 26 |

**Best SAM3**: ET on T1C = 40.6% Dice
**Best MedSAM3**: ET on T1C = 31.4% Dice

---

## BraTS2023-PED Results (Pediatric Glioma)

| Label | Contrast | SAM3 Dice | SAM3 IoU | MedSAM3 Dice | MedSAM3 IoU | N |
|-------|----------|-----------|----------|--------------|-------------|---|
| NC | T1N | 37.8% | 27.7% | 57.8% | 47.4% | 10 |
| NC | T1C | 50.7% | 40.2% | 61.9% | 52.7% | 10 |
| NC | T2W | 33.1% | 26.3% | **68.0%** | **60.1%** | 10 |
| NC | T2F | 44.8% | 34.4% | 60.6% | 51.6% | 10 |
| ED | T1N | 57.2% | 44.5% | 30.9% | 23.5% | 4 |
| ED | T1C | 68.5% | 54.6% | 47.3% | 36.0% | 4 |
| ED | T2W | 67.6% | 56.5% | 36.8% | 27.2% | 4 |
| ED | T2F | **78.0%** | **65.7%** | 71.1% | 57.7% | 4 |
| ET | T1N | 32.7% | 23.0% | 33.9% | 23.2% | 5 |
| ET | T1C | 46.8% | 35.6% | 35.0% | 23.3% | 5 |
| ET | T2W | 27.7% | 18.2% | 28.8% | 18.9% | 5 |
| ET | T2F | 26.2% | 17.3% | 30.2% | 20.6% | 5 |

**Best SAM3**: ED on T2F = 78.0% Dice
**Best MedSAM3**: ED on T2F = 71.1% Dice (NC on T2W = 68.0% Dice)

---

## BraTS2023-SSA Results (Sub-Saharan Africa Glioma)

| Label | Contrast | SAM3 Dice | SAM3 IoU | MedSAM3 Dice | MedSAM3 IoU | N |
|-------|----------|-----------|----------|--------------|-------------|---|
| NC | T1N | 67.7% | 59.2% | 66.7% | 58.4% | 9 |
| NC | T1C | **82.3%** | **74.1%** | 69.2% | 63.4% | 9 |
| NC | T2W | 69.3% | 59.4% | 61.8% | 50.7% | 9 |
| NC | T2F | 68.5% | 56.8% | 51.8% | 42.0% | 9 |
| ED | T1N | 55.6% | 41.6% | 10.6% | 7.2% | 14 |
| ED | T1C | 47.9% | 36.8% | 10.9% | 7.6% | 14 |
| ED | T2W | 62.5% | 48.9% | 36.2% | 26.6% | 14 |
| ED | T2F | 69.0% | 56.0% | 32.6% | 25.5% | 14 |
| ET | T1N | 43.0% | 29.9% | 34.0% | 23.5% | 9 |
| ET | T1C | 66.7% | 56.6% | 58.5% | 47.6% | 9 |
| ET | T2W | 51.0% | 37.5% | 46.2% | 34.3% | 9 |
| ET | T2F | 52.3% | 39.2% | 44.2% | 32.4% | 9 |

**Best SAM3**: NC on T1C = 82.3% Dice
**Best MedSAM3**: NC on T1C = 69.2% Dice

---

## BraTS2023-MEN Results (Meningioma)

*Note: NETC has only N=1 sample.*

| Label | Contrast | SAM3 Dice | SAM3 IoU | MedSAM3 Dice | MedSAM3 IoU | N |
|-------|----------|-----------|----------|--------------|-------------|---|
| NETC | T1N | **95.6%** | **91.6%** | 91.0% | 83.4% | 1 |
| NETC | T1C | 0.0% | 0.0% | 87.8% | 78.2% | 1 |
| NETC | T2W | 80.9% | 67.9% | 38.1% | 23.5% | 1 |
| NETC | T2F | 69.4% | 53.1% | 72.9% | 57.3% | 1 |
| SNFH | T1N | 63.0% | 50.1% | 39.4% | 29.7% | 6 |
| SNFH | T1C | 64.1% | 54.6% | 43.1% | 32.4% | 6 |
| SNFH | T2W | 82.2% | 71.8% | 43.2% | 35.2% | 6 |
| SNFH | T2F | 78.5% | 66.4% | 48.1% | 39.6% | 6 |
| ET | T1N | 55.2% | 45.8% | 46.5% | 30.8% | 4 |
| ET | T1C | **95.8%** | **92.0%** | 94.5% | 89.8% | 4 |
| ET | T2W | 76.5% | 62.5% | 76.6% | 64.1% | 4 |
| ET | T2F | 70.7% | 58.8% | 65.6% | 55.6% | 4 |

**Best SAM3**: ET on T1C = 95.8% Dice
**Best MedSAM3**: ET on T1C = 94.5% Dice

---

## Summary: Best Performance Per Dataset

| Dataset | SAM3 Best | Config | MedSAM3 Best | Config |
|---------|-----------|--------|--------------|--------|
| BraTS2018 | 83.9% | ET/T1CE | 59.2% | ET/T1CE |
| BraTS2021 | 76.4% | ED/FLAIR | 61.6% | ET/T1CE |
| BraTS2023-MET | 40.6% | ET/T1C | 31.4% | ET/T1C |
| BraTS2023-PED | 78.0% | ED/T2F | 71.1% | ED/T2F |
| BraTS2023-SSA | 82.3% | NC/T1C | 69.2% | NC/T1C |
| BraTS2023-MEN | 95.8% | ET/T1C | 94.5% | ET/T1C |

---

## Key Observations

1. **SAM3 vs MedSAM3**: SAM3 generally outperforms MedSAM3 on zero-shot text prompt evaluation for MRI brain tumors, suggesting the fine-tuning may have caused some forgetting of natural image priors useful for this domain.

2. **Best Contrasts by Label**:
   - **Enhancing Tumor (ET)**: Best performance on contrast-enhanced sequences (T1CE/T1C) across all datasets
   - **Edema (ED)**: FLAIR sequences perform well, as expected for fluid-sensitive imaging
   - **Necrotic Core (NC/NCR/NETC)**: Variable performance, often best on T1CE/T1C

3. **Dataset Difficulty**:
   - **Easiest**: BraTS2023-MEN (meningioma) - highest scores overall
   - **Hardest**: BraTS2023-MET (metastases) - lowest scores, challenging due to heterogeneous metastasis appearance

4. **Sample Size Considerations**: Results for some labels (e.g., NETC in BraTS2023-MEN with N=1) should be interpreted with caution due to limited sample sizes.

---

## Files and Locations

- **LaTeX appendix**: `/home/jtu9/SAM/paper/Medical-SAM3.../Medical Finetuneing/sec/appendix_mri.tex`
- **This summary**: `/home/jtu9/SAM/medsam/inference/results/summary_MRI/mri_results_summary.md`

### Source Data CSVs

| Dataset | SAM3 CSV | MedSAM3 CSV |
|---------|----------|-------------|
| BraTS2018 | `brats2018/sam3_brats18_text_prompt_by_label.csv` | `brats2018/medsam3_brats18_text_prompt_by_label.csv` |
| BraTS2021 | `brats2021/sam3_brats21_text_prompt_by_label.csv` | `brats2021/medsam3_brats21_text_prompt_by_label.csv` |
| BraTS2023-MET | `brats2023_met/sam3_brats_by_label.csv` | `brats2023_met/medsam3_brats_by_label.csv` |
| BraTS2023-PED | `brats2023_ped/sam3_ped_text_prompt_by_label.csv` | `brats2023_ped/medsam3_ped_text_prompt_by_label.csv` |
| BraTS2023-SSA | `brats2023_ssa/sam3_ssa_text_prompt_by_label.csv` | `brats2023_ssa/medsam3_ssa_text_prompt_by_label.csv` |
| BraTS2023-MEN | `brats2023_men/sam3_men_text_prompt_by_label.csv` | `brats2023_men/medsam3_men_text_prompt_by_label.csv` |
