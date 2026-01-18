# Medical-SAM3

Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation

## Setup

```bash
conda activate medsam3
```

## Evaluation

### Base SAM3

```bash
python run_evaluation.py
```

### MedSAM3

```bash
python run_medsam3_evaluation.py \
    --checkpoint /path/to/checkpoint.pt \
    --model-name medsam3
```

**Options:** `--max-samples N`, `--datasets "Dataset1,Dataset2"`

## Visualization

```bash
cd visualization
python visualize_all_datasets.py
```

## Output

- **Results:** `results/*.csv`, `*.json`, `*.md`
- **Visualizations:** `visualization/{dataset}/comparison_*.png`

## Datasets

| Dataset | Modality | Target |
|---------|----------|--------|
| CHASE_DB1 | Fundus | Retinal Vessels |
| STARE | Fundus | Retinal Vessels |
| CVC-ClinicDB | Endoscopy | Polyps |
| ETIS-Larib | Endoscopy | Polyps |
| PH2 | Dermoscopy | Skin Lesions |

**Data path:** `../medsam_data/`
