#!/usr/bin/env python3
"""
Visualization script for medical image segmentation results.

Generates comparison visualizations for all datasets:
- CHASE_DB1, STARE, CVC-ClinicDB, ETIS-Larib, PH2
- BraTS2023_MET (brain metastases MRI with 4 contrasts)

For each dataset, creates:
1. Individual images: original, GT, SAM3/MedSAM3 predictions (text & box prompts)
2. Combined 1x4 comparison layouts
3. For BraTS: multi-contrast grid visualization

Usage:
    conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3
    python visualize_all_datasets.py                    # All datasets except BraTS
    python visualize_all_datasets.py --brats            # BraTS only
    python visualize_all_datasets.py --brats --max-cases 5
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
INFERENCE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(INFERENCE_DIR))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from huggingface_hub import hf_hub_download

from dataset_loaders import load_dataset, DATASET_LOADERS, DATASET_PROMPTS
from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask
from brats_loader import (
    load_brats_sample, load_brats_all_labels, get_case_ids,
    get_cases_with_all_labels, CONTRASTS,
    STANDARD_TEXT_PROMPTS as BRATS_TEXT_PROMPTS, LABEL_PROMPTS
)
from brats_ped_loader import (
    load_brats_ped_sample, load_brats_ped_all_labels,
    get_case_ids as get_ped_case_ids,
    get_cases_with_all_labels as get_ped_cases_with_all_labels,
    CONTRASTS as PED_CONTRASTS,
    STANDARD_TEXT_PROMPTS as PED_TEXT_PROMPTS,
    LABEL_PROMPTS as PED_LABEL_PROMPTS
)
from brats_ssa_loader import (
    load_brats_ssa_sample, load_brats_ssa_all_labels,
    get_case_ids as get_ssa_case_ids,
    get_cases_with_all_labels as get_ssa_cases_with_all_labels,
    CONTRASTS as SSA_CONTRASTS,
    STANDARD_TEXT_PROMPTS as SSA_TEXT_PROMPTS,
    LABEL_PROMPTS as SSA_LABEL_PROMPTS
)
from brats_men_loader import (
    load_brats_men_sample, load_brats_men_all_labels,
    get_case_ids as get_men_case_ids,
    get_cases_with_all_labels as get_men_cases_with_all_labels,
    CONTRASTS as MEN_CONTRASTS,
    STANDARD_TEXT_PROMPTS as MEN_TEXT_PROMPTS,
    LABEL_PROMPTS as MEN_LABEL_PROMPTS
)
from brats21_loader import (
    load_brats21_sample, load_brats21_all_labels,
    get_case_ids as get_brats21_case_ids,
    get_cases_with_all_labels as get_brats21_cases_with_all_labels,
    CONTRASTS as BRATS21_CONTRASTS,
    STANDARD_TEXT_PROMPTS as BRATS21_TEXT_PROMPTS,
    LABEL_PROMPTS as BRATS21_LABEL_PROMPTS
)
from brats18_loader import (
    load_brats18_sample, load_brats18_all_labels,
    get_case_ids as get_brats18_case_ids,
    get_cases_with_all_labels as get_brats18_cases_with_all_labels,
    CONTRASTS as BRATS18_CONTRASTS,
    STANDARD_TEXT_PROMPTS as BRATS18_TEXT_PROMPTS,
    LABEL_PROMPTS as BRATS18_LABEL_PROMPTS
)


# Output directory
OUTPUT_DIR = SCRIPT_DIR


def download_medsam3_checkpoint() -> str:
    """
    Download MedSAM3 checkpoint from HuggingFace.

    Returns:
        Path to the downloaded checkpoint file
    """
    repo_id = "Potestates/medsam3_stage1_text_full"
    filename = "checkpoint.pt"
    print(f"Downloading MedSAM3 checkpoint from {repo_id}...")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded to: {checkpoint_path}")
    return checkpoint_path


def create_overlay(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Create a semi-transparent mask overlay on an image.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB color tuple for the mask
        alpha: Transparency (0=invisible, 1=opaque)

    Returns:
        Overlaid image (H, W, 3)
    """
    overlay = image.copy().astype(np.float32)

    # Resize mask if needed
    if mask.shape[:2] != image.shape[:2]:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = (np.array(mask_pil) > 127).astype(np.uint8)

    # Apply color overlay where mask is positive
    mask_bool = mask > 0
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c]
        )

    return overlay.astype(np.uint8)


def create_mask_visualization(mask: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Create a colored visualization of a binary mask.

    Args:
        mask: Binary mask (H, W)
        color: RGB color tuple

    Returns:
        Colored mask image (H, W, 3)
    """
    h, w = mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_bool = mask > 0
    for c in range(3):
        vis[:, :, c] = np.where(mask_bool, color[c], 0)
    return vis


# Label colors for multiclass visualization (BraTS2023_MET)
LABEL_COLORS = {
    'netc': (255, 0, 0),      # Red - Non-Enhancing Tumor Core
    'snfh': (0, 255, 0),      # Green - Edema/FLAIR Hyperintensity
    'et': (0, 0, 255),        # Blue - Enhancing Tumor
}

# Label colors for BraTS2023_PED (Pediatric)
PED_LABEL_COLORS = {
    'nc': (255, 0, 0),        # Red - Non-Enhancing Core
    'ed': (0, 255, 0),        # Green - Edema
    'et': (0, 0, 255),        # Blue - Enhancing Tumor
}

# Label colors for BraTS2021 (Adult Glioma) - labels 1, 2, 4
BRATS21_LABEL_COLORS = {
    'ncr': (255, 0, 0),       # Red - Necrotic/Non-Enhancing Tumor Core (label 1)
    'ed': (0, 255, 0),        # Green - Peritumoral Edema (label 2)
    'et': (0, 0, 255),        # Blue - Enhancing Tumor (label 4!)
}

# Label colors for BraTS2018 (HGG) - same labels as BraTS2021: 1, 2, 4
BRATS18_LABEL_COLORS = {
    'ncr': (255, 0, 0),       # Red - Necrotic/Non-Enhancing Tumor Core (label 1)
    'ed': (0, 255, 0),        # Green - Peritumoral Edema (label 2)
    'et': (0, 0, 255),        # Blue - Enhancing Tumor (label 4!)
}


def create_multiclass_overlay(
    image: np.ndarray,
    netc_mask: np.ndarray,
    snfh_mask: np.ndarray,
    et_mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay with different colors for each tumor class.

    Colors:
    - NETC (label 1): Red (255, 0, 0)
    - SNFH (label 2): Green (0, 255, 0)
    - ET (label 3): Blue (0, 0, 255)

    Args:
        image: RGB image (H, W, 3)
        netc_mask: Binary mask for NETC
        snfh_mask: Binary mask for SNFH
        et_mask: Binary mask for ET
        alpha: Transparency (0=invisible, 1=opaque)

    Returns:
        Overlaid image (H, W, 3)
    """
    overlay = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    # Resize masks if needed
    def resize_if_needed(mask):
        if mask.shape[:2] != (h, w):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            return (np.array(mask_pil) > 127).astype(np.uint8)
        return mask

    netc_mask = resize_if_needed(netc_mask)
    snfh_mask = resize_if_needed(snfh_mask)
    et_mask = resize_if_needed(et_mask)

    # Apply each color overlay
    for mask, color in [(netc_mask, LABEL_COLORS['netc']),
                        (snfh_mask, LABEL_COLORS['snfh']),
                        (et_mask, LABEL_COLORS['et'])]:
        mask_bool = mask > 0
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    return overlay.astype(np.uint8)


def create_per_label_figure(
    image: np.ndarray,
    gt_masks: dict,
    sam3_preds: dict,
    medsam3_preds: dict,
    case_id: str,
    contrast: str,
    output_path: Path
):
    """
    Create 3-row x 4-column grid showing per-label comparison.

    Rows: NETC, SNFH, ET (one per label)
    Columns: Original+GT | SAM3 Pred | MedSAM3 Pred | Overlap

    Args:
        image: Original RGB image
        gt_masks: Dict with 'netc', 'snfh', 'et' binary masks
        sam3_preds: Dict with 'netc', 'snfh', 'et' prediction masks
        medsam3_preds: Dict with 'netc', 'snfh', 'et' prediction masks
        case_id: Case identifier
        contrast: MRI contrast
        output_path: Path to save figure
    """
    label_names = ['netc', 'snfh', 'et']
    label_titles = {
        'netc': 'NETC\nPrompt: "Non-Enhancing Tumor Core"',
        'snfh': 'SNFH\nPrompt: "Peritumoral Edema"',
        'et': 'ET\nPrompt: "Enhancing Tumor"'
    }

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for row, label in enumerate(label_names):
        gt_mask = gt_masks.get(label)
        sam3_pred = sam3_preds.get(label)
        medsam3_pred = medsam3_preds.get(label)
        color = LABEL_COLORS[label]

        # Column 0: Original + GT overlay
        if gt_mask is not None and gt_mask.sum() > 0:
            gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
        else:
            gt_overlay = image.copy()
        axes[row, 0].imshow(gt_overlay)
        axes[row, 0].set_ylabel(label_titles[label], fontsize=10, fontweight='bold')

        # Column 1: SAM3 prediction overlay
        if sam3_pred is not None and sam3_pred.sum() > 0:
            sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)  # Orange
        else:
            sam3_overlay = image.copy()
        axes[row, 1].imshow(sam3_overlay)

        # Column 2: MedSAM3 prediction overlay
        if medsam3_pred is not None and medsam3_pred.sum() > 0:
            medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)  # Cyan
        else:
            medsam3_overlay = image.copy()
        axes[row, 2].imshow(medsam3_overlay)

        # Column 3: GT vs MedSAM3 overlap
        if gt_mask is not None and medsam3_pred is not None:
            overlap_img = image.copy().astype(np.float32)
            # Green for GT only, Cyan for pred only, White for overlap
            gt_only = (gt_mask > 0) & ~(medsam3_pred > 0)
            pred_only = (medsam3_pred > 0) & ~(gt_mask > 0)
            both = (gt_mask > 0) & (medsam3_pred > 0)

            for c, (g, p, b) in enumerate([(0, 0, 255), (255, 0, 255), (255, 255, 255)]):
                # GT only = green, Pred only = magenta, Both = white
                pass

            overlap_overlay = create_overlay(image, gt_mask, (0, 255, 0), alpha=0.3)
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                overlap_overlay = create_overlay(overlap_overlay, medsam3_pred, (0, 255, 255), alpha=0.3)
        else:
            overlap_overlay = image.copy()
        axes[row, 3].imshow(overlap_overlay)

        # Remove axis ticks
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Column titles
    axes[0, 0].set_title('Ground Truth', fontsize=12)
    axes[0, 1].set_title('SAM3 Prediction', fontsize=12)
    axes[0, 2].set_title('MedSAM3 Prediction', fontsize=12)
    axes[0, 3].set_title('GT + MedSAM3 Overlap', fontsize=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(LABEL_COLORS['netc'])/255, label='NETC (GT)'),
        mpatches.Patch(color=np.array(LABEL_COLORS['snfh'])/255, label='SNFH (GT)'),
        mpatches.Patch(color=np.array(LABEL_COLORS['et'])/255, label='ET (GT)'),
        mpatches.Patch(color=np.array((255, 165, 0))/255, label='SAM3 Pred'),
        mpatches.Patch(color=np.array((0, 255, 255))/255, label='MedSAM3 Pred'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=10)

    plt.suptitle(f'BraTS2023_MET - {case_id} - {contrast.upper()}\nPer-Label Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_single_label_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam3_pred: np.ndarray,
    medsam3_pred: np.ndarray,
    label_name: str,
    text_prompt: str,
    case_id: str,
    contrast: str,
    output_path: Path
):
    """
    Create 1x4 comparison figure for a single label.

    Columns: Original | GT | SAM3 | MedSAM3
    Title includes the text prompt used.

    Args:
        image: Original RGB image
        gt_mask: Ground truth mask for this label
        sam3_pred: SAM3 prediction mask
        medsam3_pred: MedSAM3 prediction mask
        label_name: Label name (netc, snfh, et)
        text_prompt: Text prompt used for inference
        case_id: Case identifier
        contrast: MRI contrast
        output_path: Path to save figure
    """
    label_full_names = {
        'netc': 'NETC (Non-Enhancing Tumor Core)',
        'snfh': 'SNFH (Peritumoral Edema)',
        'et': 'ET (Enhancing Tumor)'
    }

    gt_color = LABEL_COLORS.get(label_name, (0, 255, 0))
    sam3_color = (255, 165, 0)     # Orange
    medsam3_color = (0, 255, 255)  # Cyan

    # Create overlays
    if gt_mask is not None and gt_mask.sum() > 0:
        gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)
    else:
        gt_overlay = image.copy()

    if sam3_pred is not None and sam3_pred.sum() > 0:
        sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
    else:
        sam3_overlay = image.copy()

    if medsam3_pred is not None and medsam3_pred.sum() > 0:
        medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
    else:
        medsam3_overlay = image.copy()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Panel 2: GT
    axes[1].imshow(gt_overlay)
    gt_pixels = gt_mask.sum() if gt_mask is not None else 0
    axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
    axes[1].axis("off")

    # Panel 3: SAM3
    axes[2].imshow(sam3_overlay)
    sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
    axes[2].set_title(f"SAM3 Prediction\n({sam3_pixels} pixels)", fontsize=12)
    axes[2].axis("off")

    # Panel 4: MedSAM3
    axes[3].imshow(medsam3_overlay)
    medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
    axes[3].set_title(f"MedSAM3 Prediction\n({medsam3_pixels} pixels)", fontsize=12)
    axes[3].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
        mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
        mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

    label_title = label_full_names.get(label_name, label_name.upper())
    plt.suptitle(f'BraTS2023_MET - {case_id} - {contrast.upper()}\n'
                 f'{label_title}\nText Prompt: "{text_prompt}"',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.88])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_raw_masks(
    output_dir: Path,
    gt_mask: np.ndarray,
    sam3_text_pred: np.ndarray,
    medsam3_text_pred: np.ndarray,
    sam3_box_pred: np.ndarray,
    medsam3_box_pred: np.ndarray,
    img_size: tuple
):
    """
    Save raw binary masks (white on black).

    Args:
        output_dir: Directory to save masks
        gt_mask: Ground truth mask
        sam3_text_pred: SAM3 text prompt prediction
        medsam3_text_pred: MedSAM3 text prompt prediction
        sam3_box_pred: SAM3 box prompt prediction
        medsam3_box_pred: MedSAM3 box prompt prediction
        img_size: Target size (H, W) for masks
    """
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    def save_mask(mask: np.ndarray, filename: str):
        """Save a binary mask as white (255) on black (0)."""
        if mask is None:
            # Save empty black image
            empty = np.zeros(img_size, dtype=np.uint8)
            Image.fromarray(empty).save(masks_dir / filename)
        else:
            # Resize if needed
            if mask.shape[:2] != img_size:
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
                mask = np.array(mask_pil)
            else:
                mask = (mask * 255).astype(np.uint8)
            Image.fromarray(mask).save(masks_dir / filename)

    # Save all masks
    save_mask(gt_mask, "gt_mask.png")
    save_mask(sam3_text_pred, "sam3_text_mask.png")
    save_mask(sam3_box_pred, "sam3_box_mask.png")
    save_mask(medsam3_text_pred, "medsam3_text_mask.png")
    save_mask(medsam3_box_pred, "medsam3_box_mask.png")


def save_individual_images(
    output_dir: Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam3_text_pred: np.ndarray,
    medsam3_text_pred: np.ndarray,
    sam3_box_pred: np.ndarray,
    medsam3_box_pred: np.ndarray
):
    """
    Save individual visualization images.

    Args:
        output_dir: Directory to save images
        image: Original RGB image
        gt_mask: Ground truth mask
        sam3_text_pred: SAM3 text prompt prediction
        medsam3_text_pred: MedSAM3 text prompt prediction
        sam3_box_pred: SAM3 box prompt prediction
        medsam3_box_pred: MedSAM3 box prompt prediction
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    gt_color = (0, 255, 0)       # Green for GT
    sam3_color = (255, 0, 0)     # Red for SAM3
    medsam3_color = (0, 0, 255)  # Blue for MedSAM3

    # Save original image
    Image.fromarray(image).save(output_dir / "original.png")

    # Save GT mask (as overlay on image)
    gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)
    Image.fromarray(gt_overlay).save(output_dir / "gt.png")

    # Save SAM3 text prediction
    if sam3_text_pred is not None:
        sam3_text_overlay = create_overlay(image, sam3_text_pred, sam3_color, alpha=0.5)
    else:
        sam3_text_overlay = image.copy()
    Image.fromarray(sam3_text_overlay).save(output_dir / "sam3_text.png")

    # Save MedSAM3 text prediction
    if medsam3_text_pred is not None:
        medsam3_text_overlay = create_overlay(image, medsam3_text_pred, medsam3_color, alpha=0.5)
    else:
        medsam3_text_overlay = image.copy()
    Image.fromarray(medsam3_text_overlay).save(output_dir / "medsam3_text.png")

    # Save SAM3 box prediction
    if sam3_box_pred is not None:
        sam3_box_overlay = create_overlay(image, sam3_box_pred, sam3_color, alpha=0.5)
    else:
        sam3_box_overlay = image.copy()
    Image.fromarray(sam3_box_overlay).save(output_dir / "sam3_box.png")

    # Save MedSAM3 box prediction
    if medsam3_box_pred is not None:
        medsam3_box_overlay = create_overlay(image, medsam3_box_pred, medsam3_color, alpha=0.5)
    else:
        medsam3_box_overlay = image.copy()
    Image.fromarray(medsam3_box_overlay).save(output_dir / "medsam3_box.png")

    # Save raw binary masks
    save_raw_masks(
        output_dir,
        gt_mask,
        sam3_text_pred,
        medsam3_text_pred,
        sam3_box_pred,
        medsam3_box_pred,
        gt_mask.shape[:2]
    )


def create_comparison_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    sam3_pred: np.ndarray,
    medsam3_pred: np.ndarray,
    title: str,
    output_path: Path,
    prompt_type: str = "text"
):
    """
    Create and save a 1x4 horizontal comparison figure.

    Args:
        image: Original RGB image
        gt_mask: Ground truth mask
        sam3_pred: SAM3 prediction
        medsam3_pred: MedSAM3 prediction
        title: Figure title
        output_path: Path to save the figure
        prompt_type: "text" or "box"
    """
    # Colors
    gt_color = (0, 255, 0)       # Green
    sam3_color = (255, 0, 0)     # Red
    medsam3_color = (0, 0, 255)  # Blue

    # Create overlays
    gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

    if sam3_pred is not None:
        sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
    else:
        sam3_overlay = image.copy()

    if medsam3_pred is not None:
        medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
    else:
        medsam3_overlay = image.copy()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Panel 2: GT
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis("off")

    # Panel 3: SAM3
    axes[2].imshow(sam3_overlay)
    sam3_title = f"SAM3 ({prompt_type.capitalize()} Prompt)"
    if sam3_pred is None:
        sam3_title += "\n(No prediction)"
    axes[2].set_title(sam3_title, fontsize=14)
    axes[2].axis("off")

    # Panel 4: MedSAM3
    axes[3].imshow(medsam3_overlay)
    medsam3_title = f"MedSAM3 ({prompt_type.capitalize()} Prompt)"
    if medsam3_pred is None:
        medsam3_title += "\n(No prediction)"
    axes[3].set_title(medsam3_title, fontsize=14)
    axes[3].axis("off")

    # Add legend
    legend_patches = [
        mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
        mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
        mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    dataset_name: str,
    output_base_dir: Path
) -> bool:
    """
    Process one dataset: load sample, run inference, save visualizations.

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        dataset_name: Name of the dataset
        output_base_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Text Prompt: {DATASET_PROMPTS[dataset_name]}")
    print(f"{'='*60}")

    # Load first sample from dataset
    samples = list(load_dataset(dataset_name, max_samples=1))
    if not samples:
        print(f"  ERROR: No samples found for {dataset_name}")
        return False

    sample = samples[0]
    print(f"  Sample ID: {sample.sample_id}")
    print(f"  Image shape: {sample.image.shape}")

    # Get image size for mask resizing
    img_size = sample.gt_mask.shape

    # Generate bounding box from GT mask
    bbox = generate_bbox_from_mask(sample.gt_mask)
    if bbox is None:
        print(f"  ERROR: Could not generate bbox from GT mask")
        return False
    print(f"  BBox: {bbox}")

    # Run SAM3 inference
    print("  Running SAM3 inference...")
    sam3_state = sam3_model.encode_image(sample.image)

    # SAM3 text prediction
    sam3_text_pred = sam3_model.predict_text(sam3_state, sample.text_prompt)
    if sam3_text_pred is not None and sam3_text_pred.shape != img_size:
        sam3_text_pred = resize_mask(sam3_text_pred, img_size)

    # SAM3 box prediction
    sam3_box_pred = sam3_model.predict_box(sam3_state, bbox, img_size)
    if sam3_box_pred is not None and sam3_box_pred.shape != img_size:
        sam3_box_pred = resize_mask(sam3_box_pred, img_size)

    # Run MedSAM3 inference
    print("  Running MedSAM3 inference...")
    medsam3_state = medsam3_model.encode_image(sample.image)

    # MedSAM3 text prediction
    medsam3_text_pred = medsam3_model.predict_text(medsam3_state, sample.text_prompt)
    if medsam3_text_pred is not None and medsam3_text_pred.shape != img_size:
        medsam3_text_pred = resize_mask(medsam3_text_pred, img_size)

    # MedSAM3 box prediction
    medsam3_box_pred = medsam3_model.predict_box(medsam3_state, bbox, img_size)
    if medsam3_box_pred is not None and medsam3_box_pred.shape != img_size:
        medsam3_box_pred = resize_mask(medsam3_box_pred, img_size)

    # Create output directory for this dataset
    output_dir = output_base_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual images
    print("  Saving individual images...")
    save_individual_images(
        output_dir,
        sample.image,
        sample.gt_mask,
        sam3_text_pred,
        medsam3_text_pred,
        sam3_box_pred,
        medsam3_box_pred
    )

    # Create and save comparison figures
    print("  Creating comparison figures...")

    # Text prompt comparison
    create_comparison_figure(
        sample.image,
        sample.gt_mask,
        sam3_text_pred,
        medsam3_text_pred,
        f"{dataset_name} - Text Prompt Comparison\n(Prompt: \"{sample.text_prompt}\")",
        output_dir / "comparison_text.png",
        prompt_type="text"
    )

    # Box prompt comparison
    create_comparison_figure(
        sample.image,
        sample.gt_mask,
        sam3_box_pred,
        medsam3_box_pred,
        f"{dataset_name} - Box Prompt Comparison",
        output_dir / "comparison_box.png",
        prompt_type="box"
    )

    print(f"  Saved outputs to: {output_dir}")
    return True


def create_multi_contrast_figure(
    contrast_data: Dict,
    case_id: str,
    output_path: Path,
    text_prompt: str = "Brain Tumor"
):
    """
    Create a 4-row x 4-column grid showing all contrasts.

    Rows: t1n, t1c, t2w, t2f
    Columns: Original | GT | SAM3 Text | MedSAM3 Text

    Args:
        contrast_data: Dict with contrast names as keys, containing image/mask data
        case_id: Case identifier for title
        output_path: Path to save the figure
        text_prompt: Text prompt used for inference
    """
    # Colors
    gt_color = (0, 255, 0)       # Green
    sam3_color = (255, 0, 0)     # Red
    medsam3_color = (0, 0, 255)  # Blue

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    contrast_labels = {
        't1n': 'T1 Native',
        't1c': 'T1 Contrast',
        't2w': 'T2 Weighted',
        't2f': 'T2 FLAIR'
    }

    for row, contrast in enumerate(CONTRASTS):
        if contrast not in contrast_data:
            continue

        data = contrast_data[contrast]
        image = data['image']
        gt_mask = data['gt_mask']
        sam3_pred = data.get('sam3_text_pred')
        medsam3_pred = data.get('medsam3_text_pred')

        # Create overlays
        gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

        if sam3_pred is not None:
            sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
        else:
            sam3_overlay = image.copy()

        if medsam3_pred is not None:
            medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
        else:
            medsam3_overlay = image.copy()

        # Plot row
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(sam3_overlay)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(medsam3_overlay)
        axes[row, 3].axis('off')

        # Add column titles for first row
        if row == 0:
            axes[0, 0].set_title('Original', fontsize=14)
            axes[0, 1].set_title('Ground Truth', fontsize=14)
            axes[0, 2].set_title('SAM3 Text', fontsize=14)
            axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

    # Add legend
    legend_patches = [
        mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
        mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
        mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

    plt.suptitle(f'BraTS2023_MET - {case_id}\nText Prompt: "{text_prompt}"',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_brats_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = False
) -> bool:
    """
    Process one BraTS case across all 4 contrasts with per-label visualization.

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS case identifier
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = BRATS_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using new function
        data = load_brats_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NETC pixels: {data['netc_mask'].sum()}")
        print(f"    SNFH pixels: {data['snfh_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Generate bbox from combined mask
        bbox = generate_bbox_from_mask(data['combined_mask'])
        if bbox is None:
            print(f"    ERROR: Could not generate bbox")
            continue

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks
        gt_masks = {
            'netc': data['netc_mask'],
            'snfh': data['snfh_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Box predictions (using combined mask bbox)
        sam3_box_pred = sam3_model.predict_box(sam3_state, bbox, img_size)
        if sam3_box_pred is not None and sam3_box_pred.shape != img_size:
            sam3_box_pred = resize_mask(sam3_box_pred, img_size)

        medsam3_box_pred = medsam3_model.predict_box(medsam3_state, bbox, img_size)
        if medsam3_box_pred is not None and medsam3_box_pred.shape != img_size:
            medsam3_box_pred = resize_mask(medsam3_box_pred, img_size)

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
            'sam3_box_pred': sam3_box_pred,
            'medsam3_box_pred': medsam3_box_pred,
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2023_MET" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        gt_multiclass = create_multiclass_overlay(
            image, data['netc_mask'], data['snfh_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = LABEL_COLORS[label]
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['netc', 'snfh', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create per-label comparison figure (3-row grid)
        create_per_label_figure(
            image, gt_masks, sam3_preds, medsam3_preds,
            case_id, contrast,
            output_dir / "comparison_by_label.png"
        )

        # Create individual 1x4 comparison figures for each label
        for label in ['netc', 'snfh', 'et']:
            prompt = LABEL_PROMPTS[label.upper()]
            create_single_label_comparison(
                image,
                gt_masks.get(label),
                sam3_preds.get(label),
                medsam3_preds.get(label),
                label, prompt, case_id, contrast,
                output_dir / f"comparison_{label}.png"
            )

        # Box prompt comparison (combined mask) - skip if text_only
        if not text_only:
            create_comparison_figure(
                image,
                data['combined_mask'],
                sam3_box_pred,
                medsam3_box_pred,
                f"BraTS2023_MET - {case_id} - {contrast.upper()}\nBox Prompt (Combined Mask)",
                output_dir / "comparison_box.png",
                prompt_type="box"
            )

    # Create multi-contrast grid figure
    if contrast_data:
        print(f"\n  Creating multi-contrast grid...")
        grid_output = output_base_dir / "BraTS2023_MET" / case_id / "all_contrasts_comparison.png"
        create_multi_contrast_figure(contrast_data, case_id, grid_output, "Enhancing Tumor")
        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def process_brats_ped_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = True
) -> bool:
    """
    Process one BraTS-PED case across all 4 contrasts with per-label visualization.

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS-PED case identifier
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons (default True for PED)

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = PED_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS-PED case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in PED_CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using brats_ped_loader
        data = load_brats_ped_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NC pixels: {data['nc_mask'].sum()}")
        print(f"    ED pixels: {data['ed_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks (PED uses nc, ed, et instead of netc, snfh, et)
        gt_masks = {
            'nc': data['nc_mask'],
            'ed': data['ed_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in PED_LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2023_PED" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        gt_multiclass = create_multiclass_overlay(
            image, data['nc_mask'], data['ed_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = PED_LABEL_COLORS[label]
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['nc', 'ed', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create individual 1x4 comparison figures for each label
        ped_label_full_names = {
            'nc': 'NC (Non-Enhancing Core)',
            'ed': 'ED (Edema)',
            'et': 'ET (Enhancing Tumor)'
        }
        for label in ['nc', 'ed', 'et']:
            prompt = PED_LABEL_PROMPTS[label.upper()]
            # Create simple comparison figure
            gt_mask = gt_masks.get(label)
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            color = PED_LABEL_COLORS[label]

            # Create overlays
            if gt_mask is not None and gt_mask.sum() > 0:
                gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
            else:
                gt_overlay = image.copy()

            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            gt_pixels = gt_mask.sum() if gt_mask is not None else 0
            axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(sam3_overlay)
            sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
            axes[2].set_title(f"SAM3 Prediction\n({sam3_pixels} pixels)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(medsam3_overlay)
            medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
            axes[3].set_title(f"MedSAM3 Prediction\n({medsam3_pixels} pixels)", fontsize=12)
            axes[3].axis("off")

            # Legend
            legend_patches = [
                mpatches.Patch(color=np.array(color)/255, label='Ground Truth'),
                mpatches.Patch(color=np.array((255, 165, 0))/255, label='SAM3'),
                mpatches.Patch(color=np.array((0, 255, 255))/255, label='MedSAM3'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

            label_title = ped_label_full_names.get(label, label.upper())
            plt.suptitle(f'BraTS2023_PED - {case_id} - {contrast.upper()}\n'
                         f'{label_title}\nText Prompt: "{prompt}"',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.88])

            output_path = output_dir / f"comparison_{label}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Create multi-contrast grid figure
    if contrast_data:
        print(f"\n  Creating multi-contrast grid...")
        grid_output = output_base_dir / "BraTS2023_PED" / case_id / "all_contrasts_comparison.png"

        # Colors
        gt_color = (0, 255, 0)
        sam3_color = (255, 0, 0)
        medsam3_color = (0, 0, 255)

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        contrast_labels = {
            't1n': 'T1 Native',
            't1c': 'T1 Contrast',
            't2w': 'T2 Weighted',
            't2f': 'T2 FLAIR'
        }

        for row, contrast in enumerate(PED_CONTRASTS):
            if contrast not in contrast_data:
                for col in range(4):
                    axes[row, col].axis('off')
                continue

            data = contrast_data[contrast]
            image = data['image']
            gt_mask = data['gt_mask']
            sam3_pred = data.get('sam3_text_pred')
            medsam3_pred = data.get('medsam3_text_pred')

            # Create overlays
            gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

            if sam3_pred is not None:
                sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None:
                medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Plot row
            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(sam3_overlay)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(medsam3_overlay)
            axes[row, 3].axis('off')

            # Add column titles for first row
            if row == 0:
                axes[0, 0].set_title('Original', fontsize=14)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 2].set_title('SAM3 Text', fontsize=14)
                axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
            mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
            mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

        plt.suptitle(f'BraTS2023_PED - {case_id}\nText Prompt: "Enhancing Tumor"',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def process_brats_ssa_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = True
) -> bool:
    """
    Process one BraTS-SSA case across all 4 contrasts with per-label visualization.

    Uses same labels as PED (NC, ED, ET).

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS-SSA case identifier
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons (default True for SSA)

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = SSA_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS-SSA case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in SSA_CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using brats_ssa_loader
        data = load_brats_ssa_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NC pixels: {data['nc_mask'].sum()}")
        print(f"    ED pixels: {data['ed_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks (SSA uses nc, ed, et - same as PED)
        gt_masks = {
            'nc': data['nc_mask'],
            'ed': data['ed_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in SSA_LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2023_SSA" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        gt_multiclass = create_multiclass_overlay(
            image, data['nc_mask'], data['ed_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = PED_LABEL_COLORS[label]  # Same colors as PED
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['nc', 'ed', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create individual 1x4 comparison figures for each label
        ssa_label_full_names = {
            'nc': 'NC (Non-Enhancing Core)',
            'ed': 'ED (Edema)',
            'et': 'ET (Enhancing Tumor)'
        }
        for label in ['nc', 'ed', 'et']:
            prompt = SSA_LABEL_PROMPTS[label.upper()]
            # Create simple comparison figure
            gt_mask = gt_masks.get(label)
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            color = PED_LABEL_COLORS[label]  # Same colors as PED

            # Create overlays
            if gt_mask is not None and gt_mask.sum() > 0:
                gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
            else:
                gt_overlay = image.copy()

            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            gt_pixels = gt_mask.sum() if gt_mask is not None else 0
            axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(sam3_overlay)
            sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
            axes[2].set_title(f"SAM3 Prediction\n({sam3_pixels} pixels)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(medsam3_overlay)
            medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
            axes[3].set_title(f"MedSAM3 Prediction\n({medsam3_pixels} pixels)", fontsize=12)
            axes[3].axis("off")

            # Legend
            legend_patches = [
                mpatches.Patch(color=np.array(color)/255, label='Ground Truth'),
                mpatches.Patch(color=np.array((255, 165, 0))/255, label='SAM3'),
                mpatches.Patch(color=np.array((0, 255, 255))/255, label='MedSAM3'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

            label_title = ssa_label_full_names.get(label, label.upper())
            plt.suptitle(f'BraTS2023_SSA - {case_id} - {contrast.upper()}\n'
                         f'{label_title}\nText Prompt: "{prompt}"',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.88])

            output_path = output_dir / f"comparison_{label}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Create multi-contrast grid figure
    if contrast_data:
        print(f"\n  Creating multi-contrast grid...")
        grid_output = output_base_dir / "BraTS2023_SSA" / case_id / "all_contrasts_comparison.png"

        # Colors
        gt_color = (0, 255, 0)
        sam3_color = (255, 0, 0)
        medsam3_color = (0, 0, 255)

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        contrast_labels = {
            't1n': 'T1 Native',
            't1c': 'T1 Contrast',
            't2w': 'T2 Weighted',
            't2f': 'T2 FLAIR'
        }

        for row, contrast in enumerate(SSA_CONTRASTS):
            if contrast not in contrast_data:
                for col in range(4):
                    axes[row, col].axis('off')
                continue

            data = contrast_data[contrast]
            image = data['image']
            gt_mask = data['gt_mask']
            sam3_pred = data.get('sam3_text_pred')
            medsam3_pred = data.get('medsam3_text_pred')

            # Create overlays
            gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

            if sam3_pred is not None:
                sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None:
                medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Plot row
            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(sam3_overlay)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(medsam3_overlay)
            axes[row, 3].axis('off')

            # Add column titles for first row
            if row == 0:
                axes[0, 0].set_title('Original', fontsize=14)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 2].set_title('SAM3 Text', fontsize=14)
                axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
            mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
            mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

        plt.suptitle(f'BraTS2023_SSA - {case_id}\nText Prompt: "Enhancing Tumor"',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def process_brats_men_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = True
) -> bool:
    """
    Process one BraTS-MEN case across all 4 contrasts with per-label visualization.

    Uses same labels as MET (NETC, SNFH, ET).

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS-MEN case identifier
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons (default True for MEN)

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = MEN_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS-MEN case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in MEN_CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using brats_men_loader
        data = load_brats_men_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NETC pixels: {data['netc_mask'].sum()}")
        print(f"    SNFH pixels: {data['snfh_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks (MEN uses netc, snfh, et - same as MET)
        gt_masks = {
            'netc': data['netc_mask'],
            'snfh': data['snfh_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in MEN_LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2023_MEN" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        gt_multiclass = create_multiclass_overlay(
            image, data['netc_mask'], data['snfh_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = LABEL_COLORS[label]  # Same colors as MET
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['netc', 'snfh', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create individual 1x4 comparison figures for each label
        men_label_full_names = {
            'netc': 'NETC (Non-Enhancing Tumor Core)',
            'snfh': 'SNFH (Peritumoral Edema)',
            'et': 'ET (Enhancing Tumor)'
        }
        for label in ['netc', 'snfh', 'et']:
            prompt = MEN_LABEL_PROMPTS[label.upper()]
            # Create simple comparison figure
            gt_mask = gt_masks.get(label)
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            color = LABEL_COLORS[label]  # Same colors as MET

            # Create overlays
            if gt_mask is not None and gt_mask.sum() > 0:
                gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
            else:
                gt_overlay = image.copy()

            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            gt_pixels = gt_mask.sum() if gt_mask is not None else 0
            axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(sam3_overlay)
            sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
            axes[2].set_title(f"SAM3 Prediction\n({sam3_pixels} pixels)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(medsam3_overlay)
            medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
            axes[3].set_title(f"MedSAM3 Prediction\n({medsam3_pixels} pixels)", fontsize=12)
            axes[3].axis("off")

            # Legend
            legend_patches = [
                mpatches.Patch(color=np.array(color)/255, label='Ground Truth'),
                mpatches.Patch(color=np.array((255, 165, 0))/255, label='SAM3'),
                mpatches.Patch(color=np.array((0, 255, 255))/255, label='MedSAM3'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

            label_title = men_label_full_names.get(label, label.upper())
            plt.suptitle(f'BraTS2023_MEN - {case_id} - {contrast.upper()}\n'
                         f'{label_title}\nText Prompt: "{prompt}"',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.88])

            output_path = output_dir / f"comparison_{label}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Create multi-contrast grid figure
    if contrast_data:
        print(f"\n  Creating multi-contrast grid...")
        grid_output = output_base_dir / "BraTS2023_MEN" / case_id / "all_contrasts_comparison.png"

        # Colors
        gt_color = (0, 255, 0)
        sam3_color = (255, 0, 0)
        medsam3_color = (0, 0, 255)

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        contrast_labels = {
            't1n': 'T1 Native',
            't1c': 'T1 Contrast',
            't2w': 'T2 Weighted',
            't2f': 'T2 FLAIR'
        }

        for row, contrast in enumerate(MEN_CONTRASTS):
            if contrast not in contrast_data:
                for col in range(4):
                    axes[row, col].axis('off')
                continue

            data = contrast_data[contrast]
            image = data['image']
            gt_mask = data['gt_mask']
            sam3_pred = data.get('sam3_text_pred')
            medsam3_pred = data.get('medsam3_text_pred')

            # Create overlays
            gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

            if sam3_pred is not None:
                sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None:
                medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Plot row
            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(sam3_overlay)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(medsam3_overlay)
            axes[row, 3].axis('off')

            # Add column titles for first row
            if row == 0:
                axes[0, 0].set_title('Original', fontsize=14)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 2].set_title('SAM3 Text', fontsize=14)
                axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
            mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
            mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

        plt.suptitle(f'BraTS2023_MEN - {case_id}\nText Prompt: "Enhancing Tumor"',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def process_brats21_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = True
) -> bool:
    """
    Process one BraTS2021 case across all 4 contrasts with per-label visualization.

    BraTS2021 key differences:
    - Labels: NCR (1), ED (2), ET (4) - note ET is label 4!
    - Contrasts: t1, t1ce, t2, flair

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS2021 case identifier (e.g., "BraTS2021_00000")
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons (default True)

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = BRATS21_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS2021 case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in BRATS21_CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using brats21_loader
        data = load_brats21_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NCR pixels: {data['ncr_mask'].sum()}")
        print(f"    ED pixels: {data['ed_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks (BraTS2021 uses ncr, ed, et)
        gt_masks = {
            'ncr': data['ncr_mask'],
            'ed': data['ed_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in BRATS21_LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2021" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        # NCR=Red, ED=Green, ET=Blue
        gt_multiclass = create_multiclass_overlay(
            image, data['ncr_mask'], data['ed_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = BRATS21_LABEL_COLORS[label]
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['ncr', 'ed', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create individual 1x4 comparison figures for each label
        brats21_label_full_names = {
            'ncr': 'NCR (Necrotic/Non-Enhancing Tumor Core)',
            'ed': 'ED (Peritumoral Edema)',
            'et': 'ET (Enhancing Tumor - label 4)'
        }
        for label in ['ncr', 'ed', 'et']:
            prompt = BRATS21_LABEL_PROMPTS[label.upper()]
            # Create simple comparison figure
            gt_mask = gt_masks.get(label)
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            color = BRATS21_LABEL_COLORS[label]

            # Create overlays
            if gt_mask is not None and gt_mask.sum() > 0:
                gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
            else:
                gt_overlay = image.copy()

            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            gt_pixels = gt_mask.sum() if gt_mask is not None else 0
            axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(sam3_overlay)
            sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
            axes[2].set_title(f"SAM3 Prediction\n({sam3_pixels} pixels)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(medsam3_overlay)
            medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
            axes[3].set_title(f"MedSAM3 Prediction\n({medsam3_pixels} pixels)", fontsize=12)
            axes[3].axis("off")

            # Legend
            legend_patches = [
                mpatches.Patch(color=np.array(color)/255, label='Ground Truth'),
                mpatches.Patch(color=np.array((255, 165, 0))/255, label='SAM3'),
                mpatches.Patch(color=np.array((0, 255, 255))/255, label='MedSAM3'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

            label_title = brats21_label_full_names.get(label, label.upper())
            plt.suptitle(f'BraTS2021 - {case_id} - {contrast.upper()}\n'
                         f'{label_title}\nText Prompt: "{prompt}"',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 0.88])

            output_path = output_dir / f"comparison_{label}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Create multi-contrast grid figure
    if contrast_data:
        print(f"\n  Creating multi-contrast grid...")
        grid_output = output_base_dir / "BraTS2021" / case_id / "all_contrasts_comparison.png"

        # Colors
        gt_color = (0, 255, 0)
        sam3_color = (255, 0, 0)
        medsam3_color = (0, 0, 255)

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        # BraTS2021 contrast labels
        contrast_labels = {
            't1': 'T1',
            't1ce': 'T1 Contrast-Enhanced',
            't2': 'T2',
            'flair': 'FLAIR'
        }

        for row, contrast in enumerate(BRATS21_CONTRASTS):
            if contrast not in contrast_data:
                for col in range(4):
                    axes[row, col].axis('off')
                continue

            data = contrast_data[contrast]
            image = data['image']
            gt_mask = data['gt_mask']
            sam3_pred = data.get('sam3_text_pred')
            medsam3_pred = data.get('medsam3_text_pred')

            # Create overlays
            gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

            if sam3_pred is not None:
                sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None:
                medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Plot row
            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(sam3_overlay)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(medsam3_overlay)
            axes[row, 3].axis('off')

            # Add column titles for first row
            if row == 0:
                axes[0, 0].set_title('Original', fontsize=14)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 2].set_title('SAM3 Text', fontsize=14)
                axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
            mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
            mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

        plt.suptitle(f'BraTS2021 - {case_id}\nText Prompt: "Enhancing Tumor"',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def process_brats18_sample(
    sam3_model: SAM3Model,
    medsam3_model: SAM3Model,
    case_id: str,
    output_base_dir: Path,
    text_prompts: List[str] = None,
    text_only: bool = True
) -> bool:
    """
    Process one BraTS2018 case across all 4 contrasts with per-label visualization.

    BraTS2018 key differences from BraTS2021:
    - File extension: .nii (uncompressed)
    - HGG only (190 cases)
    - Labels: NCR (1), ED (2), ET (4) - same as BraTS2021

    Args:
        sam3_model: Base SAM3 model
        medsam3_model: Fine-tuned MedSAM3 model
        case_id: BraTS2018 case identifier (e.g., "Brats18_2013_10_1")
        output_base_dir: Base output directory
        text_prompts: List of text prompts to visualize (for combined mask)
        text_only: If True, skip box prompt comparisons (default True)

    Returns:
        True if successful, False otherwise
    """
    if text_prompts is None:
        text_prompts = BRATS18_TEXT_PROMPTS

    print(f"\n{'='*60}")
    print(f"Processing BraTS2018 case: {case_id}")
    print(f"{'='*60}")

    # Store data for all contrasts
    contrast_data = {}

    for contrast in BRATS18_CONTRASTS:
        print(f"\n  Processing contrast: {contrast.upper()}")

        # Load all label masks using brats18_loader
        data = load_brats18_all_labels(case_id, contrast)
        if data is None:
            print(f"    ERROR: Could not load sample")
            continue

        # Check if any tumor present
        if data['combined_mask'].sum() == 0:
            print(f"    WARNING: No tumor in middle slice, skipping")
            continue

        image = data['image']
        img_size = data['combined_mask'].shape

        print(f"    Image shape: {image.shape}")
        print(f"    NCR pixels: {data['ncr_mask'].sum()}")
        print(f"    ED pixels: {data['ed_mask'].sum()}")
        print(f"    ET pixels: {data['et_mask'].sum()}")

        # Run inference
        print(f"    Running SAM3 inference...")
        sam3_state = sam3_model.encode_image(image)

        print(f"    Running MedSAM3 inference...")
        medsam3_state = medsam3_model.encode_image(image)

        # Store GT masks (BraTS2018 uses ncr, ed, et - same as BraTS2021)
        gt_masks = {
            'ncr': data['ncr_mask'],
            'ed': data['ed_mask'],
            'et': data['et_mask'],
        }

        # Run per-label inference with label-specific prompts
        sam3_preds = {}
        medsam3_preds = {}

        for label, prompt in BRATS18_LABEL_PROMPTS.items():
            label_lower = label.lower()

            # SAM3 prediction
            sam3_pred = sam3_model.predict_text(sam3_state, prompt)
            if sam3_pred is not None and sam3_pred.shape != img_size:
                sam3_pred = resize_mask(sam3_pred, img_size)
            sam3_preds[label_lower] = sam3_pred

            # MedSAM3 prediction
            medsam3_pred = medsam3_model.predict_text(medsam3_state, prompt)
            if medsam3_pred is not None and medsam3_pred.shape != img_size:
                medsam3_pred = resize_mask(medsam3_pred, img_size)
            medsam3_preds[label_lower] = medsam3_pred

        # Store for multi-contrast grid
        contrast_data[contrast] = {
            'image': image,
            'gt_mask': data['combined_mask'],
            'gt_masks': gt_masks,
            'sam3_preds': sam3_preds,
            'medsam3_preds': medsam3_preds,
            'sam3_text_pred': sam3_preds.get('et'),  # Use ET as default
            'medsam3_text_pred': medsam3_preds.get('et'),
        }

        # Create output directory for this contrast
        output_dir = output_base_dir / "BraTS2018" / case_id / contrast
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # === Save individual GT label images ===
        print(f"    Saving per-label visualizations...")

        # Save original image
        Image.fromarray(image).save(output_dir / "original.png")

        # Save multiclass GT overlay (all 3 labels with distinct colors)
        # NCR=Red, ED=Green, ET=Blue
        gt_multiclass = create_multiclass_overlay(
            image, data['ncr_mask'], data['ed_mask'], data['et_mask']
        )
        Image.fromarray(gt_multiclass).save(output_dir / "gt_multiclass.png")

        # Save individual GT label overlays
        for label, mask in gt_masks.items():
            color = BRATS18_LABEL_COLORS[label]
            if mask.sum() > 0:
                overlay = create_overlay(image, mask, color, alpha=0.5)
            else:
                overlay = image.copy()
            Image.fromarray(overlay).save(output_dir / f"gt_{label}.png")
            # Save raw mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"gt_{label}.png")

        # Save SAM3/MedSAM3 predictions per label
        for label in ['ncr', 'ed', 'et']:
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            # SAM3 overlay
            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()
            Image.fromarray(sam3_overlay).save(output_dir / f"sam3_{label}.png")

            # MedSAM3 overlay
            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()
            Image.fromarray(medsam3_overlay).save(output_dir / f"medsam3_{label}.png")

            # Save raw prediction masks
            if sam3_pred is not None:
                Image.fromarray((sam3_pred * 255).astype(np.uint8)).save(masks_dir / f"sam3_{label}.png")
            if medsam3_pred is not None:
                Image.fromarray((medsam3_pred * 255).astype(np.uint8)).save(masks_dir / f"medsam3_{label}.png")

        # Create individual 1x4 comparison figures for each label
        brats18_label_full_names = {
            'ncr': 'NCR (Necrotic/Non-Enhancing Tumor Core)',
            'ed': 'ED (Peritumoral Edema)',
            'et': 'ET (Enhancing Tumor - label 4)'
        }
        for label in ['ncr', 'ed', 'et']:
            prompt = BRATS18_LABEL_PROMPTS[label.upper()]
            # Create simple comparison figure
            gt_mask = gt_masks.get(label)
            sam3_pred = sam3_preds.get(label)
            medsam3_pred = medsam3_preds.get(label)

            color = BRATS18_LABEL_COLORS[label]

            # Create overlays
            if gt_mask is not None and gt_mask.sum() > 0:
                gt_overlay = create_overlay(image, gt_mask, color, alpha=0.5)
            else:
                gt_overlay = image.copy()

            if sam3_pred is not None and sam3_pred.sum() > 0:
                sam3_overlay = create_overlay(image, sam3_pred, (255, 165, 0), alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None and medsam3_pred.sum() > 0:
                medsam3_overlay = create_overlay(image, medsam3_pred, (0, 255, 255), alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(gt_overlay)
            gt_pixels = gt_mask.sum() if gt_mask is not None else 0
            axes[1].set_title(f"Ground Truth\n({gt_pixels} pixels)", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(sam3_overlay)
            sam3_pixels = sam3_pred.sum() if sam3_pred is not None else 0
            axes[2].set_title(f"SAM3 Text\n({sam3_pixels} pixels)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(medsam3_overlay)
            medsam3_pixels = medsam3_pred.sum() if medsam3_pred is not None else 0
            axes[3].set_title(f"MedSAM3 Text\n({medsam3_pixels} pixels)", fontsize=12)
            axes[3].axis("off")

            plt.suptitle(f'BraTS2018 - {case_id} - {contrast.upper()}\n'
                         f'{brats18_label_full_names[label]}\n'
                         f'Text Prompt: "{prompt}"',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            fig_path = output_dir / f"comparison_{label}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()

    # === Create multi-contrast grid (4x4: 4 contrasts x 4 views) ===
    if len(contrast_data) > 0:
        print(f"\n  Creating multi-contrast grid visualization...")

        grid_output = output_base_dir / "BraTS2018" / case_id / "all_contrasts_comparison.png"

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))

        gt_color = (255, 0, 0)
        sam3_color = (255, 165, 0)
        medsam3_color = (0, 255, 255)

        contrast_labels = {
            't1': 'T1',
            't1ce': 'T1CE',
            't2': 'T2',
            'flair': 'FLAIR'
        }

        for row, contrast in enumerate(BRATS18_CONTRASTS):
            if contrast not in contrast_data:
                for col in range(4):
                    axes[row, col].axis('off')
                continue

            data = contrast_data[contrast]
            image = data['image']
            gt_mask = data['gt_mask']
            sam3_pred = data.get('sam3_text_pred')
            medsam3_pred = data.get('medsam3_text_pred')

            # Create overlays
            gt_overlay = create_overlay(image, gt_mask, gt_color, alpha=0.5)

            if sam3_pred is not None:
                sam3_overlay = create_overlay(image, sam3_pred, sam3_color, alpha=0.5)
            else:
                sam3_overlay = image.copy()

            if medsam3_pred is not None:
                medsam3_overlay = create_overlay(image, medsam3_pred, medsam3_color, alpha=0.5)
            else:
                medsam3_overlay = image.copy()

            # Plot row
            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(contrast_labels[contrast], fontsize=14, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(sam3_overlay)
            axes[row, 2].axis('off')

            axes[row, 3].imshow(medsam3_overlay)
            axes[row, 3].axis('off')

            # Add column titles for first row
            if row == 0:
                axes[0, 0].set_title('Original', fontsize=14)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 2].set_title('SAM3 Text', fontsize=14)
                axes[0, 3].set_title('MedSAM3 Text', fontsize=14)

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(gt_color)/255, label='Ground Truth'),
            mpatches.Patch(color=np.array(sam3_color)/255, label='SAM3'),
            mpatches.Patch(color=np.array(medsam3_color)/255, label='MedSAM3'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=12)

        plt.suptitle(f'BraTS2018 (HGG) - {case_id}\nText Prompt: "Enhancing Tumor"',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved multi-contrast grid to: {grid_output}")

    print(f"\n  Completed {case_id}")
    return len(contrast_data) > 0


def main():
    """Main function: load models, process all datasets."""
    parser = argparse.ArgumentParser(
        description="Generate visualization comparisons for SAM3 vs MedSAM3"
    )
    parser.add_argument(
        "--brats", action="store_true",
        help="Process BraTS2023_MET dataset"
    )
    parser.add_argument(
        "--max-cases", type=int, default=5,
        help="Maximum number of BraTS cases to visualize (default: 5)"
    )
    parser.add_argument(
        "--original", action="store_true",
        help="Process original 5 datasets (CHASE_DB1, STARE, etc.)"
    )
    parser.add_argument(
        "--all-labels", action="store_true",
        help="Only process BraTS cases where NETC, SNFH, and ET are all present"
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Only generate text prompt visualizations (skip box prompt comparisons)"
    )
    parser.add_argument(
        "--brats-ped", action="store_true",
        help="Process BraTS2023_PED (Pediatric) dataset"
    )
    parser.add_argument(
        "--brats-ssa", action="store_true",
        help="Process BraTS2023_SSA (Sub-Saharan Africa) dataset"
    )
    parser.add_argument(
        "--brats-men", action="store_true",
        help="Process BraTS2023_MEN (Meningioma) dataset"
    )
    parser.add_argument(
        "--brats21", action="store_true",
        help="Process BraTS2021 (Adult Glioma) dataset"
    )
    parser.add_argument(
        "--brats18", action="store_true",
        help="Process BraTS2018 (High-Grade Glioma) dataset"
    )
    args = parser.parse_args()

    # Default: if no flag specified, run original datasets
    if not args.brats and not args.original and not args.brats_ped and not args.brats_ssa and not args.brats_men and not args.brats21 and not args.brats18:
        args.original = True

    print("="*60)
    print("Medical Image Segmentation Visualization")
    print("SAM3 vs MedSAM3 Comparison")
    print("="*60)

    # Initialize SAM3 (base model from HuggingFace)
    print("\nLoading SAM3 (base model)...")
    sam3_model = SAM3Model(confidence_threshold=0.1)
    sam3_model.load_model()

    # Initialize MedSAM3 (fine-tuned model)
    print("\nLoading MedSAM3...")
    medsam3_checkpoint = download_medsam3_checkpoint()
    medsam3_model = SAM3Model(
        confidence_threshold=0.1,
        checkpoint_path=medsam3_checkpoint
    )
    medsam3_model.load_model()

    successful = 0
    total = 0

    # Process original datasets
    if args.original:
        datasets = list(DATASET_LOADERS.keys())
        print(f"\nOriginal datasets to process: {datasets}")

        for dataset_name in datasets:
            total += 1
            if process_dataset(sam3_model, medsam3_model, dataset_name, OUTPUT_DIR):
                successful += 1

    # Process BraTS dataset
    if args.brats:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2023_MET Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NETC, SNFH, ET)...")
            case_ids = get_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(case_ids)} cases with all 3 labels")
        else:
            case_ids = get_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(case_ids)} cases, targeting {args.max_cases} with tumor")

        brats_successful = 0
        for case_id in case_ids:
            if brats_successful >= args.max_cases:
                break

            total += 1
            if process_brats_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=BRATS_TEXT_PROMPTS,
                text_only=args.text_only
            ):
                successful += 1
                brats_successful += 1

    # Process BraTS-PED (Pediatric) dataset
    if args.brats_ped:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2023_PED (Pediatric) Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NC, ED, ET)...")
            ped_case_ids = get_ped_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(ped_case_ids)} cases with all 3 labels")
        else:
            ped_case_ids = get_ped_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(ped_case_ids)} cases, targeting {args.max_cases} with tumor")

        ped_successful = 0
        for case_id in ped_case_ids:
            if ped_successful >= args.max_cases:
                break

            total += 1
            if process_brats_ped_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=PED_TEXT_PROMPTS,
                text_only=True  # Always text-only for PED
            ):
                successful += 1
                ped_successful += 1

    # Process BraTS-SSA (Sub-Saharan Africa) dataset
    if args.brats_ssa:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2023_SSA (Sub-Saharan Africa) Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NC, ED, ET)...")
            ssa_case_ids = get_ssa_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(ssa_case_ids)} cases with all 3 labels")
        else:
            ssa_case_ids = get_ssa_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(ssa_case_ids)} cases, targeting {args.max_cases} with tumor")

        ssa_successful = 0
        for case_id in ssa_case_ids:
            if ssa_successful >= args.max_cases:
                break

            total += 1
            if process_brats_ssa_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=SSA_TEXT_PROMPTS,
                text_only=True  # Always text-only for SSA
            ):
                successful += 1
                ssa_successful += 1

    # Process BraTS-MEN (Meningioma) dataset
    if args.brats_men:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2023_MEN (Meningioma) Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NETC, SNFH, ET)...")
            men_case_ids = get_men_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(men_case_ids)} cases with all 3 labels")
        else:
            men_case_ids = get_men_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(men_case_ids)} cases, targeting {args.max_cases} with tumor")

        men_successful = 0
        for case_id in men_case_ids:
            if men_successful >= args.max_cases:
                break

            total += 1
            if process_brats_men_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=MEN_TEXT_PROMPTS,
                text_only=True  # Always text-only for MEN
            ):
                successful += 1
                men_successful += 1

    # Process BraTS2021 (Adult Glioma) dataset
    if args.brats21:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2021 (Adult Glioma) Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NCR, ED, ET)...")
            brats21_case_ids = get_brats21_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(brats21_case_ids)} cases with all 3 labels")
        else:
            brats21_case_ids = get_brats21_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(brats21_case_ids)} cases, targeting {args.max_cases} with tumor")

        brats21_successful = 0
        for case_id in brats21_case_ids:
            if brats21_successful >= args.max_cases:
                break

            total += 1
            if process_brats21_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=BRATS21_TEXT_PROMPTS,
                text_only=True  # Always text-only for BraTS2021
            ):
                successful += 1
                brats21_successful += 1

    # Process BraTS2018 (High-Grade Glioma) dataset
    if args.brats18:
        print(f"\n{'#'*60}")
        print("# Processing BraTS2018 (High-Grade Glioma) Dataset")
        print(f"{'#'*60}")

        # Get case IDs
        if args.all_labels:
            print(f"\nFinding cases with all 3 labels (NCR, ED, ET)...")
            brats18_case_ids = get_brats18_cases_with_all_labels(max_samples=args.max_cases)
            print(f"Found {len(brats18_case_ids)} cases with all 3 labels")
        else:
            brats18_case_ids = get_brats18_case_ids(max_samples=args.max_cases * 2, seed=42)
            print(f"Sampling from {len(brats18_case_ids)} cases, targeting {args.max_cases} with tumor")

        brats18_successful = 0
        for case_id in brats18_case_ids:
            if brats18_successful >= args.max_cases:
                break

            total += 1
            if process_brats18_sample(
                sam3_model, medsam3_model, case_id, OUTPUT_DIR,
                text_prompts=BRATS18_TEXT_PROMPTS,
                text_only=True  # Always text-only for BraTS2018
            ):
                successful += 1
                brats18_successful += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed {successful}/{total} cases successfully")
    print(f"Output directory: {OUTPUT_DIR}")

    # List generated files for original datasets
    if args.original:
        print("\nOriginal datasets:")
        for dataset_name in DATASET_LOADERS.keys():
            dataset_dir = OUTPUT_DIR / dataset_name
            if dataset_dir.exists():
                files = sorted(dataset_dir.glob("*.png"))
                print(f"\n  {dataset_name}/")
                for f in files:
                    print(f"    - {f.name}")

    # List generated files for BraTS
    if args.brats:
        brats_dir = OUTPUT_DIR / "BraTS2023_MET"
        if brats_dir.exists():
            print("\n  BraTS2023_MET/")
            for case_dir in sorted(brats_dir.iterdir()):
                if case_dir.is_dir():
                    print(f"    {case_dir.name}/")
                    for contrast_dir in sorted(case_dir.iterdir()):
                        if contrast_dir.is_dir():
                            n_files = len(list(contrast_dir.glob("*.png")))
                            print(f"      {contrast_dir.name}/ ({n_files} images)")

    # List generated files for BraTS-PED
    if args.brats_ped:
        ped_dir = OUTPUT_DIR / "BraTS2023_PED"
        if ped_dir.exists():
            print("\n  BraTS2023_PED/")
            for case_dir in sorted(ped_dir.iterdir()):
                if case_dir.is_dir():
                    print(f"    {case_dir.name}/")
                    for contrast_dir in sorted(case_dir.iterdir()):
                        if contrast_dir.is_dir():
                            n_files = len(list(contrast_dir.glob("*.png")))
                            print(f"      {contrast_dir.name}/ ({n_files} images)")

    # List generated files for BraTS-SSA
    if args.brats_ssa:
        ssa_dir = OUTPUT_DIR / "BraTS2023_SSA"
        if ssa_dir.exists():
            print("\n  BraTS2023_SSA/")
            for case_dir in sorted(ssa_dir.iterdir()):
                if case_dir.is_dir():
                    print(f"    {case_dir.name}/")
                    for contrast_dir in sorted(case_dir.iterdir()):
                        if contrast_dir.is_dir():
                            n_files = len(list(contrast_dir.glob("*.png")))
                            print(f"      {contrast_dir.name}/ ({n_files} images)")

    # List generated files for BraTS-MEN
    if args.brats_men:
        men_dir = OUTPUT_DIR / "BraTS2023_MEN"
        if men_dir.exists():
            print("\n  BraTS2023_MEN/")
            for case_dir in sorted(men_dir.iterdir()):
                if case_dir.is_dir():
                    print(f"    {case_dir.name}/")
                    for contrast_dir in sorted(case_dir.iterdir()):
                        if contrast_dir.is_dir():
                            n_files = len(list(contrast_dir.glob("*.png")))
                            print(f"      {contrast_dir.name}/ ({n_files} images)")

    # List generated files for BraTS2021
    if args.brats21:
        brats21_dir = OUTPUT_DIR / "BraTS2021"
        if brats21_dir.exists():
            print("\n  BraTS2021/")
            for case_dir in sorted(brats21_dir.iterdir()):
                if case_dir.is_dir():
                    print(f"    {case_dir.name}/")
                    for contrast_dir in sorted(case_dir.iterdir()):
                        if contrast_dir.is_dir():
                            n_files = len(list(contrast_dir.glob("*.png")))
                            print(f"      {contrast_dir.name}/ ({n_files} images)")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
