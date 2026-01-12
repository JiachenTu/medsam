#!/usr/bin/env python3
"""
Visualization script for medical image segmentation results.

Generates comparison visualizations for all datasets:
- CHASE_DB1, STARE, CVC-ClinicDB, ETIS-Larib, PH2

For each dataset, creates:
1. Individual images: original, GT, SAM3/MedSAM3 predictions (text & box prompts)
2. Combined 1x4 comparison layouts

Usage:
    conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3
    python visualize_all_datasets.py
"""

import sys
from pathlib import Path

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


def main():
    """Main function: load models, process all datasets."""
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

    # Process each dataset
    datasets = list(DATASET_LOADERS.keys())
    print(f"\nDatasets to process: {datasets}")

    successful = 0
    for dataset_name in datasets:
        if process_dataset(sam3_model, medsam3_model, dataset_name, OUTPUT_DIR):
            successful += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed {successful}/{len(datasets)} datasets successfully")
    print(f"Output directory: {OUTPUT_DIR}")

    # List generated files
    print("\nGenerated files:")
    for dataset_name in datasets:
        dataset_dir = OUTPUT_DIR / dataset_name
        if dataset_dir.exists():
            files = sorted(dataset_dir.glob("*.png"))
            print(f"\n  {dataset_name}/")
            for f in files:
                print(f"    - {f.name}")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
