"""
BraTS2023_MET dataset loader for brain metastases MRI segmentation.
Handles NIfTI 3D volumes and extracts 2D slices for SAM3/MedSAM3 evaluation.
"""

import os
import random
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from PIL import Image


# Dataset configuration
BRATS_ROOT = Path("/shared/BIOE486/SP25/dataset/BrainMVP/BrainMVP-ds/BraTS2023_MET/BraTS2023_MET_Training")

# MRI contrasts available
CONTRASTS = ["t1n", "t1c", "t2w", "t2f"]

# Tumor labels in segmentation mask
TUMOR_LABELS = {
    0: "Background",
    1: "NETC",   # Non-Enhancing Tumor Core
    2: "SNFH",   # Surrounding Non-enhancing FLAIR Hyperintensity (Edema)
    3: "ET",     # Enhancing Tumor
}

# Text prompts for brain metastases segmentation
BRATS_PROMPTS = {
    "combined": "Brain Tumor",
    "NETC": "Non-Enhancing Tumor Core",
    "SNFH": "Peritumoral Edema",
    "ET": "Enhancing Tumor",
}

# Label-specific prompts for per-class evaluation
LABEL_PROMPTS = {
    "NETC": "Non-Enhancing Tumor Core",
    "SNFH": "Peritumoral Edema",
    "ET": "Enhancing Tumor",
}

# Standard text prompts to evaluate
STANDARD_TEXT_PROMPTS = [
    "Brain Tumor",
    "Brain Metastasis",
    "Enhancing Tumor",
    "Tumor",
]


@dataclass
class BraTSSample:
    """A single BraTS dataset sample (2D slice)."""
    image: np.ndarray          # RGB image (H, W, 3)
    gt_mask: np.ndarray        # Binary mask (H, W)
    case_id: str               # e.g., "BraTS-MET-00002-000"
    contrast: str              # e.g., "t1c"
    slice_idx: int             # Slice index in volume
    text_prompt: str           # Text prompt for segmentation
    tumor_label: Optional[str] # Label name: "combined", "NETC", "SNFH", "ET"


def load_nifti_volume(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return the 3D volume.

    Args:
        nifti_path: Path to .nii.gz file

    Returns:
        3D numpy array (H, W, D)
    """
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata()
    return volume


def normalize_mri_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize MRI slice to 0-255 range and convert to RGB.

    Args:
        slice_2d: 2D MRI slice (H, W)

    Returns:
        RGB image (H, W, 3) as uint8
    """
    # Handle empty or constant slices
    if slice_2d.max() == slice_2d.min():
        slice_uint8 = np.zeros(slice_2d.shape, dtype=np.uint8)
    else:
        # Clip outliers (1st and 99th percentile)
        p1, p99 = np.percentile(slice_2d, [1, 99])
        slice_clipped = np.clip(slice_2d, p1, p99)

        # Normalize to 0-255
        slice_normalized = (slice_clipped - p1) / (p99 - p1)
        slice_uint8 = (slice_normalized * 255).astype(np.uint8)

    # Convert grayscale to RGB (SAM3 expects RGB input)
    rgb_image = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1)

    return rgb_image


def get_middle_slice_idx(volume: np.ndarray) -> int:
    """Get the middle slice index along the axial (z) axis."""
    return volume.shape[2] // 2


def get_tumor_slices(seg_volume: np.ndarray, min_tumor_pixels: int = 100) -> List[int]:
    """
    Find slices that contain tumor (non-zero mask values).

    Args:
        seg_volume: 3D segmentation volume
        min_tumor_pixels: Minimum number of tumor pixels required

    Returns:
        List of slice indices containing tumor
    """
    tumor_slices = []
    for z in range(seg_volume.shape[2]):
        slice_mask = seg_volume[:, :, z]
        if np.sum(slice_mask > 0) >= min_tumor_pixels:
            tumor_slices.append(z)
    return tumor_slices


def get_tumor_slices_for_label(
    seg_volume: np.ndarray,
    label_id: int,
    min_pixels: int = 50
) -> List[int]:
    """
    Find slices containing a specific tumor label.

    Args:
        seg_volume: 3D segmentation volume with label values 0-3
        label_id: Specific label to look for (1=NETC, 2=SNFH, 3=ET)
        min_pixels: Minimum number of pixels required for that label

    Returns:
        List of slice indices containing the specified label
    """
    slices = []
    for z in range(seg_volume.shape[2]):
        slice_mask = seg_volume[:, :, z]
        if (slice_mask == label_id).sum() >= min_pixels:
            slices.append(z)
    return slices


def get_case_ids(
    max_samples: Optional[int] = None,
    sample_ratio: float = 0.2,
    seed: int = 42
) -> List[str]:
    """
    Get list of case IDs, optionally sampling a subset.

    Args:
        max_samples: Maximum number of samples (overrides sample_ratio)
        sample_ratio: Fraction of cases to sample (default 20%)
        seed: Random seed for reproducibility

    Returns:
        List of case IDs
    """
    all_cases = sorted([d.name for d in BRATS_ROOT.iterdir() if d.is_dir()])

    if max_samples is not None:
        n_samples = min(max_samples, len(all_cases))
    else:
        n_samples = int(len(all_cases) * sample_ratio)

    random.seed(seed)
    sampled_cases = random.sample(all_cases, n_samples)

    return sorted(sampled_cases)


def create_tumor_mask(
    seg_slice: np.ndarray,
    tumor_label: Optional[str] = None
) -> np.ndarray:
    """
    Create binary tumor mask from segmentation.

    Args:
        seg_slice: 2D segmentation slice with label values 0-3
        tumor_label: "NETC", "SNFH", "ET", or None for combined

    Returns:
        Binary mask (H, W) as uint8
    """
    if tumor_label is None or tumor_label == "combined":
        # Combined: all tumor labels (1, 2, 3)
        return (seg_slice > 0).astype(np.uint8)
    elif tumor_label == "NETC":
        return (seg_slice == 1).astype(np.uint8)
    elif tumor_label == "SNFH":
        return (seg_slice == 2).astype(np.uint8)
    elif tumor_label == "ET":
        return (seg_slice == 3).astype(np.uint8)
    else:
        raise ValueError(f"Unknown tumor label: {tumor_label}")


def load_brats_sample(
    case_id: str,
    contrast: str = "t1c",
    tumor_label: Optional[str] = None,
    text_prompt: str = "Brain Tumor",
    slice_idx: Optional[int] = None
) -> Optional[BraTSSample]:
    """
    Load a single sample from a BraTS case.

    Args:
        case_id: Case identifier
        contrast: MRI contrast type (t1n, t1c, t2w, t2f)
        tumor_label: Tumor label to extract (None = combined)
        text_prompt: Text prompt for segmentation
        slice_idx: Specific slice index (None = middle slice)

    Returns:
        BraTSSample or None if loading fails
    """
    case_dir = BRATS_ROOT / case_id

    # Load MRI volume
    mri_path = case_dir / f"{case_id}-{contrast}.nii.gz"
    if not mri_path.exists():
        return None

    mri_volume = load_nifti_volume(mri_path)

    # Load segmentation
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if not seg_path.exists():
        return None

    seg_volume = load_nifti_volume(seg_path)

    # Get slice index
    if slice_idx is None:
        slice_idx = get_middle_slice_idx(mri_volume)

    # Extract 2D slice
    mri_slice = mri_volume[:, :, slice_idx]
    seg_slice = seg_volume[:, :, slice_idx]

    # Normalize MRI to RGB
    image = normalize_mri_slice(mri_slice)

    # Create tumor mask
    gt_mask = create_tumor_mask(seg_slice, tumor_label)

    return BraTSSample(
        image=image,
        gt_mask=gt_mask,
        case_id=case_id,
        contrast=contrast,
        slice_idx=slice_idx,
        text_prompt=text_prompt,
        tumor_label=tumor_label if tumor_label else "combined"
    )


def load_brats_all_labels(
    case_id: str,
    contrast: str = "t1c",
    slice_idx: Optional[int] = None
) -> Optional[dict]:
    """
    Load image and all three label masks for a single slice.

    Args:
        case_id: Case identifier
        contrast: MRI contrast type (t1n, t1c, t2w, t2f)
        slice_idx: Specific slice index (None = middle slice)

    Returns:
        dict with keys:
            'image': RGB image (H, W, 3)
            'seg_slice': Raw segmentation slice with label values 0-3
            'netc_mask': Binary mask for NETC (label 1)
            'snfh_mask': Binary mask for SNFH (label 2)
            'et_mask': Binary mask for ET (label 3)
            'combined_mask': Binary mask for all tumor (labels 1-3)
            'case_id': Case identifier
            'contrast': MRI contrast
            'slice_idx': Slice index
        Returns None if loading fails
    """
    case_dir = BRATS_ROOT / case_id

    # Load MRI volume
    mri_path = case_dir / f"{case_id}-{contrast}.nii.gz"
    if not mri_path.exists():
        return None

    mri_volume = load_nifti_volume(mri_path)

    # Load segmentation
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if not seg_path.exists():
        return None

    seg_volume = load_nifti_volume(seg_path)

    # Get slice index
    if slice_idx is None:
        slice_idx = get_middle_slice_idx(mri_volume)

    # Extract 2D slice
    mri_slice = mri_volume[:, :, slice_idx]
    seg_slice = seg_volume[:, :, slice_idx]

    # Normalize MRI to RGB
    image = normalize_mri_slice(mri_slice)

    # Create individual label masks
    netc_mask = (seg_slice == 1).astype(np.uint8)
    snfh_mask = (seg_slice == 2).astype(np.uint8)
    et_mask = (seg_slice == 3).astype(np.uint8)
    combined_mask = (seg_slice > 0).astype(np.uint8)

    return {
        'image': image,
        'seg_slice': seg_slice,
        'netc_mask': netc_mask,
        'snfh_mask': snfh_mask,
        'et_mask': et_mask,
        'combined_mask': combined_mask,
        'case_id': case_id,
        'contrast': contrast,
        'slice_idx': slice_idx,
    }


def get_cases_with_all_labels(max_samples: int = 10) -> List[str]:
    """
    Find BraTS cases where NETC, SNFH, and ET are all present in the middle slice.

    Args:
        max_samples: Maximum number of cases to return

    Returns:
        List of case IDs with all 3 labels present in middle slice
    """
    all_cases = sorted([d.name for d in BRATS_ROOT.iterdir() if d.is_dir()])
    valid_cases = []

    print(f"Scanning {len(all_cases)} cases for slices with all 3 labels...")

    for case_id in all_cases:
        # Load segmentation and check middle slice
        seg_path = BRATS_ROOT / case_id / f"{case_id}-seg.nii.gz"
        if not seg_path.exists():
            continue

        seg_volume = load_nifti_volume(seg_path)
        mid_slice = seg_volume[:, :, seg_volume.shape[2] // 2]

        # Check if all 3 labels present
        has_netc = (mid_slice == 1).sum() > 0
        has_snfh = (mid_slice == 2).sum() > 0
        has_et = (mid_slice == 3).sum() > 0

        if has_netc and has_snfh and has_et:
            valid_cases.append(case_id)
            print(f"  Found: {case_id} (NETC={int((mid_slice==1).sum())}, "
                  f"SNFH={int((mid_slice==2).sum())}, ET={int((mid_slice==3).sum())})")

        if len(valid_cases) >= max_samples:
            break

    print(f"Found {len(valid_cases)} cases with all 3 labels")
    return valid_cases


def load_brats_dataset(
    contrasts: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    sample_ratio: float = 0.2,
    tumor_label: Optional[str] = None,
    text_prompt: str = "Brain Tumor",
    require_tumor: bool = True,
    seed: int = 42
) -> Iterator[BraTSSample]:
    """
    Generator that yields BraTS samples for middle slices.

    Args:
        contrasts: List of contrasts to load (default: all 4)
        max_samples: Maximum number of cases to sample
        sample_ratio: Fraction of cases to sample (default 20%)
        tumor_label: Specific tumor label (None = combined)
        text_prompt: Text prompt for segmentation
        require_tumor: Skip slices without tumor in mask
        seed: Random seed for reproducibility

    Yields:
        BraTSSample objects
    """
    if contrasts is None:
        contrasts = CONTRASTS

    case_ids = get_case_ids(max_samples, sample_ratio, seed)

    for case_id in case_ids:
        for contrast in contrasts:
            sample = load_brats_sample(
                case_id, contrast, tumor_label, text_prompt
            )

            if sample is None:
                continue

            # Skip if no tumor and require_tumor is True
            if require_tumor and sample.gt_mask.sum() == 0:
                continue

            yield sample


def load_brats_by_contrast(
    contrast: str,
    max_samples: Optional[int] = None,
    sample_ratio: float = 0.2,
    seed: int = 42
) -> Iterator[BraTSSample]:
    """Load BraTS samples for a single contrast."""
    return load_brats_dataset(
        contrasts=[contrast],
        max_samples=max_samples,
        sample_ratio=sample_ratio,
        text_prompt="Brain Tumor",
        seed=seed
    )


def load_brats_by_label(
    tumor_label: str,
    max_samples: Optional[int] = None,
    sample_ratio: float = 0.2,
    seed: int = 42
) -> Iterator[BraTSSample]:
    """Load BraTS samples for a specific tumor label."""
    prompt = BRATS_PROMPTS.get(tumor_label, "Brain Tumor")
    return load_brats_dataset(
        max_samples=max_samples,
        sample_ratio=sample_ratio,
        tumor_label=tumor_label,
        text_prompt=prompt,
        seed=seed
    )


def get_dataset_info() -> dict:
    """Get information about the BraTS dataset."""
    all_cases = [d.name for d in BRATS_ROOT.iterdir() if d.is_dir()]

    return {
        "name": "BraTS2023_MET",
        "total_cases": len(all_cases),
        "contrasts": CONTRASTS,
        "tumor_labels": TUMOR_LABELS,
        "root_path": str(BRATS_ROOT),
    }


if __name__ == "__main__":
    # Test loading
    print("Testing BraTS2023_MET dataset loader...")
    print("=" * 60)

    # Dataset info
    info = get_dataset_info()
    print(f"\nDataset: {info['name']}")
    print(f"Total cases: {info['total_cases']}")
    print(f"Contrasts: {info['contrasts']}")
    print(f"Root path: {info['root_path']}")

    # Test loading samples
    print("\n" + "-" * 60)
    print("Testing sample loading (2 cases, all contrasts)...")

    case_ids = get_case_ids(max_samples=2, seed=42)
    print(f"Sampled cases: {case_ids}")

    for case_id in case_ids:
        print(f"\n{case_id}:")
        for contrast in CONTRASTS:
            sample = load_brats_sample(case_id, contrast)
            if sample:
                print(f"  {contrast}: image={sample.image.shape}, "
                      f"mask_sum={sample.gt_mask.sum()}, "
                      f"slice={sample.slice_idx}")
            else:
                print(f"  {contrast}: FAILED to load")

    # Test per-label loading
    print("\n" + "-" * 60)
    print("Testing per-label loading...")

    for label in ["combined", "NETC", "SNFH", "ET"]:
        samples = list(load_brats_by_label(label, max_samples=2))
        n_with_tumor = sum(1 for s in samples if s.gt_mask.sum() > 0)
        print(f"  {label}: {len(samples)} samples, {n_with_tumor} with tumor")

    print("\n" + "=" * 60)
    print("BraTS loader test complete!")
