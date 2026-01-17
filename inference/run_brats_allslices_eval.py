#!/usr/bin/env python3
"""
Multi-GPU parallel evaluation of BraTS2023_MET on ALL slices.

Evaluates SAM3 and MedSAM3 on 5 cases with all 3 tumor labels across
CUDA devices 1, 2, 4, 5, 6 in parallel.

Usage:
    conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3
    python run_brats_allslices_eval.py

Features:
- All slices with tumor content (not just middle)
- All 4 MRI contrasts (T1N, T1C, T2W, T2F)
- All 3 tumor labels (NETC, SNFH, ET) with label-specific prompts
- Both box prompts and text prompts
- Multi-GPU parallelization
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


# Configuration
MEDSAM3_CHECKPOINT = "/srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt"
OUTPUT_DIR = Path(__file__).parent / "results" / "brats_allslices"

# Target cases with all 3 labels present
TARGET_CASES = [
    "BraTS-MET-00002-000",
    "BraTS-MET-00003-000",
    "BraTS-MET-00006-000",
    "BraTS-MET-00009-000",
    "BraTS-MET-00011-000"
]

# Available GPUs
AVAILABLE_GPUS = [1, 2, 4, 5, 6]

# Label configuration
LABEL_CONFIG = {
    1: {"name": "NETC", "prompt": "Non-Enhancing Tumor Core"},
    2: {"name": "SNFH", "prompt": "Peritumoral Edema"},
    3: {"name": "ET", "prompt": "Enhancing Tumor"},
}

CONTRASTS = ["t1n", "t1c", "t2w", "t2f"]


def evaluate_case_worker(case_id: str, gpu_id: int, output_dir: Path):
    """
    Worker function to evaluate a single case on assigned GPU.

    Evaluates both SAM3 and MedSAM3 on all slices, contrasts, and labels.
    """
    # Set CUDA device - MUST be done before importing torch/loading model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now import torch-dependent modules
    import torch
    from brats_loader import (
        load_nifti_volume, normalize_mri_slice, create_tumor_mask,
        get_tumor_slices, BRATS_ROOT, LABEL_PROMPTS
    )
    from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask
    from metrics import compute_all_metrics

    print(f"[GPU {gpu_id}] Starting evaluation for {case_id}")
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    case_dir = BRATS_ROOT / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation volume to find all tumor slices
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    seg_volume = load_nifti_volume(seg_path)

    # Find ALL slices with any tumor (combined)
    all_tumor_slices = get_tumor_slices(seg_volume, min_tumor_pixels=50)
    print(f"[GPU {gpu_id}] {case_id}: Found {len(all_tumor_slices)} slices with tumor")

    def evaluate_with_model(model: SAM3Model, model_name: str) -> List[Dict]:
        """Evaluate all slices with given model."""
        results = []

        for contrast in CONTRASTS:
            # Load MRI volume
            mri_path = case_dir / f"{case_id}-{contrast}.nii.gz"
            if not mri_path.exists():
                print(f"[GPU {gpu_id}] Warning: {mri_path} not found, skipping")
                continue

            mri_volume = load_nifti_volume(mri_path)

            for slice_idx in tqdm(all_tumor_slices,
                                  desc=f"[GPU {gpu_id}] {model_name} {contrast}",
                                  leave=False):
                # Get slice data
                mri_slice = mri_volume[:, :, slice_idx]
                seg_slice = seg_volume[:, :, slice_idx]
                image = normalize_mri_slice(mri_slice)
                img_size = image.shape[:2]

                # Encode image once
                inference_state = model.encode_image(image)

                # Evaluate each label
                for label_id, label_info in LABEL_CONFIG.items():
                    label_name = label_info["name"]
                    text_prompt = label_info["prompt"]

                    # Create label-specific mask
                    gt_mask = (seg_slice == label_id).astype(np.uint8)

                    # Skip if label not present in this slice
                    if gt_mask.sum() < 10:
                        continue

                    result_base = {
                        "case_id": case_id,
                        "contrast": contrast,
                        "slice_idx": slice_idx,
                        "label": label_name,
                        "label_id": label_id,
                        "model": model_name,
                        "gt_pixels": int(gt_mask.sum()),
                    }

                    # === Box Prompt Evaluation ===
                    bbox = generate_bbox_from_mask(gt_mask)
                    if bbox is not None:
                        pred_box = model.predict_box(inference_state, bbox, img_size)
                        if pred_box is not None:
                            if pred_box.shape != img_size:
                                pred_box = resize_mask(pred_box, img_size)
                            metrics = compute_all_metrics(pred_box, gt_mask)
                            results.append({
                                **result_base,
                                "prompt_type": "box",
                                "text_prompt": None,
                                "dice": metrics.dice,
                                "iou": metrics.iou,
                                "precision": metrics.precision,
                                "recall": metrics.recall,
                            })

                    # === Text Prompt Evaluation ===
                    pred_text = model.predict_text(inference_state, text_prompt)
                    if pred_text is not None:
                        if pred_text.shape != img_size:
                            pred_text = resize_mask(pred_text, img_size)
                        metrics = compute_all_metrics(pred_text, gt_mask)
                    else:
                        # No prediction
                        metrics = type('obj', (object,), {
                            'dice': 0.0, 'iou': 0.0,
                            'precision': 0.0, 'recall': 0.0
                        })()

                    results.append({
                        **result_base,
                        "prompt_type": "text",
                        "text_prompt": text_prompt,
                        "dice": metrics.dice,
                        "iou": metrics.iou,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                    })

        return results

    # === Evaluate SAM3 ===
    print(f"[GPU {gpu_id}] Loading SAM3...")
    sam3 = SAM3Model(confidence_threshold=0.1)
    sam3_results = evaluate_with_model(sam3, "sam3")

    # Save SAM3 results
    sam3_output = output_dir / f"{case_id}_sam3.json"
    with open(sam3_output, "w") as f:
        json.dump(sam3_results, f, indent=2)
    print(f"[GPU {gpu_id}] Saved SAM3 results: {len(sam3_results)} entries -> {sam3_output}")

    # Free GPU memory
    del sam3
    import torch
    torch.cuda.empty_cache()

    # === Evaluate MedSAM3 ===
    print(f"[GPU {gpu_id}] Loading MedSAM3...")
    medsam3 = SAM3Model(confidence_threshold=0.1, checkpoint_path=MEDSAM3_CHECKPOINT)
    medsam3_results = evaluate_with_model(medsam3, "medsam3")

    # Save MedSAM3 results
    medsam3_output = output_dir / f"{case_id}_medsam3.json"
    with open(medsam3_output, "w") as f:
        json.dump(medsam3_results, f, indent=2)
    print(f"[GPU {gpu_id}] Saved MedSAM3 results: {len(medsam3_results)} entries -> {medsam3_output}")

    del medsam3
    torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Completed {case_id}")


def aggregate_results(output_dir: Path):
    """Aggregate results from all worker JSON files."""
    print("\n" + "=" * 60)
    print("Aggregating Results")
    print("=" * 60)

    raw_dir = output_dir / "raw"
    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_results = []
    for json_file in raw_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            all_results.extend(data)

    if not all_results:
        print("ERROR: No results found!")
        return

    df = pd.DataFrame(all_results)
    print(f"Total results: {len(df)}")

    # Save all results CSV
    all_csv = agg_dir / "all_slices_results.csv"
    df.to_csv(all_csv, index=False)
    print(f"Saved: {all_csv}")

    # === Summary by Case ===
    case_summary = df.groupby(["case_id", "model", "prompt_type"]).agg({
        "dice": ["mean", "std", "count"],
        "iou": "mean"
    }).round(4)
    case_summary.columns = ["dice_mean", "dice_std", "n_samples", "iou_mean"]
    case_summary = case_summary.reset_index()
    case_csv = agg_dir / "by_case_summary.csv"
    case_summary.to_csv(case_csv, index=False)
    print(f"Saved: {case_csv}")

    # === Summary by Label ===
    label_summary = df.groupby(["label", "model", "prompt_type"]).agg({
        "dice": ["mean", "std", "count"],
        "iou": "mean"
    }).round(4)
    label_summary.columns = ["dice_mean", "dice_std", "n_samples", "iou_mean"]
    label_summary = label_summary.reset_index()
    label_csv = agg_dir / "by_label_summary.csv"
    label_summary.to_csv(label_csv, index=False)
    print(f"Saved: {label_csv}")

    # === Summary by Contrast ===
    contrast_summary = df.groupby(["contrast", "model", "prompt_type"]).agg({
        "dice": ["mean", "std", "count"],
        "iou": "mean"
    }).round(4)
    contrast_summary.columns = ["dice_mean", "dice_std", "n_samples", "iou_mean"]
    contrast_summary = contrast_summary.reset_index()
    contrast_csv = agg_dir / "by_contrast_summary.csv"
    contrast_summary.to_csv(contrast_csv, index=False)
    print(f"Saved: {contrast_csv}")

    # === Box vs Text Comparison ===
    box_text = df.groupby(["model", "prompt_type", "label"]).agg({
        "dice": "mean",
        "iou": "mean"
    }).round(4).reset_index()
    box_text_csv = agg_dir / "box_vs_text_comparison.csv"
    box_text.to_csv(box_text_csv, index=False)
    print(f"Saved: {box_text_csv}")

    # === Generate Markdown Report ===
    generate_report(df, output_dir)


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive markdown report."""
    report_path = output_dir / "report.md"

    with open(report_path, "w") as f:
        f.write("# BraTS2023_MET All-Slices Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Total evaluations**: {len(df)}\n\n")

        # Overview stats
        n_cases = df["case_id"].nunique()
        n_slices = df.groupby("case_id")["slice_idx"].nunique().sum()
        f.write(f"**Cases**: {n_cases}\n\n")
        f.write(f"**Total slices evaluated**: {n_slices}\n\n")

        # Box vs Text Overall
        f.write("## Box vs Text Prompt Performance (Overall)\n\n")
        f.write("| Model | Prompt | Dice (mean) | IoU (mean) | N |\n")
        f.write("|-------|--------|-------------|------------|---|\n")

        for model in ["sam3", "medsam3"]:
            for prompt_type in ["box", "text"]:
                subset = df[(df["model"] == model) & (df["prompt_type"] == prompt_type)]
                if len(subset) > 0:
                    dice = subset["dice"].mean() * 100
                    iou = subset["iou"].mean() * 100
                    n = len(subset)
                    f.write(f"| {model} | {prompt_type} | {dice:.1f}% | {iou:.1f}% | {n} |\n")

        # By Label
        f.write("\n## Performance by Tumor Label\n\n")
        f.write("| Label | Model | Box Dice | Text Dice | Box-Text Gap |\n")
        f.write("|-------|-------|----------|-----------|-------------|\n")

        for label in ["NETC", "SNFH", "ET"]:
            for model in ["sam3", "medsam3"]:
                box_subset = df[(df["model"] == model) & (df["prompt_type"] == "box") & (df["label"] == label)]
                text_subset = df[(df["model"] == model) & (df["prompt_type"] == "text") & (df["label"] == label)]

                box_dice = box_subset["dice"].mean() * 100 if len(box_subset) > 0 else 0
                text_dice = text_subset["dice"].mean() * 100 if len(text_subset) > 0 else 0
                gap = box_dice - text_dice

                f.write(f"| {label} | {model} | {box_dice:.1f}% | {text_dice:.1f}% | {gap:+.1f}% |\n")

        # By Contrast
        f.write("\n## Performance by MRI Contrast\n\n")
        f.write("| Contrast | Model | Box Dice | Text Dice |\n")
        f.write("|----------|-------|----------|----------|\n")

        for contrast in ["t1n", "t1c", "t2w", "t2f"]:
            for model in ["sam3", "medsam3"]:
                box_subset = df[(df["model"] == model) & (df["prompt_type"] == "box") & (df["contrast"] == contrast)]
                text_subset = df[(df["model"] == model) & (df["prompt_type"] == "text") & (df["contrast"] == contrast)]

                box_dice = box_subset["dice"].mean() * 100 if len(box_subset) > 0 else 0
                text_dice = text_subset["dice"].mean() * 100 if len(text_subset) > 0 else 0

                f.write(f"| {contrast.upper()} | {model} | {box_dice:.1f}% | {text_dice:.1f}% |\n")

        # Label-specific prompts used
        f.write("\n## Label-Specific Text Prompts\n\n")
        f.write("| Label | Text Prompt Used |\n")
        f.write("|-------|------------------|\n")
        for label_id, info in LABEL_CONFIG.items():
            f.write(f"| {info['name']} | \"{info['prompt']}\" |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Best performing combinations
        best_box = df[df["prompt_type"] == "box"].groupby(["model", "label", "contrast"])["dice"].mean()
        if len(best_box) > 0:
            best_idx = best_box.idxmax()
            best_val = best_box.max() * 100
            f.write(f"- **Best box prompt**: {best_idx[0]} on {best_idx[1]} ({best_idx[2].upper()}) = {best_val:.1f}% Dice\n")

        best_text = df[df["prompt_type"] == "text"].groupby(["model", "label", "contrast"])["dice"].mean()
        if len(best_text) > 0:
            best_idx = best_text.idxmax()
            best_val = best_text.max() * 100
            f.write(f"- **Best text prompt**: {best_idx[0]} on {best_idx[1]} ({best_idx[2].upper()}) = {best_val:.1f}% Dice\n")

        # MedSAM3 vs SAM3 improvement
        sam3_text_mean = df[(df["model"] == "sam3") & (df["prompt_type"] == "text")]["dice"].mean()
        medsam3_text_mean = df[(df["model"] == "medsam3") & (df["prompt_type"] == "text")]["dice"].mean()
        improvement = (medsam3_text_mean - sam3_text_mean) * 100
        f.write(f"- **MedSAM3 text improvement over SAM3**: {improvement:+.1f}% Dice\n")

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU BraTS2023_MET All-Slices Evaluation"
    )
    parser.add_argument(
        "--worker", action="store_true",
        help="Run as worker process (internal use)"
    )
    parser.add_argument(
        "--case", type=str,
        help="Case ID to evaluate (worker mode)"
    )
    parser.add_argument(
        "--gpu", type=int,
        help="GPU ID to use (worker mode)"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR / "raw"),
        help="Output directory"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results"
    )
    args = parser.parse_args()

    # Worker mode
    if args.worker:
        if not args.case or args.gpu is None:
            print("ERROR: --case and --gpu required in worker mode")
            sys.exit(1)
        evaluate_case_worker(args.case, args.gpu, Path(args.output))
        return

    # Aggregate only mode
    if args.aggregate_only:
        aggregate_results(OUTPUT_DIR)
        return

    # === Orchestrator mode ===
    print("=" * 60)
    print("BraTS2023_MET All-Slices Multi-GPU Evaluation")
    print("=" * 60)
    print(f"Cases: {TARGET_CASES}")
    print(f"GPUs: {AVAILABLE_GPUS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Create output directory
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Launch worker processes
    processes = []
    for case_id, gpu_id in zip(TARGET_CASES, AVAILABLE_GPUS):
        print(f"Launching worker: {case_id} on GPU {gpu_id}")
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--case", case_id,
            "--gpu", str(gpu_id),
            "--output", str(raw_dir)
        ]
        p = subprocess.Popen(cmd)
        processes.append((case_id, gpu_id, p))

    print(f"\nLaunched {len(processes)} parallel workers")
    print("Waiting for completion...")

    # Wait for all processes
    for case_id, gpu_id, p in processes:
        p.wait()
        if p.returncode == 0:
            print(f"  [GPU {gpu_id}] {case_id}: SUCCESS")
        else:
            print(f"  [GPU {gpu_id}] {case_id}: FAILED (code {p.returncode})")

    # Aggregate results
    aggregate_results(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
