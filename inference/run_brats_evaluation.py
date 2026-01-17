#!/usr/bin/env python3
"""
SAM3 and MedSAM3 Evaluation on BraTS2023_MET Dataset.

Evaluates brain metastases segmentation across:
- 4 MRI contrasts (t1n, t1c, t2w, t2f)
- 3 tumor labels (NETC, SNFH, ET) + combined
- Both box prompts and text prompts

Usage:
    conda activate /srv/local/shared/temp/tmp1/jtu9/envs/medsam3

    # SAM3 baseline
    python run_brats_evaluation.py --model sam3 --max-samples 48

    # MedSAM3 with checkpoint
    python run_brats_evaluation.py --model medsam3 \
        --checkpoint /srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt

    # Quick test (5 samples)
    python run_brats_evaluation.py --model sam3 --max-samples 5 --contrasts t1c
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from brats_loader import (
    load_brats_dataset, load_brats_sample, get_case_ids,
    CONTRASTS, TUMOR_LABELS, STANDARD_TEXT_PROMPTS, BraTSSample
)
from sam3_inference import SAM3Model, generate_bbox_from_mask, resize_mask
from metrics import compute_all_metrics


# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "brats2023_met"


def evaluate_samples(
    sam3: SAM3Model,
    samples: List[BraTSSample],
    text_prompts: List[str],
    desc: str = "Evaluating"
) -> Dict:
    """
    Evaluate model on a list of samples.

    Args:
        sam3: SAM3Model instance
        samples: List of BraTSSample objects
        text_prompts: List of text prompts to evaluate
        desc: Description for progress bar

    Returns:
        Dict with box_metrics and text_metrics
    """
    box_metrics = []
    text_metrics = {prompt: [] for prompt in text_prompts}

    for sample in tqdm(samples, desc=desc):
        img_size = sample.gt_mask.shape

        # Skip if no tumor in slice
        if sample.gt_mask.sum() == 0:
            continue

        # Encode image
        inference_state = sam3.encode_image(sample.image)

        # Box prompt evaluation
        bbox = generate_bbox_from_mask(sample.gt_mask)
        if bbox is not None:
            pred_box = sam3.predict_box(inference_state, bbox, img_size)
            if pred_box is not None:
                if pred_box.shape != img_size:
                    pred_box = resize_mask(pred_box, img_size)
                metrics = compute_all_metrics(pred_box, sample.gt_mask)
                box_metrics.append({
                    'case_id': sample.case_id,
                    'contrast': sample.contrast,
                    'slice_idx': sample.slice_idx,
                    'tumor_label': sample.tumor_label,
                    'dice': metrics.dice,
                    'iou': metrics.iou,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                })

        # Text prompt evaluation
        for prompt in text_prompts:
            pred_text = sam3.predict_text(inference_state, prompt)
            if pred_text is not None:
                if pred_text.shape != img_size:
                    pred_text = resize_mask(pred_text, img_size)
                metrics = compute_all_metrics(pred_text, sample.gt_mask)
                text_metrics[prompt].append({
                    'case_id': sample.case_id,
                    'contrast': sample.contrast,
                    'slice_idx': sample.slice_idx,
                    'tumor_label': sample.tumor_label,
                    'dice': metrics.dice,
                    'iou': metrics.iou,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                })
            else:
                text_metrics[prompt].append({
                    'case_id': sample.case_id,
                    'contrast': sample.contrast,
                    'slice_idx': sample.slice_idx,
                    'tumor_label': sample.tumor_label,
                    'dice': 0.0,
                    'iou': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                })

    return {
        'box_metrics': box_metrics,
        'text_metrics': text_metrics,
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics into summary statistics."""
    if not metrics_list:
        return {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'n': 0}

    df = pd.DataFrame(metrics_list)
    return {
        'dice': df['dice'].mean(),
        'iou': df['iou'].mean(),
        'precision': df['precision'].mean(),
        'recall': df['recall'].mean(),
        'n': len(df),
    }


def evaluate_by_contrast(
    sam3: SAM3Model,
    case_ids: List[str],
    contrasts: List[str],
    text_prompts: List[str],
    tumor_label: Optional[str] = None
) -> Dict:
    """
    Evaluate model across contrasts.

    Returns dict with results per contrast.
    """
    results = {}

    for contrast in contrasts:
        print(f"\n{'='*60}")
        print(f"Evaluating contrast: {contrast.upper()}")
        if tumor_label:
            print(f"Tumor label: {tumor_label}")
        print(f"{'='*60}")

        # Load samples for this contrast
        samples = []
        for case_id in case_ids:
            sample = load_brats_sample(
                case_id, contrast,
                tumor_label=tumor_label,
                text_prompt="Brain Tumor"
            )
            if sample is not None and sample.gt_mask.sum() > 0:
                samples.append(sample)

        print(f"Loaded {len(samples)} samples with tumor")

        if not samples:
            results[contrast] = {
                'box_prompt': aggregate_metrics([]),
                'text_prompts': {p: aggregate_metrics([]) for p in text_prompts}
            }
            continue

        # Evaluate
        eval_results = evaluate_samples(
            sam3, samples, text_prompts, desc=f"{contrast}"
        )

        # Aggregate
        results[contrast] = {
            'box_prompt': aggregate_metrics(eval_results['box_metrics']),
            'text_prompts': {
                p: aggregate_metrics(eval_results['text_metrics'][p])
                for p in text_prompts
            }
        }

        # Print summary
        box_dice = results[contrast]['box_prompt']['dice']
        print(f"\n  Box Prompt: Dice={box_dice:.2%}")
        for prompt in text_prompts:
            text_dice = results[contrast]['text_prompts'][prompt]['dice']
            print(f"  '{prompt}': Dice={text_dice:.2%}")

    return results


def run_full_evaluation(
    sam3: SAM3Model,
    model_name: str,
    max_samples: int = 48,
    contrasts: Optional[List[str]] = None,
    text_prompts: Optional[List[str]] = None,
    seed: int = 42
) -> Dict:
    """
    Run full evaluation across contrasts and tumor labels.

    Returns comprehensive results dict.
    """
    if contrasts is None:
        contrasts = CONTRASTS
    if text_prompts is None:
        text_prompts = STANDARD_TEXT_PROMPTS

    # Get sampled case IDs
    case_ids = get_case_ids(max_samples=max_samples, seed=seed)
    print(f"\nSampled {len(case_ids)} cases")

    all_results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'n_cases': len(case_ids),
        'contrasts': contrasts,
        'text_prompts': text_prompts,
        'by_contrast': {},
        'by_label': {},
    }

    # Evaluate combined tumor mask
    print("\n" + "#" * 60)
    print("# COMBINED TUMOR MASK (all labels)")
    print("#" * 60)

    all_results['by_contrast']['combined'] = evaluate_by_contrast(
        sam3, case_ids, contrasts, text_prompts, tumor_label=None
    )

    # Evaluate individual tumor labels
    for label_id, label_name in [(1, "NETC"), (2, "SNFH"), (3, "ET")]:
        print("\n" + "#" * 60)
        print(f"# TUMOR LABEL: {label_name} (label {label_id})")
        print("#" * 60)

        all_results['by_label'][label_name] = evaluate_by_contrast(
            sam3, case_ids, contrasts, text_prompts, tumor_label=label_name
        )

    return all_results


def generate_report(results: Dict, output_dir: Path):
    """Generate evaluation reports (CSV, JSON, Markdown)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = results['model_name']

    # === Summary CSV by contrast ===
    summary_rows = []
    for contrast in results['contrasts']:
        row = {'Contrast': contrast.upper()}

        # Combined mask results
        combined = results['by_contrast']['combined'].get(contrast, {})
        row['Box Dice (%)'] = combined.get('box_prompt', {}).get('dice', 0) * 100
        row['Box IoU (%)'] = combined.get('box_prompt', {}).get('iou', 0) * 100

        # Text prompt results
        for prompt in results['text_prompts']:
            text_dice = combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0)
            row[f'{prompt} (%)'] = text_dice * 100

        row['N Samples'] = combined.get('box_prompt', {}).get('n', 0)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"{model_name}_brats_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    # === By-label CSV ===
    label_rows = []
    for label_name in ["NETC", "SNFH", "ET"]:
        label_data = results['by_label'].get(label_name, {})
        for contrast in results['contrasts']:
            contrast_data = label_data.get(contrast, {})
            row = {
                'Label': label_name,
                'Contrast': contrast.upper(),
                'Box Dice (%)': contrast_data.get('box_prompt', {}).get('dice', 0) * 100,
                'Box IoU (%)': contrast_data.get('box_prompt', {}).get('iou', 0) * 100,
                'N Samples': contrast_data.get('box_prompt', {}).get('n', 0),
            }
            # Best text prompt
            best_prompt = None
            best_dice = 0
            for prompt in results['text_prompts']:
                dice = contrast_data.get('text_prompts', {}).get(prompt, {}).get('dice', 0)
                if dice > best_dice:
                    best_dice = dice
                    best_prompt = prompt
            row['Best Text Prompt'] = best_prompt or "N/A"
            row['Best Text Dice (%)'] = best_dice * 100
            label_rows.append(row)

    df_labels = pd.DataFrame(label_rows)
    labels_csv_path = output_dir / f"{model_name}_brats_by_label.csv"
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"Saved by-label results to: {labels_csv_path}")

    # === Detailed JSON ===
    json_path = output_dir / f"{model_name}_brats_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved detailed results to: {json_path}")

    # === Markdown Report ===
    report_path = output_dir / f"{model_name}_brats_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# {model_name} Evaluation on BraTS2023_MET\n\n")
        f.write(f"**Generated**: {results['timestamp']}\n\n")
        f.write(f"**Cases evaluated**: {results['n_cases']}\n\n")

        f.write("## Results by MRI Contrast (Combined Tumor Mask)\n\n")
        f.write("| Contrast | Box Dice | Box IoU | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |\n")
        f.write("|----------|----------|---------|-------------|------------------|-----------------|-------|\n")

        for contrast in results['contrasts']:
            combined = results['by_contrast']['combined'].get(contrast, {})
            box_dice = combined.get('box_prompt', {}).get('dice', 0) * 100
            box_iou = combined.get('box_prompt', {}).get('iou', 0) * 100

            text_values = []
            for prompt in results['text_prompts']:
                dice = combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
                text_values.append(f"{dice:.1f}%")

            f.write(f"| {contrast.upper()} | {box_dice:.1f}% | {box_iou:.1f}% | "
                    f"{' | '.join(text_values)} |\n")

        # Averages
        avg_box_dice = np.mean([
            results['by_contrast']['combined'].get(c, {}).get('box_prompt', {}).get('dice', 0)
            for c in results['contrasts']
        ]) * 100
        f.write(f"\n**Average Box Prompt Dice**: {avg_box_dice:.1f}%\n")

        # By-label section
        f.write("\n## Results by Tumor Label\n\n")
        f.write("| Label | Best Contrast | Box Dice | Best Text Prompt | Text Dice |\n")
        f.write("|-------|---------------|----------|------------------|----------|\n")

        for label_name in ["NETC", "SNFH", "ET"]:
            label_data = results['by_label'].get(label_name, {})

            # Find best contrast for this label
            best_contrast = None
            best_box_dice = 0
            for contrast in results['contrasts']:
                dice = label_data.get(contrast, {}).get('box_prompt', {}).get('dice', 0)
                if dice > best_box_dice:
                    best_box_dice = dice
                    best_contrast = contrast

            # Find best text prompt for best contrast
            best_text_prompt = None
            best_text_dice = 0
            if best_contrast:
                for prompt in results['text_prompts']:
                    dice = label_data.get(best_contrast, {}).get('text_prompts', {}).get(prompt, {}).get('dice', 0)
                    if dice > best_text_dice:
                        best_text_dice = dice
                        best_text_prompt = prompt

            f.write(f"| {label_name} | {(best_contrast or 'N/A').upper()} | "
                    f"{best_box_dice*100:.1f}% | {best_text_prompt or 'N/A'} | "
                    f"{best_text_dice*100:.1f}% |\n")

        f.write("\n## Notes\n\n")
        f.write("- Evaluated on middle slices of sampled cases (~20%)\n")
        f.write("- Box prompts derived from ground truth bounding boxes\n")
        f.write("- Text prompts evaluated zero-shot\n")
        f.write("- **NETC**: Non-Enhancing Tumor Core\n")
        f.write("- **SNFH**: Surrounding Non-enhancing FLAIR Hyperintensity (Edema)\n")
        f.write("- **ET**: Enhancing Tumor\n")

    print(f"Saved report to: {report_path}")

    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="SAM3/MedSAM3 Evaluation on BraTS2023_MET Dataset"
    )
    parser.add_argument(
        "--model", choices=["sam3", "medsam3"], required=True,
        help="Model to evaluate (sam3 or medsam3)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MedSAM3 checkpoint (required for medsam3)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=48,
        help="Maximum number of cases to sample (default: 48, ~20%%)"
    )
    parser.add_argument(
        "--contrasts", type=str, default=None,
        help="Comma-separated list of contrasts (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BraTS2023_MET Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")

    # Parse contrasts
    if args.contrasts:
        contrasts = [c.strip() for c in args.contrasts.split(",")]
    else:
        contrasts = CONTRASTS
    print(f"Contrasts: {contrasts}")

    # Initialize model
    if args.model == "sam3":
        sam3 = SAM3Model(confidence_threshold=0.1)
        model_name = "sam3"
    else:
        if args.checkpoint is None:
            # Default MedSAM3 checkpoint path
            default_ckpt = "/srv/local/shared/temp/tmp1/jtu9/medsam3_weights/checkpoint.pt"
            if os.path.exists(default_ckpt):
                args.checkpoint = default_ckpt
            else:
                print("ERROR: --checkpoint required for medsam3 model")
                sys.exit(1)

        print(f"Checkpoint: {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        sam3 = SAM3Model(
            confidence_threshold=0.1,
            checkpoint_path=args.checkpoint
        )
        model_name = "medsam3"

    # Run evaluation
    results = run_full_evaluation(
        sam3, model_name,
        max_samples=args.max_samples,
        contrasts=contrasts,
        seed=args.seed
    )

    # Generate reports
    print("\n" + "=" * 60)
    print("Generating Reports")
    print("=" * 60)

    df_summary = generate_report(results, OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (Combined Tumor Mask)")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
