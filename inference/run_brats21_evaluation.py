#!/usr/bin/env python3
"""
SAM3 and MedSAM3 Evaluation on BraTS2021 (Adult Glioma) Dataset.

Evaluates brain tumor segmentation using TEXT PROMPTS ONLY:
- 4 MRI contrasts (t1, t1ce, t2, flair)
- 3 tumor labels (NCR, ED, ET) + combined
- Middle slices only

KEY DIFFERENCES FROM BraTS2023:
- Contrast names: t1, t1ce, t2, flair (not t1n, t1c, t2w, t2f)
- Label scheme: 0, 1, 2, 4 (NOT 0, 1, 2, 3 - ET is label 4!)
- Case naming: BraTS2021_XXXXX

Usage:
    conda activate medsam3

    # SAM3 baseline
    python run_brats21_evaluation.py --model sam3 --max-samples 20

    # MedSAM3 fine-tuned
    python run_brats21_evaluation.py --model medsam3 --max-samples 20

    # Quick test (5 samples)
    python run_brats21_evaluation.py --model sam3 --max-samples 5 --contrasts t1ce
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

from brats21_loader import (
    load_brats21_sample, get_case_ids, load_brats21_all_labels,
    CONTRASTS, TUMOR_LABELS, STANDARD_TEXT_PROMPTS, LABEL_PROMPTS,
    BraTS21Sample
)
from sam3_inference import SAM3Model, resize_mask
from metrics import compute_dice, compute_iou


# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "brats2021"


# Text prompts to evaluate for glioma (different from meningioma!)
GLIOMA_TEXT_PROMPTS = [
    "Brain Tumor",
    "Glioma",
    "Enhancing Tumor",
    "Tumor",
]

# Per-label specific prompts
LABEL_TEXT_PROMPTS = {
    "NCR": [
        "Non-Enhancing Tumor Core",
        "Necrotic Core",
        "Tumor Core",
    ],
    "ED": [
        "Peritumoral Edema",
        "Edema",
        "Brain Edema",
    ],
    "ET": [
        "Enhancing Tumor",
        "Enhancing Core",
        "Contrast-Enhancing Tumor",
    ],
}


def evaluate_text_prompts(
    sam3: SAM3Model,
    samples: List[BraTS21Sample],
    text_prompts: List[str],
    desc: str = "Evaluating"
) -> Dict:
    """
    Evaluate model on samples using TEXT PROMPTS ONLY.

    Args:
        sam3: SAM3Model instance
        samples: List of BraTS21Sample objects
        text_prompts: List of text prompts to evaluate
        desc: Description for progress bar

    Returns:
        Dict with text_metrics per prompt
    """
    text_metrics = {prompt: [] for prompt in text_prompts}

    for sample in tqdm(samples, desc=desc):
        img_size = sample.gt_mask.shape

        # Skip if no tumor in slice
        if sample.gt_mask.sum() == 0:
            continue

        # Encode image
        inference_state = sam3.encode_image(sample.image)

        # Text prompt evaluation
        for prompt in text_prompts:
            pred_text = sam3.predict_text(inference_state, prompt)
            if pred_text is not None:
                if pred_text.shape != img_size:
                    pred_text = resize_mask(pred_text, img_size)
                dice = compute_dice(pred_text, sample.gt_mask)
                iou = compute_iou(pred_text, sample.gt_mask)
                text_metrics[prompt].append({
                    'case_id': sample.case_id,
                    'contrast': sample.contrast,
                    'slice_idx': sample.slice_idx,
                    'tumor_label': sample.tumor_label,
                    'dice': dice,
                    'iou': iou,
                })
            else:
                text_metrics[prompt].append({
                    'case_id': sample.case_id,
                    'contrast': sample.contrast,
                    'slice_idx': sample.slice_idx,
                    'tumor_label': sample.tumor_label,
                    'dice': 0.0,
                    'iou': 0.0,
                })

    return text_metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics into summary statistics."""
    if not metrics_list:
        return {'dice': 0, 'iou': 0, 'n': 0}

    df = pd.DataFrame(metrics_list)
    return {
        'dice': df['dice'].mean(),
        'iou': df['iou'].mean(),
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
    Evaluate model across contrasts using text prompts only.

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
            sample = load_brats21_sample(
                case_id, contrast,
                tumor_label=tumor_label,
                text_prompt="Brain Tumor"
            )
            if sample is not None and sample.gt_mask.sum() > 0:
                samples.append(sample)

        print(f"Loaded {len(samples)} samples with tumor")

        if not samples:
            results[contrast] = {
                'text_prompts': {p: aggregate_metrics([]) for p in text_prompts}
            }
            continue

        # Evaluate text prompts
        text_metrics = evaluate_text_prompts(
            sam3, samples, text_prompts, desc=f"{contrast}"
        )

        # Aggregate
        results[contrast] = {
            'text_prompts': {
                p: aggregate_metrics(text_metrics[p])
                for p in text_prompts
            }
        }

        # Print summary
        print(f"\n  Text Prompt Performance (Dice %):")
        for prompt in text_prompts:
            text_dice = results[contrast]['text_prompts'][prompt]['dice']
            print(f"    '{prompt}': {text_dice:.2%}")

    return results


def run_full_evaluation(
    sam3: SAM3Model,
    model_name: str,
    max_samples: int = 20,
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
        text_prompts = GLIOMA_TEXT_PROMPTS

    # Get sampled case IDs
    case_ids = get_case_ids(max_samples=max_samples, seed=seed)
    print(f"\nSampled {len(case_ids)} cases")

    all_results = {
        'model_name': model_name,
        'dataset': 'BraTS2021',
        'timestamp': datetime.now().isoformat(),
        'n_cases': len(case_ids),
        'contrasts': contrasts,
        'text_prompts': text_prompts,
        'by_contrast': {},
        'by_label': {},
    }

    # Evaluate combined tumor mask
    print("\n" + "#" * 60)
    print("# COMBINED TUMOR MASK (all labels: 1, 2, 4)")
    print("#" * 60)

    all_results['by_contrast']['combined'] = evaluate_by_contrast(
        sam3, case_ids, contrasts, text_prompts, tumor_label=None
    )

    # Evaluate individual tumor labels with label-specific prompts
    # NOTE: Labels are NCR (1), ED (2), ET (4) - not 3!
    for label_name in ["NCR", "ED", "ET"]:
        print("\n" + "#" * 60)
        print(f"# TUMOR LABEL: {label_name}")
        print("#" * 60)

        # Use both standard prompts and label-specific prompts
        label_prompts = text_prompts + LABEL_TEXT_PROMPTS.get(label_name, [])
        # Remove duplicates while preserving order
        label_prompts = list(dict.fromkeys(label_prompts))

        all_results['by_label'][label_name] = evaluate_by_contrast(
            sam3, case_ids, contrasts, label_prompts, tumor_label=label_name
        )

    return all_results


def generate_report(results: Dict, output_dir: Path):
    """Generate evaluation reports (CSV, JSON, Markdown)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = results['model_name']

    # === Summary CSV by contrast (combined mask) ===
    summary_rows = []
    for contrast in results['contrasts']:
        row = {'Contrast': contrast.upper()}

        # Combined mask results
        combined = results['by_contrast']['combined'].get(contrast, {})

        # Text prompt results
        for prompt in results['text_prompts']:
            text_data = combined.get('text_prompts', {}).get(prompt, {})
            row[f'{prompt} Dice (%)'] = round(text_data.get('dice', 0) * 100, 2)
            row[f'{prompt} IoU (%)'] = round(text_data.get('iou', 0) * 100, 2)

        row['N Samples'] = combined.get('text_prompts', {}).get(
            results['text_prompts'][0], {}
        ).get('n', 0)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"{model_name}_brats21_text_prompt_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")

    # === By-label CSV ===
    label_rows = []
    for label_name in ["NCR", "ED", "ET"]:
        label_data = results['by_label'].get(label_name, {})
        for contrast in results['contrasts']:
            contrast_data = label_data.get(contrast, {})
            text_prompts_data = contrast_data.get('text_prompts', {})

            # Find best text prompt
            best_prompt = None
            best_dice = 0
            best_iou = 0

            for prompt, metrics in text_prompts_data.items():
                dice = metrics.get('dice', 0)
                if dice > best_dice:
                    best_dice = dice
                    best_iou = metrics.get('iou', 0)
                    best_prompt = prompt

            n_samples = next(iter(text_prompts_data.values()), {}).get('n', 0)

            label_rows.append({
                'Label': label_name,
                'Contrast': contrast.upper(),
                'Best Text Prompt': best_prompt or "N/A",
                'Dice (%)': round(best_dice * 100, 2),
                'IoU (%)': round(best_iou * 100, 2),
                'N Samples': n_samples,
            })

    df_labels = pd.DataFrame(label_rows)
    labels_csv_path = output_dir / f"{model_name}_brats21_text_prompt_by_label.csv"
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"Saved by-label results to: {labels_csv_path}")

    # === Detailed JSON ===
    json_path = output_dir / f"{model_name}_brats21_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved detailed results to: {json_path}")

    # === Markdown Report ===
    report_path = output_dir / f"{model_name}_brats21_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# {model_name.upper()} Evaluation on BraTS2021 (Adult Glioma)\n\n")
        f.write(f"**Generated**: {results['timestamp']}\n\n")
        f.write(f"**Dataset**: BraTS2021 (Adult Glioma Brain Tumors)\n\n")
        f.write(f"**Cases evaluated**: {results['n_cases']}\n\n")
        f.write("**Evaluation type**: Middle slice, text prompts only\n\n")

        # Combined tumor mask results
        f.write("## Combined Tumor Mask - Text Prompt Performance\n\n")
        f.write("| Contrast |")
        for prompt in results['text_prompts']:
            f.write(f" {prompt} |")
        f.write("\n")
        f.write("|----------|" + "----------|" * len(results['text_prompts']) + "\n")

        for contrast in results['contrasts']:
            combined = results['by_contrast']['combined'].get(contrast, {})
            f.write(f"| {contrast.upper()} |")
            for prompt in results['text_prompts']:
                dice = combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
                f.write(f" {dice:.1f}% |")
            f.write("\n")

        # Per-label results
        f.write("\n## Per-Label Best Performance\n\n")
        f.write("| Label | Contrast | Best Text Prompt | Dice | IoU |\n")
        f.write("|-------|----------|------------------|------|-----|\n")

        for label_name in ["NCR", "ED", "ET"]:
            label_data = results['by_label'].get(label_name, {})

            # Find best contrast/prompt combination
            best_contrast = None
            best_prompt = None
            best_dice = 0
            best_iou = 0

            for contrast in results['contrasts']:
                contrast_data = label_data.get(contrast, {})
                text_prompts_data = contrast_data.get('text_prompts', {})

                for prompt, metrics in text_prompts_data.items():
                    dice = metrics.get('dice', 0)
                    if dice > best_dice:
                        best_dice = dice
                        best_iou = metrics.get('iou', 0)
                        best_prompt = prompt
                        best_contrast = contrast

            f.write(f"| {label_name} | {(best_contrast or 'N/A').upper()} | "
                    f"{best_prompt or 'N/A'} | {best_dice*100:.1f}% | {best_iou*100:.1f}% |\n")

        # Label explanations
        f.write("\n## Label Definitions\n\n")
        f.write("- **NCR**: Necrotic/Non-Enhancing Tumor Core (label 1, ~97.4% of cases)\n")
        f.write("- **ED**: Peritumoral Edema (label 2, ~96.6% of cases)\n")
        f.write("- **ET**: Enhancing Tumor (label 4, ~99.9% of cases) - NOTE: label 4, not 3!\n")

        f.write("\n## Notes\n\n")
        f.write("- Evaluated on middle slices of sampled cases\n")
        f.write("- Text prompts evaluated zero-shot (no prompt tuning)\n")
        f.write("- BraTS2021: Adult glioma brain tumors\n")
        f.write("- Contrasts: T1, T1CE, T2, FLAIR (different from BraTS2023)\n")
        f.write("- 1,251 total cases in dataset\n")

    print(f"Saved report to: {report_path}")

    return df_summary


def create_comparison_report(sam3_json: Path, medsam3_json: Path, output_dir: Path):
    """Create comparison report between SAM3 and MedSAM3."""
    with open(sam3_json) as f:
        sam3_results = json.load(f)
    with open(medsam3_json) as f:
        medsam3_results = json.load(f)

    # Comparison CSV
    rows = []
    for contrast in CONTRASTS:
        sam3_combined = sam3_results['by_contrast']['combined'].get(contrast, {})
        medsam3_combined = medsam3_results['by_contrast']['combined'].get(contrast, {})

        for prompt in GLIOMA_TEXT_PROMPTS:
            sam3_dice = sam3_combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
            medsam3_dice = medsam3_combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
            improvement = medsam3_dice - sam3_dice

            rows.append({
                'Contrast': contrast.upper(),
                'Text Prompt': prompt,
                'SAM3 Dice (%)': round(sam3_dice, 2),
                'MedSAM3 Dice (%)': round(medsam3_dice, 2),
                'Improvement (%)': round(improvement, 2),
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "comparison_text_prompt.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison to: {csv_path}")

    # Comparison Markdown
    report_path = output_dir / "text_prompt_report.md"
    with open(report_path, 'w') as f:
        f.write("# BraTS2021 Text Prompt-Only Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("**Evaluation Type**: Middle slice, text prompts only\n\n")
        f.write(f"**Cases**: {sam3_results['n_cases']}\n\n")

        # SAM3 Results
        f.write("## SAM3 (Base Model) - Text Prompt Performance\n\n")
        f.write("| Contrast |")
        for prompt in GLIOMA_TEXT_PROMPTS:
            f.write(f" {prompt} |")
        f.write("\n")
        f.write("|----------|" + "----------|" * len(GLIOMA_TEXT_PROMPTS) + "\n")

        for contrast in CONTRASTS:
            sam3_combined = sam3_results['by_contrast']['combined'].get(contrast, {})
            f.write(f"| {contrast.upper()} |")
            for prompt in GLIOMA_TEXT_PROMPTS:
                dice = sam3_combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
                f.write(f" {dice:.2f}% |")
            f.write("\n")

        # MedSAM3 Results
        f.write("\n## MedSAM3 (Fine-tuned) - Text Prompt Performance\n\n")
        f.write("| Contrast |")
        for prompt in GLIOMA_TEXT_PROMPTS:
            f.write(f" {prompt} |")
        f.write("\n")
        f.write("|----------|" + "----------|" * len(GLIOMA_TEXT_PROMPTS) + "\n")

        for contrast in CONTRASTS:
            medsam3_combined = medsam3_results['by_contrast']['combined'].get(contrast, {})
            f.write(f"| {contrast.upper()} |")
            for prompt in GLIOMA_TEXT_PROMPTS:
                dice = medsam3_combined.get('text_prompts', {}).get(prompt, {}).get('dice', 0) * 100
                f.write(f" {dice:.2f}% |")
            f.write("\n")

        # Comparison table
        f.write("\n## MedSAM3 Improvement over SAM3\n\n")
        f.write("| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |\n")
        f.write("|----------|-------------|-----------|--------------|-------------|\n")

        for _, row in df.iterrows():
            improvement = row['Improvement (%)']
            imp_str = f"+{improvement}%" if improvement > 0 else f"{improvement}%"
            f.write(f"| {row['Contrast']} | {row['Text Prompt']} | "
                    f"{row['SAM3 Dice (%)']}% | {row['MedSAM3 Dice (%)']}% | "
                    f"**{imp_str}** |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Best prompt for MedSAM3
        best_row = df.loc[df['MedSAM3 Dice (%)'].idxmax()]
        f.write(f"- **Best MedSAM3 configuration**: {best_row['Text Prompt']} on {best_row['Contrast']} = "
                f"{best_row['MedSAM3 Dice (%)']}% Dice\n")

        # Average improvement
        avg_improvement = df['Improvement (%)'].mean()
        f.write(f"- **Average improvement over SAM3**: {avg_improvement:+.1f}% Dice\n")

        # Best improvement
        best_imp_row = df.loc[df['Improvement (%)'].idxmax()]
        f.write(f"- **Largest improvement**: {best_imp_row['Text Prompt']} on {best_imp_row['Contrast']} = "
                f"{best_imp_row['Improvement (%)']}% Dice improvement\n")

        f.write("\n## Notes\n\n")
        f.write("- Results from middle slice of each 3D MRI volume\n")
        f.write("- Combined tumor mask (all tumor labels 1, 2, 4)\n")
        f.write("- Text prompts evaluated zero-shot (no prompt tuning)\n")
        f.write("- BraTS2021: Adult glioma brain tumors\n")
        f.write("- Labels: NCR (1), ED (2), ET (4) - note ET is label 4\n")
        f.write("- 1,251 total cases in dataset\n")

    print(f"Saved comparison report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3/MedSAM3 Evaluation on BraTS2021 Dataset (Text Prompts Only)"
    )
    parser.add_argument(
        "--model", choices=["sam3", "medsam3"], default=None,
        help="Model to evaluate (sam3 or medsam3)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MedSAM3 checkpoint (optional, uses HuggingFace default)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=20,
        help="Maximum number of cases to sample (default: 20)"
    )
    parser.add_argument(
        "--contrasts", type=str, default=None,
        help="Comma-separated list of contrasts (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--create-comparison", action="store_true",
        help="Create comparison report (requires both SAM3 and MedSAM3 results)"
    )
    args = parser.parse_args()

    # Handle comparison report (doesn't need --model)
    if args.create_comparison:
        sam3_json = OUTPUT_DIR / "sam3_brats21_detailed.json"
        medsam3_json = OUTPUT_DIR / "medsam3_brats21_detailed.json"

        if not sam3_json.exists() or not medsam3_json.exists():
            print("ERROR: Both SAM3 and MedSAM3 results required for comparison")
            print(f"  SAM3: {sam3_json} {'(exists)' if sam3_json.exists() else '(missing)'}")
            print(f"  MedSAM3: {medsam3_json} {'(exists)' if medsam3_json.exists() else '(missing)'}")
            sys.exit(1)

        create_comparison_report(sam3_json, medsam3_json, OUTPUT_DIR)
        return

    # Model is required for evaluation
    if args.model is None:
        print("ERROR: --model is required for evaluation")
        print("Use --model sam3 or --model medsam3")
        print("Use --create-comparison to generate comparison report from existing results")
        sys.exit(1)

    print("=" * 60)
    print("BraTS2021 Evaluation (Adult Glioma Brain Tumors)")
    print("TEXT PROMPTS ONLY")
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
        # MedSAM3 - use HuggingFace checkpoint
        if args.checkpoint is None:
            # Download from HuggingFace
            from huggingface_hub import hf_hub_download
            print("Downloading MedSAM3 checkpoint from HuggingFace...")
            args.checkpoint = hf_hub_download(
                repo_id="Potestates/medsam3_stage1_text_full",
                filename="checkpoint.pt"
            )
            print(f"Downloaded to: {args.checkpoint}")
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
    print("FINAL SUMMARY (Combined Tumor Mask) - Text Prompts Only")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # Check if we can create comparison
    sam3_json = OUTPUT_DIR / "sam3_brats21_detailed.json"
    medsam3_json = OUTPUT_DIR / "medsam3_brats21_detailed.json"
    if sam3_json.exists() and medsam3_json.exists():
        print("\nBoth SAM3 and MedSAM3 results available.")
        print("Run with --create-comparison to generate comparison report.")


if __name__ == "__main__":
    main()
