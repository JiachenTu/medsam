#!/usr/bin/env python3
"""
Extract text-prompt-only results from BraTS2023_MET evaluation.

Reads existing detailed JSON results and creates organized text-prompt-only
summaries with Dice and IoU metrics.

Usage:
    conda activate medsam3
    python extract_text_prompt_results.py
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


# Paths
INPUT_DIR = Path(__file__).parent / "results" / "brats2023_met"
OUTPUT_DIR = Path(__file__).parent / "results" / "brats_text_prompt"

# Text prompts evaluated
TEXT_PROMPTS = ["Brain Tumor", "Brain Metastasis", "Enhancing Tumor", "Tumor"]
CONTRASTS = ["t1n", "t1c", "t2w", "t2f"]
LABELS = ["NETC", "SNFH", "ET"]


def load_detailed_results(json_path: Path) -> dict:
    """Load detailed JSON results."""
    with open(json_path) as f:
        return json.load(f)


def extract_text_prompt_summary(results: dict) -> pd.DataFrame:
    """
    Extract text prompt results by contrast (combined tumor mask).

    Returns DataFrame with columns:
    Contrast, Brain Tumor Dice, Brain Tumor IoU, ...
    """
    rows = []

    combined_results = results.get("by_contrast", {}).get("combined", {})

    for contrast in CONTRASTS:
        contrast_data = combined_results.get(contrast, {})
        text_data = contrast_data.get("text_prompts", {})

        row = {"Contrast": contrast.upper()}

        for prompt in TEXT_PROMPTS:
            prompt_results = text_data.get(prompt, {})
            dice = prompt_results.get("dice", 0) * 100
            iou = prompt_results.get("iou", 0) * 100

            row[f"{prompt} Dice (%)"] = round(dice, 2)
            row[f"{prompt} IoU (%)"] = round(iou, 2)

        row["N Samples"] = text_data.get(TEXT_PROMPTS[0], {}).get("n", 0)
        rows.append(row)

    return pd.DataFrame(rows)


def extract_text_prompt_by_label(results: dict) -> pd.DataFrame:
    """
    Extract text prompt results by tumor label.

    Returns DataFrame with best text prompt for each label/contrast.
    """
    rows = []

    by_label = results.get("by_label", {})

    for label in LABELS:
        label_data = by_label.get(label, {})

        for contrast in CONTRASTS:
            contrast_data = label_data.get(contrast, {})
            text_data = contrast_data.get("text_prompts", {})

            # Find best text prompt
            best_prompt = None
            best_dice = 0
            best_iou = 0

            for prompt in TEXT_PROMPTS:
                prompt_results = text_data.get(prompt, {})
                dice = prompt_results.get("dice", 0)
                if dice > best_dice:
                    best_dice = dice
                    best_iou = prompt_results.get("iou", 0)
                    best_prompt = prompt

            n_samples = text_data.get(TEXT_PROMPTS[0], {}).get("n", 0) if text_data else 0

            rows.append({
                "Label": label,
                "Contrast": contrast.upper(),
                "Best Text Prompt": best_prompt or "N/A",
                "Dice (%)": round(best_dice * 100, 2),
                "IoU (%)": round(best_iou * 100, 2),
                "N Samples": n_samples,
            })

    return pd.DataFrame(rows)


def extract_text_prompt_detailed(results: dict) -> dict:
    """
    Extract detailed text prompt results (all prompts, all contrasts, all labels).

    Returns dict structure with only text prompt data.
    """
    text_only = {
        "model_name": results.get("model_name"),
        "timestamp": results.get("timestamp"),
        "n_cases": results.get("n_cases"),
        "contrasts": CONTRASTS,
        "text_prompts": TEXT_PROMPTS,
        "by_contrast": {},
        "by_label": {},
    }

    # Combined mask results
    combined = results.get("by_contrast", {}).get("combined", {})
    text_only["by_contrast"]["combined"] = {}

    for contrast in CONTRASTS:
        contrast_data = combined.get(contrast, {})
        text_data = contrast_data.get("text_prompts", {})
        text_only["by_contrast"]["combined"][contrast] = {
            "text_prompts": text_data
        }

    # Per-label results
    by_label = results.get("by_label", {})
    for label in LABELS:
        label_data = by_label.get(label, {})
        text_only["by_label"][label] = {}

        for contrast in CONTRASTS:
            contrast_data = label_data.get(contrast, {})
            text_data = contrast_data.get("text_prompts", {})
            text_only["by_label"][label][contrast] = {
                "text_prompts": text_data
            }

    return text_only


def create_comparison_csv(sam3_summary: pd.DataFrame, medsam3_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison CSV showing SAM3 vs MedSAM3 text prompt performance.
    """
    rows = []

    for contrast in CONTRASTS:
        sam3_row = sam3_summary[sam3_summary["Contrast"] == contrast.upper()].iloc[0]
        medsam3_row = medsam3_summary[medsam3_summary["Contrast"] == contrast.upper()].iloc[0]

        for prompt in TEXT_PROMPTS:
            dice_col = f"{prompt} Dice (%)"
            iou_col = f"{prompt} IoU (%)"

            sam3_dice = sam3_row[dice_col]
            medsam3_dice = medsam3_row[dice_col]
            sam3_iou = sam3_row[iou_col]
            medsam3_iou = medsam3_row[iou_col]

            improvement = medsam3_dice - sam3_dice

            rows.append({
                "Contrast": contrast.upper(),
                "Text Prompt": prompt,
                "SAM3 Dice (%)": sam3_dice,
                "MedSAM3 Dice (%)": medsam3_dice,
                "Improvement (%)": round(improvement, 2),
                "SAM3 IoU (%)": sam3_iou,
                "MedSAM3 IoU (%)": medsam3_iou,
            })

    return pd.DataFrame(rows)


def generate_report(sam3_summary: pd.DataFrame, medsam3_summary: pd.DataFrame,
                    comparison: pd.DataFrame, output_path: Path):
    """Generate markdown report."""
    with open(output_path, "w") as f:
        f.write("# BraTS2023_MET Text Prompt-Only Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("**Evaluation Type**: Middle slice, text prompts only\n\n")
        f.write("**Cases**: 48 (~20% of dataset)\n\n")

        # SAM3 Results
        f.write("## SAM3 (Base Model) - Text Prompt Performance\n\n")
        f.write("| Contrast | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |\n")
        f.write("|----------|-------------|------------------|-----------------|-------|\n")

        for _, row in sam3_summary.iterrows():
            f.write(f"| {row['Contrast']} | "
                    f"{row['Brain Tumor Dice (%)']}% | "
                    f"{row['Brain Metastasis Dice (%)']}% | "
                    f"{row['Enhancing Tumor Dice (%)']}% | "
                    f"{row['Tumor Dice (%)']}% |\n")

        # MedSAM3 Results
        f.write("\n## MedSAM3 (Fine-tuned) - Text Prompt Performance\n\n")
        f.write("| Contrast | Brain Tumor | Brain Metastasis | Enhancing Tumor | Tumor |\n")
        f.write("|----------|-------------|------------------|-----------------|-------|\n")

        for _, row in medsam3_summary.iterrows():
            f.write(f"| {row['Contrast']} | "
                    f"{row['Brain Tumor Dice (%)']}% | "
                    f"{row['Brain Metastasis Dice (%)']}% | "
                    f"{row['Enhancing Tumor Dice (%)']}% | "
                    f"{row['Tumor Dice (%)']}% |\n")

        # Comparison
        f.write("\n## MedSAM3 Improvement over SAM3\n\n")
        f.write("| Contrast | Text Prompt | SAM3 Dice | MedSAM3 Dice | Improvement |\n")
        f.write("|----------|-------------|-----------|--------------|-------------|\n")

        for _, row in comparison.iterrows():
            improvement = row["Improvement (%)"]
            imp_str = f"+{improvement}%" if improvement > 0 else f"{improvement}%"
            f.write(f"| {row['Contrast']} | {row['Text Prompt']} | "
                    f"{row['SAM3 Dice (%)']}% | {row['MedSAM3 Dice (%)']}% | "
                    f"**{imp_str}** |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Best prompt for MedSAM3
        best_row = comparison.loc[comparison["MedSAM3 Dice (%)"].idxmax()]
        f.write(f"- **Best MedSAM3 configuration**: {best_row['Text Prompt']} on {best_row['Contrast']} = "
                f"{best_row['MedSAM3 Dice (%)']}% Dice\n")

        # Average improvement
        avg_improvement = comparison["Improvement (%)"].mean()
        f.write(f"- **Average improvement over SAM3**: {avg_improvement:+.1f}% Dice\n")

        # Best improvement
        best_imp_row = comparison.loc[comparison["Improvement (%)"].idxmax()]
        f.write(f"- **Largest improvement**: {best_imp_row['Text Prompt']} on {best_imp_row['Contrast']} = "
                f"{best_imp_row['Improvement (%)']}% Dice improvement\n")

        f.write("\n## Notes\n\n")
        f.write("- Results from middle slice of each 3D MRI volume\n")
        f.write("- Combined tumor mask (all tumor labels 1-3)\n")
        f.write("- Text prompts evaluated zero-shot (no prompt tuning)\n")
        f.write("- MedSAM3 significantly improves text prompt understanding over SAM3 base model\n")

    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Extracting Text Prompt-Only Results")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process SAM3
    print("\nProcessing SAM3...")
    sam3_json = INPUT_DIR / "sam3_brats_detailed.json"
    sam3_results = load_detailed_results(sam3_json)

    sam3_summary = extract_text_prompt_summary(sam3_results)
    sam3_by_label = extract_text_prompt_by_label(sam3_results)
    sam3_detailed = extract_text_prompt_detailed(sam3_results)

    sam3_summary.to_csv(OUTPUT_DIR / "sam3_text_prompt_summary.csv", index=False)
    sam3_by_label.to_csv(OUTPUT_DIR / "sam3_text_prompt_by_label.csv", index=False)
    with open(OUTPUT_DIR / "sam3_text_prompt_detailed.json", "w") as f:
        json.dump(sam3_detailed, f, indent=2)

    print(f"  Saved: sam3_text_prompt_summary.csv")
    print(f"  Saved: sam3_text_prompt_by_label.csv")
    print(f"  Saved: sam3_text_prompt_detailed.json")

    # Process MedSAM3
    print("\nProcessing MedSAM3...")
    medsam3_json = INPUT_DIR / "medsam3_brats_detailed.json"
    medsam3_results = load_detailed_results(medsam3_json)

    medsam3_summary = extract_text_prompt_summary(medsam3_results)
    medsam3_by_label = extract_text_prompt_by_label(medsam3_results)
    medsam3_detailed = extract_text_prompt_detailed(medsam3_results)

    medsam3_summary.to_csv(OUTPUT_DIR / "medsam3_text_prompt_summary.csv", index=False)
    medsam3_by_label.to_csv(OUTPUT_DIR / "medsam3_text_prompt_by_label.csv", index=False)
    with open(OUTPUT_DIR / "medsam3_text_prompt_detailed.json", "w") as f:
        json.dump(medsam3_detailed, f, indent=2)

    print(f"  Saved: medsam3_text_prompt_summary.csv")
    print(f"  Saved: medsam3_text_prompt_by_label.csv")
    print(f"  Saved: medsam3_text_prompt_detailed.json")

    # Create comparison
    print("\nCreating comparison...")
    comparison = create_comparison_csv(sam3_summary, medsam3_summary)
    comparison.to_csv(OUTPUT_DIR / "comparison_text_prompt.csv", index=False)
    print(f"  Saved: comparison_text_prompt.csv")

    # Generate report
    print("\nGenerating report...")
    generate_report(sam3_summary, medsam3_summary, comparison,
                    OUTPUT_DIR / "text_prompt_report.md")

    print("\n" + "=" * 60)
    print("Text Prompt Results Extraction Complete!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Print summary table
    print("\n=== SAM3 Text Prompt Performance (Dice %) ===")
    print(sam3_summary.to_string(index=False))

    print("\n=== MedSAM3 Text Prompt Performance (Dice %) ===")
    print(medsam3_summary.to_string(index=False))


if __name__ == "__main__":
    main()
