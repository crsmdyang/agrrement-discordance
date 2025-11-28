"""
CLI tool for automatic statistical analysis of MDT/AI/Guideline concordance datasets.

Usage:
    python cli_analyze.py path/to/data.xlsx --output-dir results/

The input Excel file should follow the structure described in README and used by the
Streamlit app (sheets: `recommendations`, `concordance_ratings`).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

import core_concordance


def format_float(value: float, digits: int = 3) -> str:
    """Safely format floats, returning "NA" for missing values."""
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def analyze_file(input_path: Path, output_dir: Path) -> None:
    """Run full analysis pipeline and save outputs."""
    df_rec, df_conc = core_concordance.load_data(input_path)

    # Exact concordance summary
    exact_summary = core_concordance.build_global_exact_concordance_summary(df_rec)
    save_dataframe(exact_summary, output_dir / "exact_concordance_summary.csv")

    # Confusion matrices for each pair
    confusion_outputs = []
    for pair_name, col1, col2 in [
        ("MDT vs Guideline", "mdt_rec", "guideline_rec"),
        ("GPT vs Guideline", "gpt_rec", "guideline_rec"),
        ("MDT vs GPT", "mdt_rec", "gpt_rec"),
    ]:
        res = core_concordance.compute_pairwise_exact_concordance(df_rec, col1, col2)
        confusion_path = output_dir / f"confusion_{pair_name.replace(' ', '_').replace('/', '_')}.csv"
        save_dataframe(res["confusion_matrix"].reset_index(), confusion_path)
        confusion_outputs.append((pair_name, res, confusion_path))

    # Ratings summary and ICC
    ratings_results = core_concordance.summarize_concordance_scores(df_conc)
    save_dataframe(ratings_results["descriptive"], output_dir / "ratings_summary.csv")
    save_dataframe(ratings_results["icc"], output_dir / "ratings_icc.csv")

    # Discordance reasons
    reasons_df = core_concordance.analyze_discordance_reasons(df_rec)
    if not reasons_df.empty:
        save_dataframe(reasons_df, output_dir / "discordance_reasons.csv")

    # -----------------------
    # Console report
    # -----------------------
    print("\n=== Exact Concordance (string match) ===")
    if exact_summary.empty:
        print("No valid cases found.")
    else:
        print(exact_summary.assign(
            agreement_rate=lambda d: (d["agreement_rate"] * 100).map(lambda v: f"{v:.1f}%"),
            kappa=lambda d: d["kappa"].map(lambda v: format_float(v)),
        ).to_string(index=False))

    print("\nConfusion matrices saved to:")
    for pair_name, res, path in confusion_outputs:
        print(f"- {pair_name}: {path}")
        if not res["confusion_matrix"].empty:
            print(res["confusion_matrix"].to_string())
        else:
            print("  (no data)")

    print("\n=== Concordance Ratings (0-5 scale) ===")
    descriptive = ratings_results["descriptive"].copy()
    if descriptive.empty:
        print("No ratings available.")
    else:
        print(descriptive.assign(
            mean_score=lambda d: d["mean_score"].map(lambda v: format_float(v, 2)),
            sd_score=lambda d: d["sd_score"].map(lambda v: format_float(v, 2)),
        ).to_string(index=False))

    print("\nReliability (ICC)")
    icc_df = ratings_results["icc"].copy()
    if icc_df.empty:
        print("No ICC results available.")
    else:
        print(icc_df.assign(
            ICC_single=lambda d: d["ICC_single"].map(format_float),
            ICC_average=lambda d: d["ICC_average"].map(format_float),
        ).to_string(index=False))

    print("\n=== Discordance Reasons ===")
    if reasons_df.empty:
        print("No discordant cases or reasons provided.")
    else:
        print(reasons_df.assign(percent=lambda d: d["percent"].map(lambda v: f"{v:.1f}%")).to_string(index=False))
        print(f"Saved to: {output_dir / 'discordance_reasons.csv'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically analyze MDT/AI/Guideline concordance datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to Excel file (with recommendations and concordance_ratings sheets)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save CSV reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    analyze_file(args.input, args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
