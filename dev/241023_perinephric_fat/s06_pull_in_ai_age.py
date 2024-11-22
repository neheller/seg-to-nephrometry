import json
from pathlib import Path

import pandas as pd


AI_AGE_PTH = Path(
    "/home/nick/code/ccf/c4kc-py-stats/age-stats/full_ds/full_ds_agg.json"
)
CLIN_MET_PTH = Path(__file__).parent / "stats" / "clinical_metrics.csv"


def main():
    # Read AI Age Data
    ai_age_df = pd.read_json(AI_AGE_PTH)

    # Read Clinical Metrics
    clin_metrics_df = pd.read_csv(CLIN_MET_PTH)

    # Join on case_id
    df = pd.merge(ai_age_df, clin_metrics_df, on="case_id")

    # Save merged result
    df.to_csv(Path(__file__).parent / "stats" / "ai_age_clinical.csv", index=False)

    # Sort by case_id
    df = df.sort_values(by="case_id")

    # Subset the data
    df = df[
        [
            "case_id",
            "normalized_residual",  # AI Age Discrepancy
            "ips_2cm_adipose_frac",
            "cnt_2cm_adipose_frac",
            "adipose_frac_diff",
            "ips_hu_at_ad_frac_max",
            "cnt_hu_at_ad_frac_max",
            "hu_diff_at_ad_frac_max"
        ]
    ]

    # Rename normalized residual to age discrepancy score
    df = df.rename(columns={"normalized_residual": "age_discrepancy_score"})

    # Save result
    df.to_csv(Path(__file__).parent / "stats" / "241122_radiomics_subset.csv", index=False)


if __name__ == "__main__":
    main()
