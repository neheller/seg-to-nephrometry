import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


# Make destination folder
DST_PTH = Path(__file__).parent / "stats"
DST_PTH.mkdir(exist_ok=True)

# Load KiTS clinical data
CLINICAL_PTH = Path(
    "/home/nick/code/repos/kits23/dataset/kits23.json"
)


def get_clinical_row(data, case_id, headers):
    for row in data:
        if row["case_id"] == case_id:
            return {k: row[k] for k in headers}
    return None


def main():
    # Load metrics
    metrics_pth = Path(__file__).parent / "metrics.json"
    with open(metrics_pth, "r") as f:
        metrics = json.load(f)
    
    # Load clinical data
    with open(CLINICAL_PTH, "r") as f:
        clinical_data = json.load(f)
    
    # Create DataFrame
    headers = [
        # From KiTS
        "case_id",
        "gender",
        "vital_status",
        "vital_days_after_surgery",
        "bmi",
        "age_at_nephrectomy",
        "surgery_type",
        "surgical_procedure",
        "estimated_blood_loss",
        "operative_time",
        "smoking_history",
        "pack_years",
        "radiographic_size",
        "pathologic_size",
        "pathology_t_stage",
        "pathology_n_stage",
        "pathology_m_stage",
        "tumor_isup_grade",
        "positive_resection_margins",
        "sarcomatoid_features",
        "rhabdoid_features",
        "tumor_necrosis",
        "malignant",
        "aua_risk_score",
    ]
    new_headers = [
        # From our metrics
        "ips_2cm_adipose_frac",
        "cnt_2cm_adipose_frac",
        "adipose_frac_diff",
        "ips_hu_at_ad_frac_max",
        "cnt_hu_at_ad_frac_max",
        "hu_diff_at_ad_frac_max"
    ]

    data = []
    for case_id, case_data in metrics.items():
        # Get clinical data
        clinical_row = get_clinical_row(clinical_data, case_id, headers)
        if clinical_row is None:
            print("No clinical data for", case_id)
            continue

        # Get metrics
        ips_metrics = case_data["ips_metrics"]
        cnt_metrics = case_data["cnt_metrics"]

        # Compute adipose fraction difference
        adipose_frac_diff = (
            ips_metrics["2cm_adipose_frac"] - cnt_metrics["2cm_adipose_frac"]
            if all([ips_metrics["2cm_adipose_frac"], cnt_metrics["2cm_adipose_frac"]])
            else None
        )

        # Compute HU difference
        hu_diff = (
            ips_metrics["hu_at_ad_frac_max"] - cnt_metrics["hu_at_ad_frac_max"]
            if all([ips_metrics["hu_at_ad_frac_max"], cnt_metrics["hu_at_ad_frac_max"]])
            else None
        )

        # Append to data
        full_row = []
        full_row.extend([clinical_row[k] for k in headers])
        full_row.extend([
            ips_metrics["2cm_adipose_frac"],
            cnt_metrics["2cm_adipose_frac"],
            adipose_frac_diff,
            ips_metrics["hu_at_ad_frac_max"],
            cnt_metrics["hu_at_ad_frac_max"],
            hu_diff
        ])
        data.append(full_row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers + new_headers)
    df.to_csv(DST_PTH / "clinical_metrics.csv", index=False)

    # Create a binary for pT3+
    df["pT3or4"] = df["pathology_t_stage"].apply(
        lambda x: 1 if str(x)[0] == "3" or str(x)[0] == "4" else 0
    )

    # Create a binary for AUA risk score
    df["aua_high_risk"] = df["aua_risk_score"].apply(
        lambda x: 1 if x in ["high_risk", "very_high_risk"] else 0
    )

    # Create a binary for psm
    df["positive_resection_margins"] = df["positive_resection_margins"].apply(
        lambda x: 1 if str(x) == "True" else 0
    )

    # Create a binary for current or previous smoker
    df["cur_or_prev_smoker"] = df["smoking_history"].apply(
        lambda x: 1 if x in ["current_smoker", "previous_smoker"] else 0
    )

    # Make a binary variable for maligant
    df["malignant"] = df["malignant"].apply(
        lambda x: 1 if str(x) == "True" else 0
    )

    # Run some basic stats models

    # pT3 vs HU in adipose
    model = smf.logit(
        "pT3or4 ~ hu_diff_at_ad_frac_max", data=df
    )
    results = model.fit()
    print(results.summary())

    # Survival vs HU in adipose
    df["bin_vital_status"] = df["vital_status"].apply(
        lambda x: 1 if x == "dead" else 0
    )
    model = smf.phreg(
        "vital_days_after_surgery ~ hu_diff_at_ad_frac_max", data=df,
        status="bin_vital_status"
    )
    results = model.fit()
    print(results.summary())

    # Survival vs. absolute HU in adipose
    model = smf.phreg(
        "vital_days_after_surgery ~ ips_hu_at_ad_frac_max", data=df,
        status="bin_vital_status"
    )
    results = model.fit()
    print(results.summary())

    # Contralateral fat frac vs BMI
    model = smf.ols(
        "cnt_2cm_adipose_frac ~ bmi", data=df
    )
    results = model.fit()
    print(results.summary())

    # Multivariate model to predict survival
    model = smf.phreg(
        (
            "vital_days_after_surgery ~ cnt_2cm_adipose_frac"
                "+ radiographic_size"
        ),
        data=df,
        status="bin_vital_status"
    )
    results = model.fit()
    print(results.summary())

    # Multivariate model to predict pT3+
    model = smf.logit(
        "pT3or4 ~ ips_hu_at_ad_frac_max + radiographic_size", data=df
    )
    results = model.fit()
    print(results.summary())

    # Multivariate model to predict high AUA risk
    model = smf.logit(
        "aua_high_risk ~ ips_hu_at_ad_frac_max + radiographic_size", data=df
    )
    results = model.fit()
    print(results.summary())

    # Predict positive surgical margins
    model = smf.logit(
        "positive_resection_margins ~ ips_hu_at_ad_frac_max", data=df
    )
    results = model.fit()
    print(results.summary())

    # Predict operative time
    model = smf.ols(
        "operative_time ~ ips_2cm_adipose_frac", data=df
    )
    results = model.fit()
    print(results.summary())

    # Predict estimated blood loss
    model = smf.ols(
        "estimated_blood_loss ~ ips_2cm_adipose_frac", data=df
    )
    results = model.fit()
    print(results.summary())

    # Predict HU vals using smoking status
    model = smf.ols(
        "adipose_frac_diff ~ cur_or_prev_smoker", data=df
    )
    results = model.fit()
    print(results.summary())

    # Predict malignant status using HU vals
    model = smf.logit(
        "malignant ~ hu_diff_at_ad_frac_max", data=df
    )
    results = model.fit()
    print(results.summary())


if __name__ == "__main__":
    main()
