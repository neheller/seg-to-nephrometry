import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_2cm_adipose_frac(data):
    # Initialize aggregators to zero
    numerator = 0.0
    denominator = 0.0

    # Iterate over slices
    for slice_data in data:
        # Get slice with max adipose to start at
        max_frac_ind = np.argmax(slice_data["fracs_adipose"])

        # Iterate until 2cm
        cur_slice = max_frac_ind
        dist = slice_data["dists"][cur_slice]
        while dist <= 20:
            # Get adipose fraction at this step
            adipose_frac = slice_data["fracs_adipose"][cur_slice]
            adipose_tot = slice_data["tots_adipose"][cur_slice]

            # If no adipose, skip
            if adipose_tot > 1 and not np.isnan(adipose_frac):
                # Update aggregators
                numerator += adipose_tot
                denominator += adipose_tot/adipose_frac

            # Move to next slice
            cur_slice += 1
            dist = slice_data["dists"][cur_slice]
        
    # Compute final fraction
    if denominator == 0:
        print("No denominator for 2cm_adipose_frac")
        return None
    return numerator/denominator


def get_hu_at_ad_frac_max(data):
    # Initialize aggregators to zero
    numerator = 0.0
    denominator = 0.0

    # Iterate over slices
    for slice_data in data:
        # Get slice with max adipose to start at
        max_frac_ind = np.argmax(slice_data["fracs_adipose"])
        tot_adipose = slice_data["tots_adipose"][max_frac_ind]

        # Get HU mean at this slice
        hu_mean = slice_data["mean_vals"][max_frac_ind]

        # Ensure it's not NaN
        if not np.isnan(hu_mean) and tot_adipose > 1:
            # Update aggregators
            numerator += hu_mean
            denominator += 1
    
    # Compute final fraction
    if denominator == 0:
        print("No denominator for ad_frac_max")
        return None
    return numerator/denominator


def main():
    # Load agg_.json
    agg_pth = Path(__file__).parent / "agg_.json"
    with open(agg_pth, "r") as f:
        agg_data = json.load(f)

    metrics = {}

    case_id_q = sorted(list(agg_data.keys()))

    # Compute aggregate statistics for each case
    for case_id in tqdm(list(case_id_q)):
        case_data = agg_data[case_id]
        ips_data = case_data["ips_data"]
        cnt_data = case_data["cnt_data"]

        # Skip cases with missing data
        if ips_data is None or cnt_data is None:
            print("Skipping case", case_id)
            continue

        # Compute metrics for ipsilateral kidney
        ips_metrics = {
            "2cm_adipose_frac": get_2cm_adipose_frac(ips_data),
            "hu_at_ad_frac_max": get_hu_at_ad_frac_max(ips_data),
        }
        cnt_metrics = {
            "2cm_adipose_frac": get_2cm_adipose_frac(cnt_data),
            "hu_at_ad_frac_max": get_hu_at_ad_frac_max(cnt_data),
        }
        metrics[case_id] = {
            "ips_metrics": ips_metrics,
            "cnt_metrics": cnt_metrics,
        }

        # If any of these are NaN, print the raw data for inspection
        if (
            any([x is None or np.isnan(x) for x in ips_metrics.values()]) or
            any([x is None or np.isnan(x) for x in cnt_metrics.values()])
        ):
            print("Case", case_id)

    with open(Path(__file__).parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
