import os
import json
from pathlib import Path


# Create directory for intermediate results
INT_PTH = Path(__file__).parent / "intermediate_"
DST_PTH = Path(__file__).parent / "agg_.json"


def main():
    agg_data = {}

    # Load intermediate date for each case
    for case_num in range(589):
        # Skip range from 300 to 400
        if 300 <= case_num <= 400:
            continue

        case_id = f"case_{case_num:05d}"
        ips_pth = INT_PTH / f"{case_id}_data.json"
        cnt_pth = INT_PTH / f"{case_id}_contralateral_mask_data.json"
        try:
            with open(ips_pth, "r") as f:
                ips_data = json.load(f)
        except Exception:
            ips_data = None
        try:
            with open(cnt_pth, "r") as f:
                cnt_data = json.load(f)
        except Exception:
            cnt_data = None
        
        agg_data[case_id] = {
            "ips_data": ips_data,
            "cnt_data": cnt_data
        }

    # Save data
    with open(DST_PTH, "w") as f:
        json.dump(agg_data, f, indent=2)


if __name__ == "__main__":
    main()
