import json
from pathlib import Path

import matplotlib.pyplot as plt


# Make destination folder
DST_PTH = Path(__file__).parent / "figures"
DST_PTH.mkdir(exist_ok=True)


def main():
    metrics_pth = Path(__file__).parent / "metrics.json"
    with open(metrics_pth, "r") as f:
        metrics = json.load(f)

    # Plot histogram of 2cm adipose fraction
    ips_adipose_fracs = [
        v["ips_metrics"]["2cm_adipose_frac"] for v in metrics.values()
    ]
    cnt_adipose_fracs = [
        v["cnt_metrics"]["2cm_adipose_frac"] for v in metrics.values()
        if v["cnt_metrics"]["2cm_adipose_frac"] is not None
    ]
    plt.hist(ips_adipose_fracs, bins=20, label="Ipsilateral", alpha=0.5)
    plt.hist(cnt_adipose_fracs, bins=20, label="Contralateral", alpha=0.5)
    plt.title("Histogram of 2cm Adipose Fraction")
    plt.legend()
    plt.savefig(DST_PTH / "hist_2cm_adipose_frac.png")
    plt.close()

    # Plot histogram of 2cm adipose fraction difference
    adipose_diffs = [
        ips - cnt
        for ips, cnt in zip(ips_adipose_fracs, cnt_adipose_fracs)
        if not any([x is None for x in [ips, cnt]])
    ]
    plt.hist(adipose_diffs, bins=20)
    plt.title("Histogram of 2cm Adipose Fraction Difference")
    plt.savefig(DST_PTH / "hist_2cm_adipose_frac_diff.png")
    plt.close()

    # Plot histogram of HU at max adipose fraction
    ips_hu_vals = [
        v["ips_metrics"]["hu_at_ad_frac_max"] for v in metrics.values()
    ]
    cnt_hu_vals = [
        v["cnt_metrics"]["hu_at_ad_frac_max"] for v in metrics.values()
        if v["cnt_metrics"]["hu_at_ad_frac_max"] is not None
    ]
    plt.hist(ips_hu_vals, bins=20, label="Ipsilateral", alpha=0.5)
    plt.hist(cnt_hu_vals, bins=20, label="Contralateral", alpha=0.5)
    plt.title("Histogram of HU at Max Adipose Fraction")
    plt.legend()
    plt.savefig(DST_PTH / "hist_hu_at_ad_frac_max.png")
    plt.close()

    # Plot histogram of HU diff at max adipose fraction
    hu_diffs = [
        ips - cnt
        for ips, cnt in zip(ips_hu_vals, cnt_hu_vals)
        if not any([x is None for x in [ips, cnt]])
    ]
    plt.hist(hu_diffs, bins=20)
    plt.title("Histogram of HU Difference at Max Adipose Fraction")
    plt.savefig(DST_PTH / "hist_hu_diff_at_ad_frac_max.png")
    plt.close()


if __name__ == "__main__":
    main()
