import json
import argparse
from pathlib import Path

import nibabel as nib

from seg_to_nephrometry import compute_scores_nib


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img-pth', type=str, required=True,
        help='Path to the image pixel data'
    )
    parser.add_argument(
        '--seg-pth', type=str, required=True,
        help='Path to the segmentation of the image'
    )
    parser.add_argument(
        '--out-json-pth', type=str, required=True,
        help='Path to the output file (will be overwritten if exists)'
    )
    args = parser.parse_args()

    # Load image and segmentation
    img_nib = nib.load(args.img_pth)
    seg_nib = nib.load(args.seg_pth)

    # Ensure output path can be written
    out_json_pth = Path(args.out_json_pth)
    assert out_json_pth.suffix == '.json', "Output file must be a JSON file"
    with out_json_pth.open('w') as _:
        pass

    # Get scores
    out = compute_scores_nib(img_nib, seg_nib)

    # Write scores to file
    with out_json_pth.open('w') as f:
        json.dump(out, f, indent=4)


if __name__ == '__main__':
    main()
