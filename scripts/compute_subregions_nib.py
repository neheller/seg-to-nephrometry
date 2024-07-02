import json
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from seg_to_nephrometry.utils import get_affected_kidney_subregions
from seg_to_nephrometry import standardize_orientation


def prep_affected_subregions(vol_nib, seg_nib):
    # Transform to expected orientation
    vol_nib, seg_nib = standardize_orientation(vol_nib, seg_nib)

    # Get voxel data
    seg = np.round(seg_nib.get_fdata()).astype(np.uint8)
    vol = vol_nib.get_fdata().astype(np.float32)

    # Get affected subregions
    subregions = get_affected_kidney_subregions(seg, vol)

    # Return
    sub_reg_nib = nib.Nifti1Image(subregions, seg_nib.affine, seg_nib.header)
    return sub_reg_nib


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
        '--out-nib-pth', type=str, required=True,
        help='Path to the output file (will be overwritten if exists)'
    )
    args = parser.parse_args()

    # Load image and segmentation
    img_nib = nib.load(args.img_pth)
    seg_nib = nib.load(args.seg_pth)

    # Get scores
    out_nib = prep_affected_subregions(img_nib, seg_nib)

    # Save result
    nib.save(out_nib, args.out_nib_pth)


if __name__ == '__main__':
    main()
