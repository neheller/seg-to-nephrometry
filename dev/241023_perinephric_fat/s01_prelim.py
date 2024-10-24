import os
import json
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
import imageio as iio
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from scipy.ndimage import label
import matplotlib.pyplot as plt

from seg_to_nephrometry.utils import get_affected_kidney_subregions


# Resolve path to KiTS23 repository
KITS23_PATH = Path(os.environ.get("KITS23_PATH")).resolve(strict=True)

# Create directory for intermediate results
OUT_PTH = Path(__file__).parent / "intermediate"
OUT_PTH.mkdir(exist_ok=True)


def load_data_and_subregions(case_id: str, cache: bool=True):
    case_path = KITS23_PATH / "dataset" / case_id
    image_path = case_path / "imaging.nii.gz"
    seg_path = case_path / "segmentation.nii.gz"
    img_nib = nib.load(str(image_path))
    seg_nib = nib.load(str(seg_path))

    # Extract pixel data
    img_np = img_nib.get_fdata()
    seg_np = seg_nib.get_fdata().astype(int)

    # Don't actually need subregion data anymore
    subr_nib = None
    subr_np = None
    return {
        "img_nib": img_nib,
        "img_np": img_np,
        "seg_nib": seg_nib,
        "seg_np": seg_np,
        "subr_nib": subr_nib,
        "subr_np": subr_np
    }


def visualize_frame(
    out_pth, i, render_repr_img, exp_rim, expu_rim, dists, mean_vals,
    fracs_adipose, rim_vals, rm_extraneous
):
    # Save an overlay
    overlay = np.copy(render_repr_img)
    overlay[exp_rim > 0] = 255
    overlay_pth = out_pth / f"rim_overlay_{i:02d}.png"
    iio.imsave(str(overlay_pth), overlay.astype(np.uint8))
    uoverlay = np.copy(render_repr_img)
    uoverlay[expu_rim > 0] = 255
    uoverlay_pth = out_pth / f"undeterred_rim_overlay_{i:02d}.png"
    iio.imsave(str(uoverlay_pth), uoverlay.astype(np.uint8))

    # Create a plot
    plot_pth = out_pth / f"rim_plot_{i:02d}.png"
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].hist(rim_vals, bins=20, range=(-200, -20))
    ax[0].set_title("Histogram of Perinephric HU Values")
    ax[1].plot(dists, mean_vals)
    # ax[1].set_title("Mean Intensity in Rim")
    # ax[1].set_xlabel("Distance Away (mm)")
    ax[1].set_ylabel("Mean Intensity")
    ax[1].set_ylim(-200, -20)
    ax[1].set_xlim(0, 30)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[2].plot(dists, fracs_adipose)
    # ax[2].set_title("Fraction of Adipose in Rim")
    ax[2].set_xlabel("Distance Away (mm)")
    ax[2].set_ylabel("Fraction of Adipose")
    ax[2].set_ylim(0, 1.0)
    ax[2].set_xlim(0, 30)
    ax[2].grid()
    ax[2].set_axisbelow(True)
    plt.savefig(plot_pth)
    plt.close()

    # Concatenate with overlayed rim image on left
    sbs_pth = out_pth / f"rim_sbs_{i:02d}.png"
    rim_overlay = cv2.imread(str(overlay_pth))
    plt_img = cv2.imread(str(plot_pth))
    plt_img = cv2.resize(plt_img, (rim_overlay.shape[1], rim_overlay.shape[0]))
    # Make rim_overlay in color
    # rim_overlay = np.stack([rim_overlay, rim_overlay, rim_overlay], axis=2)
    plot_img = np.concatenate((rim_overlay, plt_img), axis=1)
    cv2.imwrite(str(sbs_pth), plot_img)

    # Remove extraneous files
    if rm_extraneous:
        overlay_pth.unlink()
        uoverlay_pth.unlink()
        plot_pth.unlink()


def generate_traces(
    repr_img, repr_seg, repr_subr, total_steps, buffer_steps, mm_per_step,
    viz_pth=None, rm_extraneous=True
):
    # Initialize visualization
    if viz_pth is not None:
        viz_pth.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(repr_img, -128, 256)
        normed = (clipped + 128) / 384
        render_repr_img = (normed * 255).astype(np.uint8)

    # Slightly blur repr_img for stability
    blurred_repr_img = cv2.GaussianBlur(repr_img, (5, 5), 0)

    # Convolve to expand rim
    prev_mask = repr_subr > 0
    prev_mask_undeterred = np.copy(prev_mask)
    rim_seg = np.zeros_like(repr_seg)
    rim_seg[repr_subr > 0] = 1
    undeterred_rim = np.copy(rim_seg)
    
    # Initialize aggregators
    dists = []
    mean_vals = []
    std_vals = []
    fracs_adipose = []
    tots_adipose = []
    for i in range(total_steps + 1):
        # Expand the size of the rims
        exp_rim = binary_dilation(rim_seg, iterations=1)
        expu_rim = binary_dilation(undeterred_rim, iterations=1)

        # Remove parts that were part of previous iterations, or the interior
        # of the kidney
        exp_rim[prev_mask] = 0
        expu_rim[prev_mask_undeterred] = 0

        # Mask out parts of the rim that are too bright, for later iterations
        # This doesn't apply to the undeterred rim
        if i > buffer_steps:
            exp_rim[blurred_repr_img > -20] = 0
            exp_rim[blurred_repr_img < -200] = 0

        # Remove parts of the rim that are too small
        # This doesn't apply to the undeterred rim
        labeled, n_labels = label(exp_rim, structure=np.ones((3, 3)))
        for label_idx in range(1, n_labels + 1):
            if np.sum(labeled == label_idx) < 15:
                exp_rim[labeled == label_idx] = 0

        # Update rim_seg to new form
        rim_seg = exp_rim
        undeterred_rim = expu_rim

        # Update prev_mask to include the new rim so it's not double counted
        # in the future
        prev_mask[rim_seg > 0] = 1
        prev_mask_undeterred[undeterred_rim > 0] = 1

        # Filter out but don't remove the rim too close to the kidney
        met_rim_seg = np.copy(rim_seg)
        met_rim_seg[-200 < blurred_repr_img < -20] = 0

        # Compute metrics
        dist_away = i * mm_per_step
        rim_area = np.sum(met_rim_seg)
        undeterred_rim_area = np.sum(undeterred_rim)
        rim_vals = list(repr_img[met_rim_seg > 0])
        frac_adipose = rim_area/undeterred_rim_area
        mean_rim_val = np.mean(rim_vals)
        std_rim_val = np.std(rim_vals)

        # Add to aggregators
        dists.append(dist_away)
        mean_vals.append(mean_rim_val)
        std_vals.append(std_rim_val)
        fracs_adipose.append(frac_adipose)
        tots_adipose.append(rim_area)

        # Visualize, if requested
        if viz_pth is not None:
            visualize_frame(
                viz_pth, i, render_repr_img, met_rim_seg, expu_rim, dists,
                mean_vals, fracs_adipose, rim_vals, rm_extraneous
            )

    # Return final results
    return {
        "dists": [float(x) for x in dists],
        "mean_vals": [float(x) for x in mean_vals],
        "std_vals": [float(x) for x in std_vals],
        "fracs_adipose": [float(x) for x in fracs_adipose],
        "tots_adipose": [float(x) for x in tots_adipose]
    }


def main():
    for case_ind in range(589):
        # Skip cases between 300 and 400
        if 300 <= case_ind < 400:
            continue
        
        # Get case_id
        case_id = f"case_{case_ind:05d}"

        # Load a single case
        subr_dat = load_data_and_subregions(case_id)

        # Determine how many millimeters are in each step
        mm_per_step = np.max(np.abs(subr_dat["img_nib"].affine[0, :3]))

        # Determine how many buffer steps we need to get about 0.3 cm away
        buffer_steps = int(np.ceil(3 / mm_per_step))

        # Determine how many steps we need to get about 3 cm away
        total_steps = int(np.ceil(30 / mm_per_step))

        # Start with one for visualization -- frame with largest tumor
        sel_ind = np.equal(subr_dat["subr_np"], 2).sum(axis=(1, 2)).argmax()
        repr_img = subr_dat["img_np"][sel_ind, :, :]
        repr_seg = subr_dat["seg_np"][sel_ind, :, :]
        repr_subr = subr_dat["seg_np"][sel_ind, :, :]

        # Generate traces
        out_pth = OUT_PTH / case_id / f"frame_{sel_ind:03d}"
        frame_data = generate_traces(
            repr_img, repr_seg, repr_subr, total_steps, buffer_steps,
            mm_per_step, viz_pth=out_pth
        )

        # Select axial slice containing largest tumor
        # sel_ind = np.equal(subr_dat["subr_np"], 2).sum(axis=(1, 2)).argmax()
        frame_queue = []
        for sel_ind in range(subr_dat["img_nib"].shape[0]):
            if np.equal(subr_dat["subr_np"][sel_ind], 2).sum() < 1:
                continue
            frame_queue.append(sel_ind)

        print("Working on", case_id)
        data_by_frame = []
        for sel_ind in tqdm(frame_queue):
            repr_img = subr_dat["img_np"][sel_ind, :, :]
            repr_seg = subr_dat["seg_np"][sel_ind, :, :]
            repr_subr = subr_dat["seg_np"][sel_ind, :, :]

            # Generate traces
            # out_pth = OUT_PTH / case_id / f"frame_{sel_ind:03d}"
            out_pth = None
            frame_data = generate_traces(
                repr_img, repr_seg, repr_subr, total_steps, buffer_steps,
                mm_per_step, viz_pth=out_pth
            )
            data_by_frame.append(frame_data)

        # Save results
        with (OUT_PTH / f"{case_id}_data.json").open("w") as f:
            json.dump(data_by_frame, f)


if __name__ == "__main__":
    main()
