""" A collection of functions which are useful for computing the renal score of 
a kidney tumor given semantic segmentation of the kidney and the tumor """

import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label
from scipy.stats import mode

import matplotlib.pyplot as plt

from seg_to_nephrometry.utils import prep_seg_shape, get_centroid


def count_tumor_voxels_by_type(tum_slc, kid_thresh_slc):
    # Set OR of all convex hulls to zeros. Will add to this over time
    convex_or = np.zeros(np.shape(tum_slc), dtype=np.uint8)

    # Get contours of kidney thresholded image
    contours, _ = cv2.findContours(
        kid_thresh_slc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate over connected components and add convex hull to OR
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(convex_or, hull, color=1)

    # Count voxels of each type
    endophytic_count = np.sum(
        np.logical_and(
            np.equal(tum_slc, 1),
            np.equal(convex_or, 1)
        ).astype(np.int32)
    )
    exophytic_count = np.sum(tum_slc) - endophytic_count

    return exophytic_count, endophytic_count


def get_R(max_radius_mm):
    diameter_mm = 2*max_radius_mm
    if diameter_mm >= 70:
        return 3
    elif diameter_mm > 40:
        return 2
    else:
        return 1


def get_E(subregions):
    # Get kidney (not tumor) voxels as though it's the 2nd output of cv2 thresh
    kidney_no_tumor_thresh = 255*np.logical_or(
        np.equal(
            subregions, 1
        ),
        np.greater(
            subregions, 2.5
        )
    ).astype(np.uint8)
    # Get a 3d array with tumor as value 1 only 
    tumor_no_kidney_bin = np.equal(
        subregions, 2
    ).astype(np.uint8)

    tot_exophytic = 0
    tot_endophytic = 0
    for i in range(kidney_no_tumor_thresh.shape[0]):
        if np.sum(tumor_no_kidney_bin[i]) > 0:
            this_exophytic, this_endophytic = count_tumor_voxels_by_type(
                tumor_no_kidney_bin[i], kidney_no_tumor_thresh[i]
            )
            tot_exophytic = tot_exophytic + this_exophytic
            tot_endophytic = tot_endophytic + this_endophytic

    if tot_exophytic == 0:
        return 3, 1.0
    elif tot_endophytic > tot_exophytic:
        return 2, tot_endophytic/(tot_endophytic+tot_exophytic)
    else:
        return 1, tot_endophytic/(tot_endophytic+tot_exophytic)


def get_N(distance_mm):
    if distance_mm >= 7:
        return 1
    elif distance_mm > 4:
        return 2
    else:
        return 3


def get_A(subregions, pixel_width):
    component_bin = np.greater(subregions, 0.5).astype(np.int32)
    component_centroid = get_centroid(component_bin)
    tumor_bin = np.equal(subregions, 2).astype(np.int32)
    tumor_centroid = get_centroid(tumor_bin)
    dst = (tumor_centroid[1] - component_centroid[1])*pixel_width

    # TODO neither?
    if np.abs(dst) < 10:
        return "x", dst
    elif tumor_centroid[1] < component_centroid[1]:
        return "a", dst
    else:
        return "p", dst


def get_L(subregions):
    # Get slices representing polar lines (interior of them)
    start, end = get_polar_line_indicies(subregions)

    # Define midline index/indices
    if (start + end)%2 == 0:
        midline = [(start + end)//2]
    else:
        midline = [(start + end - 1)//2, (start + end + 1)//2]

    # If tumor involves the midline, it is a three
    for m in midline:
        if 2 in subregions[m]:
            return 3, -1

    # Count total tumor pixels
    tumor_volume = np.sum(np.equal(subregions, 2).astype(np.int32))

    # Count tumor pixels between the polar lines
    count = 0
    for i in range(start, end+1):
        count = count + np.sum(np.equal(subregions[i], 2).astype(np.int32))

    # If more than halfway across the line, it's a 3
    # If 0 < t <= 0.5, then it's a 2, else it's a 1
    if count/tumor_volume > 0.5:
        return 3, count/tumor_volume
    if count/tumor_volume > 1e-4:
        return 2, count/tumor_volume
    else:
        return 1, count/tumor_volume


def get_polar_line_indicies(subregions):
    hilum_bin = np.logical_or(
        np.equal(subregions, 3),
        np.equal(subregions, 4)
    ).astype(np.int32)

    idx_1 = -1
    idx_2 = -1
    for i, slc in enumerate(hilum_bin):
        if not 3 in subregions[i] and 4 not in subregions[i]:
            continue 
        hilum_exterior_edge = np.logical_and(
            np.greater(
                convolve2d(
                    slc, np.ones((3,3), dtype=np.int32), mode='same'
                ),
                0
            ),
            np.equal(subregions[i], 0)
        ).astype(np.int32)
        if 1 in hilum_exterior_edge:
            if idx_1 == -1:
                idx_1 = i
            else:
                idx_2 = i

    # print(idx_1, idx_2)

    return (idx_1, idx_2)
