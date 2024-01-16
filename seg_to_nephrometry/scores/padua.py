""" A collection of functions which are useful for computing the padua score of
a kidney tumor given semantic segmentation of the kidney and the tumor """

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label
from scipy.stats import mode

from seg_to_nephrometry.utils import prep_seg_shape, get_nearest_rim_point, get_centroid


# Location relative to sinus lines
def get_a(subregions):
    # Get slices representing polar lines (interior of them)
    start, end = get_sinus_line_indicies(subregions)

    # Count total tumor pixels
    tumor_volume = np.sum(np.equal(subregions, 2).astype(np.int32))

    # Count tumor pixels between the sinus lines
    count = 0
    for i in range(start, end+1):
        count = count + np.sum(np.equal(subregions[i], 2).astype(np.int32))

    if count/tumor_volume > 0.5:
        return 2, count/tumor_volume
    else:
        return 1, count/tumor_volume


# Closer to medial or lateral rim
def get_b(subregions, boundaries, pixel_width, slice_thickness):
    # Find nearest point on rim
    rim_pt = get_nearest_rim_point(boundaries, pixel_width, slice_thickness)

    # Find centroid of kidney
    affected_centroid = get_centroid(
        np.greater(subregions, 0.5).astype(np.int32)
    )

    # Classify that rim point as medial or lateral
    if affected_centroid[2] > subregions.shape[2]//2:
        medial_side = -1
    else:
        medial_side = 1

    if (rim_pt[2] - affected_centroid[2])*medial_side > 0:
        return 2, (rim_pt[2] - affected_centroid[2])*medial_side
    else:
        return 1, (rim_pt[2] - affected_centroid[2])*medial_side


# Sinus involvement 
def get_c(distance_mm):
    if distance_mm < 2:
        return 2
    else:
        return 1


# Collecting system involvement
def get_d(distance_mm):
    if distance_mm < 2:
        return 2
    else:
        return 1


# Endophycity - taken care of by function from "RENAL"
# def get_e(seg):
#     pass


# Radius - taken care of by function from "RENAL"
# def get_f(max_radius_mm):
#     pass

# Anterior or Posterior suffix taken care of by function from "RENAL"


def get_sinus_line_indicies(subregions):
    idx_1 = -1
    idx_2 = -1
    for i, slc in enumerate(subregions):
        if 3 not in slc and 4 not in slc:
            continue
        if idx_1 == -1:
            idx_1 = i
        else:
            idx_2 = i

    # print(idx_1, idx_2)

    return idx_1, idx_2
