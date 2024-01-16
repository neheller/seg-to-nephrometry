""" A collection of functions which are useful for computing the c-index of a 
kidney tumor given semantic segmentation of the kidney and the tumor """

import numpy as np
from scipy.ndimage.measurements import label
from scipy.stats import mode

from seg_to_nephrometry.utils import prep_seg_shape, furthest_pair_distance


def _get_plane_center(seg_slice):
    """ Get the centroid pixel of a binary 2d slice of the segmentation """
    # Get center of mass in y-direction
    y_margin = np.sum(seg_slice, axis=1)
    y_range = np.arange(0, y_margin.shape[0])
    y_center = np.sum(np.multiply(y_margin, y_range))/np.sum(y_margin)
    # Get center of mass in x-direction
    x_margin = np.sum(seg_slice, axis=0)
    x_range = np.arange(0, x_margin.shape[0])
    x_center = np.sum(np.multiply(x_margin, x_range))/np.sum(x_margin)
    return int(y_center), int(x_center)


def get_max_tumor_radius(label_slice):
    """ Get the maximum radius of the tumor in pixels """
    # Optimally takes boundaries
    binseg = np.equal(label_slice, 2).astype(np.uint8)
    return furthest_pair_distance(np.nonzero(binseg))/2


def get_tumor_center(seg):
    """ Get the centroid of the tumor """
    seg = prep_seg_shape(seg)
    binseg = np.equal(seg, 2).astype(np.int32)
    sums = np.sum(binseg, axis=(1,2))
    z_center = np.argmax(sums)
    y_center, x_center = _get_plane_center(binseg[z_center])
    return np.array((z_center, y_center, x_center), dtype=np.int32)


def get_kidney_center(seg):
    """ Get the centroid of the kidney that the tumor is touching """
    seg = prep_seg_shape(seg)
    binseg = np.greater(seg, 0.5).astype(np.int32)
    sums = np.sum(binseg, axis=(1,2))
    nonempty = np.arange(0, sums.shape[0])[np.greater(sums, 0)]
    start = np.min(nonempty)
    end = np.max(nonempty)
    z_center = (start+end)//2   
    y_center, x_center = _get_plane_center(binseg[z_center])
    return np.array((z_center, y_center, x_center), dtype=np.int32)
