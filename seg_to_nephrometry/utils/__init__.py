""" A collection of functions which are useful for getting the necessary
information from the volume in order to compute nephrometry metrics """

from pathlib import Path

import numpy as np
import pydicom
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform
from seg_to_nephrometry.utils.pyfastnns import NNS
import matplotlib.pyplot as plt
import time
import cv2
from tqdm import tqdm


def get_centroid(volume):
    coordinates = np.transpose(np.array(np.nonzero(volume)))
    centroid = np.mean(coordinates, axis=0)
    return centroid


def _blur_thresh(vol):
    kernel = np.ones((3,3))/9.0
    ret = np.zeros(np.shape(vol), dtype=np.float32)
    for i in range(vol.shape[0]):
        ret[i] = convolve2d(
            vol[i], kernel, mode="same", boundary="fill", fillvalue=0
        )
    return ret


def _get_distance(c1, c2, x_width=1, y_width=1, z_width=1):
    return np.linalg.norm(
        np.multiply(c1 - c2, np.array((x_width, y_width, z_width))), ord=2
    )


def distance_between_regions(first_coordinates, second_coordinates):
    nns = NNS(first_coordinates)
    _, distance = nns.search(second_coordinates)
    min_distance = np.min(distance)
    return min_distance


def nearest_pair(first_coordinates, second_coordinates):
    nns = NNS(first_coordinates)
    pts, distances = nns.search(second_coordinates)
    min_distance_idx = np.argmin(distances)
    
    sp = second_coordinates[min_distance_idx]
    fp = first_coordinates[pts[min_distance_idx]]

    return fp, sp


def furthest_pair_distance(coordinates):
    coordinates = np.array(coordinates).T
    D = pdist(coordinates)
    return np.nanmax(D)


def get_nearest_rim_point(region_boundaries, pixel_width, slice_thickness):
    # Get coordinates of collecting system voxels
    rim_bin = np.equal(region_boundaries, 5).astype(np.int32)
    rim_coordinates = np.transpose(np.array(np.nonzero(rim_bin)))
    if rim_coordinates.shape[0] == 0:
        raise ValueError("Renal rim could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.transpose(np.array(np.nonzero(tumor_bin)))

    # Scale coordinates such that they correspond to the real world (mm)
    multiplier = np.array(
        [[slice_thickness, pixel_width, pixel_width]]
    ).astype(np.float32)
    rim_coordinates = np.multiply(rim_coordinates, multiplier)
    tumor_coordinates = np.multiply(tumor_coordinates, multiplier)

    nearest_pt, _ = nearest_pair(rim_coordinates, tumor_coordinates)
    return np.divide(nearest_pt, multiplier[0])


def get_distance_to_collecting_system(region_boundaries, pixel_width, 
    slice_thickness):
    # Get coordinates of collecting system voxels
    ucs_bin = np.equal(region_boundaries, 4).astype(np.int32)
    ucs_coordinates = np.transpose(np.array(np.nonzero(ucs_bin)))
    if ucs_coordinates.shape[0] == 0:
        return get_distance_to_sinus(
            region_boundaries, pixel_width, slice_thickness
        )
        # raise ValueError("Collecting system could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.transpose(np.array(np.nonzero(tumor_bin)))

    # Scale coordinates such that they correspond to the real world (mm)
    ucs_coordinates = np.multiply(
        ucs_coordinates, 
        np.array([[slice_thickness, pixel_width, pixel_width]])
    )
    tumor_coordinates = np.multiply(
        tumor_coordinates, 
        np.array([[slice_thickness, pixel_width, pixel_width]])
    )

    # Find nearest point between the two (quickly pls)
    min_distance = distance_between_regions(
        ucs_coordinates, tumor_coordinates
    )

    return min_distance


def get_distance_to_sinus(region_boundaries, pixel_width, 
    slice_thickness):
    # Get coordinates of collecting system voxels
    sinus_bin = np.equal(region_boundaries, 3).astype(np.int32)
    sinus_coordinates = np.array(np.nonzero(sinus_bin), dtype=np.float32).T
    if sinus_coordinates.shape[0] == 0:
        return np.inf
        # raise ValueError("Sinus could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.array(np.nonzero(tumor_bin), dtype=np.float32).T

    # Scale coordinates such that they correspond to the real world (mm)
    multiplier = np.array(
        [[slice_thickness, pixel_width, pixel_width]]
    ).astype(np.float32)

    tumor_coordinates = np.multiply(tumor_coordinates, multiplier) 

    sinus_coordinates = np.multiply(sinus_coordinates, multiplier)

    # Find nearest point between the two (quickly pls)
    min_distance = distance_between_regions(
        sinus_coordinates, tumor_coordinates
    )

    return min_distance



def prep_seg_shape(seg):
    """ Make sure segmentation is of the shape (slices, height, width) """
    if len(seg.shape) > 3:
        return np.reshape(seg, [seg.shape[0], seg.shape[1], seg.shape[2]])
    return seg


def get_pixel_width(dicom_directory):
    """ Returns the distance between adjacent pixel centers in millimeters 
    
    Needs a Path object where the volume dicoms live
    """
    for p in dicom_directory.glob("*"):
        try:
            dcm = pydicom.dcmread(str(p))
            return float(dcm[0x0028, 0x0030].value[0])
        except:
            continue
    raise IOError(
        "Unable to get a pixel spacing value for this directory: {0}".format(
            str(dicom_directory)
        )
    )
    return None


def get_slice_thickness(dicom_directory):
    """ Returns the distance between adjacent slices in millimeters
    
    Needs a Path object where the volume dicoms live
    """
    for p in dicom_directory.glob("*"):
        try:
            dcm = pydicom.dcmread(str(p))
            return float(dcm[0x0018, 0x0050].value)
        except:
            continue
    raise IOError("Unable to get a slices thickness value for this directory")
    return None


def load_volume(dicom_path, plat_id=None):
    if plat_id is not None:
        pth = Path(
            "/home/helle246/data/umnkcid/intermediate/volumes/{}.npy".format(
                plat_id
            )
        )
        if pth.exists():
            print("loading volume from {}".format(str(pth)))
            return np.load(str(pth))
    dcms = [pydicom.dcmread(str(slc)) for slc in dicom_path.glob("*")]
    instance_nums = [int(dcm[0x20,0x13].value) for dcm in dcms]
    spatial_shape = dcms[0].pixel_array.shape
    ret = np.zeros((len(dcms), spatial_shape[0], spatial_shape[1]))
    for i, ind in enumerate(np.argsort(instance_nums).tolist()):
        dcm = dcms[ind]
        data = dcm.pixel_array
        try:
            slope = float(dcm[0x28, 0x1053].value)
        except KeyError:
            slope = 1.0
        try:
            intercept = float(dcm[0x28, 0x1052].value)
        except KeyError:
            intercept = -1024.0 - data[0,0]

        ret[i] = slope*data + intercept

    return ret


def get_interior_seg_boundaries(seg):
    conv_kernel = np.ones((3,3), dtype=np.int32)
    ret = np.zeros(seg.shape, dtype=np.int32)
    for i in range(ret.shape[0]):
        for v in np.unique(seg[i]).tolist():
            if v != 0:
                bin_arr = np.zeros(seg[i].shape, dtype=np.int32)
                bin_arr[seg[i] == v] = 1
                conv = convolve2d(
                    bin_arr, conv_kernel, 
                    mode="same", boundary="fill", fillvalue=0
                )
                bin_arr = np.logical_and(
                    np.greater(bin_arr, 0),
                    np.less(conv, 9)
                )
                ret[i] = ret[i] + v*bin_arr

    return ret


def fill_affected_seg(seg):
    out = np.copy(seg)
    struc = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    )
    unconnected_sinus = np.zeros_like(seg)
    exterior_hull = np.zeros_like(seg)
    binseg = np.greater(seg, 0).astype(np.uint8)
    seg_counts = np.apply_over_axes(np.sum, binseg, [1, 2])

    inds_to_fill = []
    for i in range(seg.shape[0]):
        # Nothing to do if this slice is empty
        if seg_counts[i] != 0:
            inds_to_fill.append(i)

    exp_slices = []
    for i in tqdm(inds_to_fill):
        # Copy this slice
        this_slice = np.copy(binseg[i])

        # Expand this slice
        this_slice_exp = np.greater(
            convolve2d(this_slice, struc, mode="same"),
            0
        ).astype(np.uint8)
        exp_slices.append(this_slice_exp)

        # Flood fill from the top
        mask = np.zeros(
            (this_slice.shape[0] + 2, this_slice.shape[1] + 2), np.uint8
        )
        for_filling = np.copy(this_slice)
        cv2.floodFill(for_filling, mask, (0,0), 1)
        
        # Store the pixels that still are empty
        unconnected_sinus[i][np.less(for_filling, 1)] = 1

        # Wrap the region in a convex hull
        contours, _ = cv2.findContours(
            255*this_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        fincont = contours[0]
        for cont in contours[1:]:
            fincont = np.concatenate((fincont, cont), axis=0)
        hull = cv2.convexHull(fincont)
        hull_filled = np.zeros_like(this_slice)
        cv2.fillConvexPoly(hull_filled, hull, color=1)
        exterior_hull[i][np.logical_and(
            np.equal(hull_filled, 1),
            np.equal(this_slice_exp, 0)
        )] = 1

    # Get connected components within exterior hull
    components, n_cmp = label(exterior_hull, structure=np.ones([3, 3, 3]))

    # Get centroid of interior sinus
    sinus_centroid = get_centroid(unconnected_sinus)

    # ID the sinus component by comparing centroids
    sin_cmp = np.zeros_like(components)
    min_diff = np.inf
    for i in range(1, n_cmp + 1):
        cmp = np.equal(components, i)
        centroid_size = np.sum(cmp)
        if centroid_size < 50:
            continue
        centroid = get_centroid(cmp)
        centroid_diff = np.linalg.norm(sinus_centroid - centroid)
        if centroid_diff < min_diff:
            min_diff = centroid_diff
            sin_cmp = np.copy(cmp)
    
    # Apply fill
    for i, ind in tqdm(list(enumerate(inds_to_fill))):
        exp_slice = exp_slices[i]
        unc_sinus = unconnected_sinus[ind]
        hil_sinus = sin_cmp[ind]
        exp_sinus = np.greater(
            convolve2d(hil_sinus, struc, mode="same"),
            0
        ).astype(np.uint8)
        gap_filler = np.logical_and(
            np.greater(exp_slice, 0),
            np.greater(exp_sinus, 0)
        )
        ret = np.copy(out[ind])
        ret[np.logical_and(
            ret == 0,
            unc_sinus + hil_sinus + gap_filler > 0
        )] = 1

        out[ind] = np.copy(ret)

        # Fill in the gaps
        cv2.floodFill(ret, None, (0,0), 1)
        out[ind][np.less(ret, 1)] = 1
        

    return out        


def get_affected_kidney_subregions(seg, vol):
    print("GETTING AFFECTED SUBREGIONS")
    # Get affected region, set seg to zero elsewhere
    print("Isolating side")
    components, _ = label(seg, structure=np.ones((3,3,3)))
    tumor_pixel_components = components[seg == 2]
    try:
        affected_component_ind = mode(tumor_pixel_components, axis=None)[0][0]
    except IndexError:
        print("Warning: could not identify tumor subregion")
        return None
    affected_seg = np.where(
        np.equal(components, affected_component_ind),
        seg, np.zeros(np.shape(seg), dtype=seg.dtype)
    )

    # Fill affected seg so that sinus can be identified
    print("Filling hilum and sinus")
    affected_seg = fill_affected_seg(affected_seg)

    # Get outer boundary of affected region
    print("Getting interior seg boundaries")
    affected_region = np.greater(
        affected_seg, 0.5
    ).astype(seg.dtype)
    affected_interior = get_interior_seg_boundaries(affected_region)

    # Get sinus by blurring volume and finding kidney pixels below the 
    # threshold
    print("Identifying sinus")
    conv_kernel = np.ones((3,3), dtype=np.float32)/9
    blurred_volume = np.zeros(np.shape(vol))
    for i in range(vol.shape[0]):
        blurred_volume[i] = convolve2d(
            vol[i], conv_kernel, 
            mode='same', boundary='fill', fillvalue=vol[0,0,0]
        )
    sinus = np.where(
        np.logical_and(
            np.logical_and(
                np.less(blurred_volume, -30),
                np.greater(affected_seg, 0)
            ),
            np.less(affected_interior, 0.5)
        ),
        np.ones(np.shape(seg), dtype=seg.dtype),
        np.zeros(np.shape(seg), dtype=seg.dtype)
    )
    grown_sinus = sinus.copy()
    big_conv_kernel = np.ones((15,15), dtype=np.int32)
    for i in range(grown_sinus.shape[0]):
        grown_sinus[i] = np.where(
            np.greater(
                convolve2d(grown_sinus[i], big_conv_kernel, mode='same'), 0
            ),
            np.ones(np.shape(grown_sinus[i]), dtype=seg.dtype),
            np.zeros(np.shape(grown_sinus[i]), dtype=seg.dtype)
        )
    # Set sinus equal to largest connectect sinus component
    components, _ = label(grown_sinus, structure=np.ones((3,3,3)))
    try:
        largest_component = mode(components[components != 0], axis=None)[0][0]
    except IndexError:
        largest_component = -1
    sinus = np.logical_and(
        np.equal(components, largest_component),
        np.equal(sinus, 1)
    ).astype(seg.dtype)

    print("Identifying UCS")
    ucs = np.zeros(np.shape(sinus), dtype=seg.dtype)
    for i in range(sinus.shape[0]):
        if 1 not in sinus[i]:
            continue
        # Compute binary image of convex hull of sinus
        contours, _ = cv2.findContours(
            255*sinus[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        fincont = contours[0]
        for cont in contours[1:]:
            fincont = np.concatenate((fincont, cont), axis=0)
        hull = cv2.convexHull(fincont)
        cv2.fillConvexPoly(ucs[i], hull, color=1)

        # Everything labeled kidney but not sinus in this is ucs
        ucs[i] = np.logical_and(
            np.logical_and(
                np.less(sinus[i], 1),
                np.greater(affected_seg[i], 0)
            ),
            np.greater(ucs[i], 0)
        ).astype(seg.dtype)

    # Get rim
    print("Identifying rim")
    rim = np.logical_and(
        np.greater(affected_interior, 0),
        np.logical_and(
            np.less(sinus, 1),
            np.less(ucs, 1)
        )
    ).astype(seg.dtype)

    print("Returning result with correct labels")
    subregions = np.greater(affected_seg, 0).astype(seg.dtype)
    subregions = subregions + 2*sinus + 3*ucs + 4*rim
    subregions = np.where(np.equal(affected_seg, 2), affected_seg, subregions)
    return subregions

