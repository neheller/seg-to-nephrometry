# Copyright 2018 Nicholas Heller. All Rights Reserved.
#
# Licensed under the MIT License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" A script to compute nephrometry scores for a kidney tumor given a CT scan 
in DICOM format along with a npy file of the same shape with a semantic 
segmentation of the kidneys and the tumor """

import argparse
from pathlib import Path

import numpy as np

from seg_to_nephrometry.utils import get_distance_to_collecting_system
from seg_to_nephrometry.utils import get_distance_to_sinus
from seg_to_nephrometry.utils import get_interior_seg_boundaries
from seg_to_nephrometry.utils import get_affected_kidney_subregions
import seg_to_nephrometry.scores.cindex as cindex
import seg_to_nephrometry.scores.renal as renal
import seg_to_nephrometry.scores.padua as padua


def get_custom(subregions, subregion_boundaries, slice_thickness, pixel_width, 
    verbose=True):
    c = slice_thickness*pixel_width*pixel_width # Voxel volume
    return (
        c*np.sum(np.equal(subregion_boundaries, 2).astype(np.int32)), # Shell V
        c*np.sum(np.equal(subregions, 2).astype(np.int32)) # Tumor V
    )


def get_cindex(seg, subregions, boundaries, slice_thickness, pixel_width, 
    verbose=True):
    # Get centers of largest slice for tumor and kidney
    tumor_center = cindex.get_tumor_center(seg)
    kidney_center = cindex.get_kidney_center(subregions)

    # Get distances 
    res = np.abs(tumor_center - kidney_center)
    z_px = res[0]
    y_px = res[1]
    x_px = res[2]
    plane_distance = np.sqrt(
        (x_px*pixel_width)**2 + (y_px*pixel_width)**2
    )
    z_distance = z_px*slice_thickness
    distance = np.sqrt(plane_distance**2 + z_distance**2)

    # Get max tumor radius for ratio
    max_tumor_radius = cindex.get_max_tumor_radius(
        boundaries[tumor_center[0]]
    )*pixel_width

    # Compute C-index 
    try:
        score = distance/max_tumor_radius
    except ZeroDivisionError:
        print("Error, distance between tumor and kidney centroid is zero.")
        score = np.inf

    if verbose:
        sr = "***********************"
        print("{0} Computed C-index: {1:.3f} {0}".format(sr, score))
        print("Plane distance:   {0:.3f} mm".format(plane_distance))
        print("Slice distance:   {0:.3f} mm".format(z_distance))
        print("Max tumor radius: {0:.3f} mm".format(max_tumor_radius))

    return {
        "composite": score,
        "radius": max_tumor_radius,
        "z_distance": z_distance,
        "plane_distance": plane_distance
    }


def get_renal(subregions, boundaries, slice_thickness, pixel_width, 
    verbose=True):
    """ Get individual components """
    # R (Radius)
    tumor_center = cindex.get_tumor_center(subregions)
    max_tumor_radius = cindex.get_max_tumor_radius(
        boundaries[tumor_center[0]]
    )*pixel_width
    R = renal.get_R(max_tumor_radius)
    R_cont = max_tumor_radius*2

    # E (Exophytic)
    E, E_cont = renal.get_E(subregions)

    # N (Nearness to Collecting System)
    N_cont = get_distance_to_collecting_system(
        boundaries, pixel_width, slice_thickness
    )
    N = renal.get_N(N_cont)

    # A (Anterior vs. Posterior)
    A, A_cont = renal.get_A(subregions, pixel_width)

    # L (Location Relative to Polar Lines)
    L, L_cont = renal.get_L(subregions)

    """ Compute Score """
    score = str(R + E + N + L) + A

    # Maybe print and return
    if verbose:
        sr = "***********************"
        print("{0} Computed RENAL: {1:s} {0}".format(sr, score))
        print("R: {0:d}".format(R))
        print("E: {0:d}".format(E))
        print("N: {0:d}".format(N))
        print("A: {0:s}".format(A))
        print("L: {0:d}".format(L))

    return {
        "composite": score,
        "R": R,
        "E": E,
        "N": N,
        "A": A,
        "L": L,
        "R_cont": R_cont,
        "E_cont": E_cont,
        "N_cont": N_cont,
        "A_cont": A_cont,
        "L_cont": L_cont
    }


def get_padua(subregions, boundaries, slice_thickness, pixel_width, 
    prev_renal=None, verbose=True):
    """ Get individual components """
    # (a) Location Relative to Sinus Lines
    a, a_cont = padua.get_a(subregions)

    # (b) Location Relative to Renal Rim (Lateral vs Medial)
    b, b_cont = padua.get_b(subregions, boundaries, pixel_width, slice_thickness)

    # (c) Sinus Involvement
    c_cont = get_distance_to_sinus(
        boundaries, pixel_width, slice_thickness
    )
    c = padua.get_c(c_cont)

    # (d) Collecting System Involvement
    d_cont = get_distance_to_collecting_system(
        boundaries, pixel_width, slice_thickness
    )
    d = padua.get_d(d_cont)

    # (e) Exophycity
    e, e_cont = renal.get_E(subregions)
    
    # (f) Radius
    tumor_center = cindex.get_tumor_center(subregions)
    max_tumor_radius = cindex.get_max_tumor_radius(
        boundaries[tumor_center[0]]
    )*pixel_width
    f = renal.get_R(max_tumor_radius)
    f_cont = max_tumor_radius*2

    # A (Anterior vs. Posterior)
    if prev_renal is not None:
        A = prev_renal[4]
    else:
        A, A_cont = renal.get_A(subregions, pixel_width)

    """ Compute Score """
    score = str(a + b + c + d + e + f) + A

    # Maybe print and return
    if verbose:
        sr = "***********************"
        print("{0} Computed PADUA: {1:s} {0}".format(sr, score))
        print("a: {0:d}".format(a))
        print("b: {0:d}".format(b))
        print("c: {0:d}".format(c))
        print("d: {0:d}".format(d))
        print("e: {0:d}".format(e))
        print("f: {0:d}".format(f))
        print("suffix: {0:s}".format(A))

    return {
        "composite": score,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "f": f,
        "a_cont": a_cont,
        "b_cont": b_cont,
        "c_cont": c_cont,
        "d_cont": d_cont,
        "e_cont": e_cont,
        "f_cont": f_cont,
        "A_cont": A_cont
    }


def compute_scores_nib(vol_nib, seg_nib, subregions=None, verbose=False):
    # Get useful meta
    pixel_width = np.abs(vol_nib.affine[0,2])
    slice_thickness = np.abs(vol_nib.affine[2,0])

    # Get voxel data
    seg = seg_nib.get_fdata().astype(np.uint8)
    vol = vol_nib.get_fdata()

    # Sanity checking
    has_kidney = np.sum(np.greater(seg, 0.5).astype(np.float32)) > 0
    if not has_kidney:
        print("This segmentation does not have any kidney pixels")
        return
    has_tumor = np.sum(np.greater(seg, 1.5).astype(np.float32)) > 0
    if not has_tumor:
        print("This segmentation does not have any tumor pixels")
        return
    
    # 0 background, 1 kidney, 2 tumor, 3 sinus, 4 ucs, 5 rim
    if subregions is None:
        subregions = get_affected_kidney_subregions(seg, vol)
    subregion_boundaries = get_interior_seg_boundaries(subregions)

    if verbose:
        print("Finished loading data. Computing scores...\n")

    c = None
    r = None
    p = None
    # Compute c-index, leave verbose True so it prints
    c = get_cindex(
        seg, subregions, subregion_boundaries, slice_thickness, pixel_width,
        verbose=verbose
    )
    if verbose:
        print()
    # Compute RENAL, leave verbose True so it prints
    r = get_renal(
        subregions, subregion_boundaries, slice_thickness, pixel_width,
        verbose=verbose
    )
    if verbose:
        print()
    # Compute PADUA, leave verbose True so it prints
    p = get_padua(
        subregions, subregion_boundaries, slice_thickness, pixel_width,
        verbose=verbose
    )
    if verbose:
        print()

    return c, r, p
