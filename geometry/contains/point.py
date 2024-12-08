"""Functions for testing containment of points in shapes."""

import numpy as np
from numba import njit, float64, boolean
import math
from ..tools import test


@njit(boolean(float64[:,:], float64[:], float64[:]))
def pol(pts_pol: np.array, dom_pol: np.array, xy: np.array) -> boolean:
    """
    Return true if a polygon contains the point xy.

    Sums up the angles of vector pairs from point xy to consecutive pairs of
    vertices  from the polygon exterior. If the angles sum up to 2 pi, the
    point is contained. Source:
    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html

    Parameters
    ----------
    pts_pol : np.array
        DESCRIPTION.
    dom_pol : np.array
        DESCRIPTION.
    xy : np.array
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dom_xy = np.array((xy[0], xy[0], xy[1], xy[1]))
    if not test.overlaps_domain(dom_pol, dom_xy):
        return False

    # Get polygon edges and loop over edges
    vec = xy - pts_pol
    norm = (vec[:,0]**2 + vec[:,1]**2)**0.5
    rad_sum = 0

    # Get edge pairs
    num_edges = len(pts_pol)
    edge_idx = np.arange(num_edges)
    pairs_idx = np.stack((edge_idx, np.roll(edge_idx, -1))).T

    for i, j in pairs_idx:
        inner = vec[i][0] * vec[j][0] + vec[i][1] * vec[j][1]
        norms = norm[i] * norm[j]

        # If norms equals zero, point lies on polygon vertex
        if norms == 0:
            return False

        # Compute angle in radians and add to rad_sum
        ratio = inner / norms
        if ratio < -1:
            ratio = -1
        elif ratio > 1:
            ratio = 1
        rad_sum += np.arccos(ratio)
    return 1.99999999*math.pi < rad_sum < 2.00000001*math.pi



@njit(boolean(float64[:], float64, float64[:]))
def circle(midpoint: np.array, radius: float64, xy: np.array):
    """
    Return True if point lies within a circle.

    Parameters
    ----------
    midpoint : np.array
        midpoint of circle (x,y).
    radius : float64
        Redius of circle.
    xy : np.array
        Coordinates of point to test (x,y).

    Returns
    -------
    boolean
        True if points is within circle.

    """
    return ((xy[0] - midpoint[0])**2 + (xy[1] - midpoint[1])**2) < radius**2


