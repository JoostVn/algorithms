import numpy as np
from numba import njit, float64
from ..tools import test
from ..tools import getinfo


@njit(float64[:](
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:]))
def line_line(pts1, pts2, vec1, vec2, dom1, dom2):
    """
    Find the intersection point of two line segments, and return None if no
    intersection exists. Based on PyEuclid module.
    """
    nan = np.array([np.nan, np.nan])
    if not test.overlaps_domain(dom1, dom2):
        return nan
    d = vec2[1] * vec1[0] - vec2[0] * vec1[1]
    if d == 0:
        return nan
    dy = pts1[0][1] - pts2[0][1]
    dx = pts1[0][0] - pts2[0][0]
    ua = (vec2[0] * dy - vec2[1] * dx) / d
    if not (0 <= ua <= 1):
        return nan
    ub = (vec1[0] * dy - vec1[1] * dx) / d
    if not (0 <= ub <= 1):
        return nan
    return pts1[0] + ua * vec1
    

@njit(float64[:,:](
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:]))
def line_pol(pts_line, pts_pol, dom_line, dom_pol, vec_line):
    """
    Find the intersection points between a line and a polygon exterior. Return 
    the (n,x,y) coordinates of intersection points as a numpy array.
    """
    
    # Cheap domain check
    if not test.overlaps_domain(dom_line, dom_pol):
        return np.array([[np.nan],[np.nan]])
    
    # Get edge node pair indices and initialize intersections
    n_edges, pts_edges, vec_edges, dom_edges = getinfo.edge_data(pts_pol)
    intersections = np.ones((n_edges, 2))
    
    # Loop over edges to find intersections with line
    for i in range(n_edges):
        inter = line_line(
            pts_line, pts_edges[i], 
            vec_line, vec_edges[i], 
            dom_line, dom_edges[i])
        intersections[i][0] = inter[0]
        intersections[i][1] = inter[1]
    return intersections


@njit(float64[:,:,:](float64[:,:], float64[:,:], float64[:], float64[:]))
def pol_pol(pts1, pts2, dom1, dom2):
    """
    TODO: docstring
    """
    # Cheap domain check
    nan = np.array([[[np.nan, np.nan]]])
    if not test.overlaps_domain(dom1, dom2):
        return nan
    
    # Get polygon edges and initialize intersections array
    n_e1, pts_e1, vec_e1, dom_e1 = getinfo.edge_data(pts1)
    n_e2, pts_e2, vec_e2, dom_e2 = getinfo.edge_data(pts2)
    intersections = np.ones((n_e1, n_e2, 2))
    
    # Loop over edges for both polygons and check edge intersections
    for i in range(n_e1):
        for j in range(n_e2):
            inter = line_line(
                pts_e1[i], pts_e2[j], 
                vec_e1[i], vec_e2[j], 
                dom_e1[i], dom_e2[j])
            intersections[i,j,0] = inter[0]
            intersections[i,j,1] = inter[1]
    return intersections

