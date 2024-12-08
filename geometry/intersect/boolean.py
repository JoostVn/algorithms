from numba import njit, float64, boolean
from ..tools import test
from ..tools import getinfo

@njit(boolean(
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:]))
def line_line(pts1, pts2, vec1, vec2, dom1, dom2):
    """
    TODO: optimize, only check if lines intersect
    """
    if not test.overlaps_domain(dom1, dom2):
        return False
    
    d = vec2[1] * vec1[0] - vec2[0] * vec1[1]
    if d == 0:
        return False
    dy = pts1[0][1] - pts2[0][1]
    dx = pts1[0][0] - pts2[0][0]
    ua = (vec2[0] * dy - vec2[1] * dx) / d
    if not (0 <= ua <= 1):
        return False
    ub = (vec1[0] * dy - vec1[1] * dx) / d
    if not (0 <= ub <= 1):
        return False
    return True
    

@njit(boolean(
    float64[:,:], float64[:,:], float64[:], float64[:], float64[:]))
def line_pol(pts_line, pts_pol, dom_line, dom_pol, vec_line):
    """
    TODO: optimize
    """
    
    # Cheap domain check
    if not test.overlaps_domain(dom_line, dom_pol):
        return False
    
    # Get edge node pair indices and initialize intersections
    n_edges, pts_edges, vec_edges, dom_edges = getinfo.edge_data(pts_pol)
    
    # Loop over edges to find intersections with line
    for i in range(n_edges):
        inter = line_line(
            pts_line, pts_edges[i], 
            vec_line, vec_edges[i], 
            dom_line, dom_edges[i])
        if inter:
            return True
    return False


@njit(boolean(float64[:,:], float64[:,:], float64[:], float64[:]))
def pol_pol(pts1, pts2, dom1, dom2):
    """
    TODO: optimize
    """
    # Cheap domain check
    if not test.overlaps_domain(dom1, dom2):
        return False
    
    # Get polygon edges and initialize intersections array
    n_e1, pts_e1, vec_e1, dom_e1 = getinfo.edge_data(pts1)
    n_e2, pts_e2, vec_e2, dom_e2 = getinfo.edge_data(pts2)
    
    # Loop over edges for both polygons and check edge intersections
    for i in range(n_e1):
        for j in range(n_e2):
            inter = line_line(
                pts_e1[i], pts_e2[j], 
                vec_e1[i], vec_e2[j], 
                dom_e1[i], dom_e2[j])
            if inter:
                return True
    return False