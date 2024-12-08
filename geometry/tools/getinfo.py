import numpy as np
import numba
from numba import njit, float64, int64
        

@njit(float64[:](float64[:,:]))
def domain(points):
    """
    returns the domain of an (n, 2) list of points as 
    (min(x), max(x), min(y), max(y)). 
    """
    x, y = points.T
    domain = np.array((min(x), max(x), min(y), max(y)), dtype=float64)
    return domain



@njit(numba.types.Tuple((int64, float64[:,:,:], float64[:,:], float64[:,:]))(
        float64[:,:]))
def edge_data(pts):
    """
    Get the number of polygon edges, and for each edge an array of point pairs, 
    vectors and domains. 
    """
    n = len(pts)
    idx = list(range(n))
    pairs_idx = zip(idx, idx[1:] + [idx[0]])
    points = np.ones((n,2,2), dtype=float64)
    vectors = np.ones((n,2), dtype=float64)
    domains = np.ones((n,4), dtype=float64)
    for i, j in pairs_idx:
        points[i][0][0] = pts[i][0]
        points[i][0][1] = pts[i][1]
        points[i][1][0] = pts[j][0]
        points[i][1][1] = pts[j][1]
        vectors[i][0] = pts[j][0] - pts[i][0]
        vectors[i][1] = pts[j][1] - pts[i][1]
        domains[i][0] = min(pts[i][0], pts[j][0])
        domains[i][1] = max(pts[i][0], pts[j][0])
        domains[i][2] = min(pts[i][1], pts[j][1])
        domains[i][3] = max(pts[i][1], pts[j][1])
    return n, points, vectors, domains


