from numba import njit, float64, boolean



@njit(boolean(float64[:], float64[:]))
def overlaps_domain(d1, d2):
    """
    Cheap function that returns True if there is some overlap in
    domain d1 and domain d2. The domains are given as (xmin, xmax, ymin, ymax).
    """
    return not (
        d1[1] < d2[0] or d2[1] < d1[0] or d1[3] < d2[2] or d2[3] < d1[2])

