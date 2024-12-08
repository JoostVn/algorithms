import numpy as np
from numba import njit, float64



@njit(float64[:,:](float64[:,:], float64, float64))
def translate(points, x, y):
    """
    Translate the geometry points in a given (x,y) direction.
    """
    for i in range(len(points)):
        points[i][0] = points[i][0] + x
        points[i][1] = points[i][1] + y
    return points


@njit(float64[:,:](float64[:,:], float64, float64, float64))
def rotate(points, delta_angle, center_x, center_y):
    """ 
    Rotate the geometry by delta_angle radians around a center point.
    """
    cosang, sinang = np.cos(-delta_angle), np.sin(-delta_angle)
    for i in range(len(points)):
        tx = points[i][0] - center_x
        ty = points[i][1] - center_y
        points[i][0] = center_x + (tx * cosang + ty * sinang)
        points[i][1] = center_y + (-tx * sinang + ty * cosang)
    return points
