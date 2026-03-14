"""
Legacy. Might delete later.
"""

import time
import shapely.geometry
import pyshape.geometry
import numpy as np
from math import pi, sin, cos



######################## TIMER FUNCTIONS

def time_func(func, iterable_args):
    
    # Set args to list if only single argument is given
    if not type(iterable_args[0]) is list:
        iterable_args = [[args] for args in iterable_args]
    
    # Run function for multiple iterations and record time
    t_start = time.perf_counter()
    for args in iterable_args:
        func(*args)
    t_end = time.perf_counter()
   
    # Print results
    module = func.__module__.split('.')[0]
    funcname = func.__name__
    name = f'{module}.{funcname}'
    t_it = round(1000 * (t_end - t_start),2)
    print(name.ljust(29), str(t_it).rjust(6), ' ms') 


def time_method(iterable_method, iterable_args):
        
    # Set args to list if only single argument is given
    if not type(iterable_args[0]) is list:
        iterable_args = [[args] for args in iterable_args]
    
    # Run function for multiple iterations and record time
    t_start = time.perf_counter()
    for method, args in zip(iterable_method, iterable_args):
        method(*args)
    t_end = time.perf_counter()
   
    # Print results
    func = iterable_method[0]
    module = func.__module__.split('.')[0]
    funcname = func.__name__
    name = f'{module}.{funcname}'
    t_it = round(1000 * (t_end - t_start),2)
    print(name.ljust(29), str(t_it).rjust(6), ' ms') 



######################## SHAPES AND POINTS

it = 1000

# Lines
line_points = np.random.uniform(0,10,(it,2,2))
shap_lines = [shapely.geometry.LineString(points) for points in line_points]
geom_lines = [pyshape.geometry.Line(points) for points in line_points]

# Polygons
pol_points = []
for i in range(it):
    num = np.random.randint(3, 20)
    size = np.random.randint(1, 100)
    angles = np.linspace(0, 2*pi, num, endpoint=False)
    points = tuple([(cos(a)* size, sin(a) * size) for a in angles])
    pol_points.append(points)
geom_pols = [pyshape.geometry.Polygon(points) for points in pol_points]
shap_pols = [shapely.geometry.Polygon(points) for points in pol_points]

# Translation and rotation
delta_xy = np.random.uniform(-10, 10, (it, 2))
delta_angle = np.random.uniform(-pi, pi, it)



######################## TESTING

print('\nRESULTS FOR 1000 ITERATIONS AND RANDOM SHAPES')

print('\nSHAPE CREATION')

# Line creation
print()
time_func(shapely.geometry.LineString, line_points)
time_func(pyshape.geometry.Line, line_points)

# Polygon creation
print()
time_func(shapely.geometry.Polygon, pol_points)
time_func(pyshape.geometry.Polygon, pol_points)


print('\nPOLYGON TRANSLATION AND ROTATION')

# Polygon translation
print()
shap_args = [[pol, xy[0], xy[1]] for pol, xy in zip(shap_pols, delta_xy)]
time_func(shapely.affinity.translate, shap_args)
time_method([pol.translate for pol in geom_pols], delta_xy)

# Polygon rotation
print()
shap_args = [[pol, a] for pol, a in zip(shap_pols, delta_angle)]
time_func(shapely.affinity.rotate, shap_args)
geom_args = [[a, (0,0)] for a in delta_angle]
time_method([pol.rotate for pol in geom_pols], geom_args)


print('\nINTERSECTION POINTS')

# Line line
print()
time_method([l.intersection for l in shap_lines], list(reversed(shap_lines)))
time_method([l.intersect_line for l in geom_lines], list(reversed(geom_lines)))

# line  pol
print()
time_method([l.intersection for l in shap_lines], shap_pols)
time_method([l.intersect_pol for l in geom_lines], geom_pols)

# pol pol
print()
time_method([p.intersection for p in shap_pols], list(reversed(shap_pols)))
time_method([p.intersect_pol for p in geom_pols], list(reversed(geom_pols)))




print('\nINTERSECTION BOOL')

# Line line
print()

# line  pol
print()

# pol pol
print()
time_method([p.intersection for p in shap_pols], list(reversed(shap_pols)))
time_method([p.intersect_pol_bool for p in geom_pols], list(reversed(geom_pols)))

