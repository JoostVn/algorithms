import numpy as np
from math import pi
from . import contains
from . import intersect
from .tools import getinfo, test, transform
from abc import ABC, abstractmethod

"""
TODO:

    - Remove nan statements from intersection methods and move to numba
    - Optimize bool intersections
    - Implement "multiple" versions of intersection functions
"""



class GeometryBase(ABC):

    def __init__(self, points):
        """
        Base class for all geometry types.

        Points must be an array or list with shape (n,2). The last point in
        the list is linked to the first one to create a closed loop.
        """
        self.set_points(np.array(points, dtype=np.float64))

    def __repr__(self):
        shape_name = self.__class__.__name__
        str_points  = ','.join([f'({x},{y})' for (x,y) in self.points])
        return f'{shape_name} [{str_points}]'

    def set_points(self, points):
        """
        Set or adjust the geometry points. Expects a numpy array of floats.
        """
        self.points = points
        self.domain = getinfo.domain(self.points)

    def copy(self):
        """
        Return a copy of the geometry instance.
        """
        return self.__class__(self.points)

    def center(self):
        """
        Returns the average (x,y) of all polygon points.
        """
        return self.points.mean(axis=0)

    def radius(self, xy):
        """
        Returns the radius of a circle centered around a point xy that
        completely surrounds the geometry.
        """
        return np.max(np.linalg.norm(xy - self.points, axis=1))

    def translate(self, delta_xy):
        """
        Translate the polygon points in a given (x,y) direction.
        """
        self.set_points(transform.translate(self.points, *delta_xy))

    def rotate(self, delta_angle, center):
        """
        Rotate the geometry by delta_angle radians around a center point.
        """
        self.set_points(transform.rotate(self.points, delta_angle, *center))

    def overlaps_domain(self, other_geom):
        """
        Returns True if there is some overlap in self.domain and
        other_geom.domain.
        """
        return test.overlaps_domain(self.domain, other_geom.domain)



class Line(GeometryBase):

    def set_points(self, points):
        super().set_points(points)
        self.vec = self.points[1] - self.points[0]

    def intersect_line(self, other):
        """
        Find the intersection points between self and another line.
        """
        intersection = intersect.position.line_line(
            self.points, other.points, self.vec, other.vec, self.domain,
            other.domain)
        return intersection[~np.isnan(intersection).any()].flatten()

    def intersect_pol(self, polygon):
        """
        Find the intersection points between self and a polygon exterior.
        """
        intersections = intersect.position.line_pol(
            self.points, polygon.points, self.domain, polygon.domain, self.vec)
        return intersections[~np.isnan(intersections).any(axis=1)]

    def intersect_line_bool(self, other):
        """
        Find the intersection points between self and another line.
        """
        return intersect.boolean.line_line(
            self.points, other.points, self.vec, other.vec,self.domain,
            other.domain)

    def intersect_pol_bool(self, polygon):
        """
        Find the intersection points between self and a polygon exterior.
        """
        return intersect.boolean.line_pol(
            self.points, polygon.points, self.domain, polygon.domain, self.vec)



class Polygon(GeometryBase):

    def intersect_line(self, line):
        """
        Find the intersection points between polygon exterior and a line.
        """
        intersections = intersect.position.line_pol(
            line.points, self.points, line.domain, self.domain, line.vec)
        return intersections[~np.isnan(intersections).any(axis=1)]

    def intersect_pol(self, other):
        """
        Find the intersection points between self and other polygon exteriors.
        """
        intersections = intersect.position.pol_pol(
            self.points, other.points, self.domain, other.domain)
        intersections = intersections.reshape(-1,2)
        return intersections[~np.isnan(intersections).any(axis=1).flatten()]

    def intersect_line_bool(self, line):
        """
        Find the intersection points between polygon exterior and a line.
        """
        return intersect.boolean.line_pol(
            line.points, self.points, line.domain, self.domain, line.vec)

    def intersect_pol_bool(self, other):
        """
        Return True if self and other polygon exteriors intersect.
        """
        return intersect.boolean.pol_pol(
            self.points, other.points, self.domain, other.domain)

    def contains_point(self, xy):
        """
        Returns true if polygon contains the point (x,y).
        """
        return contains.point.pol(self.points, self.domain, xy)



class Circle(Polygon):

    def __init__(self, midpoint, radius, resolution):
        """
        Polygon object that is defined by a midpoint, radius and resolution.
        The resolution parameter determines the number of points that form the
        circle exterior.
        """
        assert resolution >= 3
        angles = np.linspace(0, 2*pi, resolution)
        points = midpoint + radius * np.stack(
            (np.cos(angles), np.sin(angles)), axis=1)
        super().__init__(points)
        self.midpoint = np.array(midpoint, dtpye=np.float64)

    def contains_point(self, xy):
        """
        Returns true if circle contains the point (x,y).
        """
        contains.point.circle(self.midpoint, self.radius, xy)








