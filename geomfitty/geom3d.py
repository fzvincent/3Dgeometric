from abc import ABC, abstractmethod

import numpy as np
import math

from ._descriptor import Direction, Position, PositiveNumber
from ._util import distance_line_point, distance_plane_point, distance_point_point, vec2vec_rotation


class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        """Calculates the smallest distance from a point to the shape"""

    # @abstractmethod
    # def project_point(self, point):
    # pass


class Line(GeometricShape):
    anchor_point = Position(3)
    direction = Direction(3)

    def __init__(self, anchor_point, direction):
        self.anchor_point = anchor_point
        self.direction = direction

    def __repr__(self):
        return f"Line(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()})"

    def distance_to_point(self, point):
        return distance_line_point(self.anchor_point, self.direction, point)


# class myLine(Line):
#     length = PositiveNumber()
#
#     def __init__(self, anchor_point, direction, length):
#         super().__init__(anchor_point, direction)
#         self.length = length
#
#     def __repr__(self):
#         return f"Cylinder(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()}, length={self.length})"
#
#     def distance_to_point(self, point):
#         # return np.abs(super().distance_to_point(point) - self.radius)
#         return np.abs(super().distance_to_point(point))


class Plane(GeometricShape):
    anchor_point = Position(3)
    normal = Direction(3)

    def __init__(self, anchor_point, normal):
        self.anchor_point = anchor_point
        self.normal = normal

    def __repr__(self):
        return f"Plane(anchor_point={self.anchor_point.tolist()}, normal={self.normal.tolist()})"

    def distance_to_point(self, point):
        return distance_plane_point(self.anchor_point, self.normal, point)

    def abs_distance_to_point(self, point):
        return np.abs(self.distance_to_point(point))

    def rotation(self, direction):
        rotation = vec2vec_rotation([0, 0, 1], direction)
        self.anchor_point = np.matmul(rotation, self.anchor_point)
        self.normal = np.matmul(rotation, self.normal)
    # def abs_distance_to_point(self, point):
    #     return np.abs(distance_plane_point(self.anchor_point, self.normal, point))


class Sphere(GeometricShape):
    center = Position(3)
    radius = PositiveNumber()

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.center.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(distance_point_point(point, self.center) - self.radius)


class Cylinder(Line):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius):
        super().__init__(anchor_point, direction)
        self.radius = radius

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.radius)


class Cylinder(Line):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius, length=1):
        super().__init__(anchor_point, direction)
        self.radius = radius
        self.length = length

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, " \
               f"direction={self.direction.tolist()}, " \
               f"radius={self.radius},"\
               f"length={self.length})"

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.radius)


class Circle3D(GeometricShape):
    center = Position(3)
    direction = Direction(3)
    radius = PositiveNumber()

    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction
        self.radius = radius

    def __repr__(self):
        return f"Circle3D(center={self.center.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        delta_p = point - self.center
        x1 = np.matmul(
            np.expand_dims(np.dot(delta_p, self.direction), axis=-1),
            np.atleast_2d(self.direction),
        )
        x2 = delta_p - x1
        return np.sqrt(
            np.linalg.norm(x1, axis=-1) ** 2
            + (np.linalg.norm(x2, axis=-1) - self.radius) ** 2
        )

class partialCircle3D(Circle3D):
    # begin_degree = PositiveNumber()
    # end_degree = PositiveNumber()

    def __init__(self, center, direction, radius, begin_degree, end_degree):
        super().__init__(center, direction, radius)
        self.begin_degree = begin_degree
        self.end_degree = end_degree

    def __repr__(self):
        return f"Circle3D(center={self.center.tolist()}, " \
               f"direction={self.direction.tolist()}, " \
               f"radius={self.radius},"\
               f"begin_degree={self.begin_degree}, " \
               f"end_degree={self.end_degree})"

class Torus(Circle3D):
    minor_radius = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius):
        super().__init__(center, direction, major_radius)
        self.minor_radius = minor_radius

    def __repr__(self):
        return f"Torus(center={self.center.tolist()}, direction={self.direction.tolist()}, major_radius={self.major_radius}, minor_radius={self.minor_radius})"

    @property
    def major_radius(self):
        return self.radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.minor_radius)




class partialTorus(Torus):
    # begin_degree = PositiveNumber()
    # end_degree = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius, begin_degree, end_degree):
        super().__init__(center, direction, major_radius, minor_radius)
        self.begin_degree = begin_degree
        self.end_degree = end_degree

        # defined with direction of [0,0,1], needs to be rotated.
        # direction pointing to the outside of the 3D structure
        self.begin_plane = Plane([major_radius * math.cos(begin_degree),
                                  major_radius * math.sin(begin_degree),
                                  0],
                                 [math.sin(begin_degree),
                                  -math.cos(begin_degree),
                                  0])
        self.end_plane = Plane([major_radius * math.cos(end_degree),
                                major_radius * math.sin(end_degree),
                                0],
                               [-math.sin(end_degree),
                                math.cos(end_degree),
                                0])
        # rotation based on direction
        self.begin_plane.rotation(direction)
        self.end_plane.rotation(direction)

    def __repr__(self):
        return f"Torus(center={self.center.tolist()}, " \
               f"direction={self.direction.tolist()}, " \
               f"major_radius={self.major_radius}, " \
               f"minor_radius={self.minor_radius}, " \
               f"begin_degree={self.begin_degree}, " \
               f"end_degree={self.end_degree})"

    @property
    def major_radius(self):
        return self.radius

    # def distance_to_point(self, point):
    #     return np.abs(super().distance_to_point(point) - self.minor_radius)
    def distance_to_point(self, point):
        # if the point is inside, value shall be negative
        dis2begin_plane = self.begin_plane.distance_to_point(self, point)
        dis2end_plane = self.end_plane.distance_to_point(self, point)
        if dis2begin_plane < 0 and dis2end_plane < 0:
            return np.abs(super().distance_to_point(point) - self.minor_radius)
        else:
            return "outside"


class myCylinder(Cylinder):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius, length):
        super().__init__(anchor_point, direction, radius)
        self.length = length

        # defined with direction of [0,0,1], needs to be rotated.
        # direction pointing to the outside of the 3D structure
        self.begin_plane = Plane(anchor_point, -direction)
        self.end_plane = Plane(anchor_point+ length, -direction)

        # rotation based on direction
        self.begin_degree_rotated = self.begin_plane.rotation(direction)
        self.end_plane_rotated = self.end_plane.rotation(direction)


    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        distance2begin_plane = self.begin_plane(point)
        distance2end_plane = self.begin_end(point)
        return np.abs(super().distance_to_point(point) - self.radius)
