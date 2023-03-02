import numpy as np
import scipy.spatial  # type: ignore
import open3d as o3d
import math


# unit_vec_1 is setted as [0,0,1]
# vec_2 may need to be normalized

def vec2vec_rotation(unit_vec_1, vec_2):
    norm_vec_2 = np.linalg.norm(vec_2)
    if np.abs(norm_vec_2 - 1) > 1e-3:
        unit_vec_2 = vec_2 / norm_vec_2
    else:
        unit_vec_2 = vec_2

    if np.abs(np.linalg.norm(unit_vec_1) - 1) > 1e-8:
        unit_vec_1 /= np.linalg.norm(unit_vec_1)

    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))
    if angle < 1e-8:
        return np.identity(3, dtype=np.float64)

    if angle > (np.pi - 1e-8):
        # WARNING this only works because all geometries are rotationaly invariant
        # minus identity is not a proper rotation matrix
        return -np.identity(3, dtype=np.float64)

    rot_vec = np.cross(unit_vec_1, unit_vec_2)
    rot_vec = rot_vec.astype(float)
    rot_vec /= np.linalg.norm(rot_vec)

    return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)


def vector_equal(v1, v2):
    return v1.shape == v2.shape and np.allclose(
        v1, v2, rtol=1e-12, atol=1e-12, equal_nan=False
    )


def distance_point_point(p1, p2):
    """Calculates the euclidian distance between two points or sets of points
    >>> distance_point_point(np.array([1, 0]), np.array([0, 1]))
    1.4142135623730951
    >>> distance_point_point(np.array([[1, 1], [0, 0]]), np.array([0, 1]))
    array([1., 1.])
    >>> distance_point_point(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, -3]]))
    array([1., 3.])
    """
    return scipy.spatial.minkowski_distance(p1, p2)


def distance_plane_point(plane_point, plane_normal, point):
    """Calculates the signed distance from a plane to one or more points
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([2, 2, 2]))
    1
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[2, 2, 2], [2, 2, 3]]))
    array([1, 2])
    """
    assert np.allclose(
        np.linalg.norm(plane_normal), 1.0, rtol=1e-12, atol=1e-12, equal_nan=False
    )
    return np.dot(point - plane_point, plane_normal)


def distance_line_point(line_point, line_direction, point):
    """Calculates the distance from a line to a point
    >>> distance_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([1, 1, 2]))
    1.4142135623730951
    >>> distance_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[1, 0, 1], [0, 2, 3]]))
    array([1., 2.])
    """
    assert np.allclose(
        np.linalg.norm(line_direction), 1.0, rtol=1e-12, atol=1e-12, equal_nan=False
    )
    delta_p = point - line_point
    return distance_point_point(
        delta_p,
        np.matmul(
            np.expand_dims(np.dot(delta_p, line_direction), axis=-1),
            np.atleast_2d(line_direction),
        ),
    )


def two_lines_parallel(d1, d2):
    if np.allclose(d1, d2):
        return True
    return False


def closest_points_on_two_skew_lines(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if two_lines_parallel(d1, d2):
        return None, None, np.linalg.norm(np.cross(p2 - p1, d1)) / np.linalg.norm(d2)
    v3 = np.cross(d1, d2)
    v3 = v3 / np.linalg.norm(v3)

    mat = np.column_stack((d1, -d2, v3))
    t1, t2, t3 = np.linalg.solve(mat, p2 - p1)
    q1 = p1 + t1 * d1
    q2 = p2 + t2 * d2
    return q1, q2, np.abs(t3)


def onoff_ramp_points(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if two_lines_parallel(d1, d2):
        return None, None
    q1, q2, t3 = closest_points_on_two_skew_lines(p1, d1, p2, d2)
    middle = (q1 + q2) / 2

    r = t3 / 2
    # centric, begin, end
    c1 = middle - d1 * r
    b1 = q1 - d1 * t3 / 2
    e1 = middle

    c2 = middle + d2 * r
    b2 = middle
    e2 = q2 + d2 * r

    return [c1, b1, e1], [c2, b2, e2], r


def rotate_to_xy_plane(torus_points):
    # c: centric, a: begin point, b: end point
    c, b, e = torus_points

    # Find the normal vector of the plane
    normal_vec = np.cross(b - c, e - c)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    # Find the rotation matrix
    rotation_matrix = vec2vec_rotation(normal_vec, [0, 0, 1])

    # Rotate b and c to the x-y plane
    b_new = np.matmul(rotation_matrix, b - c)
    e_new = np.matmul(rotation_matrix, e - c)

    return b_new, e_new


def ratio_to_degree(x):
    return math.degrees(math.atan2(x[1], x[0]))


def distance_line_line(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    if two_lines_parallel(d1, d2):
        return np.linalg.norm(np.cross(p2 - p1, d1)) / 1  # np.linalg.norm(d1)
    else:
        return distance_two_skew_lines(p1, d1, p2, d2)


def distance_two_skew_lines(p1, d1, p2, d2):
    v = np.cross(d1, d2)

    # Calculate the denominator of the line parameter equations
    denom = np.linalg.norm(v)

    w = p1 - p2

    # Calculate the line parameters
    a = np.dot(v, w)
    distance = np.linalg.norm(a) / denom

    return distance
