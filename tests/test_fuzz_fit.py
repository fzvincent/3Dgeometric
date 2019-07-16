from geomfitty import fit3d, geom3d
from .test_util import (
    assert_float_equal,
    assert_vector_equal,
    assert_direction_equivalent,
)

import numpy as np
import numpy.random
from scipy import optimize
import pytest


def brute_force_fit(Shape, args, points, weights=None, x0=None):
    csum = np.cumsum((0,) + args)

    def error(x, points, weights):
        geom = Shape(*(x[i:j] for i, j in zip(csum[:-1], csum[1:])))

        weights = 1.0 if weights is None else weights
        distances = geom.distance_to_point(points)
        return np.sum(weights * distances ** 2)

    x0 = np.random.uniform(low=-1, high=1, size=sum(args)) if x0 is None else x0
    results = optimize.minimize(error, x0=x0, args=(points, weights))
    return Shape(*(results.x[i:j] for i, j in zip(csum[:-1], csum[1:]))), results


def test_fuzz_line():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, 2] *= 10

    line1 = fit3d.line_fit(points)
    line2, results = brute_force_fit(
        geom3d.Line, (3, 3), points, x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    )

    assert_direction_equivalent(line1.direction, line2.direction)
    assert_float_equal(
        np.sum(line1.distance_to_point(points) ** 2),
        np.sum(line2.distance_to_point(points) ** 2),
    )


def test_fuzz_line_with_weights():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, 2] *= 10

    weights = np.random.uniform(size=(100,))

    line1 = fit3d.line_fit(points, weights=weights)
    line2, results = brute_force_fit(
        geom3d.Line,
        (3, 3),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
    )

    assert_direction_equivalent(line1.direction, line2.direction)
    assert_float_equal(
        np.sum(line1.distance_to_point(points) ** 2),
        np.sum(line2.distance_to_point(points) ** 2),
    )


def test_fuzz_plane():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, :2] *= 10

    plane1 = fit3d.plane_fit(points)
    plane2, results = brute_force_fit(
        geom3d.Plane, (3, 3), points, x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    )

    assert_direction_equivalent(plane1.normal, plane2.normal)
    assert_float_equal(
        np.sum(plane1.distance_to_point(points) ** 2),
        np.sum(plane2.distance_to_point(points) ** 2),
    )


def test_fuzz_plane_with_weights():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, :2] *= 10

    weights = np.random.uniform(size=(100,))

    plane1 = fit3d.plane_fit(points, weights=weights)
    plane2, results = brute_force_fit(
        geom3d.Plane,
        (3, 3),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
    )

    assert_direction_equivalent(plane1.normal, plane2.normal)
    assert_float_equal(
        np.sum(plane1.distance_to_point(points) ** 2),
        np.sum(plane2.distance_to_point(points) ** 2),
    )


@pytest.mark.parametrize("initial_guess", [geom3d.Sphere([0, 0, 0], 1), None])
def test_fuzz_sphere(initial_guess):
    points = np.random.uniform(low=-1, high=1, size=(3, 100))
    points /= np.linalg.norm(points, axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    sphere1 = fit3d.sphere_fit(points, initial_guess=initial_guess)
    sphere2, results = brute_force_fit(
        geom3d.Sphere, (3, 1), points, x0=np.array([0, 0, 0, 1], dtype=np.float64)
    )

    assert_vector_equal(sphere1.center, sphere2.center)
    assert_float_equal(sphere1.radius, sphere2.radius)


@pytest.mark.parametrize("initial_guess", [geom3d.Sphere([0, 0, 0], 1), None])
def test_fuzz_sphere_with_weights(initial_guess):
    points = np.random.uniform(low=-1, high=1, size=(3, 100))
    points /= np.linalg.norm(points, axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    weights = np.random.uniform(size=(100,))

    sphere1 = fit3d.sphere_fit(points, weights=weights, initial_guess=initial_guess)
    sphere2, results = brute_force_fit(
        geom3d.Sphere,
        (3, 1),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 1], dtype=np.float64),
    )

    assert_vector_equal(sphere1.center, sphere2.center)
    assert_float_equal(sphere1.radius, sphere2.radius)
