"""Test suite for CollatzViz."""

import unittest
import collatz_viz.collatz_viz as target
import numpy as np
from numpy.testing import assert_almost_equal
import manim as mnm


class TestCollatzViz(unittest.TestCase):

    def test_monotonic_bbox_moving_camera_update_interior_bbox_single(self):
        # Default extremal bounds.
        camera = target.MonotonicBBoxMovingCamera()
        camera.add_to_bbox([-1, -2, 3, 4])
        assert_almost_equal(camera.get_content_bbox(), [-1, -2, 3, 4])

    def test_monotonic_bbox_moving_camera_update_interior_bbox_multi(self):
        # Default extremal bounds.
        camera = target.MonotonicBBoxMovingCamera()
        camera.add_to_bbox([-1, -2, 3, 4])
        camera.add_to_bbox([-1, -1, 1, 1])  # BBox unchanged
        camera.add_to_bbox([2, 2, 3, 5])
        camera.add_to_bbox([-4, -4, -2, -2])
        assert_almost_equal(camera.get_content_bbox(), [-4, -4, 3, 5])
        
    def test_monotonic_bbox_moving_camera_update_interior_mobject_single(self):
        # Default extremal bounds.
        camera = target.MonotonicBBoxMovingCamera()
        camera.add_to_bbox(mnm.Circle(radius=2).move_to([-1, -2, 0]))
        assert_almost_equal(camera.get_content_bbox(), [-3, -4, 1, 0])

    def test_monotonic_bbox_moving_camera_update_interior_mobject_multi(self):
        # Default extramal bounds.
        camera = target.MonotonicBBoxMovingCamera()
        camera.add_to_bbox(mnm.Circle(radius=2).move_to([-1, -2, 0]))
        camera.add_to_bbox(mnm.Square(side_length=4).move_to([5, 8, 0]))
        camera.add_to_bbox(mnm.Square(side_length=0.03))  # BBox unchanged
        assert_almost_equal(camera.get_content_bbox(), [-3, -4, 7, 10])

    def test_monotonic_bbox_moving_camera_update_interior_maintains_bounds(self):
        # Explicit extremal bounds.
        camera = target.MonotonicBBoxMovingCamera(content_bbox_bounds=[-3, -1, 5, 8])
        camera.add_to_bbox(mnm.Circle(radius=0.001).move_to([1000, 1000, 0]))
        camera.add_to_bbox(mnm.Circle(radius=1).move_to([-1000, -1000, 0]))
        assert_almost_equal(camera.get_content_bbox(), [-3, -1, 5, 8])

    def test_polar_transforms_default_origin(self):
        for point in [[3, 4, 0],
                      [1, 1, 1],
                      [-7, -4, -2],
                      [-8, 2, 0],
                      [17, -11, 2]]:
            point = np.asarray(point)
            assert_almost_equal(target.polar_to_cartesian(target.cartesian_to_polar(point)), point,
                                err_msg=f'Identity mapping failed for point={point}')

    def test_polar_transforms_specified_origin(self):
        origin = np.array([142, -3871, 12])
        for point in [[3, 4, 0],
                      [1, 1, 1],
                      [-7, -4, -2],
                      [-8, 2, 0],
                      [17, -11, 2]]:
            point = np.asarray(point)
            assert_almost_equal(target.polar_to_cartesian(target.cartesian_to_polar(point, origin=origin), origin=origin), point,
                                err_msg=f'Identity mapping failed for point={point}')

