"""Test suite for CollatzViz."""

import unittest
import collatz_viz.collatz_viz as target
import numpy as np
from numpy.testing import assert_almost_equal
from dataclasses import dataclass
import manim as mnm


@dataclass(order=True)
class PseudoNodeInfo(object):
    shell: int
    value: int


class TestHelpers(unittest.TestCase):

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

    def test_popall_empty(self):
        queue = target.PriorityNodeQueue()
        actual = queue.pop_all()
        self.assertEqual(actual, [])
        self.assertEqual(len(queue.queue), 0)

    def test_popall_singleton(self):
        queue = target.PriorityNodeQueue([PseudoNodeInfo(shell=4, value=17)])
        actual = queue.pop_all()
        self.assertEqual([(n.shell, n.value) for n in actual], [(4, 17)])
        self.assertEqual(len(queue.queue), 0)

    def test_popall_whole_list(self):
        queue = target.PriorityNodeQueue()
        for n in [
                PseudoNodeInfo(shell=4, value=17),
                PseudoNodeInfo(shell=4, value=3),
                PseudoNodeInfo(shell=4, value=93),
            ]:
            queue.enqueue(n)
        actual = queue.pop_all()
        self.assertEqual([(n.shell, n.value) for n in actual],
                         [(4, 3), (4, 17), (4, 93)])
        self.assertEqual(len(queue.queue), 0)

    def test_popall_partial_list(self):
        queue = target.PriorityNodeQueue()
        for n in [
                PseudoNodeInfo(shell=9, value=-22),
                PseudoNodeInfo(shell=4, value=17),
                PseudoNodeInfo(shell=5, value=3),
                PseudoNodeInfo(shell=4, value=3),
                PseudoNodeInfo(shell=62, value=17),
                PseudoNodeInfo(shell=4, value=93),
                PseudoNodeInfo(shell=5, value=5),
            ]:
            queue.enqueue(n)
        actual = queue.pop_all()
        self.assertEqual([(n.shell, n.value) for n in actual],
                         [(4, 3), (4, 17), (4, 93)])
        self.assertEqual(len(queue.queue), 4)

class TestCollatzViz(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.viz = target.CollatzViz(0)
        theta = mnm.PI / 4.0
        r = 70.0
        circle_center = target.polar_to_cartesian([theta, r, 0])
        c = self.viz.circle_factory(5)
        c.move_to(circle_center)
        self.default_node = target.NodeInfo(shell=5,
                                            value=11,
                                            display_node=c,
                                            display_text=mnm.Text('11'),
                                            polar_final_angle=theta,
                                            polar_final_radius=r,
                                            polar_segment_width=mnm.PI / 12.0)

    def test_circle_factory_decreases_circle_sizes(self):
        c1 = self.viz.circle_factory(3)
        c2 = self.viz.circle_factory(7)
        self.assertGreaterEqual(1, c1.radius)
        self.assertGreaterEqual(c1.radius, c2.radius)
        self.assertGreater(c2.radius, 0)

    def test_node_factory(self):
        n = self.viz.node_factory(parent=self.default_node, child_val=19)
        self.assertEqual(n.value, 19)
        self.assertEqual(n.polar_segment_width, mnm.PI / 12.0)
        assert_almost_equal(n.display_node.get_center(),
                            self.default_node.display_node.get_center())
        self.assertEqual(n.display_text.text, '19')

    def test_generate_doubling_node_geometry(self):
        n = self.viz.generate_doubling_node(parent=self.default_node)
        self.assertEqual(n.value, 2 * self.default_node.value)
        self.assertEqual(n.shell, self.default_node.shell + 1)
        self.assertEqual(n.polar_final_angle, self.default_node.polar_final_angle)
        self.assertGreater(n.polar_final_radius, self.default_node.polar_final_radius)
 
    def test_generate_doubling_node_segment_declared(self):
        n = self.viz.generate_doubling_node(parent=self.default_node, polar_segment_width=93.0)
        self.assertEqual(n.polar_segment_width, 93.0)

    def test_generate_doubling_node_segment_not_declared(self):
        n = self.viz.generate_doubling_node(parent=self.default_node)
        self.assertEqual(n.polar_segment_width, self.default_node.polar_segment_width)
        
    def test_generate_division_node(self):
        n = self.viz.generate_division_node(parent=self.default_node)
        self.assertEqual(n.value, int((2 * self.default_node.value - 1) / 3))
        self.assertEqual(n.shell, self.default_node.shell + 1)
        self.assertGreater(n.polar_final_radius, self.default_node.polar_final_radius)
        self.assertGreater(n.polar_final_angle, self.default_node.polar_final_angle)
        self.assertAlmostEqual(n.polar_segment_width, (n.polar_final_angle - self.default_node.polar_final_angle))
