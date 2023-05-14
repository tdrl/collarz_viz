# An attempt to visualize the Collatz graph.

import manim as mnm  # ignore: reportMissingStubTypes
from dataclasses import dataclass, field
from typing import Optional, List, Iterable
import numpy as np
from collections import deque
from numpy.linalg import norm
import heapq


@dataclass(order=True)
class NodeInfo:
    shell: int = field(compare=True)
    value: int = field(compare=True)
    color: mnm.color.Color = field(compare=False)
    display_node: mnm.Circle = field(compare=False)
    display_text: mnm.Text = field(compare=False)
    display_trace: Optional[mnm.TracedPath] = field(default=None, compare=False)
    polar_final_angle: float = field(default=0.0, compare=False)
    polar_final_radius: float = field(default=0.0, compare=False)
    animation_list: List[mnm.Animation] = field(default_factory=list, compare=False)


class PriorityNodeQueue(object):
    def __init__(self, first_elements: Iterable[NodeInfo] = []) -> None:
        self.queue: List[NodeInfo] = list()
        for e in first_elements:
            self.enqueue(e)

    def enqueue(self, element: NodeInfo):
        heapq.heappush(self.queue, element)

    def pop(self) -> NodeInfo:
        return heapq.heappop(self.queue)


class CollatzViz(mnm.MovingCameraScene):

    def __init__(self, nodes_to_generate: int = 20):
        super().__init__()
        self.nodes_to_generate = nodes_to_generate
        self.origin = mnm.ORIGIN
        self.circle_radius = 1.0
        self.circle_fill_color_start = mnm.PURPLE
        self.circle_fill_color_end = mnm.RED
        self.circle_fill_opacity = 1.0
        self.circle_stroke_color = mnm.DARK_BLUE
        self.circle_stroke_width = 8
        self.trace_stroke_color = mnm.BLUE_A
        self.trace_stroke_width = 8
        self.step_size = 4.0
        self.arc_angle = 135 * mnm.DEGREES
        self.circle_radius_decay = 0.95
        self.angular_decay = 0.75
        self.distance_decay = 0.93

    def circle_factory(self, shell: int) -> mnm.Circle():
        decay = np.power(self.circle_radius_decay, shell)
        c = mnm.Circle(radius=self.circle_radius * decay)
        c.set_stroke(color=self.circle_fill_color_start, width=self.circle_stroke_width * decay)
        fill_color = mnm.utils.color.interpolate_color(self.circle_fill_color_start,
                                                       self.circle_fill_color_end,
                                                       1.0 - decay)
        c.set_fill(color=fill_color, opacity=self.circle_fill_opacity)
        return c

    def node_factory(self, parent: NodeInfo, child_val: int) -> NodeInfo:
        child_shell = parent.shell + 1
        child_node = NodeInfo(
            value=child_val,
            shell=child_shell,
            color=parent.color,
            display_node=self.circle_factory(child_shell),
            display_text=mnm.Text(str(child_val))
        )
        child_node.display_node.move_to(parent.display_node)
        child_node.display_text.match_style(parent.display_text)
        child_node.display_text.clear_updaters()
        child_node.display_text.add_updater(lambda x: x.next_to(child_node.display_node, mnm.DOWN))
        child_node.display_trace = mnm.TracedPath(child_node.display_node.get_center,
                                                  stroke_width=self.trace_stroke_width,
                                                  stroke_color=self.trace_stroke_color,
                                                  z_index=-2)
        child_node.animation_list.append(mnm.FadeIn(child_node.display_node,
                                                    child_node.display_text,
                                                    child_node.display_trace))
        return child_node

    def generate_doubling_node(self, parent: NodeInfo) -> NodeInfo:
        child_val = 2 * parent.value
        child_node = self.node_factory(parent=parent, child_val=child_val)
        move_dist = np.power(self.distance_decay, child_node.shell) * self.step_size
        child_node.polar_final_angle = parent.polar_final_angle
        child_node.polar_final_radius = parent.polar_final_radius + move_dist
        here = child_node.display_node.get_center()
        move_path = mnm.Line(start=here,
                             end=[here[0] + move_dist * np.cos(child_node.polar_final_angle),
                                  here[1] + move_dist * np.sin(child_node.polar_final_angle),
                                  0.0])
        child_node.animation_list.append(mnm.MoveAlongPath(child_node.display_node, move_path))
        return child_node

    def generate_division_node(self, parent: NodeInfo) -> NodeInfo:
        child_val = int((2 * parent.value - 1) / 3)
        child_node = self.node_factory(parent=parent, child_val=child_val)
        here = child_node.display_node.get_center()
        move_dist = np.power(self.distance_decay, child_node.shell) * self.step_size
        there = here + np.array([move_dist / 2 * np.cos(parent.polar_final_angle),
                                 move_dist / 2 * np.sin(parent.polar_final_angle),
                                 0.0])
        first_segment = mnm.Line(start=here, end=there)
        arc_init_radius = norm(there - self.origin)
        arc_init_angle = np.arctan2(here[1], here[0])
        turn_angle = np.power(self.angular_decay, child_node.shell) * self.arc_angle
        arc_final_angle = arc_init_angle + turn_angle
        arc_segment = mnm.Arc(radius=arc_init_radius, arc_center=self.origin, start_angle=arc_init_angle, angle=turn_angle)
        here = arc_segment.get_end()
        there = arc_segment.get_end() + np.array([move_dist / 2 * np.cos(arc_final_angle),
                                                  move_dist / 2 * np.sin(arc_final_angle),
                                                  0.0])
        second_segment = mnm.Line(start=here, end=there)
        child_node.animation_list.append(mnm.MoveAlongPath(child_node.display_node, first_segment, run_time=1./3., rate_func=mnm.rate_functions.ease_in_sine))
        child_node.animation_list.append(mnm.MoveAlongPath(child_node.display_node, arc_segment, run_time=1./3., rate_func=mnm.rate_functions.linear))
        child_node.animation_list.append(mnm.MoveAlongPath(child_node.display_node, second_segment, run_time=1./3., rate_func=mnm.rate_functions.ease_out_sine))
        child_node.polar_final_angle = arc_final_angle
        child_node.polar_final_radius = norm(second_segment.get_end())
        return child_node

    def play_all(self, animations: List[mnm.Animation]):
        for a in animations:
            self.play(a)

    def construct(self):
        self.camera.frame.save_state()
        origin_node = self.circle_factory(shell=0)
        origin_text = mnm.Text("1").next_to(origin_node, mnm.DOWN)
        primal_ancestor = NodeInfo(
            value=1,
            shell=0,
            color=mnm.PURPLE,
            display_node=origin_node,
            display_text=origin_text,
            display_trace=None,
            polar_final_angle=0.0,
            polar_final_radius=0.0,
            animation_list=[mnm.FadeIn(origin_node, origin_text)]
        )
        open = PriorityNodeQueue([primal_ancestor])
        closed: List[NodeInfo] = []
        self.camera.frame.set(height=50)
        for _ in range(self.nodes_to_generate):
            curr_node = open.pop()
            # self.play(self.camera.frame.animate.set(width=self.step_size * next_node.value + 3))
            self.play_all(curr_node.animation_list)
            open.enqueue(self.generate_doubling_node(curr_node))
            if (curr_node.value == 2):
                self.play(mnm.Create(mnm.ArcBetweenPoints(curr_node.display_node.get_center(),
                                                          primal_ancestor.display_node.get_center(),
                                                          angle=mnm.PI,
                                                          z_index=-2,
                                                          stroke_width=self.trace_stroke_width,
                                                          stroke_color=self.trace_stroke_color)))
            if (curr_node.value > 2) and (curr_node.value % 3 == 2):
                open.enqueue(self.generate_division_node(curr_node))
            closed.append(curr_node)
