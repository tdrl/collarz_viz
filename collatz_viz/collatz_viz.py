"""An attempt to visualize the Collatz graph.

The Collatz conjecture is a significant outstanding mathematical question that,
quite unusually, is both incredibly simple to explain and yet incredibly difficult
to prove. Mathematicians have been struggling with it for over a hundred years,
yet it remains open. For detail on the full conjecture, see the Wikipedia article:
    https://en.wikipedia.org/wiki/Collatz_conjecture

This script produces a visualization of the conjecture, as a tree growth process.
It uses the "reverse" form of the conjecture, which generates values starting from
1, rather than working from a chosen number down to 1.

To compile the visualization, use:

    manim -p -qm collatz_viz.py CollatzViz
"""

import manim as mnm  # ignore: reportMissingStubTypes
from dataclasses import dataclass, field
from typing import Optional, List, Iterable
import numpy as np
from numpy.linalg import norm
import heapq


@dataclass(order=True)
class NodeInfo:
    """A container for all of the information we need to track about a single visual node.
    
    This contains the "value" of the node (i.e., the int represented in the Collatz process)
    as well as a number of visual elements for display and metadata used for layout.

    Note: We require these nodes to be sortable so that we can order them with a
    priority queue (see PriorityNodeQueue). However, we only need to order on the
    "shell" and the "value", so we turn off comparison for all other fields.
    """
    # The "shell" of the circular visualization in which this node is placed.
    shell: int = field(compare=True)
    # The Collatz value of this node.
    value: int = field(compare=True)
    # Color of the node.
    color: mnm.color.Color = field(compare=False)
    # The displayable circle mobject representing this value.
    display_node: mnm.Circle = field(compare=False)
    # The displayable text attached to this node.
    display_text: mnm.Text = field(compare=False)
    # A visual element that traces the path the node takes as we animate it.
    display_trace: Optional[mnm.TracedPath] = field(default=None, compare=False)
    # The final placed location of the node, as measured in polar coordinates.
    polar_final_angle: float = field(default=0.0, compare=False)
    polar_final_radius: float = field(default=0.0, compare=False)
    # The set of animations required to draw this node and its ancillary
    # visualizations (text, trace, etc.) To render a node appearing and then moving
    # into its final position, call CollatzViz.play_all(animation_list)
    animation_list: List[mnm.Animation] = field(default_factory=list, compare=False)


class PriorityNodeQueue(object):
    """A bog standard priority queue, specialized for NodeInfos."""
    def __init__(self, first_elements: Iterable[NodeInfo] = []) -> None:
        self.queue: List[NodeInfo] = list()
        for e in first_elements:
            self.enqueue(e)

    def enqueue(self, element: NodeInfo):
        heapq.heappush(self.queue, element)

    def pop(self) -> NodeInfo:
        return heapq.heappop(self.queue)


class CollatzViz(mnm.MovingCameraScene):
    """The main visualization.
    
    This animates the growth of the reverse Collatz process as a tree, laid out with
    a circular layout style and exponentially decaying node sizes and arc lengths, so
    that we can fit it all in a bounded space.
    """

    def __init__(self, nodes_to_generate: int = 100):
        """Configure Collatz process viz.

        Args:
            nodes_to_generate (int, optional): Number of nodes to generate/animate.
                Larger numbers makes bigger, more elaborate visualizations, but takes longer
                to render. Defaults to 100.
        """
        super().__init__()
        self.nodes_to_generate = nodes_to_generate
        # Starting point of the visualization.
        self.origin = mnm.ORIGIN
        # Initial circle radius for root node.
        self.circle_radius = 1.0
        # We use a color gradient of nodes. These are the start and end of the color
        # gradient.
        self.circle_fill_color_start = mnm.PURPLE
        self.circle_fill_color_end = mnm.RED
        # How opaque to color each node circle.
        self.circle_fill_opacity = 1.0
        # Color of node edge.
        self.circle_stroke_color = mnm.DARK_BLUE
        # Thickness of node edge.
        self.circle_stroke_width = 8
        # Color of trace that we draw as we move nodes.
        self.trace_stroke_color = mnm.BLUE_A
        # Thickness of trace.
        self.trace_stroke_width = 8
        # Fundamental "step size": distance between final placement of nodes.
        self.step_size = 4.0
        # Fundamental rotational angle for each arc in the layout.
        self.arc_angle = 135 * mnm.DEGREES
        # Exponential decay factors: How quickly we decay circle sizes, polar angles
        # of rotation, and step sizes as we move out in "shells" from the origin.
        self.circle_radius_decay = 0.95
        self.angular_decay = 0.75
        self.distance_decay = 0.93

    def circle_factory(self, shell: int) -> mnm.Circle():
        """Create a single node circle, with standard styling.
        
        Generates a node laying on the 'shell'th circular radius from the origin.
        The node is sized and colored according to an exponential decay from the
        origin.

        Args:
            shell (int): The "shell" on which a the node is rendered. This is the number
                of steps out from the origin in polar radii. All nodes that are the same number of
                Collatz steps from 1 are on the same shell.
        """
        decay = np.power(self.circle_radius_decay, shell)
        c = mnm.Circle(radius=self.circle_radius * decay)
        c.set_stroke(color=self.circle_fill_color_start, width=self.circle_stroke_width * decay)
        fill_color = mnm.utils.color.interpolate_color(self.circle_fill_color_start,
                                                       self.circle_fill_color_end,
                                                       1.0 - decay)
        c.set_fill(color=fill_color, opacity=self.circle_fill_opacity)
        return c

    def node_factory(self, parent: NodeInfo, child_val: int) -> NodeInfo:
        """Factory to build renderable nodes & co, with standard styling.

        Args:
            parent (NodeInfo): The parent of this node in the Collatz tree.
            child_val (int): Collatz value of this node.

        Returns:
            NodeInfo: Styled node with ancillary structures (text, trace, etc.) and
                initial animations for creating (but not moving) it. Node is initialized
                at the location of its parent. 
        """
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
        """Create and animate a node in the "2n" part of the reverse Collatz process.

        This builds a node at the "2n" branch from its parent and animates moving it
        purely radially from the parent.

        Args:
            parent (NodeInfo): Parent node.

        Returns:
            NodeInfo: Fully populated child node, including necessary rendering, animation, and ancilliary
                visualization info.
        """
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
        """Generate a node from the "(2n - 1)/3" part of the reverse Collatz process.

        This creates and animates a node from the "(2n - 1)/3" branch of the tree growth
        process. We render these as radial steps around a circular "shell".

        Args:
            parent (NodeInfo): Fully populated parent node.

        Returns:
            NodeInfo: Fully populated child node, including necessary rendering, animation, and ancilliary
                visualization info.
        """
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
        """Play all animations sequentially.

        Args:
            animations (List[mnm.Animation]): Set of animations to run.
        """
        for a in animations:
            self.play(a)

    def construct(self):
        """Main 'script' for the animation.

        This sets up the "root node" at Collatz value 1 and then recursively constructs/
        animates the tree.
        """
        self.camera.frame.save_state()
        root_circle = self.circle_factory(shell=0)
        root_text = mnm.Text("1").next_to(root_circle, mnm.DOWN)
        root_node = NodeInfo(
            value=1,
            shell=0,
            color=mnm.PURPLE,
            display_node=root_circle,
            display_text=root_text,
            display_trace=None,
            polar_final_angle=0.0,
            polar_final_radius=0.0,
            animation_list=[mnm.FadeIn(root_circle, root_text)]
        )
        open = PriorityNodeQueue([root_node])
        closed: List[NodeInfo] = []
        self.camera.frame.set(height=50)
        for _ in range(self.nodes_to_generate):
            curr_node = open.pop()
            # self.play(self.camera.frame.animate.set(width=self.step_size * next_node.value + 3))
            self.play_all(curr_node.animation_list)
            open.enqueue(self.generate_doubling_node(curr_node))
            if (curr_node.value == 2):
                self.play(mnm.Create(mnm.ArcBetweenPoints(curr_node.display_node.get_center(),
                                                          root_node.display_node.get_center(),
                                                          angle=mnm.PI,
                                                          z_index=-2,
                                                          stroke_width=self.trace_stroke_width,
                                                          stroke_color=self.trace_stroke_color)))
            if (curr_node.value > 2) and (curr_node.value % 3 == 2):
                open.enqueue(self.generate_division_node(curr_node))
            closed.append(curr_node)
