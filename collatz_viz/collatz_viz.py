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

import manim as mnm
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Any, Dict, Union, Mapping
import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike
import heapq


LOG = mnm.logger
CONSOLE = mnm.console


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
    # The displayable circle mobject representing this value.
    display_node: mnm.Circle = field(compare=False)
    # The displayable text attached to this node.
    display_text: mnm.Text = field(compare=False)
    # A visual element that traces the path the node takes as we animate it.
    display_trace: Optional[mnm.TracedPath] = field(default=None, compare=False)
    # A dot to animate on the number line.
    display_dot: mnm.Dot = field(default_factory=mnm.Dot, compare=False)
    # The final placed location of the node, as measured in polar coordinates.
    polar_final_angle: float = field(default=0.0, compare=False)
    polar_final_radius: float = field(default=0.0, compare=False)
    # The polar width subtended by the most recent ancestor branching node.
    # (The (n-1)/3 "division" node, which, in this visualization, generates a
    # branch along a circular arc.)
    polar_segment_width: float = field(default=mnm.PI, compare=False)
    # The set of animations required to draw this node and its ancillary
    # visualizations (text, trace, etc.) This can be a single animation or, to
    # compose multiple animations, use mnm.Succession or mnm.AnimationGroup
    animations: Optional[mnm.Animation] = field(default=None, compare=False)


class PriorityNodeQueue(object):
    """A bog standard priority queue, specialized for NodeInfos."""
    def __init__(self, first_elements: Iterable[NodeInfo] = []) -> None:
        self.queue: List[NodeInfo] = list()
        for e in first_elements:
            self.enqueue(e)

    def enqueue(self, element: NodeInfo):
        heapq.heappush(self.queue, element)

    def enqueue_all(self, elements: Iterable[NodeInfo]):
        for e in elements:
            self.enqueue(e)

    def pop(self) -> NodeInfo:
        return heapq.heappop(self.queue)

    def pop_all(self) -> List[NodeInfo]:
        result = []
        if len(self.queue) < 1:
            return result
        target_shell = self.queue[0].shell
        while len(self.queue) > 0 and self.queue[0].shell == target_shell:
            result.append(self.pop())
        return result


class MonotonicBBoxMovingCamera(mnm.MovingCamera):
    def __init__(self, content_bbox_bounds: ArrayLike = np.array([-5, -5, 36, 36]), margin: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        # A bounding box for all visible elements, represented as [x_ll, y_ll, x_ur, y_ur].
        # Used to set the frame for the scene camera.
        self.content_bbox = np.array([+np.Inf, +np.Inf, -np.Inf, -np.Inf])
        # Some sanity bounds to keep the bbox from escaping to infinity as we add more
        # and more distant nodes.
        self.content_bbox_bounds: np.ndarray = np.asarray(content_bbox_bounds)
        self.margin = margin

    def add_to_bbox(self, mobject: Union[mnm.Mobject, ArrayLike]):
        """Add the bounding boxes of the arguments to the current scene content bounding box.

        Args:
            mobject (Union[mnm.Mobject, NodeInfo, ArrayLike]): Object or coordinates to add to
                the scene bounding box. Special handling by input type:
                    * Mobject: Adds bounding box directly.
                    * ArrayLike: Assumed to be a bbox in [x_ll, y_ll, x_ur, y_ur] coordinates.
        """
        if isinstance(mobject, mnm.Mobject):
            bbox_to_add = np.array([mobject.get_left()[0],
                                    mobject.get_bottom()[1],
                                    mobject.get_right()[0],
                                    mobject.get_top()[1]])
        else:
            bbox_to_add = np.asarray(mobject)
            if bbox_to_add.shape != (4,):
                raise ValueError(f'Caller provided bad bounding box shape '
                                 f'({mobject} with shape {bbox_to_add.shape}) when shape (4,) expected')
        self.content_bbox[0:2] = np.minimum(self.content_bbox[0:2], bbox_to_add[0:2])
        self.content_bbox[0:2] = np.maximum(self.content_bbox[0:2], self.content_bbox_bounds[0:2])
        self.content_bbox[2:4] = np.maximum(self.content_bbox[2:4], bbox_to_add[2:4])
        self.content_bbox[2:4] = np.minimum(self.content_bbox[2:4], self.content_bbox_bounds[2:4])

    def get_content_bbox(self):
        return self.content_bbox

    def set_frame_from_bbox(self, animate: bool = True) -> Union[mnm.Mobject, mnm.Animation]:
        # This logic is borrowed and lightly adapted from MovingCamera.auto_zoom().
        # I'm not _totally_ sure that it's the right thing for our use, but it makes
        # a good starting point.
        center = (self.content_bbox[2:4] + self.content_bbox[0:2]) / 2
        new_width_height = np.abs(self.content_bbox[2:4] - self.content_bbox[0:2])
        m_target = self.frame.animate if animate else self.frame
        if (new_width_height[0] / self.frame.width) > (new_width_height[1] / self.frame.height):
            return m_target.set_x(center[0]).set_y(center[1]).set(width=new_width_height[0] + self.margin)
        else:
            return m_target.set_x(center[0]).set_y(center[1]).set(height=new_width_height[1] + self.margin)

    def set_content_bbox_bounds(self, bounds: ArrayLike):
        bounds = np.asarray(bounds)
        if bounds.shape != (4,):
            raise ValueError(f'Proposed content bounding box outer bounds is wrong shape. Got {bounds.shape}, '
            'but expected (4,)')
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f'Illegal/ill-formed content outer bounds for {self.__class__.__name__}: '
                             f'upper right corner, {bounds[2:4]}, either not above or to the right of lower '
                             f'left corner, {bounds[0:2]}')
        self.content_bbox_bounds = bounds


def polar_to_cartesian(point_p: ArrayLike, origin: ArrayLike = mnm.ORIGIN) -> np.ndarray:
    """Convert polar to Cartesian coordinates.

    Args:
        point_p: (ArrayLike): Polar coordinate point in [theta, radius, z] format.
        origin (ArrayLike, optional): [x, y, z] coordinates of Cartesian origin. Defaults to mnm.ORIGIN.

    Returns:
        np.ndarray: [x, y, z] coordinate of point in Cartesian space, where (x, y) are
            computed from (theta, radius) and z is copied from point_p.
    """
    point_p = np.asarray(point_p)
    origin = np.asarray(origin)
    return np.array([point_p[1] * np.cos(point_p[0]), point_p[1] * np.sin(point_p[0]), point_p[2]]) + origin


def cartesian_to_polar(c_point: ArrayLike, origin: ArrayLike = mnm.ORIGIN) -> np.ndarray:
    """Convert Cartesian to polar coordinates.

    Args:
        c_point (ArrayLike): Cartesian point in [x, y, z] coordinate form.
        origin (ArrayLike, optional): Origin of the Cartesian coordinate system. Defaults to mnm.ORIGIN.

    Returns:
        np.ndarray: [theta, radius, z], where (theta, radius) are computed from (x, y) and
            z is copied from origin.
    """
    c_point = np.asarray(c_point)
    origin = np.asarray(origin)
    delta = c_point - origin
    return np.array([np.arctan2(delta[1], delta[0]),
                     norm(delta[0:2]),
                     delta[2]])


class CollatzViz(mnm.MovingCameraScene):
    """The main visualization.

    This animates the growth of the reverse Collatz process as a tree, laid out with
    a circular layout style and exponentially decaying node sizes and arc lengths, so
    that we can fit it all in a bounded space.
    """

    def __init__(self, shells_to_generate: int = 5, **kwargs: Dict[str, Any]):
        """Configure Collatz process viz.

        Args:
            nodes_to_generate (int, optional): Number of nodes to generate/animate.
                Larger numbers makes bigger, more elaborate visualizations, but takes longer
                to render. Defaults to 100.
        """
        super().__init__(camera_class=MonotonicBBoxMovingCamera, **kwargs)
        self.open: PriorityNodeQueue = PriorityNodeQueue()
        self.closed: Mapping[int, NodeInfo] = dict()
        self.shells_to_generate = shells_to_generate
        # Starting point of the visualization.
        self.origin = mnm.ORIGIN
        # Initial circle radius for root node.
        self.circle_radius = 1.0
        # We use a color gradient of nodes. These are the start and end of the color
        # gradient.
        self.circle_fill_color_start = mnm.DARK_BLUE
        self.circle_fill_color_end = mnm.RED
        # How opaque to color each node circle.
        self.circle_fill_opacity = 1.0
        # Color of node edge.
        self.circle_stroke_color = mnm.PURPLE
        # Thickness of node edge.
        self.circle_stroke_width = 8
        # How big of location dots to render on the number line.
        self.dot_radius = 0.45
        # Dot opacity.
        self.dot_opacity = 0.7
        # Color of trace that we draw as we move nodes.
        self.trace_stroke_color = mnm.BLUE_A
        # Thickness of trace.
        self.trace_stroke_width = 8
        # Fundamental "step size": distance between final placement of nodes.
        self.step_size = 6.0 * self.circle_radius
        # Starting angular width of zero'th branching segment. In practice, this will
        # immediately be halved when drawing the first branch.
        self.initial_segment_angle = mnm.PI / 2.0
        # Exponential decay factors: How quickly we decay circle sizes, and step sizes as we
        # move out in "shells" (i.e., tree depth) from the origin.
        #
        # Decay in the length of radial segments with shell.
        self.distance_decay = 0.92
        # Decay of node circle radius with shell.
        self.circle_radius_decay = self.distance_decay - 0.05
        # Distance that the geometric decay of radial distance will converge to.
        self.convergence_distance = self.step_size * self.distance_decay / (1 - self.distance_decay)
        self.number_line: mnm.NumberLine = self.number_line_factory()
        # Set some outer bounds to the camera bounding box based on most distant
        # expected nodes.
        self.edge_buffer = 5 * self.circle_radius
        self.camera.set_content_bbox_bounds([
            (self.origin + self.convergence_distance * mnm.LEFT)[0] - self.edge_buffer,
            self.number_line.get_bottom()[1] - 2 * self.edge_buffer,
            (self.origin + self.convergence_distance * mnm.RIGHT)[0] + self.edge_buffer,
            (self.origin + self.convergence_distance * mnm.UP)[1] + self.edge_buffer
        ])
        self.base_animation_run_time_seconds = 1.5

    def get_color_by_shell(self, shell: int) -> mnm.color.Color:
        """Generate a color scaled by radial shell from the origin.

        :param shell: The tree depth (i.e., shell) at which we're rendering.
        :type shell: int
        :return: Color for this shell, interpolated between start and end colors.
        :rtype: mnm.color.Color
        """
        decay = np.power(self.circle_radius_decay, shell)
        return mnm.utils.color.interpolate_color(self.circle_fill_color_start,
                                                 self.circle_fill_color_end,
                                                 1.0 - decay)

    def circle_factory(self, shell: int) -> mnm.Circle:
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
        c.set_fill(color=self.get_color_by_shell(shell=shell), opacity=self.circle_fill_opacity)
        return c

    def dot_factory(self, shell: int) -> mnm.Dot:
        return mnm.Dot(radius=self.dot_radius,
                       fill_opacity=self.dot_opacity,
                       color=self.get_color_by_shell(shell=shell))

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
            display_node=self.circle_factory(child_shell),
            display_text=mnm.Text(str(child_val)),
            display_dot=self.dot_factory(child_shell),
            polar_segment_width=parent.polar_segment_width,
        )
        child_node.display_node.move_to(parent.display_node)
        child_node.display_text.match_style(parent.display_text)
        child_node.display_text.clear_updaters()
        child_node.display_text.add_updater(lambda x: x.next_to(child_node.display_node, mnm.DOWN))
        child_node.display_trace = mnm.TracedPath(child_node.display_node.get_center,
                                                  stroke_width=self.trace_stroke_width,
                                                  stroke_color=self.trace_stroke_color,
                                                  z_index=-2)
        child_node.display_dot.move_to(self.number_line.number_to_point(parent.value) + mnm.OUT)
        child_node.animations = mnm.AnimationGroup(mnm.FadeIn(child_node.display_dot),
                                                   mnm.FadeIn(child_node.display_node,
                                                              child_node.display_text,
                                                              child_node.display_trace))
        return child_node

    def number_line_factory(self) -> mnm.NumberLine:
        numberline_end = int(np.floor(self.convergence_distance * 2))
        result = mnm.NumberLine(x_range=[0, numberline_end, 2],
                                numbers_with_elongated_ticks=range(0, numberline_end, 2),
                                longer_tick_multiple=3,
                                include_numbers=True,
                                stroke_width=2 * self.trace_stroke_width,
                                font_size=64,
                                line_to_number_buff=mnm.LARGE_BUFF)
        result.move_to(mnm.DOWN * 5 * self.circle_radius - result.number_to_point(1) + mnm.IN)
        return result

    def generate_doubling_node(self, parent: NodeInfo, polar_segment_width: Optional[float] = None) -> NodeInfo:
        """Create and animate a node in the "2n" part of the reverse Collatz process.

        This builds a node at the "2n" branch from its parent and animates moving it
        purely radially from the parent.

        Args:
            parent (NodeInfo): Parent node.
            polar_segment_width (NodeInfo): Declare that this node is a member of a circular arc subtending
                this angle (radians).

        Returns:
            NodeInfo: Fully populated child node, including necessary rendering, animation, and ancilliary
                visualization info.
        """
        child_val = 2 * parent.value
        child_node = self.node_factory(parent=parent, child_val=child_val)
        move_dist = np.power(self.distance_decay, child_node.shell) * self.step_size
        child_node.polar_final_angle = parent.polar_final_angle
        child_node.polar_final_radius = parent.polar_final_radius + move_dist
        if polar_segment_width is not None:
            child_node.polar_segment_width = polar_segment_width
        here = child_node.display_node.get_center()
        move_path = mnm.Line(start=here,
                             end=np.array([here[0] + move_dist * np.cos(child_node.polar_final_angle),
                                           here[1] + move_dist * np.sin(child_node.polar_final_angle),
                                           0.0]))
        dot_animation = child_node.display_dot.animate(run_time=self.base_animation_run_time_seconds).move_to(self.number_line.number_to_point(child_node.value) + mnm.OUT)
        child_node.animations = mnm.Succession(child_node.animations,
                                               mnm.AnimationGroup(mnm.MoveAlongPath(child_node.display_node, move_path, run_time=self.base_animation_run_time_seconds),
                                                                  dot_animation))
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
        path_start_polar = cartesian_to_polar(child_node.display_node.get_center(), origin=self.origin)
        move_dist = np.power(self.distance_decay, child_node.shell) * self.step_size
        arc_start_polar = path_start_polar + np.array([0, move_dist / 2, 0])
        first_segment = mnm.Line(start=polar_to_cartesian(path_start_polar, origin=self.origin),
                                 end=polar_to_cartesian(arc_start_polar, origin=self.origin))
        # TODO(hlane) This can probably be made smarter. Start here and see if this at least works.
        angular_fraction = (3.0 / 4.0) if (child_val % 3 == 0) else (1.0 / 2.0)
        turn_angle = parent.polar_segment_width * angular_fraction
        arc_end_polar = arc_start_polar + np.array([turn_angle, 0, 0])
        arc_segment = mnm.Arc(radius=arc_start_polar[1],
                              arc_center=self.origin,
                              start_angle=arc_start_polar[0],
                              angle=turn_angle)
        second_segment_end_polar = arc_end_polar + np.array([0, move_dist / 2, 0])
        second_segment = mnm.Line(start=polar_to_cartesian(arc_end_polar, origin=self.origin),
                                  end=polar_to_cartesian(second_segment_end_polar, origin=self.origin))
        dot_animation = child_node.display_dot.animate(run_time=self.base_animation_run_time_seconds).move_to(self.number_line.number_to_point(child_node.value) + mnm.OUT)
        path_anim = mnm.Succession(
            child_node.animations,
            mnm.AnimationGroup(dot_animation,
                               mnm.Succession(mnm.MoveAlongPath(child_node.display_node, first_segment),
                                              mnm.MoveAlongPath(child_node.display_node, arc_segment),
                                              mnm.MoveAlongPath(child_node.display_node, second_segment),
                                              run_time=self.base_animation_run_time_seconds)))
        child_node.animations = path_anim
        child_node.polar_final_angle = second_segment_end_polar[0]
        child_node.polar_final_radius = second_segment_end_polar[1]
        child_node.polar_segment_width = turn_angle
        return child_node

    def animate_back_arc(self, ancestor_node: NodeInfo, curr_node: NodeInfo) -> mnm.Animation:
        """Produce an animation of an arc being drawn back from curr_node to ancestor_node.

        Args:
            curr_node (NodeInfo): Node originating the back arc.
            ancestor_node (NodeInfo): Target of the back arc.

        Returns:
            mnm.Animation: Animation that draws the back arc.
        """
        arc_start = curr_node.display_node.get_center() + self.distance_decay * self.step_size / 2 * mnm.RIGHT
        first_segment = mnm.Line(curr_node.display_node.get_center(),
                                 arc_start,
                                 z_index=-2,
                                 stroke_width=self.trace_stroke_width,
                                 stroke_color=self.trace_stroke_color)
        back_arc = mnm.ArcBetweenPoints(arc_start,
                                        ancestor_node.display_node.get_center(),
                                        angle=mnm.PI,
                                        z_index=-2,
                                        stroke_width=self.trace_stroke_width,
                                        stroke_color=self.trace_stroke_color)
        self.camera.add_to_bbox(back_arc)
        # TODO(hlane): SUPER ANNOYING! How to synchronize this to the movement of the doubling
        # node that should be taking place simultaneously? If I omit
        #   run_time=self.base_animation_run_time_seconds
        # here, then this runs faster than the doubling node. But if I _include_ the run_time
        # parameter, then manim _sequences_ the two and this runs to completion before the other
        # runs at all! Grrr.
        return mnm.AnimationGroup(mnm.Succession(mnm.Create(first_segment),
                                                 mnm.Create(back_arc)))

    def update_camera_from_node(self, new_node: NodeInfo):
        """Expand the camera's viewport to include the contents of the new node.

        Args:
            new_node (NodeInfo): Object to include in the viewport.
        """
        self.camera.add_to_bbox(new_node.display_node)
        self.camera.add_to_bbox(new_node.display_text)
        # Final position of display dot.
        dot_center = self.number_line.number_to_point(new_node.value)
        dot_radius = new_node.display_dot.radius
        dot_bbox = np.array([dot_center[0] - dot_radius,
                             dot_center[1] - dot_radius,
                             dot_center[0] + dot_radius,
                             dot_center[1] + dot_radius])
        self.camera.add_to_bbox(dot_bbox)
        # Final position of display node.
        node_center = polar_to_cartesian([new_node.polar_final_angle,
                                          new_node.polar_final_radius,
                                          self.origin[2]])
        node_radius = new_node.display_node.radius
        node_bbox = np.array([node_center[0] - node_radius,
                              node_center[1] - node_radius,
                              node_center[0] + node_radius,
                              node_center[1] + node_radius])
        self.camera.add_to_bbox(node_bbox)

    def generate_child_nodes(self, parent: NodeInfo) -> List[NodeInfo]:
        """Generate a list of all children of the given node.

        Args:
            parent (NodeInfo): Starting point for generation.

        Returns:
            List[NodeInfo]: Both the doubling value and, when applicable, the (2n - 1) / 3
                child nodes. SPECIAL CASE: If this detects that the child has previously been
                generated, a child will still be returned, but its animation will contain the
                reverse arc, leading from parent to the already generated ancestor node.
        """
        result: List[NodeInfo] = []
        polar_segment_width = None
        if (parent.value % 3 == 2):
            branching_node = self.generate_division_node(parent)
            result.append(branching_node)
            polar_segment_width = branching_node.polar_segment_width
        result.append(self.generate_doubling_node(parent,
                                                  polar_segment_width=polar_segment_width))
        for child in result:
            if child.value in self.closed:
                child.animations = self.animate_back_arc(self.closed[child.value], parent)
        return result

    def construct(self):
        """Main 'script' for the animation.

        This sets up the "root node" at Collatz value 1 and then recursively constructs/
        animates the tree.
        """
        self.camera.frame.save_state()
        self.add(self.number_line)
        root_circle = self.circle_factory(shell=0)
        root_text = mnm.Text("1").next_to(root_circle, mnm.DOWN)
        root_dot = self.dot_factory(shell=0).move_to(self.number_line.number_to_point(1) + mnm.OUT)
        root_node = NodeInfo(
            value=1,
            shell=0,
            display_node=root_circle,
            display_text=root_text,
            display_trace=None,
            display_dot=root_dot,
            polar_final_angle=0.0,
            polar_final_radius=0.0,
            polar_segment_width=self.initial_segment_angle,
            animations=mnm.FadeIn(root_circle, root_text, root_dot),
        )
        self.open.enqueue(root_node)
        # Initial camera view.
        self.camera.add_to_bbox(np.array([
            np.minimum(root_circle.get_left()[0], self.number_line.get_left()[0]) - self.edge_buffer,
            self.number_line.get_bottom()[1] - self.edge_buffer,
            root_circle.get_right()[0] + self.edge_buffer,
            root_circle.get_top()[1] + self.edge_buffer
        ]))
        self.camera.set_frame_from_bbox(animate=False)
        for _ in range(self.shells_to_generate):
            nodes_to_close = self.open.pop_all()
            LOG.info(f'Working shell = {nodes_to_close[0].shell}; node count = {len(nodes_to_close)}')
            # Note: We need to process the list of nodes to work _twice_:
            #  - In the first phase, we accumulate all of the animations to be played for this shell's
            #    nodes, then we play them. This updates their state (specifically, their location).
            #  - In the second phase, we compute and open the children of each node.
            # It's important that we play the animations for each node before opening its children;
            # otherwise, their state isn't up to date and we're starting their children from their
            # _start_ location, not their _end_ location. (Net result being that everything starts from
            # the origin.)
            for curr_node in nodes_to_close:
                self.update_camera_from_node(curr_node)
            self.play(self.camera.set_frame_from_bbox(animate=True), *[n.animations for n in nodes_to_close])
            for curr_node in nodes_to_close:
                if curr_node.value not in self.closed:
                    self.closed[curr_node.value] = curr_node
                    self.open.enqueue_all(self.generate_child_nodes(curr_node))
