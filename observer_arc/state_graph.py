"""ARC-native object/relation state graph utilities.

The observer policy uses this layer to reason over visible state transitions
rather than prose summaries. It intentionally consumes only rendered frame
signatures and available action metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .state import FrameComponent, FrameSignature
from .vision import frame_distance


@dataclass(frozen=True)
class ObjectNode:
    object_id: int
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    edge_touch: bool
    size_bucket: int
    x_bucket: int
    y_bucket: int

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    def abstract_signature(self) -> str:
        return (
            f"c{self.color}:s{self.size_bucket}:"
            f"x{self.x_bucket}:y{self.y_bucket}:e{int(self.edge_touch)}"
        )


@dataclass(frozen=True)
class RelationEdge:
    kind: str
    source: int
    target: int
    strength: float = 1.0

    def abstract_signature(self, objects: list[ObjectNode]) -> str:
        left = objects[self.source]
        right = objects[self.target]
        colors = sorted([left.color, right.color])
        return f"{self.kind}:c{colors[0]}-c{colors[1]}"


@dataclass(frozen=True)
class StateGraph:
    frame_hash: str
    width: int
    height: int
    background: int
    objects: list[ObjectNode] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    profile: str = ""

    def relation_signatures(self) -> set[str]:
        return {relation.abstract_signature(self.objects) for relation in self.relations}


@dataclass(frozen=True)
class ObjectMotion:
    before: ObjectNode
    after: ObjectNode
    dx: float
    dy: float

    @property
    def distance(self) -> float:
        return abs(self.dx) + abs(self.dy)


@dataclass(frozen=True)
class TransitionDelta:
    previous: StateGraph
    current: StateGraph
    action_key: str
    frame_change: float
    motions: list[ObjectMotion] = field(default_factory=list)
    created_count: int = 0
    destroyed_count: int = 0
    relation_changes: int = 0
    information_gain: float = 0.0
    family: str = "no_change"

    @property
    def moved_count(self) -> int:
        return len(self.motions)

    def compact_signature(self) -> str:
        motion_bucket = _bucket(float(self.moved_count), [0, 1, 2, 4])
        create_bucket = _bucket(float(self.created_count), [0, 1, 2, 4])
        destroy_bucket = _bucket(float(self.destroyed_count), [0, 1, 2, 4])
        relation_bucket = _bucket(float(self.relation_changes), [0, 1, 3, 8])
        return (
            f"{self.previous.profile}|{self.action_key}|{self.family}|"
            f"m{motion_bucket}:c{create_bucket}:d{destroy_bucket}:r{relation_bucket}"
        )


def build_state_graph(signature: FrameSignature, max_objects: int = 24) -> StateGraph:
    objects = [
        _object_node(index, component, signature.width, signature.height)
        for index, component in enumerate(signature.components[:max_objects])
    ]
    relations = _relations(objects)
    profile = _graph_profile(signature, objects)
    return StateGraph(
        frame_hash=signature.frame_hash,
        width=signature.width,
        height=signature.height,
        background=signature.background,
        objects=objects,
        relations=relations,
        profile=profile,
    )


def transition_delta(
    previous_signature: FrameSignature,
    current_signature: FrameSignature,
    action_key: str,
) -> TransitionDelta:
    previous = build_state_graph(previous_signature)
    current = build_state_graph(current_signature)
    matches = _match_objects(previous.objects, current.objects)
    matched_previous = {left for left, _ in matches}
    matched_current = {right for _, right in matches}
    motions: list[ObjectMotion] = []
    for left_index, right_index in matches:
        before = previous.objects[left_index]
        after = current.objects[right_index]
        dx = after.centroid[0] - before.centroid[0]
        dy = after.centroid[1] - before.centroid[1]
        if abs(dx) + abs(dy) >= 0.5:
            motions.append(ObjectMotion(before, after, dx, dy))
    created_count = max(0, len(current.objects) - len(matched_current))
    destroyed_count = max(0, len(previous.objects) - len(matched_previous))
    relation_changes = len(
        previous.relation_signatures() ^ current.relation_signatures()
    )
    frame_change = frame_distance(previous_signature, current_signature)
    information_gain = min(
        1.0,
        frame_change
        + 0.08 * min(4, len(motions))
        + 0.06 * min(5, created_count + destroyed_count)
        + 0.025 * min(8, relation_changes),
    )
    family = classify_transition(
        frame_change,
        len(motions),
        created_count,
        destroyed_count,
        relation_changes,
    )
    return TransitionDelta(
        previous=previous,
        current=current,
        action_key=action_key,
        frame_change=frame_change,
        motions=motions,
        created_count=created_count,
        destroyed_count=destroyed_count,
        relation_changes=relation_changes,
        information_gain=information_gain,
        family=family,
    )


def classify_transition(
    frame_change: float,
    moved_count: int,
    created_count: int,
    destroyed_count: int,
    relation_changes: int,
) -> str:
    if frame_change <= 0.004 and moved_count == 0 and created_count == 0 and destroyed_count == 0:
        return "no_change"
    if created_count > destroyed_count:
        return "reveal_or_create"
    if destroyed_count > created_count:
        return "remove_or_hide"
    if moved_count and relation_changes:
        return "relational_move"
    if moved_count:
        return "move"
    if relation_changes:
        return "relation_change"
    return "state_change"


def _object_node(
    index: int,
    component: FrameComponent,
    width: int,
    height: int,
) -> ObjectNode:
    x_bucket = _bucket(component.centroid[0] / max(1.0, width), [0.25, 0.50, 0.75])
    y_bucket = _bucket(component.centroid[1] / max(1.0, height), [0.25, 0.50, 0.75])
    size_bucket = _bucket(float(component.area), [3, 8, 24, 80, 240])
    return ObjectNode(
        object_id=index,
        color=component.color,
        area=component.area,
        bbox=component.bbox,
        centroid=component.centroid,
        edge_touch=component.edge_touch,
        size_bucket=size_bucket,
        x_bucket=x_bucket,
        y_bucket=y_bucket,
    )


def _relations(objects: list[ObjectNode]) -> list[RelationEdge]:
    out: list[RelationEdge] = []
    for left_index, left in enumerate(objects[:16]):
        for right_index, right in enumerate(objects[:16]):
            if left_index >= right_index:
                continue
            distance = _distance(left.centroid, right.centroid)
            if _bbox_gap(left.bbox, right.bbox) <= 1:
                out.append(RelationEdge("contact", left_index, right_index, 1.0))
            if distance <= max(6.0, math.sqrt(max(1, left.area + right.area)) * 2.2):
                out.append(RelationEdge("near", left_index, right_index, 1.0))
            if abs(left.centroid[0] - right.centroid[0]) <= 1.5:
                out.append(RelationEdge("aligned_x", left_index, right_index, 1.0))
            if abs(left.centroid[1] - right.centroid[1]) <= 1.5:
                out.append(RelationEdge("aligned_y", left_index, right_index, 1.0))
    return out


def _match_objects(
    previous: list[ObjectNode],
    current: list[ObjectNode],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(previous):
        for right_index, right in enumerate(current):
            color_cost = 0.0 if left.color == right.color else 3.0
            area_cost = abs(left.area - right.area) / max(1.0, left.area + right.area)
            distance_cost = _distance(left.centroid, right.centroid) / 16.0
            candidates.append((color_cost + area_cost + distance_cost, left_index, right_index))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int]] = []
    for cost, left_index, right_index in candidates:
        if cost > 5.0:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index))
    return matches


def _graph_profile(signature: FrameSignature, objects: list[ObjectNode]) -> str:
    density = _bucket(signature.non_background_ratio, [0.04, 0.12, 0.28, 0.50])
    object_bucket = _bucket(float(len(objects)), [1, 3, 8, 16])
    largest = max((obj.area for obj in objects), default=0)
    largest_bucket = _bucket(float(largest), [4, 16, 64, 192])
    colors = sorted({obj.color for obj in objects[:8]})
    color_key = "-".join(str(color) for color in colors[:4]) or "none"
    return (
        f"{min(64, signature.width)}x{min(64, signature.height)}"
        f":d{density}:o{object_bucket}:l{largest_bucket}:c{color_key}"
    )


def _bbox_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    x_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]) - 1)
    y_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]) - 1)
    return max(x_gap, y_gap)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _bucket(value: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)
