"""Frame summarization for observer-centric ARC policies."""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Iterable

from .state import ClickPoint, FrameComponent, FrameSignature


def latest_grid(frame_stack: object) -> list[list[int]]:
    """Return the latest 2D rendered frame as plain Python ints."""
    if frame_stack is None:
        return []
    data = frame_stack.tolist() if hasattr(frame_stack, "tolist") else frame_stack
    if not data:
        return []
    if _is_2d(data):
        candidate = data
    else:
        candidate = data[-1]
        if hasattr(candidate, "tolist"):
            candidate = candidate.tolist()
    if not _is_2d(candidate):
        return []
    return [[int(value) for value in row] for row in candidate]


def summarize_frame(frame_stack: object, max_components: int = 96) -> FrameSignature:
    grid = latest_grid(frame_stack)
    if not grid:
        return FrameSignature(
            frame_hash="empty",
            width=0,
            height=0,
            background=0,
            histogram={},
            non_background_ratio=0.0,
        )

    height = len(grid)
    width = max(len(row) for row in grid)
    flat = [value for row in grid for value in row]
    histogram = dict(Counter(flat))
    background = _background_color(grid)
    frame_hash = _hash_grid(grid)
    non_background = sum(1 for value in flat if value != background and value >= 0)
    ratio = non_background / max(1, len(flat))
    components = _components(grid, background, max_components)
    salience_points = _salient_points(grid, background, components)

    return FrameSignature(
        frame_hash=frame_hash,
        width=width,
        height=height,
        background=background,
        histogram=histogram,
        non_background_ratio=ratio,
        components=components,
        salience_points=salience_points,
    )


def frame_distance(left: FrameSignature, right: FrameSignature) -> float:
    if left.frame_hash == right.frame_hash:
        return 0.0
    colors = set(left.histogram) | set(right.histogram)
    total = max(1, left.width * left.height, right.width * right.height)
    diff = sum(abs(left.histogram.get(color, 0) - right.histogram.get(color, 0)) for color in colors)
    shape_delta = abs(len(left.components) - len(right.components)) / max(1, len(left.components) + len(right.components))
    return min(1.0, diff / total + 0.3 * shape_delta)


def _is_2d(data: object) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], list)
        and (not data[0] or not isinstance(data[0][0], list))
    )


def _background_color(grid: list[list[int]]) -> int:
    border: list[int] = []
    if not grid:
        return 0
    height = len(grid)
    width = max(len(row) for row in grid)
    for x in range(width):
        if x < len(grid[0]):
            border.append(grid[0][x])
        if height > 1 and x < len(grid[-1]):
            border.append(grid[-1][x])
    for y in range(height):
        if grid[y]:
            border.append(grid[y][0])
            border.append(grid[y][-1])
    if border:
        return Counter(border).most_common(1)[0][0]
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def _hash_grid(grid: list[list[int]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for row in grid:
        digest.update(bytes((value + 128) % 256 for value in row))
        digest.update(b"|")
    return digest.hexdigest()


def _components(
    grid: list[list[int]], background: int, max_components: int
) -> list[FrameComponent]:
    height = len(grid)
    width = max(len(row) for row in grid)
    visited: set[tuple[int, int]] = set()
    components: list[FrameComponent] = []

    for y in range(height):
        for x in range(len(grid[y])):
            if (x, y) in visited:
                continue
            color = grid[y][x]
            if color == background or color < 0:
                visited.add((x, y))
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited.add((x, y))
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in _neighbors(cx, cy, width, height):
                    if (nx, ny) in visited or nx >= len(grid[ny]):
                        continue
                    if grid[ny][nx] != color:
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            if cells:
                xs = [cell[0] for cell in cells]
                ys = [cell[1] for cell in cells]
                area = len(cells)
                bbox = (min(xs), min(ys), max(xs), max(ys))
                centroid = (sum(xs) / area, sum(ys) / area)
                edge_touch = bbox[0] == 0 or bbox[1] == 0 or bbox[2] >= width - 1 or bbox[3] >= height - 1
                components.append(
                    FrameComponent(
                        color=color,
                        area=area,
                        bbox=bbox,
                        centroid=centroid,
                        edge_touch=edge_touch,
                    )
                )
                if len(components) >= max_components:
                    return _rank_components(components, width, height)
    return _rank_components(components, width, height)


def _neighbors(x: int, y: int, width: int, height: int) -> Iterable[tuple[int, int]]:
    if x > 0:
        yield (x - 1, y)
    if x + 1 < width:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y + 1 < height:
        yield (x, y + 1)


def _rank_components(
    components: list[FrameComponent], width: int, height: int
) -> list[FrameComponent]:
    def score(component: FrameComponent) -> tuple[float, int]:
        cx, cy = component.centroid
        center = 1.0 - min(1.0, ((cx - width / 2) ** 2 + (cy - height / 2) ** 2) ** 0.5 / max(width, height))
        compact = min(1.0, component.area / max(1, component.width * component.height))
        edge_penalty = -0.25 if component.edge_touch and component.area > width * height * 0.12 else 0.0
        return (0.45 * center + 0.35 * compact + 0.2 * min(1.0, component.area / 64.0) + edge_penalty, component.area)

    return sorted(components, key=score, reverse=True)


def _salient_points(
    grid: list[list[int]],
    background: int,
    components: list[FrameComponent],
    limit: int = 40,
) -> list[ClickPoint]:
    height = len(grid)
    width = max(len(row) for row in grid)
    total = max(1, width * height)
    color_counts = Counter(value for row in grid for value in row)
    points: list[ClickPoint] = []

    for component in components[:24]:
        rarity = 1.0 - min(1.0, color_counts[component.color] / total)
        area_score = min(1.0, component.area / 48.0)
        edge_penalty = 0.2 if component.edge_touch else 0.0
        salience = max(0.05, 0.55 * rarity + 0.35 * area_score - edge_penalty)
        cx = int(round(component.centroid[0]))
        cy = int(round(component.centroid[1]))
        points.append(ClickPoint(_clamp(cx), _clamp(cy), "component", salience, component.color))

    for x, y in _coarse_grid_points(width, height):
        if y < len(grid) and x < len(grid[y]) and grid[y][x] != background:
            points.append(ClickPoint(_clamp(x), _clamp(y), "contrast-grid", 0.35, grid[y][x]))

    for x, y in _ui_probe_points():
        points.append(ClickPoint(x, y, "coverage-grid", 0.12, None))

    return _dedupe_points(points, limit)


def _coarse_grid_points(width: int, height: int) -> list[tuple[int, int]]:
    xs = [max(0, min(width - 1, value)) for value in (8, 16, 24, 32, 40, 48, 56)]
    ys = [max(0, min(height - 1, value)) for value in (8, 16, 24, 32, 40, 48, 56)]
    return [(x, y) for y in ys for x in xs]


def _ui_probe_points() -> list[tuple[int, int]]:
    return [
        (8, 8),
        (32, 8),
        (56, 8),
        (8, 32),
        (32, 32),
        (56, 32),
        (8, 56),
        (24, 56),
        (40, 56),
        (56, 56),
    ]


def _dedupe_points(points: list[ClickPoint], limit: int) -> list[ClickPoint]:
    ordered = sorted(points, key=lambda point: point.salience, reverse=True)
    out: list[ClickPoint] = []
    occupied: set[tuple[int, int]] = set()
    for point in ordered:
        key = point.key(bucket=4)
        if key in occupied:
            continue
        occupied.add(key)
        out.append(point)
        if len(out) >= limit:
            break
    return out


def _clamp(value: int) -> int:
    return max(0, min(63, value))
