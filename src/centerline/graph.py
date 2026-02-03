"""Skeleton topology analysis and vessel graph construction.

Analyses a 3D skeleton volume to identify endpoints, branch points, and
individual vessel segments.  Optionally builds a NetworkX graph.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 26-connectivity offsets
# ---------------------------------------------------------------------------
_OFFSETS_26 = np.array(
    [
        [dz, dy, dx]
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.intp,
)  # shape (26, 3)


def _neighbours_26(pt: tuple[int, int, int], skel_set: set[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Return 26-connected skeleton neighbours of *pt*."""
    z, y, x = pt
    nbrs: list[tuple[int, int, int]] = []
    for dz, dy, dx in _OFFSETS_26:
        nb = (z + dz, y + dy, x + dx)
        if nb in skel_set:
            nbrs.append(nb)
    return nbrs


def _segment_length(segment: np.ndarray) -> float:
    """Euclidean length of a polyline given as an (N, 3) array."""
    if segment.shape[0] < 2:
        return 0.0
    diffs = np.diff(segment.astype(np.float64), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def build_vessel_graph(
    skeleton: np.ndarray,
) -> dict[str, Any]:
    """Analyse the topology of a 3D skeleton and extract vessel segments.

    The skeleton is examined using 26-connectivity to classify every voxel:

    * **endpoint** — exactly 1 skeleton neighbour
    * **continuation** — exactly 2 skeleton neighbours
    * **branch / junction** — 3 or more skeleton neighbours

    Individual vessel segments are traced between branch-points and/or
    endpoints using breadth-first search.  Isolated loops (cycles with no
    branch point) are detected in a second pass.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean (or 0/1) 3D array representing the one-voxel-wide
        centerline.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        - ``endpoints`` — ``(M, 3)`` int array of endpoint coordinates.
        - ``branch_points`` — ``(K, 3)`` int array of branch-point
          coordinates.
        - ``segments`` — list of ``(N_i, 3)`` int arrays, each an ordered
          sequence of voxel coordinates forming one vessel segment.
        - ``num_segments`` — total number of segments.
        - ``total_centerline_length_voxels`` — sum of all segment lengths
          (Euclidean, in voxel units).
        - ``segment_lengths`` — list of per-segment Euclidean lengths.
        - ``graph`` — a ``networkx.Graph`` if NetworkX is importable,
          otherwise ``None``.  Nodes are 3-tuples ``(z, y, x)`` with a
          ``"type"`` attribute (``"endpoint"``, ``"branch"``, or
          ``"continuation"``).  Edges carry a ``"segment"`` attribute with
          the ordered point array and a ``"length"`` attribute.
    """
    skel_bool = skeleton.astype(bool, copy=False)
    all_coords = np.argwhere(skel_bool)  # (N, 3)

    # Fast lookup set of skeleton voxel tuples.
    skel_set: set[tuple[int, int, int]] = set(map(tuple, all_coords.tolist()))

    # ------------------------------------------------------------------
    # 1. Classify every skeleton voxel
    # ------------------------------------------------------------------
    neighbor_count: dict[tuple[int, int, int], int] = {}
    for pt in skel_set:
        neighbor_count[pt] = len(_neighbours_26(pt, skel_set))

    endpoints: list[tuple[int, int, int]] = []
    branch_points: list[tuple[int, int, int]] = []

    for pt, nc in neighbor_count.items():
        if nc == 1:
            endpoints.append(pt)
        elif nc >= 3:
            branch_points.append(pt)

    endpoint_set = set(endpoints)
    branch_set = set(branch_points)
    seed_set = endpoint_set | branch_set  # starting points for tracing

    # ------------------------------------------------------------------
    # 2. Trace segments between seed points (endpoints / branch points)
    # ------------------------------------------------------------------
    segments: list[np.ndarray] = []
    visited_edges: set[frozenset[tuple[int, int, int]]] = set()
    globally_visited: set[tuple[int, int, int]] = set()

    def _trace_segment(
        start: tuple[int, int, int],
        first_step: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        """Trace from *start* through *first_step* until hitting another seed or dead end."""
        path = [start, first_step]
        current = first_step
        prev = start
        while True:
            if current in seed_set and current != start:
                # Reached another seed — segment complete.
                break
            nbrs = _neighbours_26(current, skel_set)
            # Filter out the previous point to avoid backtracking.
            forward = [n for n in nbrs if n != prev]
            if len(forward) == 0:
                # Dead end (isolated 2-voxel stub or similar).
                break
            if len(forward) == 1:
                prev, current = current, forward[0]
                path.append(current)
            else:
                # Multiple forward neighbours — current should have been
                # classified as a branch point.  Stop here.
                break
        return path

    for seed in seed_set:
        nbrs = _neighbours_26(seed, skel_set)
        for nb in nbrs:
            edge_key = frozenset((seed, nb))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            path = _trace_segment(seed, nb)
            seg_arr = np.array(path, dtype=np.intp)
            segments.append(seg_arr)

            # Mark interior (non-seed) points as globally visited.
            for pt in path:
                globally_visited.add(pt)

    # ------------------------------------------------------------------
    # 3. Detect isolated loops (cycles without any seed point)
    # ------------------------------------------------------------------
    unvisited = skel_set - globally_visited
    while unvisited:
        start = next(iter(unvisited))
        # BFS around the loop.
        loop: list[tuple[int, int, int]] = [start]
        visited_loop: set[tuple[int, int, int]] = {start}
        queue: deque[tuple[int, int, int]] = deque()

        # Walk in one direction to collect the loop.
        current = start
        while True:
            nbrs = [n for n in _neighbours_26(current, skel_set) if n not in visited_loop]
            # Also restrict to unvisited pool to avoid merging with already-traced segments.
            nbrs = [n for n in nbrs if n in unvisited]
            if not nbrs:
                break
            nxt = nbrs[0]
            loop.append(nxt)
            visited_loop.add(nxt)
            current = nxt

        # Close the loop if the last point neighbours the first.
        if len(loop) > 2:
            last_nbrs = _neighbours_26(loop[-1], skel_set)
            if start in last_nbrs:
                loop.append(start)  # close it

        seg_arr = np.array(loop, dtype=np.intp)
        segments.append(seg_arr)

        unvisited -= visited_loop

    # ------------------------------------------------------------------
    # 4. Compute segment lengths
    # ------------------------------------------------------------------
    seg_lengths = [_segment_length(s) for s in segments]
    total_length = sum(seg_lengths)

    # ------------------------------------------------------------------
    # 5. Optionally build a NetworkX graph
    # ------------------------------------------------------------------
    nx_graph: Any = None
    try:
        import networkx as nx

        G = nx.Graph()

        # Add all skeleton nodes with type labels.
        for pt in skel_set:
            nc = neighbor_count[pt]
            if nc == 1:
                ntype = "endpoint"
            elif nc >= 3:
                ntype = "branch"
            else:
                ntype = "continuation"
            G.add_node(pt, type=ntype)

        # Add edges for each segment.
        for seg_arr, length in zip(segments, seg_lengths):
            if seg_arr.shape[0] < 2:
                continue
            u = tuple(seg_arr[0].tolist())
            v = tuple(seg_arr[-1].tolist())
            G.add_edge(u, v, segment=seg_arr, length=length)

        nx_graph = G
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 6. Assemble result
    # ------------------------------------------------------------------
    endpoints_arr = np.array(endpoints, dtype=np.intp).reshape(-1, 3) if endpoints else np.empty((0, 3), dtype=np.intp)
    branch_arr = np.array(branch_points, dtype=np.intp).reshape(-1, 3) if branch_points else np.empty((0, 3), dtype=np.intp)

    return {
        "endpoints": endpoints_arr,
        "branch_points": branch_arr,
        "segments": segments,
        "num_segments": len(segments),
        "total_centerline_length_voxels": total_length,
        "segment_lengths": seg_lengths,
        "graph": nx_graph,
    }
