import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def load_xyz(path):
    with np.load(path) as data:
        if "coords" in data:
            coords = np.asarray(data["coords"])
            if coords.ndim != 2 or coords.shape[1] != 4:
                raise ValueError(f"{path}: 'coords' must have shape [N, 4], got {coords.shape}")
            xyz = coords[:, 1:4]
        elif "q_xyz" in data:
            xyz = np.asarray(data["q_xyz"])
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(f"{path}: 'q_xyz' must have shape [N, 3], got {xyz.shape}")
        else:
            raise ValueError(f"{path}: expected 'coords' [N, 4] or 'q_xyz' [N, 3]")

    if not np.issubdtype(xyz.dtype, np.integer):
        if not np.all(np.isfinite(xyz)) or not np.all(xyz == np.round(xyz)):
            raise ValueError(f"{path}: coordinates must be integer-valued")
        xyz = np.round(xyz)
    return xyz.astype(np.int64, copy=False)


def neighbor_offsets(connectivity):
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan == 1:
                    offsets.append((dx, dy, dz))
                elif connectivity == 18 and manhattan <= 2:
                    offsets.append((dx, dy, dz))
                elif connectivity == 26:
                    offsets.append((dx, dy, dz))
    return offsets


OFFSETS = {
    6: neighbor_offsets(6),
    18: neighbor_offsets(18),
    26: neighbor_offsets(26),
}


def connected_components(points, offsets):
    point_set = set(points)
    seen = set()
    components = 0

    for point in points:
        if point in seen:
            continue
        components += 1
        seen.add(point)
        queue = deque([point])
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in point_set and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return components


def neighbor_counts(points, offsets):
    point_set = set(points)
    counts = []
    for x, y, z in points:
        counts.append(sum((x + dx, y + dy, z + dz) in point_set for dx, dy, dz in offsets))
    return np.asarray(counts, dtype=np.int64)


def axis_histograms(xyz, resolution):
    histograms = {}
    for axis, name in enumerate(("x", "y", "z")):
        counts, edges = np.histogram(xyz[:, axis], bins=32, range=(0, resolution))
        histograms[name] = {
            "counts": counts.astype(int).tolist(),
            "bin_edges": edges.tolist(),
        }
    return histograms


def profile_file(path, resolution):
    xyz = load_xyz(path)
    token_count = int(xyz.shape[0])
    unique_xyz = np.unique(xyz, axis=0)
    unique_count = int(unique_xyz.shape[0])
    duplicate_count = token_count - unique_count
    in_bounds = bool(np.all((xyz >= 0) & (xyz < resolution)))

    if token_count:
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        xyz_extent = xyz_max - xyz_min
        bbox_volume = int(np.prod(xyz_extent + 1))
    else:
        xyz_min = xyz_max = xyz_extent = np.zeros(3, dtype=np.int64)
        bbox_volume = 0

    graph_xyz = unique_xyz[(unique_xyz >= 0).all(axis=1) & (unique_xyz < resolution).all(axis=1)]
    points = [tuple(row.tolist()) for row in graph_xyz]
    counts_6 = neighbor_counts(points, OFFSETS[6]) if points else np.asarray([], dtype=np.int64)
    counts_18 = neighbor_counts(points, OFFSETS[18]) if points else np.asarray([], dtype=np.int64)
    counts_26 = neighbor_counts(points, OFFSETS[26]) if points else np.asarray([], dtype=np.int64)
    density_hist, density_edges = np.histogram(counts_26, bins=np.arange(28))

    if resolution % 2 == 0 and token_count:
        parent_count = int(np.unique(xyz // 2, axis=0).shape[0])
    elif resolution % 2 == 0:
        parent_count = 0
    else:
        parent_count = None

    return {
        "path": str(path),
        "resolution": int(resolution),
        "token_count": token_count,
        "xyz_min": xyz_min.astype(int).tolist(),
        "xyz_max": xyz_max.astype(int).tolist(),
        "xyz_extent": xyz_extent.astype(int).tolist(),
        "occupancy_ratio": token_count / float(resolution ** 3),
        "unique_count": unique_count,
        "duplicate_count": int(duplicate_count),
        "in_bounds": in_bounds,
        "connected_components_6": connected_components(points, OFFSETS[6]) if points else 0,
        "connected_components_18": connected_components(points, OFFSETS[18]) if points else 0,
        "connected_components_26": connected_components(points, OFFSETS[26]) if points else 0,
        "neighbor_degree_mean_6": float(counts_6.mean()) if counts_6.size else 0.0,
        "neighbor_degree_mean_18": float(counts_18.mean()) if counts_18.size else 0.0,
        "neighbor_degree_mean_26": float(counts_26.mean()) if counts_26.size else 0.0,
        "isolated_token_count_26": int((counts_26 == 0).sum()) if counts_26.size else 0,
        "bbox_fill_ratio": token_count / float(bbox_volume) if bbox_volume else 0.0,
        "per_axis_histograms": axis_histograms(xyz, resolution) if token_count else {},
        "local_density_histogram_26": {
            "counts": density_hist.astype(int).tolist(),
            "bin_edges": density_edges.astype(int).tolist(),
        },
        "parent_count": parent_count,
        "parent_child_ratio": token_count / float(parent_count) if parent_count else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Profile sparse coordinate support .npz files.")
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="One or more .npz coordinate files.")
    parser.add_argument("--resolution", required=True, type=int, help="Coordinate grid resolution.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    profiles = [profile_file(path, args.resolution) for path in args.input]
    payload = profiles[0] if len(profiles) == 1 else profiles
    text = json.dumps(payload, indent=2)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
