
import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image
import torch


# ----------------------------
# Analytic SDFs
# ----------------------------

def torus_sdf(points: torch.Tensor, major: float = 0.62, minor: float = 0.18) -> torch.Tensor:
    xy = torch.linalg.norm(points[:, :2], dim=1)
    return torch.sqrt((xy - major) ** 2 + points[:, 2] ** 2) - minor


def sphere_sdf(points: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.linalg.norm(points - center[None, :], dim=1) - radius


def capsule_sdf(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, radius: float) -> torch.Tensor:
    pa = points - a[None, :]
    ba = b - a
    h = torch.clamp((pa @ ba) / torch.clamp(ba @ ba, min=1e-8), 0.0, 1.0)
    closest = a[None, :] + h[:, None] * ba[None, :]
    return torch.linalg.norm(points - closest, dim=1) - radius


def circular_tube_sdf(
    points: torch.Tensor,
    center_x: float,
    center_y: float,
    center_z: float,
    ring_radius: float,
    tube_radius: float,
) -> torch.Tensor:
    # A torus-like circular tube lying in the XY plane, with tube axis around Z.
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    dz = points[:, 2] - center_z
    radial = torch.sqrt(dx * dx + dy * dy)
    return torch.sqrt((radial - ring_radius) ** 2 + dz ** 2) - tube_radius


def dumbbell_sdf(
    points: torch.Tensor,
    sphere_offset: float = 0.48,
    sphere_radius: float = 0.28,
    tube_radius: float = 0.055,
) -> torch.Tensor:
    device = points.device
    left_c = torch.tensor([-sphere_offset, 0.0, 0.0], device=device)
    right_c = torch.tensor([sphere_offset, 0.0, 0.0], device=device)

    left = sphere_sdf(points, left_c, sphere_radius)
    right = sphere_sdf(points, right_c, sphere_radius)

    a = torch.tensor([-sphere_offset, 0.0, 0.0], device=device)
    b = torch.tensor([sphere_offset, 0.0, 0.0], device=device)
    tube = capsule_sdf(points, a, b, tube_radius)

    return torch.minimum(torch.minimum(left, right), tube)


def glasses_sdf(
    points: torch.Tensor,
    tube_radius: float = 0.020,
    rim_center_x: float = 0.32,
    rim_radius: float = 0.22,
    bridge_y: float = 0.045,
    arm_drop_z: float = -0.22,
) -> torch.Tensor:
    # Synthetic glasses frame:
    # - two circular tube rims in the XY plane
    # - a nose bridge capsule between inner rim points
    # - two temple arms extending backwards in -Z
    device = points.device

    left_rim = circular_tube_sdf(
        points,
        center_x=-rim_center_x,
        center_y=0.0,
        center_z=0.0,
        ring_radius=rim_radius,
        tube_radius=tube_radius,
    )
    right_rim = circular_tube_sdf(
        points,
        center_x=rim_center_x,
        center_y=0.0,
        center_z=0.0,
        ring_radius=rim_radius,
        tube_radius=tube_radius,
    )

    bridge_a = torch.tensor([-rim_center_x + rim_radius * 0.86, bridge_y, 0.0], device=device)
    bridge_b = torch.tensor([rim_center_x - rim_radius * 0.86, bridge_y, 0.0], device=device)
    bridge = capsule_sdf(points, bridge_a, bridge_b, tube_radius)

    left_arm_a = torch.tensor([-rim_center_x - rim_radius * 0.96, 0.02, 0.0], device=device)
    left_arm_b = torch.tensor([-0.98, 0.06, arm_drop_z], device=device)
    right_arm_a = torch.tensor([rim_center_x + rim_radius * 0.96, 0.02, 0.0], device=device)
    right_arm_b = torch.tensor([0.98, 0.06, arm_drop_z], device=device)

    left_arm = capsule_sdf(points, left_arm_a, left_arm_b, tube_radius)
    right_arm = capsule_sdf(points, right_arm_a, right_arm_b, tube_radius)

    return torch.minimum(
        torch.minimum(torch.minimum(left_rim, right_rim), bridge),
        torch.minimum(left_arm, right_arm),
    )


def shape_sdf(points: torch.Tensor, shape: str, tube_radius: float = 0.055) -> torch.Tensor:
    if shape == "torus":
        return torus_sdf(points)
    if shape == "dumbbell":
        return dumbbell_sdf(points, tube_radius=tube_radius)
    if shape == "glasses":
        return glasses_sdf(points, tube_radius=tube_radius)
    raise ValueError(f"unknown shape: {shape}")


# ----------------------------
# Surface projection / extraction
# ----------------------------

def numerical_sdf_gradient(
    points: torch.Tensor,
    shape: str,
    tube_radius: float = 0.055,
    eps: float = 1e-3,
) -> torch.Tensor:
    grads = []

    for axis in range(3):
        d = torch.zeros_like(points)
        d[:, axis] = eps
        fp = shape_sdf(points + d, shape, tube_radius=tube_radius)
        fm = shape_sdf(points - d, shape, tube_radius=tube_radius)
        grads.append((fp - fm) / (2.0 * eps))

    return torch.stack(grads, dim=1)


def project_to_surface(
    points: torch.Tensor,
    shape: str,
    tube_radius: float = 0.055,
    steps: int = 5,
) -> torch.Tensor:
    # Newton-like SDF projection along the numerical gradient.
    p = points.clone()

    for _ in range(steps):
        phi = shape_sdf(p, shape, tube_radius=tube_radius)
        grad = numerical_sdf_gradient(p, shape, tube_radius=tube_radius)
        denom = torch.sum(grad * grad, dim=1).clamp_min(1e-8)
        p = p - (phi / denom)[:, None] * grad

    return p


def make_vertices(res: int, bounds: float, device: torch.device) -> torch.Tensor:
    n = res + 1
    axis = torch.linspace(-bounds, bounds, n, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def extract_zero_crossings(
    shape: str,
    res: int,
    bounds: float,
    device: torch.device,
    tube_radius: float = 0.055,
) -> torch.Tensor:
    n = res + 1
    verts = make_vertices(res, bounds, device)
    sdf = shape_sdf(verts, shape, tube_radius=tube_radius)

    vid = torch.arange(n ** 3, device=device, dtype=torch.long).reshape(n, n, n)

    ax = vid[:-1, :, :].reshape(-1)
    bx = vid[1:, :, :].reshape(-1)

    ay = vid[:, :-1, :].reshape(-1)
    by = vid[:, 1:, :].reshape(-1)

    az = vid[:, :, :-1].reshape(-1)
    bz = vid[:, :, 1:].reshape(-1)

    a = torch.cat([ax, ay, az], dim=0)
    b = torch.cat([bx, by, bz], dim=0)

    sdf_a = sdf[a]
    sdf_b = sdf[b]
    mask = (sdf_a * sdf_b) < 0.0

    a = a[mask]
    b = b[mask]

    sdf_a_abs = sdf[a].abs()
    sdf_b_abs = sdf[b].abs()

    va = verts[a]
    vb = verts[b]
    denom = (sdf_a_abs + sdf_b_abs).clamp_min(1e-8)

    surface_points = (
        sdf_b_abs[:, None] * va + sdf_a_abs[:, None] * vb
    ) / denom[:, None]

    return surface_points


# ----------------------------
# Sparse index path
# ----------------------------

def q_from_points(points: torch.Tensor, origin: torch.Tensor, h: float, res: int) -> torch.Tensor:
    q = torch.floor((points - origin) / h).to(torch.long)
    return torch.clamp(q, 0, res - 1)


def centers_from_q(q: torch.Tensor, origin: torch.Tensor, h: float) -> torch.Tensor:
    return origin + (q.to(torch.float32) + 0.5) * h


def hash_q(q: torch.Tensor, res: int) -> torch.Tensor:
    q = q.to(torch.long)
    return q[:, 0] + res * (q[:, 1] + res * q[:, 2])


def unique_q(q: torch.Tensor, res: int) -> torch.Tensor:
    hashes = hash_q(q, res)
    unique_hashes = torch.unique(hashes)
    x = unique_hashes % res
    y = (unique_hashes // res) % res
    z = unique_hashes // (res * res)
    return torch.stack([x, y, z], dim=1).to(torch.long)


def make_offsets(factor: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(factor, device=device, dtype=torch.long)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def children_band(
    shape: str,
    coarse_q: torch.Tensor,
    coarse_res: int,
    target_res: int,
    origin: torch.Tensor,
    h_target: float,
    band_voxels: float,
    device: torch.device,
    tube_radius: float = 0.055,
) -> torch.Tensor:
    factor = target_res // coarse_res
    if target_res % coarse_res != 0:
        raise ValueError("target_res must be divisible by coarse_res")

    offsets = make_offsets(factor, device)
    children = coarse_q[:, None, :] * factor + offsets[None, :, :]
    children = children.reshape(-1, 3)

    valid = torch.all((children >= 0) & (children < target_res), dim=1)
    children = children[valid]

    centers = centers_from_q(children, origin, h_target)
    sdf_abs = shape_sdf(centers, shape, tube_radius=tube_radius).abs()

    keep = sdf_abs <= (band_voxels * h_target)
    return unique_q(children[keep], target_res)


# ----------------------------
# Camera / visibility path
# ----------------------------

def project(points_view: torch.Tensor, width: int, height: int, fx: float, fy: float):
    cx = width * 0.5
    cy = height * 0.5
    z = points_view[:, 2].clamp_min(1e-6)
    u = fx * (points_view[:, 0] / z) + cx
    v = fy * (points_view[:, 1] / z) + cy
    return u, v, z


def scatter_zbuffer(
    surface_points: torch.Tensor,
    width: int,
    height: int,
    fx: float,
    fy: float,
    z_shift: float,
    splat_radius: int,
    device: torch.device,
):
    points_view = surface_points + torch.tensor([0.0, 0.0, z_shift], device=device)
    u, v, z = project(points_view, width, height, fx, fy)

    base_u = torch.floor(u).to(torch.long)
    base_v = torch.floor(v).to(torch.long)

    depth = torch.full((height * width,), float("inf"), device=device)

    offsets = []
    for dv in range(-splat_radius, splat_radius + 1):
        for du in range(-splat_radius, splat_radius + 1):
            offsets.append((du, dv))

    all_idx = []
    all_depths = []

    for du, dv in offsets:
        uu = base_u + du
        vv = base_v + dv
        valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height) & torch.isfinite(z)
        if valid.any():
            idx = vv[valid] * width + uu[valid]
            penalty = 1e-4 * float(du * du + dv * dv)
            all_idx.append(idx)
            all_depths.append(z[valid] + penalty)

    if all_idx:
        idx = torch.cat(all_idx)
        vals = torch.cat(all_depths)
        depth.scatter_reduce_(0, idx, vals, reduce="amin", include_self=True)

    return depth.reshape(height, width)


def sample_depth_nearest(depth: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    height, width = depth.shape
    uu = torch.floor(u).to(torch.long)
    vv = torch.floor(v).to(torch.long)

    valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
    out = torch.full_like(u, float("inf"), dtype=torch.float32)
    out[valid] = depth[vv[valid], uu[valid]]
    return out, valid


# ----------------------------
# Metrics / export
# ----------------------------

def count_components_26(q_cpu: np.ndarray, max_nodes: int = 250_000) -> int | None:
    if q_cpu.shape[0] > max_nodes:
        return None

    pts = [tuple(map(int, row)) for row in q_cpu]
    remaining = set(pts)

    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))

    comps = 0
    while remaining:
        comps += 1
        start = remaining.pop()
        dq = deque([start])

        while dq:
            p = dq.popleft()
            px, py, pz = p
            for dx, dy, dz in offsets:
                n = (px + dx, py + dy, pz + dz)
                if n in remaining:
                    remaining.remove(n)
                    dq.append(n)

    return comps


def torus_hole_metrics(centers: torch.Tensor):
    xy_radius = torch.linalg.norm(centers[:, :2], dim=1)
    hole_mask = xy_radius < 0.35
    return int(hole_mask.sum().item()), float(hole_mask.float().mean().item())


def dumbbell_bridge_metrics(
    centers: torch.Tensor,
    tube_radius: float = 0.055,
    sphere_offset: float = 0.48,
    sphere_radius: float = 0.28,
    h: float = 0.01,
    band_voxels: float = 1.0,
):
    x = centers[:, 0]
    yz = torch.linalg.norm(centers[:, 1:3], dim=1)

    # The tube surface is only exposed where it lies outside both spheres.
    exposed_half_len = sphere_offset - float(max(sphere_radius ** 2 - tube_radius ** 2, 0.0) ** 0.5)

    x_margin = 2.0 * h
    x_min = -exposed_half_len + x_margin
    x_max = exposed_half_len - x_margin

    if x_min >= x_max:
        x_min = -sphere_offset * 0.25
        x_max = sphere_offset * 0.25

    tube_margin = max(2.0 * h, band_voxels * h)
    bridge_mask = (
        (x >= x_min)
        & (x <= x_max)
        & ((yz - tube_radius).abs() <= tube_margin)
    )

    center_fog_mask = (
        (x >= x_min)
        & (x <= x_max)
        & (yz < tube_radius * 0.55)
    )

    left_mask = x < -sphere_offset * 0.55
    right_mask = x > sphere_offset * 0.55

    bridge_bins = 24
    bridge_x = x[bridge_mask]

    if bridge_x.numel() > 0:
        bin_float = (bridge_x - x_min) / max(x_max - x_min, 1e-8) * bridge_bins
        bin_idx = torch.clamp(torch.floor(bin_float).to(torch.long), 0, bridge_bins - 1)

        bin_counts = torch.zeros(bridge_bins, device=centers.device, dtype=torch.long)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long))

        occupied = bin_counts > 0
        bins_occupied = int(occupied.sum().item())
        min_tokens_occupied = int(bin_counts[occupied].min().item()) if occupied.any() else 0
        min_tokens_any = int(bin_counts.min().item())
        bin_counts_list = [int(v) for v in bin_counts.detach().cpu().tolist()]
    else:
        bins_occupied = 0
        min_tokens_occupied = 0
        min_tokens_any = 0
        bin_counts_list = [0 for _ in range(bridge_bins)]

    return {
        "bridge_exposed_x_min": float(x_min),
        "bridge_exposed_x_max": float(x_max),
        "bridge_exposed_length": float(x_max - x_min),
        "bridge_shell_tokens": int(bridge_mask.sum().item()),
        "bridge_shell_ratio": float(bridge_mask.float().mean().item()),
        "bridge_center_fog_tokens": int(center_fog_mask.sum().item()),
        "bridge_center_fog_ratio": float(center_fog_mask.float().mean().item()),
        "left_lobe_tokens": int(left_mask.sum().item()),
        "right_lobe_tokens": int(right_mask.sum().item()),
        "bridge_x_bins_total": bridge_bins,
        "bridge_x_bins_occupied": bins_occupied,
        "bridge_x_coverage_ratio": float(bins_occupied / bridge_bins),
        "bridge_x_min_tokens_per_occupied_bin": min_tokens_occupied,
        "bridge_x_min_tokens_any_bin": min_tokens_any,
        "bridge_x_bin_counts": bin_counts_list,
    }


def line_surface_coverage_metrics(
    centers: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tube_radius: float,
    h: float,
    band_voxels: float,
    prefix: str,
    bins: int = 24,
):
    ba = b - a
    pa = centers - a[None, :]
    denom = torch.clamp(ba @ ba, min=1e-8)
    t = torch.clamp((pa @ ba) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ba[None, :]
    dist_to_axis = torch.linalg.norm(centers - closest, dim=1)

    margin = max(2.0 * h, band_voxels * h)
    mask = (t >= 0.0) & (t <= 1.0) & ((dist_to_axis - tube_radius).abs() <= margin)

    line_t = t[mask]
    if line_t.numel() > 0:
        bin_float = line_t * bins
        bin_idx = torch.clamp(torch.floor(bin_float).to(torch.long), 0, bins - 1)
        bin_counts = torch.zeros(bins, device=centers.device, dtype=torch.long)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long))

        occupied = bin_counts > 0
        occupied_count = int(occupied.sum().item())
        min_tokens_occupied = int(bin_counts[occupied].min().item()) if occupied.any() else 0
        min_tokens_any = int(bin_counts.min().item())
        bin_counts_list = [int(v) for v in bin_counts.detach().cpu().tolist()]
    else:
        occupied_count = 0
        min_tokens_occupied = 0
        min_tokens_any = 0
        bin_counts_list = [0 for _ in range(bins)]

    return {
        f"{prefix}_tokens": int(mask.sum().item()),
        f"{prefix}_bins_total": bins,
        f"{prefix}_bins_occupied": occupied_count,
        f"{prefix}_coverage_ratio": float(occupied_count / bins),
        f"{prefix}_min_tokens_per_occupied_bin": min_tokens_occupied,
        f"{prefix}_min_tokens_any_bin": min_tokens_any,
        f"{prefix}_bin_counts": bin_counts_list,
    }


def rim_coverage_metrics(
    centers: torch.Tensor,
    center_x: float,
    tube_radius: float,
    ring_radius: float,
    h: float,
    band_voxels: float,
    prefix: str,
    bins: int = 48,
):
    dx = centers[:, 0] - center_x
    dy = centers[:, 1]
    dz = centers[:, 2]

    radial = torch.sqrt(dx * dx + dy * dy)
    tube_sdf_abs = torch.sqrt((radial - ring_radius) ** 2 + dz * dz).sub(tube_radius).abs()

    margin = max(2.0 * h, band_voxels * h)
    mask = tube_sdf_abs <= margin

    theta = torch.atan2(dy[mask], dx[mask])
    theta = torch.where(theta < 0, theta + 2.0 * torch.pi, theta)

    if theta.numel() > 0:
        bin_float = theta / (2.0 * torch.pi) * bins
        bin_idx = torch.clamp(torch.floor(bin_float).to(torch.long), 0, bins - 1)
        bin_counts = torch.zeros(bins, device=centers.device, dtype=torch.long)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long))

        occupied = bin_counts > 0
        occupied_count = int(occupied.sum().item())
        min_tokens_occupied = int(bin_counts[occupied].min().item()) if occupied.any() else 0
        min_tokens_any = int(bin_counts.min().item())
        bin_counts_list = [int(v) for v in bin_counts.detach().cpu().tolist()]
    else:
        occupied_count = 0
        min_tokens_occupied = 0
        min_tokens_any = 0
        bin_counts_list = [0 for _ in range(bins)]

    # Tokens inside the lens hole near the frame plane.
    hole_mask = (
        (radial < ring_radius - 3.0 * tube_radius)
        & (dz.abs() < max(2.0 * h, tube_radius))
    )

    return {
        f"{prefix}_rim_tokens": int(mask.sum().item()),
        f"{prefix}_rim_bins_total": bins,
        f"{prefix}_rim_bins_occupied": occupied_count,
        f"{prefix}_rim_coverage_ratio": float(occupied_count / bins),
        f"{prefix}_rim_min_tokens_per_occupied_bin": min_tokens_occupied,
        f"{prefix}_rim_min_tokens_any_bin": min_tokens_any,
        f"{prefix}_rim_bin_counts": bin_counts_list,
        f"{prefix}_lens_hole_fog_tokens": int(hole_mask.sum().item()),
        f"{prefix}_lens_hole_fog_ratio": float(hole_mask.float().mean().item()),
    }


def glasses_metrics(
    centers: torch.Tensor,
    tube_radius: float = 0.020,
    h: float = 0.01,
    band_voxels: float = 1.0,
    rim_center_x: float = 0.32,
    rim_radius: float = 0.22,
    bridge_y: float = 0.045,
    arm_drop_z: float = -0.22,
):
    device = centers.device

    metrics = {}

    metrics.update(
        rim_coverage_metrics(
            centers,
            center_x=-rim_center_x,
            tube_radius=tube_radius,
            ring_radius=rim_radius,
            h=h,
            band_voxels=band_voxels,
            prefix="left",
        )
    )
    metrics.update(
        rim_coverage_metrics(
            centers,
            center_x=rim_center_x,
            tube_radius=tube_radius,
            ring_radius=rim_radius,
            h=h,
            band_voxels=band_voxels,
            prefix="right",
        )
    )

    bridge_a = torch.tensor([-rim_center_x + rim_radius * 0.86, bridge_y, 0.0], device=device)
    bridge_b = torch.tensor([rim_center_x - rim_radius * 0.86, bridge_y, 0.0], device=device)
    metrics.update(
        line_surface_coverage_metrics(
            centers,
            bridge_a,
            bridge_b,
            tube_radius=tube_radius,
            h=h,
            band_voxels=band_voxels,
            prefix="nose_bridge",
            bins=16,
        )
    )

    left_arm_a = torch.tensor([-rim_center_x - rim_radius * 0.96, 0.02, 0.0], device=device)
    left_arm_b = torch.tensor([-0.98, 0.06, arm_drop_z], device=device)
    right_arm_a = torch.tensor([rim_center_x + rim_radius * 0.96, 0.02, 0.0], device=device)
    right_arm_b = torch.tensor([0.98, 0.06, arm_drop_z], device=device)

    metrics.update(
        line_surface_coverage_metrics(
            centers,
            left_arm_a,
            left_arm_b,
            tube_radius=tube_radius,
            h=h,
            band_voxels=band_voxels,
            prefix="left_arm",
            bins=24,
        )
    )
    metrics.update(
        line_surface_coverage_metrics(
            centers,
            right_arm_a,
            right_arm_b,
            tube_radius=tube_radius,
            h=h,
            band_voxels=band_voxels,
            prefix="right_arm",
            bins=24,
        )
    )

    return metrics


def save_depth_png(depth: torch.Tensor, path: Path):
    d = depth.detach().cpu().numpy()
    valid = np.isfinite(d)
    img = np.zeros_like(d, dtype=np.uint8)

    if valid.any():
        vals = d[valid]
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        hi = max(hi, lo + 1e-6)
        norm = np.clip((d - lo) / (hi - lo), 0, 1)
        img[valid] = (255 * (1.0 - norm[valid])).astype(np.uint8)

    Image.fromarray(img).save(path)


def save_ply(path: Path, points: np.ndarray, scalar: np.ndarray | None = None):
    points = np.asarray(points, dtype=np.float32)

    if scalar is None:
        colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)
    else:
        s = np.asarray(scalar, dtype=np.float32)
        s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
        if abs(s_max - s_min) < 1e-8:
            s_norm = np.ones_like(s)
        else:
            s_norm = np.clip((s - s_min) / (s_max - s_min), 0, 1)
        colors = np.stack(
            [
                (255 * s_norm).astype(np.uint8),
                (255 * (1 - s_norm)).astype(np.uint8),
                np.full_like((255 * s_norm).astype(np.uint8), 80),
            ],
            axis=1,
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="torus", choices=["torus", "dumbbell", "glasses"])
    parser.add_argument("--res", type=int, default=96, help="SDF cell resolution")
    parser.add_argument("--coarse-res", type=int, default=64)
    parser.add_argument("--target-res", type=int, default=256)
    parser.add_argument("--bounds", type=float, default=1.2)
    parser.add_argument("--band-voxels", type=float, default=1.0)
    parser.add_argument("--tube-radius", type=float, default=0.055)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fx", type=float, default=520.0)
    parser.add_argument("--fy", type=float, default=520.0)
    parser.add_argument("--z-shift", type=float, default=3.0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--out", type=str, default="runs/test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("shape:", args.shape)
    if args.shape in {"dumbbell", "glasses"}:
        print("tube_radius:", args.tube_radius)
    print("extracting zero-crossings...")

    surface_points = extract_zero_crossings(
        args.shape,
        args.res,
        args.bounds,
        device,
        tube_radius=args.tube_radius,
    )
    nan_surface = torch.isnan(surface_points).sum().item()

    origin = torch.tensor([-args.bounds, -args.bounds, -args.bounds], device=device)
    h_target = (2.0 * args.bounds) / args.target_res
    factor = args.target_res // args.coarse_res

    if args.target_res % args.coarse_res != 0:
        raise ValueError("target-res must be divisible by coarse-res")

    print("building index path...")

    q_core = q_from_points(surface_points, origin, h_target, args.target_res)
    q_core = unique_q(q_core, args.target_res)

    q_coarse = unique_q(q_core // factor, args.coarse_res)

    q_final = children_band(
        shape=args.shape,
        coarse_q=q_coarse,
        coarse_res=args.coarse_res,
        target_res=args.target_res,
        origin=origin,
        h_target=h_target,
        band_voxels=args.band_voxels,
        device=device,
        tube_radius=args.tube_radius,
    )

    centers = centers_from_q(q_final, origin, h_target)

    print("building geometry path...")

    y_hat = project_to_surface(
        centers,
        args.shape,
        tube_radius=args.tube_radius,
        steps=5,
    )
    sdf_center = shape_sdf(centers, args.shape, tube_radius=args.tube_radius)
    abs_sdf_vox = sdf_center.abs() / h_target

    rho = torch.exp(-(sdf_center.abs() ** 2) / (2.0 * (args.band_voxels * h_target) ** 2))

    print("building conditioning path / z-buffer...")

    depth = scatter_zbuffer(
        surface_points=surface_points,
        width=args.image_size,
        height=args.image_size,
        fx=args.fx,
        fy=args.fy,
        z_shift=args.z_shift,
        splat_radius=args.splat_radius,
        device=device,
    )

    y_hat_view = y_hat + torch.tensor([0.0, 0.0, args.z_shift], device=device)
    u, v, z = project(y_hat_view, args.image_size, args.image_size, args.fx, args.fy)

    depth_at, inside = sample_depth_nearest(depth, u, v)
    eps = 2.0 * h_target
    tau = h_target
    visibility = torch.sigmoid((depth_at - z + eps) / tau)
    visibility = torch.where(torch.isfinite(depth_at) & inside, visibility, torch.zeros_like(visibility))

    print("validating dyadic parentage...")

    parent_hash = hash_q(q_final // factor, args.coarse_res)
    coarse_hash = hash_q(q_coarse, args.coarse_res)
    parent_ok = torch.isin(parent_hash, coarse_hash).all().item()

    print("counting connected components...")

    q_cpu = q_final.detach().cpu().numpy()
    components_26 = count_components_26(q_cpu)

    nan_count = (
        torch.isnan(surface_points).sum()
        + torch.isnan(centers).sum()
        + torch.isnan(y_hat).sum()
        + torch.isnan(rho).sum()
        + torch.isnan(visibility).sum()
    ).item()

    valid_depth_pixels = torch.isfinite(depth).sum().item()

    shape_metrics = {}
    if args.shape == "torus":
        hole_tokens, hole_ratio = torus_hole_metrics(centers)
        shape_metrics.update({
            "torus_hole_tokens_xy_radius_lt_0p35": hole_tokens,
            "torus_hole_ratio": hole_ratio,
        })

    if args.shape == "dumbbell":
        shape_metrics.update(
            dumbbell_bridge_metrics(
                centers,
                tube_radius=args.tube_radius,
                h=h_target,
                band_voxels=args.band_voxels,
            )
        )

    if args.shape == "glasses":
        shape_metrics.update(
            glasses_metrics(
                centers,
                tube_radius=args.tube_radius,
                h=h_target,
                band_voxels=args.band_voxels,
            )
        )

    metrics = {
        "device": str(device),
        "shape": args.shape,
        "tube_radius": args.tube_radius if args.shape in {"dumbbell", "glasses"} else None,
        "sdf_res": args.res,
        "coarse_res": args.coarse_res,
        "target_res": args.target_res,
        "bounds": args.bounds,
        "target_voxel_size": h_target,
        "surface_samples": int(surface_points.shape[0]),
        "core_tokens": int(q_core.shape[0]),
        "coarse_tokens": int(q_coarse.shape[0]),
        "final_tokens_children_band": int(q_final.shape[0]),
        **shape_metrics,
        "abs_sdf_vox_min": float(abs_sdf_vox.min().item()),
        "abs_sdf_vox_mean": float(abs_sdf_vox.mean().item()),
        "abs_sdf_vox_max": float(abs_sdf_vox.max().item()),
        "near_surface_ratio_1_vox": float((abs_sdf_vox <= 1.0).float().mean().item()),
        "near_surface_ratio_2_vox": float((abs_sdf_vox <= 2.0).float().mean().item()),
        "near_surface_ratio_3_vox": float((abs_sdf_vox <= 3.0).float().mean().item()),
        "components_26": components_26,
        "nan_count_total": int(nan_count),
        "nan_surface": int(nan_surface),
        "dyadic_parentage_ok": bool(parent_ok),
        "rho_min": float(rho.min().item()),
        "rho_mean": float(rho.mean().item()),
        "rho_max": float(rho.max().item()),
        "inside_projection_ratio": float(inside.float().mean().item()),
        "visibility_min": float(visibility.min().item()),
        "visibility_mean": float(visibility.mean().item()),
        "visibility_max": float(visibility.max().item()),
        "valid_depth_pixels": int(valid_depth_pixels),
    }

    print(json.dumps(metrics, indent=2))

    print("saving artifacts...")

    save_depth_png(depth, out_dir / "depth.png")

    save_ply(
        out_dir / "surface_zero_crossings.ply",
        surface_points.detach().cpu().numpy(),
    )

    save_ply(
        out_dir / "token_centers_rho.ply",
        centers.detach().cpu().numpy(),
        rho.detach().cpu().numpy(),
    )

    save_ply(
        out_dir / "y_hat_visibility.ply",
        y_hat.detach().cpu().numpy(),
        visibility.detach().cpu().numpy(),
    )

    np.savez_compressed(
        out_dir / "tensors.npz",
        q_final=q_final.detach().cpu().numpy(),
        centers=centers.detach().cpu().numpy(),
        y_hat=y_hat.detach().cpu().numpy(),
        rho=rho.detach().cpu().numpy(),
        visibility=visibility.detach().cpu().numpy(),
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPASS: artifacts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
