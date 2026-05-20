import argparse
import json
from pathlib import Path

import numpy as np


def unique_rows_int32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 array, got shape {arr.shape}")

    arr = np.ascontiguousarray(arr)
    view = arr.view([("", arr.dtype)] * arr.shape[1])
    unique = np.unique(view)
    return unique.view(arr.dtype).reshape(-1, arr.shape[1])


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_input_res(tensors_path: Path, explicit_res: int | None) -> int:
    if explicit_res is not None:
        return explicit_res

    metrics = load_metrics(tensors_path.parent / "metrics.json")
    if "target_res" not in metrics:
        raise ValueError(
            "Could not infer input resolution. Pass --input-res explicitly."
        )
    return int(metrics["target_res"])


def downsample_q(q: np.ndarray, input_res: int, output_res: int) -> np.ndarray:
    if output_res > input_res:
        raise ValueError(
            f"output_res={output_res} cannot exceed input_res={input_res} for this exporter."
        )
    if input_res % output_res != 0:
        raise ValueError(
            f"input_res={input_res} must be divisible by output_res={output_res}."
        )

    factor = input_res // output_res
    q_out = q // factor
    q_out = np.clip(q_out, 0, output_res - 1)
    return unique_rows_int32(q_out)


def make_pixal_coords(q_xyz: np.ndarray, batch_idx: int = 0) -> np.ndarray:
    q_xyz = np.asarray(q_xyz, dtype=np.int32)
    batch = np.full((q_xyz.shape[0], 1), batch_idx, dtype=np.int32)
    coords = np.concatenate([batch, q_xyz], axis=1)
    return unique_rows_int32(coords)


def summarize_coords(coords: np.ndarray, output_res: int) -> dict:
    xyz = coords[:, 1:4]
    return {
        "count": int(coords.shape[0]),
        "dtype": str(coords.dtype),
        "shape": list(coords.shape),
        "min_xyz": xyz.min(axis=0).astype(int).tolist() if coords.shape[0] else None,
        "max_xyz": xyz.max(axis=0).astype(int).tolist() if coords.shape[0] else None,
        "output_res": int(output_res),
        "in_bounds": bool(
            np.all(xyz >= 0) and np.all(xyz < output_res)
        ) if coords.shape[0] else True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tensors.npz from synthetic_router_test.py")
    parser.add_argument("--input-res", type=int, default=None, help="Resolution of q_final in tensors.npz. Usually inferred from metrics.json.")
    parser.add_argument("--out-dir", required=True, help="Output directory for Pixal coord npz files.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64], help="Pixal coordinate resolutions to export.")
    parser.add_argument("--batch-idx", type=int, default=0)
    args = parser.parse_args()

    tensors_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(tensors_path)
    if "q_final" not in data:
        raise KeyError(f"{tensors_path} does not contain q_final")

    q_final = np.asarray(data["q_final"], dtype=np.int32)
    if q_final.ndim != 2 or q_final.shape[1] != 3:
        raise ValueError(f"Expected q_final shape [N, 3], got {q_final.shape}")

    input_res = infer_input_res(tensors_path, args.input_res)

    all_metrics = {
        "source": str(tensors_path),
        "input_res": int(input_res),
        "input_q_count": int(q_final.shape[0]),
        "exports": {},
    }

    for output_res in args.resolutions:
        q_out = downsample_q(q_final, input_res=input_res, output_res=output_res)
        coords = make_pixal_coords(q_out, batch_idx=args.batch_idx)

        out_path = out_dir / f"pixal_coords_{output_res}.npz"
        np.savez_compressed(
            out_path,
            coords=coords,
            q_xyz=q_out,
            input_res=np.array(input_res, dtype=np.int32),
            output_res=np.array(output_res, dtype=np.int32),
            batch_idx=np.array(args.batch_idx, dtype=np.int32),
        )

        summary = summarize_coords(coords, output_res)
        summary["path"] = str(out_path)
        all_metrics["exports"][str(output_res)] = summary

    metrics_path = out_dir / "pixal_coord_export_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(json.dumps(all_metrics, indent=2))
    print(f"\nPASS: Pixal coordinate exports written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()