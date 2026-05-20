import argparse
import os
import sys
import types
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(REPO_ROOT / "autotune_cache.json"),
)

# Coord-dump-only script: avoid mesh decode dependencies such as cumesh.
representations_stub = types.ModuleType("pixal3d.representations")
representations_stub.Mesh = object
representations_stub.MeshWithVoxel = object
sys.modules.setdefault("pixal3d.representations", representations_stub)

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.pipelines import pixal3d_image_to_3d as pixal_pipeline_module


IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
}


def build_image_cond_model(config):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def coord_summary(name, coords):
    xyz = coords[:, 1:]
    if coords.shape[0] == 0:
        print(f"{name}: tokens=0")
        return
    print(
        f"{name}: tokens={coords.shape[0]}, "
        f"min={xyz.min(dim=0).values.tolist()}, max={xyz.max(dim=0).values.tolist()}"
    )


def save_coords(path, coords):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords_np = coords.detach().cpu().to(torch.int32).numpy()
    import numpy as np

    np.savez(path, coords=coords_np)
    print(f"Wrote {path}")


class DisabledRembg:
    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        return self

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise RuntimeError(
            "RMBG is disabled for coordinate dumping. Use --load-rembg if preprocessing is needed."
        )


def load_pipeline(args, device):
    if args.preprocess_image and not args.load_rembg:
        raise ValueError("--preprocess-image requires --load-rembg because background removal may be needed.")

    print(f"Loading Pixal3D pipeline from {args.model_path}...")
    had_model_names_to_load = hasattr(Pixal3DImageTo3DPipeline, "model_names_to_load")
    original_model_names_to_load = getattr(Pixal3DImageTo3DPipeline, "model_names_to_load", None)
    original_rembg_birefnet = pixal_pipeline_module.rembg.BiRefNet
    if args.skip_hr:
        print("--skip-hr enabled: loading sparse-structure models only.")
        Pixal3DImageTo3DPipeline.model_names_to_load = {
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
        }
    if not args.load_rembg:
        print("--load-rembg not supplied: using disabled RMBG placeholder.")
        pixal_pipeline_module.rembg.BiRefNet = DisabledRembg
    try:
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model_path)
    finally:
        pixal_pipeline_module.rembg.BiRefNet = original_rembg_birefnet
        if had_model_names_to_load:
            Pixal3DImageTo3DPipeline.model_names_to_load = original_model_names_to_load
        else:
            delattr(Pixal3DImageTo3DPipeline, "model_names_to_load")

    pipeline.low_vram = args.low_vram
    pipeline._device = device

    print("Building image conditioning model for sparse structure...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])

    if not args.skip_hr:
        print("Building image conditioning model for LR shape...")
        pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])

    if args.low_vram:
        if pipeline.rembg_model is not None:
            pipeline.rembg_model.cpu()
    else:
        pipeline.to(device)
        pipeline.image_cond_model_ss.to(device)
        if pipeline.image_cond_model_shape_512 is not None:
            pipeline.image_cond_model_shape_512.to(device)

    return pipeline


@torch.no_grad()
def dump_native_coords(args):
    device = torch.device(args.device)
    pipeline = load_pipeline(args, device)
    image = Image.open(args.image).convert("RGB")
    if args.preprocess_image:
        image = pipeline.preprocess_image(image)

    torch.manual_seed(args.seed)

    cond_ss = pipeline.get_proj_cond_ss(
        [image],
        camera_angle_x=args.camera_angle_x,
        distance=args.distance,
        mesh_scale=args.mesh_scale,
    )
    coords = pipeline.sample_sparse_structure(
        cond_ss,
        resolution=32,
        num_samples=1,
        sampler_params={},
    )
    del cond_ss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    coord_summary("native_coords_32", coords)
    save_coords(args.out_dir / "native_coords_32.npz", coords)

    if args.skip_hr:
        return

    try:
        cond_shape_lr = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_512,
            [image],
            coords,
            camera_angle_x=args.camera_angle_x,
            distance=args.distance,
            mesh_scale=args.mesh_scale,
        )
        lr_slat = pipeline.sample_shape_slat(
            cond_shape_lr,
            pipeline.models["shape_slat_flow_model_512"],
            coords,
            sampler_params={},
        )
        del cond_shape_lr
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if pipeline.low_vram:
            pipeline.models["shape_slat_decoder"].to(device)
            pipeline.models["shape_slat_decoder"].low_vram = True
        hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models["shape_slat_decoder"].cpu()
            pipeline.models["shape_slat_decoder"].low_vram = False

        quant_coords = torch.cat(
            [
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / 512 * (64 - 1)).round().int(),
            ],
            dim=1,
        )
        hr_coords_unique = quant_coords.unique(dim=0)
        coord_summary("native_coords_64", hr_coords_unique)
        save_coords(args.out_dir / "native_coords_64.npz", hr_coords_unique)
    except Exception as exc:
        print(f"Skipped native_coords_64 due to error: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dump native Pixal sparse coordinate supports.")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face repo for Pixal3D weights.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for native coordinate .npz files.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-angle-x", type=float, default=0.8575560450553894)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--preprocess-image", action="store_true")
    parser.add_argument("--load-rembg", action="store_true", help="Load the RMBG background-removal model.")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-hr", action="store_true", help="Only dump native sparse structure coords at 32^3.")
    return parser.parse_args()


def main():
    dump_native_coords(parse_args())


if __name__ == "__main__":
    main()
