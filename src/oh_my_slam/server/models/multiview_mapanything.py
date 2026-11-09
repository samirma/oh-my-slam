"""MapAnything (Apache-2.0 checkpoint): metric multi-view poses and depth, optionally conditioned
on known intrinsics and cam-to-world poses for some views (used to anchor chunks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from oh_my_slam.client.protocol import MultiviewRequest
from oh_my_slam.core.atomic import atomic_save_npy
from oh_my_slam.core.images import load_rgb, upright_size
from oh_my_slam.server.models.hub import cached_file

REPO_ID = "facebook/map-anything-apache"  # the Apache-2.0 checkpoint (THIRD_PARTY_LICENSES.md)
# The checkpoint's revision: read from the Hugging Face cache without asking the Hub, and never
# replaced by a newer upload (one start re-downloaded the 4.5 GB checkpoint for 249 s).
REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
_LOAD_SIDE = 1024


def build_on_meta(cls: Any, config: dict[str, Any], weights: str, device: str) -> Any:
    """``cls`` (MapAnything) with ``config``'s arguments and the checkpoint ``weights``
    (safetensors), on ``device``: built on the ``meta`` device, so that no weight is allocated
    and randomly initialised before the checkpoint replaces it (about 16 s for the ViT-G), then
    given the checkpoint's tensors (``load_state_dict(assign=True)``). The weights are those
    ``from_pretrained`` loads, bit for bit. None when a tensor of the model is not in the
    checkpoint (it would stay on ``meta``): the caller builds it the usual way."""
    import inspect

    import torch
    from safetensors.torch import load_file

    params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in config.items() if k in params}
    enc = kwargs.get("encoder_config")
    if isinstance(enc, dict) and enc.get("encoder_str") == "dinov2":
        # as MapAnything._from_pretrained: the checkpoint holds the encoder's weights
        kwargs["encoder_config"] = {**enc, "torch_hub_pretrained": False}
    with torch.device("meta"):
        model = cls(**kwargs)
    state = load_file(weights, device="cpu")
    dtypes = {k: t.dtype for k, t in model.state_dict().items()}
    # as load_state_dict copies them: in the dtype of the model's own tensors
    state = {k: v if k not in dtypes or v.dtype == dtypes[k] else v.to(dtypes[k])
             for k, v in state.items()}
    model.load_state_dict(state, strict=False, assign=True)
    del state
    tensors = [*model.parameters(), *model.buffers()]
    if any(t.is_meta for t in tensors):
        return None
    return model.to(torch.device(device)).eval()


class MapAnythingMultiview:
    key = "multiview"
    name = "MapAnything (apache)"

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import json

        import torch
        from mapanything.models import MapAnything

        self.device = device
        config = cached_file(REPO_ID, "config.json", REVISION)
        weights = cached_file(REPO_ID, "model.safetensors", REVISION)
        model = None
        if config is not None and weights is not None:
            model = build_on_meta(MapAnything, json.loads(Path(config).read_text()), weights,
                                  device)
        if model is None:  # not cached yet (downloads it), or a tensor not in the checkpoint
            model = MapAnything.from_pretrained(REPO_ID, revision=REVISION)
            model = model.to(torch.device(device)).eval()
        self.model = model

    def warmup(self) -> None:
        pass  # a forward pass needs >= 1 real view; the first request warms up

    def run(self, req: MultiviewRequest) -> dict[str, Any]:
        import torch
        from mapanything.utils.image import preprocess_inputs

        views = []
        n = len(req.image_paths)
        for i, p in enumerate(req.image_paths):
            path = Path(p)
            rgb = load_rgb(path, max_side=_LOAD_SIDE)
            ow, oh = upright_size(path)
            view: dict[str, Any] = {"img": torch.from_numpy(rgb)}
            K = None if req.intrinsics is None else req.intrinsics[i]
            if K is not None:
                Kn = np.asarray(K, dtype=np.float32).copy()
                sx, sy = rgb.shape[1] / ow, rgb.shape[0] / oh
                Kn[0] *= sx
                Kn[1] *= sy
                view["intrinsics"] = torch.from_numpy(Kn)
            T = None if req.poses is None else req.poses[i]
            if T is not None:
                view["camera_poses"] = torch.from_numpy(np.asarray(T, dtype=np.float32))
                view["is_metric_scale"] = torch.tensor([bool(req.poses_metric)])
            views.append(view)
        processed = preprocess_inputs(views, resize_mode="longest_side", size=req.resolution)
        with torch.inference_mode():
            preds = self.model.infer(
                processed,
                memory_efficient_inference=n > 8,
                use_amp=self.device != "cpu",
                amp_dtype="fp16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )
        out_dir = Path(req.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result_views = []
        scale = 1.0
        for i, pred in enumerate(preds):
            pose = pred["camera_poses"][0].detach().float().cpu().numpy()
            K = pred["intrinsics"][0].detach().float().cpu().numpy()
            depth = pred["depth_z"][0, ..., 0].detach().float().cpu().numpy()
            mask = pred["mask"][0, ..., 0].detach().cpu().numpy().astype(bool)
            conf = pred["conf"][0].detach().float().cpu().numpy()
            depth = np.where(mask & np.isfinite(depth), depth, 0.0).astype(np.float32)
            entry: dict[str, Any] = {
                "pose": pose.tolist(),
                "intrinsics": K.tolist(),
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
            }
            if req.want_depth:
                atomic_save_npy(out_dir / f"mv_depth_{i:04d}.npy", depth)
                atomic_save_npy(out_dir / f"mv_conf_{i:04d}.npy", conf.astype(np.float16))
                entry["depth_path"] = str(out_dir / f"mv_depth_{i:04d}.npy")
                entry["conf_path"] = str(out_dir / f"mv_conf_{i:04d}.npy")
            result_views.append(entry)
            if "metric_scaling_factor" in pred:
                scale = float(pred["metric_scaling_factor"].reshape(-1)[0])
        if self.device == "mps":
            torch.mps.empty_cache()
        return {"views": result_views, "metric_scale": scale, "timings": {}}
