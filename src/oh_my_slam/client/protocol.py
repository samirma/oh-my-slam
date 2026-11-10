"""Request/response models shared by the inference client and server.

Requests carry file paths (client and server run as the same user on the same machine); large
arrays come back as ``.npy`` files the server writes into the caller's ``out_dir``. Masks come back
as COCO RLE on the same pixel grid as the geometry output for the same ``max_side``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ROUTE_HEALTH = "/health"
ROUTE_GEOMETRY = "/v1/geometry"
ROUTE_GRAVITY = "/v1/gravity"
ROUTE_SEGMENT = "/v1/segment"
ROUTE_MULTIVIEW = "/v1/multiview"
# a request's priority (inference.md, single worker): the device queue serves ``interactive``
# requests (locate, dry runs, frame analysis) before the others (a job's fusion)
PRIORITY_HEADER = "x-oms-priority"
INTERACTIVE = "interactive"
# ``ErrorBody.error`` (HTTP 500) of a request the device ran out of memory for: a smaller one may
# fit
OUT_OF_MEMORY = "out_of_memory"

# ``degraded``: the models are loaded, but the last job ran out of device memory or the memory
# still in use after it is near the cap (``Health.memory``); requests are accepted and may fail
# until a restart frees the memory.
ServerStatus = Literal["loading", "ready", "degraded", "error", "stopping"]

# Long side at which the server reads a request's image (``core.images.load_rgb``): the request's
# ``max_side`` for geometry and detection, these for gravity and multi-view (the server's model
# adapters). The client sends images already at that size (``client.images``).
GRAVITY_SIDE = 640
MULTIVIEW_SIDE = 1024


class ModelStatus(BaseModel):
    name: str
    loaded: bool = False
    error: str | None = None
    detail: str | None = None


class DeviceMemory(BaseModel):
    """The device's memory after the last job (taken on the GPU thread), in GiB: in use by
    tensors (what the cap is checked against), held from the driver (in use, cached and the
    frameworks' own), and the per-process cap."""

    allocated_gib: float
    driver_gib: float
    limit_gib: float


class Health(BaseModel):
    status: ServerStatus
    models: dict[str, ModelStatus] = Field(default_factory=dict)
    device: str = "cpu"
    precision: str = "fp32"
    versions: dict[str, str] = Field(default_factory=dict)
    pid: int = 0
    protocol: int = 1
    queue_depth: int = 0
    queue_limit: int = 8
    uptime_s: float = 0.0
    memory: DeviceMemory | None = None  # None until a job has run on MPS
    last_oom: str | None = None  # the last out-of-memory error since the start

    def failures(self) -> str:
        """The models that failed to load, as ``name: error`` joined by ``; `` (empty if none)."""
        return "; ".join(f"{m.name}: {m.error}" for m in self.models.values() if m.error)

    def memory_note(self) -> str:
        """Why the server is ``degraded``: the device memory in use against the cap, and the last
        out-of-memory error ("no detail" if neither is known)."""
        notes = [] if self.memory is None else [
            f"{self.memory.allocated_gib:.2f} of {self.memory.limit_gib:.2f} GiB in use"]
        notes += [] if self.last_oom is None else [f"last error: {self.last_oom}"]
        return "; ".join(notes) or "no detail"


class Timings(BaseModel):
    queue_s: float = 0.0
    compute_s: float = 0.0


# --- geometry (MoGe-2) ---------------------------------------------------------------------------


class GeometryRequest(BaseModel):
    image_path: str
    out_dir: str
    max_side: int = 1024
    fov_x_deg: float | None = None
    num_tokens: int = 2500
    want_descriptor: bool = True
    # Keep the network output: this image will be asked again with another ``fov_x_deg``. A
    # request for the same pixels and tokens without it re-solves a kept output (only MoGe's
    # post-processing depends on the focal length) instead of running the network again.
    keep_forward: bool = False


class PixelIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class GeometryResponse(BaseModel):
    width: int
    height: int
    orig_width: int
    orig_height: int
    intrinsics: PixelIntrinsics  # on the output grid (width x height)
    fov_x_deg: float
    depth_path: str  # float32 (H, W), metres, 0 where invalid
    mask_path: str  # uint8 (H, W), 1 = valid
    descriptor: list[float] | None = None  # L2-normalised global image descriptor
    timings: Timings = Field(default_factory=Timings)


# --- gravity (GeoCalib) --------------------------------------------------------------------------


class GravityRequest(BaseModel):
    image_path: str
    focal_px: float | None = None  # prior, in pixels of the upright full-resolution image


class GravityResponse(BaseModel):
    up_cam: list[float]  # unit "up" direction in OpenCV camera coordinates
    roll_deg: float
    pitch_deg: float
    roll_unc_deg: float
    pitch_unc_deg: float
    focal_px: float  # estimated focal, full-resolution pixels
    focal_unc_px: float
    vfov_deg: float
    timings: Timings = Field(default_factory=Timings)


# --- segmentation (YOLOE) ------------------------------------------------------------------------------


class SegmentRequest(BaseModel):
    image_path: str
    labels: list[str]
    max_side: int = 1024
    conf: float = 0.05  # score pre-filter
    iou: float = 0.6
    imgsz: int = 1024
    # every label a candidate scores above ``conf``, as instances of their own (the same box and
    # mask), instead of its best label alone (class-agnostic selection)
    per_label: bool = False


class Instance(BaseModel):
    label: str
    score: float  # model score
    source: Literal["yoloe"]
    box_xyxy: list[float]  # on the mask grid
    mask: dict[str, object]  # COCO RLE {"size": [h, w], "counts": str}


class SegmentResponse(BaseModel):
    width: int
    height: int
    instances: list[Instance]
    timings: Timings = Field(default_factory=Timings)


# --- multi-view (MapAnything) --------------------------------------------------------------------


class MultiviewRequest(BaseModel):
    image_paths: list[str]
    out_dir: str
    intrinsics: list[list[list[float]] | None] | None = None  # per view 3x3, full-res pixels
    poses: list[list[list[float]] | None] | None = None  # per view 4x4 cam-to-world (OpenCV)
    poses_metric: bool = True
    resolution: int = 518
    want_depth: bool = True


class MultiviewView(BaseModel):
    pose: list[list[float]]  # 4x4 cam-to-world (OpenCV axes), metres
    intrinsics: list[list[float]]  # 3x3 on the processed grid
    width: int
    height: int
    depth_path: str | None = None  # float32 (H, W) z-depth on the processed grid
    conf_path: str | None = None


class MultiviewResponse(BaseModel):
    views: list[MultiviewView]
    metric_scale: float = 1.0
    timings: Timings = Field(default_factory=Timings)


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
