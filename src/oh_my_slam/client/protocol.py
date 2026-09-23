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

ServerStatus = Literal["loading", "ready", "degraded", "error", "stopping"]


class ModelStatus(BaseModel):
    name: str
    loaded: bool = False
    error: str | None = None
    detail: str | None = None


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
    detail: str | None = None


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
    want_normals: bool = True
    want_descriptor: bool = True


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
    normals_path: str | None = None  # float16 (H, W, 3), camera frame (OpenCV)
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


# --- segmentation (YOLOE -> SAM 3) ---------------------------------------------------------------


class SegmentRequest(BaseModel):
    image_path: str
    labels: list[str]
    max_side: int = 1024
    conf: float = 0.05  # raw pre-filter, well below any calibrated threshold
    iou: float = 0.6
    imgsz: int = 1024
    refine: bool = True
    max_concepts: int = 8


class Instance(BaseModel):
    label: str
    score: float  # raw model score (calibrated by the client)
    source: Literal["yoloe", "sam3"]
    box_xyxy: list[float]  # on the mask grid
    mask: dict[str, object]  # COCO RLE {"size": [h, w], "counts": str}


class SegmentResponse(BaseModel):
    width: int
    height: int
    mode: Literal["full", "degraded"]
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
