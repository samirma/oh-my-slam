# Third-party components and licences

oh-my-slam depends on the following components. None is redistributed in this repository except
the vendored browser libraries under `src/oh_my_slam/viewer/static/vendor/` (licences included
there) and the vendored ASAM OpenLABEL JSON schema.

| Component | Use | Licence | Notes |
|---|---|---|---|
| MoGe-2 (`microsoft/MoGe` @ 925b8ed, weights `Ruicheng/moge-2-vitl-normal`) | Single-image metric geometry, intrinsics, normals | MIT (code and weights) | Pinned pre-V3 commit (gate G1) |
| GeoCalib (`cvg/GeoCalib` @ 97b8968) | Gravity and field-of-view cross-check | Apache-2.0 code, CC-BY-4.0 weights | Weights downloaded from the GitHub release on first start |
| Ultralytics YOLOE-26x-seg + MobileCLIP2-B text encoder | Open-vocabulary instance segmentation | AGPL-3.0 | Network use of AGPL software may impose obligations |
| MapAnything (`facebookresearch/map-anything` @ 3d10cf7, weights `facebook/map-anything-apache`) | Metric multi-view fallback for poses | Apache-2.0 code and weights | The Apache-licensed checkpoint is used instead of the CC-BY-NC one |
| COLMAP 4.2 / GLOMAP (Homebrew CLI, PyPI `pycolmap`) | Features, matching, SfM, bundle adjustment | BSD-3-Clause | ALIKED/LightGlue ONNX models downloaded by COLMAP |
| OpenMVS 2.4.0 (prebuilt macOS arm64) | Mesh texturing (`TextureMesh`) | AGPL-3.0 | Optional; Open3D texturing is the fallback |
| Open3D 0.20 | TSDF fusion, meshing, UV atlas, texture projection | MIT | |
| three.js r186, lil-gui | Browser viewer | MIT | Vendored |
| ASAM OpenLABEL 1.0.0 JSON schema | Scene description validation | ASAM (schema published for implementers) | Vendored, sha256-pinned |
| PyTorch, torchvision, transformers, FastAPI, uvicorn, httpx, NumPy, SciPy, Pillow, pillow-heif, PyAV, OpenCV, trimesh | Runtime | BSD / MIT / Apache-2.0 / LGPL (FFmpeg in PyAV/OpenCV wheels) | |
