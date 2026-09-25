# Third-party components and licences

oh-my-slam is for personal and research use (`high_level_spec.md` §4). Copyleft components
(AGPL-3.0, GPL, LGPL, MPL) and non-commercial or research-only licences are therefore acceptable
here. Anyone using the project commercially, or offering it as a network service, would have to
re-check every row below. These matter most:

* the AGPL-3.0 components: Ultralytics and its CLIP fork
* the Apple research-only MobileCLIP2 encoder
* the GPL components: plyfile, the codecs bundled in the PyAV, OpenCV and pillow-heif wheels, and
  the Homebrew COLMAP build

**Redistributed in this repository:**

* three.js (vendored browser library)
* the ASAM OpenLABEL JSON schema
* 10 of the 19 palette colours and the 11 viridis samples in `segmentation/colors.py`
* the label list in `segmentation/data/default_labels.txt`

Every other component is installed by `uv sync`, by Homebrew, or on the first server start.

**Sources.** Versions are those locked in `uv.lock`, or installed, on 2026-09-25. Licences were
checked online on 2026-09-25 through:

* PyPI JSON (`https://pypi.org/pypi/<pkg>/json`)
* the Hugging Face model API
* GitHub `LICENSE` files, at the pinned commit for the git dependencies
* `formulae.brew.sh`

The contents of binary wheels were read from the installed files. **UNVERIFIED** marks a licence
that could not be confirmed.

## Models and weights (loaded by the inference server)

| Component | Source | Used for | Licence |
|---|---|---|---|
| MoGe-2 ViT-L normal | HF `Ruicheng/moge-2-vitl-normal` | Metric depth, point map, intrinsics; its encoder's class token is the retrieval descriptor | MIT (model card) |
| DINOv2 ViT-L backbone | Part of the MoGe-2 checkpoint; code in `moge/model/dinov2` | MoGe encoder | Apache-2.0 (Meta AI; MoGe README and `facebookresearch/dinov2`) |
| GeoCalib pinhole weights | GitHub release `cvg/GeoCalib` v1.0 (`geocalib-pinhole.tar`, torch hub cache) | Gravity direction | CC-BY-4.0 (weights; GeoCalib README) |
| YOLOE-26x-seg | `yoloe-26x-seg.pt`, downloaded by Ultralytics | Open-vocabulary instance segmentation | AGPL-3.0, or an Ultralytics Enterprise licence |
| MobileCLIP2-B text encoder | `mobileclip2_b.ts`, the Ultralytics TorchScript export of Apple's MobileCLIP2-B | YOLOE text prompts | Apple Machine Learning Research Model License (`apple-amlr`, HF `apple/MobileCLIP2-B`): research purposes only, no commercial use |
| MapAnything, Apache variant | HF `facebook/map-anything-apache` | Metric multi-view poses (mapping fallback) | Apache-2.0 (model card). The default `facebook/map-anything` checkpoint is CC-BY-NC-4.0 and is not used unless `OH_MY_SLAM_MAPANYTHING_REPO` selects it. |
| ALIKED (`ALIKED_N16ROT`) + LightGlue ONNX models | Fetched by COLMAP; used only with `OH_MY_SLAM_FEATURES=aliked` | Learned features and matching | Upstream: ALIKED BSD-3-Clause (`Shiaoming/ALIKED`), LightGlue Apache-2.0 (`cvg/LightGlue`). The licence of COLMAP's ONNX exports is **UNVERIFIED**. |

## Python runtime dependencies (`pyproject.toml`)

| Package | Version | Licence | Notes |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | |
| scipy | 1.18.1 | BSD-3-Clause | |
| pillow | 12.3.0 | MIT-CMU | |
| pillow-heif | 1.8.0 | BSD-3-Clause source; **binary wheel GPL-2.0** | The wheel bundles libheif (LGPL-3.0), libde265 (LGPL-3.0) and x265 (GPL-2.0), per its `LICENSES_bundled.txt`. |
| av (PyAV) | 18.1.0 | BSD-3-Clause | The macOS wheel bundles FFmpeg 8 with libx264 and libx265 (GPL-2.0-or-later), so the binary is effectively GPL. |
| opencv-python-headless | 4.14.0.94 | Apache-2.0 (OpenCV) | Wheels ship FFmpeg under LGPL-2.1 (PyPI description). The macOS arm64 wheel also bundles libx264 and libx265 (GPL-2.0-or-later). |
| jsonschema | 4.26.0 | MIT | |
| psutil | 7.2.2 | BSD-3-Clause | |
| open3d | 0.20.0 | MIT | TSDF fusion of the map cloud |
| pycolmap | 4.2.0 | BSD-3-Clause | Mapping, triangulation, bundle adjustment; the wheel bundles only libomp |
| fastapi | 0.141.1 | MIT | |
| uvicorn | 0.53.0 | BSD-3-Clause | |
| httpx | 0.28.1 | BSD-3-Clause | |
| pydantic | 2.13.5 | MIT | |
| torch | 2.14.0 | BSD-3-Clause (PyTorch). The PyPI expression also lists Apache-2.0, Apache-2.0 WITH LLVM-exception and BSD-2-Clause for bundled parts. | Server process only |
| torchvision | 0.29.0 | BSD-3-Clause | |
| transformers | 5.17.0 | Apache-2.0 | |
| ultralytics | 8.4.159 | AGPL-3.0 | YOLOE runtime |
| clip (`ultralytics/CLIP` @ a13192f) | 1.0 | AGPL-3.0 (`LICENSE` at the pinned commit) | A fork of OpenAI CLIP, which is MIT |
| moge (`microsoft/MoGe` @ 925b8ed) | 2.0.0 | MIT; `moge/model/dinov2` is Apache-2.0 | The last pre-V3 commit, so MoGe-2 |
| utils3d (`EasternJournalist/utils3d` @ 3fab839) | 1.3 | MIT | MoGe dependency |
| pipeline (`EasternJournalist/pipeline` @ 866f059) | 1.0.0 | MIT | MoGe dependency |
| geocalib (`cvg/GeoCalib` @ 97b8968) | 1.0 | Apache-2.0 (code) | Weights: see above |
| mapanything (`facebookresearch/map-anything` @ 3d10cf7) | 1.1.4 | Apache-2.0 (code) | Weights: see above |

### Notable transitive dependencies

Every PyPI package in `uv.lock` (164 of them) was checked on PyPI. Apart from the rows below,
each one reports a permissive licence (MIT, BSD, Apache-2.0, ISC, PSF or similar). The one
exception is `mypy-extensions` (a mypy dependency), whose PyPI entry reports no licence. These
rows are listed because they are copyleft or otherwise notable:

| Package | Version | Licence | Pulled in by |
|---|---|---|---|
| plyfile | 1.1.5 | GPL-3.0-or-later | MapAnything |
| ultralytics-thop | 2.1.6 | AGPL-3.0 | ultralytics |
| ultralytics-platform | 0.1.54 | AGPL-3.0-only | ultralytics |
| certifi | 2026.7.22 | MPL-2.0 | httpx, requests |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | transformers, huggingface-hub, clip, MapAnything |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | MapAnything |
| pathspec | 1.1.1 | MPL-2.0 | mypy (dev) |
| trimesh | 5.1.0 | MIT | MoGe, MapAnything (no longer a direct dependency) |
| uniception | 0.1.7 | BSD-3-Clause | MapAnything |

## External tools (not in `.venv`)

| Component | Version | Used for | Licence |
|---|---|---|---|
| COLMAP (Homebrew `colmap`) | 4.2.0 | Feature extraction and matching CLI (SIFT by default; ALIKED/LightGlue through ONNX Runtime + CoreML); GLOMAP is part of COLMAP 4.2 | BSD-3-Clause (COLMAP, GLOMAP). COLMAP's licence notes that its dependencies may change the licence of the built binary. The Homebrew formula depends on CGAL 6.2.1 (GPL-3.0-or-later), Qt 6 `qtbase` (LGPL-3.0 / GPL), SuiteSparse (mixed, including GPL and LGPL) and ONNX Runtime 1.30.0 (MIT), so the Homebrew binary should be treated as GPL-3.0-or-later. |
| Microsoft Edge or Google Chrome | system-installed | Playwright browser tests and the evaluator's page timing | Proprietary; not bundled |
| uv | 0.9.18 (installed) | Environment and dependency manager | MIT OR Apache-2.0 |

## Browser libraries (vendored in `src/oh_my_slam/viewer/static/vendor/`)

| Component | Version | Licence |
|---|---|---|
| three.js: `three.module.js`, `three.core.js` and the addons `OrbitControls`, `CSS2DRenderer`, `LineSegments2`, `LineMaterial`, `LineSegmentsGeometry` | 0.186.0 (r186), from the npm tarball recorded in `VERSIONS.txt` | MIT (`vendor/three/LICENSE`) |

## Data, specifications and design assets

| Component | Where | Licence |
|---|---|---|
| ASAM OpenLABEL 1.0.0 JSON schema | Vendored as `src/oh_my_slam/schema/openlabel_json_schema.json`, sha256-pinned; published at `https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json` | **UNVERIFIED**. ASAM publishes the schema openly and the standard is free of charge with registration. The schema repository's licence text (`code.asam.net`) could not be read. |
| LVIS vocabulary | Label names curated from LVIS and COCO nouns in `segmentation/data/default_labels.txt`; the ontology URI `https://www.lvisdataset.org/` | LVIS annotations CC-BY-4.0; COCO annotations CC-BY-4.0 (COCO Consortium). No images are used. |
| Distinct-colour palette | Sasha Trubetskoy, "List of 20 Simple, Distinct Colors" (sashamaps.net); 10 colours used (the other 9 palette colours are this project's own) | No licence is stated on the source page, which offers the list for free download. The colours are credited in `segmentation/colors.py`. |
| Viridis colour map | 11 samples used for `color=height` | CC0-1.0 (mpl-colormaps by Nathaniel Smith and Stéfan van der Walt) |

## Development tools (`[dependency-groups] dev` and build)

| Package | Version | Licence |
|---|---|---|
| pytest | 9.1.1 | MIT |
| pytest-timeout | 2.4.0 | MIT |
| pytest-cov | 7.1.0 | MIT |
| coverage | 7.16.1 | Apache-2.0 |
| ruff | 0.16.8 | MIT |
| mypy | 2.3.1 | MIT |
| import-linter | 2.15 | BSD-2-Clause |
| playwright | 1.63.0 | Apache-2.0 |
| hatchling (build backend, `>=1.25`, not locked) | — | MIT |

## Removed since the previous revision

* **OpenMVS.** The surface mesh and texturing are gone, and the map's geometry is a point cloud.
* **trimesh as a direct dependency.** It remains transitive, through MoGe and MapAnything.
* **The combined "Ultralytics YOLOE-26x-seg + MobileCLIP2-B text encoder: AGPL-3.0" entry.**
  YOLOE and the MobileCLIP2-B encoder now have separate rows. The encoder is under Apple's
  research-only licence.
* **The catch-all "PyTorch, torchvision, …" row.** Each package now has its own row.
