# Third-party components and licences

oh-my-slam is for personal and research use (`high_level_spec.md` §4). Copyleft components
(AGPL-3.0, GPL, LGPL, MPL) and non-commercial or research-only licences are therefore acceptable
here. Anyone using the project commercially, or offering it as a network service, would have to
re-check every row below. These matter most:

* the AGPL-3.0 components: Ultralytics and its CLIP fork
* the Apple research-only MobileCLIP2 encoder
* the GPL components: plyfile, the codecs bundled in the PyAV, OpenCV and pillow-heif wheels, and
  the Homebrew COLMAP build (and, for one unit test only, the Homebrew FFmpeg CLI)
* the vendored ASAM OpenLABEL JSON schema, whose licensing terms (ASAM e.V.) let neither research
  institutions nor other non-members pass it on (see "Data, specifications and design assets")

**Redistributed in this repository:**

* three.js (vendored browser library)
* axe-core (vendored for the browser tests only; MPL-2.0)
* the ASAM OpenLABEL JSON schema
* 10 of the 19 palette colours and the 11 viridis samples in `segmentation/colors.py`
* the label list in `segmentation/data/default_labels.txt`

Every other component is installed by `uv sync`, by Homebrew, or on the first server start.

**Sources.** Versions are those locked in `uv.lock`, or installed, on 2026-10-07. Licences were
checked online on 2026-09-25, and again on 2026-10-04 for the versions the regenerated lock
changed, through the sources below. The table of every runtime package was generated on
2026-10-07, and that of every development package on 2026-10-08, from `uv.lock` and the
installed metadata in `.venv` (`importlib.metadata`). `tests/unit/test_third_party_licenses.py`
fails when a package of `uv.lock` has no row here.

* PyPI JSON (`https://pypi.org/pypi/<pkg>/json`)
* the Hugging Face model API
* GitHub `LICENSE` files, at the pinned commit for the git dependencies
* `formulae.brew.sh`

The contents of binary wheels were read from the installed files. Every licence below has been
confirmed; none is left unverified.

## Models and weights (loaded by the inference server)

| Component | Source | Used for | Licence |
|---|---|---|---|
| MoGe-2 ViT-L normal | HF `Ruicheng/moge-2-vitl-normal` | Metric depth, point map, intrinsics; its encoder's class token is the retrieval descriptor | MIT (model card) |
| DINOv2 ViT-L backbone | Part of the MoGe-2 checkpoint; code in `moge/model/dinov2` | MoGe encoder | Apache-2.0 (Meta AI; MoGe README and `facebookresearch/dinov2`) |
| GeoCalib pinhole weights | GitHub release `cvg/GeoCalib` v1.0 (`geocalib-pinhole.tar`, torch hub cache) | Gravity direction | CC-BY-4.0 (weights; GeoCalib README) |
| YOLOE-26x-seg | `yoloe-26x-seg.pt`, downloaded by Ultralytics | Open-vocabulary instance segmentation | AGPL-3.0, or an Ultralytics Enterprise licence |
| MobileCLIP2-B text encoder | `mobileclip2_b.ts`, the Ultralytics TorchScript export of Apple's MobileCLIP2-B | YOLOE text prompts | Apple Machine Learning Research Model License (`apple-amlr`, HF `apple/MobileCLIP2-B`): research purposes only, no commercial use |
| MapAnything, Apache variant | HF `facebook/map-anything-apache` | Metric multi-view poses (mapping fallback) | Apache-2.0 (model card). The default `facebook/map-anything` checkpoint (CC-BY-NC-4.0) is not used. |
| DINOv2 hub code | GitHub `facebookresearch/dinov2` (`main`), fetched by `torch.hub.load` into the torch hub cache when MapAnything builds its encoder; no DINOv2 weights are downloaded (`torch_hub_pretrained=False`: they are in the MapAnything checkpoint) | MapAnything's ViT-G encoder architecture | Apache-2.0 (`LICENSE` of the fetched repository) |

## Python runtime dependencies (`pyproject.toml`)

| Package | Version | Licence | Notes |
|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | |
| scipy | 1.18.1 | BSD-3-Clause | |
| pillow | 12.3.0 | MIT-CMU | |
| av (PyAV) | 19.0.1 | BSD-3-Clause | The macOS wheel bundles FFmpeg 8 with libx264 and libx265 (GPL-2.0-or-later), so the binary is effectively GPL. |
| opencv-python-headless | 4.14.0.94 | Apache-2.0 (OpenCV) | Wheels ship FFmpeg under LGPL-2.1 (PyPI description). The macOS arm64 wheel also bundles libx264 and libx265 (GPL-2.0-or-later). |
| jsonschema | 4.26.0 | MIT | |
| psutil | 7.2.2 | BSD-3-Clause | |
| open3d | 0.20.0 | MIT | TSDF fusion of the map cloud |
| pycolmap | 4.2.1 | BSD-3-Clause | Mapping, triangulation, bundle adjustment; the wheel bundles only libomp |
| fastapi | 0.142.2 | MIT | |
| starlette | 1.7.0 | BSD-3-Clause | Also a fastapi dependency; the web service (`server.sh`) imports it directly |
| uvicorn | 0.54.0 | BSD-3-Clause | |
| httpx | 0.28.1 | BSD-3-Clause | |
| pydantic | 2.13.5 | MIT | |
| torch | 2.14.1 | BSD-3-Clause (PyTorch). The PyPI expression also lists Apache-2.0, Apache-2.0 WITH LLVM-exception and BSD-2-Clause for bundled parts. | Server process only |
| torchvision | 0.29.1 | BSD-3-Clause | |
| ultralytics | 8.4.159 | AGPL-3.0 | YOLOE runtime |
| clip (`ultralytics/CLIP` @ a13192f) | 1.0 | AGPL-3.0 (`LICENSE` at the pinned commit) | A fork of OpenAI CLIP, which is MIT |
| moge (`microsoft/MoGe` @ 925b8ed) | 2.0.0 | MIT; `moge/model/dinov2` is Apache-2.0 | The last pre-V3 commit, so MoGe-2 |
| utils3d (`EasternJournalist/utils3d` @ 3fab839) | 1.3 | MIT | MoGe dependency |
| pipeline (`EasternJournalist/pipeline` @ 866f059) | 1.0.0 | MIT | MoGe dependency |
| geocalib (`cvg/GeoCalib` @ 97b8968) | 1.0 | Apache-2.0 (code) | Weights: see above |
| mapanything (`facebookresearch/map-anything` @ 3d10cf7) | 1.1.4 | Apache-2.0 (code) | Weights: see above |

### Notable transitive dependencies

These rows are listed because they are copyleft or otherwise notable. Every runtime package,
with its licence, is in [All runtime packages](#all-runtime-packages-generated) below.

| Package | Version | Licence | Pulled in by |
|---|---|---|---|
| plyfile | 1.1.5 | GPL-3.0-or-later | MapAnything |
| pillow-heif | 1.8.0 | BSD-3-Clause source; **binary wheel GPL-2.0** (it bundles libheif and libde265, LGPL-3.0, and x265, GPL-2.0, per its `LICENSES_bundled.txt`) | MapAnything (oh-my-slam itself reads no HEIC/HEIF) |
| ultralytics-thop | 2.2.2 | AGPL-3.0 | ultralytics |
| ultralytics-platform | 0.1.79 | AGPL-3.0-only | ultralytics |
| certifi | 2026.7.22 | MPL-2.0 | httpx, requests |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | huggingface-hub, clip, MapAnything |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | MapAnything |
| pathspec | 1.1.1 | MPL-2.0 | mypy (dev; see [Development tools](#development-tools-dependency-groups-dev-and-build)) |
| trimesh | 5.1.1 | MIT | MoGe, MapAnything (no longer a direct dependency) |
| uniception | 0.1.7 | BSD-3-Clause | MapAnything |

### All runtime packages (generated)

Every package that `uv.lock` resolves for `[project] dependencies`, directly or transitively
(147 packages; the dev group's are in [Development tools](#development-tools-dependency-groups-dev-and-build)).
The licence is what the installed package's metadata states: its `License-Expression`, else its
`License` field, else its licence classifiers, else the heading of its `LICENSE` file. "Required
by" names the packages of this list that depend on it. Versions are those of `uv.lock`.

| Package | Version | Licence (installed metadata) | Required by |
|---|---|---|---|
| absl-py | 2.5.0 | Apache-2.0 | tensorboard |
| annotated-doc | 0.0.5 | MIT | fastapi |
| annotated-types | 0.8.0 | MIT | pydantic |
| antlr4-python3-runtime | 4.9.3 | BSD | hydra-core, omegaconf |
| anyio | 4.15.1 | MIT | httpx, starlette |
| argon2-cffi | 25.1.0 | MIT | minio |
| argon2-cffi-bindings | 26.1.0 | MIT | argon2-cffi |
| asttokens | 3.0.2 | Apache 2.0 | stack-data |
| attrs | 26.1.0 | MIT | jsonschema, referencing, rerun-sdk |
| av | 19.0.1 | BSD-3-Clause | oh-my-slam (direct) |
| blinker | 1.9.0 | MIT License | flask |
| certifi | 2026.7.22 | MPL-2.0 | httpcore, httpx, minio, requests |
| cffi | 2.1.1 | MIT-0 | argon2-cffi-bindings |
| charset-normalizer | 3.5.2 | MIT | requests |
| click | 8.5.0 | BSD-3-Clause | flask, huggingface-hub, moge, uvicorn |
| clip | 1.0 | AGPL-3.0 (no licence metadata; its LICENSE file) | oh-my-slam (direct) |
| cloudpickle | 3.1.2 | BSD-3-Clause | joblib, ultralytics |
| comm | 0.2.3 | BSD License | dash, ipywidgets |
| configargparse | 1.8.0 | MIT | open3d |
| contourpy | 1.4.0 | BSD-3-Clause | matplotlib |
| cycler | 0.12.1 | BSD License | matplotlib |
| dash | 4.4.1 | MIT | open3d |
| einops | 0.8.2 | MIT | uniception |
| executing | 2.2.1 | MIT | stack-data |
| fastapi | 0.142.2 | MIT | oh-my-slam (direct) |
| fastjsonschema | 2.22.2 | BSD-3-Clause | nbformat |
| filelock | 4.0.10 | MIT | huggingface-hub, torch, ultralytics |
| flask | 3.1.3 | BSD-3-Clause | dash, open3d |
| fonttools | 4.66.1 | MIT | matplotlib |
| fsspec | 2026.9.0 | BSD-3-Clause | huggingface-hub, torch |
| ftfy | 6.3.1 | Apache-2.0 | clip |
| geocalib | 1.0 | Apache Software License | oh-my-slam (direct) |
| glcontext | 3.0.0 | MIT | moderngl |
| grpcio | 1.84.0 | Apache-2.0 | tensorboard |
| h11 | 0.16.0 | MIT | httpcore, uvicorn |
| hf-xet | 1.6.0 | Apache-2.0 | huggingface-hub |
| httpcore | 1.0.9 | BSD-3-Clause | httpx |
| httpx | 0.28.1 | BSD-3-Clause | oh-my-slam (direct), huggingface-hub, ultralytics-platform |
| huggingface-hub | 1.33.0 | Apache-2.0 | mapanything, moge, timm |
| hydra-core | 1.3.7 | MIT | mapanything |
| idna | 3.20 | BSD-3-Clause | anyio, httpx, requests |
| importlib-metadata | 9.0.1 | Apache-2.0 | dash |
| ipython | 9.17.1 | BSD-3-Clause | ipywidgets |
| ipython-pygments-lexers | 1.1.1 | BSD License | ipython |
| ipywidgets | 8.1.9 | BSD 3-Clause License | open3d |
| itsdangerous | 2.2.0 | BSD License | flask |
| janus | 2.0.0 | Apache 2 | dash |
| jaxtyping | 0.3.11 | MIT License | uniception |
| jedi | 0.20.0 | MIT | ipython |
| jinja2 | 3.1.6 | BSD License | flask, torch |
| joblib | 1.6.0 | BSD-3-Clause | scikit-learn |
| jsonschema | 4.26.0 | MIT | oh-my-slam (direct), nbformat |
| jsonschema-specifications | 2025.9.1 | MIT | jsonschema |
| jupyter-core | 5.9.1 | BSD-3-Clause | nbformat |
| jupyterlab-widgets | 3.0.17 | BSD License | ipywidgets |
| kiwisolver | 1.5.1 | BSD License | matplotlib |
| kornia | 0.8.3 | Apache-2.0 | geocalib |
| kornia-rs | 0.2.0 | Apache Software License | kornia |
| lightning-utilities | 0.15.3 | Apache-2.0 | torchmetrics |
| mapanything | 1.1.4 | Apache-2.0 (no licence metadata; its LICENSE file) | oh-my-slam (direct) |
| markdown | 3.11 | BSD-3-Clause | tensorboard |
| markupsafe | 3.0.4 | BSD-3-Clause | flask, jinja2, werkzeug |
| matplotlib | 3.11.2 | Python Software Foundation License | geocalib, moge, ultralytics, uniception, utils3d |
| matplotlib-inline | 0.2.2 | BSD-3-Clause | ipython |
| minio | 7.2.20 | Apache-2.0 | uniception |
| moderngl | 5.12.0 | MIT | utils3d |
| moge | 2.0.0 | MIT | oh-my-slam (direct) |
| mpmath | 1.3.0 | BSD | sympy |
| narwhals | 2.26.0 | MIT | plotly, scikit-learn |
| natsort | 8.4.0 | MIT | mapanything |
| nbformat | 5.11.1 | BSD License | open3d |
| nest-asyncio | 1.6.0 | BSD | dash |
| networkx | 3.7 | BSD-3-Clause | torch |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | oh-my-slam (direct), contourpy, matplotlib, moge, open3d, opencv-python-headless, plyfile, pycolmap, rerun-sdk, scikit-learn, scipy, tensorboard, torchmetrics, torchvision, trimesh, ultralytics, ultralytics-thop, uniception, utils3d |
| nvidia-ml-py | 13.615.71 | BSD | ultralytics |
| omegaconf | 2.3.1 | BSD License | hydra-core |
| open3d | 0.20.0 | MIT | oh-my-slam (direct) |
| opencv-python-headless | 4.14.0.94 | Apache 2.0 | oh-my-slam (direct), mapanything |
| opentelemetry-api | 1.45.0 | Apache-2.0 | fastapi |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | mapanything |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | huggingface-hub, hydra-core, kornia, lightning-utilities, matplotlib, plotly, tensorboard, torchmetrics |
| parso | 0.8.7 | MIT | jedi |
| pexpect | 4.9.0 | ISC license | ipython |
| pillow | 12.3.0 | MIT-CMU | oh-my-slam (direct), matplotlib, moge, pillow-heif, rerun-sdk, tensorboard, torchvision, ultralytics, uniception |
| pillow-heif | 1.8.0 | BSD-3-Clause | mapanything |
| pipeline | 1.0.0 | MIT | oh-my-slam (direct), moge |
| platformdirs | 4.12.3 | MIT | jupyter-core |
| plotly | 7.1.0 | MIT | dash |
| plyfile | 1.1.5 | GNU General Public License v3 or later (GPLv3+) | mapanything |
| polars | 1.44.2 | MIT License | ultralytics |
| polars-runtime-32 | 1.44.2 | MIT | polars |
| prompt-toolkit | 3.0.53 | BSD License | ipython |
| protobuf | 7.36.2 | 3-Clause BSD License | tensorboard |
| psutil | 7.2.2 | BSD-3-Clause | oh-my-slam (direct), ipython, ultralytics |
| ptyprocess | 0.7.0 | ISC License (ISCL) | pexpect |
| pure-eval | 0.2.4 | MIT | stack-data |
| pyarrow | 25.0.1 | Apache-2.0 | rerun-sdk |
| pycolmap | 4.2.1 | BSD-3-Clause | oh-my-slam (direct) |
| pycparser | 3.0 | BSD-3-Clause | cffi |
| pycryptodome | 3.24.0 | BSD, Public Domain | minio |
| pydantic | 2.13.5 | MIT | oh-my-slam (direct), dash, fastapi |
| pydantic-core | 2.46.5 | MIT | pydantic |
| pygments | 2.21.0 | BSD-2-Clause | ipython, ipython-pygments-lexers |
| pyparsing | 3.3.3 | MIT | matplotlib |
| python-box | 7.4.1 | MIT | mapanything |
| python-dateutil | 2.9.0.post0 | Dual License (BSD License, Apache Software License) | matplotlib |
| pyyaml | 6.0.3 | MIT | huggingface-hub, omegaconf, timm, ultralytics |
| referencing | 0.37.0 | MIT | jsonschema, jsonschema-specifications |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | clip |
| requests | 2.34.2 | Apache-2.0 | dash, mapanything, ultralytics |
| rerun-sdk | 0.24.1 | MIT OR Apache-2.0 | mapanything, uniception |
| retrying | 1.4.2 | Apache-2.0 | dash |
| rpds-py | 2026.9.1 | MIT | jsonschema, referencing |
| safetensors | 0.8.0 | Apache Software License | mapanything, timm |
| scikit-learn | 1.9.1 | BSD-3-Clause | uniception |
| scipy | 1.18.1 | BSD License | oh-my-slam (direct), moge, scikit-learn, utils3d |
| setuptools | 84.0.0 | MIT | dash, tensorboard, torch |
| six | 1.17.0 | MIT | python-dateutil |
| stack-data | 0.6.3 | MIT | ipython |
| starlette | 1.7.0 | BSD-3-Clause | oh-my-slam (direct), fastapi |
| sympy | 1.14.0 | BSD | torch |
| tensorboard | 2.21.0 | Apache 2.0 | mapanything |
| tensorboard-data-server | 0.7.2 | Apache 2.0 | tensorboard |
| termcolor | 3.3.0 | MIT | uniception |
| threadpoolctl | 3.7.0 | BSD-3-Clause | scikit-learn |
| timm | 1.0.30 | Apache-2.0 | uniception |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | oh-my-slam (direct), clip, geocalib, kornia, moge, timm, torchmetrics, torchvision, ultralytics, ultralytics-thop, uniception |
| torchaudio | 2.11.0 | BSD License | uniception |
| torchmetrics | 1.9.0 | Apache-2.0 | uniception |
| torchvision | 0.29.1 | BSD | oh-my-slam (direct), clip, geocalib, moge, timm, ultralytics, uniception |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | clip, huggingface-hub, mapanything |
| traitlets | 5.16.1 | BSD License | ipython, ipywidgets, jupyter-core, matplotlib-inline, nbformat |
| trimesh | 5.1.1 | MIT License | mapanything, moge |
| typing-extensions | 4.16.0 | PSF-2.0 | anyio, dash, fastapi, grpcio, huggingface-hub, lightning-utilities, minio, opentelemetry-api, pydantic, pydantic-core, referencing, rerun-sdk, starlette, torch, typing-inspection |
| typing-inspection | 0.4.4 | MIT | fastapi, pydantic |
| ultralytics | 8.4.159 | AGPL-3.0 | oh-my-slam (direct) |
| ultralytics-platform | 0.1.79 | AGPL-3.0-only | ultralytics |
| ultralytics-thop | 2.2.2 | AGPL-3.0 | ultralytics |
| uniception | 0.1.7 | BSD 3-Clause | mapanything |
| urllib3 | 2.8.0 | MIT | minio, requests |
| utils3d | 1.3 | MIT | oh-my-slam (direct), moge |
| uvicorn | 0.54.0 | BSD-3-Clause | oh-my-slam (direct) |
| wadler-lindig | 0.1.7 | Apache Software License | jaxtyping |
| wcwidth | 0.9.1 | MIT License | ftfy, prompt-toolkit |
| werkzeug | 3.1.9 | BSD-3-Clause | dash, flask, open3d, tensorboard |
| widgetsnbextension | 4.0.16 | BSD 3-Clause License | ipywidgets |
| zipp | 4.1.1 | MIT | importlib-metadata |

## External tools (not in `.venv`)

| Component | Version | Used for | Licence |
|---|---|---|---|
| COLMAP (Homebrew `colmap`) | 4.2.0 | SIFT feature extraction and matching CLI (LightGlue on a video's weak links through its ONNX Runtime); GLOMAP is part of COLMAP 4.2 | BSD-3-Clause (COLMAP, GLOMAP). COLMAP's licence notes that its dependencies may change the licence of the built binary. The Homebrew formula depends on CGAL 6.2.1 (GPL-3.0-or-later), Qt 6 `qtbase` (LGPL-3.0 / GPL), SuiteSparse (mixed, including GPL and LGPL) and ONNX Runtime 1.30.0 (MIT), so the Homebrew binary should be treated as GPL-3.0-or-later. |
| LightGlue for SIFT (`sift-lightglue.onnx`) | COLMAP release asset 3.13.0, sha256 `e0500228…096e`; the `colmap` CLI downloads it into `~/.cache/colmap` on first use | Matching the SIFT keypoints of a video's weak links again (`mapping/sfm.py`) | Apache-2.0 (code and pre-trained weights, `cvg/LightGlue` README, checked 2026-10-02); the ONNX export is COLMAP's (BSD-3-Clause) |
| Microsoft Edge or Google Chrome | system-installed | Playwright browser tests and the evaluator's page timing | Proprietary; not bundled |
| uv | 0.9.18 (installed) | Environment and dependency manager | MIT OR Apache-2.0 |
| FFmpeg CLI (Homebrew `ffmpeg`) | 9.0.1 (installed) | One unit test writes a rotated video with it (`tests/unit/test_video.py`; skipped without it); development only, not bundled | LGPL-2.1-or-later source; the Homebrew build is configured `--enable-gpl --enable-version3` with libx264 and libx265, so the binary is GPL-3.0-or-later |
| bun (Homebrew `bun`) | 1.3.11 (installed) | Unit tests of the browser modules (`bun test`, `tests/js`, run by `tests/unit/test_js_units.py`); development only, not bundled | MIT (Bun itself; it statically links JavaScriptCore/WebKit, LGPL-2, and tinycc, LGPL-2.1) |

## Browser libraries (vendored in `src/oh_my_slam/viewer/static/vendor/`)

| Component | Version | Licence |
|---|---|---|
| three.js: `three.module.js`, `three.core.js` and the addons `OrbitControls`, `LineSegments2`, `LineMaterial`, `LineSegmentsGeometry` | 0.186.0 (r186), from the npm tarball recorded in `VERSIONS.txt` | MIT (`vendor/three/LICENSE`) |

The `server.sh` web application (`src/oh_my_slam/web/static/`) vendors nothing more: it reuses the
viewer's modules and three.js above, which it serves under `/static/viewer/`.

## Test-only browser libraries (vendored in `tests/browser/vendor/`, never served)

| Component | Version | Used for | Licence |
|---|---|---|---|
| axe-core: `axe.min.js` | 4.13.0, from the npm package `axe-core@4.13.0` (sha256 `c24f097b…a0c1`, which matches its published SRI `sha256-wk8Je9L0…RwoME=`) | The browser tests' automatic accessibility check of every page (WCAG 2.1 A and AA rules) | MPL-2.0 (`vendor/axe-core/LICENSE`; its bundled third-party notices in `LICENSE-3RD-PARTY.txt`) |

## Data, specifications and design assets

| Component | Where | Licence |
|---|---|---|
| ASAM OpenLABEL 1.0.0 JSON schema | Vendored as `src/oh_my_slam/schema/openlabel_json_schema.json`, sha256-pinned; published at `https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json` | **Proprietary, © ASAM e.V.; not an open licence** (checked online on 2026-10-07). The schema file carries no licence notice, and ASAM publishes no open-source licence for it (no OpenLABEL repository under `github.com/asam-ev`). The standard is free of charge, but downloading it means accepting ASAM's terms. ASAM's "Licensing Terms for Grant of Rights to Use ASAM Products" (`asam.net/license`, version of 2017-10-26) count schemata as part of a standard. Members and purchasers may use it commercially, and the research and education communities non-commercially. Research institutions and other licensees "are not entitled to pass standards on to third parties". Vendoring the schema in this repository is therefore covered only for ASAM members or purchasers. Anyone else should fetch it from the URL instead. |
| LVIS vocabulary | Label names curated from LVIS and COCO nouns in `segmentation/data/default_labels.txt`; the ontology URI `https://www.lvisdataset.org/` | LVIS annotations CC-BY-4.0; COCO annotations CC-BY-4.0 (COCO Consortium). No images are used. |
| Distinct-colour palette | Sasha Trubetskoy, "List of 20 Simple, Distinct Colors" (sashamaps.net); 10 colours used (the other 9 palette colours are this project's own) | No licence is stated on the source page, which offers the list for free download. The colours are credited in `segmentation/colors.py`. |
| Viridis colour map | 11 samples used for `color=height` | CC0-1.0 (mpl-colormaps by Nathaniel Smith and Stéfan van der Walt) |

## Development tools (`[dependency-groups] dev` and build)

Every package that `uv.lock` resolves for the `dev` group, directly or transitively (24 packages;
`uv sync` installs them by default). Generated as the runtime table above. click, packaging,
pygments and typing-extensions are runtime packages too.

| Package | Version | Licence (installed metadata) | Required by | Notes |
|---|---|---|---|---|
| ast-serialize | 0.12.1 | MIT | mypy | |
| click | 8.5.0 | BSD-3-Clause | import-linter | Also a runtime package |
| coverage | 7.16.2 | Apache-2.0 | pytest-cov | |
| greenlet | 3.5.6 | MIT AND PSF-2.0 | playwright | |
| grimp | 3.17 | BSD 2-Clause License | import-linter | |
| import-linter | 2.15 | BSD 2-Clause License | oh-my-slam (dev, direct) | |
| iniconfig | 2.3.0 | MIT | pytest | |
| librt | 0.16.0 | MIT | mypy | |
| markdown-it-py | 4.2.0 | MIT License | rich | |
| mdurl | 0.1.2 | MIT License | markdown-it-py | |
| mypy | 2.4.0 | MIT | oh-my-slam (dev, direct) | |
| mypy-extensions | 1.1.0 | MIT | mypy | |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | pytest | Also a runtime package |
| pathspec | 1.1.1 | MPL-2.0 | mypy | |
| playwright | 1.63.0 | Apache-2.0 | oh-my-slam (dev, direct) | The wheel bundles the Playwright driver (Apache-2.0; its `ThirdPartyNotices.txt`) and a Node.js 24.21.0 runtime (MIT, with the licences of its bundled parts in `driver/LICENSE`) |
| pluggy | 1.6.0 | MIT | pytest, pytest-cov | |
| pyee | 13.0.1 | MIT | playwright | |
| pygments | 2.21.0 | BSD-2-Clause | pytest, rich | Also a runtime package |
| pytest | 9.1.1 | MIT | oh-my-slam (dev, direct), pytest-cov, pytest-timeout | |
| pytest-cov | 7.1.0 | MIT | oh-my-slam (dev, direct) | |
| pytest-timeout | 2.4.0 | MIT | oh-my-slam (dev, direct) | |
| rich | 15.0.0 | MIT | import-linter | |
| ruff | 0.16.10 | MIT | oh-my-slam (dev, direct) | |
| typing-extensions | 4.16.0 | PSF-2.0 | import-linter, mypy, pyee | Also a runtime package |
| hatchling (build backend, `>=1.25`, not locked) | — | MIT | `[build-system]` | |
