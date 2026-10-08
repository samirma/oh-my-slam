// One point cloud read from a PLY's bytes and drawn by this viewer as view.sh draws a cloud: the
// Viewer of viewer.js with the materials of cloud.js (the same point size, the colours it carries
// written exactly, shading by its normals, the same background and initial view), for a page that
// holds the bytes — the server.sh web application's point-cloud results (http_server.md "Image").
//
// * The bytes are read off the page's thread (ply.js), within the display budget (controls.js):
//   above it every k-th point is drawn, and budgetNote says so.
// * A cloud whose header names a single image's camera frame is shown upright as view.sh -i shows
//   one, here for a level camera: a PLY does not carry the estimated up direction. A map's cloud
//   (z up) is drawn as it is.
// * The host page draws its own chrome (text alternative, buttons); CloudView makes the moves they
//   ask for (rotate, pan, zoom, reset) and answers the keys of a focused host.
import { Viewer } from './viewer.js';
import { loadPly, plyHeader } from './ply.js';
import { budgetNote, DISPLAY_POINT_BUDGET } from './controls.js';

export { budgetNote, DISPLAY_POINT_BUDGET, plyHeader };

// reconstruction.cloud.IMAGE_FRAME: the header comment of a cloud in a single image's camera frame
export const IMAGE_FRAME = 'oh-my-slam camera frame (OpenCV axes: x right, y down, z forward), metres';
// viewer/bundle.py upright_transform(reconstruction.gravity.DEFAULT_UP_CAM), row-major: a level
// camera's frame shown upright (x right; z forward becomes +y; y down becomes -z, so up is +z)
export const LEVEL_UPRIGHT = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]];
const IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

export const ROTATE_STEP_DEG = 15;  // one rotation step (a button, an arrow key)
export const PAN_STEP = 0.1;        // one pan step, a fraction of the view's height
export const ZOOM_STEP = 1.25;      // one zoom step

export function isUpright(header) { return (header.comments || []).includes(IMAGE_FRAME); }

// The cloud of PLY bytes (an ArrayBuffer, handed over to the reader), within `budget` points.
export function readCloud(buffer, budget = DISPLAY_POINT_BUDGET) { return loadPly(buffer, budget); }

// What a text alternative of the cloud states: the points drawn and in the file, what each point
// carries, the attributes its writer recorded ("key=value,…"), the frame its header names, its
// format, whether it is shown upright, and the display-budget notice ('' when every point is drawn).
export function cloudFacts({ header, arrays }) {
  const recorded = new Map((header.attrs || '').split(',').filter(Boolean).map((p) => p.split('=')));
  const carries = ['position'];
  if (arrays.color) carries.push(recorded.has('color') ? `colour (color=${recorded.get('color')})` : 'colour');
  if (arrays.normal) carries.push('normal');
  if (arrays.label) carries.push('object id');
  return {
    count: header.count, total: header.total, carries, attrs: header.attrs || '',
    frame: (header.comments || []).find((c) => / frame\b|, metres/.test(c)) || '',
    format: header.format || '', upright: isUpright(header), note: budgetNote(header),
  };
}

// The move a key asks of a focused view, as [CloudView method, ...its arguments]: arrows rotate,
// Shift + arrows pan, + and - zoom, 0 or Home resets; null for any other key or a modifier.
export function keyMove(e) {
  if (e.altKey || e.ctrlKey || e.metaKey) return null;
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  const a = arrows[e.key];
  if (a && e.shiftKey) return ['pan', a[0] * PAN_STEP, a[1] * PAN_STEP];
  if (a) return ['rotate', -a[0] * ROTATE_STEP_DEG, a[1] * ROTATE_STEP_DEG];
  if (e.key === '+' || e.key === '=') return ['zoom', ZOOM_STEP];
  if (e.key === '-' || e.key === '_') return ['zoom', 1 / ZOOM_STEP];
  if (e.key === '0' || e.key === 'Home') return ['reset'];
  return null;
}

export class CloudView {
  constructor(host) {
    this.host = host;
    this.viewer = new Viewer(host);
    this._onKey = (e) => { if (this.key(e)) e.preventDefault(); };
    host.addEventListener('keydown', this._onKey);
  }

  get drawn() { return this.viewer.cloudDrawn; }  // a frame showing the cloud has been drawn
  get frames() { return this.viewer.frames; }

  // Draw `cloud` (readCloud) and frame it as view.sh's initial view does.
  show(cloud) {
    const v = this.viewer;
    v.setDisplayTransform(isUpright(cloud.header) ? LEVEL_UPRIGHT : IDENTITY);
    v.setCloud(cloud);
    v.setLayer('points', true);  // applies every layer's state (no scene here: no segmentation layer)
    v.resize();
    v.resetView();
  }

  // Orbit by `left` degrees around the vertical axis and `up` degrees towards the top view.
  rotate(left, up = 0) {
    const c = this.viewer.controls;
    if (left) c.rotateLeft((left * Math.PI) / 180);
    if (up) c.rotateUp((up * Math.PI) / 180);
  }

  // Move the viewpoint `right` and `up`, in fractions of the view's height.
  pan(right, up = 0) {
    const h = Math.max(1, this.host.clientHeight);
    this.viewer.controls.pan(-right * h, up * h);
  }

  // Move `factor` times closer to the point the view turns around (< 1: farther).
  zoom(factor) { this.viewer.controls.dollyIn(1 / factor); }

  reset() { this.viewer.resetView(); }

  // The keys of a focused host (keyMove). Returns true when the key moved the view.
  key(e) {
    const move = keyMove(e);
    if (!move) return false;
    const [name, ...args] = move;
    this[name](...args);
    return true;
  }

  dispose() {
    this.host.removeEventListener('keydown', this._onKey);
    this.viewer.dispose();
  }
}
