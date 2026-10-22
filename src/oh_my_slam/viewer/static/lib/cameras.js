// Cameras: the cameras of the scene as the server lists them (api/meta, viewer/bundle.py
// scene_cameras), drawn as frustums and listed with their centres and a "Go to".
//
// A camera is { name, T (camera-to-scene, 4 x 4 row-major, OpenCV axes: x right, y down,
// z forward), position (its centre, metres), K: [fx, fy, cx, cy], size: [width, height],
// source (image file name) }, and, for a camera a result names besides the scene's own
// (cloudview.js recordCamera), its `kind`: such a camera is drawn in a colour of its kind
// (KIND_COLORS) and labelled with its name and kind.
import * as THREE from 'three';
import { el } from './dom.js';
import { srgb } from './obbs.js';

export const FRAME_COLOR = '#c9d1dc';
// the colours of the cameras of a kind (white, and a pale cream: neither is an object colour, whose
// chroma and lightness the palette bounds), and what each kind is called on its label
export const KIND_COLORS = { located: '#ffffff', last_known: '#f4e3b5' };
export const KIND_NAMES = { located: 'located', last_known: 'last known camera' };

export function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }

// A camera's frustum colour: its kind's, else the scene's cameras' FRAME_COLOR.
export function frustumColor(f) { return KIND_COLORS[f.kind] || FRAME_COLOR; }

// The depth of the frustums of `count` cameras in a scene of `size`.
export function frustumDepth(count, size) { return Math.max(0.05, size * (count === 1 ? 0.05 : 0.025)); }

// A frustum per camera, its depth a fraction of the scene's `size`: one line set per colour (the
// scene's own cameras' `frustums`, a kind's `frustums-<kind>`).
export function buildFrustums(cams, size) {
  const group = new THREE.Group();
  if (!cams.length) return group;
  const d = frustumDepth(cams.length, size);
  const byColor = new Map();
  for (const f of cams) {
    const color = frustumColor(f);
    if (!byColor.has(color)) byColor.set(color, { name: KIND_COLORS[f.kind] ? `frustums-${f.kind}` : 'frustums', pos: [] });
    const { pos } = byColor.get(color);
    const T = poseMatrix(f);
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    for (let i = 0; i < 4; i++) pos.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
  }
  for (const [color, { name, pos }] of byColor) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: srgb(color) }));
    lines.name = name;
    lines.frustumCulled = false;
    group.add(lines);
  }
  return group;
}

// The label of a camera of a kind (none for the scene's own cameras), in its kind's colour: as its
// tag, the input it placed ("input 2": its `input` number, as the result's tables and its
// bird's-eye image number it) or else its name, then its kind; its name in the tooltip.
export function cameraLabel(f) {
  if (!KIND_COLORS[f.kind]) return null;
  const tag = f.input ? `input ${f.input}` : f.name;
  return { id: `camera-${f.name}`, tag, name: KIND_NAMES[f.kind], background: KIND_COLORS[f.kind], ink: '#000000',
    title: `${tag}: ${KIND_NAMES[f.kind]}${f.input ? ` (${f.name})` : ''}`, note: `${tag} ${KIND_NAMES[f.kind]}`, group: 'cameras' };
}

// The viewpoint at a camera: its centre, looking along its optical axis, with the vertical field
// of view that shows its whole image at `aspect`; the orbit pivot straight ahead, towards the
// middle of `bbox` (0.5 m at least). `matrix` maps the scene frame to the display frame.
export function cameraView(f, matrix, bbox, aspect) {
  const M = poseMatrix(f).premultiply(matrix);
  const eye = new THREE.Vector3().setFromMatrixPosition(M);
  const fwd = new THREE.Vector3(0, 0, 1).transformDirection(M);
  const [fx, fy] = f.K; const [w, h] = f.size;
  const half = Math.max(h / (2 * fy), w / (2 * fx) / Math.max(aspect, 1e-3));
  const fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(half)), 5, 150);
  let dist = 1.0;
  if (!bbox.isEmpty()) {
    const ahead = bbox.getCenter(new THREE.Vector3()).sub(eye).dot(fwd);
    const size = bbox.getSize(new THREE.Vector3()).length();
    dist = THREE.MathUtils.clamp(ahead, 0.5, Math.max(0.5, size / 2));
  }
  return { eye, target: eye.clone().addScaledVector(fwd, dist), fov };
}

export function fmtCoord(v) { return (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3); }

// What the camera table shows of each camera: { name, source (its image file name when that is
// not its name, else null), coords: its centre's x, y, z as shown }.
export function cameraRows(cams) {
  return cams.map((f) => {
    const [x, y, z] = f.position;
    return { name: f.name, source: f.source && f.source !== f.name ? f.source : null,
      coords: [fmtCoord(x), fmtCoord(y), fmtCoord(z)] };
  });
}

// Rows of a camera table (camera, x, y, z, Go to) into `tbody`; `onGo(i)` moves the viewpoint.
export function fillCameraTable(tbody, cams, onGo) {
  tbody.replaceChildren();
  cameraRows(cams).forEach((r, i) => {
    const go = el('button', { type: 'button', class: 'goto', title: 'Move the viewpoint to this camera',
      'aria-label': `Go to camera ${r.name}` }, 'Go to');
    go.addEventListener('click', () => onGo(i));
    const name = el('td', { class: 'cam-name' }, el('span', {}, r.name));
    if (r.source) name.append(el('small', {}, r.source));
    tbody.appendChild(el('tr', { 'data-index': i }, name,
      ...r.coords.map((c) => el('td', { class: 'num' }, c)), el('td', {}, go)));
  });
  if (!cams.length) tbody.append(el('tr', {}, el('td', { colspan: 5, class: 'muted' }, 'No cameras.')));
}
