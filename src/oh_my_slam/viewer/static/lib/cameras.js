// Cameras: the cameras of the scene as the server lists them (api/meta, viewer/bundle.py
// scene_cameras), drawn as frustums and listed with their centres and a "Go to".
//
// A camera is { name, frame, T (camera-to-scene, 4 x 4 row-major, OpenCV axes: x right, y down,
// z forward), position (its centre, metres), K: [fx, fy, cx, cy], size: [width, height],
// update, source (image file name) }.
import * as THREE from 'three';
import { el } from './dom.js';
import { srgb } from './obbs.js';

export const FRAME_COLOR = '#c9d1dc';

export function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }

// A frustum per camera, its depth a fraction of the scene's `size`.
export function buildFrustums(cams, size) {
  const group = new THREE.Group();
  if (!cams.length) return group;
  const d = Math.max(0.05, size * (cams.length === 1 ? 0.05 : 0.025));
  const pos = [];
  for (const f of cams) {
    const T = poseMatrix(f);
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    for (let i = 0; i < 4; i++) pos.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: srgb(FRAME_COLOR) }));
  lines.name = 'frustums';
  lines.frustumCulled = false;
  group.add(lines);
  return group;
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

// Rows of a camera table (camera, x, y, z, Go to) into `tbody`; `onGo(i)` moves the viewpoint.
export function fillCameraTable(tbody, cams, onGo) {
  tbody.replaceChildren();
  cams.forEach((f, i) => {
    const [x, y, z] = f.position;
    const go = el('button', { type: 'button', class: 'goto', title: 'Move the viewpoint to this camera',
      'aria-label': `Go to camera ${f.name}` }, 'Go to');
    go.addEventListener('click', () => onGo(i));
    const name = el('td', { class: 'cam-name' }, el('span', {}, f.name));
    if (f.source && f.source !== f.name) name.append(el('small', {}, f.source));
    tbody.appendChild(el('tr', { 'data-index': i }, name,
      el('td', { class: 'num' }, fmtCoord(x)), el('td', { class: 'num' }, fmtCoord(y)),
      el('td', { class: 'num' }, fmtCoord(z)), el('td', {}, go)));
  });
  if (!cams.length) tbody.append(el('tr', {}, el('td', { colspan: 5, class: 'muted' }, 'No cameras.')));
}
