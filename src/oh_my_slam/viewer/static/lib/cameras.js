// Cameras: read from an OpenLABEL scene (sceneCameras, the mirror of viewer/bundle.py
// scene_cameras, which /api/meta serves) or from the header of a PLY written by `mapper.sh locate`
// (plyCameras); drawn as frustums; listed with their centres and a "Go to".
//
// A camera is { name, frame, T (camera-to-scene, 4 x 4 row-major, OpenCV axes: x right, y down,
// z forward), position (its centre, metres), K: [fx, fy, cx, cy], size: [width, height],
// update, source (image file name), located (placed by `mapper.sh locate`) }.
import * as THREE from 'three';
import { el } from './dom.js';
import { srgb } from './obbs.js';

export const FRAME_COLOR = '#c9d1dc';    // a map's own frames (and an image's camera): solid lines
export const LOCATED_COLOR = '#ffb02e';  // located cameras: dashed lines, "located" in their label

function quatToMatrix(q, t) {
  const [x, y, z, w] = q;  // scalar last
  const n = Math.hypot(x, y, z, w) || 1;
  const m = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x / n, y / n, z / n, w / n));
  m.setPosition(t[0], t[1], t[2]);
  return m;
}
function rows(m) {  // THREE.Matrix4 (column-major) -> 4 x 4 row-major
  const e = m.elements;
  return [0, 1, 2, 3].map((r) => [e[r], e[4 + r], e[8 + r], e[12 + r]]);
}
function baseName(uri) { return String(uri || '').split(/[\\/]/).pop(); }

function camera(name, frame, transform, pinhole, extra) {
  const T = transform ? rows(quatToMatrix(transform.quaternion, transform.translation))
    : [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  if (transform) for (let i = 0; i < 3; i++) T[i][3] = transform.translation[i];  // exactly as stated
  const m = pinhole.camera_matrix;
  return { name, frame, T, position: [T[0][3], T[1][3], T[2][3]], K: [m[0], m[5], m[2], m[6]],
    size: [pinhole.width_px, pinhole.height_px], update: null, source: '', located: false, ...extra };
}

// The camera of every frame of an OpenLABEL scene, in the objects' coordinate system: the frame's
// `<stream> -> <cs>` transform (a map keyframe, a located image), or the identity for a stream
// whose sensor frame is itself a root coordinate system (a single image).
export function sceneCameras(doc) {
  const root = doc.openlabel || {};
  const streams = root.streams || {};
  const systems = root.coordinate_systems || {};
  const out = [];
  const frames = Object.entries(root.frames || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
  for (const [fid, fr] of frames) {
    const props = fr.frame_properties || {};
    const bySrc = {};
    for (const t of Object.values(props.transforms || {})) bySrc[t.src] = t;
    for (const [name, stream] of Object.entries(props.streams || {})) {
      const pin = ((streams[name] || {}).stream_properties || {}).intrinsics_pinhole;
      if (!pin) continue;
      let transform = null;
      if (bySrc[name]) transform = bySrc[name].transform_src_to_dst;
      else if (((systems[name] || {}).parent ?? '') !== '') continue;
      out.push(camera(props.keyframe ?? name, Number(fid), transform, pin, {
        update: props.update_id ?? null, source: baseName(stream.uri), located: props.located === true,
      }));
    }
  }
  return out;
}

// The cameras `mapper.sh locate` writes in a PLY header (mapping/locate.py pose_comment): one
// comment `located_<k> {json}` per input image, with `transform_src_to_dst` (camera to map) and
// `stream_properties` for an image it located, `"located": false` for one it could not.
// Returns { cameras, unlocated: [image, ...] }.
export function plyCameras(comments) {
  const cameras = [], unlocated = [];
  for (const c of comments) {
    const m = /^(located_\d+) (\{.*\})$/.exec(c);
    if (!m) continue;
    let d;
    try { d = JSON.parse(m[2]); } catch { continue; }
    const pin = (d.stream_properties || {}).intrinsics_pinhole;
    if (d.located !== true || !d.transform_src_to_dst || !pin) { unlocated.push(d.image ?? m[1]); continue; }
    cameras.push(camera(m[1], Number(m[1].slice('located_'.length)), d.transform_src_to_dst, pin, {
      source: baseName(d.image), located: true,
    }));
  }
  return { cameras, unlocated };
}

export function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }

// A frustum per camera, its depth a fraction of the scene's `size`: map frames solid, located
// cameras dashed and in their own colour (never the colour alone).
export function buildFrustums(cams, size) {
  const group = new THREE.Group();
  if (!cams.length) return group;
  const d = Math.max(0.05, size * (cams.length === 1 ? 0.05 : 0.025));
  for (const located of [false, true]) {
    const pos = [];
    for (const f of cams.filter((c) => !!c.located === located)) {
      const T = poseMatrix(f);
      const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
      const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
      const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
        new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
      for (let i = 0; i < 4; i++) pos.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
    }
    if (!pos.length) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const material = located
      ? new THREE.LineDashedMaterial({ color: srgb(LOCATED_COLOR), dashSize: d / 6, gapSize: d / 10 })
      : new THREE.LineBasicMaterial({ color: srgb(FRAME_COLOR) });
    const lines = new THREE.LineSegments(g, material);
    if (located) lines.computeLineDistances();
    lines.name = located ? 'located-frustums' : 'frustums';
    lines.frustumCulled = false;
    group.add(lines);
  }
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
    if (f.located) name.append(el('small', { class: 'located' }, 'located'));
    if (f.source && f.source !== f.name) name.append(el('small', {}, f.source));
    tbody.appendChild(el('tr', { 'data-index': i, 'data-located': f.located || null }, name,
      el('td', { class: 'num' }, fmtCoord(x)), el('td', { class: 'num' }, fmtCoord(y)),
      el('td', { class: 'num' }, fmtCoord(z)), el('td', {}, go)));
  });
  if (!cams.length) tbody.append(el('tr', {}, el('td', { colspan: 5, class: 'muted' }, 'No cameras.')));
}
