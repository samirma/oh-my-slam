// oh-my-slam viewer (spec §2.5): display only. Every point cloud is derived by the server
// (GET /api/cloud, the shared derivation of segmentation.cloud); objects, OBBs, colours and camera
// poses come from the scene JSON and /api/meta. Nothing is recomputed here.
//
// What the page offers is what §2.5 asks for: independent point-cloud, camera-pose, segmentation,
// label and OBB layers; live controls for the applicable point-cloud attributes; the position of
// every camera with a "Go to" that moves the viewpoint there; and, for an image, the segmented
// image and the object catalogue.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const $ = (sel) => document.querySelector(sel);
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  e.append(...children);
  return e;
}

// [key, name, what it shows]
const LAYERS = [
  ['points', 'Point cloud', 'The derived cloud, coloured by the color attribute below'],
  ['segments', 'Segmentation',
    'Only the points of each object, in the object\'s colour, drawn over the point cloud'],
  ['cameras', 'Camera poses', 'A frustum at each camera\'s pose'],
  ['labels', 'Labels', 'Each box\'s id and label, in its colour'],
  ['obbs', 'Oriented boxes', 'The objects\' oriented bounding boxes, in their colours'],
];
const DEBOUNCE_MS = 250;
const POINT_SIZE_CM = 2.0;
const BACKGROUND = '#15171c';
const state = {
  meta: null, scene: null, objects: [], objectRgb: new Map(),
  layers: { points: true, segments: false, cameras: true, labels: true, obbs: true },
  attrs: {},          // current point-cloud attributes, as the -p values the server parses
  cloud: null,        // header of the cloud on screen
  cloudSeq: 0, abort: null, debounce: null, busy: false, framed: false, cloudInScene: false,
  bbox: new THREE.Box3(), ready: false, labelsDirty: true,
  frames: 0,          // frames drawn so far (drawn on demand only; read by the tests)
};
window.__viewer = state; // for tests and debugging

// ---------------------------------------------------------------- renderer, scene, controls
const host = $('#canvas-host');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;   // material colours are converted back exactly
renderer.toneMapping = THREE.NoToneMapping;          // no tone mapping: colours stay exact
host.appendChild(renderer.domElement);
const labelLayer = el('div', { id: 'labels', 'aria-label': 'Object labels' });
host.appendChild(labelLayer);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BACKGROUND);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
camera.up.set(0, 0, 1);  // map and display frames are z-up
camera.position.set(-3, -3.6, 2.7);  // until the cloud is framed (e.g. an empty map)
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;

// Frames are drawn on demand: an idle page draws nothing, so that a large cloud does not keep the
// GPU busy next to the inference server. Whatever changes the canvas calls invalidate(); the
// animation loop also draws whenever the viewpoint moved (orbiting and its damping, go-to).
const REDRAW_FRAMES = 2;  // frames drawn after each change: a margin for anything that lands late
let redraw = 1;           // the first frame: the empty view's background
function invalidate(frames = REDRAW_FRAMES) { redraw = Math.max(redraw, frames); }
window.__viewerInvalidate = invalidate;  // for tests that change the scene directly
controls.addEventListener('change', () => invalidate());
for (const type of ['pointerdown', 'pointerup', 'wheel', 'keydown', 'input', 'change', 'click']) {
  window.addEventListener(type, () => invalidate(), { capture: true, passive: true });
}
document.addEventListener('visibilitychange', () => invalidate());
renderer.domElement.addEventListener('webglcontextrestored', () => invalidate());

const root = new THREE.Group();  // display transform (image mode: camera frame → z-up)
scene.add(root);
const groups = {};
for (const k of ['points', 'segments', 'cameras', 'obbs', 'labels']) {
  groups[k] = new THREE.Group();
  groups[k].name = k;
  root.add(groups[k]);
}
window.__viewerGroups = groups;  // read by the browser tests
window.__viewerCamera = camera;
window.__viewerControls = controls;

function focalPx() {
  const h = Math.max(host.clientHeight, 1);
  return (h * renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
}
function pointMaterials() {
  const out = [];
  for (const g of [groups.points, groups.segments]) g.traverse((m) => { if (m.material?.uniforms?.focal) out.push(m.material); });
  return out;
}
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  for (const o of state.objects) o.line.material.resolution.set(w, h);
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
  state.labelsDirty = true;
  invalidate();  // resizing the canvas clears it
  if (state.homePending && w > 0 && h > 0) resetView();
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- loading
async function errorOf(res) {
  try { return (await res.json()).error || `HTTP ${res.status}`; } catch { return `HTTP ${res.status}`; }
}
async function fetchBuffer(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(await errorOf(res));
  return res.arrayBuffer();
}
async function fetchJSON(url) {
  return JSON.parse(new TextDecoder().decode(await fetchBuffer(url)));
}

// ---------------------------------------------------------------- point clouds (raw sRGB)
// The binary document of /api/cloud: uint32 LE header length J, JSON header, 4-byte aligned buffers.
const TYPED = { float32: Float32Array, uint8: Uint8Array, int32: Int32Array };
function parseCloud(buffer) {
  const hlen = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, hlen)));
  const base = 4 + hlen;
  const arrays = {};
  for (const b of header.buffers) {
    const T = TYPED[b.type];
    arrays[b.name] = new T(buffer, base + b.offset, b.bytes / T.BYTES_PER_ELEMENT);
  }
  return { header, arrays };
}
// Points are drawn in their sRGB bytes as they are; with normals (normals=on) they are shaded by
// them (headlight), so that the attribute shows, except with color=segment, whose object colours
// and unsegmented grey must reach the screen exactly (§2.4 colour contract).
function pointMaterial({ color, normal, exact }) {
  const defines = {};
  if (color) defines.HAS_COLOR = '';
  if (normal) defines.HAS_NORMAL = '';
  return new THREE.ShaderMaterial({
    defines,
    uniforms: {
      size: { value: POINT_SIZE_CM },               // point diameter in centimetres
      shade: { value: exact ? 0 : 1 },              // 0: colours exactly as sent
      focal: { value: focalPx() },                  // viewport focal length in pixels
      base: { value: new THREE.Vector3(0.80, 0.82, 0.86) },  // color=none
    },
    vertexShader: `
      #ifdef HAS_COLOR
      attribute vec3 rgb;
      #endif
      uniform float size; uniform float focal; uniform int shade; uniform vec3 base;
      varying vec3 vColor;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        #ifdef HAS_COLOR
        vec3 c = rgb;
        #else
        vec3 c = base;
        #endif
        #ifdef HAS_NORMAL
        if (shade == 1) {
          vec3 n = normalize(normalMatrix * normal);
          c *= 0.3 + 0.7 * abs(dot(n, normalize(-mv.xyz)));
        }
        #endif
        vColor = c;
        gl_PointSize = clamp(size * 0.01 * focal / max(-mv.z, 0.05), 1.0, 24.0);
        gl_Position = projectionMatrix * mv;
      }`,
    // No colour-space conversion: the bytes are sRGB already and are written as they are.
    fragmentShader: `
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }`,
  });
}
function clearGroup(g) {
  for (const c of [...g.children]) { g.remove(c); c.geometry?.dispose(); c.material?.dispose(); }
}
function showCloud({ header, arrays }) {
  clearGroup(groups.points);
  clearGroup(groups.segments);
  const n = header.count;
  const position = new THREE.BufferAttribute(arrays.position, 3);
  // point-cloud layer: the derived cloud in the colour the `color` attribute selects
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', position);
  if (arrays.color) g.setAttribute('rgb', new THREE.BufferAttribute(arrays.color, 3, true));
  if (arrays.normal) g.setAttribute('normal', new THREE.BufferAttribute(arrays.normal, 3));
  const exact = (header.attrs || '').split(',').includes('color=segment');
  const pts = new THREE.Points(g, pointMaterial({ color: !!arrays.color, normal: !!arrays.normal, exact }));
  pts.name = 'points';
  pts.frustumCulled = false;
  groups.points.add(pts);
  // segmentation layer: the segmented points (object id != 0) in their object's colour, exactly
  // the colour the scene JSON states for that id (§2.4); drawn over the cloud
  if (arrays.label) {
    const idx = new Uint32Array(n);
    const col = new Uint8Array(n * 3);
    let m = 0;
    for (let i = 0; i < n; i++) {
      const rgb = arrays.label[i] > 0 ? state.objectRgb.get(arrays.label[i]) : undefined;
      if (rgb) { idx[m++] = i; col[3 * i] = rgb[0]; col[3 * i + 1] = rgb[1]; col[3 * i + 2] = rgb[2]; }
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', position);
    sg.setAttribute('rgb', new THREE.BufferAttribute(col, 3, true));
    sg.setIndex(new THREE.BufferAttribute(idx.slice(0, m), 1));
    const seg = new THREE.Points(sg, pointMaterial({ color: true, exact: true }));
    seg.name = 'segments';
    seg.frustumCulled = false;
    seg.renderOrder = 1;  // after the cloud: same positions, depth test passes (less-or-equal)
    groups.segments.add(seg);
  }
  state.cloud = header;
  state.cloudInScene = true;
  // a cloud above the display limit is thinned for display only: say so (spec §2.5: complete cloud)
  const note = $('#cloud-note');
  note.hidden = !(header.step > 1);
  note.textContent = header.step > 1
    ? `Showing every ${header.step}th of ${header.total.toLocaleString()} points (display limit).` : '';
  invalidate();
  const pos = arrays.position;
  if (!state.framed && pos.length) {
    state.framed = true;
    state.bbox.union(robustBox(pos, root.matrixWorld));
  }
}

// ---------------------------------------------------------------- OBBs and labels
function srgb(hex) { return new THREE.Color().setStyle(hex, THREE.SRGBColorSpace); }
function obbCorners(c) {
  const [x, y, z, qx, qy, qz, qw, sx, sy, sz] = c;
  const q = new THREE.Quaternion(qx, qy, qz, qw);
  const pts = [];
  for (const dx of [-0.5, 0.5]) for (const dy of [-0.5, 0.5]) for (const dz of [-0.5, 0.5]) {
    pts.push(new THREE.Vector3(dx * sx, dy * sy, dz * sz).applyQuaternion(q).add(new THREE.Vector3(x, y, z)));
  }
  return pts;
}
const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
function prop(o, kind, name) { const v = (o.object_data[kind] || []).find((n) => n.name === name); return v ? v.val : null; }
function attr(cub, name) { const v = ((cub.attributes || {}).num || []).find((n) => n.name === name); return v ? v.val : 0; }
// text on a tag of this background: black or white, whichever contrasts more (WCAG luminance)
function inkFor(rgb) {
  const lin = rgb.map((v) => { const u = v / 255; return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4; });
  const y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  return (y + 0.05) / 0.05 >= 1.05 / (y + 0.05) ? '#000' : '#fff';
}

function buildObjects(doc) {
  const objs = doc.openlabel.objects || {};
  for (const [id, o] of Object.entries(objs)) {
    const hex = prop(o, 'text', 'color_hex');
    const rgb = prop(o, 'vec', 'color');
    if (rgb) state.objectRgb.set(Number(id), rgb);
    const cub = (o.object_data.cuboid || [])[0];
    if (!cub) continue;
    const corners = obbCorners(cub.val);
    const pos = [];
    for (const [a, b] of EDGES) { pos.push(...corners[a].toArray(), ...corners[b].toArray()); }
    const geom = new LineSegmentsGeometry().setPositions(pos);
    const mat = new LineMaterial({ color: srgb(hex), linewidth: 2.0, worldUnits: false });
    const line = new LineSegments2(geom, mat);
    line.userData.id = Number(id);
    groups.obbs.add(line);
    // label: the id tag in the object's colour, then the name; anchored at the top face's centre
    const tag = el('span', { class: 'tag' }, String(id));
    tag.style.background = hex;
    tag.style.color = inkFor(rgb || [128, 128, 128]);
    const div = el('div', { class: 'obj-label', 'data-id': id, title: `${o.type} ${id}` }, tag,
      el('span', { class: 'name' }, o.type));
    labelLayer.appendChild(div);
    const display = corners.map((p) => p.clone().applyMatrix4(root.matrixWorld));
    display.sort((a, b) => b.z - a.z);
    const anchor = display.slice(0, 4).reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(0.25);
    const [, , , , , , , sx, sy, sz] = cub.val;
    state.objects.push({
      id: Number(id), label: o.type, hex, score: prop(o, 'num', 'score'), volume: attr(cub, 'volume_m3'),
      dims: [attr(cub, 'width_m'), attr(cub, 'depth_m'), attr(cub, 'height_m')],
      line, div, anchor, size: Math.cbrt(Math.max(sx * sy * sz, 1e-9)),
      w: 0, wTag: 0, h: 0, mode: null, rect: null,
    });
  }
}

// Labels (spec §2.5: labelled OBBs), kept legible: no label ever covers another. Each box whose
// top is in view gets its id tag next to the top face's centre, or on a ring farther out, larger
// boxes on screen first. A box whose tag finds no free place keeps no tag on screen; its id and
// label are listed under the Labels layer ("No room for: …"), and appear once the view is zoomed
// in. Then the names are added, in the same order, wherever the longer label covers nothing.
const LABEL_GAP = 2;          // px between two labels, and between a label and the view's edge
const RINGS = [22, 34, 48, 64, 84];  // px from the anchor to the label's centre
const DIRS = [0, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6].map((k) => {  // from straight up, both ways
  const a = -Math.PI / 2 + (k * Math.PI) / 6;
  return [Math.cos(a), Math.sin(a)];
});
function measureLabels() {
  for (const o of state.objects) {
    o.div.classList.remove('compact');
    const r = o.div.getBoundingClientRect();
    o.w = Math.ceil(r.width); o.h = Math.ceil(r.height);
    o.wTag = Math.ceil(o.div.firstChild.getBoundingClientRect().width);
  }
  state.measured = state.objects.every((o) => o.w > 0);
}
function layoutLabels() {
  const show = state.layers.labels;
  groups.labels.visible = show;
  labelLayer.hidden = !show;
  if (!show) $('#labels-note').hidden = true;
  if (!show || !state.objects.length) return;
  if (!state.measured) measureLabels();
  const W = host.clientWidth, H = host.clientHeight;
  const taken = [];
  const free = (x, y, bw, bh, ignore = null) => !taken.some((r) => r !== ignore
    && x - LABEL_GAP < r[2] && r[0] < x + bw + LABEL_GAP && y - LABEL_GAP < r[3] && r[1] < y + bh + LABEL_GAP);
  const clampX = (x, bw) => Math.min(Math.max(x, LABEL_GAP), W - bw - LABEL_GAP);
  const clampY = (y, bh) => Math.min(Math.max(y, LABEL_GAP), H - bh - LABEL_GAP);
  const candidates = (sx, sy, bw, bh) => {
    const out = [[sx - bw / 2, sy - bh - 3], [sx + 4, sy - bh / 2], [sx - bw - 4, sy - bh / 2],
      [sx - bw / 2, sy + 3]];
    for (const r of RINGS) for (const [dx, dy] of DIRS) out.push([sx + dx * r - bw / 2, sy + dy * r - bh / 2]);
    return out.map(([x, y]) => [clampX(x, bw), clampY(y, bh)]);
  };
  const spot = (sx, sy, bw, bh) => candidates(sx, sy, bw, bh).find(([x, y]) => free(x, y, bw, bh)) || null;
  const eye = camera.position;
  const p = new THREE.Vector3();
  const boxes = [];
  for (const o of state.objects) {
    o.mode = null;
    p.copy(o.anchor).project(camera);
    const sx = (p.x + 1) / 2 * W, sy = (1 - p.y) / 2 * H;
    if (!(p.z < 1 && p.z > -1 && sx >= 0 && sx <= W && sy >= 0 && sy <= H)) continue;
    o.sx = sx; o.sy = sy;
    o.rank = o.size / Math.max(eye.distanceTo(o.anchor), 1e-3);
    boxes.push(o);
  }
  boxes.sort((a, b) => (b.rank - a.rank) || (a.id - b.id));
  const placed = [], crowded = [];
  for (const o of boxes) {
    const at = spot(o.sx, o.sy, o.wTag, o.h);
    if (!at) { crowded.push(o); continue; }
    o.mode = 'compact';
    o.rect = [at[0], at[1], at[0] + o.wTag, at[1] + o.h];
    taken.push(o.rect);
    placed.push(o);
  }
  const note = $('#labels-note');
  crowded.sort((a, b) => a.id - b.id);
  note.hidden = !crowded.length;
  note.textContent = crowded.length
    ? `No room for: ${crowded.map((o) => `${o.id} ${o.label}`).join(', ')} (zoom in to show)` : '';
  for (const o of placed) {  // names: the tag grows rightwards, else leftwards, where that is free
    const r = o.rect;
    for (const x of [r[0], r[2] - o.w]) {
      if (x >= LABEL_GAP && x + o.w <= W - LABEL_GAP && free(x, r[1], o.w, o.h, r)) {
        r[0] = x; r[2] = x + o.w; o.mode = 'full'; break;
      }
    }
  }
  for (const o of state.objects) {
    const d = o.div;
    if (!o.mode) { if (!d.hidden) d.hidden = true; continue; }
    if (d.hidden) d.hidden = false;
    d.classList.toggle('compact', o.mode === 'compact');
    d.style.transform = `translate(${Math.round(o.rect[0])}px, ${Math.round(o.rect[1])}px)`;
  }
}

// ---------------------------------------------------------------- camera poses (scene JSON)
// f.T is the served camera-to-scene pose (row-major 4 x 4, OpenCV axes: x right, y down, z forward).
function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }
function buildFrustums(cams, size) {
  if (!cams.length) return;
  const d = Math.max(0.05, size * (cams.length === 1 ? 0.05 : 0.025));
  const pos = [];
  for (const f of cams) {
    const T = poseMatrix(f);
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    for (let i = 0; i < 4; i++) { pos.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray()); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: srgb('#c9d1dc') }));
  lines.name = 'frustums';
  lines.frustumCulled = false;
  groups.cameras.add(lines);
}

// ---------------------------------------------------------------- cameras: list and go to
// The list shows each served camera centre (f.position, scene frame, metres). "Go to" moves the
// viewpoint to that centre, looking along the camera's optical axis with a field of view that
// shows the camera's whole image; the pose is the served one, only mapped into the display frame.
function cameraView(f) {
  const M = poseMatrix(f).premultiply(root.matrixWorld);  // camera → display frame
  const eye = new THREE.Vector3().setFromMatrixPosition(M);
  const fwd = new THREE.Vector3(0, 0, 1).transformDirection(M);
  const [fx, fy] = f.K; const [w, h] = f.size;
  // vertical field of view of the viewport that shows the camera's whole image
  const half = Math.max(h / (2 * fy), w / (2 * fx) / Math.max(camera.aspect, 1e-3));
  const fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(half)), 5, 150);
  // orbit pivot straight ahead: towards the middle of the scene, 0.5 m at least
  let dist = 1.0;
  if (!state.bbox.isEmpty()) {
    const ahead = state.bbox.getCenter(new THREE.Vector3()).sub(eye).dot(fwd);
    const size = state.bbox.getSize(new THREE.Vector3()).length();
    dist = THREE.MathUtils.clamp(ahead, 0.5, Math.max(0.5, size / 2));
  }
  return { eye, target: eye.clone().addScaledVector(fwd, dist), fov };
}
function placeView(eye, target, near, far) {
  camera.position.copy(eye);
  controls.target.copy(target);
  camera.near = near; camera.far = far;
  camera.updateProjectionMatrix();
  controls.enableDamping = false;  // no leftover orbit inertia moves the camera afterwards
  controls.update();
  controls.enableDamping = true;
}
function goToCamera(i) {
  const f = state.meta.cameras[i];
  if (!f) return;
  const to = cameraView(f);
  setFov(to.fov);
  placeView(to.eye, to.target, 0.01, Math.max(camera.far, 1000));
}
function fmtCoord(v) { return (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3); }
function buildCameraList() {
  const cams = state.meta.cameras;
  const tbody = $('#cameras tbody');
  const cs = state.scene.openlabel.coordinate_systems || {};
  const axes = cs.map?.axes?.replaceAll(',', ', ');
  $('#cam-note').textContent = state.meta.mode === 'map'
    ? `Camera centres in the map frame${axes ? ` (${axes})` : ''}, metres.`
    : 'Camera centre in the scene frame (the image\'s camera frame, OpenCV axes), metres.';
  cams.forEach((f, i) => {
    const [x, y, z] = f.position;
    const go = el('button', { type: 'button', class: 'goto', title: 'Move the viewpoint to this camera',
      'aria-label': `Go to camera ${f.name}` }, 'Go to');
    go.addEventListener('click', () => goToCamera(i));
    tbody.appendChild(el('tr', { 'data-index': i },
      el('td', { class: 'cam-name' }, el('span', {}, f.name),
        f.source && f.source !== f.name ? el('small', {}, f.source) : ''),
      el('td', { class: 'num' }, fmtCoord(x)), el('td', { class: 'num' }, fmtCoord(y)),
      el('td', { class: 'num' }, fmtCoord(z)), el('td', {}, go)));
  });
  if (!cams.length) tbody.append(el('tr', {}, el('td', { colspan: 5, class: 'muted' }, 'No cameras.')));
}

// ---------------------------------------------------------------- framing
// The initial view: a bird's-eye three-quarter view, HOME_PITCH below the horizon, of the whole
// scene (the points' 2nd-98th percentile box and the camera centres near it), from the south-west
// (HOME_AZIMUTH) so that the up axis stays vertical on screen.
const DEFAULT_FOV = 55;
const HOME_PITCH = 60, HOME_AZIMUTH = new THREE.Vector2(-1, -1.2).normalize();
// camera centres further than this many diagonals of the cloud's box from it do not widen the
// initial view (a keyframe posed far off would shrink the whole cloud to a dot); they are drawn
const HOME_CAMERA_REACH = 2;
function setFov(fov) {
  camera.fov = fov;
  camera.updateProjectionMatrix();
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
  state.labelsDirty = true;
  invalidate();
}
// the distance from `target` along `dir` (unit, target → eye) at which all corners of `box` lie
// inside the viewport, with `pad` of margin
function fitDistance(box, target, dir, pad = 1.06) {
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / pad;
  const tanH = tanV * camera.aspect;
  const fwd = dir.clone().negate();
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
  const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
  let dist = 0.1;
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const rel = new THREE.Vector3(x, y, z).sub(target);
    const along = rel.dot(dir);  // towards the eye
    dist = Math.max(dist, Math.abs(rel.dot(right)) / tanH + along, Math.abs(rel.dot(up)) / tanV + along);
  }
  return dist;
}
function resetView() {
  setFov(DEFAULT_FOV);
  // a page loaded without a size (a hidden tab or pane) has no aspect to fit to: it is framed on
  // the first resize that gives it one
  state.homePending = !(host.clientWidth > 0 && host.clientHeight > 0);
  if (state.homePending || state.bbox.isEmpty()) return;
  const box = state.bbox;
  const pitch = THREE.MathUtils.degToRad(HOME_PITCH);
  const dir = new THREE.Vector3(HOME_AZIMUTH.x * Math.cos(pitch), HOME_AZIMUTH.y * Math.cos(pitch), Math.sin(pitch));
  const target = box.getCenter(new THREE.Vector3());
  const dist = fitDistance(box, target, dir);
  const size = box.getSize(new THREE.Vector3()).length();
  placeView(target.clone().addScaledVector(dir, dist), target, Math.max(dist / 1000, 0.005),
    dist * 50 + size * 10);
}
window.__viewerResetView = resetView;  // for tests: back to the initial view
function robustBox(positions, matrix) {
  // 2nd-98th percentile bounds of a sample: far outliers (windows, sky) do not shrink the view
  const n = positions.length / 3;
  const step = Math.max(1, Math.floor(n / 50000));
  const xs = [], ys = [], zs = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    v.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]).applyMatrix4(matrix);
    xs.push(v.x); ys.push(v.y); zs.push(v.z);
  }
  const q = (a, f) => { a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(f * a.length))]; };
  return new THREE.Box3(new THREE.Vector3(q(xs, 0.02), q(ys, 0.02), q(zs, 0.02)),
    new THREE.Vector3(q(xs, 0.98), q(ys, 0.98), q(zs, 0.98)));
}

// ---------------------------------------------------------------- panel tabs
function showTab(name) {
  document.querySelectorAll('#tabs button').forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
}
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ---------------------------------------------------------------- Layers
function applyLayers() {
  for (const k of ['points', 'segments', 'cameras', 'obbs']) groups[k].visible = state.layers[k];
  groups.labels.visible = state.layers.labels;
  state.labelsDirty = true;
  invalidate();
}
function buildLayers() {
  for (const [k, name, help] of LAYERS) {
    const cb = el('input', { type: 'checkbox', id: `layer-${k}` });
    cb.checked = state.layers[k];
    cb.addEventListener('change', () => { state.layers[k] = cb.checked; applyLayers(); });
    const row = el('label', { class: 'row check', 'data-layer': k, for: `layer-${k}`, title: help }, cb,
      el('span', {}, name));
    if (k === 'cameras' && !state.meta.cameras.length) { cb.disabled = true; row.classList.add('disabled'); }
    $('#layers').append(row);
  }
}

// ---------------------------------------------------------------- Point cloud (attributes)
// One control per attribute in meta.controls (core.cloud_attrs: applicable to this scope, no
// PLY-only keys). Values are the -p strings; the server validates and derives.
function decimals(step) { const s = String(step); return s.includes('.') ? s.split('.')[1].length : 0; }
function isOff(c, v) { return c.off != null && (v === c.off || Number(v) === Number(c.off)); }
function formatValue(c, v) {
  if (c.kind !== 'int' && c.kind !== 'float') return v;
  if (isOff(c, v)) return c.off === 'inf' ? '∞' : 'off';
  return `${Number(v).toFixed(decimals(c.step))}${c.unit ? ` ${c.unit}` : ''}`;
}
const cloudControls = new Map();
function buildCloudControls() {
  for (const c of state.meta.controls) {
    const id = `attr-${c.key}`;
    const row = el('div', { class: 'row attr', 'data-attr': c.key, title: `${c.help} — ${c.values}` });
    const out = el('output', { for: id });
    let input;
    let set;  // (value) → update the widget
    if (c.kind === 'choice') {
      input = el('select', { id }, ...c.options.map((o) => el('option', { value: o }, o)));
      input.addEventListener('change', () => setAttr(c.key, input.value));
      set = (v) => { input.value = v; };
    } else if (c.kind === 'toggle') {
      input = el('input', { type: 'checkbox', id, role: 'switch', class: 'switch' });
      input.addEventListener('change', () => { out.value = input.checked ? 'on' : 'off'; setAttr(c.key, out.value); });
      set = (v) => { input.checked = v === 'on'; out.value = v; };
    } else if (c.kind === 'int' || c.kind === 'float') {
      input = el('input', { type: 'range', id, min: c.min, max: c.max, step: c.step });
      const read = () => {
        const x = Number(input.value);
        if (c.off === 'inf' && x >= Number(c.max)) return 'inf';
        return c.kind === 'int' ? String(Math.round(x)) : String(Number(x.toFixed(decimals(c.step))));
      };
      input.addEventListener('input', () => { const v = read(); out.value = formatValue(c, v); setAttr(c.key, v); });
      set = (v) => { input.value = v === 'inf' ? c.max : Number(v); out.value = formatValue(c, v); };
    } else {
      input = el('input', { type: 'text', id });
      input.addEventListener('change', () => setAttr(c.key, input.value.trim()));
      set = (v) => { input.value = v; };
    }
    set(state.attrs[c.key]);
    cloudControls.set(c.key, { c, row, input, set });
    row.append(el('label', { for: id }, c.key), input, out);
    $('#cloud-controls').append(row);
  }
}
function attrQuery() {
  return new URLSearchParams(state.meta.controls.map((c) => [c.key, state.attrs[c.key]])).toString();
}
function setAttr(key, value) {
  state.attrs[key] = value;
  attrsChanged();
}
function attrsChanged() {
  state.busy = true;
  clearTimeout(state.debounce);
  state.debounce = setTimeout(reloadCloud, DEBOUNCE_MS);
}
function showError(msg) {
  const box = $('#cloud-error');
  box.hidden = !msg;
  box.textContent = msg ? `Not updated: ${msg}` : '';
  const key = msg ? msg.split(/[\s(=]/)[0] : null;
  for (const { row, c } of cloudControls.values()) row.classList.toggle('invalid', c.key === key);
}
async function reloadCloud() {
  const seq = ++state.cloudSeq;
  state.abort?.abort();
  const ctl = new AbortController();
  state.abort = ctl;
  try {
    const buffer = await fetchBuffer(`/api/cloud?${attrQuery()}`, ctl.signal);
    if (seq !== state.cloudSeq) return;
    showCloud(parseCloud(buffer));
    showError(null);
  } catch (err) {
    if (err.name === 'AbortError' || seq !== state.cloudSeq) return;
    showError(err.message);
  } finally {
    if (seq === state.cloudSeq) state.busy = false;
  }
}

// ---------------------------------------------------------------- Catalogue
function buildCatalogue() {
  const tbody = $('#catalogue tbody');
  const rows = [...state.objects].sort((a, b) => b.volume - a.volume);
  for (const o of rows) {
    const [w, d, h] = o.dims;
    tbody.appendChild(el('tr', { 'data-id': o.id },
      el('td', {}, el('span', { class: 'swatch', style: `background:${o.hex}`, title: o.hex })),
      el('td', { class: 'num' }, String(o.id)), el('td', { class: 'label', title: o.label }, o.label),
      el('td', { class: 'num' }, o.score != null ? o.score.toFixed(2) : ''),
      el('td', { class: 'num dims' }, `${w.toFixed(2)}×${d.toFixed(2)}×${h.toFixed(2)}`),
      el('td', { class: 'num' }, o.volume.toFixed(3))));
  }
  if (!rows.length) tbody.append(el('tr', {}, el('td', { colspan: 6, class: 'muted' }, 'No objects.')));
}

// ---------------------------------------------------------------- main
async function main() {
  resize();
  state.meta = await fetchJSON('/api/meta');
  state.scene = await fetchJSON('/api/scene');
  document.title = `${state.meta.title} — oh-my-slam`;
  const T = new THREE.Matrix4().fromArray(state.meta.display_transform.flat()).transpose();
  root.matrixAutoUpdate = false;
  root.matrix.copy(T);
  root.updateMatrixWorld(true);
  buildObjects(state.scene);
  for (const c of state.meta.controls) state.attrs[c.key] = c.default;
  const buffer = await fetchBuffer(`/api/cloud?${attrQuery()}`);
  showCloud(parseCloud(buffer));
  const size = state.bbox.isEmpty() ? 1 : state.bbox.getSize(new THREE.Vector3()).length();
  buildFrustums(state.meta.cameras, size);
  const reach = state.bbox.isEmpty() ? null
    : state.bbox.clone().expandByScalar(HOME_CAMERA_REACH * state.bbox.getSize(new THREE.Vector3()).length());
  for (const f of state.meta.cameras) {
    const c = new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]).applyMatrix4(root.matrixWorld);
    if (!reach || reach.containsPoint(c)) state.bbox.expandByPoint(c);
  }
  if (state.meta.mode === 'image') $('#tab-catalogue-btn').hidden = false;
  if (state.meta.has_segmented) {
    $('#tab-image-btn').hidden = false;
    $('#segmented').src = '/api/segmented.png';
  }
  state.layers.cameras = state.meta.cameras.length > 0;
  buildLayers();
  buildCloudControls();
  if (state.meta.mode === 'image') buildCatalogue();
  buildCameraList();
  applyLayers();
  resize();
  resetView();
  $('#loading').classList.add('done');
  state.ready = true;
  invalidate();
}

// The viewpoint as last seen by the loop; a change beyond VIEW_EPS (1 µm, 1 µrad: far below a
// pixel) moves the labels and draws. Orbit damping's last creep stays below it, so a settled view
// stops drawing.
const VIEW_EPS = 1e-6;
const lastView = new THREE.Matrix4(), lastProj = new THREE.Matrix4();
function differs(a, b) {
  const x = a.elements, y = b.elements;
  for (let i = 0; i < 16; i++) if (Math.abs(x[i] - y[i]) > VIEW_EPS) return true;
  return false;
}
// <body data-rendered="true"> once a frame showing the point cloud has been drawn and presented:
// set in the animation frame after the first one that drew the loaded cloud; never removed.
let cloudDrawn = false;
function animate() {
  requestAnimationFrame(animate);
  if (cloudDrawn && !document.body.dataset.rendered) document.body.dataset.rendered = 'true';
  controls.update();
  camera.updateMatrixWorld();
  if (differs(lastView, camera.matrixWorld) || differs(lastProj, camera.projectionMatrix)) {
    lastView.copy(camera.matrixWorld);
    lastProj.copy(camera.projectionMatrix);
    state.labelsDirty = true;
    invalidate();
  }
  if (state.labelsDirty && state.ready) { state.labelsDirty = false; layoutLabels(); }
  if (redraw > 0) {
    redraw--;
    renderer.render(scene, camera);
    state.frames++;
    if (state.ready && state.cloudInScene) cloudDrawn = true;
  }
}
requestAnimationFrame(animate);
main().catch((err) => {
  console.error(err);
  document.body.dataset.error = err.message;
  $('#loading-title').textContent = `Failed to load: ${err.message}`;
});
