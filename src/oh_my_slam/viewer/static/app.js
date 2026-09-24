// oh-my-slam viewer: display only. Every point cloud is derived by the server (GET /api/cloud,
// the shared derivation of segmentation.cloud); objects, OBBs, colours and camera poses come from
// the scene JSON and /api/meta. Nothing is recomputed here.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
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

const LAYERS = [
  ['points', 'Point cloud'],
  ['segments', 'Segmentation (object colours)'],
  ['cameras', 'Camera poses'],
  ['labels', 'Labels'],
  ['obbs', 'Oriented boxes'],
];
const DEBOUNCE_MS = 250;
const state = {
  meta: null, scene: null, objects: [], objectRgb: new Map(), selected: null, hovered: null,
  layers: { points: true, segments: false, cameras: true, labels: true, obbs: true },
  display: { pointSize: 2.0, labelDensity: 12, background: '#15171c', normals: 'shade' },
  attrs: {},          // current point-cloud attributes, as the -p values the server parses
  cloud: null,        // header of the cloud on screen
  cloudSeq: 0, abort: null, debounce: null, framed: false, cloudInScene: false,
  bbox: new THREE.Box3(), ready: false,
};
window.__viewer = state; // for tests and debugging

// ---------------------------------------------------------------- renderer, scene, controls
const host = $('#canvas-host');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;   // material colours are converted back exactly
renderer.toneMapping = THREE.NoToneMapping;          // no tone mapping: colours stay exact
host.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
host.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(state.display.background);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
camera.up.set(0, 0, 1);  // map and display frames are z-up
camera.position.set(-3, -3.6, 2.7);  // until the cloud is framed (e.g. an empty map)
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;

const root = new THREE.Group();  // display transform (image mode: camera frame → z-up)
scene.add(root);
const groups = {};
for (const k of ['points', 'segments', 'cameras', 'obbs', 'labels', 'pick']) {
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
  labelRenderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  for (const o of state.objects) o.line.material.resolution.set(w, h);
  state.camHighlight?.material.resolution.set(w, h);
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- loading
const loadingList = $('#loading-list');
function loadRow(name) {
  const li = el('li', {}, el('span', {}, name), el('span', { class: 'pct' }, '…'));
  loadingList.appendChild(li);
  const pct = li.querySelector('.pct');
  return {
    progress(p) { pct.textContent = `${Math.round(p * 100)}%`; },
    ok(msg = 'done') { li.classList.add('ok'); pct.textContent = msg; },
    fail(msg) { li.classList.add('err'); pct.textContent = msg; },
  };
}
async function errorOf(res) {
  try { return (await res.json()).error || `HTTP ${res.status}`; } catch { return `HTTP ${res.status}`; }
}
async function fetchWithProgress(url, row, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) { const msg = await errorOf(res); row?.fail(`HTTP ${res.status}`); throw new Error(msg); }
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader();
  const buf = new Uint8Array(total || 0);
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (total && got + value.length <= total) buf.set(value, got); else chunks.push(value);
    got += value.length;
    if (total) row?.progress(got / total);
  }
  row?.ok(got > 1e6 ? `${(got / 1e6).toFixed(1)} MB` : 'done');
  if (!chunks.length && got === total) return buf.buffer;
  const all = new Uint8Array(got); let off = 0;  // no or wrong Content-Length
  if (total) { all.set(buf.subarray(0, Math.min(total, got))); off = Math.min(total, got); }
  for (const c of chunks) { all.set(c, off); off += c.length; }
  return all.buffer;
}
async function fetchJSON(url, name) {
  const buf = await fetchWithProgress(url, loadRow(name));
  return JSON.parse(new TextDecoder().decode(buf));
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
const SHADE = { off: 0, shade: 1, normals: 2 };
function pointMaterial({ color, normal, exact }) {
  const defines = {};
  if (color) defines.HAS_COLOR = '';
  if (normal && !exact) defines.HAS_NORMAL = '';
  return new THREE.ShaderMaterial({
    defines,
    uniforms: {
      size: { value: state.display.pointSize },  // point diameter in centimetres
      focal: { value: focalPx() },               // viewport focal length in pixels
      shade: { value: SHADE[state.display.normals] },
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
        if (shade == 1) {         // headlight shading by the derived normals
          vec3 n = normalize(normalMatrix * normal);
          c *= 0.3 + 0.7 * abs(dot(n, normalize(-mv.xyz)));
        } else if (shade == 2) {  // the normal itself, display frame (z up = blue)
          c = 0.5 * normalize(mat3(modelMatrix) * normal) + 0.5;
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
function ordinal(n) {
  const s = (n % 100 >= 11 && n % 100 <= 13) ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th');
  return `${n}${s}`;
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
  const pts = new THREE.Points(g, pointMaterial({ color: !!arrays.color, normal: !!arrays.normal }));
  pts.name = 'points';
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
    seg.renderOrder = 1;  // after the cloud: same positions, depth test passes (less-or-equal)
    groups.segments.add(seg);
  }
  state.cloud = header;
  state.cloudInScene = true;
  updateNormalsControl();
  const pos = arrays.position;
  if (!state.framed && pos.length) {
    state.framed = true;
    state.bbox.union(robustBox(pos, root.matrixWorld));
    if (state.meta.mode === 'image') {
      // look-at point: median depth straight ahead of the photo's camera (camera frame +z)
      const zs = [];
      for (let i = 2; i < pos.length; i += 30) zs.push(pos[i]);
      zs.sort((a, b) => a - b);
      const zmed = zs[Math.floor(zs.length / 2)] || 1;
      state.photoTarget = new THREE.Vector3(0, 0, zmed).applyMatrix4(root.matrixWorld);
    }
  }
  const shown = header.count.toLocaleString();
  const thin = header.step > 1
    ? ` of ${header.total.toLocaleString()} shown (every ${ordinal(header.step)}, for display)` : '';
  $('#cloud-status').textContent = `${shown} points${thin} · derived in ${Math.round(header.seconds * 1000)} ms`;
  updateStats();
}

// ---------------------------------------------------------------- OBBs, labels, picking
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
    const [x, y, z, qx, qy, qz, qw, sx, sy, sz] = cub.val;
    const pick = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }));
    pick.position.set(x, y, z);
    pick.quaternion.set(qx, qy, qz, qw);
    pick.userData.id = Number(id);
    groups.pick.add(pick);
    const div = el('div', { class: 'obj-label' }, `${o.type} ${id}`);
    div.style.borderLeftColor = hex;
    div.addEventListener('click', (e) => { e.stopPropagation(); select(Number(id), true); });
    const label = new CSS2DObject(div);
    const top = corners.reduce((m, p) => (p.z > m.z ? p : m), corners[0]);
    label.position.copy(top);
    groups.labels.add(label);
    state.objects.push({
      id: Number(id), label: o.type, hex, score: prop(o, 'num', 'score'), volume: attr(cub, 'volume_m3'),
      dims: [attr(cub, 'width_m'), attr(cub, 'depth_m'), attr(cub, 'height_m')],
      center: new THREE.Vector3(x, y, z), corners, line, pick, labelObj: label, div,
    });
  }
}

// ---------------------------------------------------------------- camera poses (scene JSON)
// f.T is the served camera-to-scene pose (row-major 4 x 4, OpenCV axes: x right, y down, z forward).
function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }
function buildFrustums(cams, size) {
  if (!cams.length) return;
  const d = Math.max(0.05, size * (cams.length === 1 ? 0.05 : 0.025));
  const pos = [];
  state.frustumSegments = [];
  for (const f of cams) {
    const T = poseMatrix(f);
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    const segs = [];
    for (let i = 0; i < 4; i++) { segs.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray()); }
    state.frustumSegments.push(segs);
    pos.push(...segs);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial(
    { color: srgb('#c9d1dc'), transparent: true, opacity: cams.length === 1 ? 0.6 : 0.85 }));
  lines.name = 'frustums';
  groups.cameras.add(lines);
}

// ---------------------------------------------------------------- cameras: list, highlight, go to
// The list shows each served camera centre (f.position, scene frame, metres). "Go to" flies the
// viewpoint to that centre, looking along the camera's optical axis with a field of view that
// shows the camera's whole image; the pose is the served one, only mapped into the display frame.
const FLIGHT_MS = 650;
function selectCamera(i) {
  state.cameraIndex = i;
  document.querySelectorAll('#cameras tbody tr[data-index]').forEach((tr) => {
    const on = Number(tr.dataset.index) === i;
    tr.classList.toggle('selected', on);
    tr.setAttribute('aria-selected', String(on));
    if (on) tr.scrollIntoView({ block: 'nearest' });
  });
  if (state.camHighlight) {
    groups.cameras.remove(state.camHighlight);
    state.camHighlight.geometry.dispose();
    state.camHighlight.material.dispose();
    state.camHighlight = null;
  }
  if (i == null || !state.frustumSegments) return;
  const geom = new LineSegmentsGeometry().setPositions(state.frustumSegments[i]);
  const mat = new LineMaterial({ color: srgb('#5b8cff'), linewidth: 3.0, worldUnits: false });
  mat.resolution.set(host.clientWidth, host.clientHeight);
  state.camHighlight = new LineSegments2(geom, mat);
  state.camHighlight.name = 'selected-camera';
  state.camHighlight.renderOrder = 2;
  groups.cameras.add(state.camHighlight);
}
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
function goToCamera(i) {
  const f = state.meta.cameras[i];
  if (!f) return;
  selectCamera(i);
  const to = cameraView(f);
  camera.near = 0.01;
  camera.far = Math.max(camera.far, 1000);
  state.flight = { from: { pos: camera.position.clone(), target: controls.target.clone(), fov: camera.fov },
    to, t0: performance.now() };
  state.flying = true;
  controls.enabled = false;
  controls.enableDamping = false;  // no leftover orbit inertia moves the camera after arrival
}
function stepFlight(now) {
  const f = state.flight;
  if (!f) return;
  const t = Math.min(1, (now - f.t0) / FLIGHT_MS);
  const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;  // ease in-out
  camera.position.lerpVectors(f.from.pos, f.to.eye, e);
  controls.target.lerpVectors(f.from.target, f.to.target, e);
  setFov(f.from.fov + (f.to.fov - f.from.fov) * e);
  if (t < 1) return;
  camera.position.copy(f.to.eye);
  controls.target.copy(f.to.target);
  endFlight();
}
function endFlight() {
  state.flight = null;
  state.flying = false;
  controls.enabled = true;
  controls.enableDamping = true;
}
function stepCamera(delta) {
  const rows = [...document.querySelectorAll('#cameras tbody tr[data-index]')].filter((tr) => !tr.hidden);
  if (!rows.length) return;
  const at = rows.findIndex((tr) => Number(tr.dataset.index) === state.cameraIndex);
  const next = at < 0 ? (delta > 0 ? 0 : rows.length - 1) : (at + delta + rows.length) % rows.length;
  goToCamera(Number(rows[next].dataset.index));
}
function fmtCoord(v) { return (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3); }
function buildCameraList() {
  const cams = state.meta.cameras;
  const tbody = $('#cameras tbody');
  $('#cam-count').textContent = cams.length ? String(cams.length) : '';
  const cs = state.scene.openlabel.coordinate_systems || {};
  const axes = cs.map?.axes?.replaceAll(',', ', ');
  $('#cam-note').textContent = state.meta.mode === 'map'
    ? `Camera centres in the map frame${axes ? ` (${axes})` : ''}, metres. Go to moves the view to a camera, looking where it looked.`
    : 'Camera centre in the scene frame (the image\'s camera frame, OpenCV axes), metres. Go to shows the scene from the photo\'s viewpoint.';
  cams.forEach((f, i) => {
    const [x, y, z] = f.position;
    const title = [f.source && `image ${f.source}`, `frame ${f.frame}`, f.update != null && `update ${f.update}`]
      .filter(Boolean).join(' · ');
    const go = el('button', { type: 'button', class: 'goto', title: 'Move the viewpoint to this camera',
      'aria-label': `Go to camera ${f.name}` }, 'Go to');
    go.addEventListener('click', (e) => { e.stopPropagation(); goToCamera(i); });
    const tr = el('tr', { 'data-index': i, title, 'aria-selected': 'false' },
      el('td', { class: 'cam-name' }, el('span', {}, f.name),
        f.source && f.source !== f.name ? el('small', {}, f.source) : ''),
      el('td', { class: 'num' }, fmtCoord(x)), el('td', { class: 'num' }, fmtCoord(y)),
      el('td', { class: 'num' }, fmtCoord(z)), el('td', {}, go));
    tr.addEventListener('click', () => selectCamera(state.cameraIndex === i ? null : i));
    tr.addEventListener('dblclick', () => goToCamera(i));
    tbody.appendChild(tr);
  });
  if (!cams.length) tbody.append(el('tr', {}, el('td', { colspan: 5, class: 'muted' }, 'No cameras.')));
  $('#cam-prev').disabled = $('#cam-next').disabled = !cams.length;
  $('#cam-prev').addEventListener('click', () => stepCamera(-1));
  $('#cam-next').addEventListener('click', () => stepCamera(1));
  $('#cam-filter').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    tbody.querySelectorAll('tr[data-index]').forEach((tr) => {
      const f = cams[Number(tr.dataset.index)];
      tr.hidden = q && !`${f.name} ${f.source || ''}`.toLowerCase().includes(q);
    });
  });
}

// ---------------------------------------------------------------- framing
const DEFAULT_FOV = 55;
function setFov(fov) {
  camera.fov = fov;
  camera.updateProjectionMatrix();
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
}
function frameBox(box, pad = 1.15) {
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const r = Math.max(sphere.radius, 0.05) * pad;
  const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  const dir = new THREE.Vector3(-1, -1.2, 0.9).normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, dist);
  camera.near = Math.max(dist / 1000, 0.005);
  camera.far = dist * 50 + r * 10;
  camera.updateProjectionMatrix();
  controls.update();
}
function resetView() {
  if (state.flight) endFlight();
  setFov(DEFAULT_FOV);
  if (state.meta && state.meta.mode === 'image' && state.photoTarget) {
    // image mode: start just behind the photo's own viewpoint, looking forward
    const eye = new THREE.Vector3(0, 0, 0).applyMatrix4(root.matrixWorld);
    controls.target.copy(state.photoTarget);
    camera.position.copy(eye).addScaledVector(state.photoTarget.clone().sub(eye).normalize(), -0.3);
    camera.near = 0.01; camera.far = 5000;
    camera.updateProjectionMatrix();
    controls.update();
    return;
  }
  if (!state.bbox.isEmpty()) frameBox(state.bbox);
}
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
function frameObject(o) {
  if (state.flight) endFlight();
  const box = new THREE.Box3().setFromPoints(o.corners.map((p) => p.clone().applyMatrix4(root.matrixWorld)));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const r = Math.max(sphere.radius, 0.1) * 2.2;
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)));
  controls.update();
}

// ---------------------------------------------------------------- selection, hover, labels
function select(id, frame = false) {
  state.selected = id;
  for (const o of state.objects) {
    const on = o.id === id;
    o.line.material.linewidth = on ? 5.0 : 2.0;  // thicker, never a different colour
    o.div.classList.toggle('selected', on);
  }
  document.querySelectorAll('#catalogue tbody tr').forEach((tr) => {
    const on = Number(tr.dataset.id) === id;
    tr.classList.toggle('selected', on);
    if (on) tr.scrollIntoView({ block: 'nearest' });
  });
  const o = state.objects.find((x) => x.id === id);
  if (o && frame) frameObject(o);
  updateLabels();
}
function updateLabels() {
  const show = state.layers.labels;
  groups.labels.visible = show;
  if (!show) return;
  const camPos = camera.position.clone();
  const ranked = state.objects
    .map((o) => ({ o, d: o.center.clone().applyMatrix4(root.matrixWorld).distanceTo(camPos) }))
    .sort((a, b) => (b.o.volume / (1 + b.d)) - (a.o.volume / (1 + a.d)));
  const keep = new Set(ranked.slice(0, state.display.labelDensity).map((r) => r.o.id));
  if (state.selected != null) keep.add(state.selected);
  if (state.hovered != null) keep.add(state.hovered);
  // declutter: drop labels whose screen positions collide with a higher-ranked label
  const placed = [];
  const w = host.clientWidth, h = host.clientHeight;
  for (const { o } of ranked) {
    let vis = keep.has(o.id);
    if (vis) {
      const p = o.labelObj.position.clone().applyMatrix4(root.matrixWorld).project(camera);
      const sx = (p.x + 1) / 2 * w, sy = (1 - p.y) / 2 * h;
      const pri = o.id === state.selected || o.id === state.hovered;
      if (!pri && placed.some(([x, y]) => Math.abs(x - sx) < 70 && Math.abs(y - sy) < 16)) vis = false;
      if (vis) placed.push([sx, sy]);
    }
    o.labelObj.visible = vis;
  }
}
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
function pickAt(ev) {
  if (!state.layers.obbs) return null;
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(groups.pick.children, false);
  if (!hits.length) return null;
  const volume = (hit) => state.objects.find((o) => o.id === hit.object.userData.id).volume;
  hits.sort((a, b) => volume(a) - volume(b));  // prefer the smallest box (inner objects)
  return hits[0].object.userData.id;
}
const tooltip = $('#tooltip');
renderer.domElement.addEventListener('pointermove', (ev) => {
  const id = pickAt(ev);
  if (id !== state.hovered) { state.hovered = id; updateLabels(); }
  if (id == null) { tooltip.hidden = true; renderer.domElement.style.cursor = ''; return; }
  const o = state.objects.find((x) => x.id === id);
  const [w, d, h] = o.dims;
  tooltip.replaceChildren(
    el('div', { class: 't-title' }, el('span', { class: 'sw', style: `background:${o.hex}` }), `${o.label} `,
      el('span', { class: 'muted' }, `#${o.id}`)),
    `score ${o.score != null ? o.score.toFixed(2) : '–'}`, el('br'),
    `W×D×H ${w.toFixed(2)} × ${d.toFixed(2)} × ${h.toFixed(2)} m`, el('br'),
    `volume ${o.volume.toFixed(3)} m³`);
  const r = host.getBoundingClientRect();
  tooltip.style.left = `${Math.max(4, Math.min(ev.clientX - r.left + 14, r.width - 190))}px`;
  tooltip.style.top = `${ev.clientY - r.top + 14}px`;
  tooltip.hidden = false;
  renderer.domElement.style.cursor = 'pointer';
});
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
renderer.domElement.addEventListener('pointerup', (ev) => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
  const id = pickAt(ev);
  if (id != null) { select(id, false); showTab('catalogue'); }
});

// ---------------------------------------------------------------- panel and tabs
function showTab(name) {
  document.querySelectorAll('#tabs button').forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  });
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
}
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
$('#toggle-panel').addEventListener('click', () => {
  const hidden = $('#app').classList.toggle('panel-hidden');
  $('#toggle-panel').setAttribute('aria-expanded', String(!hidden));
  setTimeout(resize, 200);
});
$('#reset-view').addEventListener('click', resetView);
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'Escape') { select(null); selectCamera(null); }
  if (e.key === '[') stepCamera(-1);
  if (e.key === ']') stepCamera(1);
});

// ---------------------------------------------------------------- Layers
function applyLayers() {
  for (const k of ['points', 'segments', 'cameras', 'obbs']) groups[k].visible = state.layers[k];
  groups.pick.visible = state.layers.obbs;
  if (!state.layers.obbs) tooltip.hidden = true;
  updateLabels();
}
function buildLayers() {
  const counts = { cameras: state.meta.cameras.length, labels: state.objects.length, obbs: state.objects.length };
  for (const [k, name] of LAYERS) {
    const cb = el('input', { type: 'checkbox', id: `layer-${k}` });
    cb.checked = state.layers[k];
    cb.addEventListener('change', () => { state.layers[k] = cb.checked; applyLayers(); });
    const text = k === 'cameras' && counts.cameras === 1 ? 'Camera pose' : name;
    const row = el('label', { class: 'row check', 'data-layer': k, for: `layer-${k}` }, cb,
      el('span', {}, text), el('span', { class: 'count' }, k in counts ? String(counts[k]) : ''));
    if (k === 'cameras' && !counts.cameras) { cb.disabled = true; row.classList.add('disabled'); }
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
  $('#cloud-defaults').addEventListener('click', () => {
    for (const c of state.meta.controls) { state.attrs[c.key] = c.default; cloudControls.get(c.key).set(c.default); }
    attrsChanged();
  });
  updateCli();
}
function attrQuery() {
  return new URLSearchParams(state.meta.controls.map((c) => [c.key, state.attrs[c.key]])).toString();
}
function updateCli() {
  const changed = state.meta.controls.filter((c) => state.attrs[c.key] !== c.default)
    .map((c) => `${c.key}=${state.attrs[c.key]}`);
  $('#cloud-cli').textContent = changed.length ? `-p ${changed.join(',')}` : 'default attributes';
}
function setAttr(key, value) {
  state.attrs[key] = value;
  attrsChanged();
}
function attrsChanged() {
  updateCli();
  setBusy(true);
  clearTimeout(state.debounce);
  state.debounce = setTimeout(reloadCloud, DEBOUNCE_MS);
}
function setBusy(on) {
  state.busy = on;
  $('#busy').hidden = !on;
  $('#cloud-busy').hidden = !on;
  $('#group-cloud').setAttribute('aria-busy', String(on));
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
    const buffer = await fetchWithProgress(`/api/cloud?${attrQuery()}`, null, ctl.signal);
    if (seq !== state.cloudSeq) return;
    showCloud(parseCloud(buffer));
    showError(null);
  } catch (err) {
    if (err.name === 'AbortError' || seq !== state.cloudSeq) return;
    showError(err.message);
  } finally {
    if (seq === state.cloudSeq) setBusy(false);
  }
}

// ---------------------------------------------------------------- Display
function rangeRow(label, id, min, max, step, value, fmt, onInput) {
  const input = el('input', { type: 'range', id, min, max, step });
  input.value = value;
  const out = el('output', { for: id }, fmt(value));
  input.addEventListener('input', () => { const v = Number(input.value); out.value = fmt(v); onInput(v); });
  return el('div', { class: 'row attr' }, el('label', { for: id }, label), input, out);
}
function updateNormalsControl() {
  const sel = $('#display-normals');
  if (!sel) return;
  const has = !!groups.points.children[0]?.geometry.getAttribute('normal');
  sel.disabled = !has;
  sel.closest('.row').title = has ? 'How the derived normals are shown' : 'Turn on normals under Point cloud';
}
function buildDisplay() {
  const host = $('#display');
  host.append(rangeRow('point size', 'display-size', 0.5, 8, 0.1, state.display.pointSize,
    (v) => `${Number(v).toFixed(1)} cm`, (v) => {
      state.display.pointSize = v;
      for (const m of pointMaterials()) m.uniforms.size.value = v;
    }));
  const sel = el('select', { id: 'display-normals' },
    el('option', { value: 'shade' }, 'shading'), el('option', { value: 'normals' }, 'normal colours'),
    el('option', { value: 'off' }, 'hidden'));
  sel.value = state.display.normals;
  sel.addEventListener('change', () => {
    state.display.normals = sel.value;
    for (const m of pointMaterials()) m.uniforms.shade.value = SHADE[sel.value];
  });
  host.append(el('div', { class: 'row attr' }, el('label', { for: 'display-normals' }, 'normals'), sel, el('output')));
  host.append(rangeRow('max labels', 'display-labels', 0, 60, 1, state.display.labelDensity,
    (v) => `${v}`, (v) => { state.display.labelDensity = v; updateLabels(); }));
  const bg = el('input', { type: 'color', id: 'display-bg' });
  bg.value = state.display.background;
  bg.addEventListener('input', () => { state.display.background = bg.value; scene.background.set(bg.value); });
  host.append(el('div', { class: 'row attr' }, el('label', { for: 'display-bg' }, 'background'), bg, el('output')));
  updateNormalsControl();
}

// ---------------------------------------------------------------- Catalogue
function buildCatalogue() {
  const tbody = $('#catalogue tbody');
  const rows = [...state.objects].sort((a, b) => b.volume - a.volume);
  $('#obj-count').textContent = String(rows.length);
  for (const o of rows) {
    const [w, d, h] = o.dims;
    const tr = el('tr', { 'data-id': o.id },
      el('td', {}, el('span', { class: 'swatch', style: `background:${o.hex}`, title: o.hex })),
      el('td', {}, String(o.id)), el('td', {}, o.label),
      el('td', {}, o.score != null ? o.score.toFixed(2) : ''),
      el('td', {}, `${w.toFixed(2)}×${d.toFixed(2)}×${h.toFixed(2)}`), el('td', {}, o.volume.toFixed(3)));
    tr.addEventListener('click', () => select(o.id, true));
    tr.addEventListener('mouseenter', () => { state.hovered = o.id; updateLabels(); });
    tr.addEventListener('mouseleave', () => { state.hovered = null; updateLabels(); });
    tbody.appendChild(tr);
  }
  if (!rows.length) tbody.append(el('tr', {}, el('td', { colspan: 6, class: 'muted' }, 'No objects.')));
  $('#filter').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    tbody.querySelectorAll('tr[data-id]').forEach((tr) => {
      tr.hidden = q && !tr.children[2].textContent.toLowerCase().includes(q);
    });
  });
}
function updateStats() {
  const s = state.meta.stats;
  const pts = state.cloud ? state.cloud.count : 0;
  $('#stats').textContent = `${pts.toLocaleString()} points · ${s.objects} objects · ${s.frames} frame${s.frames === 1 ? '' : 's'}`;
}

// ---------------------------------------------------------------- main
async function main() {
  resize();
  state.meta = await fetchJSON('/api/meta', 'metadata');
  state.scene = await fetchJSON('/api/scene', 'scene description');
  $('#title').textContent = `${state.meta.mode === 'map' ? 'Map' : 'Image'} · ${state.meta.title}`;
  document.title = `${state.meta.title} — oh-my-slam`;
  const T = new THREE.Matrix4().fromArray(state.meta.display_transform.flat()).transpose();
  root.matrixAutoUpdate = false;
  root.matrix.copy(T);
  root.updateMatrixWorld(true);
  buildObjects(state.scene);
  for (const c of state.meta.controls) state.attrs[c.key] = c.default;
  const buffer = await fetchWithProgress(`/api/cloud?${attrQuery()}`, loadRow('point cloud'));
  showCloud(parseCloud(buffer));
  const size = state.bbox.isEmpty() ? 1 : state.bbox.getSize(new THREE.Vector3()).length();
  buildFrustums(state.meta.cameras, size);
  for (const f of state.meta.cameras) state.bbox.expandByPoint(new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]).applyMatrix4(root.matrixWorld));
  if (state.meta.has_segmented) {
    $('#tab-image-btn').hidden = false;
    $('#segmented').src = '/api/segmented.png';
  }
  state.layers.cameras = state.meta.cameras.length > 0;
  buildLayers();
  buildCloudControls();
  buildDisplay();
  buildCatalogue();
  buildCameraList();
  applyLayers();
  resize();
  resetView();
  $('#loading').classList.add('done');
  state.ready = true;
}

// <body data-rendered="true"> once a frame showing the point cloud has been drawn and presented:
// set in the animation frame after the first one that rendered the loaded cloud; never removed.
let cloudFrames = 0;
let lastLabelUpdate = 0;
function animate(t) {
  requestAnimationFrame(animate);
  stepFlight(performance.now());
  controls.update();
  if (t - lastLabelUpdate > 150) { lastLabelUpdate = t; if (state.objects.length) updateLabels(); }
  if (state.ready && state.cloudInScene && !document.body.dataset.rendered && ++cloudFrames > 1) {
    document.body.dataset.rendered = 'true';
  }
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
requestAnimationFrame(animate);
main().catch((err) => {
  console.error(err);
  document.body.dataset.error = err.message;
  $('#loading-title').textContent = `Failed to load: ${err.message}`;
});
