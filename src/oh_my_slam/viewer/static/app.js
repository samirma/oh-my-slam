// oh-my-slam viewer: display only. Every point cloud is derived by the server (GET /api/cloud,
// the shared derivation of segmentation.cloud); objects, OBBs, colours and camera poses come from
// the scene JSON and /api/meta. Nothing is recomputed here.
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
  ['segments', 'Segmentation overlay',
    'Only the points of each object, in the object\'s colour, drawn over the point cloud whatever its color attribute'],
  ['cameras', 'Camera poses', 'A frustum at each camera\'s pose; frustums next to the viewpoint fade out'],
  ['labels', 'Labels',
    'Every box gets its id tag in its colour, or a "+N" chip where tags are crowded; names where they fit'],
  ['obbs', 'Oriented boxes', 'The objects\' oriented bounding boxes, in their colours'],
];
// the color attribute's options, as the menu words them (the value stays the -p value)
const COLOR_WORDS = {
  rgb: 'rgb · image colours', segment: 'segment · object colours',
  height: 'height · up-axis ramp', none: 'none · plain',
};
const DEBOUNCE_MS = 250;
const state = {
  meta: null, scene: null, objects: [], objectRgb: new Map(), selected: null, hovered: null, peek: null,
  chips: [], leaders: [], cluster: null,  // "+N" chips, leader lines, the chip whose list is open
  layers: { points: true, segments: false, cameras: true, labels: true, obbs: true },
  display: { pointSize: 2.0, labels: 'fit', background: '#15171c', normals: 'shade' },
  attrs: {},          // current point-cloud attributes, as the -p values the server parses
  cloud: null,        // header of the cloud on screen
  cloudSeq: 0, abort: null, debounce: null, framed: false, cloudInScene: false,
  bbox: new THREE.Box3(), pointsBox: new THREE.Box3(), ready: false, labelsDirty: true,
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
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  for (const o of state.objects) o.line.material.resolution.set(w, h);
  state.camHighlight?.material.resolution.set(w, h);
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
  closeCluster();
  state.labelsDirty = true;
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
  updateNormalsControl();
  updateLayerNotes();
  const pos = arrays.position;
  if (!state.framed && pos.length) {
    state.framed = true;
    state.pointsBox = robustBox(pos, root.matrixWorld);
    state.bbox.union(state.pointsBox);
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
    ? ` of ${header.total.toLocaleString()} shown (every ${ordinal(header.step)}: display limit ${state.meta.max_points.toLocaleString()})` : '';
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
    const [x, y, z, qx, qy, qz, qw, sx, sy, sz] = cub.val;
    const pick = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }));
    pick.position.set(x, y, z);
    pick.quaternion.set(qx, qy, qz, qw);
    pick.userData.id = Number(id);
    groups.pick.add(pick);
    // label: the id tag in the object's colour, then the name; anchored at the top face's centre
    const tag = el('span', { class: 'tag' }, String(id));
    tag.style.background = hex;
    const ink = inkFor(rgb || [128, 128, 128]);
    tag.style.color = ink;
    const div = el('div', { class: 'obj-label', 'data-id': id, title: `${o.type} ${id}` }, tag,
      el('span', { class: 'name' }, o.type));
    div.addEventListener('click', (e) => { e.stopPropagation(); select(Number(id), true); });
    div.addEventListener('pointerenter', () => setHover(Number(id)));
    div.addEventListener('pointerleave', () => { if (state.hovered === Number(id)) setHover(null); });
    labelLayer.appendChild(div);
    const display = corners.map((p) => p.clone().applyMatrix4(root.matrixWorld));
    display.sort((a, b) => b.z - a.z);
    const anchor = display.slice(0, 4).reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(0.25);
    const inv = new THREE.Quaternion(qx, qy, qz, qw).invert();
    state.objects.push({
      id: Number(id), label: o.type, hex, ink, score: prop(o, 'num', 'score'), volume: attr(cub, 'volume_m3'),
      dims: [attr(cub, 'width_m'), attr(cub, 'depth_m'), attr(cub, 'height_m')],
      center: new THREE.Vector3(x, y, z), half: new THREE.Vector3(sx / 2, sy / 2, sz / 2), inv,
      corners, line, pick, div, anchor, size: Math.cbrt(Math.max(sx * sy * sz, 1e-9)),
      w: 0, wTag: 0, h: 0, mode: null, chip: null, rect: null, sx: 0, sy: 0,
    });
  }
}

// ---------------------------------------------------------------- camera poses (scene JSON)
// f.T is the served camera-to-scene pose (row-major 4 x 4, OpenCV axes: x right, y down, z forward).
// Each frustum fades out as the viewpoint comes near it (within FADE_NEAR x its depth: hidden;
// beyond FADE_FAR x: fully drawn), so the cameras next to the eye — the one being looked through
// and its neighbours in a capture that turns in place — never draw lines across the view.
const FADE_NEAR = 2.0, FADE_FAR = 5.0;
function poseMatrix(f) { return new THREE.Matrix4().fromArray(f.T.flat()).transpose(); }
function frustumFade(dist) {
  const d = state.frustumDepth || 1;
  return THREE.MathUtils.smoothstep(dist, FADE_NEAR * d, FADE_FAR * d);
}
// how much of camera i's frustum is drawn from the current viewpoint (0 hidden … 1 full); tests
window.__viewerFrustumFade = (i) =>
  frustumFade(camera.position.distanceTo(state.frustumCentres[i].clone().applyMatrix4(root.matrixWorld)));
function buildFrustums(cams, size) {
  if (!cams.length) return;
  const d = Math.max(0.05, size * (cams.length === 1 ? 0.05 : 0.025));
  state.frustumDepth = d;
  const pos = [], centre = [];
  state.frustumSegments = [];
  state.frustumCentres = [];
  for (const f of cams) {
    const T = poseMatrix(f);
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    const segs = [];
    for (let i = 0; i < 4; i++) { segs.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray()); }
    const mid = new THREE.Vector3(0, 0, d / 2).applyMatrix4(T);  // the frustum's middle
    state.frustumSegments.push(segs);
    state.frustumCentres.push(mid);
    pos.push(...segs);
    for (let k = 0; k < 16; k++) centre.push(mid.x, mid.y, mid.z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('centre', new THREE.Float32BufferAttribute(centre, 3));
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      color: { value: new THREE.Vector3(0xc9 / 255, 0xd1 / 255, 0xdc / 255) },  // sRGB, as written
      opacity: { value: cams.length === 1 ? 0.6 : 0.85 },
      fadeNear: { value: FADE_NEAR * d }, fadeFar: { value: FADE_FAR * d },
    },
    vertexShader: `
      attribute vec3 centre;
      uniform float fadeNear; uniform float fadeFar;
      varying float vFade;
      void main() {
        vec3 c = (modelMatrix * vec4(centre, 1.0)).xyz;
        vFade = smoothstep(fadeNear, fadeFar, distance(cameraPosition, c));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform vec3 color; uniform float opacity;
      varying float vFade;
      void main() { if (vFade <= 0.001) discard; gl_FragColor = vec4(color, opacity * vFade); }`,
    transparent: true, depthWrite: false,
  });
  const lines = new THREE.LineSegments(g, mat);
  lines.name = 'frustums';
  lines.frustumCulled = false;
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
  const mat = new LineMaterial({ color: srgb('#5b8cff'), linewidth: 3.0, worldUnits: false, transparent: true });
  mat.resolution.set(host.clientWidth, host.clientHeight);
  state.camHighlight = new LineSegments2(geom, mat);
  state.camHighlight.name = 'selected-camera';
  state.camHighlight.renderOrder = 2;
  groups.cameras.add(state.camHighlight);
  fadeHighlight();
}
// the selected camera's highlight fades like its frustum: hidden while looking through it
function fadeHighlight() {
  const hl = state.camHighlight;
  if (!hl) return;
  const c = state.frustumCentres[state.cameraIndex].clone().applyMatrix4(root.matrixWorld);
  const a = frustumFade(camera.position.distanceTo(c));
  hl.material.opacity = a;
  hl.visible = a > 0.001;
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
// Map overview: a bird's-eye three-quarter view, HOME_PITCH below the horizon, of the whole scene
// (the points' 2nd-98th percentile box and every camera centre). Steep enough that the walls of a
// room hide little of its floor and objects and the camera cluster shows inside it; from the
// south-west (HOME_AZIMUTH) so that the up axis stays vertical on screen.
const HOME_PITCH = 60, HOME_AZIMUTH = new THREE.Vector2(-1, -1.2).normalize();
function setFov(fov) {
  camera.fov = fov;
  camera.updateProjectionMatrix();
  const focal = focalPx();
  for (const m of pointMaterials()) m.uniforms.focal.value = focal;
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
function homeView(box) {
  const pitch = THREE.MathUtils.degToRad(HOME_PITCH);
  const dir = new THREE.Vector3(HOME_AZIMUTH.x * Math.cos(pitch), HOME_AZIMUTH.y * Math.cos(pitch), Math.sin(pitch));
  const target = box.getCenter(new THREE.Vector3());
  const dist = fitDistance(box, target, dir);
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, dist);
  const size = box.getSize(new THREE.Vector3()).length();
  camera.near = Math.max(dist / 1000, 0.005);
  camera.far = dist * 50 + size * 10;
  camera.updateProjectionMatrix();
  controls.update();
}
function resetView() {
  if (state.flight) endFlight();
  setFov(DEFAULT_FOV);
  if (state.meta && state.meta.mode === 'image' && state.photoTarget) {
    // image mode: start just behind the photo's own viewpoint, looking forward — close enough
    // (at most one frustum depth) that the photo's own frustum is faded out, not drawn across it
    const eye = new THREE.Vector3(0, 0, 0).applyMatrix4(root.matrixWorld);
    const back = Math.min(0.3, state.frustumDepth || 0.3);
    controls.target.copy(state.photoTarget);
    camera.position.copy(eye).addScaledVector(state.photoTarget.clone().sub(eye).normalize(), -back);
    camera.near = 0.01; camera.far = 5000;
    camera.updateProjectionMatrix();
    controls.update();
    return;
  }
  if (!state.bbox.isEmpty()) homeView(state.bbox);
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
// Box outlines: the selected box thickest, the hovered one (in 3D, its tag or its catalogue row)
// and the one peeked at in a chip's list thicker; never a different colour.
function lineWidths() {
  for (const o of state.objects) {
    const w = o.id === state.selected ? 5.0 : (o.id === state.hovered || o.id === state.peek ? 3.5 : 2.0);
    if (o.line.material.linewidth !== w) o.line.material.linewidth = w;
  }
}
function select(id, frame = false) {
  state.selected = id;
  for (const o of state.objects) o.div.classList.toggle('selected', o.id === id);
  lineWidths();
  document.querySelectorAll('#catalogue tbody tr').forEach((tr) => {
    const on = Number(tr.dataset.id) === id;
    tr.classList.toggle('selected', on);
    if (on) tr.scrollIntoView({ block: 'nearest' });
  });
  const o = state.objects.find((x) => x.id === id);
  if (o && frame) frameObject(o);
  fadeBoxes();
  closeCluster();
  state.labelsDirty = true;
}
// Hovering thickens the box and shows its label's name over the others (or lights up the chip
// that holds its tag) without moving any label.
function setHover(id) {
  if (id === state.hovered) return;
  const mark = (x, on) => {
    const o = state.objects.find((q) => q.id === x);
    o?.div.classList.toggle('hover', on);
    o?.chip?.div?.classList.toggle('hover', on);
  };
  mark(state.hovered, false);
  state.hovered = id;
  mark(id, true);
  lineWidths();
}
function setPeek(id) {
  if (id === state.peek) return;
  state.peek = id;
  lineWidths();
  if (state.cluster) drawClusterLeaders();
}

// Labels (spec §2.5: labelled OBBs). No label covers another label, the header or the help line,
// and none leaves the view:
// 1. the selected box first, with its name;
// 2. every other box whose top is in view, by apparent size (largest first): its id tag in its
//    colour next to the top face's centre (above, right, left or below it) or, where those are
//    taken, on rings farther out, joined to that point by a leader line in its colour;
// 3. the tags that fit nowhere near their box go into a "+N" chip placed there (farther out if
//    need be), one per neighbourhood of CLUSTER_RADIUS; a chip of a single box becomes that box's
//    tag, with a leader. Hovering or clicking a chip lists its boxes (id tag and name; each entry
//    selects its box) and draws a line from the chip to each of them;
// 4. names are added to the tags, largest boxes first, wherever the longer label still fits.
// Whatever the layout, hovering a box shows its id and label, and its catalogue row selects it.
const LABEL_GAP = 2;        // px between two labels, and between a label and the view's edge
const OBSTACLE_PAD = 4;     // px kept clear around the header and the help line
const LEADER_MIN = 5;       // px: a label farther than this from its box gets a leader line
const CLUSTER_RADIUS = 60;  // px: a tag that fits nowhere joins a chip started this close
const TAG_RINGS = [22, 34, 48, 64];  // px from the anchor to the label's centre
const CHIP_RINGS = [...TAG_RINGS, 84, 110, 145, 190, 250, 330, 430];
const DIRS = [0, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6].map((k) => {  // from straight up, both ways
  const a = -Math.PI / 2 + (k * Math.PI) / 6;
  return [Math.cos(a), Math.sin(a)];
});
const CELL = 64;
// the rectangles taken so far ([x0, y0, x1, y1] in px), bucketed on a coarse grid
class RectIndex {
  constructor() { this.cells = new Map(); }
  add(r) {
    for (let i = Math.floor(r[0] / CELL); i <= Math.floor(r[2] / CELL); i++) {
      for (let j = Math.floor(r[1] / CELL); j <= Math.floor(r[3] / CELL); j++) {
        const k = (i + 64) * 4096 + j + 64;
        const list = this.cells.get(k);
        if (!list) this.cells.set(k, [r]); else if (!list.includes(r)) list.push(r);
      }
    }
  }
  hits(x0, y0, x1, y1, ignore) {
    for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++) {
      for (let j = Math.floor(y0 / CELL); j <= Math.floor(y1 / CELL); j++) {
        const list = this.cells.get((i + 64) * 4096 + j + 64);
        if (!list) continue;
        for (const r of list) if (r !== ignore && x0 < r[2] && r[0] < x1 && y0 < r[3] && r[1] < y1) return true;
      }
    }
    return false;
  }
}
// the header (title card and buttons) and the help line, in the label layer's pixels
function obstacles() {
  const at = host.getBoundingClientRect();
  const out = [];
  for (const sel of ['#info', '#actions', '#help']) {
    const e = $(sel);
    const r = e && e.offsetParent ? e.getBoundingClientRect() : null;
    if (r && r.width && r.height) {
      out.push([r.left - at.left - OBSTACLE_PAD, r.top - at.top - OBSTACLE_PAD,
        r.right - at.left + OBSTACLE_PAD, r.bottom - at.top + OBSTACLE_PAD]);
    }
  }
  return out;
}
function measureLabels() {
  const width = (e) => Math.ceil(e.getBoundingClientRect().width);
  for (const o of state.objects) {
    o.div.classList.remove('compact');
    o.w = width(o.div); o.h = Math.ceil(o.div.getBoundingClientRect().height);
    o.wTag = width(o.div.firstChild);
  }
  const probe = chipAt(0);  // room for up to three digits
  probe.button.textContent = '+999';
  probe.key = null;
  probe.div.hidden = false;
  state.chipSize = [width(probe.div), Math.ceil(probe.div.getBoundingClientRect().height)];
  state.measured = state.objects.every((o) => o.w > 0) && state.chipSize[0] > 0;
}
function layoutLabels() {
  const show = state.layers.labels;
  groups.labels.visible = show;
  labelLayer.hidden = !show;
  if (!show || !state.objects.length) return;
  if (!state.measured) measureLabels();
  const W = host.clientWidth, H = host.clientHeight;
  const taken = new RectIndex();
  for (const r of obstacles()) taken.add(r);
  const free = (x, y, bw, bh, ignore = null) => x >= LABEL_GAP && y >= LABEL_GAP
    && x + bw <= W - LABEL_GAP && y + bh <= H - LABEL_GAP
    && !taken.hits(x - LABEL_GAP, y - LABEL_GAP, x + bw + LABEL_GAP, y + bh + LABEL_GAP, ignore);
  const clampX = (x, bw) => Math.min(Math.max(x, LABEL_GAP), W - bw - LABEL_GAP);
  const clampY = (y, bh) => Math.min(Math.max(y, LABEL_GAP), H - bh - LABEL_GAP);
  // the first free bw x bh place around the anchor (sx, sy): next to it, then on the rings
  const spot = (sx, sy, bw, bh, rings) => {
    for (const [x0, y0] of [[sx - bw / 2, sy - bh - 3], [sx + 4, sy - bh / 2], [sx - bw - 4, sy - bh / 2],
      [sx - bw / 2, sy + 3]]) {
      const x = clampX(x0, bw), y = clampY(y0, bh);
      if (free(x, y, bw, bh)) return [x, y];
    }
    for (const r of rings) {
      for (const [dx, dy] of DIRS) {
        const x = clampX(sx + dx * r - bw / 2, bw), y = clampY(sy + dy * r - bh / 2, bh);
        if (free(x, y, bw, bh)) return [x, y];
      }
    }
    return null;
  };
  const eye = camera.position;
  const p = new THREE.Vector3();
  const boxes = [];  // the boxes whose top is in view, largest on screen first
  for (const o of state.objects) {
    o.mode = null; o.chip = null;
    p.copy(o.anchor).project(camera);
    const sx = (p.x + 1) / 2 * W, sy = (1 - p.y) / 2 * H;
    if (!(p.z < 1 && p.z > -1 && sx >= 0 && sx <= W && sy >= 0 && sy <= H)) continue;
    o.sx = sx; o.sy = sy;
    o.rank = o.size / Math.max(eye.distanceTo(o.anchor), 1e-3);
    boxes.push(o);
  }
  boxes.sort((a, b) => (b.rank - a.rank) || (a.id - b.id));
  const labelled = [];
  const put = (o, [x, y], bw, mode) => {
    o.rect = [x, y, x + bw, y + o.h];
    o.mode = mode;
    taken.add(o.rect);
    labelled.push(o);
  };
  // 1. the selected box, with its name
  const sel = boxes.find((o) => o.id === state.selected);
  if (sel) {
    const full = spot(sel.sx, sel.sy, sel.w, sel.h, TAG_RINGS);
    const tag = full ? null : spot(sel.sx, sel.sy, sel.wTag, sel.h, TAG_RINGS);
    if (full) put(sel, full, sel.w, 'full');
    else if (tag) put(sel, tag, sel.wTag, 'compact');
    else put(sel, [clampX(sel.sx - sel.w / 2, sel.w), clampY(sel.sy - sel.h - 3, sel.h)], sel.w, 'full');
  }
  // 2. an id tag next to every other box
  const loose = [];
  for (const o of boxes) {
    if (o === sel) continue;
    const at = spot(o.sx, o.sy, o.wTag, o.h, TAG_RINGS);
    if (at) put(o, at, o.wTag, 'compact'); else loose.push(o);
  }
  // 3. the tags that fit nowhere near their box: "+N" chips
  const chips = [];
  const [cw, chh] = state.chipSize;
  const nearest = (o, limit) => {
    let best = null, d0 = limit;
    for (const c of chips) { const d = Math.hypot(c.hx - o.sx, c.hy - o.sy); if (d <= d0) { d0 = d; best = c; } }
    return best;
  };
  for (const o of loose) {
    let chip = nearest(o, CLUSTER_RADIUS);
    if (!chip) {
      const at = spot(o.sx, o.sy, cw, chh, CHIP_RINGS);
      if (at) {
        chip = { hx: o.sx, hy: o.sy, rect: [at[0], at[1], at[0] + cw, at[1] + chh], members: [] };
        taken.add(chip.rect);
        chips.push(chip);
      } else {
        chip = nearest(o, Infinity);  // no room left near it: the nearest chip anywhere
      }
    }
    if (chip) { chip.members.push(o); o.mode = 'cluster'; o.chip = chip; }
  }
  // a chip of one box is its tag, in the chip's place (which is free and no smaller than the tag)
  const lone = chips.filter((c) => c.members.length === 1 && c.members[0].wTag <= cw && c.members[0].h <= chh);
  for (const c of lone) {
    const o = c.members[0];
    c.rect[2] = c.rect[0] + o.wTag; c.rect[3] = c.rect[1] + o.h;
    o.rect = c.rect; o.mode = 'compact'; o.chip = null;
    labelled.push(o);
  }
  chips.splice(0, chips.length, ...chips.filter((c) => !lone.includes(c)));
  // 4. names, wherever the longer label still fits (growing rightwards, else leftwards)
  if (state.display.labels === 'fit') {
    for (const o of labelled) {
      if (o.mode !== 'compact') continue;
      const r = o.rect;
      for (const x of [r[0], r[2] - o.w]) {
        if (free(x, r[1], o.w, o.h, r)) { r[0] = x; r[2] = x + o.w; taken.add(r); o.mode = 'full'; break; }
      }
    }
  }
  state.chips = chips;
  drawLabels(labelled);
}
window.__viewerLayoutLabels = layoutLabels;  // for tests and timing
function drawLabels(labelled) {
  for (const o of state.objects) {
    const d = o.div;
    if (o.mode !== 'full' && o.mode !== 'compact') { if (!d.hidden) d.hidden = true; continue; }
    if (d.hidden) d.hidden = false;
    d.classList.toggle('compact', o.mode === 'compact');
    d.style.transform = `translate(${Math.round(o.rect[0])}px, ${Math.round(o.rect[1])}px)`;
    d.style.zIndex = o.id === state.selected ? '3' : '2';
  }
  state.chips.forEach((c, i) => {
    const e = chipAt(i);
    c.div = e.div; c.button = e.button;
    const key = c.members.map((o) => o.id).join(',');
    if (e.key !== key) {
      e.key = key;
      const n = c.members.length;
      e.button.textContent = `+${n}`;
      const names = c.members.slice(0, 12).map((o) => `${o.id} ${o.label}`).join(', ');
      e.button.title = `${n} more object${n === 1 ? '' : 's'} here: ${names}${n > 12 ? ', …' : ''}`;
      e.button.setAttribute('aria-label', e.button.title);
    }
    e.div.classList.toggle('hover', c.members.some((o) => o.id === state.hovered));
    if (e.div.hidden) e.div.hidden = false;
    e.div.style.transform = `translate(${Math.round(c.rect[0])}px, ${Math.round(c.rect[1])}px)`;
  });
  for (let i = state.chips.length; i < chipPool.length; i++) if (!chipPool[i].div.hidden) chipPool[i].div.hidden = true;
  // a leader from the box (its top face's centre) to the nearest point of its label
  state.leaders = [];
  for (const o of labelled) {
    const [x0, y0, x1, y1] = o.rect;
    const px = Math.min(Math.max(o.sx, x0), x1), py = Math.min(Math.max(o.sy, y0), y1);
    if (Math.hypot(px - o.sx, py - o.sy) > LEADER_MIN) state.leaders.push([o.sx, o.sy, px, py, o.hex, false]);
  }
  drawLeaders(state.leaders);
}

// leader lines: one dark halo path under all of them, then a line and a dot at the box per leader
const SVG_NS = 'http://www.w3.org/2000/svg';
const leaderSvg = document.createElementNS(SVG_NS, 'svg');
leaderSvg.id = 'leaders';
leaderSvg.setAttribute('aria-hidden', 'true');
const leaderHalo = document.createElementNS(SVG_NS, 'path');
leaderHalo.setAttribute('class', 'halo');
leaderSvg.append(leaderHalo);
labelLayer.appendChild(leaderSvg);
const leaderPool = [];
function drawLeaders(list) {
  let d = '';
  list.forEach(([ax, ay, bx, by, hex, strong], i) => {
    if (i === leaderPool.length) {
      const line = document.createElementNS(SVG_NS, 'line');
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('r', '2.5');
      leaderSvg.append(line, dot);
      leaderPool.push({ line, dot });
    }
    const { line, dot } = leaderPool[i];
    const [a, b, c, e] = [ax, ay, bx, by].map((v) => v.toFixed(1));
    line.setAttribute('x1', a); line.setAttribute('y1', b); line.setAttribute('x2', c); line.setAttribute('y2', e);
    line.setAttribute('stroke', hex);
    line.classList.toggle('strong', strong);
    dot.setAttribute('cx', a); dot.setAttribute('cy', b); dot.setAttribute('fill', hex);
    line.style.display = dot.style.display = '';
    d += `M${a} ${b}L${c} ${e}`;
  });
  leaderHalo.setAttribute('d', d);
  for (let i = list.length; i < leaderPool.length; i++) leaderPool[i].line.style.display = leaderPool[i].dot.style.display = 'none';
}

// "+N" chips (a pool, reused by index) and the list one of them opens
const chipPool = [];
function chipAt(i) {
  while (chipPool.length <= i) {
    const k = chipPool.length;
    const button = el('button', { type: 'button', class: 'chip', 'aria-haspopup': 'menu', 'aria-expanded': 'false' });
    const div = el('div', { class: 'obj-cluster' }, button);
    button.addEventListener('click', (e) => {
      e.stopPropagation();
      const c = state.chips[k];
      if (!c) return;
      if (state.cluster?.chip === c && state.cluster.pinned) closeCluster(); else openCluster(c, true);
    });
    div.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse' && state.chips[k]) openCluster(state.chips[k], false); });
    div.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') scheduleCloseCluster(); });
    labelLayer.appendChild(div);
    chipPool.push({ div, button, key: null });
  }
  return chipPool[i];
}
const clusterPop = el('div', { id: 'cluster-pop', role: 'menu', 'aria-label': 'Objects under this chip' });
clusterPop.hidden = true;
clusterPop.addEventListener('pointerenter', () => clearTimeout(state.clusterTimer));
clusterPop.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') scheduleCloseCluster(); });
labelLayer.appendChild(clusterPop);
function clusterEntry(o) {
  const tag = el('span', { class: 'tag' }, String(o.id));
  tag.style.background = o.hex;
  tag.style.color = o.ink;
  const b = el('button', { type: 'button', role: 'menuitem', 'data-id': o.id, title: `Select ${o.label} ${o.id}` },
    tag, el('span', { class: 'name' }, o.label));
  b.addEventListener('click', (e) => { e.stopPropagation(); select(o.id, true); });
  b.addEventListener('pointerenter', () => setPeek(o.id));
  b.addEventListener('pointerleave', () => setPeek(null));
  b.addEventListener('focus', () => setPeek(o.id));
  b.addEventListener('blur', () => setPeek(null));
  return b;
}
function openCluster(chip, pinned) {
  clearTimeout(state.clusterTimer);
  if (state.cluster?.chip === chip) { state.cluster.pinned ||= pinned; return; }
  closeCluster();
  state.cluster = { chip, pinned, view: camera.matrixWorld.clone() };
  chip.div.classList.add('open');
  chip.button.setAttribute('aria-expanded', 'true');
  clusterPop.replaceChildren(...[...chip.members].sort((a, b) => a.id - b.id).map(clusterEntry));
  clusterPop.hidden = false;
  // below the chip if the list fits there (or has more room there), else above; inside the view,
  // clear of the header and the help line
  const W = host.clientWidth, H = host.clientHeight;
  const obs = obstacles();
  const top = Math.max(LABEL_GAP, ...obs.filter((r) => r[1] < H / 2).map((r) => r[3]));
  const bottom = Math.min(H - LABEL_GAP, ...obs.filter((r) => r[1] >= H / 2).map((r) => r[1]));
  const [x0, y0] = chip.rect;
  const cw = chip.div.offsetWidth, ch = chip.div.offsetHeight;
  const below = bottom - (y0 + ch + 4), above = y0 - 4 - top;
  clusterPop.style.maxHeight = '';
  const natural = clusterPop.offsetHeight;
  const down = below >= natural || below >= above;
  clusterPop.style.maxHeight = `${Math.max(48, Math.floor(down ? below : above))}px`;
  const pw = clusterPop.offsetWidth, ph = clusterPop.offsetHeight;
  const x = Math.min(Math.max(x0 + cw / 2 - pw / 2, LABEL_GAP), W - pw - LABEL_GAP);
  const y = down ? y0 + ch + 4 : y0 - 4 - ph;
  clusterPop.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  drawClusterLeaders();
}
// while a chip's list is open: a line from the chip to each of its boxes (the peeked one bolder)
function drawClusterLeaders() {
  const c = state.cluster?.chip;
  if (!c) return;
  const cx = c.rect[0] + c.div.offsetWidth / 2, cy = (c.rect[1] + c.rect[3]) / 2;
  drawLeaders([...state.leaders, ...c.members.map((o) => [o.sx, o.sy, cx, cy, o.hex, o.id === state.peek])]);
}
function scheduleCloseCluster() {
  clearTimeout(state.clusterTimer);
  if (state.cluster && !state.cluster.pinned) state.clusterTimer = setTimeout(closeCluster, 250);
}
function closeCluster() {
  clearTimeout(state.clusterTimer);
  const cur = state.cluster;
  if (!cur) return;
  state.cluster = null;
  cur.chip.div.classList.remove('open');
  cur.chip.button.setAttribute('aria-expanded', 'false');
  clusterPop.hidden = true;
  clusterPop.replaceChildren();
  setPeek(null);
  drawLeaders(state.leaders);
  state.labelsDirty = true;
}
// whether the view has moved since the list opened (orbit damping's last creep moves it far less)
function clusterViewMoved() {
  const a = state.cluster.view.elements, b = camera.matrixWorld.elements;
  for (let i = 0; i < 16; i++) if (Math.abs(a[i] - b[i]) > 1e-3) return true;
  return false;
}

// Boxes the viewpoint is inside of, or next to, fade to OBB_FADE_MIN so that their edges do not
// cross the whole view (e.g. after "Go to" a camera standing in a box); the selected one stays.
const OBB_FADE_MIN = 0.2;
function fadeBoxes() {
  const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const eye = camera.position.clone().applyMatrix4(inv);  // scene frame
  const reach = Math.max(0.15, Math.min(1.0, 0.03 * (state.sceneSize || 5)));
  const v = new THREE.Vector3();
  for (const o of state.objects) {
    v.copy(eye).sub(o.center).applyQuaternion(o.inv);
    const out = Math.hypot(Math.max(Math.abs(v.x) - o.half.x, 0), Math.max(Math.abs(v.y) - o.half.y, 0),
      Math.max(Math.abs(v.z) - o.half.z, 0));
    const a = o.id === state.selected ? 1 : OBB_FADE_MIN + (1 - OBB_FADE_MIN) * THREE.MathUtils.smoothstep(out, 0, reach);
    const m = o.line.material;
    if (a >= 0.999) { if (m.transparent) { m.transparent = false; m.opacity = 1; m.needsUpdate = true; } } else {
      if (!m.transparent) { m.transparent = true; m.needsUpdate = true; }
      m.opacity = a;
    }
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
  setHover(id);
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
renderer.domElement.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; closeCluster(); });
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
  if (!$('#lightbox').hidden) { if (e.key === 'Escape') closeLightbox(); return; }
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'Escape') { select(null); selectCamera(null); }
  if (e.key === '[') stepCamera(-1);
  if (e.key === ']') stepCamera(1);
});

// ---------------------------------------------------------------- segmented image: lightbox
function openLightbox() {
  const box = $('#lightbox');
  box.hidden = false;
  box.classList.remove('actual');
  $('#lightbox-close').focus();
}
function closeLightbox() { $('#lightbox').hidden = true; $('#segmented-open').focus(); }
$('#segmented-open').addEventListener('click', openLightbox);
$('#lightbox-close').addEventListener('click', closeLightbox);
$('#lightbox').addEventListener('click', (e) => { if (e.target.id === 'lightbox') closeLightbox(); });
$('#lightbox img').addEventListener('click', () => $('#lightbox').classList.toggle('actual'));

// ---------------------------------------------------------------- Layers
function applyLayers() {
  for (const k of ['points', 'segments', 'cameras', 'obbs']) groups[k].visible = state.layers[k];
  groups.labels.visible = state.layers.labels;
  groups.pick.visible = state.layers.obbs;
  if (!state.layers.obbs) tooltip.hidden = true;
  closeCluster();
  state.labelsDirty = true;
}
function buildLayers() {
  const counts = { cameras: state.meta.cameras.length, labels: state.objects.length, obbs: state.objects.length };
  for (const [k, name, help] of LAYERS) {
    const cb = el('input', { type: 'checkbox', id: `layer-${k}` });
    cb.checked = state.layers[k];
    cb.addEventListener('change', () => { state.layers[k] = cb.checked; applyLayers(); });
    const text = k === 'cameras' && counts.cameras === 1 ? 'Camera pose' : name;
    const row = el('label', { class: 'row check', 'data-layer': k, for: `layer-${k}`, title: help }, cb,
      el('span', {}, text, el('small', { class: 'layer-note', id: `layer-note-${k}` })),
      el('span', { class: 'count' }, k in counts ? String(counts[k]) : ''));
    if (k === 'cameras' && !counts.cameras) { cb.disabled = true; row.classList.add('disabled'); }
    $('#layers').append(row);
  }
  updateLayerNotes();
}
// what the two object-colour displays are doing, so that they cannot be confused
function updateLayerNotes() {
  const note = $('#layer-note-segments');
  if (!note) return;
  note.textContent = state.attrs.color === 'segment'
    ? 'the point cloud is coloured by object too (color=segment)'
    : 'object points in their colours, over the cloud';
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
      const words = c.key === 'color' ? COLOR_WORDS : {};
      input = el('select', { id }, ...c.options.map((o) => el('option', { value: o }, words[o] || o)));
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
  updateLayerNotes();
  setBusy(true);
  clearTimeout(state.debounce);
  state.debounce = setTimeout(reloadCloud, DEBOUNCE_MS);
}
function setBusy(on) {
  state.busy = on;
  $('#busy').hidden = !on;
  state.labelsDirty = true;
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
  const lab = el('select', { id: 'display-labels' },
    el('option', { value: 'fit' }, 'id tags + names that fit'), el('option', { value: 'ids' }, 'id tags only'));
  lab.value = state.display.labels;
  lab.addEventListener('change', () => { state.display.labels = lab.value; closeCluster(); state.labelsDirty = true; });
  host.append(el('div', { class: 'row attr',
    title: 'Every box gets its id tag (or a "+N" chip where tags are crowded); names where they fit, on hover and on selection' },
    el('label', { for: 'display-labels' }, 'labels'), lab, el('output')));
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
      el('td', { class: 'num' }, String(o.id)), el('td', { class: 'label', title: o.label }, o.label),
      el('td', { class: 'num' }, o.score != null ? o.score.toFixed(2) : ''),
      el('td', { class: 'num dims' }, `${w.toFixed(2)}×${d.toFixed(2)}×${h.toFixed(2)}`),
      el('td', { class: 'num' }, o.volume.toFixed(3)));
    tr.addEventListener('click', () => select(o.id, true));
    tr.addEventListener('mouseenter', () => setHover(o.id));
    tr.addEventListener('mouseleave', () => setHover(null));
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
  const c = state.cloud;
  const pts = c ? c.count : 0;
  const shown = c && c.step > 1 ? `${pts.toLocaleString()} of ${c.total.toLocaleString()} points (display limit)` : `${pts.toLocaleString()} points`;
  const items = [shown, `${s.objects} objects`, `${s.frames} frame${s.frames === 1 ? '' : 's'}`];
  $('#stats').replaceChildren(...items.flatMap((t, i) => [i ? ' · ' : '', el('span', { class: 'stat' }, t)]));
  state.labelsDirty = true;
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
  state.sceneSize = size;
  buildFrustums(state.meta.cameras, size);
  for (const f of state.meta.cameras) state.bbox.expandByPoint(new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]).applyMatrix4(root.matrixWorld));
  if (state.meta.has_segmented) {
    $('#tab-image-btn').hidden = false;
    $('#segmented').src = '/api/segmented.png';
    $('#lightbox img').src = '/api/segmented.png';
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
const lastView = new THREE.Matrix4();
function animate() {
  requestAnimationFrame(animate);
  stepFlight(performance.now());
  controls.update();
  camera.updateMatrixWorld();
  if (!lastView.equals(camera.matrixWorld)) {  // the viewpoint moved: labels and fades follow
    lastView.copy(camera.matrixWorld);
    state.labelsDirty = true;
    if (state.objects.length) fadeBoxes();
    fadeHighlight();
  }
  if (state.cluster && clusterViewMoved()) closeCluster();
  // an open chip list holds the labels still (they would move under the pointer)
  if (state.labelsDirty && state.ready && !state.cluster) { state.labelsDirty = false; layoutLabels(); }
  if (state.ready && state.cloudInScene && !document.body.dataset.rendered && ++cloudFrames > 1) {
    document.body.dataset.rendered = 'true';
  }
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);
main().catch((err) => {
  console.error(err);
  document.body.dataset.error = err.message;
  $('#loading-title').textContent = `Failed to load: ${err.message}`;
});
