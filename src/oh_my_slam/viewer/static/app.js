// oh-my-slam viewer: display only. Geometry, colours, OBBs and labels come from the server
// (reconstruction / mapping / segmentation outputs); nothing is recomputed here.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import GUI from '/static/vendor/lil-gui/lil-gui.esm.min.js';

const $ = (sel) => document.querySelector(sel);
const state = {
  meta: null, scene: null, objects: [], selected: null, hovered: null,
  layers: { points: true, segments: false, mesh: true, cameras: true, labels: true, obbs: true },
  display: { pointSize: 2.0, labelDensity: 12, background: '#15171c' },
  bbox: new THREE.Box3(),
};
window.__viewer = state; // for tests and debugging

// ---------------------------------------------------------------- renderer, scene, controls
const host = $('#canvas-host');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;   // exact sRGB output
renderer.toneMapping = THREE.NoToneMapping;          // no tone mapping, colours stay exact
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
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;

const root = new THREE.Group();  // display transform (image mode: camera → z-up)
scene.add(root);
const groups = {};
for (const k of ['points', 'segments', 'mesh', 'cameras', 'obbs', 'labels', 'pick']) {
  groups[k] = new THREE.Group();
  groups[k].name = k;
  root.add(groups[k]);
}
window.__viewerGroups = groups;  // read by the browser tests
window.__viewerCamera = camera;
window.__viewerControls = controls;

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  for (const o of state.objects) o.line.material.resolution.set(w, h);
  const focal = (h * renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  for (const g of [groups.points, groups.segments]) g.traverse((m) => { if (m.material?.uniforms?.focal) m.material.uniforms.focal.value = focal; });
}
window.addEventListener('resize', resize);

// ---------------------------------------------------------------- loading with progress
const loadingList = $('#loading-list');
function loadRow(name) {
  const li = document.createElement('li');
  li.innerHTML = `<span>${name}</span><span class="pct">0%</span>`;
  loadingList.appendChild(li);
  return {
    progress(p) { li.querySelector('.pct').textContent = `${Math.round(p * 100)}%`; },
    ok(msg = 'done') { li.classList.add('ok'); li.querySelector('.pct').textContent = msg; },
    fail(msg) { li.classList.add('err'); li.querySelector('.pct').textContent = msg; },
  };
}
async function fetchWithProgress(url, name, as = 'arrayBuffer') {
  const row = loadRow(name);
  const res = await fetch(url);
  if (!res.ok) { row.fail(`HTTP ${res.status}`); throw new Error(`${url}: ${res.status}`); }
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) row.progress(got / total);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  row.ok(total > 1e6 ? `${(total / 1e6).toFixed(1)} MB` : 'done');
  if (as === 'json') return JSON.parse(new TextDecoder().decode(buf));
  return buf.buffer;
}

// ---------------------------------------------------------------- point clouds (raw sRGB)
function pointMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { size: { value: state.display.pointSize }, focal: { value: 800.0 } },
    // size is the point diameter in centimetres; focal the viewport focal length in pixels
    vertexShader: `
      attribute vec3 rgb;
      uniform float size; uniform float focal;
      varying vec3 vColor;
      void main() {
        vColor = rgb;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(size * 0.01 * focal / max(-mv.z, 0.05), 1.0, 24.0);
        gl_Position = projectionMatrix * mv;
      }`,
    // No colorspace conversion: the bytes are sRGB already and are written as they are.
    fragmentShader: `
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }`,
  });
}
function makePoints(positions, colors, name) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('rgb', new THREE.BufferAttribute(colors, 3, true));
  g.computeBoundingBox();
  const p = new THREE.Points(g, pointMaterial());
  p.name = name;
  return p;
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
function num(o, name) { const v = (o.object_data.num || []).find((n) => n.name === name); return v ? v.val : null; }
function attr(cub, name) { const v = ((cub.attributes || {}).num || []).find((n) => n.name === name); return v ? v.val : 0; }

function buildObjects(doc) {
  const objs = doc.openlabel.objects || {};
  for (const [id, o] of Object.entries(objs)) {
    const cub = (o.object_data.cuboid || [])[0];
    if (!cub) continue;
    const hex = (o.object_data.text || []).find((t) => t.name === 'color_hex').val;
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
    const div = document.createElement('div');
    div.className = 'obj-label';
    div.style.borderLeftColor = hex;
    div.textContent = `${o.type} ${id}`;
    div.addEventListener('click', (e) => { e.stopPropagation(); select(Number(id), true); });
    const label = new CSS2DObject(div);
    const top = corners.reduce((m, p) => (p.z > m.z ? p : m), corners[0]);
    label.position.set(x, y, z);
    label.position.lerp(top, 1.0);
    groups.labels.add(label);
    state.objects.push({
      id: Number(id), label: o.type, hex, score: num(o, 'score'), volume: attr(cub, 'volume_m3'),
      dims: [attr(cub, 'width_m'), attr(cub, 'depth_m'), attr(cub, 'height_m')],
      center: new THREE.Vector3(x, y, z), corners, line, pick, labelObj: label, div,
    });
  }
}

// ---------------------------------------------------------------- camera frustums
function buildFrustums(frustums, size) {
  if (!frustums.length) return;
  const d = Math.max(0.05, size * 0.025);
  const pos = [];
  for (const f of frustums) {
    const T = new THREE.Matrix4().fromArray(f.T.flat()).transpose();
    const [fx, fy, cx, cy] = f.K; const [w, h] = f.size;
    const c = new THREE.Vector3(0, 0, 0).applyMatrix4(T);
    const cs = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
      new THREE.Vector3((u - cx) / fx * d, (v - cy) / fy * d, d).applyMatrix4(T));
    for (let i = 0; i < 4; i++) { pos.push(...c.toArray(), ...cs[i].toArray(), ...cs[i].toArray(), ...cs[(i + 1) % 4].toArray()); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: srgb('#b8c0cc') }));
  lines.name = 'frustums';
  groups.cameras.add(lines);
}

// ---------------------------------------------------------------- mesh (unlit, exact texture)
async function loadMesh() {
  const buf = await fetchWithProgress('/api/mesh.glb', 'textured mesh');
  const gltf = await new Promise((resolve, reject) => new GLTFLoader().parse(buf, '', resolve, reject));
  let tris = 0;
  gltf.scene.traverse((m) => {
    if (!m.isMesh) return;
    const old = m.material;
    const map = old.map || null;
    if (map) map.colorSpace = THREE.SRGBColorSpace;
    m.material = new THREE.MeshBasicMaterial({ map, vertexColors: !map && !!m.geometry.attributes.color, side: THREE.DoubleSide });
    tris += m.geometry.index ? m.geometry.index.count / 3 : m.geometry.attributes.position.count / 3;
  });
  groups.mesh.add(gltf.scene);
  return tris;
}

// ---------------------------------------------------------------- framing
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
  if (state.meta && state.meta.mode === 'image' && state.photoTarget) {
    // image mode: start from the photo's own viewpoint (camera centre, looking forward)
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
  // prefer the smallest box among the hits (inner objects)
  hits.sort((a, b) => {
    const va = state.objects.find((o) => o.id === a.object.userData.id).volume;
    const vb = state.objects.find((o) => o.id === b.object.userData.id).volume;
    return va - vb;
  });
  return hits[0].object.userData.id;
}
const tooltip = $('#tooltip');
renderer.domElement.addEventListener('pointermove', (ev) => {
  const id = pickAt(ev);
  if (id !== state.hovered) { state.hovered = id; updateLabels(); }
  if (id == null) { tooltip.hidden = true; renderer.domElement.style.cursor = ''; return; }
  const o = state.objects.find((x) => x.id === id);
  const [w, d, h] = o.dims;
  tooltip.innerHTML = `<div class="t-title"><span class="sw" style="background:${o.hex}"></span>${o.label} <span style="color:var(--muted)">#${o.id}</span></div>
    score ${o.score != null ? o.score.toFixed(2) : '–'}<br>W×D×H ${w.toFixed(2)} × ${d.toFixed(2)} × ${h.toFixed(2)} m<br>volume ${o.volume.toFixed(3)} m³`;
  const r = host.getBoundingClientRect();
  tooltip.style.left = `${Math.min(ev.clientX - r.left + 14, r.width - 190)}px`;
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

// ---------------------------------------------------------------- panel, tabs, GUI, catalogue
function showTab(name) {
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
}
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
$('#toggle-panel').addEventListener('click', () => { $('#app').classList.toggle('panel-hidden'); setTimeout(resize, 200); });
$('#reset-view').addEventListener('click', resetView);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'Escape') select(null);
});

function applyLayers() {
  for (const k of ['points', 'segments', 'mesh', 'cameras', 'obbs']) groups[k].visible = state.layers[k];
  groups.pick.visible = state.layers.obbs;
  updateLabels();
}
function buildGui() {
  const layers = new GUI({ container: $('#tab-layers'), title: 'Layers' });
  const avail = {
    points: 'Point cloud (RGB)', segments: 'Segmentation (object colours)', mesh: 'Textured mesh',
    cameras: 'Camera poses', labels: 'Labels', obbs: 'Oriented boxes',
  };
  for (const [k, name] of Object.entries(avail)) {
    const c = layers.add(state.layers, k).name(name).onChange(applyLayers);
    c.domElement.dataset.layer = k;
    if (k === 'mesh' && !state.meta.has_mesh) c.disable();
    if (k === 'cameras' && !state.meta.frustums.length) c.disable();
  }
  const disp = new GUI({ container: $('#tab-display'), title: 'Display' });
  disp.add(state.display, 'pointSize', 0.5, 8, 0.1).name('Point size').onChange((v) => {
    for (const g of [groups.points, groups.segments]) g.traverse((m) => { if (m.material?.uniforms?.size) m.material.uniforms.size.value = v; });
  });
  disp.add(state.display, 'labelDensity', 0, 60, 1).name('Label density').onChange(updateLabels);
  disp.addColor(state.display, 'background').name('Background').onChange((v) => scene.background.set(v));
  disp.add({ reset: resetView }, 'reset').name('Reset view (R)');
}
function buildCatalogue() {
  const tbody = $('#catalogue tbody');
  const rows = [...state.objects].sort((a, b) => b.volume - a.volume);
  for (const o of rows) {
    const tr = document.createElement('tr');
    tr.dataset.id = o.id;
    const [w, d, h] = o.dims;
    tr.innerHTML = `<td><span class="swatch" style="background:${o.hex}" title="${o.hex}"></span></td><td>${o.id}</td><td>${o.label}</td><td>${o.score != null ? o.score.toFixed(2) : ''}</td><td>${w.toFixed(2)}×${d.toFixed(2)}×${h.toFixed(2)}</td><td>${o.volume.toFixed(3)}</td>`;
    tr.addEventListener('click', () => select(o.id, true));
    tr.addEventListener('mouseenter', () => { state.hovered = o.id; updateLabels(); });
    tr.addEventListener('mouseleave', () => { state.hovered = null; updateLabels(); });
    tbody.appendChild(tr);
  }
  $('#filter').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    tbody.querySelectorAll('tr').forEach((tr) => {
      tr.hidden = q && !tr.children[2].textContent.toLowerCase().includes(q);
    });
  });
}

// ---------------------------------------------------------------- main
async function main() {
  resize();
  state.meta = await fetchWithProgress('/api/meta', 'metadata', 'json');
  state.scene = await fetchWithProgress('/api/scene', 'scene description', 'json');
  $('#title').textContent = `${state.meta.mode === 'map' ? 'Map' : 'Image'} · ${state.meta.title}`;
  document.title = `${state.meta.title} — oh-my-slam`;
  const T = new THREE.Matrix4().fromArray(state.meta.display_transform.flat()).transpose();
  root.matrixAutoUpdate = false;
  root.matrix.copy(T);
  root.updateMatrixWorld(true);
  const [pos, col, seg] = await Promise.all([
    fetchWithProgress('/api/points.bin', `points (${state.meta.points.toLocaleString()})`),
    fetchWithProgress('/api/colors.bin', 'colours'),
    fetchWithProgress('/api/segments.bin', 'segment colours'),
  ]);
  const positions = new Float32Array(pos);
  if (positions.length) {
    groups.points.add(makePoints(positions, new Uint8Array(col), 'points'));
    groups.segments.add(makePoints(positions, new Uint8Array(seg), 'segments'));
    state.bbox.union(robustBox(positions, root.matrixWorld));
    if (state.meta.mode === 'image') {
      // look-at point: median depth straight ahead of the photo's camera (camera frame +z)
      const zsorted = [];
      for (let i = 2; i < positions.length; i += 30) zsorted.push(positions[i]);
      zsorted.sort((a, b) => a - b);
      const zmed = zsorted[Math.floor(zsorted.length / 2)] || 1;
      state.photoTarget = new THREE.Vector3(0, 0, zmed).applyMatrix4(root.matrixWorld);
    }
  }
  buildObjects(state.scene);
  let tris = 0;
  if (state.meta.has_mesh) {
    try { tris = await loadMesh(); } catch (err) { console.warn('mesh failed to load', err); }
  }
  const size = state.bbox.isEmpty() ? 1 : state.bbox.getSize(new THREE.Vector3()).length();
  buildFrustums(state.meta.frustums, size);
  for (const f of state.meta.frustums) state.bbox.expandByPoint(new THREE.Vector3(f.T[0][3], f.T[1][3], f.T[2][3]));
  if (state.meta.has_segmented) {
    $('#tab-image-btn').hidden = false;
    $('#segmented').src = '/api/segmented.png';
  }
  state.layers.segments = false;
  state.layers.cameras = state.meta.frustums.length > 0;
  state.layers.mesh = state.meta.has_mesh;
  state.layers.points = !state.meta.has_mesh;
  buildGui();
  buildCatalogue();
  applyLayers();
  const s = state.meta.stats;
  $('#stats').textContent = `${Number(s.points).toLocaleString()} points · ${Number(tris || s.triangles || 0).toLocaleString()} triangles · ${s.objects} objects · ${s.frames} frame${s.frames === 1 ? '' : 's'}`;
  resize();
  resetView();
  $('#loading').classList.add('done');
  state.ready = true;
}

let lastLabelUpdate = 0;
function animate(t) {
  requestAnimationFrame(animate);
  controls.update();
  if (t - lastLabelUpdate > 150) { lastLabelUpdate = t; if (state.objects.length) updateLabels(); }
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
requestAnimationFrame(animate);
main().catch((err) => {
  console.error(err);
  $('#loading-title').textContent = `Failed to load: ${err.message}`;
});
