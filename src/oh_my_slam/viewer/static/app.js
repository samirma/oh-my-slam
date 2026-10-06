// view.sh's page (spec §2.5): display only. Every point cloud is derived by the server (api/cloud,
// the shared derivation of segmentation.cloud); objects, OBBs, colours and camera poses come from
// the scene JSON and api/meta. Nothing is recomputed here.
//
// What the page offers is what §2.5 asks for: independent point-cloud, camera-pose, segmentation,
// label and OBB layers; live controls for the applicable point-cloud attributes, with the
// display-budget notice; the position of every camera with a "Go to" that moves the viewpoint
// there; and, for an image, the segmented image and the object catalogue. The drawing is the
// Viewer of lib/viewer.js; data comes through lib/data.js relative to the page's URL.
import { DataSource } from './lib/data.js';
import { el } from './lib/dom.js';
import { Viewer } from './lib/viewer.js';
import { sceneObjects } from './lib/obbs.js';
import { fillCameraTable } from './lib/cameras.js';
import { buildLayerControls } from './lib/layers.js';
import { buildAttributeControls, budgetNote } from './lib/controls.js';
import { crowdedNote } from './lib/labels.js';

const DEBOUNCE_MS = 250;
const $ = (sel) => document.querySelector(sel);

const data = new DataSource();
const viewer = new Viewer($('#canvas-host'));
const state = {
  meta: null, scene: null,
  attrs: {},          // current point-cloud attributes, as the -p values the server parses
  cloudSeq: 0, abort: null, debounce: null, busy: false, ready: false,
  get cloud() { return viewer.cloud; },
  get objects() { return viewer.objects; },
  get layers() { return viewer.layers; },
  get frames() { return viewer.frames; },  // frames drawn so far (drawn on demand only)
};
// for tests and debugging
window.__viewer = state;
window.__viewerGroups = viewer.groups;
window.__viewerCamera = viewer.camera;
window.__viewerControls = viewer.controls;
window.__viewerInvalidate = (n) => viewer.invalidate(n);  // for tests that change the scene directly
window.__viewerResetView = () => viewer.resetView();     // for tests: back to the initial view

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

// ---------------------------------------------------------------- labels with no room
viewer.onCrowded((crowded) => {
  const note = $('#labels-note');
  const text = crowdedNote(crowded);
  if (note.textContent !== text) note.textContent = text;
  note.hidden = !text;
});

// ---------------------------------------------------------------- point cloud
let attrControls = null;
function showCloud(cloud) {
  viewer.setCloud(cloud);
  const note = $('#cloud-note');
  note.textContent = budgetNote(cloud.header);
  note.hidden = !note.textContent;
}
function showError(msg) {
  const box = $('#cloud-error');
  box.hidden = !msg;
  box.textContent = msg ? `Not updated: ${msg}` : '';
  attrControls?.markInvalid(msg ? msg.split(/[\s(=]/)[0] : null);
}
function attrsChanged(key, value) {
  state.attrs[key] = value;
  state.busy = true;
  clearTimeout(state.debounce);
  state.debounce = setTimeout(reloadCloud, DEBOUNCE_MS);
}
function currentAttrs() { return state.meta.controls.map((c) => [c.key, state.attrs[c.key]]); }
async function reloadCloud() {
  const seq = ++state.cloudSeq;
  state.abort?.abort();
  const ctl = new AbortController();
  state.abort = ctl;
  try {
    const cloud = await data.cloud(currentAttrs(), ctl.signal);
    if (seq !== state.cloudSeq) return;
    showCloud(cloud);
    showError(null);
  } catch (err) {
    if (err.name === 'AbortError' || seq !== state.cloudSeq) return;
    showError(err.message);
  } finally {
    if (seq === state.cloudSeq) state.busy = false;
  }
}

// ---------------------------------------------------------------- catalogue (an image only)
// segmentation's own catalogue (api/catalog: the rows of catalog.csv), shown as served: the
// server's row order and its columns, values unchanged. Only a colour swatch is added, next to the
// id, so colour is never the only cue.
function buildCatalogue(rows) {
  const columns = rows.length ? Object.keys(rows[0]) : [];
  const thead = $('#catalogue thead');
  const tbody = $('#catalogue tbody');
  thead.replaceChildren(el('tr', {}, ...columns.map((c) =>
    el('th', { scope: 'col', class: typeof rows[0][c] === 'number' ? `num c-${c}` : `c-${c}` }, c))));
  for (const row of rows) {
    tbody.appendChild(el('tr', { 'data-id': row.id }, ...columns.map((c) => {
      const v = row[c];
      const cell = el('td', { class: typeof v === 'number' ? `num c-${c}` : `c-${c}` });
      if (c === 'id') cell.append(el('span', { class: 'swatch', style: `background:${row.color_hex}`,
        title: row.color_hex, 'aria-hidden': 'true' }), ' ');
      cell.append(v == null ? '' : String(v));
      if (c === 'label') cell.title = String(v);
      return cell;
    })));
  }
  if (!rows.length) tbody.append(el('tr', {}, el('td', { class: 'muted' }, 'No objects.')));
}

// ---------------------------------------------------------------- main
async function main() {
  state.meta = await data.meta();
  state.scene = await data.scene();
  const { meta } = state;
  document.title = `${meta.title} — oh-my-slam`;
  viewer.setDisplayTransform(meta.display_transform);
  viewer.setObjects(sceneObjects(state.scene));
  for (const c of meta.controls) state.attrs[c.key] = c.default;
  showCloud(await data.cloud(currentAttrs()));
  viewer.setCameras(meta.cameras);
  if (meta.mode === 'image') $('#tab-catalogue-btn').hidden = false;
  if (meta.has_segmented) {
    $('#tab-image-btn').hidden = false;
    $('#segmented').src = data.segmentedUrl();
  }
  viewer.layers.cameras = meta.cameras.length > 0;
  buildLayerControls($('#layers'), viewer.layers, (k, on) => viewer.setLayer(k, on),
    new Set(meta.cameras.length ? [] : ['cameras']));
  attrControls = buildAttributeControls($('#cloud-controls'), meta.controls, state.attrs, attrsChanged);
  if (meta.mode === 'image') buildCatalogue(await data.catalog());
  const cs = state.scene.openlabel.coordinate_systems || {};
  const axes = cs.map?.axes?.replaceAll(',', ', ');
  $('#cam-note').textContent = meta.mode === 'map'
    ? `Camera centres in the map frame${axes ? ` (${axes})` : ''}, metres.`
    : 'Camera centre in the scene frame (the image\'s camera frame, OpenCV axes), metres.';
  fillCameraTable($('#cameras tbody'), meta.cameras, (i) => viewer.goToCamera(i));
  viewer.setLayer('points', viewer.layers.points);  // apply every layer's state
  viewer.resize();
  viewer.resetView();
  $('#loading').classList.add('done');
  state.ready = true;
  viewer.invalidate();
}

// <body data-rendered="true"> once a frame showing the point cloud has been drawn and presented:
// set in the animation frame after the first one that drew the loaded cloud; never removed.
let drawn = false;
function watchRendered() {
  if (drawn) { document.body.dataset.rendered = 'true'; return; }
  drawn = state.ready && viewer.cloudDrawn;
  requestAnimationFrame(watchRendered);
}
requestAnimationFrame(watchRendered);
main().catch((err) => {
  console.error(err);
  document.body.dataset.error = err.message;
  $('#loading-title').textContent = `Failed to load: ${err.message}`;
});
