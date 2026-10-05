// The 3D scene viewer (http_server.md "Structure"): opens, alone or together in one view, a point
// cloud (PLY) and a scene description (JSON, spec §3) produced by the commands — a job's result or
// file (#/scene?ply=<url>&json=<url>, a stable URL), or files opened from disk, which are read in
// the browser and never uploaded. The drawing is the §2.5 viewer's own (lib/viewer.js and its
// modules): the cloud with the attributes its PLY carries and the controls the file allows; each
// object's OBB from its cuboid in its colour with its label and id; every camera of the JSON or of
// a `mapper.sh locate` PLY header, with its coordinates and a go-to, located cameras drawn apart
// from a map's frames. Both files share map coordinates; a single image's camera frame is shown
// upright by the viewer's own display transform (/api/display-transform). Each layer has its
// toggle; the list of the JSON's objects is linked to their boxes. The layers on, the point-cloud
// controls and the selected object are in the URL. An invalid PLY, a document that does not
// validate against the scene schema, or a disk PLY above the display budget is refused with the
// reason (scene/files.js reads the files, in a worker).
import * as THREE from 'three';
import { el, clear, notice, fmtBytes } from '../dom.js';
import { getJson } from '../api.js';
import { Viewer } from '/static/viewer/lib/viewer.js';
import { sceneObjects, obbCorners } from '/static/viewer/lib/obbs.js';
import { sceneCameras, plyCameras, fillCameraTable } from '/static/viewer/lib/cameras.js';
import { buildLayerControls, LAYERS } from '/static/viewer/lib/layers.js';
import { buildAttributeControls, budgetNote } from '/static/viewer/lib/controls.js';
import { crowdedNote } from '/static/viewer/lib/labels.js';
import { plyFromDisk, plyFromService, sceneFromBytes, fetchBuffer } from '../scene/files.js';
import { Selection } from '../selection.js';
import { urlSelection, setQuery, queryParams } from '../url.js';
import { objectsTable } from '../jobview.js';
import { clickToSelect } from '../pick.js';

const fmtN = (n) => n.toLocaleString('en-US');

// The point-cloud controls a PLY file allows: its colours or none (when it carries colours), and
// shading by its normals or not (when it carries normals). They only choose what to draw of the
// file; nothing is re-derived.
function plyControls(cloud) {
  const out = [];
  const recorded = /(?:^|,)color=([^,]+)/.exec(cloud.header.attrs || '');
  if (cloud.arrays.color) {
    const as = recorded ? recorded[1] : 'rgb';
    out.push({ key: 'color', kind: 'choice', options: [as, 'none'], default: as,
      help: 'per-point colour as the file carries it, or none', values: `${as}|none` });
  }
  if (cloud.arrays.normal) {
    out.push({ key: 'normals', kind: 'toggle', default: 'on', help: 'shade the points by the normals the file carries', values: 'on|off' });
  }
  return out;
}

function shownCloud(cloud, attrs) {
  const arrays = { position: cloud.arrays.position };
  if (cloud.arrays.label) arrays.label = cloud.arrays.label;
  if (cloud.arrays.color && attrs.color !== 'none') arrays.color = cloud.arrays.color;
  if (cloud.arrays.normal && attrs.normals !== 'off') arrays.normal = cloud.arrays.normal;
  const parts = (cloud.header.attrs || '').split(',').filter((p) => p && !/^(color|normals)=/.test(p));
  if (arrays.color) parts.unshift(`color=${attrs.color}`);
  return { header: { ...cloud.header, attrs: parts.join(',') }, arrays };
}

function plyEntry(name, cloud) {
  const { cameras, unlocated } = plyCameras(cloud.header.comments || []);
  return { name, cloud, cameras, unlocated };
}

function sceneEntry(name, doc, errors) {
  if (errors.length) {
    const head = errors.slice(0, 8).join('; ');
    throw new Error(`${name} does not validate against the scene schema (ASAM OpenLABEL 1.0.0, spec §3): ${head}`
      + (errors.length > 8 ? ` …and ${errors.length - 8} more problems.` : '.'));
  }
  return { name, doc, objects: sceneObjects(doc), cameras: sceneCameras(doc) };
}

// The display transform of what is open: a scene in a single image's camera frame (no scene
// coordinate system; or a PLY whose header names that frame) is shown upright, by the viewer's own
// transform, asked of the service.
async function displayTransform(state) {
  const q = new URLSearchParams();
  if (state.json) {
    const ol = state.json.doc.openlabel;
    const scene = Object.values(ol.coordinate_systems || {}).some((c) => c && c.type === 'scene_cs');
    if (scene) return null;
    q.set('camera', 'true');
    const up = ol.metadata?.gravity?.up_cam;
    if (Array.isArray(up) && up.length === 3 && up.every(Number.isFinite)) q.set('up', up.join(','));
  } else if (state.ply) {
    for (const c of state.ply.cloud.header.comments || []) if (!/^located_\d+ /.test(c)) q.append('comment', c);
  } else return null;
  const r = await getJson(`/api/display-transform?${q}`);
  return r.camera_frame ? r.display_transform : null;
}

export function scenePage(main, { query }) {
  const selection = urlSelection(new Selection());
  const state = { ply: null, json: null, attrs: {} };
  const errors = el('div', { class: 'scene-errors', 'aria-live': 'assertive', 'data-testid': 'scene-errors' });
  const file = el('input', { type: 'file', id: 'scene-files', multiple: true, accept: '.ply,.json,application/json' });
  const sources = el('ul', { class: 'sources', 'data-testid': 'sources' });
  const host = el('div', { class: 'scene-canvas' });
  const empty = el('p', { class: 'scene-empty' }, 'Open a PLY file, a scene JSON, or both.');
  const view = el('div', { class: 'scene-view', 'data-testid': 'scene-view' }, host, empty);
  const layersBox = el('div', { class: 'rows', id: 'scene-layers' });
  const labelsNote = el('p', { class: 'hint', 'aria-live': 'off', hidden: true });
  const cloudBox = el('div', { class: 'rows', id: 'scene-cloud' });
  const cloudNote = el('p', { class: 'hint' });
  const frameNote = el('p', { class: 'hint' });
  const objectsBox = el('div', {});
  const camerasBody = el('tbody', {});
  const camNote = el('p', { class: 'hint' });
  const panel = el('aside', { class: 'scene-panel', 'aria-label': 'Scene' },
    el('section', {}, el('h2', {}, 'Files'), sources, frameNote),
    el('section', {}, el('h2', {}, 'Layers'), layersBox, labelsNote),
    el('section', {}, el('h2', {}, 'Point cloud'), cloudBox, cloudNote),
    el('section', {}, el('h2', {}, 'Objects'), objectsBox),
    el('section', {}, el('h2', {}, 'Cameras'), camNote, el('div', { class: 'table-wrap' },
      el('table', { class: 'data cameras', 'data-testid': 'scene-cameras' },
        el('caption', { class: 'visually-hidden' }, 'Cameras: centre and go-to'),
        el('thead', {}, el('tr', {}, ...['camera', 'x', 'y', 'z', ''].map((h, i) => el('th', { scope: 'col', class: i && i < 4 ? 'num' : null }, h || el('span', { class: 'visually-hidden' }, 'go to'))))),
        camerasBody))));
  main.append(el('h1', {}, '3D scene viewer'),
    el('div', { class: 'scene-open' },
      el('label', { for: 'scene-files' }, 'Open a PLY and/or a scene JSON from this computer'),
      file, el('p', { class: 'help' }, 'Files opened here are read by your browser and never uploaded. A PLY and a JSON opened together share map coordinates.')),
    errors, el('div', { class: 'scene-layout' }, view, panel));

  let viewer = null;
  const ensureViewer = () => {
    if (viewer) return viewer;
    viewer = new Viewer(host);
    window.__sceneViewer = viewer;
    viewer.onCrowded((c) => { const t = crowdedNote(c); labelsNote.textContent = t; labelsNote.hidden = !t; });
    viewer.onSelect((id) => selection.set(id));
    selection.join((id) => viewer.select(id));
    clickToSelect(viewer, (id) => selection.set(id));
    return viewer;
  };

  function drawSources() {
    clear(sources);
    for (const kind of ['ply', 'json']) {
      const s = state[kind];
      if (!s) continue;
      const rm = el('button', { type: 'button', class: 'icon', 'aria-label': `Close ${s.name}` }, '✕');
      rm.addEventListener('click', () => { state[kind] = null; setQuery({ [kind]: null }); rebuild(); });
      const h = s.cloud?.header;
      const what = kind === 'ply'
        ? `${fmtN(h.count)}${h.count < h.total ? ` of ${fmtN(h.total)}` : ''} points; ${['color', 'normal', 'label'].filter((k) => s.cloud.arrays[k]).join(', ') || 'positions only'}${s.cameras.length ? `; ${s.cameras.length} located camera(s)` : ''}`
        : `${s.objects.length} objects, ${s.cameras.length} cameras`;
      sources.append(el('li', { 'data-kind': kind }, el('strong', {}, s.name), el('span', { class: 'muted' }, ` ${kind.toUpperCase()}: ${what}${s.size ? `, ${fmtBytes(s.size)}` : ''}`), rm));
    }
  }

  function applyCloud() {
    if (!state.ply) return;
    const c = shownCloud(state.ply.cloud, state.attrs);
    viewer.setCloud(c);
    cloudNote.textContent = budgetNote(c.header) || `All ${fmtN(c.header.count)} points of ${state.ply.name}.`;
  }

  const layersOn = (v) => Object.keys(v.layers).filter((k) => v.layers[k]).join(',');

  let building = 0;
  async function rebuild() {
    const seq = ++building;
    drawSources();
    const any = state.ply || state.json;
    let matrix = null;
    if (any) {
      try { matrix = await displayTransform(state); } catch (err) { errors.append(notice('error', `The display transform could not be read: ${err.message}`)); }
      if (seq !== building) return;
    }
    clear(layersBox); clear(cloudBox); clear(objectsBox); clear(camerasBody);
    empty.hidden = !!any;
    view.dataset.loaded = any ? 'true' : 'false';
    if (!any) {
      if (viewer) { viewer.setObjects([]); viewer.setCameras([]); viewer.groups.points.clear(); viewer.groups.segments.clear(); viewer.invalidate(); }
      frameNote.textContent = '';
      return;
    }
    const v = ensureViewer();
    v.setDisplayTransform(matrix || [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
    view.dataset.upright = matrix ? 'true' : 'false';
    frameNote.textContent = matrix
      ? 'A single image\'s camera frame, shown upright as view.sh shows it.'
      : 'Map coordinates (z up).';
    v.cloudBox.makeEmpty();
    v.setObjects(state.json ? state.json.objects : []);
    const q = queryParams();
    if (state.ply) {
      const controls = plyControls(state.ply.cloud);
      state.attrs = Object.fromEntries(controls.map((c) => {
        const wanted = q.get(c.key);
        const ok = wanted !== null && (c.kind === 'choice' ? c.options.includes(wanted) : ['on', 'off'].includes(wanted));
        return [c.key, ok ? wanted : c.default];
      }));
      applyCloud();
      buildAttributeControls(cloudBox, controls, state.attrs, (k, val) => { state.attrs[k] = val; setQuery({ [k]: val }); applyCloud(); });
      if (!controls.length) cloudBox.append(el('p', { class: 'hint' }, 'The file carries no colours or normals to choose from.'));
    } else {
      v.groups.points.clear(); v.groups.segments.clear(); v.cloud = null;
      cloudBox.append(el('p', { class: 'hint' }, 'No point cloud: open a PLY to draw one.'));
      cloudNote.textContent = '';
    }
    if (!state.ply) {  // no cloud to frame: the boxes are the scene
      const box = new THREE.Box3();
      for (const o of state.json.objects) if (o.cuboid) for (const p of obbCorners(o.cuboid)) box.expandByPoint(p.applyMatrix4(v.root.matrixWorld));
      v.cloudBox = box;
    }
    const cams = [...(state.json ? state.json.cameras : []), ...(state.ply ? state.ply.cameras : [])];
    v.setCameras(cams);
    const source = {
      points: state.ply?.name, segments: state.ply && state.json ? `${state.ply.name} labels, ${state.json.name} colours` : null,
      cameras: [state.json?.cameras.length ? state.json.name : null, state.ply?.cameras.length ? state.ply.name : null].filter(Boolean).join(', '),
      labels: state.json?.objects.length ? state.json.name : null, obbs: state.json?.objects.length ? state.json.name : null,
    };
    const canSegment = !!(state.ply?.cloud.arrays.label && state.json?.objects.some((o) => o.rgb));
    const disabled = new Set(LAYERS.map(([k]) => k).filter((k) => !(k === 'segments' ? canSegment : source[k])));
    const wanted = q.get('layers');
    const on = wanted === null ? null : new Set(wanted.split(','));
    for (const k of Object.keys(v.layers)) v.layers[k] = !disabled.has(k) && (on ? on.has(k) : k !== 'segments');
    buildLayerControls(layersBox, v.layers, (k, isOn) => { v.setLayer(k, isOn); setQuery({ layers: layersOn(v) }); }, disabled);
    for (const row of layersBox.querySelectorAll('[data-layer]')) {
      const s = source[row.dataset.layer];
      row.append(el('small', { class: 'muted' }, s ? ` ${s}` : ' (nothing to show)'));
    }
    v.setLayer('points', v.layers.points);
    if (state.json) objectsBox.append(objectsTable(state.json.objects, `The objects of ${state.json.name}`, selection));
    else objectsBox.append(el('p', { class: 'hint' }, 'No scene JSON: open one to list its objects.'));
    fillCameraTable(camerasBody, cams, (i) => v.goToCamera(i));
    const unlocated = state.ply?.unlocated || [];
    camNote.textContent = (cams.length ? 'Camera centres in the files\' common frame, metres. Located cameras are dashed and marked "located".' : '')
      + (unlocated.length ? ` Not located: ${unlocated.join(', ')}.` : '');
    v.select(selection.id);
    v.resize();
    v.resetView();
    v.invalidate();
    view.dataset.objects = String(v.objects.length);
    view.dataset.built = String(seq);
  }

  // read one file: `kind` 'ply' | 'json'; `read` gives the cloud, or the JSON's bytes
  async function open(name, kind, read, size) {
    try {
      if (kind === 'ply') state.ply = { ...plyEntry(name, await read()), size };
      else {
        const { doc, errors: problems } = await sceneFromBytes(await read());
        state.json = { ...sceneEntry(name, doc, problems), size };
      }
      return true;
    } catch (err) {
      errors.append(notice('error', el('strong', {}, `${name} was refused. `), err.message));
      return false;
    }
  }

  const kindOf = (name, type) => (/\.ply$/i.test(name) ? 'ply' : /\.json$/i.test(name) || type === 'application/json' ? 'json' : null);

  file.addEventListener('change', async () => {
    clear(errors);
    const files = [...file.files];
    file.value = '';
    empty.textContent = 'Reading…';
    for (const f of files) {
      const kind = kindOf(f.name, f.type);
      if (!kind) { errors.append(notice('error', el('strong', {}, `${f.name} was refused. `), 'Open a .ply point cloud or a .json scene description.')); continue; }
      await open(f.name, kind, () => (kind === 'ply' ? plyFromDisk(f) : f.arrayBuffer()), f.size);
    }
    empty.textContent = 'Open a PLY file, a scene JSON, or both.';
    setQuery({ ply: null, json: null });  // files from disk cannot come back on a reload
    rebuild();
  });
  for (const t of ['dragenter', 'dragover']) view.addEventListener(t, (e) => { e.preventDefault(); view.classList.add('over'); });
  view.addEventListener('dragleave', () => view.classList.remove('over'));
  view.addEventListener('drop', (e) => {
    e.preventDefault();
    view.classList.remove('over');
    const dt = new DataTransfer();
    for (const f of e.dataTransfer.files) dt.items.add(f);
    file.files = dt.files;
    file.dispatchEvent(new Event('change'));
  });

  // sources named by the URL: files of the service (a job's result or file)
  (async () => {
    const wanted = [['ply', query.get('ply')], ['json', query.get('json')]].filter(([, u]) => u);
    if (!wanted.length) return;
    empty.textContent = 'Loading…';
    for (const [kind, url] of wanted) {
      if (!url.startsWith('/api/')) { errors.append(notice('error', `${url} is not a file of this service.`)); continue; }
      const last = decodeURIComponent(url.split('/').pop());
      const name = last === 'result' ? `${url.split('/').slice(-2, -1)[0]} result` : last;
      await open(name, kind, () => (kind === 'ply' ? plyFromService(url) : fetchBuffer(url)));
    }
    rebuild();
  })();

  return () => { if (viewer) viewer.renderer.dispose(); window.__sceneViewer = null; };
}
