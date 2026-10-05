// Maps (http_server.md "Structure"): the workspace's maps as cards (name, a keyframe thumbnail and
// the summary figures of the map's own metadata) with a filter; a map's page with the embedded
// viewer, the map's objects, its update history with timings, and every export the commands offer
// for a map: one generated form per operation that takes a map (the mode that writes maps is the
// guided update flow; a mode whose output is the browser is the embedded viewer itself).
import { el, clear, notice, fmtTime, fmtSeconds, humanize, objectBadge } from '../dom.js';
import { getJson, enc } from '../api.js';
import { store } from '../store.js';
import { opCard } from '../opcard.js';
import { Selection } from '../selection.js';
import { embeddedViewer } from '../embed.js';
import { objectsTable } from '../jobview.js';
import { sceneObjects } from '/static/viewer/lib/obbs.js';

const HIDDEN = new Set(['name', 'path', 'thumbnail', 'meta']);

function figure(k, v) {
  if (v && typeof v === 'object') {
    return Object.entries(v).filter(([, x]) => x !== null && typeof x !== 'object')
      .map(([kk, x]) => [`${humanize(k)} ${humanize(kk)}`, kk === 'at' && typeof x === 'number' ? fmtTime(x) : kk.endsWith('_s') ? fmtSeconds(x) : String(x)]);
  }
  return [[humanize(k), typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(3) : String(v)]];
}

// the summary figures of a map: every field of its summary, as the service reads it from map.json
export function figures(summary) {
  const out = [];
  for (const [k, v] of Object.entries(summary)) if (!HIDDEN.has(k) && v !== null && v !== undefined) out.push(...figure(k, v));
  return out;
}

function figureList(summary) {
  return el('dl', { class: 'figures' }, ...figures(summary).map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', {}, v))));
}

export function mapsPage(main, { filter }) {
  const input = el('input', { type: 'search', id: 'map-filter', value: filter, placeholder: 'name or any figure', autocomplete: 'off' });
  const grid = el('ul', { class: 'cards', 'aria-live': 'polite', 'aria-busy': 'true' });
  const count = el('p', { class: 'muted', 'aria-live': 'polite' }, 'Loading maps…');
  const writer = [...store.ops.values()].find((o) => o.writesMap);
  main.append(el('div', { class: 'page-head' }, el('h1', {}, 'Maps'),
    writer ? el('a', { href: '#/maps/new', class: 'button primary' }, 'New map') : ''),
  el('div', { class: 'filter' }, el('label', { for: 'map-filter' }, 'Filter maps'), input), count, grid);
  let maps = [];
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    const shown = maps.filter((m) => !q || m.name.toLowerCase().includes(q)
      || figures(m).some(([k, v]) => `${k} ${v}`.toLowerCase().includes(q)));
    clear(grid);
    for (const m of shown) {
      const thumb = m.thumbnail
        ? el('img', { src: `/api/maps/${enc(m.name)}/files/${m.thumbnail.split('/').map(enc).join('/')}`, alt: `A keyframe of ${m.name}`, loading: 'lazy' })
        : el('div', { class: 'no-thumb' }, 'no keyframe');
      grid.append(el('li', { class: 'card map-card', 'data-map': m.name },
        el('a', { href: `#/maps/${enc(m.name)}`, class: 'card-link' }, thumb, el('h2', {}, m.name)), figureList(m)));
    }
    count.textContent = maps.length ? `${shown.length} of ${maps.length} maps` : 'No maps yet: create one with New map.';
    history.replaceState(null, '', q ? `#/maps?filter=${enc(input.value.trim())}` : '#/maps');
  };
  input.addEventListener('input', draw);
  getJson('/api/maps').then((m) => { maps = m; grid.setAttribute('aria-busy', 'false'); draw(); })
    .catch((err) => count.replaceChildren(notice('error', err.message)));
  return null;
}

function historyTable(updates) {
  if (!updates || !updates.length) return el('p', { class: 'muted' }, 'No update recorded.');
  const stageNames = [];
  for (const u of updates) for (const k of Object.keys(u.timings?.stages_s || {})) if (!stageNames.includes(k)) stageNames.push(k);
  const rows = updates.map((u, i) => el('tr', {},
    el('th', { scope: 'row' }, String(u.id ?? i + 1)),
    el('td', {}, typeof u.at === 'number' ? fmtTime(u.at) : String(u.at ?? '')),
    el('td', {}, String(u.kind ?? '')),
    el('td', { class: 'num' }, String((u.inputs || []).length || '')),
    el('td', { class: 'num' }, String((u.frames_added || []).length)),
    el('td', { class: 'num' }, fmtSeconds(u.timings?.total_s)),
    el('td', {}, el('ul', { class: 'stages' }, ...Object.entries(u.timings?.stages_s || {}).map(([k, s]) => el('li', {}, el('span', { class: 'stage-name' }, k), ' ', fmtSeconds(s)))))));
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data', 'data-testid': 'history' },
    el('caption', {}, 'Update history (latest last), with each update\'s per-stage timings'),
    el('thead', {}, el('tr', {}, ...['update', 'at', 'kind', 'inputs', 'frames added', 'total', 'stages'].map((h) => el('th', { scope: 'col' }, h)))),
    el('tbody', {}, ...rows)));
}

export function mapPage(main, { name }) {
  const selection = new Selection();
  const head = el('div', { class: 'page-head' }, el('h1', {}, `Map ${name}`));
  const summary = el('div', {});
  const objectsBox = el('section', { 'aria-labelledby': 'objects-h' }, el('h2', { id: 'objects-h' }, 'Objects'));
  const historyBox = el('section', { 'aria-labelledby': 'history-h' }, el('h2', { id: 'history-h' }, 'Update history'));
  const exportsBox = el('section', { 'aria-labelledby': 'exports-h' }, el('h2', { id: 'exports-h' }, 'Exports'),
    el('p', { class: 'muted' }, 'Everything the commands offer for a map. Each runs as a job; the map is opened read-only.'));
  main.append(head, summary);
  const ops = [...store.ops.values()].filter((o) => o.mapParam);
  const writer = ops.find((o) => o.writesMap);
  if (writer) head.append(el('a', { href: `#/maps/${enc(name)}/update`, class: 'button primary' }, 'Update this map'));
  const viewerOp = ops.find((o) => o.browser);
  const layout = el('div', { class: 'map-layout' });
  if (viewerOp) layout.append(embeddedViewer(`/viewer/map/${enc(name)}/`, `Viewer of map ${name}`, selection));
  layout.append(objectsBox);
  main.append(layout, historyBox, exportsBox);
  const stops = [];
  for (const op of ops.filter((o) => !o.writesMap && !o.browser)) {
    const c = opCard(op, { [op.mapParam.name]: name });
    exportsBox.append(c.el);
    stops.push(c.stop);
  }
  getJson(`/api/maps/${enc(name)}`).then((m) => {
    summary.append(figureList(m));
    historyBox.append(historyTable(m.meta?.updates));
  }).catch((err) => summary.append(notice('error', err.message)));
  getJson(`/api/maps/${enc(name)}/viewer/api/scene`).then((doc) => {
    objectsBox.append(objectsTable(sceneObjects(doc), `The objects of ${name}`, selection));
  }).catch((err) => objectsBox.append(notice('error', err.message)));
  const sel = el('p', { class: 'selected-note', 'aria-live': 'polite' });
  objectsBox.append(sel);
  selection.join((id) => { sel.textContent = id == null ? '' : `Selected object ${id}.`; });
  return () => stops.forEach((s) => s());
}

export { objectBadge };
