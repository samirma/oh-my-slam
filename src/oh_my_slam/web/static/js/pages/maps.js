// Maps (http_server.md "Structure"): the workspace's maps as cards (name, a keyframe thumbnail and
// the summary figures of the map's own metadata) with a filter; a map's page with the embedded
// viewer, the map's objects, its update history with timings, and every export the commands offer
// for a map: one generated form per operation that takes a map (the mode that writes maps is the
// guided update flow; a mode whose output is the browser is the embedded viewer itself).
import { el, clear, notice, fmtSeconds, humanize, objectBadge } from '../dom.js';
import { getJson, enc } from '../api.js';
import { store } from '../store.js';
import { opCard } from '../opcard.js';
import { Selection } from '../selection.js';
import { urlSelection, setQuery } from '../url.js';
import { embeddedViewer } from '../embed.js';
import { objectsTable } from '../jobview.js';
import { sceneObjects } from '/static/viewer/lib/obbs.js';

const HIDDEN = new Set(['name', 'path', 'thumbnail', 'meta']);

// A record's figures, flattened: nested objects give "parent child" names, a list gives its length
// (or its values when they are a few scalars), and a name ending in _s is a duration in seconds.
function figure(k, v, depth = 0) {
  if (v === null || v === undefined) return [];
  if (Array.isArray(v)) {
    const scalars = v.every((x) => x === null || typeof x !== 'object');
    return [[humanize(k), scalars && v.length <= 4 ? v.join(', ') : `${v.length} item${v.length === 1 ? '' : 's'}`]];
  }
  if (typeof v === 'object') {
    if (depth > 3) return [];
    return Object.entries(v).flatMap(([kk, x]) => figure(`${k} ${kk}`, x, depth + 1));
  }
  if (typeof v === 'number') {
    return [[humanize(k), /_s$/.test(k) ? fmtSeconds(v) : Number.isInteger(v) ? String(v) : v.toFixed(3)]];
  }
  return [[humanize(k), String(v)]];
}

// the summary figures of a map: every field of its summary, as the service reads it from map.json
export function figures(summary) {
  const out = [];
  for (const [k, v] of Object.entries(summary)) if (!HIDDEN.has(k)) out.push(...figure(k, v));
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
    setQuery({ filter: input.value.trim() });
  };
  input.addEventListener('input', draw);
  getJson('/api/maps').then((m) => { maps = m; grid.setAttribute('aria-busy', 'false'); draw(); })
    .catch((err) => count.replaceChildren(notice('error', err.message)));
  return null;
}

// The map's update history (map.json's list of updates, latest last): each update record with every
// figure it holds, its per-stage timings included, whatever fields the mapper records.
function historyList(updates) {
  if (!updates || !updates.length) return el('p', { class: 'muted' }, 'No update recorded.');
  return el('ol', { class: 'history', 'data-testid': 'history' }, ...updates.map((u, i) => el('li', { class: 'card' },
    el('h3', {}, `Update ${i + 1}`),
    el('dl', { class: 'figures' }, ...figures(u).map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', {}, v)))))));
}

export function mapPage(main, { name }) {
  const selection = urlSelection(new Selection());
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
    historyBox.append(historyList(m.meta?.updates));
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
