// Maps (http_server.md "Structure"): the workspace's maps as cards (name and summary figures from
// the map's metadata) with a filter (#/maps?filter=<text>); a map's page (#/maps/<name>) with its
// summary, its viewer (started and stopped here, with a link to it while it runs), its update
// history with timings (map.json's `updates`), the operations that take a map without writing it
// (#/maps/<name>?op=<id>, e.g. locating images in it), each with its generated form, the running
// request and the result, and its removal, after a confirmation that it is for good. The
// operation that writes maps is the guided flow (mapflow.js).
import { el, clear, notice, facts, fmtDate, fmtSeconds, fmtNumber, humanize } from '../dom.js';
import { getJson, enc, startViewer, stopViewer, removeMap } from '../api.js';
import { confirmAction } from '../dialog.js';
import { store } from '../store.js';
import { OpForm } from '../form.js';
import { runPanel } from '../runpanel.js';
import { stagesTable } from '../result.js';
import { setQuery } from '../url.js';

// the map store's own bookkeeping, not figures of the map, and the state of its viewer
const HIDDEN = new Set(['name', 'path', 'meta', 'next_object_id', 'next_frame_index', 'updates_count',
  'scale_count', 'map_frame_count', 'viewer']);

// One figure of a map's metadata, readable: *_at a date, *_s a duration, a list its length.
function figureValue(k, v) {
  if (v === null || v === undefined) return '—';
  if (/(^|_)at$/.test(k) && typeof v === 'number') return fmtDate(v);
  if (/_s$/.test(k) && typeof v === 'number') return fmtSeconds(v);
  if (Array.isArray(v)) return v.every((x) => x === null || typeof x !== 'object') && v.length <= 4 ? v.join(', ') || '—' : String(v.length);
  if (typeof v === 'number') return fmtNumber(v);
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return String(v);
}

// A figure's name: a duration's `_s` and a date's `_at` are said by its value
function label(k) { return humanize(String(k).replace(/_s$/, '').replace(/_at$/, '')); }

// The flat figures of a record (nested records one level down, as "parent child").
export function figures(record, hidden = HIDDEN) {
  const out = [];
  for (const [k, v] of Object.entries(record || {})) {
    if (hidden.has(k)) continue;
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [kk, x] of Object.entries(v)) {
        if (x === null || typeof x !== 'object' || Array.isArray(x)) out.push([`${humanize(k)} ${label(kk).toLowerCase()}`, figureValue(kk, x)]);
      }
    } else {
      out.push([label(k), figureValue(k, v)]);
    }
  }
  return out;
}

// The figures a card shows first, when the summary has them
const CARD = [['frames', 'Frames'], ['objects', 'Objects'], ['update_count', 'Updates'], ['updated_at', 'Updated']];

// A map card's figures: those of CARD the summary has, and how long its last update took; else
// its first four figures.
export function cardFigures(m) {
  const shown = CARD.filter(([k]) => k in m).map(([k, label]) => [label, figureValue(k, m[k])]);
  const last = m.last_update;
  if (last && last.total_s != null) shown.push(['Last update took', fmtSeconds(last.total_s)]);
  return shown.length ? shown : figures(m).slice(0, 4);
}

function card(m) {
  return el('li', { class: 'card map-card', 'data-map': m.name },
    el('h2', {}, el('a', { href: `#/maps/${enc(m.name)}`, class: 'card-link' }, m.name)),
    facts(cardFigures(m)));
}

// Whether a map's card shows for the filter `q` (lower case; '' shows every map): its name or one
// of its figures ("name value") contains it.
export function mapMatches(m, q) {
  return !q || m.name.toLowerCase().includes(q) || figures(m).some(([k, v]) => `${k} ${v}`.toLowerCase().includes(q));
}

export function mapsPage(main, { filter }) {
  const writer = [...store.ops.values()].find((o) => o.writesMap);
  const input = el('input', { type: 'search', id: 'map-filter', value: filter, autocomplete: 'off', spellcheck: 'false',
    'aria-describedby': 'map-count' });
  const grid = el('ul', { class: 'cards', 'data-testid': 'maps', 'aria-label': 'Maps' });
  const count = el('p', { class: 'muted', id: 'map-count', role: 'status' }, 'Loading the maps…');
  main.append(el('div', { class: 'page-head' }, el('h1', {}, 'Maps'),
    writer ? el('a', { href: '#/maps/new', class: 'button primary', 'data-action': 'new-map' }, 'New map') : null),
  el('p', { class: 'lead' }, 'The maps of this workspace. Open one to see its summary and update history, and to run the operations that take a map.'),
  el('div', { class: 'filter' }, el('label', { for: 'map-filter' }, 'Filter by name or figure'), input), count, grid);
  let maps = [];
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    const shown = maps.filter((m) => mapMatches(m, q));
    clear(grid).append(...shown.map(card));
    count.textContent = maps.length
      ? `${shown.length} of ${maps.length} map${maps.length === 1 ? '' : 's'}${q ? ` match “${input.value.trim()}”` : ''}.`
      : `No map yet${writer ? ': create one with New map' : ''}.`;
    setQuery({ filter: input.value.trim() });
  };
  input.addEventListener('input', draw);
  getJson('/api/maps').then((m) => { maps = m; draw(); })
    .catch((err) => count.replaceChildren(notice('error', `The maps could not be listed: ${err.message}`)));
  return null;
}

// The map's update history (map.json's `updates`, oldest first): one row per update with its
// figures (a list counts its entries), and per update its per-stage timings (the command's own
// timing record) and every other figure it records.
const UPDATE_COLUMNS = [['kind', 'Input'], ['inputs', 'Files'], ['frames_added', 'Frames added'],
  ['frames_rejected', 'Rejected'], ['sfm', 'Registration']];

// An update's figure in the history table: a list counts its entries.
export function count(v) { return Array.isArray(v) ? String(v.length) : figureValue('', v); }

// The per-stage timings of an update's timing record (map.json updates[].timings), `total` last.
export function updateStages(t) {
  const stages = Object.entries(t.stages_s || {}).map(([name, sec]) => ({ name, ms: sec * 1000 }));
  if (t.total_s != null) stages.push({ name: 'total', ms: t.total_s * 1000 });
  return stages;
}

function history(updates) {
  if (!updates || !updates.length) return el('p', { class: 'muted' }, 'No update recorded.');
  const cols = UPDATE_COLUMNS.filter(([k]) => updates.some((u) => k in u));
  const objectsCol = updates.some((u) => u.objects && typeof u.objects === 'object' && 'total' in u.objects);
  const rows = [];
  updates.forEach((u, i) => {
    const n = u.id ?? i + 1;
    const t = u.timings || {};
    const stages = updateStages(t);
    rows.push(el('tr', { 'data-update': n },
      el('th', { scope: 'row' }, `Update ${n}`),
      el('td', {}, u.at ? fmtDate(u.at) : '—'),
      ...cols.map(([k]) => el('td', { class: Array.isArray(u[k]) ? 'num' : null }, k in u ? count(u[k]) : '—')),
      objectsCol ? el('td', { class: 'num' }, u.objects?.total != null ? String(u.objects.total) : '—') : null,
      el('td', { class: 'num' }, t.total_s != null ? fmtSeconds(t.total_s) : '—')));
    const more = figures(u, new Set(['id', 'at', 'timings', 'notes', 'inputs', 'frames_added', 'frames_rejected']));
    rows.push(el('tr', { class: 'update-detail' }, el('td', { colspan: cols.length + 3 + (objectsCol ? 1 : 0) },
      el('details', { class: 'stages-box' }, el('summary', {}, `Timings and figures of update ${n}`),
        stages.length ? stagesTable(stages, `Stages of update ${n}`) : el('p', { class: 'muted' }, 'No timings recorded.'),
        facts(more)))));
  });
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data updates', 'data-testid': 'history' },
    el('caption', { class: 'visually-hidden' }, 'Updates of the map, oldest first'),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'Update'), el('th', { scope: 'col' }, 'When'),
      ...cols.map(([, label]) => el('th', { scope: 'col' }, label)),
      objectsCol ? el('th', { scope: 'col', class: 'num' }, 'Objects') : null,
      el('th', { scope: 'col', class: 'num' }, 'Time'))),
    el('tbody', {}, ...rows)));
}

// What the state of a map's viewer ({running, url}, as the service answers it) says.
export function viewerStatus(v) {
  return v && v.running
    ? 'The viewer runs. Its page is served on the machine that runs the service only: open it there.'
    : 'The viewer is stopped.';
}

// The confirmation of a map's removal: it states that the map is deleted for good (and that its
// running viewer stops).
export function removeQuestion(name, viewer) {
  return {
    title: `Remove map ${name} for good?`,
    body: [`Map ${name} is deleted from this workspace for good: its keyframes, objects and update history cannot be recovered.`,
      ...(viewer && viewer.running ? ['Its viewer is stopped first.'] : [])],
    yes: 'Remove the map for good', no: 'Keep the map', danger: true,
  };
}

// The viewer's controls in `box`: its state, a button that starts or stops it (the action answers
// at once; the button stays focusable), a link to it while it runs, and why an action failed.
// Returns { show(state), state }.
function viewerControls(box, name) {
  const status = el('p', { role: 'status', 'data-testid': 'viewer-status' }, 'Loading…');
  const button = el('button', { type: 'button', 'data-action': 'viewer', 'aria-describedby': 'viewer-about' }, 'Start the viewer');
  const link = el('a', { target: '_blank', rel: 'noopener', 'data-testid': 'viewer-link', hidden: true }, `Open the viewer of map ${name}`);
  const problem = el('div', {});
  box.append(el('p', { class: 'muted', id: 'viewer-about' }, 'The viewer shows the map in 3D, with its objects and cameras, in a page of its own.'),
    status, el('div', { class: 'actions' }, button, link), problem);
  let state = { running: false, url: null };
  let busy = false;
  const show = (v) => {
    state = v || { running: false, url: null };
    status.textContent = viewerStatus(state);
    button.textContent = state.running ? 'Stop the viewer' : 'Start the viewer';
    link.hidden = !state.running;
    if (state.running) link.href = state.url; else link.removeAttribute('href');
    box.dataset.running = String(state.running);
  };
  button.addEventListener('click', async () => {
    if (busy) return;
    const running = state.running;
    busy = true;
    button.setAttribute('aria-busy', 'true');
    problem.replaceChildren();
    status.textContent = running ? 'Stopping the viewer…' : 'Starting the viewer…';
    try {
      show(await (running ? stopViewer(name) : startViewer(name)));
    } catch (err) {
      show(state);
      problem.replaceChildren(notice('error', `The viewer could not be ${running ? 'stopped' : 'started'}: ${err.message}`));
    } finally {
      busy = false;
      button.removeAttribute('aria-busy');
    }
  });
  return { show, get state() { return state; } };
}

// The map's removal in `box`: a button that asks first, then deletes the map; the page then says
// it is gone. `viewer` is the viewer's controls (its state goes into the question).
function removeControls(main, box, name, viewer) {
  const problem = el('div', {});
  const button = el('button', { type: 'button', class: 'danger', 'data-action': 'remove-map' }, 'Remove this map');
  box.append(el('p', {}, 'Removing the map deletes it from this workspace for good.'), el('div', { class: 'actions' }, button), problem);
  button.addEventListener('click', async () => {
    problem.replaceChildren();
    if (!(await confirmAction(removeQuestion(name, viewer.state)))) return;
    button.setAttribute('aria-busy', 'true');
    try {
      await removeMap(name);
    } catch (err) {
      button.removeAttribute('aria-busy');
      problem.replaceChildren(notice('error', `Map ${name} was not removed: ${err.message}`));
      return;
    }
    const h1 = el('h1', { tabindex: '-1' }, `Map ${name} removed`);
    main.replaceChildren(h1, notice('ok', `Map ${name} was deleted from this workspace for good. `,
      el('a', { href: '#/maps' }, 'See the maps of this workspace.')));
    document.title = `${h1.textContent} — oh-my-slam`;
    h1.focus();
  });
}

export function mapPage(main, { name, op: wanted }) {
  const ops = [...store.ops.values()].filter((o) => o.mapParam && !o.writesMap);
  const writer = [...store.ops.values()].find((o) => o.writesMap);
  main.append(el('div', { class: 'page-head' }, el('h1', {}, `Map ${name}`),
    writer ? el('a', { href: `#/maps/${enc(name)}/update`, class: 'button', 'data-action': 'update-map' }, 'Update this map') : null));
  const summary = el('section', { 'aria-labelledby': 'summary-h' }, el('h2', { id: 'summary-h' }, 'Summary'),
    el('p', { class: 'muted' }, 'Loading…'));
  const runBox = el('section', { 'aria-labelledby': 'run-h', class: 'step' }, el('h2', { id: 'run-h' }, 'Run on this map'));
  const historyBox = el('section', { 'aria-labelledby': 'history-h' }, el('h2', { id: 'history-h' }, 'Update history'));
  const viewerBox = el('section', { 'aria-labelledby': 'viewer-h', 'data-testid': 'viewer' }, el('h2', { id: 'viewer-h' }, 'Viewer'));
  const removeBox = el('section', { 'aria-labelledby': 'remove-h', 'data-testid': 'remove' }, el('h2', { id: 'remove-h' }, 'Remove'));
  main.append(summary, viewerBox, runBox, historyBox, removeBox);
  const viewer = viewerControls(viewerBox, name);
  removeControls(main, removeBox, name, viewer);
  let form = null;
  let panel = null;
  if (ops.length) {
    let op = ops.find((o) => o.id === wanted) || ops[0];
    const formBox = el('div', {});
    const choices = el('fieldset', { class: 'modes', 'data-testid': 'operations' }, el('legend', {}, 'Operation'),
      ...ops.map((o) => {
        const id = `op-${o.id}`;
        const r = el('input', { type: 'radio', name: 'op', id, value: o.id, 'aria-describedby': `${id}-desc` });
        r.checked = o === op;
        r.addEventListener('change', () => { if (r.checked) choose(o); });
        return el('div', { class: 'mode' }, r, el('label', { for: id }, el('code', {}, o.label)),
          el('span', { class: 'muted', id: `${id}-desc` }, ` ${o.description} · ${o.x.inference_text}`));
      }));
    const runPanelBox = el('div', {});
    runBox.append(choices, formBox, runPanelBox);
    const choose = (o) => {
      form?.destroy();
      panel?.dispose();
      op = o;
      formBox.replaceChildren();
      form = new OpForm(formBox, o, { fixed: { [o.mapParam.name]: name } });
      panel = runPanel({ op: o, form, label: `Run ${o.label}`,
        downloadName: (fmt) => `${o.id}-${name}${fmt ? `.${fmt}` : ''}`,
        onEnd: () => { loadSummary(); } });
      runPanelBox.replaceChildren(panel.el);
      setQuery({ op: o.id });
      form.changed();
    };
    choose(op);
  } else {
    runBox.append(el('p', { class: 'muted' }, 'No operation of this service takes a map.'));
  }

  function loadSummary() {
    getJson(`/api/maps/${enc(name)}`).then((m) => {
      summary.replaceChildren(summary.firstChild, facts(figures(m)));
      historyBox.replaceChildren(historyBox.firstChild, history(m.meta?.updates));
      viewer.show(m.viewer);
    }).catch((err) => {
      summary.replaceChildren(summary.firstChild, notice('error', `${err.message}`,
        err.status === 404 ? el('p', {}, el('a', { href: '#/maps' }, 'See the maps of this workspace.')) : null));
      historyBox.replaceChildren(historyBox.firstChild);
      if (err.status === 404) { viewerBox.remove(); removeBox.remove(); } else viewer.show(null);
    });
  }
  loadSummary();
  return () => { panel?.dispose(); form?.destroy(); };
}
