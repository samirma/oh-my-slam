// Maps (http_server.md "Structure"): the workspace's maps as cards (name and summary figures from
// the map's metadata) with a filter (#/maps?filter=<text>), and what the operations that take a map
// give; a map's page (#/maps/<name>) with its summary, the operations that take a map without
// writing it (#/maps/<name>?op=<id>&<parameter>=<value>…: e.g. locating images in it, or its
// bird's-eye image; the form's values are kept in the URL), each with its generated form, the
// running request and the result, its viewer (started and stopped here, with a link to it while it
// runs), its update history with timings (map.json's `updates`), and its removal, after a
// confirmation that it is for good. The operation that writes maps is the guided flow
// (mapflow.js).
import { el, clear, notice, facts, fmtDate, fmtShortDateLines, fmtSeconds, fmtNumber, humanize, sentence } from '../dom.js';
import { getJson, enc, startViewer, stopViewer, removeMap } from '../api.js';
import { confirmAction } from '../dialog.js';
import { store, inferenceSentence } from '../store.js';
import { OpForm, fieldLabel, resultName } from '../form.js';
import { runPanel } from '../runpanel.js';
import { stagesTable } from '../result.js';
import { setQuery, formQuery, queryInitial } from '../url.js';

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

// Words joined as one of them: "a", "a or b", "a, b or c".
export function oneOf(words) {
  return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} or ${words[words.length - 1]}`;
}

// What an operation's result is, from its definition: "Its result: <each result's title
// (FORMAT)>, by its <the parameter that chooses it>."
export function resultsText(op) {
  const results = op.results.map((o) => `${o.title || o.name} (${o.format.toUpperCase()})`);
  const by = op.resultParam;
  return `Its result: ${oneOf(results)}${by && results.length > 1 ? `, by its ${fieldLabel(by).toLowerCase()}` : ''}.`;
}

// What an operation gives, from its definition: "<command>: <what it does>. Its result: …".
export function producesText(op) { return `${op.label}: ${sentence(op.description)} ${resultsText(op)}`; }

// The maps page's lead, in plain words: what a map's page gives, from the results of the
// operations that take a map (`ops`).
export function mapsLead(ops) {
  const titles = [...new Set(ops.flatMap((o) => o.results.map((r) => r.title || r.name)))];
  return 'The maps of this workspace. Open one to see its summary and history'
    + (titles.length ? `, and to get ${oneOf(titles)}.` : '.');
}

// A result's title as a link's text: "the bird's-eye image of the map" → "Bird's-eye image of the
// map".
function linkText(title) { const t = String(title).replace(/^the /, ''); return t.charAt(0).toUpperCase() + t.slice(1); }

// The links a map's card gives straight to an image result of an operation that needs nothing
// but the map (its other parameters optional): [{ text, op, query: {parameter: value} }], each
// result's choosing parameter set to the value that selects it.
export function imageLinks(ops) {
  const out = [];
  for (const op of ops) {
    if (!op.params.every((p) => p === op.mapParam || !p.required)) continue;
    for (const r of op.results) {
      const w = (r.when || []).find((c) => Array.isArray(c.in) && c.in.length);
      if (w && String(r.media_type || '').startsWith('image/')) out.push({ text: linkText(r.title || r.name), op: op.id, query: { [w.option]: w.in[0] } });
    }
  }
  return out;
}

// The route of an image link on the map `name`: the map's page, which runs it at once (`run=1`,
// autoRuns) and shows the image.
export function linkHref(name, link) { return `#/maps/${enc(name)}?${new URLSearchParams({ op: link.op, ...link.query, run: '1' })}`; }

// Whether a map's page asked to run its request at once (`run=1`) may run it unasked: its
// validation (`v`) passed and says it needs no inference server, and the operation writes
// nothing but its result (every output is the response's).
export function autoRuns(op, v) {
  return !!v && v.valid === true && v.inference === false && op.x.outputs.every((o) => o.via === 'stdout');
}

function card(m, links = []) {
  return el('li', { class: 'card map-card', 'data-map': m.name },
    el('h2', {}, el('a', { href: `#/maps/${enc(m.name)}`, class: 'card-link' }, m.name)),
    facts(cardFigures(m)),
    links.length ? el('p', { class: 'card-actions' }, ...links.flatMap((l, i) => [i ? ' · ' : null,
      el('a', { href: linkHref(m.name, l), 'data-action': 'image-link', 'aria-label': `${l.text}: map ${m.name}` }, l.text)])) : null);
}

// Whether a map's card shows for the filter `q` (lower case; '' shows every map): its name or one
// of its figures ("name value") contains it.
export function mapMatches(m, q) {
  return !q || m.name.toLowerCase().includes(q) || figures(m).some(([k, v]) => `${k} ${v}`.toLowerCase().includes(q));
}

// The operations a map's page runs: those that take a map without writing it.
function mapOps() { return [...store.ops.values()].filter((o) => o.mapParam && !o.writesMap); }

export function mapsPage(main, { filter, query = new URLSearchParams() }) {
  const writer = [...store.ops.values()].find((o) => o.writesMap);
  const ops = mapOps();
  const links = imageLinks(ops);
  const other = [...query.keys()].filter((k) => k !== 'filter');  // what the page does not keep
  const input = el('input', { type: 'search', id: 'map-filter', value: filter, autocomplete: 'off', spellcheck: 'false',
    'aria-describedby': 'map-count' });
  const grid = el('ul', { class: 'cards', 'data-testid': 'maps', 'aria-label': 'Maps' });
  const count = el('p', { class: 'muted', id: 'map-count', role: 'status' }, 'Loading the maps…');
  main.append(el('div', { class: 'page-head' }, el('h1', {}, 'Maps'),
    writer ? el('a', { href: '#/maps/new', class: 'button primary', 'data-action': 'new-map' }, 'New map') : null),
  el('p', { class: 'lead', 'data-testid': 'maps-lead' }, mapsLead(ops)),
  el('div', { class: 'filter' }, el('label', { for: 'map-filter' }, 'Filter by name or figure'), input), count, grid);
  let maps = [];
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    const shown = maps.filter((m) => mapMatches(m, q));
    clear(grid).append(...shown.map((m) => card(m, links)));
    count.textContent = maps.length
      ? `${shown.length} of ${maps.length} map${maps.length === 1 ? '' : 's'}${q ? ` match “${input.value.trim()}”` : ''}.`
      : `No map yet${writer ? ': create one with New map' : ''}.`;
    setQuery({ ...Object.fromEntries(other.map((k) => [k, null])), filter: input.value.trim() });
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

// The detections an update tracked (its `objects` record, as the mapper counts them: confirmed or
// not, before the export's checks — not the objects the map lists, which the summary counts):
// { confirmed, unconfirmed, total }, each null when it does not say.
export function trackedCounts(u) {
  const o = u && u.objects && typeof u.objects === 'object' ? u.objects : {};
  const num = (v) => (typeof v === 'number' ? v : null);
  return { confirmed: num(o.confirmed), unconfirmed: num(o.unconfirmed), total: num(o.total) };
}

// The history's Tracked cell of an update (trackedCounts), as lines: ["8 confirmed",
// "+ 22 unconfirmed"]; an older record its total alone; ["—"] when it says none.
export function trackedLines(c) {
  if (c.confirmed == null) return [c.total != null ? String(c.total) : '—'];
  return [`${c.confirmed} confirmed`, ...(c.unconfirmed != null ? [`+ ${c.unconfirmed} unconfirmed`] : [])];
}

// What the Tracked column holds, beside the summary's Objects.
export const TRACKED_TITLE = 'Detections the map tracked at that update, confirmed or not. The summary\'s Objects are '
  + 'those the map lists: confirmed, with an OBB and enough points of their own.';

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
  const counted = updates.map(trackedCounts);
  const trackedCol = counted.some((c) => c.confirmed != null || c.total != null);
  const extra = trackedCol ? 1 : 0;
  const rows = [];
  updates.forEach((u, i) => {
    const n = u.id ?? i + 1;
    const t = u.timings || {};
    const stages = updateStages(t);
    rows.push(el('tr', { 'data-update': n },
      el('th', { scope: 'row' }, `Update ${n}`),
      el('td', { class: 'when', title: u.at ? fmtDate(u.at) : null },
        ...(u.at ? fmtShortDateLines(u.at).flatMap((line, k) => (k ? [el('br'), line] : [line])) : ['—'])),
      ...cols.map(([k]) => el('td', { class: Array.isArray(u[k]) ? 'num' : null }, k in u ? count(u[k]) : '—')),
      trackedCol ? el('td', { class: 'num tracked' }, ...trackedLines(counted[i]).flatMap((line, k) => (k ? [el('br'), line] : [line]))) : null,
      el('td', { class: 'num' }, t.total_s != null ? fmtSeconds(t.total_s) : '—')));
    const more = figures(u, new Set(['id', 'at', 'timings', 'notes', 'inputs', 'frames_added', 'frames_rejected']));
    rows.push(el('tr', { class: 'update-detail' }, el('td', { colspan: cols.length + 3 + extra },
      el('details', { class: 'stages-box' }, el('summary', {}, `Timings and figures of update ${n}`),
        stages.length ? stagesTable(stages, `Stages of update ${n}`) : el('p', { class: 'muted' }, 'No timings recorded.'),
        facts(more)))));
  });
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data updates', 'data-testid': 'history' },
    el('caption', { class: 'visually-hidden' }, 'Updates of the map, oldest first'),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'Update'), el('th', { scope: 'col' }, 'When'),
      ...cols.map(([, label]) => el('th', { scope: 'col' }, label)),
      trackedCol ? el('th', { scope: 'col', class: 'num', title: TRACKED_TITLE }, 'Tracked') : null,
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

export function mapPage(main, { name, op: wanted, query = new URLSearchParams() }) {
  const ops = mapOps();
  const writer = [...store.ops.values()].find((o) => o.writesMap);
  const update = writer ? el('a', { href: `#/maps/${enc(name)}/update`, class: 'button', 'data-action': 'update-map' }, 'Update this map') : null;
  main.append(el('div', { class: 'page-head' }, el('h1', {}, `Map ${name}`), update));
  const summary = el('section', { 'aria-labelledby': 'summary-h' }, el('h2', { id: 'summary-h' }, 'Summary'),
    el('p', { class: 'muted' }, 'Loading…'));
  const runBox = el('section', { 'aria-labelledby': 'run-h', class: 'step', 'data-testid': 'run-on-map' }, el('h2', { id: 'run-h' }, 'Run on this map'));
  const historyBox = el('section', { 'aria-labelledby': 'history-h' }, el('h2', { id: 'history-h' }, 'Update history'));
  const viewerBox = el('section', { 'aria-labelledby': 'viewer-h', 'data-testid': 'viewer' }, el('h2', { id: 'viewer-h' }, 'Viewer'));
  const removeBox = el('section', { 'aria-labelledby': 'remove-h', 'data-testid': 'remove' }, el('h2', { id: 'remove-h' }, 'Remove'));
  main.append(summary, runBox, viewerBox, historyBox, removeBox);
  const viewer = viewerControls(viewerBox, name);
  removeControls(main, removeBox, name, viewer);
  let form = null;
  let panel = null;
  let offQuery = null;
  let kept = [...query.keys()].filter((k) => k !== 'op');  // the query keys to replace: the form's, or unknown
  let runAsked = query.get('run') === '1';  // run=1: the request at once, if it may run unasked
  if (ops.length) {
    let op = ops.find((o) => o.id === wanted) || ops[0];
    const formBox = el('div', {});
    // one operation: said in a sentence; several: a choice between them
    const about = (o) => [` ${sentence(o.description)} ${inferenceSentence(o)}`];
    const choices = ops.length === 1
      ? el('div', { 'data-testid': 'operations', 'data-op': op.id }, el('p', { class: 'op-lead' }, el('code', {}, op.label),
        `: ${sentence(op.description)} ${inferenceSentence(op)}`))
      : el('fieldset', { class: 'modes', 'data-testid': 'operations' }, el('legend', {}, 'Operation'),
        ...ops.map((o) => {
          const id = `op-${o.id}`;
          const r = el('input', { type: 'radio', name: 'op', id, value: o.id, 'aria-describedby': `${id}-desc` });
          r.checked = o === op;
          r.addEventListener('change', () => { if (r.checked) choose(o); });
          return el('div', { class: 'mode' }, r, el('label', { for: id }, el('code', {}, o.label)),
            el('span', { class: 'muted', id: `${id}-desc` }, ...about(o)));
        }));
    const lead = el('p', { class: 'lead', 'data-testid': 'produces' });
    const runPanelBox = el('div', {});
    runBox.append(choices, lead, formBox, runPanelBox);
    const choose = (o, initial = {}) => {
      form?.destroy();
      panel?.dispose();
      offQuery?.();
      op = o;
      lead.textContent = resultsText(o);
      formBox.replaceChildren();
      form = new OpForm(formBox, o, { fixed: { [o.mapParam.name]: name }, initial });
      const f = form;
      panel = runPanel({ op: o, form, label: `Run ${o.label}`,
        downloadName: (fmt) => resultName([o.id, name, ...f.variant()], fmt),
        onEnd: () => { loadSummary(); } });
      const p = panel;
      runPanelBox.replaceChildren(panel.el);
      const keep = () => {
        const values = f.queryValues();
        setQuery(formQuery(o.id, values, kept.filter((k) => !(k in values))));
        kept = Object.keys(values);
      };
      offQuery = form.onValidate(keep);  // after each check (the first one comes at once); drops run
      if (runAsked) {  // its first check decides: run it now, or show where to run it
        runAsked = false;
        const offAsk = form.onValidate((v) => {
          offAsk();
          if (panel !== p) return;
          if (autoRuns(o, v)) { p.run({ focus: true }); return; }
          p.submit.scrollIntoView({ block: 'center' });
          p.submit.focus();
        });
      }
      form.changed();
    };
    choose(op, op.id === wanted || !wanted ? queryInitial(query, op.params.map((p) => p.name)) : {});
  } else {
    runBox.append(el('p', { class: 'muted' }, 'No operation of this service takes a map.'));
  }

  function loadSummary() {
    getJson(`/api/maps/${enc(name)}`).then((m) => {
      summary.replaceChildren(summary.firstChild, facts(figures(m)));
      historyBox.replaceChildren(historyBox.firstChild, history(m.meta?.updates));
      viewer.show(m.viewer);
    }).catch((err) => {
      const missing = err.status === 404;
      summary.replaceChildren(summary.firstChild, notice('error', `${err.message}`,
        missing ? el('p', {}, el('a', { href: '#/maps' }, 'See the maps of this workspace'),
          ...(writer ? [' or ', el('a', { href: '#/maps/new', 'data-action': 'new-map' }, 'create a new map'), '.'] : ['.'])) : null));
      historyBox.replaceChildren(historyBox.firstChild);
      if (missing) {  // no map: nothing to run on it, update, view or remove, nor kept in the URL
        panel?.dispose(); panel = null;
        form?.destroy(); form = null;
        offQuery?.();
        setQuery(Object.fromEntries([...kept, ...query.keys(), 'op'].map((k) => [k, null])));
        for (const box of [runBox, viewerBox, historyBox, removeBox, update]) box?.remove();
      } else viewer.show(null);
    });
  }
  loadSummary();
  return () => { offQuery?.(); panel?.dispose(); form?.destroy(); };
}
