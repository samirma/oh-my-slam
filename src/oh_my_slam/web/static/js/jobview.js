// One job: its kind, inputs, state, times, live progress and per-stage timings, its failure message,
// and on success its result (http_server.md "Image", "Jobs"): the images and tables the command
// wrote, rendered; the embedded viewer; a download for the result and for every file. What is
// shown comes from the job record and its file list, never from a list of commands: a new output
// file is a new download (and, if it is an image or a table, a new rendering).
import { el, clear, fmtTime, fmtSeconds, fmtBytes, notice, stateBadge, objectBadge, humanize } from './dom.js';
import { getJson, enc } from './api.js';
import { store, PATH_IN } from './store.js';
import { actionButtons, TERMINAL } from './jobactions.js';
import { Selection, linkRows } from './selection.js';
import { urlSelection } from './url.js';
import { segmentedImage } from './segimage.js';
import { embeddedViewer } from './embed.js';
import { sceneObjects } from '/static/viewer/lib/obbs.js';

export function jobInputs(job) {
  const op = store.ops.get(job.operation);
  if (!op) return [];
  return op.params.filter((p) => PATH_IN.includes(p.kind) && job.params[p.name] != null)
    .map((p) => [p, [].concat(job.params[p.name])]);
}

export function inputsText(job) {
  return jobInputs(job).map(([p, v]) => `${p.flag} ${v.map((x) => String(x).split('/').pop()).join(' ')}`).join('; ');
}

export function progressBar(job) {
  // the viewer step after a command reports apart from the command's own stages
  const vp = job.viewer_progress;
  const pr = vp || job.progress;
  const wrap = el('div', { class: 'progress' });
  const stage = vp ? `preparing the viewer${vp.stage ? ` (${vp.stage})` : ''}`
    : job.stage || (job.state === 'queued' ? 'waiting for its turn' : 'starting');
  if (pr && pr.total) {
    const pct = Math.round((100 * pr.done) / pr.total);
    wrap.append(el('progress', { max: pr.total, value: pr.done, 'aria-label': `Stage ${stage}: ${pr.done} of ${pr.total}` }),
      el('span', {}, `${stage}: ${pr.done} / ${pr.total} (${pct} %)`));
  } else {
    wrap.append(el('progress', { 'aria-label': `Stage ${stage}` }), el('span', {}, stage));
  }
  return wrap;
}

export function stagesList(job) {
  if (!job.stages || !job.stages.length) return el('span', { class: 'muted' }, job.state === 'queued' ? 'not started' : '—');
  return el('ul', { class: 'stages' }, ...job.stages.map((s) => el('li', {}, el('span', { class: 'stage-name' }, s.stage), ' ', fmtSeconds(s.seconds))));
}

function duration(job) {
  if (!job.started_at) return '';
  return fmtSeconds((job.ended_at || Date.now() / 1000) - job.started_at);
}

function isScene(doc) { return doc && typeof doc === 'object' && doc.openlabel && typeof doc.openlabel === 'object'; }

function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += c;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.length > 1 || r[0] !== '');
}

// A CSV file as a table; with `id` and `color_hex` columns its rows are objects (swatch with the id,
// linked to the page's selection).
export function csvTable(text, caption, selection) {
  const [head, ...rows] = parseCsv(text);
  const idCol = head.indexOf('id'), hexCol = head.indexOf('color_hex'), labelCol = head.indexOf('label');
  const objects = idCol >= 0;
  const tbody = el('tbody', {}, ...rows.map((r) => el('tr', objects ? { 'data-id': r[idCol], tabindex: '0' } : {},
    ...r.map((v, i) => (i === idCol && hexCol >= 0
      ? el('td', {}, objectBadge(v, labelCol >= 0 ? '' : null, r[hexCol]))
      : el('td', { class: /^-?[\d.]+(e-?\d+)?$/i.test(v) ? 'num' : null }, v))))));
  const table = el('table', { class: 'data' }, el('caption', {}, caption),
    el('thead', {}, el('tr', {}, ...head.map((h) => el('th', { scope: 'col' }, h)))), tbody);
  if (objects && selection) linkRows(tbody, selection);
  const list = objects ? rows.map((r) => ({ id: Number(r[idCol]), label: labelCol >= 0 ? r[labelCol] : '', hex: hexCol >= 0 ? r[hexCol] : null })) : [];
  return { el: el('div', { class: 'table-wrap' }, table), objects: list };
}

// The objects of a scene document as a table linked to the page's selection.
export function objectsTable(objects, caption, selection) {
  const tbody = el('tbody', {}, ...objects.map((o) => el('tr', { 'data-id': o.id, tabindex: '0' },
    el('td', {}, objectBadge(o.id, o.label, o.hex)),
    el('td', { class: 'num' }, o.score != null ? o.score.toFixed(2) : ''),
    el('td', { class: 'num' }, o.dims.map((d) => d.toFixed(2)).join(' × ')),
    el('td', { class: 'num' }, o.volume ? o.volume.toFixed(3) : ''))));
  if (!objects.length) tbody.append(el('tr', {}, el('td', { colspan: 4, class: 'muted' }, 'No objects.')));
  const table = el('table', { class: 'data objects', 'data-testid': 'objects' }, el('caption', {}, caption),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'object'), el('th', { scope: 'col', class: 'num' }, 'score'),
      el('th', { scope: 'col', class: 'num' }, 'W × D × H (m)'), el('th', { scope: 'col', class: 'num' }, 'volume (m³)'))), tbody);
  if (selection) linkRows(tbody, selection);
  return el('div', { class: 'table-wrap' }, table);
}

function sceneLink(entries) {
  const q = new URLSearchParams();
  for (const [k, url] of entries) q.append(k, url);
  return `#/scene?${q}`;
}

function kindOf(name, media) {
  if (/\.ply$/i.test(name)) return 'ply';
  if (/\.json$/i.test(name) || media === 'application/json') return 'json';
  if ((media || '').startsWith('image/')) return 'image';
  if (/\.csv$/i.test(name) || media === 'text/csv') return 'csv';
  return 'other';
}

// The map a job of a map operation (mapper.sh update / locate, segment.sh -m) worked on, by the
// operation's map parameter (`<name>` or `maps/<name>`), when the service offers a map viewer (a
// mode whose output is the browser takes a map); else null.
function mapOf(job) {
  const p = store.ops.get(job.operation)?.mapParam;
  const value = p ? job.params[p.name] : null;
  if (value == null || ![...store.ops.values()].some((o) => o.browser && o.mapParam)) return null;
  return String(value).replace(/^maps\//, '').replace(/\/+$/, '');
}

// The result section of a succeeded job.
async function renderResult(box, job, selection) {
  const files = await getJson(`/api/jobs/${enc(job.id)}/files`).catch(() => []);
  const entries = [];
  if (job.result) entries.push({ name: job.result.name, url: job.result.url, label: 'Result', media: null, result: true });
  for (const f of files) if (!(job.result && f.path === job.result.name)) entries.push({ name: f.path, url: f.url, size: f.size, media: f.media_type });
  for (const e of entries) e.kind = kindOf(e.name, e.media);

  // the objects, for the rendered images and tables: a scene document's, else a catalogue's
  let objects = [];
  let sceneDoc = null;
  for (const e of entries.filter((x) => x.kind === 'json')) {
    try {
      const doc = await getJson(e.url);
      if (isScene(doc)) { sceneDoc = doc; objects = sceneObjects(doc); break; }
    } catch { /* not JSON after all */ }
  }
  const rendered = el('div', { class: 'rendered' });
  const tables = [];
  for (const e of entries.filter((x) => x.kind === 'csv')) {
    const text = await (await fetch(e.url)).text();
    const t = csvTable(text, e.name, selection);
    if (!objects.length) objects = t.objects;
    tables.push(t.el);
  }
  // an image whose output the API marks as painted in the objects' colours has per-object regions
  const outputs = store.ops.get(job.operation)?.x.outputs || [];
  const regions = (name) => outputs.some((o) => o.object_regions && o.name === name.split('/').pop());
  for (const e of entries.filter((x) => x.kind === 'image')) {
    const out = outputs.find((o) => o.name === e.name.split('/').pop());
    rendered.append(el('div', { class: 'card' }, el('h3', {}, e.name),
      regions(e.name)
        ? segmentedImage(e.url, objects, selection, `${e.name}: ${out.text}`)
        : el('img', { src: e.url, alt: out ? `${e.name}: ${out.text}` : e.name, class: 'result-image' })));
  }
  if (!tables.length && sceneDoc) tables.push(objectsTable(objects, 'Objects of the scene', selection));
  for (const t of tables) rendered.append(el('div', { class: 'card wide' }, t));

  const ply = entries.find((x) => x.kind === 'ply');
  const json = entries.find((x) => x.kind === 'json' && x.result) || entries.find((x) => x.kind === 'json');
  const downloads = el('ul', { class: 'downloads', 'data-testid': 'downloads' }, ...entries.map((e) => el('li', {},
    el('a', { href: e.url, download: e.name.split('/').pop(), class: 'download' }, `Download ${e.result ? `the result (${e.name})` : e.name}`),
    e.size != null ? el('span', { class: 'muted' }, ` ${fmtBytes(e.size)}`) : '',
    e.kind === 'ply' || (e.kind === 'json' && (e === json))
      ? el('a', { href: sceneLink(e.kind === 'ply' ? [['ply', e.url], ...(json ? [['json', json.url]] : [])] : [['json', e.url], ...(ply ? [['ply', ply.url]] : [])]), class: 'scene-link' }, ' Open in the 3D scene viewer')
      : '')));
  clear(box);
  if (job.viewer_error) {
    const v = job.viewer_error;
    box.append(notice('error', el('strong', {}, v.code === 'cancelled' ? 'The viewer was cancelled. ' : 'The viewer could not be prepared. '),
      v.code === 'cancelled' ? 'The job was cancelled while it prepared the viewer; the result below stands.' : `${v.message} (${v.code}). The result below stands.`));
  }
  if (job.viewer) box.append(embeddedViewer(job.viewer, `Viewer of job ${job.id}`, selection));
  else {
    const map = mapOf(job);
    if (map) box.append(embeddedViewer(`/viewer/map/${enc(map)}/`, `Viewer of map ${map}`, selection));
  }
  if (rendered.children.length) box.append(rendered);
  box.append(el('h3', {}, 'Downloads'), entries.length ? downloads : el('p', { class: 'muted' }, 'This job wrote no file.'));
}

// What the command printed on stderr (/api/jobs/<id>/log), in its own words: its warnings and
// messages, and its `timings:` line (per-stage timings) shown on its own.
const TIMINGS_LINE = /\btimings: total /;
export function jobLog(text) {
  const lines = text.split('\n').filter((l) => l !== '');
  const timing = lines.filter((l) => TIMINGS_LINE.test(l));
  const other = lines.filter((l) => !TIMINGS_LINE.test(l));
  const box = el('section', { class: 'job-log', 'data-testid': 'job-log', 'aria-label': 'Command output' });
  box.append(el('h3', {}, 'Command output'));
  if (timing.length) box.append(el('p', { class: 'timings-line', 'data-testid': 'timings-line' }, ...timing.map((l) => el('code', {}, l))));
  box.append(other.length
    ? el('pre', { class: 'log', tabindex: '0', 'aria-label': 'stderr of the command' }, other.join('\n'))
    : el('p', { class: 'muted' }, lines.length ? 'No other message.' : 'The command printed nothing.'));
  return box;
}

// The job's live view into `container`; returns a function that stops it.
export function renderJob(container, jobId, { heading = 'h2', compact = false } = {}) {
  const selection = urlSelection(new Selection());
  const head = el('div', { class: 'job-head' });
  const body = el('div', { class: 'job-body' });
  const result = el('div', { class: 'job-result', 'data-testid': 'result' });
  const logBox = el('div', { class: 'job-log-box' });
  const errors = el('div', { 'aria-live': 'assertive' });
  const section = el('section', { class: 'job', 'data-job': jobId, 'aria-label': `Job ${jobId}` }, head, errors, body, result, logBox);
  container.append(section);
  let shownResult = false;
  let lastState = null;
  let logKey = null;  // the log is read again when the job's lines or state change
  function drawLog(job) {
    const key = `${job.state}|${(job.log_tail || []).length}|${(job.log_tail || []).at(-1) || ''}`;
    if (key === logKey || job.state === 'queued') return;
    logKey = key;
    fetch(`/api/jobs/${enc(job.id)}/log`).then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
      .then((text) => { if (logKey === key) logBox.replaceChildren(jobLog(text)); })
      .catch(() => { /* the job's page still shows the rest */ });
  }

  function draw(job) {
    section.dataset.state = job.state;
    if (job.state !== lastState) {  // the buttons stay put (and keep focus) while a job progresses
      clear(head).append(
        el(heading, {}, `${job.label} `, stateBadge(job.state)),
        el('div', { class: 'actions' }, ...actionButtons(job, { onError: (err) => errors.replaceChildren(notice('error', err.message)) })),
      );
    }
    const dl = el('dl', { class: 'facts' },
      el('div', {}, el('dt', {}, 'Job'), el('dd', {}, el('a', { href: `#/jobs/${job.id}` }, job.id))),
      el('div', {}, el('dt', {}, 'Command'), el('dd', {}, el('code', {}, (job.command || []).join(' ')))),
      el('div', {}, el('dt', {}, 'Inputs'), el('dd', {}, inputsText(job) || '—')),
      el('div', {}, el('dt', {}, 'Submitted'), el('dd', {}, fmtTime(job.created_at))),
      el('div', {}, el('dt', {}, 'Started'), el('dd', {}, fmtTime(job.started_at) || '—')),
      el('div', {}, el('dt', {}, 'Ended'), el('dd', {}, fmtTime(job.ended_at) || '—')),
      el('div', {}, el('dt', {}, 'Duration'), el('dd', { 'data-fact': 'duration' }, duration(job) || '—')),
      el('div', {}, el('dt', {}, 'Stages'), el('dd', {}, stagesList(job))));
    if (job.resubmitted_from) dl.append(el('div', {}, el('dt', {}, 'Re-submission of'), el('dd', {}, el('a', { href: `#/jobs/${job.resubmitted_from}` }, job.resubmitted_from))));
    clear(body);
    if (!TERMINAL.includes(job.state)) body.append(progressBar(job));
    if (job.state === 'failed' || (job.state === 'cancelled' && job.error)) {
      const e = job.error || {};
      body.append(notice(job.state === 'failed' ? 'error' : 'warn', el('strong', {}, job.state === 'failed' ? 'Failed: ' : 'Cancelled: '),
        e.message || `exit status ${job.exit_code}`, e.code ? el('span', { class: 'muted' }, ` (${e.code})`) : ''));
    }
    if (!compact) body.append(dl);
    if (job.state === 'succeeded' && !shownResult) {
      shownResult = true;
      result.replaceChildren(el('p', { class: 'muted' }, 'Loading the result…'));
      renderResult(result, job, selection).catch((err) => result.replaceChildren(notice('error', `The result could not be shown: ${err.message}`)));
    }
    drawLog(job);
    if (job.state !== lastState) { lastState = job.state; container.dispatchEvent(new CustomEvent('jobstate', { detail: job })); }
  }

  const off = store.on((what, j) => { if (what === 'job' && j.id === jobId) draw(j); });
  const known = store.jobs.get(jobId);
  if (known) draw(known);
  getJson(`/api/jobs/${enc(jobId)}`).then((j) => { store.jobs.set(j.id, j); draw(j); })
    .catch((err) => { head.replaceChildren(el(heading, {}, `Job ${jobId}`)); body.replaceChildren(notice('error', err.message)); });
  const tick = setInterval(() => {  // the running time
    const j = store.jobs.get(jobId);
    const dd = section.querySelector('[data-fact="duration"]');
    if (j && j.state === 'running' && dd) dd.textContent = duration(j);
  }, 1000);
  return () => { off(); clearInterval(tick); };
}

export { humanize };
