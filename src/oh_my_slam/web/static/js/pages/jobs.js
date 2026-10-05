// Jobs (http_server.md "Structure"): every job with its kind, inputs, state, times and per-stage
// timings; running jobs show their progress, failed ones their message; a job can be cancelled
// (its consequence stated first) or re-submitted with the same options. Kept current by the
// server-sent events of /api/jobs/events.
import { el, clear, notice, fmtTime, stateBadge } from '../dom.js';
import { store } from '../store.js';
import { actionButtons, TERMINAL } from '../jobactions.js';
import { renderJob, inputsText, progressBar, stagesList } from '../jobview.js';
import { opCard } from '../opcard.js';

function row(job, onError) {
  const msg = job.state === 'failed' || (job.state === 'cancelled' && job.error)
    ? el('span', { class: job.state === 'failed' ? 'error-text' : 'muted' }, job.error?.message || '')
    : job.viewer_error ? el('span', { class: 'error-text' }, `viewer: ${job.viewer_error.message}`) : '';
  return el('tr', { 'data-job': job.id, 'data-state': job.state },
    el('th', { scope: 'row' }, el('a', { href: `#/jobs/${job.id}` }, job.id)),
    el('td', {}, job.label),
    el('td', { class: 'inputs' }, inputsText(job)),
    el('td', {}, stateBadge(job.state), TERMINAL.includes(job.state) ? '' : progressBar(job)),
    el('td', { class: 'times' }, el('div', {}, `submitted ${fmtTime(job.created_at)}`),
      job.started_at ? el('div', {}, `started ${fmtTime(job.started_at)}`) : '',
      job.ended_at ? el('div', {}, `ended ${fmtTime(job.ended_at)}`) : ''),
    el('td', {}, stagesList(job)),
    el('td', { class: 'message' }, msg),
    el('td', { class: 'row-actions' }, ...actionButtons(job, { onError })));
}

export function jobsPage(main) {
  const errors = el('div', { 'aria-live': 'assertive' });
  const tbody = el('tbody', {});
  const empty = el('p', { class: 'muted' }, 'No jobs yet.');
  main.append(el('h1', {}, 'Jobs'), errors,
    el('div', { class: 'table-wrap' }, el('table', { class: 'data jobs', 'data-testid': 'jobs' },
      el('caption', {}, 'Every job, newest first'),
      el('thead', {}, el('tr', {}, ...['job', 'kind', 'inputs', 'state', 'times', 'per-stage timings', 'message', 'actions']
        .map((h) => el('th', { scope: 'col' }, h)))), tbody)), empty);
  const onError = (err) => errors.replaceChildren(notice('error', err.message));
  const rows = new Map();
  const draw = (job) => {
    const old = rows.get(job.id);
    if (old && old.dataset.state === job.state) {
      // the same state: refresh the live cells only, so its buttons (and their focus) stay
      old.children[3].replaceChildren(stateBadge(job.state), TERMINAL.includes(job.state) ? '' : progressBar(job));
      old.children[5].replaceChildren(stagesList(job));
      return;
    }
    const r = row(job, onError);
    rows.set(job.id, r);
    if (old) old.replaceWith(r); else tbody.prepend(r);
    empty.hidden = true;
  };
  const all = [...store.jobs.values()].sort((a, b) => a.created_at - b.created_at);
  for (const j of all) draw(j);
  empty.hidden = all.length > 0;
  // a command mode that takes neither a single image (Image) nor a map (Maps) is run from here, so
  // that every operation of the API description has its form
  const others = [...store.ops.values()].filter((o) => !o.singleImage && !o.mapParam);
  const stops = [];
  if (others.length) {
    const box = el('section', { 'aria-labelledby': 'other-h' }, el('h2', { id: 'other-h' }, 'Other commands'));
    for (const op of others) { const c = opCard(op, {}); box.append(c.el); stops.push(c.stop); }
    main.insertBefore(box, main.children[1]);
  }
  const off = store.on((what, j) => { if (what === 'job') draw(j); });
  return () => { off(); stops.forEach((s) => s()); };
}

export function jobPage(main, { id }) {
  main.append(el('p', {}, el('a', { href: '#/jobs' }, '← All jobs')));
  const stop = renderJob(main, id, { heading: 'h1' });
  return stop;
}

export { clear };
