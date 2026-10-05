// Cancelling and re-submitting a job (http_server.md "Jobs", "Confirmation"): a cancel states its
// consequence first; a re-submission runs the same options again and asks for an uploaded input
// again, since uploads are deleted when their job ends.
import { el } from './dom.js';
import { postJson, upload, enc } from './api.js';
import { store, PATH_IN, blockedReason } from './store.js';
import { confirmAction, askFiles } from './dialog.js';

export const TERMINAL = ['succeeded', 'failed', 'cancelled'];

export async function cancelJob(job) {
  const body = job.state === 'queued'
    ? ['The job has not started: it is dropped, and its uploaded inputs are deleted.']
    : [`Cancelling interrupts ${job.label} as Ctrl-C would: it stops at once and produces no result.`,
      ...(job.writes ? ['The map is left exactly as it was before this job: nothing of this update is kept.'] : []),
      'Its uploaded inputs are deleted; re-submitting asks for them again.'];
  const ok = await confirmAction({ title: `Cancel job ${job.id}?`, body, yes: 'Cancel the job', no: 'Keep it running' });
  if (!ok) return null;
  return postJson(`/api/jobs/${enc(job.id)}/cancel`);
}

// The path parameters of a job that named uploads (gone once the job ended).
export function discardedUploads(job) {
  const op = store.ops.get(job.operation);
  if (!op || !TERMINAL.includes(job.state)) return [];
  return op.params.filter((p) => PATH_IN.includes(p.kind)
    && [].concat(job.params[p.name] ?? []).some((v) => String(v).startsWith('uploads/')));
}

// Re-submit with the same options; resolves to the new job, or null when the user cancelled.
export async function resubmitJob(job) {
  const gone = discardedUploads(job);
  const override = {};
  if (gone.length) {
    const files = await askFiles({
      title: `Re-submit job ${job.id}`,
      body: 'The uploaded inputs of this job were deleted when it ended. Choose the files again; every other option stays as it was.',
      params: gone.map((p) => ({ ...p, was: [].concat(job.params[p.name]).map((v) => String(v).split('/').pop()).join(', ') })),
    });
    if (!files) return null;
    for (const p of gone) {
      const chosen = files[p.name] || [];
      if (!chosen.length) throw new Error(`choose a file for ${p.name} (${p.flag})`);
      const ups = [];
      for (const f of chosen) ups.push((await upload(f)).path);
      override[p.name] = p.multiple ? ups : ups[0];
    }
  }
  return postJson(`/api/jobs/${enc(job.id)}/resubmit`, override);
}

export function actionButtons(job, { onDone = () => {}, onError = () => {} } = {}) {
  const out = [];
  if (!TERMINAL.includes(job.state)) {
    const b = el('button', { type: 'button', class: 'danger', 'data-action': 'cancel' }, 'Cancel…');
    b.addEventListener('click', async () => {
      try { const j = await cancelJob(job); if (j) onDone(j); } catch (err) { onError(err); }
    });
    out.push(b);
  } else {
    const op = store.ops.get(job.operation);
    const why = op ? blockedReason(op) : null;
    const b = el('button', { type: 'button', class: 'secondary', 'data-action': 'resubmit', disabled: !!why || null,
      title: why || null }, 'Re-submit');
    if (why) { out.push(b, el('span', { class: 'blocked' }, why)); return out; }
    b.addEventListener('click', async () => {
      b.disabled = true;
      b.textContent = 'Re-submitting…';
      try {
        const j = await resubmitJob(job);
        if (j) { location.hash = `#/jobs/${j.id}`; onDone(j); }
      } catch (err) { onError(err); } finally { b.disabled = false; b.textContent = 'Re-submit'; }
    });
    out.push(b);
  }
  return out;
}
