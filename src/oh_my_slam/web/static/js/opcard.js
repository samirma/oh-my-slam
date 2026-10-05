// One operation as a collapsible card: its generated form (with `fixed` values set by the page),
// what running it does, and its submission as a job (a link to the job's page). Disabled with the
// reason when it needs the inference server and that server is down.
import { el, notice } from './dom.js';
import { store, blockedReason } from './store.js';
import { OpForm } from './form.js';

export function consequence(op) {
  const parts = [`Runs ${op.label} as a job: ${op.x.inference_text}.`];
  if (op.x.inference === 'required') parts.push('Jobs that use the inference server run one at a time, in submission order.');
  parts.push('You can leave this page; the job keeps running and its result stays under Jobs.');
  return parts.join(' ');
}

export function opCard(op, fixed) {
  const box = el('details', { class: 'export', 'data-op': op.id });
  const body = el('div', {});
  const submit = el('button', { type: 'submit', class: 'primary' }, `Run ${op.label}`);
  const blocked = el('div', { 'aria-live': 'polite' });
  const out = el('div', { 'aria-live': 'polite' });
  const what = Object.values(fixed).join(', ');
  const formEl = el('form', { novalidate: true, 'aria-label': what ? `${op.label} of ${what}` : op.label }, body, blocked,
    el('p', { class: 'consequence' }, consequence(op)), el('div', { class: 'actions' }, submit), out);
  box.append(el('summary', {}, el('strong', {}, op.label), ` ${op.description}`), formEl);
  let form = null;
  const refresh = () => {
    const why = blockedReason(op);
    submit.disabled = !!why;
    blocked.replaceChildren(...(why ? [notice('warn', why)] : []));
  };
  box.addEventListener('toggle', () => {
    if (box.open && !form) { form = new OpForm(body, op, { fixed }); form.validate(); }
  });
  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    submit.disabled = true;
    submit.textContent = 'Submitting…';
    out.replaceChildren();
    try {
      const j = await form.submit();
      store.jobs.set(j.id, j);
      out.replaceChildren(notice('info', 'Submitted as ', el('a', { href: `#/jobs/${j.id}` }, `job ${j.id}`), '; its result and files are on its page.'));
    } catch (err) {
      out.replaceChildren(notice('error', err.message));
    } finally { submit.textContent = `Run ${op.label}`; refresh(); }
  });
  refresh();
  const off = store.on((w) => { if (w === 'health') refresh(); });
  return { el: box, stop: () => { off(); form?.destroy(); } };
}
