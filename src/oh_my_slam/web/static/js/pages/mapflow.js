// Creating or updating a map (http_server.md "Maps"): a guided flow over the mapping operation's
// options — the operation whose output is the map folder it names, read from the API description.
// Step 1 holds its input parameters, with their order visible and editable where it matters
// (`ordered`); step 2 the map; step 3 every other option; step 4 states what starting it does
// (the command line, the inference server, how long it takes, what a cancel leaves) before it runs.
import { el, notice, humanize } from '../dom.js';
import { store, blockedReason, PATH_IN } from '../store.js';
import { OpForm } from '../form.js';
import { confirmAction } from '../dialog.js';

export function mapFlowPage(main, { name }) {
  const op = [...store.ops.values()].find((o) => o.writesMap);
  main.append(el('h1', {}, name ? `Update map ${name}` : 'New map'));
  if (!op) { main.append(notice('info', 'No command writes a map.')); return null; }
  const mapP = op.writesMap;
  const inputs = op.params.filter((p) => PATH_IN.includes(p.kind) && p !== mapP).map((p) => p.name);
  const rest = op.params.filter((p) => !inputs.includes(p.name) && p !== mapP).map((p) => p.name);
  const step = (n, title, ...body) => el('section', { class: 'step', 'aria-labelledby': `step-${n}` },
    el('h2', { id: `step-${n}` }, el('span', { class: 'step-n' }, `${n}`), ` ${title}`), ...body);
  const s1 = el('div', {}), s2 = el('div', {}), s3 = el('div', {});
  const command = el('code', { 'data-testid': 'flow-command' });
  const blocked = el('div', { 'aria-live': 'polite' });
  const start = el('button', { type: 'submit', class: 'primary' }, name ? `Update ${name}…` : 'Create the map…');
  const out = el('div', { 'aria-live': 'assertive' });
  const formEl = el('form', { class: 'flow', novalidate: true, 'aria-label': main.querySelector('h1').textContent },
    step(1, inputs.map(humanize).join(', ') || 'Inputs', s1),
    step(2, 'Map', s2),
    step(3, 'Options', s3),
    step(4, 'Review and start',
      el('p', {}, 'The command that will run: ', command),
      el('p', { class: 'consequence' }, `${op.label} ${op.x.inference_text}. It runs as a job and may take minutes; you can leave this page meanwhile. `
        + 'The map changes only when the job succeeds: a failed or cancelled update leaves it exactly as it was.'),
      blocked, el('div', { class: 'actions' }, start), out));
  main.append(formEl);

  // one OpForm, its fields placed in the steps
  const host = el('div', {});
  const form = new OpForm(host, op, name ? { fixed: { [mapP.name]: name } } : {});
  for (const f of form.fields) (inputs.includes(f.p.name) ? s1 : f.p === mapP ? s2 : s3).append(f.el);
  if (name) s2.append(el('p', {}, `The new inputs extend map `, el('strong', {}, name), '.'));
  s3.append(form.general);
  if (!rest.length) s3.append(el('p', { class: 'muted' }, 'No other option.'));
  form.on('validated', (r) => { command.textContent = (r.command || []).join(' ') || '(complete steps 1 and 2)'; });
  command.textContent = '(complete steps 1 and 2)';

  const refresh = () => {
    const why = blockedReason(op);
    start.disabled = !!why;
    blocked.replaceChildren(...(why ? [notice('warn', why)] : []));
  };
  refresh();
  const off = store.on((w) => { if (w === 'health') refresh(); });

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    out.replaceChildren();
    const r = await form.validate();
    if (!r || !r.valid) { out.replaceChildren(notice('error', 'Fix the fields marked above first.')); form.focusFirstError(); return; }
    const values = form.values();
    const target = values[mapP.name];
    const ok = await confirmAction({
      title: name ? `Update map ${target}?` : `Create map ${target}?`,
      body: [`This runs ${r.command.join(' ')} as a job.`,
        `${op.label} ${op.x.inference_text}; it may take minutes, and jobs that use the inference server run one at a time.`,
        'The map changes only when the job succeeds. Cancelling it, or a failure, leaves the map exactly as it was. The uploaded inputs are deleted when the job ends.'],
      yes: name ? 'Start the update' : 'Start mapping', no: 'Not now',
    });
    if (!ok) return;
    start.disabled = true;
    start.textContent = 'Submitting…';
    try {
      const j = await form.submit();
      store.jobs.set(j.id, j);
      location.hash = `#/jobs/${j.id}`;
    } catch (err) {
      out.replaceChildren(notice('error', err.message));
      form.focusFirstError();
    } finally { start.textContent = name ? `Update ${name}…` : 'Create the map…'; refresh(); }
  });
  return () => { off(); form.destroy(); };
}
