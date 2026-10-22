// Running an operation from a page: its Run button (disabled, with the reason, when the operation
// needs the inference server and that server is down), what running it does, the confirmation of a
// long operation, then the request in progress and its result (RequestView). Every click shows a
// change at once (http_server.md "Responsive feedback").
import { el, notice } from './dom.js';
import { store, blockedReason, inferenceSentence } from './store.js';
import { RequestView } from './request.js';

// What running `op` does, stated before it runs. `inference`: whether this request uses the
// inference server, as its validation answered (true or false; null: not known yet, so a mode that
// always needs it does and a conditional one is not said).
export function consequence(op, inference = null) {
  const parts = [`Runs ${op.label} within this page's request and shows its result here.`, inferenceSentence(op)];
  const uses = inference ?? (op.x.inference === 'required' ? true : null);
  if (uses === true) parts.push('This request uses it: requests that use the inference server run one at a time, in arrival order, so it may wait for its turn.');
  else if (uses === false && op.x.inference === 'conditional') parts.push('This request does not use it, so it starts at once.');
  if (op.writesMap) parts.push('It is the only operation that changes a map, and it can take many minutes.');
  return parts.join(' ');
}

// `confirm(validation)`: asked before a long operation runs (resolves true to go on);
// `downloadName(format)`; `onEnd(ok)`: after the request ended (the form's uploads are renewed
// unless it returns 'keep-cleared').
export function runPanel({ op, form, label, confirm = null, downloadName, onEnd = null }) {
  const submit = el('button', { type: 'button', class: 'primary', 'data-action': 'run' }, label);
  const blocked = el('div', { class: 'blocked-box' });
  const problems = el('div', { class: 'run-problems' });
  const what = el('p', { class: 'consequence' }, consequence(op));
  const offValidate = form.onValidate((r) => { what.textContent = consequence(op, r.valid ? r.inference : null); });
  const requestBox = el('div', { class: 'request-box' });
  const rv = new RequestView(requestBox, { op, form, downloadName });
  let busy = false;
  const root = el('div', { class: 'run-panel', 'data-op': op.id }, what, blocked,
    el('div', { class: 'actions' }, submit), problems, requestBox);

  function refresh() {
    const why = blockedReason(op);
    submit.disabled = busy || !!why;
    if (why) submit.setAttribute('aria-describedby', `${op.id}-blocked`); else submit.removeAttribute('aria-describedby');
    blocked.replaceChildren(...(why ? [el('div', { id: `${op.id}-blocked` }, notice('warn', el('strong', {}, 'Not available now. '), why))] : []));
  }

  // `focus`: move the focus to the result when it ends, wherever it is (a run the page started)
  async function go({ focus = false } = {}) {
    if (busy) return;
    busy = true;
    submit.disabled = true;
    submit.textContent = 'Checking…';
    problems.replaceChildren();
    try {
      const v = await form.check();
      if (!v || !v.valid) {
        problems.replaceChildren(notice('error', 'This request would be refused: ',
          (v?.problems || []).map((p) => p.message).join(' ') || 'fix the fields marked above.',
          ' Nothing ran.'));
        form.focusFirstError();
        return;
      }
      if (confirm && !(await confirm(v))) return;
      submit.textContent = 'Running…';
      const ok = await rv.run(form.values(), v.command, { focus });
      if (!onEnd || onEnd(ok) !== 'keep-cleared') form.renew();
    } finally {
      busy = false;
      submit.textContent = label;
      refresh();
    }
  }

  submit.addEventListener('click', () => go());
  refresh();
  const off = store.on((w) => { if (w === 'health') refresh(); });
  return { el: root, submit, refresh, request: rv, run: go, dispose: () => { off(); offValidate(); rv.dispose(); } };
}
