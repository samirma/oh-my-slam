// A modal confirmation that states the consequence first (http_server.md "Confirmation").
// Resolves true when the user confirms; Escape or the other button resolves false.
import { el } from './dom.js';
import { FilesField } from './form.js';

export function confirmAction({ title, body, yes, no = 'Keep it as it is' }) {
  const dlg = document.getElementById('confirm');
  document.getElementById('confirm-title').textContent = title;
  const b = document.getElementById('confirm-body');
  b.replaceChildren(...[].concat(body).map((t) => (typeof t === 'string' ? el('p', {}, t) : t)));
  document.getElementById('confirm-yes').textContent = yes;
  document.getElementById('confirm-no').textContent = no;
  const opener = document.activeElement;
  return new Promise((resolve) => {
    dlg.addEventListener('close', () => {
      resolve(dlg.returnValue === 'ok');
      dlg.returnValue = '';
      opener?.focus?.();
    }, { once: true });
    dlg.showModal();
    document.getElementById('confirm-no').focus();
  });
}

// Ask for the files of discarded uploads again: the form's own path field per parameter, the
// previous files pre-listed in their order (each filled in place by the file of the same name).
// Resolves to {name: value} once every upload has finished, or null when cancelled; throws the
// names still missing.
export function askFiles({ title, body, params }) {
  const dlg = document.getElementById('confirm');
  document.getElementById('confirm-title').textContent = title;
  const host = { changed() {}, emit() {} };  // no form around these fields
  const fields = params.map((p) => {
    const f = new FilesField(host, p);
    f.expect(p.previous);
    return f;
  });
  document.getElementById('confirm-body').replaceChildren(el('p', {}, body), ...fields.map((f) => f.el));
  document.getElementById('confirm-yes').textContent = 'Re-submit';
  document.getElementById('confirm-no').textContent = 'Cancel';
  const opener = document.activeElement;
  return new Promise((resolve, reject) => {
    dlg.addEventListener('close', async () => {
      const ok = dlg.returnValue === 'ok';
      dlg.returnValue = '';
      opener?.focus?.();
      if (!ok) { for (const f of fields) f.destroy(); resolve(null); return; }
      await Promise.all(fields.map((f) => f.settled()));
      const missing = fields.flatMap((f) => f.missing());
      if (missing.length) { for (const f of fields) f.destroy(); reject(new Error(`choose ${missing.join(', ')} again to re-submit`)); return; }
      const out = {};
      for (const f of fields) { f.consumed(); out[f.p.name] = f.value(); }
      resolve(out);
    }, { once: true });
    dlg.showModal();
  });
}
