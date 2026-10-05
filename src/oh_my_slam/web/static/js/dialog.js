// A modal confirmation that states the consequence first (http_server.md "Confirmation").
// Resolves true when the user confirms; Escape or the other button resolves false.
import { el } from './dom.js';

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

// Ask for the files of discarded uploads again: one file input per parameter; resolves to
// {name: [File, …]} or null when cancelled.
export function askFiles({ title, body, params }) {
  const dlg = document.getElementById('confirm');
  document.getElementById('confirm-title').textContent = title;
  const inputs = new Map();
  const rows = params.map((p) => {
    const id = `again-${p.name}`;
    const input = el('input', { type: 'file', id, multiple: p.multiple || null, accept: (p.accepts || []).join(',') || null });
    inputs.set(p.name, input);
    return el('div', { class: 'field' }, el('label', { for: id }, `${p.name} (${p.flag}): was ${p.was}`), input);
  });
  document.getElementById('confirm-body').replaceChildren(el('p', {}, body), ...rows);
  document.getElementById('confirm-yes').textContent = 'Upload and re-submit';
  document.getElementById('confirm-no').textContent = 'Cancel';
  return new Promise((resolve) => {
    dlg.addEventListener('close', () => {
      const ok = dlg.returnValue === 'ok';
      dlg.returnValue = '';
      if (!ok) { resolve(null); return; }
      const out = {};
      for (const [name, input] of inputs) out[name] = [...input.files];
      resolve(out);
    }, { once: true });
    dlg.showModal();
  });
}
