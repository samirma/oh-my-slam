// A modal confirmation that states the consequence first (http_server.md "Confirmation",
// "Interruption"). Resolves true when the user confirms; Escape or the other button resolve false.
// The focus starts on the safe choice and returns to where it was.
import { el } from './dom.js';

let pending = null;

export function confirmAction({ title, body, yes, no = 'Cancel', danger = false }) {
  const dlg = document.getElementById('confirm');
  if (pending) return pending;  // one question at a time
  document.getElementById('confirm-title').textContent = title;
  document.getElementById('confirm-body').replaceChildren(
    ...[].concat(body).map((t) => (typeof t === 'string' ? el('p', {}, t) : t)));
  const yesBtn = document.getElementById('confirm-yes');
  yesBtn.textContent = yes;
  yesBtn.className = danger ? 'danger' : 'primary';
  document.getElementById('confirm-no').textContent = no;
  const opener = document.activeElement;
  pending = new Promise((resolve) => {
    dlg.addEventListener('close', () => {
      const ok = dlg.returnValue === 'ok';
      dlg.returnValue = '';
      pending = null;
      if (opener && opener.isConnected) opener.focus?.();
      resolve(ok);
    }, { once: true });
    dlg.showModal();
    document.getElementById('confirm-no').focus();
  });
  return pending;
}
