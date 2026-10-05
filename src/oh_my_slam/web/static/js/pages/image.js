// Image (http_server.md "Structure"): one image in, any command mode that takes a single image (the
// operations with one `image` parameter, read from the API description). A drop zone with a
// preview, the generated option form, then the job's live progress and, on success, its result
// with the embedded viewer (asked with ?viewer=true where the operation offers it).
import { el, clear, notice } from '../dom.js';
import { upload, discardUpload } from '../api.js';
import { store, blockedReason } from '../store.js';
import { OpForm } from '../form.js';
import { renderJob } from '../jobview.js';
import { consequence } from '../opcard.js';

// The image input as a drop zone with a preview; the form's field for the image parameter.
class DropImage {
  constructor(p, onChange) {
    this.p = p;
    this.onChange = onChange;
    this.item = null;
    const accept = (p.accepts || []).join(',');
    this.input = el('input', { type: 'file', id: 'image-file', accept: accept || null, hidden: true, 'aria-label': 'Image file' });
    this.input.addEventListener('change', () => { if (this.input.files[0]) this.take(this.input.files[0]); this.input.value = ''; });
    this.button = el('button', { type: 'button', class: 'primary', 'aria-describedby': 'image-help image-err' }, 'Choose an image…');
    this.button.addEventListener('click', () => this.input.click());
    this.preview = el('img', { class: 'preview', alt: '', hidden: true });
    this.status = el('p', { class: 'file-state', 'aria-live': 'polite' });
    this.err = el('div', { class: 'field-error', id: 'image-err', 'aria-live': 'polite' });
    this.zone = el('div', { class: 'dropzone big', 'data-testid': 'dropzone' },
      el('p', {}, 'Drop an image here, or'), this.button, this.input,
      el('p', { class: 'help', id: 'image-help' }, el('code', {}, p.flag), ` ${p.help}. Accepted: ${(p.accepts || []).join(' ')}`),
      this.preview, this.status);
    for (const t of ['dragenter', 'dragover']) this.zone.addEventListener(t, (e) => { e.preventDefault(); this.zone.classList.add('over'); });
    for (const t of ['dragleave', 'drop']) this.zone.addEventListener(t, () => this.zone.classList.remove('over'));
    this.zone.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) this.take(f); });
    this.el = el('div', { class: 'field', 'data-param': p.name }, this.zone, this.err);
  }

  take(file) {
    this.discard();
    const it = { name: file.name, state: 'uploading', progress: 0 };
    this.item = it;
    this.preview.src = URL.createObjectURL(file);
    this.preview.alt = `Preview of ${file.name}`;
    this.preview.hidden = false;
    this.status.textContent = `${file.name}: uploading…`;
    this.onChange();
    upload(file, (x) => { if (this.item === it) this.status.textContent = `${file.name}: uploading ${Math.round(x * 100)} %`; })
      .then((u) => {
        it.upload = u;
        if (this.item !== it) { discardUpload(u.id); return; }
        it.state = 'ready';
        this.status.textContent = `${file.name}: uploaded`;
      })
      .catch((err) => { it.state = 'failed'; this.status.textContent = `${file.name}: the upload failed: ${err.message}`; })
      .finally(() => this.onChange());
  }

  discard() { if (this.item?.upload && !this.item.consumed) discardUpload(this.item.upload.id); this.item = null; }
  value() { return this.item?.state === 'ready' ? this.item.upload.path : undefined; }
  pending() { return this.item?.state === 'uploading'; }
  names() { return this.item ? [this.item.name] : []; }
  consumed() { if (this.item) this.item.consumed = true; }
  destroy() { this.discard(); }
  setError(msgs) {
    const on = !!(msgs && msgs.length);
    this.err.textContent = on ? msgs.join(' ') : '';
    this.el.classList.toggle('invalid', on);
    if (on) this.button.setAttribute('aria-invalid', 'true'); else this.button.removeAttribute('aria-invalid');
  }
}

export function imagePage(main, { job, op: wanted }) {
  const ops = [...store.ops.values()].filter((o) => o.singleImage);
  let op = ops.find((o) => o.id === wanted) || ops[0];
  main.append(el('h1', {}, 'Image'));
  if (!op) { main.append(notice('info', 'No command takes a single image.')); return null; }
  const choices = el('fieldset', { class: 'modes' }, el('legend', {}, 'Command'),
    ...ops.map((o) => {
      const id = `mode-${o.id}`;
      const r = el('input', { type: 'radio', name: 'mode', id, value: o.id });
      r.checked = o === op;
      r.addEventListener('change', () => { if (r.checked) choose(o); });
      return el('div', { class: 'mode' }, r, el('label', { for: id }, el('strong', {}, o.label), el('span', { class: 'muted' }, ` ${o.description}`)));
    }));
  const formBox = el('div', { class: 'form-box' });
  const blocked = el('div', { 'aria-live': 'polite' });
  const what = el('p', { class: 'consequence' });
  const submit = el('button', { type: 'submit', class: 'primary' }, 'Run');
  const submitErr = el('div', { 'aria-live': 'assertive' });
  let form = null;
  const drop = new DropImage(op.singleImage, () => form?.changed());
  const formEl = el('form', { class: 'image-form', novalidate: true, 'aria-label': 'Image job' },
    choices, el('h2', {}, 'Image'), drop.el, el('h2', {}, 'Options'), formBox, blocked, what,
    el('div', { class: 'actions' }, submit), submitErr);
  const jobBox = el('div', { class: 'job-box' });
  main.append(formEl, jobBox);

  function refreshBlocked() {
    const why = blockedReason(op);
    submit.disabled = !!why;
    blocked.replaceChildren(...(why ? [notice('warn', why)] : []));
  }

  function choose(o) {
    op = o;
    drop.p = o.singleImage;
    form?.root.remove();
    form = new OpForm(formBox, o, { omit: [o.singleImage.name], external: { [o.singleImage.name]: drop }, viewer: o.viewerQuery });
    what.textContent = consequence(o) + (o.viewerQuery ? ' It also prepares the viewer of this image.' : '');
    submit.textContent = `Run ${o.label}`;
    const q = new URLSearchParams({ op: o.id });
    history.replaceState(null, '', `#/image${job ? `/${job}` : ''}?${q}`);
    refreshBlocked();
    if (drop.value()) form.changed();
  }
  choose(op);

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitErr.replaceChildren();
    if (!drop.item) { drop.setError(['Choose an image first.']); drop.button.focus(); return; }
    submit.disabled = true;
    submit.textContent = 'Submitting…';
    try {
      const j = await form.submit();
      store.jobs.set(j.id, j);
      location.hash = `#/image/${j.id}?op=${op.id}`;
    } catch (err) {
      submitErr.replaceChildren(notice('error', err.message));
      form.focusFirstError();
    } finally {
      submit.textContent = `Run ${op.label}`;
      refreshBlocked();
    }
  });

  const offHealth = store.on((w) => { if (w === 'health') refreshBlocked(); });
  let stopJob = null;
  if (job) {
    jobBox.append(el('h2', {}, 'Job'));
    stopJob = renderJob(jobBox, job, { heading: 'h3' });
  }
  return () => { offHealth(); stopJob?.(); form?.destroy(); drop.destroy(); };
}
