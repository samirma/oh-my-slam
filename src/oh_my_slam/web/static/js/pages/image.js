// Image (http_server.md "Structure"): one image in, any operation that takes a single image (the
// operations with one `image` parameter and no map, read from the API description). A drop zone
// with a preview, the generated option form, then the running request and, on success, the result
// with its download. The operation and the form's values are in the URL
// (#/image?op=<id>&<parameter>=<value>…; not the image, which is a file of this computer).
import { el, notice, sentence } from '../dom.js';
import { store, inferenceSentence } from '../store.js';
import { OpForm, resultName } from '../form.js';
import { runPanel } from '../runpanel.js';
import { setQuery, formQuery, queryInitial } from '../url.js';

// A file name without its folders and its suffix (the result's download name starts with it).
export function stem(name) { return String(name || '').split('/').pop().replace(/\.[^.]+$/, ''); }

export function imagePage(main, { op: wanted, query = new URLSearchParams() }) {
  const ops = [...store.ops.values()].filter((o) => o.singleImage);
  let op = ops.find((o) => o.id === wanted) || ops[0];
  main.append(el('h1', {}, 'Image'),
    el('p', { class: 'lead' }, 'One image in: choose what to run on it. The result is the command\'s own output, shown here and downloadable.'));
  if (!op) { main.append(notice('info', 'No operation of this service takes a single image.')); return null; }
  const about = (o) => ` ${sentence(o.description)} ${inferenceSentence(o)}`;
  // one operation: said in a sentence; several: a choice between them
  const choices = ops.length === 1
    ? el('div', { 'data-testid': 'operations', 'data-op': op.id }, el('p', { class: 'op-lead' }, el('code', {}, op.label),
      `: ${sentence(op.description)} ${inferenceSentence(op)}`))
    : el('fieldset', { class: 'modes', 'data-testid': 'operations' }, el('legend', {}, 'Operation'),
      ...ops.map((o) => {
        const id = `op-${o.id}`;
        const r = el('input', { type: 'radio', name: 'op', id, value: o.id, 'aria-describedby': `${id}-desc` });
        r.checked = o === op;
        r.addEventListener('change', () => { if (r.checked) choose(o); });
        return el('div', { class: 'mode' }, r, el('label', { for: id }, el('code', {}, o.label)),
          el('span', { class: 'muted', id: `${id}-desc` }, about(o)));
      }));
  const imageBox = el('section', { class: 'step', 'aria-labelledby': 'image-h' }, el('h2', { id: 'image-h' }, 'Image'));
  const optionsBox = el('section', { class: 'step', 'aria-labelledby': 'options-h' }, el('h2', { id: 'options-h' }, 'Options'));
  const runBox = el('div', {});
  main.append(choices, imageBox, optionsBox, runBox);
  let form = null;
  let panel = null;
  let offQuery = null;
  let keys = [...query.keys()].filter((k) => k !== 'op');  // the query keys to replace: the form's, or unknown

  function choose(o, initial = {}) {
    const previous = form?.field(op.singleImage.name);
    const kept = previous ? previous.release() : [];
    form?.destroy();
    panel?.dispose();
    offQuery?.();
    op = o;
    imageBox.querySelector('.field')?.remove();
    optionsBox.replaceChildren(optionsBox.firstChild);
    form = new OpForm(optionsBox, o, { files: { [o.singleImage.name]: { big: true, preview: true } }, initial });
    const f = form;
    const image = form.field(o.singleImage.name);
    imageBox.append(image.el);
    image.adopt(kept);
    const otherFields = form.fields.filter((f) => f !== image);
    if (!otherFields.length) optionsBox.insertBefore(el('p', { class: 'muted' }, 'No other option.'), form.root);
    panel = runPanel({
      op: o, form, label: `Run ${o.label}`,
      downloadName: (fmt) => resultName([o.id, stem(image.names()[0]) || 'result', ...f.variant()], fmt),
    });
    runBox.replaceChildren(panel.el);
    const keep = () => {
      const values = f.queryValues();
      setQuery(formQuery(o.id, values, keys.filter((k) => !(k in values))));
      keys = Object.keys(values);
    };
    offQuery = form.onValidate(keep);
    keep();
    form.changed();
  }
  choose(op, op.id === wanted || !wanted ? queryInitial(query, op.params.map((p) => p.name)) : {});
  return () => { offQuery?.(); panel?.dispose(); form?.destroy(); };
}
