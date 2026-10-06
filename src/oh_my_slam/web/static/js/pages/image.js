// Image (http_server.md "Structure"): one image in, any operation that takes a single image (the
// operations with one `image` parameter and no map, read from the API description). A drop zone
// with a preview, the generated option form, then the running request and, on success, the result
// with its download. The operation is in the URL (#/image?op=<id>).
import { el, notice } from '../dom.js';
import { store } from '../store.js';
import { OpForm } from '../form.js';
import { runPanel } from '../runpanel.js';
import { setQuery } from '../url.js';

function stem(name) { return String(name || '').split('/').pop().replace(/\.[^.]+$/, ''); }

export function imagePage(main, { op: wanted }) {
  const ops = [...store.ops.values()].filter((o) => o.singleImage);
  let op = ops.find((o) => o.id === wanted) || ops[0];
  main.append(el('h1', {}, 'Image'),
    el('p', { class: 'lead' }, 'One image in: choose what to run on it. The result is the command\'s own output, shown here and downloadable.'));
  if (!op) { main.append(notice('info', 'No operation of this service takes a single image.')); return null; }
  const choices = el('fieldset', { class: 'modes', 'data-testid': 'operations' }, el('legend', {}, 'Operation'),
    ...ops.map((o) => {
      const id = `op-${o.id}`;
      const r = el('input', { type: 'radio', name: 'op', id, value: o.id, 'aria-describedby': `${id}-desc` });
      r.checked = o === op;
      r.addEventListener('change', () => { if (r.checked) choose(o); });
      return el('div', { class: 'mode' }, r, el('label', { for: id }, el('code', {}, o.label)),
        el('span', { class: 'muted', id: `${id}-desc` }, ` ${o.description} · ${o.x.inference_text}`));
    }));
  const imageBox = el('section', { class: 'step', 'aria-labelledby': 'image-h' }, el('h2', { id: 'image-h' }, 'Image'));
  const optionsBox = el('section', { class: 'step', 'aria-labelledby': 'options-h' }, el('h2', { id: 'options-h' }, 'Options'));
  const runBox = el('div', {});
  main.append(choices, imageBox, optionsBox, runBox);
  let form = null;
  let panel = null;

  function choose(o) {
    const previous = form?.field(op.singleImage.name);
    const kept = previous ? previous.release() : [];
    form?.destroy();
    panel?.dispose();
    op = o;
    imageBox.querySelector('.field')?.remove();
    optionsBox.replaceChildren(optionsBox.firstChild);
    form = new OpForm(optionsBox, o, { files: { [o.singleImage.name]: { big: true, preview: true } } });
    const image = form.field(o.singleImage.name);
    imageBox.append(image.el);
    image.adopt(kept);
    const otherFields = form.fields.filter((f) => f !== image);
    if (!otherFields.length) optionsBox.insertBefore(el('p', { class: 'muted' }, 'No other option.'), form.root);
    panel = runPanel({
      op: o, form, label: `Run ${o.label}`,
      downloadName: (fmt) => `${o.id}-${stem(image.names()[0]) || 'result'}${fmt ? `.${fmt}` : ''}`,
    });
    runBox.replaceChildren(panel.el);
    setQuery({ op: o.id });
    form.changed();
  }
  choose(op);
  return () => { panel?.dispose(); form?.destroy(); };
}
