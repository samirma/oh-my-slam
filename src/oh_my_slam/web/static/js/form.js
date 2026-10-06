// Forms generated from the API description (http_server.md "Web application", "Forms"): one field
// per parameter of an operation, from its registry entry (`x-oms`: kind, flag, help, default,
// choices, bounds, multiple/ordered, accepts, applies, the -p attributes). Nothing here names a
// command or an option: a new option is a new field, and a kind this file does not know still gets
// a text field. Fields that do not apply to the current choices (`applies`) are hidden and not
// sent; every change is checked by the command's own rules (POST /api/ops/<op>/validate) and each
// message is shown next to the field it names (`by_parameter`), before submission.
import { el, clear, nextId, humanize, fmtBytes } from './dom.js';
import { getJson, postJson, upload, discardUpload, enc } from './api.js';

const VALIDATE_DEBOUNCE_MS = 250;

function suffix(name) { const m = /\.[^./\\]+$/.exec(name); return m ? m[0].toLowerCase() : ''; }

export function defaultText(p) {
  if (p.default === null || p.default === undefined || p.kind === 'attrs') return null;
  if (p.kind === 'flag') return p.default ? 'on' : 'off';
  return String(p.default);
}

// Whether one applicability condition (`applies` / an output's `when`) holds for `current(name)`
// (a parameter's value, else its default) and `names(name)` (the files a path parameter holds).
export function holds(w, current, names) {
  if (w.in) return w.in.includes(current(w.option));
  if (w.is === 'given') { const v = current(w.option); return v !== undefined && v !== null && v !== '' && v !== false; }
  if (w.is === 'video') {
    const n = names(w.option);  // the suffixes of a video come with the condition
    return n.length === 1 && (w.suffixes || []).includes(suffix(n[0]));
  }
  return true;
}

// ------------------------------------------------------------------------------------------- fields

class Field {
  constructor(form, p) {
    this.form = form;
    this.p = p;
    this.id = nextId(`f-${p.name}`);
    this.errId = `${this.id}-err`;
    this.helpId = `${this.id}-help`;
    this.active = true;
    this.el = el('div', { class: 'field', 'data-param': p.name, 'data-kind': p.kind });
    this.err = el('div', { class: 'field-error', id: this.errId, role: 'alert' });
  }

  labelText() { return humanize(this.p.name) + (this.p.required ? ' (required)' : ''); }

  help(extra = '') {
    const p = this.p;
    const parts = [el('code', { class: 'flag' }, p.flag), ` ${p.help}`];
    if (p.applies_text && !p.help.includes(p.applies_text)) parts.push(` (${p.applies_text})`);
    const d = defaultText(p);
    if (d !== null && !/\(default/i.test(p.help)) parts.push(' ', el('span', { class: 'default' }, `Default: ${d}.`));
    if (extra) parts.push(` ${extra}`);
    return el('p', { class: 'help', id: this.helpId }, ...parts);
  }

  describedBy() { return `${this.helpId} ${this.errId}`; }

  // the API value; undefined: not given
  value() { return undefined; }
  names() { return [].concat(this.value() ?? []).map(String); }

  setError(msgs) {
    const on = !!(msgs && msgs.length);
    this.err.textContent = on ? msgs.join(' ') : '';
    this.el.classList.toggle('invalid', on);
    for (const c of this.controls()) {
      if (on) c.setAttribute('aria-invalid', 'true'); else c.removeAttribute('aria-invalid');
    }
  }

  controls() { return [...this.el.querySelectorAll('input:not([type=file]), select, textarea, button.choose')]; }
  changed() { this.form.changed(this); }
  consumed() {}
  renew() {}
  destroy() {}
}

class EnumField extends Field {
  constructor(form, p) {
    super(form, p);
    this.input = el('select', { id: this.id, 'aria-describedby': this.describedBy() },
      ...(p.required || p.default != null ? [] : [el('option', { value: '' }, '(not given)')]),
      ...p.choices.map((c) => el('option', { value: c }, c + (c === p.default ? ' (default)' : ''))));
    if (p.default != null) this.input.value = p.default;
    this.input.addEventListener('change', () => this.changed());
    this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
  }

  value() { return this.input.value === '' || this.input.value === this.p.default ? undefined : this.input.value; }
  current() { return this.input.value === '' ? undefined : this.input.value; }
}

class NumberField extends Field {
  constructor(form, p) {
    super(form, p);
    this.input = el('input', { type: 'text', id: this.id, inputmode: 'decimal', autocomplete: 'off',
      spellcheck: 'false', placeholder: defaultText(p) ?? '', 'aria-describedby': this.describedBy(),
      required: p.required || null });
    this.input.addEventListener('input', () => this.changed());
    this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
  }

  value() {
    const t = this.input.value.trim();
    if (t === '') return undefined;
    const x = Number(t);
    return Number.isFinite(x) && /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t) ? x : t;
  }
}

class FlagField extends Field {
  constructor(form, p) {
    super(form, p);
    this.input = el('input', { type: 'checkbox', id: this.id, 'aria-describedby': this.describedBy() });
    this.input.checked = !!p.default;
    this.input.addEventListener('change', () => this.changed());
    this.el.classList.add('check');
    this.el.append(this.input, el('label', { for: this.id }, this.labelText()), this.help(), this.err);
  }

  value() { return this.input.checked === !!this.p.default ? undefined : this.input.checked; }
}

class TextField extends Field {
  constructor(form, p) {
    super(form, p);
    this.input = el('input', { type: 'text', id: this.id, autocomplete: 'off', spellcheck: 'false',
      placeholder: defaultText(p) || '', 'aria-describedby': this.describedBy(), required: p.required || null });
    this.input.addEventListener('input', () => this.changed());
    this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
  }

  value() { const t = this.input.value.trim(); return t === '' ? undefined : t; }
}

// -p: one control per point-cloud attribute of the mode (x-oms `attributes`), sent as key=value,…
// with only the keys changed from their defaults
class AttrsField extends Field {
  constructor(form, p) {
    super(form, p);
    this.inputs = new Map();
    const fs = el('fieldset', { class: 'attrs', 'aria-describedby': this.describedBy() },
      el('legend', {}, this.labelText()));
    for (const a of p.attributes || []) {
      const id = `${this.id}-${a.key}`;
      const s = a.schema || {};
      let input;
      if (s.type === 'enum') {
        input = el('select', { id, 'aria-describedby': `${id}-help` },
          ...s.choices.map((c) => el('option', { value: c }, c + (String(c) === String(a.default) ? ' (default)' : ''))));
        input.value = a.default;
      } else {
        input = el('input', { type: 'text', id, inputmode: s.type === 'integer' || s.type === 'number' ? 'decimal' : null,
          placeholder: a.default ?? '', autocomplete: 'off', spellcheck: 'false', 'aria-describedby': `${id}-help` });
      }
      input.addEventListener(input.tagName === 'SELECT' ? 'change' : 'input', () => this.changed());
      this.inputs.set(a.key, { input, a });
      fs.append(el('div', { class: 'attr', 'data-attr': a.key }, el('label', { for: id }, el('code', {}, a.key)), input,
        el('span', { class: 'help', id: `${id}-help` }, `${a.effect}. Default: ${a.default}.`)));
    }
    this.el.append(fs, this.help(), this.err);
  }

  value() {
    const parts = [];
    for (const [key, { input, a }] of this.inputs) {
      const v = input.value.trim();
      if (v !== '' && v !== String(a.default)) parts.push(`${key}=${v}`);
    }
    return parts.length ? parts.join(',') : undefined;
  }
}

// A map of the workspace: one of the maps, or (for the mode that writes maps) a new name too
class MapField extends Field {
  constructor(form, p, writes) {
    super(form, p);
    this.writes = writes;
    this.maps = [];
    if (writes) {
      const list = nextId('maps');
      this.input = el('input', { type: 'text', id: this.id, list, autocomplete: 'off', spellcheck: 'false',
        'aria-describedby': `${this.describedBy()} ${this.id}-note`, required: p.required || null });
      this.datalist = el('datalist', { id: list });
      this.note = el('p', { class: 'help map-note', id: `${this.id}-note`, 'aria-live': 'polite' });
      this.input.addEventListener('input', () => { this.drawNote(); this.changed(); });
      this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.datalist,
        this.help('Type a new name to create a map, or an existing one to extend it.'), this.note, this.err);
    } else {
      this.input = el('select', { id: this.id, 'aria-describedby': this.describedBy() },
        el('option', { value: '' }, 'Loading maps…'));
      this.input.addEventListener('change', () => this.changed());
      this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
    }
    this.load();
  }

  async load() {
    try { this.maps = await getJson('/api/maps'); } catch { this.maps = []; }
    const keep = this.input.value;
    if (this.writes) {
      this.datalist.replaceChildren(...this.maps.map((m) => el('option', { value: m.name })));
      this.drawNote();
    } else {
      this.input.replaceChildren(el('option', { value: '' }, this.maps.length ? 'Choose a map' : 'No maps yet'),
        ...this.maps.map((m) => el('option', { value: m.name }, m.name)));
      if (keep) this.input.value = keep;
    }
  }

  drawNote() {
    const v = this.input.value.trim();
    const known = this.maps.some((m) => m.name === v);
    this.note.textContent = !v ? '' : known ? `Map ${v} exists: the inputs extend it.` : `A new map ${v} will be created.`;
  }

  value() { const v = (this.input.value || '').trim(); return v === '' ? undefined : v; }
}

// Path inputs: files uploaded from this computer (at once, so they can be checked), or paths inside
// the workspace. Several values keep their order, which is visible and editable when the order
// matters (`ordered`). An upload is consumed by the one request it is given to: the chosen files are
// kept in the page and uploaded again for the next request (`renew`).
export class FilesField extends Field {
  constructor(form, p, { big = false, preview = false } = {}) {
    super(form, p);
    this.items = [];
    this.multiple = !!p.multiple;
    this.ordered = !!p.ordered;
    this.previewOn = preview;
    const accept = (p.accepts || []).join(',');
    this.file = el('input', { type: 'file', id: `${this.id}-file`, accept: accept || null, multiple: this.multiple || null,
      class: 'file-input', tabindex: '-1', 'aria-hidden': 'true' });
    this.file.addEventListener('change', () => { this.addFiles([...this.file.files]); this.file.value = ''; });
    this.choose = el('button', { type: 'button', class: `choose ${big ? 'primary' : 'secondary'}`, id: this.id,
      'aria-describedby': this.describedBy() },
    this.multiple ? 'Choose files…' : 'Choose a file…');
    this.choose.addEventListener('click', () => this.file.click());
    this.preview = el('img', { class: 'preview', alt: '', hidden: true, 'data-testid': 'preview' });
    this.drop = el('div', { class: `dropzone${big ? ' big' : ''}`, 'data-drop': p.name, 'data-testid': 'dropzone' },
      el('p', {}, this.multiple ? 'Drop files here, or ' : 'Drop a file here, or '), this.choose, this.file,
      accept ? el('p', { class: 'help' }, `Accepted: ${(p.accepts || []).join(' ')}`) : null,
      preview ? this.preview : null);
    for (const t of ['dragenter', 'dragover']) this.drop.addEventListener(t, (e) => { e.preventDefault(); this.drop.classList.add('over'); });
    for (const t of ['dragleave', 'drop']) this.drop.addEventListener(t, () => this.drop.classList.remove('over'));
    this.drop.addEventListener('drop', (e) => { e.preventDefault(); this.addFiles([...e.dataTransfer.files]); });
    const pathId = `${this.id}-path`;
    this.path = el('input', { type: 'text', id: pathId, autocomplete: 'off', spellcheck: 'false',
      placeholder: 'e.g. inputs/photo.jpg' });
    const addPath = el('button', { type: 'button', class: 'secondary' }, 'Add path');
    const add = () => {
      const v = this.path.value.trim();
      if (!v) return;
      this.path.value = '';
      if (!this.multiple) this.removeAll();
      this.items.push({ name: v, path: v, state: 'path' });
      this.render(); this.changed();
    };
    addPath.addEventListener('click', add);
    this.path.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    this.list = el(this.ordered ? 'ol' : 'ul', { class: 'files', 'aria-label': `${humanize(p.name)}${this.ordered ? ', in the order they are used' : ''}` });
    this.sort = el('button', { type: 'button', class: 'secondary small', hidden: true }, 'Sort by name');
    this.sort.addEventListener('click', () => {
      this.items.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      this.render(); this.changed();
    });
    this.el.append(el('fieldset', { class: 'files-group', 'aria-describedby': this.describedBy() },
      el('legend', {}, this.labelText()),
      this.drop,
      el('div', { class: 'path-row' }, el('label', { for: pathId, class: 'sub' }, 'or a path in the workspace'), this.path, addPath),
      this.ordered ? el('p', { class: 'help order-note' }, 'Used in this order, first to last (the latest observation wins). Reorder with the arrow buttons.') : null,
      this.list, this.ordered ? this.sort : null, this.help(), this.err));
  }

  addFiles(files) {
    if (!files.length) return;
    if (!this.multiple) { this.removeAll(); files = files.slice(0, 1); }
    for (const f of files) {
      const it = { name: f.name, file: f, size: f.size };
      this.items.push(it);
      this.send(it);
    }
    if (this.previewOn && files[0] && /^image\//.test(files[0].type || '')) {
      if (this.preview.src) URL.revokeObjectURL(this.preview.src);
      this.preview.src = URL.createObjectURL(files[0]);
      this.preview.alt = `Preview of ${files[0].name}`;
      this.preview.hidden = false;
    }
    this.render(); this.changed();
  }

  // upload a chosen file (its field is `it.owner`, which changes when another form adopts it)
  send(it) {
    Object.assign(it, { owner: this, state: 'uploading', progress: 0, upload: null, path: null, consumed: false, error: null });
    const req = upload(it.file, (x) => { it.progress = x; it.owner.renderItem(it); });
    it.request = req;
    it.promise = req.then((u) => {
      if (!it.owner.items.includes(it)) { discardUpload(u.id); return; }
      it.upload = u; it.path = u.path; it.state = 'ready';
    }).catch((err) => { it.state = 'failed'; it.error = err.message; })
      .finally(() => { it.request = null; it.owner.render(); it.owner.changed(); });
  }

  removeAll() { for (const it of [...this.items]) this.remove(it, false); }

  remove(it, notify = true) {
    it.request?.abort();
    if (it.upload && !it.consumed) discardUpload(it.upload.id);
    this.items = this.items.filter((x) => x !== it);
    if (!this.items.some((x) => x.file) && this.previewOn) { this.preview.hidden = true; this.preview.removeAttribute('src'); }
    if (notify) { this.render(); this.changed(); }
  }

  move(it, d) {
    const i = this.items.indexOf(it);
    const j = i + d;
    if (j < 0 || j >= this.items.length) return;
    [this.items[i], this.items[j]] = [this.items[j], this.items[i]];
    this.render(); this.changed();
    const row = this.list.children[j];
    (row?.querySelector(d < 0 ? '.up' : '.down:not([disabled])') || row?.querySelector('.up:not([disabled]), .down:not([disabled])'))?.focus();
  }

  stateText(it) {
    if (it.state === 'uploading') return `uploading ${Math.round((it.progress || 0) * 100)} %`;
    if (it.state === 'failed') return `upload failed: ${it.error}`;
    if (it.state === 'path') return 'workspace path';
    return `uploaded${it.size != null ? `, ${fmtBytes(it.size)}` : ''}`;
  }

  renderItem(it) {
    const st = it.row?.querySelector('.file-state');
    if (st) st.textContent = this.stateText(it);
  }

  render() {
    clear(this.list);
    this.items.forEach((it, i) => {
      const n = this.items.length;
      const row = el('li', { class: `file ${it.state}`, 'data-name': it.name },
        el('span', { class: 'file-name' }, it.name), el('span', { class: 'file-state' }, this.stateText(it)));
      if (this.ordered && n > 1) {
        const up = el('button', { type: 'button', class: 'icon up', 'aria-label': `Move ${it.name} earlier`, disabled: i === 0 || null }, '↑');
        const down = el('button', { type: 'button', class: 'icon down', 'aria-label': `Move ${it.name} later`, disabled: i === n - 1 || null }, '↓');
        up.addEventListener('click', () => this.move(it, -1));
        down.addEventListener('click', () => this.move(it, 1));
        row.append(up, down);
      }
      const rm = el('button', { type: 'button', class: 'icon', 'aria-label': `Remove ${it.name}` }, '✕');
      rm.addEventListener('click', () => this.remove(it));
      row.append(rm);
      it.row = row;
      this.list.append(row);
    });
    this.sort.hidden = !(this.ordered && this.items.length > 1);
  }

  settled() { return Promise.all(this.items.map((it) => it.promise).filter(Boolean)); }
  names() { return this.items.map((it) => it.name); }
  failed() { return this.items.filter((it) => it.state === 'failed'); }

  value() {
    const paths = this.items.filter((it) => it.path).map((it) => it.path);
    if (!paths.length) return undefined;
    return this.multiple ? paths : paths[0];
  }

  // the request consumed the uploads (the service deletes them when it ends)
  consumed() { for (const it of this.items) if (it.upload) it.consumed = true; }
  // upload the consumed files again, for the next request
  renew() {
    let any = false;
    for (const it of this.items) if (it.consumed && it.file) { this.send(it); any = true; }
    if (any) { this.render(); this.form.changed(); }
  }
  clearAll() { this.removeAll(); this.render(); this.form.changed(); }
  // hand the chosen files to another field of the same parameter (another operation's form)
  release() { const items = this.items; this.items = []; return items; }
  adopt(items) {
    const keep = this.multiple ? items : items.slice(0, 1);
    for (const it of items) if (!keep.includes(it) && it.upload && !it.consumed) discardUpload(it.upload.id);
    this.items = keep;
    const first = keep.find((it) => it.file);
    if (this.previewOn && first && /^image\//.test(first.file.type || '')) {
      this.preview.src = URL.createObjectURL(first.file);
      this.preview.alt = `Preview of ${first.name}`;
      this.preview.hidden = false;
    }
    for (const it of keep) it.owner = this;
    this.render();
    this.renew();
  }
  destroy() {
    for (const it of this.items) { it.request?.abort(); if (it.upload && !it.consumed) discardUpload(it.upload.id); }
    if (this.preview.src) URL.revokeObjectURL(this.preview.src);
  }
}

function makeField(form, p, op, options) {
  switch (p.kind) {
    case 'enum': return new EnumField(form, p);
    case 'number': return new NumberField(form, p);
    case 'flag': return new FlagField(form, p);
    case 'attrs': return new AttrsField(form, p);
    case 'map': return new MapField(form, p, op.writesMap === p);
    case 'image': case 'images': case 'images_or_video':
      return new FilesField(form, p, options.files?.[p.name] || {});
    default: return new TextField(form, p);  // a kind this page does not know yet still gets a field
  }
}

// ---------------------------------------------------------------------------------------- the form

// A form for `op` into `container`. `fixed`: values set by the page (not rendered); `files`:
// options of a path field by parameter name ({big, preview}).
export class OpForm {
  constructor(container, op, { fixed = {}, files = {} } = {}) {
    this.op = op;
    this.fixed = { ...fixed };
    this.fields = [];
    this.seq = 0;
    this.lastResult = null;
    this.touched = new Set();  // the fields the user changed: only their messages show, until a run
    this.reveal = false;  // a run was asked for: every message shows
    this.general = el('div', { class: 'form-problems' });
    this.command = el('code', { class: 'command', 'data-testid': 'command' });
    this.root = el('div', { class: 'op-form', 'data-op': op.id });
    for (const p of op.params) {
      if (p.name in this.fixed) continue;
      const f = makeField(this, p, op, { files });
      this.fields.push(f);
      this.root.append(f.el);
    }
    this.commandRow = el('p', { class: 'command-row' }, el('span', { class: 'sub' }, 'Command: '), this.command);
    this.root.append(this.general, this.commandRow);
    container.append(this.root);
    this.applyApplies();
    this.command.textContent = '(complete the required fields)';
  }

  field(name) { return this.fields.find((f) => f.p.name === name) || null; }

  // the current value of a parameter for `applies`: its value, else its default
  current(name) {
    if (name in this.fixed) return this.fixed[name];
    const f = this.field(name);
    const v = f ? (f.current ? f.current() : f.value()) : undefined;
    if (v !== undefined) return v;
    return this.op.param(name)?.default ?? undefined;
  }

  names(name) {
    const f = this.field(name);
    return f ? f.names() : [].concat(this.current(name) ?? []).map(String);
  }

  holdsAny(conditions) {
    if (!conditions || !conditions.length) return true;
    return conditions.some((w) => holds(w, (n) => this.current(n), (n) => this.names(n)));
  }

  applyApplies() {
    for (const f of this.fields) {
      const on = this.holdsAny(f.p.applies);
      f.el.hidden = !on;
      f.active = on;
    }
  }

  // the output the response will carry for the current values (its format names the download)
  result() { return this.op.results.find((o) => this.holdsAny(o.when)) || this.op.results[0] || null; }

  values() {
    const out = { ...this.fixed };
    for (const f of this.fields) {
      if (!f.active) continue;
      const v = f.value();
      if (v !== undefined) out[f.p.name] = v;
    }
    return out;
  }

  failedUploads() { return this.fields.flatMap((f) => (f.failed ? f.failed() : [])); }

  changed(field) {
    if (field) this.touched.add(field.p.name);
    this.applyApplies();
    clearTimeout(this._t);
    this._t = setTimeout(() => this.validate(), VALIDATE_DEBOUNCE_MS);
  }

  // the command's own checks, without running anything; messages go next to their fields
  async validate() {
    clearTimeout(this._t);
    const seq = ++this.seq;
    let r;
    try {
      r = await postJson(`/api/ops/${enc(this.op.id)}/validate`, this.values());
    } catch (err) {
      r = { valid: false, problems: [{ message: err.message, parameters: [] }], by_parameter: {}, command: [] };
    }
    if (seq !== this.seq) return this.lastResult;
    this.lastResult = r;
    this.show(r.by_parameter || {}, r.problems || []);
    this.command.textContent = (r.command || []).join(' ') || '(complete the required fields)';
    return r;
  }

  show(byParameter, problems) {
    const shown = new Set();
    const quiet = new Set();  // fields the user has not changed yet: no message before a run
    for (const f of this.fields) {
      const msgs = byParameter[f.p.name];
      const on = !f.el.hidden && (this.reveal || this.touched.has(f.p.name));
      f.setError(on ? msgs : null);
      if (msgs && on) shown.add(f.p.name);
      else if (msgs && !f.el.hidden) quiet.add(f.p.name);
    }
    // by_parameter files a problem under the first parameter it concerns; the others go here
    const rest = problems.filter((p) => !shown.has((p.parameters || [])[0]) && !quiet.has((p.parameters || [])[0]));
    clear(this.general);
    for (const p of rest) this.general.append(el('p', { class: 'problem', role: 'alert' }, p.message));
  }

  // Ready to run: the uploads finished and the command's checks pass. Resolves to the validation
  // ({valid, command, inference, problems}); the messages are on the fields.
  async check() {
    this.reveal = true;
    await Promise.all(this.fields.map((f) => f.settled?.()));
    const failed = this.failedUploads();
    if (failed.length) {
      return { valid: false, problems: failed.map((it) => ({ message: `${it.name}: ${it.error}. Remove it and choose it again.`, parameters: [] })) };
    }
    return this.validate();
  }

  focusFirstError() {
    const bad = this.root.querySelector('[aria-invalid="true"]') || this.general.querySelector('.problem');
    if (bad) { if (!bad.matches('input, select, button, textarea')) bad.tabIndex = -1; bad.focus(); }
  }

  // start afresh: no message until the user changes a field again or asks for a run
  quiet() { this.reveal = false; this.touched.clear(); }
  consumed() { for (const f of this.fields) f.consumed(); }
  renew() { for (const f of this.fields) f.renew(); }
  destroy() { clearTimeout(this._t); for (const f of this.fields) f.destroy(); }
}
