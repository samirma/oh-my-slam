// Forms generated from the API description (http_server.md "Web application", "Forms"): one field
// per parameter of an operation, from its registry entry (`x-oms`: kind, flag, help, default,
// choices, bounds, multiple/ordered, accepts, applies, the -p attributes). Nothing here names a
// command or an option: a new option is a new field. Fields that do not apply to the current
// choices (`applies`) are hidden and not sent; every change is checked by the command's own rules
// (POST /api/ops/<op>/validate) and each message is shown next to the field it names
// (`by_parameter`), before submission.
import { el, clear, nextId, humanize, fmtBytes } from './dom.js';
import { getJson, postJson, upload, discardUpload, ApiError, enc } from './api.js';
import { store } from './store.js';

const VALIDATE_DEBOUNCE_MS = 250;

function suffix(name) { const m = /\.[^./\\]+$/.exec(name); return m ? m[0].toLowerCase() : ''; }

// The suffixes of a video: those an images-or-video input takes beyond any single-image input's.
function videoSuffixes(p) {
  const images = new Set();
  for (const op of store.ops.values()) for (const q of op.ofKind('image')) for (const s of q.accepts || []) images.add(s);
  return new Set((p.accepts || []).filter((s) => !images.has(s)));
}

function defaultText(p) {
  if (p.default === null || p.default === undefined || p.kind === 'attrs') return null;
  if (p.kind === 'flag') return p.default ? 'on' : 'off';
  return String(p.default);
}

// ------------------------------------------------------------------------------------------- fields

class Field {
  constructor(form, p) {
    this.form = form;
    this.p = p;
    this.id = nextId(`f-${p.name}`);
    this.errId = `${this.id}-err`;
    this.helpId = `${this.id}-help`;
    this.el = el('div', { class: 'field', 'data-param': p.name, 'data-kind': p.kind });
    this.err = el('div', { class: 'field-error', id: this.errId, 'aria-live': 'polite' });
  }

  labelText() { return humanize(this.p.name) + (this.p.required ? ' (required)' : ''); }

  help(extra = '') {
    const p = this.p;
    const parts = [el('code', {}, p.flag), ` ${p.help}`];
    if (p.applies_text) parts.push(` (${p.applies_text})`);
    const d = defaultText(p);
    if (d !== null && !/\bdefault\b/i.test(p.help)) parts.push(el('span', { class: 'default' }, ` Default: ${d}.`));
    if (extra) parts.push(` ${extra}`);
    return el('p', { class: 'help', id: this.helpId }, ...parts);
  }

  describedBy() { return `${this.helpId} ${this.errId}`; }

  // the API value; undefined: not given
  value() { return undefined; }
  given() { const v = this.value(); return v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && !v.length); }
  pending() { return false; }

  setError(msgs) {
    const on = !!(msgs && msgs.length);
    this.err.textContent = on ? msgs.join(' ') : '';
    this.el.classList.toggle('invalid', on);
    for (const c of this.controls()) {
      if (on) c.setAttribute('aria-invalid', 'true'); else c.removeAttribute('aria-invalid');
    }
  }

  controls() { return [...this.el.querySelectorAll('input, select, textarea')]; }
  changed() { this.form.changed(this); }
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

  value() { return this.input.value === '' ? undefined : this.input.value; }
  set(v) { this.input.value = v; }
}

class NumberField extends Field {
  constructor(form, p) {
    super(form, p);
    this.input = el('input', { type: 'number', id: this.id, step: 'any', inputmode: 'decimal',
      min: p.minimum ?? (p.exclusive_minimum ?? null), placeholder: defaultText(p) ?? '',
      'aria-describedby': this.describedBy(), required: p.required || null });
    this.input.addEventListener('input', () => this.changed());
    this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
  }

  value() {
    const t = this.input.value.trim();
    if (t === '') return this.input.validity.badInput ? 'not a number' : undefined;
    const x = Number(t);
    return Number.isFinite(x) ? x : t;
  }

  set(v) { this.input.value = v; }
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
  set(v) { this.input.checked = !!v; }
}

class TextField extends Field {
  constructor(form, p, placeholder = '') {
    super(form, p);
    this.input = el('input', { type: 'text', id: this.id, autocomplete: 'off', spellcheck: 'false',
      placeholder: placeholder || defaultText(p) || '', 'aria-describedby': this.describedBy(),
      required: p.required || null });
    this.input.addEventListener('input', () => this.changed());
    this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
  }

  value() { const t = this.input.value.trim(); return t === '' ? undefined : t; }
  set(v) { this.input.value = v ?? ''; }
}

// -o: a plain name in the job's folder; the service names the result after its format by default
class FileOutField extends TextField {
  constructor(form, p) {
    super(form, p, 'result');
    this.el.querySelector('.help').append(' The service keeps the result in the job\'s folder; leave empty for result.<format>.');
  }
}

// -d: written only when asked; then a plain name in the job's folder
class FolderOutField extends Field {
  constructor(form, p) {
    super(form, p);
    const cbId = `${this.id}-on`;
    this.on = el('input', { type: 'checkbox', id: cbId, 'aria-describedby': this.describedBy() });
    this.input = el('input', { type: 'text', id: this.id, value: p.name, autocomplete: 'off',
      spellcheck: 'false', 'aria-describedby': this.describedBy(), disabled: true });
    this.on.addEventListener('change', () => { this.input.disabled = !this.on.checked; this.changed(); });
    this.input.addEventListener('input', () => this.changed());
    this.el.append(
      el('div', { class: 'check' }, this.on, el('label', { for: cbId }, `Write ${humanize(p.name)}`)),
      el('label', { for: this.id, class: 'sub' }, 'Folder name'), this.input, this.help(), this.err);
  }

  value() { return this.on.checked ? (this.input.value.trim() || undefined) : undefined; }
  set(v) { this.on.checked = v != null; this.input.disabled = !this.on.checked; if (v != null) this.input.value = v; }
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
        input = el('select', { id }, ...s.choices.map((c) => el('option', { value: c }, c + (c === a.default ? ' (default)' : ''))));
        input.value = a.default;
      } else if (s.type === 'integer' || s.type === 'number') {
        input = el('input', { type: 'text', id, inputmode: 'decimal', placeholder: a.default, autocomplete: 'off' });
      } else {
        input = el('input', { type: 'text', id, placeholder: a.default ?? '', autocomplete: 'off' });
      }
      input.addEventListener(input.tagName === 'SELECT' ? 'change' : 'input', () => this.changed());
      this.inputs.set(a.key, { input, a });
      fs.append(el('div', { class: 'attr' }, el('label', { for: id }, el('code', {}, a.key)), input,
        el('span', { class: 'help' }, `${a.effect}. Default: ${a.default}.`)));
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

  set(v) {
    for (const kv of String(v || '').split(',')) {
      const [k, x] = kv.split('=');
      const it = this.inputs.get(k);
      if (it) it.input.value = x;
    }
  }
}

// A map of the workspace: one of the maps, or (for the mode that writes maps) a new name too
class MapField extends Field {
  constructor(form, p, writes) {
    super(form, p);
    this.writes = writes;
    if (writes) {
      const list = nextId('maps');
      this.input = el('input', { type: 'text', id: this.id, list, autocomplete: 'off', spellcheck: 'false',
        'aria-describedby': this.describedBy(), required: p.required || null });
      this.datalist = el('datalist', { id: list });
      this.input.addEventListener('input', () => this.changed());
      this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.datalist,
        this.help('Pick an existing map to extend it, or type a new name to create one.'), this.err);
    } else {
      this.input = el('select', { id: this.id, 'aria-describedby': this.describedBy() },
        el('option', { value: '' }, 'Loading maps…'));
      this.input.addEventListener('change', () => this.changed());
      this.el.append(el('label', { for: this.id }, this.labelText()), this.input, this.help(), this.err);
    }
    this.load();
  }

  async load() {
    let maps = [];
    try { maps = await getJson('/api/maps'); } catch { /* shown by validation */ }
    const keep = this.input.value || this.wanted;
    if (this.writes) {
      this.datalist.replaceChildren(...maps.map((m) => el('option', { value: m.name })));
    } else {
      this.input.replaceChildren(el('option', { value: '' }, maps.length ? 'Choose a map' : 'No maps yet'),
        ...maps.map((m) => el('option', { value: m.name }, m.name)));
      if (keep) this.input.value = keep;
    }
  }

  value() { const v = (this.input.value || '').trim(); return v === '' ? undefined : v; }
  set(v) { this.wanted = v; this.input.value = v; }
}

// Path inputs: files uploaded from this computer, or paths inside the workspace. Several values
// keep their order, which is visible and editable when the order matters (`ordered`).
export class FilesField extends Field {
  constructor(form, p, { label } = {}) {
    super(form, p);
    this.items = [];
    this.multiple = !!p.multiple;
    this.ordered = !!p.ordered;
    const accept = (p.accepts || []).join(',');
    this.file = el('input', { type: 'file', id: this.id, accept: accept || null, multiple: this.multiple || null,
      hidden: true, 'aria-label': humanize(p.name) });
    this.file.addEventListener('change', () => { this.addFiles([...this.file.files]); this.file.value = ''; });
    const choose = el('button', { type: 'button', class: 'secondary', 'aria-describedby': this.describedBy() },
      this.multiple ? 'Choose files…' : 'Choose a file…');
    choose.addEventListener('click', () => this.file.click());
    this.choose = choose;
    this.drop = el('div', { class: 'dropzone', 'data-drop': p.name },
      el('p', {}, this.multiple ? 'Drop files here, or ' : 'Drop a file here, or '), choose, this.file,
      accept ? el('p', { class: 'help' }, `Accepted: ${(p.accepts || []).join(' ')}`) : '');
    for (const t of ['dragenter', 'dragover']) this.drop.addEventListener(t, (e) => { e.preventDefault(); this.drop.classList.add('over'); });
    for (const t of ['dragleave', 'drop']) this.drop.addEventListener(t, () => this.drop.classList.remove('over'));
    this.drop.addEventListener('drop', (e) => { e.preventDefault(); this.addFiles([...e.dataTransfer.files]); });
    const pathId = `${this.id}-path`;
    this.path = el('input', { type: 'text', id: pathId, autocomplete: 'off', spellcheck: 'false',
      placeholder: this.multiple ? 'e.g. inputs/a.jpg' : 'e.g. inputs/photo.jpg' });
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
    this.list = el(this.ordered ? 'ol' : 'ul', { class: 'files', 'aria-label': `${humanize(p.name)}${this.ordered ? ' in order' : ''}` });
    this.el.append(
      el('div', { class: 'label', id: `${this.id}-label` }, label || this.labelText()),
      this.drop,
      el('div', { class: 'path-row' }, el('label', { for: pathId, class: 'sub' }, 'or a path in the workspace'), this.path, addPath),
      this.ordered ? el('p', { class: 'help order-note' }, 'The inputs are used in this order, first to last; reorder them with the arrows.') : '',
      this.list, this.help(), this.err);
  }

  addFiles(files) {
    if (!files.length) return;
    if (!this.multiple) { this.removeAll(); files = files.slice(0, 1); }
    for (const f of files) {
      const it = { name: f.name, file: f, size: f.size, state: 'uploading', progress: 0 };
      this.items.push(it);
      it.promise = upload(f, (x) => { it.progress = x; this.renderItem(it); })
        .then((u) => { it.upload = u; it.path = u.path; it.state = 'ready'; })
        .catch((err) => { it.state = 'failed'; it.error = err.message; })
        .finally(() => { this.render(); this.changed(); });
    }
    this.render(); this.changed();
    this.form.emit('files', this, files);
  }

  removeAll() { for (const it of [...this.items]) this.remove(it, false); }

  remove(it, notify = true) {
    if (it.upload && !it.consumed) discardUpload(it.upload.id);
    this.items = this.items.filter((x) => x !== it);
    if (notify) { this.render(); this.changed(); }
  }

  move(it, d) {
    const i = this.items.indexOf(it);
    const j = i + d;
    if (j < 0 || j >= this.items.length) return;
    [this.items[i], this.items[j]] = [this.items[j], this.items[i]];
    this.render(); this.changed();
    this.list.children[j]?.querySelector(d < 0 ? '.up' : '.down')?.focus();
  }

  renderItem(it) {
    if (!it.row) return;
    const st = it.row.querySelector('.file-state');
    if (st) st.textContent = this.stateText(it);
  }

  stateText(it) {
    if (it.state === 'uploading') return `uploading ${Math.round(it.progress * 100)} %`;
    if (it.state === 'failed') return `upload failed: ${it.error}`;
    if (it.state === 'path') return 'workspace path';
    return `uploaded${it.size != null ? `, ${fmtBytes(it.size)}` : ''}`;
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
  }

  pending() { return this.items.some((it) => it.state === 'uploading'); }
  names() { return this.items.map((it) => it.name); }

  value() {
    const paths = this.items.filter((it) => it.path).map((it) => it.path);
    if (!paths.length) return undefined;
    return this.multiple ? paths : paths[0];
  }

  set(v) {
    this.removeAll();
    for (const x of [].concat(v ?? [])) this.items.push({ name: x, path: x, state: 'path' });
    this.render();
  }

  // the job consumed the uploads (they are deleted when it ends): never discard them from here
  consumed() { for (const it of this.items) it.consumed = true; }
  destroy() { for (const it of this.items) if (it.upload && !it.consumed) discardUpload(it.upload.id); }
}

function makeField(form, p, op) {
  switch (p.kind) {
    case 'enum': return new EnumField(form, p);
    case 'number': return new NumberField(form, p);
    case 'flag': return new FlagField(form, p);
    case 'attrs': return new AttrsField(form, p);
    case 'map': return new MapField(form, p, op.writesMap === p);
    case 'file_out': return new FileOutField(form, p);
    case 'folder_out': return new FolderOutField(form, p);
    case 'image': case 'images': case 'images_or_video': return new FilesField(form, p);
    default: return new TextField(form, p);  // a kind this page does not know yet still gets a field
  }
}

// ---------------------------------------------------------------------------------------- the form

// A form for `op`. `fixed`: values set by the page (not rendered); `omit`: parameters the page
// renders itself (`external` fields, e.g. the Image page's drop zone); `viewer`: the submission asks
// for the viewer (?viewer=true).
export class OpForm {
  constructor(container, op, { fixed = {}, omit = [], external = {}, viewer = false, only = null } = {}) {
    this.op = op;
    this.fixed = { ...fixed };
    this.viewer = viewer;
    this.fields = [];
    this.external = external;
    this._listeners = new Map();
    this.seq = 0;
    this.lastResult = null;
    this.general = el('div', { class: 'form-problems', 'aria-live': 'polite' });
    this.command = el('p', { class: 'command' });
    this.root = el('div', { class: 'op-form', 'data-op': op.id });
    for (const p of op.params) {
      if (p.name in this.fixed || omit.includes(p.name)) continue;
      if (only && !only.includes(p.name)) continue;
      const f = makeField(this, p, op);
      this.fields.push(f);
      this.root.append(f.el);
    }
    this.root.append(this.general, el('div', { class: 'command-row' }, el('span', { class: 'sub' }, 'Command: '), this.command));
    container.append(this.root);
    this.applyApplies();
  }

  on(what, cb) { if (!this._listeners.has(what)) this._listeners.set(what, new Set()); this._listeners.get(what).add(cb); }
  emit(what, ...args) { for (const cb of this._listeners.get(what) || []) cb(...args); }

  field(name) { return this.fields.find((f) => f.p.name === name) || this.external[name] || null; }

  // the current value of a parameter for `applies`: its value, else its default
  current(name) {
    if (name in this.fixed) return this.fixed[name];
    const f = this.field(name);
    const v = f ? f.value() : undefined;
    if (v !== undefined) return v;
    return this.op.param(name)?.default ?? undefined;
  }

  // the names of the files a path parameter holds (to tell a video by its suffix)
  names(name) {
    const f = this.field(name);
    if (f && f.names) return f.names();
    return [].concat(this.current(name) ?? []);
  }

  applies(p) {
    if (!p.applies || !p.applies.length) return true;
    return p.applies.some((w) => {
      if (w.in) return w.in.includes(this.current(w.option));
      if (w.is === 'given') { const v = this.current(w.option); return v !== undefined && v !== null && v !== '' && v !== false; }
      if (w.is === 'video') {
        const names = this.names(w.option);
        const op = this.op.param(w.option);
        return names.length === 1 && videoSuffixes(op || {}).has(suffix(names[0]));
      }
      return true;
    });
  }

  applyApplies() {
    for (const f of this.fields) {
      const on = this.applies(f.p);
      f.el.hidden = !on;
      f.active = on;
    }
  }

  values() {
    const out = { ...this.fixed };
    for (const [name, f] of Object.entries(this.external)) { const v = f.value(); if (v !== undefined) out[name] = v; }
    for (const f of this.fields) {
      if (!f.active) continue;
      const v = f.value();
      if (v !== undefined) out[f.p.name] = v;
    }
    return out;
  }

  pending() { return this.fields.some((f) => f.pending()) || Object.values(this.external).some((f) => f.pending()); }

  setFixed(name, v) { if (v === undefined) delete this.fixed[name]; else this.fixed[name] = v; this.changed(); }

  changed() {
    this.applyApplies();
    this.emit('change');
    clearTimeout(this._t);
    this._t = setTimeout(() => this.validate(), VALIDATE_DEBOUNCE_MS);
  }

  query() { return this.viewer ? '?viewer=true' : ''; }

  // the command's own checks, without queuing anything; messages go next to their fields
  async validate() {
    const seq = ++this.seq;
    let r;
    try {
      r = await postJson(`/api/ops/${enc(this.op.id)}/validate${this.query()}`, this.values());
    } catch (err) {
      r = { valid: false, problems: [{ message: err.message, parameters: [] }], by_parameter: {}, command: [] };
    }
    if (seq !== this.seq) return this.lastResult;
    this.lastResult = r;
    this.show(r.by_parameter || {}, r.problems || []);
    this.command.textContent = (r.command || []).join(' ');
    this.emit('validated', r);
    return r;
  }

  show(byParameter, problems) {
    const shown = new Set();
    for (const f of [...this.fields, ...Object.values(this.external)]) {
      const msgs = byParameter[f.p.name];
      f.setError(f.el.hidden ? null : msgs);
      if (msgs && !f.el.hidden) shown.add(f.p.name);
    }
    // by_parameter files a problem under the first parameter it concerns; the others go on top
    const rest = problems.filter((p) => !shown.has((p.parameters || [])[0]));
    clear(this.general);
    for (const p of rest) this.general.append(el('p', { class: 'problem', role: 'alert' }, p.message));
  }

  // Submit as a job: resolves to the job, or throws ApiError (its messages shown on the fields)
  async submit() {
    if (this.pending()) throw new ApiError(0, { error: { message: 'wait for the uploads to finish' } });
    try {
      const job = await postJson(`/api/ops/${enc(this.op.id)}${this.query()}`, this.values());
      for (const f of [...this.fields, ...Object.values(this.external)]) f.consumed?.();
      return job;
    } catch (err) {
      if (err instanceof ApiError) this.show(err.byParameter, err.problems.length ? err.problems : [{ message: err.message, parameters: [] }]);
      throw err;
    }
  }

  focusFirstError() { this.root.querySelector('[aria-invalid="true"]')?.focus(); }
  destroy() { for (const f of [...this.fields, ...Object.values(this.external)]) f.destroy(); }
}
