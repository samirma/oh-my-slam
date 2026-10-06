// DOM helpers and formatting of the web application.

// A DOM element with attributes and children (attributes that are null, undefined or false are
// left out; true gives an empty attribute; "class" sets the class name).
export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  e.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return e;
}

export function clear(node) { node.replaceChildren(); return node; }

let uid = 0;
export function nextId(prefix = 'f') { uid += 1; return `${prefix}-${uid}`; }

export function fmtDate(t) {
  if (t == null || !Number.isFinite(Number(t))) return '';
  return new Date(Number(t) * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function fmtSeconds(s) {
  if (s == null || !Number.isFinite(Number(s))) return '';
  s = Number(s);
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${Math.round(s % 60)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

// A running clock: 0:07, 1:05, 1:02:03
export function fmtClock(s) {
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

export function fmtBytes(n) {
  if (n == null) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function fmtNumber(v) {
  if (typeof v !== 'number') return String(v);
  return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

export function humanize(name) {
  const s = String(name).replaceAll('_', ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// `hex` when it is an sRGB colour #rrggbb (the only form put in a style attribute), else null.
export function hexColor(hex) { return typeof hex === 'string' && /^#[0-9a-fA-F]{6}$/.test(hex) ? hex : null; }

// An object's colour always comes with its id and label (colour is never the only cue).
export function objectBadge(id, label, hex) {
  const c = hexColor(hex);
  return el('span', { class: 'obj-badge' },
    el('span', { class: 'swatch', style: c ? `background:${c}` : null, 'aria-hidden': 'true' }),
    el('span', { class: 'obj-id' }, `#${id}`), label ? el('span', { class: 'obj-name' }, label) : null);
}

// A message area: kind is 'error' | 'warn' | 'info' | 'ok'. Errors are alerts; the rest status.
export function notice(kind, ...children) {
  const icon = { error: '✕', warn: '!', info: 'i', ok: '✓' }[kind] || 'i';
  return el('div', { class: `notice ${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    el('span', { class: 'notice-icon', 'aria-hidden': 'true' }, icon), el('div', { class: 'notice-body' }, ...children));
}

// A definition list of [label, value] pairs.
export function facts(pairs, cls = 'figures') {
  return el('dl', { class: cls }, ...pairs.map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', {}, v))));
}
