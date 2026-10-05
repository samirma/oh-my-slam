// DOM helpers of the web application (the element builder is the viewer's own).
import { el } from '/static/viewer/lib/dom.js';

export { el };

export function clear(node) { node.replaceChildren(); return node; }

let uid = 0;
export function nextId(prefix = 'f') { uid += 1; return `${prefix}-${uid}`; }

export function fmtTime(t) {
  if (t == null) return '';
  return new Date(t * 1000).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' });
}

export function fmtSeconds(s) {
  if (s == null) return '';
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 120) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function fmtBytes(n) {
  if (n == null) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function humanize(name) { return String(name).replaceAll('_', ' '); }

// A colour swatch that always comes with the object's id and label (colour is never the only cue).
// (Its class names stay clear of the viewer's label layer, lib/labels.css: .obj-label is a
// positioned 3D label there.)
export function objectBadge(id, label, hex) {
  return el('span', { class: 'obj-badge' },
    el('span', { class: 'swatch', style: hexColor(hex) ? `background:${hex}` : null, 'aria-hidden': 'true' }),
    el('span', { class: 'obj-id' }, String(id)), label ? el('span', { class: 'obj-name' }, label) : '');
}

// `hex` when it is an sRGB colour #rrggbb (the only form put in a style attribute), else null.
export function hexColor(hex) { return typeof hex === 'string' && /^#[0-9a-fA-F]{6}$/.test(hex) ? hex : null; }

// A status message area: kind is 'error' | 'info' | 'warn'.
export function notice(kind, ...children) {
  return el('div', { class: `notice ${kind}`, role: kind === 'error' ? 'alert' : 'status' }, ...children);
}

export function stateBadge(state) { return el('span', { class: `state state-${state}` }, state); }
