// Live controls for the §2.2 point-cloud attributes: one per entry of /api/meta `controls`
// (core.cloud_attrs: those that apply and affect the display). Values are the -p strings; the
// server validates and derives.
import { el } from './dom.js';

function decimals(step) { const s = String(step); return s.includes('.') ? s.split('.')[1].length : 0; }
function isOff(c, v) { return c.off != null && (v === c.off || Number(v) === Number(c.off)); }
export function formatValue(c, v) {
  if (c.kind !== 'int' && c.kind !== 'float') return v;
  if (isOff(c, v)) return c.off === 'inf' ? '∞' : 'off';
  return `${Number(v).toFixed(decimals(c.step))}${c.unit ? ` ${c.unit}` : ''}`;
}

// A slider's value (-p string) at position `x`: its maximum is "off" for a control whose off value
// is inf, an int is rounded, a float keeps its step's decimals. sliderPosition is the inverse.
export function sliderValue(c, x) {
  if (c.off === 'inf' && x >= Number(c.max)) return 'inf';
  return c.kind === 'int' ? String(Math.round(x)) : String(Number(x.toFixed(decimals(c.step))));
}
export function sliderPosition(c, v) { return v === 'inf' ? c.max : Number(v); }

// The controls into `container`, set to `values` ({key: value}); `onChange(key, value)` on every
// change. Returns { markInvalid(key or null) } to flag the control an error names.
export function buildAttributeControls(container, controls, values, onChange) {
  const rows = new Map();
  for (const c of controls) {
    const id = `attr-${c.key}`;
    const row = el('div', { class: 'row attr', 'data-attr': c.key, title: `${c.help} — ${c.values}` });
    const out = el('output', { for: id });
    let input;
    let set;  // (value) → update the widget
    if (c.kind === 'choice') {
      input = el('select', { id }, ...c.options.map((o) => el('option', { value: o }, o)));
      input.addEventListener('change', () => onChange(c.key, input.value));
      set = (v) => { input.value = v; };
    } else if (c.kind === 'toggle') {
      input = el('input', { type: 'checkbox', id, role: 'switch', class: 'switch' });
      input.addEventListener('change', () => { out.value = input.checked ? 'on' : 'off'; onChange(c.key, out.value); });
      set = (v) => { input.checked = v === 'on'; out.value = v; };
    } else if (c.kind === 'int' || c.kind === 'float') {
      input = el('input', { type: 'range', id, min: c.min, max: c.max, step: c.step });
      const read = () => sliderValue(c, Number(input.value));
      input.addEventListener('input', () => { const v = read(); out.value = formatValue(c, v); onChange(c.key, v); });
      set = (v) => { input.value = sliderPosition(c, v); out.value = formatValue(c, v); };
    } else {
      input = el('input', { type: 'text', id });
      input.addEventListener('change', () => onChange(c.key, input.value.trim()));
      set = (v) => { input.value = v; };
    }
    set(values[c.key]);
    rows.set(c.key, row);
    row.append(el('label', { for: id }, c.key), input, out);
    container.append(row);
  }
  return {
    markInvalid(key) { for (const [k, row] of rows) row.classList.toggle('invalid', k === key); },
  };
}

// The display-budget notice (spec §2.5): "Showing X of Y points" with the voxel edge whenever the
// cloud on screen omits points; '' when it shows them all. With no grid (edge 0) the shown points
// are every finite one, or one per distinct position when those still exceed the budget
// (core.geometry budget_voxel_grid); the document does not say which, so the notice names both
// reasons an omitted point can have. A PLY read in the page above the budget (ply.js) shows its
// points evenly spaced in the file's order (`step` > 1).
export function budgetNote(header) {
  if (!(header.count < header.total)) return '';
  const shown = `Showing ${header.count.toLocaleString('en-US')} of ${header.total.toLocaleString('en-US')} points`;
  const complete = '(display budget; PLY outputs and the map stay complete).';
  if (header.step > 1) return `${shown}: evenly spaced in the file's order, read in this page ${complete}`;
  const e = header.voxel;
  if (!(e > 0)) return `${shown}: no voxel grid (edge 0); each omitted point has a non-finite coordinate or repeats a shown position ${complete}`;
  const edge = e >= 1 ? `${e.toFixed(2)} m` : e >= 0.01 ? `${(e * 100).toFixed(1)} cm` : `${(e * 1000).toFixed(2)} mm`;
  return `${shown}: one per voxel of ${edge} edge ${complete}`;
}

// The display budget of spec §2.5: the viewer draws every point of a cloud of at most this many
// points. It is viewer/bundle.py DISPLAY_POINT_BUDGET (tests/unit/test_viewer_server.py checks it).
export const DISPLAY_POINT_BUDGET = 16_000_000;
