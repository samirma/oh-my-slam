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
      const read = () => {
        const x = Number(input.value);
        if (c.off === 'inf' && x >= Number(c.max)) return 'inf';
        return c.kind === 'int' ? String(Math.round(x)) : String(Number(x.toFixed(decimals(c.step))));
      };
      input.addEventListener('input', () => { const v = read(); out.value = formatValue(c, v); onChange(c.key, v); });
      set = (v) => { input.value = v === 'inf' ? c.max : Number(v); out.value = formatValue(c, v); };
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
// cloud on screen omits points; '' when it shows them all. With no grid (edge 0) the omitted points
// are duplicates (one per distinct position is shown) or have non-finite coordinates.
export function budgetNote(header) {
  if (!(header.count < header.total)) return '';
  const shown = `Showing ${header.count.toLocaleString('en-US')} of ${header.total.toLocaleString('en-US')} points`;
  const complete = '(display budget; PLY outputs and the map stay complete).';
  const e = header.voxel;
  if (!(e > 0)) return `${shown}: one per distinct position, non-finite points omitted ${complete}`;
  const edge = e >= 1 ? `${e.toFixed(2)} m` : e >= 0.01 ? `${(e * 100).toFixed(1)} cm` : `${(e * 1000).toFixed(2)} mm`;
  return `${shown}: one per voxel of ${edge} edge ${complete}`;
}
