// The segmented image (segment.sh's segmented.png) with its per-object regions. Each object's mask
// is painted opaque in exactly the object's colour (the §2.4 colour contract), so the object under a
// pixel is the one whose colour that pixel has: a click selects it, and the selected object's
// region is outlined while the rest is dimmed.
import { el, objectBadge } from './dom.js';

function hex2rgb(hex) { const n = parseInt(String(hex).slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

// `objects`: [{id, label, hex}], `selection`: the page's Selection. Returns the element.
export function segmentedImage(url, objects, selection, alt) {
  const byRgb = new Map(objects.filter((o) => o.hex).map((o) => [hex2rgb(o.hex).join(','), o]));
  const canvas = el('canvas', { class: 'seg-canvas', role: 'img', 'aria-label': alt });
  const caption = el('p', { class: 'seg-caption', 'aria-live': 'polite' }, 'Click a region to select its object.');
  const wrap = el('figure', { class: 'segmented', 'data-testid': 'segmented' }, canvas, caption);
  const img = new Image();
  let pixels = null;
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
    wrap.dataset.ready = 'true';
    draw(selection.id);
  };
  img.src = url;

  function objectAt(x, y) {
    if (!pixels) return null;
    const i = (y * pixels.width + x) * 4;
    const d = pixels.data;
    return byRgb.get(`${d[i]},${d[i + 1]},${d[i + 2]}`) || null;
  }

  function draw(id) {
    if (!pixels) return;
    const ctx = canvas.getContext('2d');
    const sel = objects.find((o) => o.id === id);
    if (!sel || !sel.hex) {
      ctx.putImageData(pixels, 0, 0);
      caption.replaceChildren('Click a region to select its object.');
      return;
    }
    const [r, g, b] = hex2rgb(sel.hex);
    const out = new ImageData(new Uint8ClampedArray(pixels.data), pixels.width, pixels.height);
    const d = out.data;
    const W = pixels.width, H = pixels.height;
    const inside = (k) => d[k] === r && d[k + 1] === g && d[k + 2] === b;
    const mask = new Uint8Array(W * H);
    for (let p = 0, k = 0; p < W * H; p++, k += 4) mask[p] = inside(k) ? 1 : 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const p = y * W + x, k = p * 4;
        if (mask[p]) {
          const edge = x === 0 || y === 0 || x === W - 1 || y === H - 1 || !mask[p - 1] || !mask[p + 1] || !mask[p - W] || !mask[p + W];
          if (edge) { d[k] = 255; d[k + 1] = 255; d[k + 2] = 255; }
        } else {
          d[k] *= 0.35; d[k + 1] *= 0.35; d[k + 2] *= 0.35;
        }
      }
    }
    ctx.putImageData(out, 0, 0);
    caption.replaceChildren('Selected: ', objectBadge(sel.id, sel.label, sel.hex));
  }

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * canvas.width);
    const y = Math.floor((e.clientY - rect.top) / rect.height * canvas.height);
    const o = objectAt(x, y);
    selection.set(o ? o.id : null);
  });
  selection.join((id) => { wrap.dataset.selected = id ?? ''; draw(id); });
  return wrap;
}
