// A result as the response carried it (http_server.md "Image": a scene description (JSON) "is shown
// in full, with its objects listed (label, id, colour, score)"; a point cloud (PLY) "is loaded in the
// page and drawn in 3D"; "every result can be downloaded byte for byte"). The download is the
// response body itself, byte for byte (the command's own output); the text shown is that body as
// received. What is shown follows the result's format and content, never the command: a JSON
// document that is a scene description (OpenLABEL) has its objects listed; a PLY is drawn in 3D by
// the viewer's rendering (cloudresult.js), with its header below; an image (PNG: a depth image, a
// segmented image) is shown as received.
import { el, fmtBytes, fmtSeconds, hexColor } from './dom.js';
import { cloudSection } from './cloudresult.js';

export const SHOW_TEXT_MAX = 8 << 20;  // a text body larger than this is offered as a download only

// The objects of an OpenLABEL scene description: {id, label, hex, score}, by id.
export function sceneObjects(doc) {
  const objs = (doc && doc.openlabel && doc.openlabel.objects) || {};
  const prop = (o, kind, name) => {
    const v = ((o.object_data || {})[kind] || []).find((n) => n.name === name);
    return v ? v.val : null;
  };
  return Object.entries(objs).map(([id, o]) => ({
    id: Number(id), label: o.type || o.name || '', hex: prop(o, 'text', 'color_hex'), score: prop(o, 'num', 'score'),
  })).sort((a, b) => a.id - b.id);
}

export function isScene(doc) { return !!(doc && typeof doc === 'object' && doc.openlabel && typeof doc.openlabel === 'object'); }

export function objectsTable(objects, caption) {
  const body = objects.length
    ? objects.map((o) => el('tr', { 'data-id': o.id },
      el('th', { scope: 'row', class: 'obj-id' }, `#${o.id}`),
      el('td', {}, o.label),
      el('td', {}, hexColor(o.hex) ? el('span', { class: 'hex' }, el('span', { class: 'swatch', style: `background:${o.hex}`, 'aria-hidden': 'true' }), o.hex) : '—'),
      el('td', { class: 'num' }, o.score != null ? Number(o.score).toFixed(2) : '—')))
    : [el('tr', {}, el('td', { colspan: 4, class: 'muted' }, 'No objects.'))];
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data objects', 'data-testid': 'objects' },
    el('caption', {}, caption),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'ID'), el('th', { scope: 'col' }, 'Label'),
      el('th', { scope: 'col' }, 'Colour'), el('th', { scope: 'col', class: 'num' }, 'Score'))),
    el('tbody', {}, ...body)));
}

// The rows of a stages table: { total (the `total` stage, if any), parts: every other stage with its
// `share` of the total (of their sum when there is no total), between 0 and 1 }.
export function stageShares(stages) {
  const total = stages.find((s) => s.name === 'total');
  const parts = stages.filter((s) => s.name !== 'total');
  const sum = total?.ms || parts.reduce((a, s) => a + (s.ms || 0), 0) || 1;
  return { total, parts: parts.map((s) => ({ ...s, share: Math.max(0, Math.min(1, (s.ms || 0) / sum)) })) };
}

// Per-stage timings ([{name, ms}], the command's own stage names; `total` last) as a table with a
// bar per stage (its share of the total, also given as text).
export function stagesTable(stages, caption = 'Stages') {
  const { total, parts } = stageShares(stages);
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data stages', 'data-testid': 'stages' },
    el('caption', {}, caption),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'Stage'), el('th', { scope: 'col', class: 'num' }, 'Time'),
      el('th', { scope: 'col' }, 'Share'))),
    el('tbody', {}, ...parts.map((s) => el('tr', {}, el('th', { scope: 'row' }, el('code', {}, s.name)),
      el('td', { class: 'num' }, fmtSeconds((s.ms || 0) / 1000)),
      el('td', { class: 'share' }, el('span', { class: 'bar', style: `width:${(s.share * 100).toFixed(1)}%` }), ` ${Math.round(s.share * 100)} %`)))),
    total ? el('tfoot', {}, el('tr', {}, el('th', { scope: 'row' }, 'Total'), el('td', { class: 'num' }, fmtSeconds(total.ms / 1000)), el('td', {}))) : null));
}

// A PLY's header (its text up to end_header) and what it declares: { header, format, elements:
// [{name, count}] }, or null when `blob` is no PLY.
export async function plyHeader(blob) {
  const head = new Uint8Array(await blob.slice(0, Math.min(blob.size, 1 << 20)).arrayBuffer());
  const text = new TextDecoder('latin1').decode(head);
  const end = text.indexOf('end_header');
  if (!text.startsWith('ply') || end < 0) return null;
  const header = text.slice(0, text.indexOf('\n', end) + 1 || end + 10);
  const lines = header.split(/\r?\n/);
  const format = (lines.find((l) => l.startsWith('format ')) || '').split(' ')[1] || '';
  const elements = lines.filter((l) => l.startsWith('element ')).map((l) => { const [, n, c] = l.split(/\s+/); return { name: n, count: Number(c) }; });
  return { header, format, elements };
}

// What the result says of a PLY's header (plyHeader) in a body of `size` bytes: the summary of its
// details, and the note on the point data after it.
export function plyHeaderText(ply, size) {
  const counts = ply.elements.map((e) => `element ${e.name} ${e.count.toLocaleString()}`).join(', ');
  return {
    summary: `Its PLY header (${ply.format}: ${counts || 'no element'})`,
    note: `The ${fmtBytes(size - ply.header.length)} of ${ply.format.replaceAll('_', ' ')} point data after the header are in the download.`,
  };
}

// How a result is shown: 'text' (a JSON or text body, in full, up to SHOW_TEXT_MAX), 'image', or
// 'other' (a PLY drawn in 3D, else a download only).
export function resultKind(mediaType, format, size) {
  if ((/json|text/.test(mediaType) || format === 'json') && size <= SHOW_TEXT_MAX) return 'text';
  return mediaType.startsWith('image/') ? 'image' : 'other';
}

// A text result in full: a JSON document formatted for reading (a switch shows it as received;
// the download is always the bytes as received).
function fullText(text, doc) {
  const pre = el('pre', { class: 'result-text', tabindex: '0', 'aria-label': 'The result in full', 'data-testid': 'result-text' });
  const out = [el('h3', {}, 'The result in full'), pre];
  if (doc === null) { pre.textContent = text; return out; }
  const formatted = JSON.stringify(doc, null, 2);
  const id = `as-received-${Math.random().toString(36).slice(2, 8)}`;
  const raw = el('input', { type: 'checkbox', id });
  const draw = () => { pre.textContent = raw.checked ? text : formatted; pre.dataset.view = raw.checked ? 'received' : 'formatted'; };
  raw.addEventListener('change', draw);
  draw();
  out.splice(1, 0, el('p', { class: 'check-row' }, raw, el('label', { for: id }, ' Show it exactly as received (the formatted view only adds line breaks and indentation)')));
  return out;
}

// The result section: facts, download, timings, then the content.
export async function resultView({ blob, mediaType, stages, format, downloadName }) {
  const url = URL.createObjectURL(blob);
  const box = el('section', { class: 'result', 'data-testid': 'result', 'data-format': format || '', 'aria-labelledby': 'result-h' });
  const total = stages.find((s) => s.name === 'total');
  box.append(el('h2', { id: 'result-h' }, 'Result'),
    el('div', { class: 'result-head' },
      el('a', { class: 'button primary download', href: url, download: downloadName, 'data-testid': 'download' },
        `Download ${downloadName}`),
      el('span', { class: 'muted' }, `${mediaType}, ${fmtBytes(blob.size)}${total ? `, ${fmtSeconds(total.ms / 1000)} in the command` : ''}`)));
  const content = el('div', { class: 'result-content' });
  box.append(content);
  if (stages.length) box.append(el('details', { class: 'stages-box' }, el('summary', {}, 'Per-stage timings'), stagesTable(stages, 'The command\'s stages (Server-Timing)')));

  let cloud = null;  // a point cloud's 3D view
  const kind = resultKind(mediaType, format, blob.size);
  if (kind === 'text') {
    const text = await blob.text();
    let doc = null;
    try { doc = JSON.parse(text); } catch { /* shown as text */ }
    if (isScene(doc)) {
      const objs = sceneObjects(doc);
      content.append(objectsTable(objs, `Objects of the scene (${objs.length})`));
    }
    content.append(...fullText(text, doc));
  } else if (kind === 'image') {
    content.append(el('figure', { class: 'result-image', 'data-testid': 'result-image' },
      el('img', { src: url, alt: `The result: a ${mediaType} image of ${fmtBytes(blob.size)}` }),
      el('figcaption', { class: 'muted' }, 'The image as received; the download holds the same bytes.')));
  } else {
    const ply = await plyHeader(blob);
    if (ply || format === 'ply') {
      cloud = cloudSection(blob);
      content.append(cloud.el);
      if (ply) {
        const { summary, note } = plyHeaderText(ply, blob.size);
        content.append(el('details', { class: 'ply-header' }, el('summary', {}, summary),
          el('pre', { class: 'result-text', tabindex: '0', 'aria-label': 'The PLY header', 'data-testid': 'result-text' }, ply.header),
          el('p', { class: 'muted' }, note)));
      }
    } else {
      content.append(el('p', { class: 'muted' }, blob.size > SHOW_TEXT_MAX
        ? `A ${fmtBytes(blob.size)} result: too large to show here; download it.`
        : 'A binary result: download it.'));
    }
  }
  return { el: box, cloud, dispose: () => { cloud?.dispose(); URL.revokeObjectURL(url); } };
}
