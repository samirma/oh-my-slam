// A result as the response carried it (http_server.md "Image": a scene description (JSON) "is shown
// in full, with its objects listed (label, id, colour, score)"; a point cloud (PLY) "is loaded in the
// page and drawn in 3D"; "every result can be downloaded byte for byte"). The download is the
// response body itself, byte for byte (the command's own output); the text shown is that body as
// received. What is shown follows the result's format and content, never the command: a JSON
// document that is a scene description (OpenLABEL) has its objects and its cameras listed; a PLY is
// drawn in 3D by the viewer's rendering (cloudresult.js), with the cameras its header names and
// its header below; an image (PNG: a depth image, a segmented image, a map's bird's-eye image) is
// shown as received, with what its own text metadata says when it has any (a bird's-eye image's
// description, scale, layers, legend and cameras), and a link to it at full size. The warnings
// the command gave come first. An input is named as the user chose it (the file's name and its
// number among several), with the path the command was given as its tooltip.
import { el, fmtBytes, fmtSeconds, hexColor, notice } from './dom.js';
import { cloudSection } from './cloudresult.js';
import { cameraTitle, chosenFor, inputLabel, warningText } from './inputs.js';

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

// What each kind of camera of a scene is (sceneCameras), in the order the cameras are listed.
export const CAMERA_KINDS = { located: 'Located', last_known: 'Last known camera', frame: 'Map frame' };

// The cameras of a scene description: every frame with a camera pose (a map's keyframes, the
// cameras mapper.sh locate placed or its last known camera) as { frame (its key), name, kind
// ('located', 'last_known' or 'frame'), position [x, y, z], image }, located and last known
// cameras first, then by frame key.
export function sceneCameras(doc) {
  const frames = (doc && doc.openlabel && doc.openlabel.frames) || {};
  const out = [];
  for (const [key, f] of Object.entries(frames)) {
    const p = (f && f.frame_properties) || {};
    const tr = Object.values(p.transforms || {}).find((t) => t && t.transform_src_to_dst);
    if (!tr) continue;
    const kind = p.located === true ? 'located' : p.last_known === true ? 'last_known' : 'frame';
    const stream = Object.values(p.streams || {})[0];
    out.push({ frame: Number(key), name: kind === 'frame' ? (p.keyframe || `frame ${key}`) : tr.src, kind,
      position: tr.transform_src_to_dst.translation || [], image: p.image || (stream && stream.uri) || '' });
  }
  const rank = Object.keys(CAMERA_KINDS);
  return out.sort((a, b) => rank.indexOf(a.kind) - rank.indexOf(b.kind) || a.frame - b.frame);
}

// A camera's centre coordinate as shown (metres, 3 decimals).
export function coord(v) { return Number.isFinite(v) ? (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3) : '—'; }

// The cameras a bird's-eye image's metadata lists ({key: its JSON result's record}), as
// sceneCameras gives a scene's: a located one, the last known one, a keyframe's (a map frame).
export function metadataCameras(records) {
  const out = Object.entries(records && typeof records === 'object' ? records : {}).map(([name, r]) => {
    const d = r && typeof r === 'object' ? r : {};
    const kind = d.located === true ? 'located' : d.last_known === true ? 'last_known' : 'frame';
    return { frame: Number.isFinite(d.frame) ? d.frame : null, name, kind,
      position: (d.transform_src_to_dst && d.transform_src_to_dst.translation) || [], image: d.image || '' };
  });
  const rank = Object.keys(CAMERA_KINDS);
  return out.sort((a, b) => rank.indexOf(a.kind) - rank.indexOf(b.kind) || a.name.localeCompare(b.name, undefined, { numeric: true }));
}

// The warnings of the command as a list, each input named as the user chose it.
export function warningsList(warnings, chosen) {
  return el('ul', { class: 'warnings-list' }, ...warnings.map((w) => el('li', {}, warningText(w, chosen))));
}

// The warnings of the command, before its result.
function warningsNotice(warnings, chosen) {
  const n = warnings.length;
  return el('div', { 'data-testid': 'warnings' }, notice('warn', el('strong', {}, `The command gave ${n} warning${n === 1 ? '' : 's'}:`),
    warningsList(warnings, chosen)));
}

// What the table shows of a camera's image: the file the user chose (with the path as its
// tooltip), else the path.
function cameraImage(c, input) {
  return input ? el('span', { title: c.image, 'data-input': input.index }, input.name) : c.image;
}

export function camerasTable(cameras, caption, chosen = null) {
  const rows = cameras.map((c) => {
    const input = chosenFor(c.image, chosen);
    return el('tr', { 'data-camera': c.name, 'data-kind': c.kind },
      el('th', { scope: 'row' }, input ? cameraTitle(c.name, input) : el('code', {}, c.name)),
      el('td', {}, CAMERA_KINDS[c.kind]),
      ...[0, 1, 2].map((i) => el('td', { class: 'num' }, coord(c.position[i]))),
      el('td', { class: 'cam-image' }, cameraImage(c, input)));
  });
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data cameras', 'data-testid': 'cameras' },
    el('caption', {}, caption),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, 'Camera'), el('th', { scope: 'col' }, 'Kind'),
      el('th', { scope: 'col', class: 'num' }, 'x (m)'), el('th', { scope: 'col', class: 'num' }, 'y (m)'),
      el('th', { scope: 'col', class: 'num' }, 'z (m)'), el('th', { scope: 'col' }, 'Image'))),
    el('tbody', {}, ...rows)));
}

// What a scene with no objects says: "No objects.", or, for a result whose scope (its metadata's
// `scope`) is a choice of one of `op`'s parameters, which scope it is and which other one to choose
// to include them.
export function noObjectsText(doc, op) {
  const scope = doc && doc.openlabel && doc.openlabel.metadata && doc.openlabel.metadata.scope;
  const p = scope && op ? op.params.find((x) => x.choice_text && Object.hasOwn(x.choice_text, scope)) : null;
  if (!p) return 'No objects.';
  const name = (p.label || p.name).toLowerCase();
  const others = p.choices.filter((c) => c !== scope).map((c) => `${name} ${c} (${p.choice_text[c]})`);
  return `No objects in this result: its ${name} is ${scope} (${p.choice_text[scope]}).`
    + (others.length ? ` Choose ${others.join(' or ')} to include them.` : '');
}

export function objectsTable(objects, caption, empty = 'No objects.') {
  const body = objects.length
    ? objects.map((o) => el('tr', { 'data-id': o.id },
      el('th', { scope: 'row', class: 'obj-id' }, `#${o.id}`),
      el('td', {}, o.label),
      el('td', {}, hexColor(o.hex) ? el('span', { class: 'hex' }, el('span', { class: 'swatch', style: `background:${o.hex}`, 'aria-hidden': 'true' }), o.hex) : '—'),
      el('td', { class: 'num' }, o.score != null ? Number(o.score).toFixed(2) : '—')))
    : [el('tr', {}, el('td', { colspan: 4, class: 'muted', 'data-testid': 'no-objects' }, empty))];
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

// ---------------------------------------------------------------------------------- an image

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

// What a PNG says of itself (`buffer`: its bytes): { width, height, text: its tEXt and
// uncompressed iTXt chunks, keyword → text }, or null when the bytes are no PNG.
export function pngInfo(buffer) {
  const b = new Uint8Array(buffer);
  if (b.length < 8 || PNG_SIGNATURE.some((x, i) => b[i] !== x)) return null;
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const latin1 = new TextDecoder('latin1');
  const utf8 = new TextDecoder('utf-8');
  const out = { width: null, height: null, text: {} };
  for (let at = 8; at + 8 <= b.length;) {
    const len = view.getUint32(at);
    const type = latin1.decode(b.subarray(at + 4, at + 8));
    const data = b.subarray(at + 8, Math.min(at + 8 + len, b.length));
    const z = data.indexOf(0);
    if (type === 'IHDR' && data.length >= 8) {
      out.width = view.getUint32(at + 8);
      out.height = view.getUint32(at + 12);
    } else if (type === 'tEXt' && z > 0) {
      out.text[latin1.decode(data.subarray(0, z))] = latin1.decode(data.subarray(z + 1));
    } else if (type === 'iTXt' && z > 0 && data[z + 1] === 0) {  // not compressed
      const lang = data.indexOf(0, z + 3);
      const translated = lang < 0 ? -1 : data.indexOf(0, lang + 1);
      if (translated >= 0) out.text[latin1.decode(data.subarray(0, z))] = utf8.decode(data.subarray(translated + 1));
    } else if (type === 'IEND') break;
    at += 12 + len;
  }
  return out;
}

function parsed(text) {
  try { return JSON.parse(text); } catch { return null; }
}

// What an image's text metadata says that the page shows: its `Description`, its scale
// (`metres_per_pixel`), the `layers` drawn, its `legend` (the marks drawn: { mark, shape, fill,
// edge, text }) and its `cameras` (metadataCameras); each null when it has none (or none that
// reads).
export function imageFacts(text) {
  const t = text || {};
  const mpp = Number(parsed(t.metres_per_pixel));
  const legend = parsed(t.legend);
  const cameras = parsed(t.cameras);
  return {
    description: t.Description || null,
    mpp: Number.isFinite(mpp) && mpp > 0 ? mpp : null,
    layers: t.layers ? t.layers.split(',').filter(Boolean) : null,
    legend: Array.isArray(legend) ? legend.filter((m) => m && typeof m.text === 'string') : null,
    cameras: cameras && typeof cameras === 'object' && !Array.isArray(cameras) ? metadataCameras(cameras) : null,
  };
}

// A length in metres as the page says it: in centimetres below a metre.
export function lengthText(m) {
  return m < 1 ? `${Number((m * 100).toPrecision(6))} cm` : `${Number(m.toPrecision(6))} m`;
}

// The scale bar of an image `width` pixels wide at `mpp` metres per pixel: the longest 1, 2 or 5
// × 10ⁿ m within a quarter of its width, { metres, label, percent (of the image's width) }; null
// without a scale or a width.
export function scaleBar(mpp, width) {
  if (!(mpp > 0) || !(width > 0)) return null;
  const most = width * mpp / 4;
  const k = Math.floor(Math.log10(most));
  const metres = [5, 2, 1].map((f) => Number((f * 10 ** k).toPrecision(6))).find((m) => m <= most * (1 + 1e-9));
  return { metres, label: lengthText(metres), percent: (metres / mpp / width) * 100 };
}

// The text alternative of an image result: what it is (its output's title), its size, and its
// own description.
export function imageAlt(title, info, facts) {
  const what = title || 'the result image';
  const size = info && info.width ? `, ${info.width} × ${info.height} px` : '';
  return `${what.charAt(0).toUpperCase()}${what.slice(1)}, as received${size}${facts.description ? `. ${facts.description}` : ''}`;
}

const SVG = 'http://www.w3.org/2000/svg';

// A mark of a legend, as the image draws it ({shape, fill, edge}: a triangle, a diamond, a disc,
// an outline or points), 18 px on the image's background.
function markSvg(m) {
  const svg = document.createElementNS(SVG, 'svg');
  for (const [k, v] of Object.entries({ viewBox: '0 0 18 18', width: '18', height: '18', 'aria-hidden': 'true', class: 'mark' })) svg.setAttribute(k, v);
  const shape = (tag, attrs) => {
    const e = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    svg.append(e);
  };
  const fill = hexColor(m.fill) || '#ffffff';
  const edge = hexColor(m.edge) || '#000000';
  if (m.shape === 'triangle') shape('polygon', { points: '15,9 4,3.5 4,14.5', fill, stroke: edge });
  else if (m.shape === 'diamond') shape('polygon', { points: '9,2 16,9 9,16 2,9', fill, stroke: edge });
  else if (m.shape === 'disc') shape('circle', { cx: '9', cy: '9', r: '4', fill, stroke: edge });
  else if (m.shape === 'outline') shape('rect', { x: '3', y: '4', width: '12', height: '10', fill: 'none', stroke: '#e6194b', 'stroke-width': '2' });
  else for (const [x, y, c] of [[3, 4, '#8a7f70'], [8, 9, '#b9b2a5'], [12, 4, '#6b7a5c'], [5, 12, '#a59a88'], [12, 12, '#7d8a99']]) shape('rect', { x: String(x), y: String(y), width: '3', height: '3', fill: c });
  return svg;
}

// The figure of an image result: the image as received (with a scale bar when its metadata gives
// a scale), its description, size, scale and layers, a link to it at full size, and its legend.
function imageFigure(url, info, facts, title) {
  const bar = scaleBar(facts.mpp, info?.width);
  const frame = el('div', { class: 'image-frame' }, el('img', { src: url, alt: imageAlt(title, info, facts) }),
    bar ? el('div', { class: 'scale', 'data-testid': 'scale-bar' }, el('span', { class: 'scale-line', style: `width:${bar.percent.toFixed(3)}%` }),
      el('span', { class: 'scale-label' }, bar.label)) : null);
  const rows = [
    info?.width ? ['Size', `${info.width} × ${info.height} px`] : null,
    facts.mpp ? ['Scale', `1 px = ${lengthText(facts.mpp)}`] : null,
    facts.layers ? ['Layers', facts.layers.join(', ')] : null,
  ].filter(Boolean);
  const caption = el('figcaption', {},
    el('p', { class: 'muted' }, facts.description || 'The image as received; the download holds the same bytes.'),
    rows.length ? el('dl', { class: 'cloud-facts', 'data-testid': 'image-facts' }, ...rows.map(([k, v]) => el('div', {}, el('dt', {}, k), el('dd', { 'data-fact': k }, v)))) : null,
    el('p', {}, el('a', { href: url, target: '_blank', rel: 'noopener', 'data-testid': 'full-size' }, 'Open the image full size'), ' (in a new tab)'),
    facts.legend?.length ? el('div', { class: 'legend' }, el('p', { class: 'sub' }, 'Legend'),
      el('ul', { 'data-testid': 'legend' }, ...facts.legend.map((m) => el('li', { 'data-mark': m.mark || '' }, markSvg(m), ` ${m.text}`)))) : null);
  return el('figure', { class: 'result-image', 'data-testid': 'result-image' }, frame, caption);
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

// The result section: facts, download, the command's warnings, timings, then the content.
// `warnings`: the command's (X-Oms-Warnings); `chosen`: the inputs as the user chose them
// (OpForm.chosen); `op`: its operation; `title`: what the result is (its output's title).
export async function resultView({ blob, mediaType, stages, format, downloadName, warnings = [], chosen = null, op = null, title = '' }) {
  const url = URL.createObjectURL(blob);
  const box = el('section', { class: 'result', 'data-testid': 'result', 'data-format': format || '', 'aria-labelledby': 'result-h' });
  const total = stages.find((s) => s.name === 'total');
  box.append(el('h2', { id: 'result-h', tabindex: '-1' }, 'Result'),
    el('div', { class: 'result-head' },
      el('a', { class: 'button primary download', href: url, download: downloadName, 'data-testid': 'download' },
        `Download ${downloadName}`),
      el('span', { class: 'muted' }, `${mediaType}, ${fmtBytes(blob.size)}${total ? `, ${fmtSeconds(total.ms / 1000)} in the command` : ''}`)));
  if (warnings.length) box.append(warningsNotice(warnings, chosen));
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
      content.append(objectsTable(objs, `Objects of the scene (${objs.length})`, noObjectsText(doc, op)));
      const cams = sceneCameras(doc);
      if (cams.length) content.append(camerasTable(cams, `Cameras of the scene (${cams.length}): position in metres, map frame`, chosen));
    }
    content.append(...fullText(text, doc));
  } else if (kind === 'image') {
    const info = pngInfo(await blob.arrayBuffer());
    const facts = imageFacts(info?.text);
    content.append(imageFigure(url, info, facts, title));
    if (facts.cameras?.length) {
      const n = facts.cameras.length;
      content.append(el('details', { class: 'stages-box', open: n <= 12 || null }, el('summary', {}, `Cameras drawn (${n}), from the image's metadata`),
        camerasTable(facts.cameras, `Cameras drawn (${n}): position in metres, map frame`, chosen)));
    }
  } else {
    const ply = await plyHeader(blob);
    if (ply || format === 'ply') {
      cloud = cloudSection(blob, { chosen });
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
