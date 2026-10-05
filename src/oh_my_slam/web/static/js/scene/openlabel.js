// Whether a document is a scene description of the commands (spec §3): it validates against the
// vendored ASAM OpenLABEL 1.0.0 schema (draft-07, served at /static/openlabel_json_schema.json) and
// passes the checks the schema does not express, as schema/validate.py does them: metadata
// schema_version / schema_url, stream intrinsics, the top-level frame_intervals, transform
// endpoints and unit quaternions, object coordinate systems, and the 10-value cuboid.
import { validate } from './jsonschema.js';

export const SCHEMA_VERSION = '1.0.0';
export const SCHEMA_URL = 'https://openlabel.asam.net/V1-0-0/schema/openlabel_json_schema.json';
const QUAT_TOL = 1e-3;

let schemaPromise = null;
export function loadSchema() {
  if (!schemaPromise) {
    schemaPromise = fetch('/static/openlabel_json_schema.json').then((r) => {
      if (!r.ok) throw new Error(`the scene schema could not be loaded (HTTP ${r.status})`);
      return r.json();
    });
  }
  return schemaPromise;
}

const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
// Python's isinstance(v, int), which a bool passes (bool is an int there); its value as a number
const isInt = (v) => Number.isInteger(v) || typeof v === 'boolean';

function checkQuat(q, where, errors) {
  if (!(Array.isArray(q) && q.length === 4 && q.every(isNum))) { errors.push(`${where}: quaternion must be 4 numbers`); return; }
  const n = Math.hypot(...q);
  if (Math.abs(n - 1) > QUAT_TOL) errors.push(`${where}: quaternion norm ${n.toFixed(6)} is not 1`);
}

function checkIntrinsics(name, stream, errors) {
  const pin = (stream.stream_properties || {}).intrinsics_pinhole;
  if (stream.type === 'camera' && pin == null) { errors.push(`streams/${name}: camera stream without intrinsics_pinhole`); return; }
  if (pin == null) return;
  const w = pin.width_px, h = pin.height_px;
  if (!(isInt(w) && isInt(h) && w > 0 && h > 0)) errors.push(`streams/${name}: width_px/height_px must be positive integers`);
  const cm = pin.camera_matrix;
  if (!(Array.isArray(cm) && cm.length === 12 && cm.every(isNum))) errors.push(`streams/${name}: camera_matrix must be 12 numbers (3x4)`);
  else {
    if (cm[0] <= 0 || cm[5] <= 0) errors.push(`streams/${name}: focal lengths must be positive`);
    if (!(cm[8] === 0 && cm[9] === 0 && cm[10] === 1 && cm[11] === 0)) errors.push(`streams/${name}: camera_matrix last row must be [0, 0, 1, 0]`);
  }
  const dist = pin.distortion_coeffs;
  if (dist != null && !(Array.isArray(dist) && [4, 5, 8, 12, 14].includes(dist.length) && dist.every(isNum))) {
    errors.push(`streams/${name}: distortion_coeffs must be a list of numbers`);
  }
}

// The checks of schema/validate.py extra_errors.
export function extraErrors(doc) {
  const errors = [];
  const ol = isObj(doc) && isObj(doc.openlabel) ? doc.openlabel : {};
  const md = ol.metadata || {};
  if (md.schema_version !== SCHEMA_VERSION) errors.push('metadata.schema_version must be 1.0.0');
  if (md.schema_url !== SCHEMA_URL) errors.push('metadata.schema_url must be the canonical OpenLABEL schema URL');
  for (const [name, stream] of Object.entries(ol.streams || {})) checkIntrinsics(name, stream || {}, errors);
  const frames = ol.frames;
  const intervals = ol.frame_intervals;
  if (intervals !== undefined && intervals !== null) {
    if (!Array.isArray(intervals)) errors.push('frame_intervals must be a list');
    else {
      // The intervals must cover exactly the frame keys. They are never enumerated beyond the
      // number of frames: a range longer than that cannot match, whatever its size (a document
      // claiming billions of frames is refused at once, not counted).
      const keys = frames != null ? new Set(Object.keys(frames).map(Number)) : null;
      const covered = new Set();
      let tooMany = false;
      intervals.forEach((fi, i) => {
        let s = (fi || {}).frame_start, e = (fi || {}).frame_end;
        if (!(isInt(s) && isInt(e) && s <= e)) { errors.push(`frame_intervals[${i}]: needs integer frame_start <= frame_end`); return; }
        s = Number(s); e = Number(e);
        if (!keys || tooMany) return;
        if (e - s + 1 > keys.size) { tooMany = true; return; }
        for (let k = s; k <= e; k++) covered.add(k);
        if (covered.size > keys.size) tooMany = true;
      });
      if (keys && (tooMany || keys.size !== covered.size || [...keys].some((k) => !covered.has(k)))) {
        errors.push('frame_intervals do not match the frame keys');
      }
    }
  } else if (frames && Object.keys(frames).length) {
    errors.push('frames present without frame_intervals');
  }
  const css = ol.coordinate_systems || {};
  const haveCss = Object.keys(css).length > 0;
  for (const [key, fr] of Object.entries(frames || {})) {
    for (const [tname, tr] of Object.entries(((fr || {}).frame_properties || {}).transforms || {})) {
      const data = (tr || {}).transform_src_to_dst || {};
      if (isObj(data) && 'quaternion' in data) checkQuat(data.quaternion, `frames/${key}/transforms/${tname}`, errors);
      for (const end of ['src', 'dst']) {
        if (haveCss && !((tr || {})[end] in css)) errors.push(`frames/${key}/transforms/${tname}: unknown ${end} ${(tr || {})[end]}`);
      }
    }
  }
  for (const [oid, obj] of Object.entries(ol.objects || {})) {
    const cs = (obj || {}).coordinate_system;
    if (cs != null && haveCss && !(cs in css)) errors.push(`objects/${oid}: unknown coordinate system ${cs}`);
    ((((obj || {}).object_data) || {}).cuboid || []).forEach((cub, j) => {
      const val = (cub || {}).val;
      const where = `objects/${oid}/cuboid[${j}]`;
      if (!(Array.isArray(val) && val.length === 10 && val.every(isNum))) { errors.push(`${where}: val must be 10 numbers (x,y,z,qx,qy,qz,qw,sx,sy,sz)`); return; }
      checkQuat(val.slice(3, 7), where, errors);
      if (val.slice(7).some((v) => v < 0)) errors.push(`${where}: negative size`);
    });
  }
  return errors;
}

// The problems of `doc`: { schema: [{path, message}], extra: [text] }; `extra` is null when the
// checks beyond the schema cannot read the document (its shape is wrong: the schema says why).
export async function sceneProblems(doc) {
  const schema = validate(await loadSchema(), doc);
  let extra;
  try { extra = extraErrors(doc); } catch { extra = null; }
  return { schema, extra };
}

// Every problem of `doc` as text (empty: a valid scene description).
export async function sceneErrors(doc) {
  const p = await sceneProblems(doc);
  const extra = p.extra ?? (p.schema.length ? [] : ['the document cannot be read as a scene description']);
  return [...p.schema.map((e) => `schema: ${e.path || '<root>'}: ${e.message}`), ...extra];
}
