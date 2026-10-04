// Objects and their oriented boxes from an OpenLABEL 1.0.0 scene document (spec §3): each
// object's `cuboid` (x, y, z, qx, qy, qz, qw, sx, sy, sz; quaternion scalar last), its colour
// (`color` vec, `color_hex` text: the §2.4 colour contract, read, never computed) and label.
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

export const OBB_WIDTH_PX = 2.0;

export function srgb(hex) { return new THREE.Color().setStyle(hex, THREE.SRGBColorSpace); }

function prop(o, kind, name) { const v = ((o.object_data || {})[kind] || []).find((n) => n.name === name); return v ? v.val : null; }
function attr(cub, name) { const v = ((cub.attributes || {}).num || []).find((n) => n.name === name); return v ? v.val : 0; }

// Text on a tag of this background: black or white, whichever contrasts more (WCAG luminance).
export function inkFor(rgb) {
  const lin = rgb.map((v) => { const u = v / 255; return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4; });
  const y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  return (y + 0.05) / 0.05 >= 1.05 / (y + 0.05) ? '#000' : '#fff';
}

// Every object of the scene: { id, label, hex, rgb, cuboid (10 values or null), score, volume,
// dims: [width, depth, height] }.
export function sceneObjects(doc) {
  const objs = (doc.openlabel || {}).objects || {};
  return Object.entries(objs).map(([id, o]) => {
    const cub = ((o.object_data || {}).cuboid || [])[0] || null;
    return {
      id: Number(id), label: o.type, hex: prop(o, 'text', 'color_hex'), rgb: prop(o, 'vec', 'color'),
      cuboid: cub ? cub.val : null, score: prop(o, 'num', 'score'),
      volume: cub ? attr(cub, 'volume_m3') : 0,
      dims: cub ? [attr(cub, 'width_m'), attr(cub, 'depth_m'), attr(cub, 'height_m')] : [0, 0, 0],
    };
  });
}

// The 8 corners of a cuboid, in its coordinate system.
export function obbCorners(c) {
  const [x, y, z, qx, qy, qz, qw, sx, sy, sz] = c;
  const q = new THREE.Quaternion(qx, qy, qz, qw);
  const pts = [];
  for (const dx of [-0.5, 0.5]) for (const dy of [-0.5, 0.5]) for (const dz of [-0.5, 0.5]) {
    pts.push(new THREE.Vector3(dx * sx, dy * sy, dz * sz).applyQuaternion(q).add(new THREE.Vector3(x, y, z)));
  }
  return pts;
}
const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

// The box of an object with a cuboid: { line (its 12 edges in the object's colour), anchor (the
// top face's centre, in the display frame `matrix` maps to), size (cube root of its volume) }.
export function buildObb(obj, matrix) {
  const corners = obbCorners(obj.cuboid);
  const pos = [];
  for (const [a, b] of EDGES) pos.push(...corners[a].toArray(), ...corners[b].toArray());
  const line = new LineSegments2(new LineSegmentsGeometry().setPositions(pos),
    new LineMaterial({ color: srgb(obj.hex), linewidth: OBB_WIDTH_PX, worldUnits: false }));
  line.userData.id = obj.id;
  const display = corners.map((p) => p.clone().applyMatrix4(matrix));
  display.sort((a, b) => b.z - a.z);
  const anchor = display.slice(0, 4).reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(0.25);
  const [, , , , , , , sx, sy, sz] = obj.cuboid;
  return { line, anchor, size: Math.cbrt(Math.max(sx * sy * sz, 1e-9)) };
}
