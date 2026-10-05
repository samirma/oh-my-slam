// Picking a box in a Viewer (lib/viewer.js) with a click: the object whose oriented box, projected
// on screen, contains the point. Works on a Viewer of this
// page or of an embedded viewer page (same origin): only plain matrix elements are read from it.
// The smallest box on screen wins where several contain the point (a plate on a table).
import * as THREE from 'three';
import { obbCorners } from '/static/viewer/lib/obbs.js';

const CLICK_PX = 5;  // a press that moves further is an orbit, not a click

function projectBox(o, root, view, proj) {
  const pts = obbCorners(o.cuboid).map((p) => p.applyMatrix4(root).applyMatrix4(view));
  if (pts.some((p) => p.z > -1e-3)) return null;  // behind or at the eye
  const s = pts.map((p) => p.applyMatrix4(proj));
  return { minX: Math.min(...s.map((p) => p.x)), maxX: Math.max(...s.map((p) => p.x)),
    minY: Math.min(...s.map((p) => p.y)), maxY: Math.max(...s.map((p) => p.y)) };
}

// the id of the box under (x, y) in CSS pixels of the viewer's host, or null
export function pickObject(viewer, x, y) {
  const host = viewer.host;
  const W = host.clientWidth, H = host.clientHeight;
  const nx = (x / W) * 2 - 1, ny = 1 - (y / H) * 2;
  const root = new THREE.Matrix4().fromArray(viewer.root.matrixWorld.elements);
  const view = new THREE.Matrix4().fromArray(viewer.camera.matrixWorldInverse.elements);
  const proj = new THREE.Matrix4().fromArray(viewer.camera.projectionMatrix.elements);
  if (!viewer.layers.obbs) return null;
  let best = null;
  for (const o of viewer.objects) {
    if (!o.cuboid) continue;
    const b = projectBox(o, root, view, proj);
    if (!b || nx < b.minX || nx > b.maxX || ny < b.minY || ny > b.maxY) continue;
    const area = (b.maxX - b.minX) * (b.maxY - b.minY);
    if (!best || area < best.area) best = { id: o.id, area };
  }
  return best ? best.id : null;
}

// Clicks (not drags) on the viewer's canvas select the box under them (or none).
export function clickToSelect(viewer, onPick) {
  const canvas = viewer.renderer.domElement;
  let down = null;
  canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > CLICK_PX) { down = null; return; }
    down = null;
    const r = canvas.getBoundingClientRect();
    onPick(pickObject(viewer, e.clientX - r.left, e.clientY - r.top));
  });
}
