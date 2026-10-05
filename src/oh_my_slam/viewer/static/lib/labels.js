// Labels anchored at 3D points (spec §2.5 "labelled OBBs"; also the located cameras), kept
// legible: no label ever covers another. Each item whose anchor is in view gets its tag next to the
// anchor, or on a ring farther out, larger items on screen first. An item whose tag finds no free
// place keeps no tag on screen and is returned as crowded (the page lists it, and it appears once
// the view is zoomed in). Then the names are added, in the same order, wherever the longer label
// covers nothing.
//
// An item is { id, tag, name, background, ink, anchor (THREE.Vector3, display frame), size (m),
// title, note (its text in a list of crowded items), group (which layer shows it) }; the layer
// adds `div`, and sets `mode` (null: not shown, 'compact': tag only, 'full': tag and name).
import * as THREE from 'three';
import { el } from './dom.js';

const LABEL_GAP = 2;          // px between two labels, and between a label and the view's edge
const RINGS = [22, 34, 48, 64, 84];  // px from the anchor to the label's centre
const DIRS = [0, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6].map((k) => {  // from straight up, both ways
  const a = -Math.PI / 2 + (k * Math.PI) / 6;
  return [Math.cos(a), Math.sin(a)];
});

export class LabelLayer {
  constructor(host) {
    this.root = el('div', { class: 'labels', 'aria-label': 'Object labels' });
    host.appendChild(this.root);
    this.items = [];
    this.measured = false;
  }

  add(item) {
    const tag = el('span', { class: 'tag' }, item.tag);
    tag.style.background = item.background;
    tag.style.color = item.ink;
    item.div = el('div', { class: 'obj-label', 'data-id': item.id, 'data-group': item.group,
      title: item.title }, tag, el('span', { class: 'name' }, item.name));
    Object.assign(item, { w: 0, wTag: 0, h: 0, mode: null, rect: null });
    this.root.appendChild(item.div);
    this.items.push(item);
    this.measured = false;
    return item;
  }

  remove(group) {
    for (const it of this.items.filter((i) => i.group === group)) it.div.remove();
    this.items = this.items.filter((i) => i.group !== group);
  }

  measure() {
    for (const o of this.items) {
      o.div.classList.remove('compact');
      const r = o.div.getBoundingClientRect();
      o.w = Math.ceil(r.width); o.h = Math.ceil(r.height);
      o.wTag = Math.ceil(o.div.firstChild.getBoundingClientRect().width);
    }
    this.measured = this.items.every((o) => o.w > 0);
  }

  // Place the labels of the shown groups for `camera` in a W x H view; returns the crowded items
  // (in view, no room for their tag), by id.
  layout(camera, W, H, shown) {
    const active = this.items.filter((o) => shown(o.group));
    this.root.hidden = !active.length;
    if (!active.length) {
      for (const o of this.items) { o.mode = null; o.div.hidden = true; }
      return [];
    }
    if (!this.measured) this.measure();
    const taken = [];
    const free = (x, y, bw, bh, ignore = null) => !taken.some((r) => r !== ignore
      && x - LABEL_GAP < r[2] && r[0] < x + bw + LABEL_GAP && y - LABEL_GAP < r[3] && r[1] < y + bh + LABEL_GAP);
    const clampX = (x, bw) => Math.min(Math.max(x, LABEL_GAP), W - bw - LABEL_GAP);
    const clampY = (y, bh) => Math.min(Math.max(y, LABEL_GAP), H - bh - LABEL_GAP);
    const candidates = (sx, sy, bw, bh) => {
      const out = [[sx - bw / 2, sy - bh - 3], [sx + 4, sy - bh / 2], [sx - bw - 4, sy - bh / 2],
        [sx - bw / 2, sy + 3]];
      for (const r of RINGS) for (const [dx, dy] of DIRS) out.push([sx + dx * r - bw / 2, sy + dy * r - bh / 2]);
      return out.map(([x, y]) => [clampX(x, bw), clampY(y, bh)]);
    };
    const spot = (sx, sy, bw, bh) => candidates(sx, sy, bw, bh).find(([x, y]) => free(x, y, bw, bh)) || null;
    const eye = camera.position;
    const p = new THREE.Vector3();
    const inView = [];
    for (const o of this.items) o.mode = null;
    for (const o of active) {
      p.copy(o.anchor).project(camera);
      const sx = (p.x + 1) / 2 * W, sy = (1 - p.y) / 2 * H;
      if (!(p.z < 1 && p.z > -1 && sx >= 0 && sx <= W && sy >= 0 && sy <= H)) continue;
      o.sx = sx; o.sy = sy;
      o.rank = o.size / Math.max(eye.distanceTo(o.anchor), 1e-3);
      inView.push(o);
    }
    inView.sort((a, b) => (b.rank - a.rank) || (a.id - b.id));
    const placed = [], crowded = [];
    for (const o of inView) {
      const at = spot(o.sx, o.sy, o.wTag, o.h);
      if (!at) { crowded.push(o); continue; }
      o.mode = 'compact';
      o.rect = [at[0], at[1], at[0] + o.wTag, at[1] + o.h];
      taken.push(o.rect);
      placed.push(o);
    }
    for (const o of placed) {  // names: the tag grows rightwards, else leftwards, where that is free
      const r = o.rect;
      for (const x of [r[0], r[2] - o.w]) {
        if (x >= LABEL_GAP && x + o.w <= W - LABEL_GAP && free(x, r[1], o.w, o.h, r)) {
          r[0] = x; r[2] = x + o.w; o.mode = 'full'; break;
        }
      }
    }
    for (const o of this.items) {
      const d = o.div;
      if (!o.mode) { if (!d.hidden) d.hidden = true; continue; }
      if (d.hidden) d.hidden = false;
      d.classList.toggle('compact', o.mode === 'compact');
      d.style.transform = `translate(${Math.round(o.rect[0])}px, ${Math.round(o.rect[1])}px)`;
    }
    return crowded.sort((a, b) => a.id - b.id);
  }
}

// The text listing crowded items: the first `max`, then how many more.
export function crowdedNote(crowded, max = 20) {
  if (!crowded.length) return '';
  const head = crowded.slice(0, max).map((o) => o.note).join(', ');
  const more = crowded.length > max ? ` …and ${crowded.length - max} more` : '';
  return `No room for: ${head}${more} (zoom in to show)`;
}
