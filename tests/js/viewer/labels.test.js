// lib/labels.js: the labels of the boxes (spec §2.5 "labelled OBBs") stay legible: no label ever
// covers another; larger items on screen are placed first, and the ones with no room are listed.
import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { crowdedNote, LABEL_GAP, placeLabels } from '../../../src/oh_my_slam/viewer/static/lib/labels.js';

function view(W, H) {
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 100);
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  // the 3D point drawn at pixel (sx, sy)
  const at = (sx, sy) => new THREE.Vector3(sx / W * 2 - 1, 1 - sy / H * 2, 0.5).unproject(camera);
  return { camera, at, W, H };
}
// a label as LabelLayer measures it: its width with the name (w), the tag's (wTag), its height
const item = (id, anchor, { size = 1, w = 80, wTag = 20, h = 16 } = {}) => ({ id, anchor, size, w, wTag, h, mode: null, note: `${id} thing` });
const near = (rect, expected) => rect.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 6));

function disjoint(a, b) {
  return a[2] + LABEL_GAP <= b[0] || b[2] + LABEL_GAP <= a[0] || a[3] + LABEL_GAP <= b[1] || b[3] + LABEL_GAP <= a[1];
}

describe('placeLabels', () => {
  test('a label in view: its tag above its anchor, its name to the right', () => {
    const v = view(800, 600);
    const o = item(1, v.at(400, 300));
    expect(placeLabels([o], v.camera, v.W, v.H)).toEqual([]);
    expect(o.sx).toBeCloseTo(400, 6);
    expect(o.sy).toBeCloseTo(300, 6);
    expect(o.mode).toBe('full');
    near(o.rect, [390, 281, 470, 297]);
  });

  test('an anchor out of view, or behind the camera, gets no label and is not crowded', () => {
    const v = view(800, 600);
    const side = item(1, new THREE.Vector3(100, 0, 0));
    const behind = item(2, new THREE.Vector3(0, 0, 20));
    expect(placeLabels([side, behind], v.camera, v.W, v.H)).toEqual([]);
    for (const o of [side, behind]) {
      expect(o.mode).toBeNull();
      expect(o.sx).toBeUndefined();
    }
  });

  test('near the right edge the name grows leftwards', () => {
    const v = view(800, 600);
    const o = item(1, v.at(780, 300));
    placeLabels([o], v.camera, v.W, v.H);
    expect(o.mode).toBe('full');
    near(o.rect, [710, 281, 790, 297]);
  });

  test('a name goes where it covers no tag: away from a neighbour\'s', () => {
    const v = view(800, 600);
    const big = item(1, v.at(400, 300), { size: 2 }), small = item(2, v.at(430, 300));
    placeLabels([small, big], v.camera, v.W, v.H);
    near(big.rect, [330, 281, 410, 297]);  // leftwards: the small one's tag is on its right
    near(small.rect, [420, 281, 500, 297]);
    expect([big.mode, small.mode]).toEqual(['full', 'full']);
  });

  test('larger items on screen first; at equal size, the lower id; the others are crowded', () => {
    const v = view(60, 40);  // room for one 40 x 20 tag, and for no name
    const label = (o) => ({ id: o.id, mode: o.mode });
    const small = item(1, v.at(30, 20), { wTag: 40, h: 20 }), large = item(2, v.at(30, 20), { size: 3, wTag: 40, h: 20 });
    expect(placeLabels([small, large], v.camera, v.W, v.H).map((o) => o.id)).toEqual([1]);
    expect([label(small), label(large)]).toEqual([{ id: 1, mode: null }, { id: 2, mode: 'compact' }]);
    const a = item(5, v.at(30, 20), { wTag: 40, h: 20 }), b = item(4, v.at(30, 20), { wTag: 40, h: 20 });
    expect(placeLabels([a, b], v.camera, v.W, v.H)).toEqual([a]);
    expect(b.mode).toBe('compact');
  });

  test('many labels at one place: every tag placed covers no other label, inside the view', () => {
    const v = view(800, 600);
    const items = Array.from({ length: 60 }, (_, i) => item(i + 1, v.at(400 + (i % 5), 300 - (i % 3)), { size: 1 + (i % 7) }));
    const crowded = placeLabels(items, v.camera, v.W, v.H);
    const placed = items.filter((o) => o.mode);
    expect(placed.length).toBeGreaterThan(10);
    expect(crowded.length).toBe(items.length - placed.length);
    expect(crowded.length).toBeGreaterThan(0);
    expect(crowded.map((o) => o.id)).toEqual([...crowded].map((o) => o.id).sort((x, y) => x - y));
    for (const o of crowded) expect(o.mode).toBeNull();
    for (const [i, a] of placed.entries()) {
      expect(a.rect[0]).toBeGreaterThanOrEqual(LABEL_GAP);
      expect(a.rect[1]).toBeGreaterThanOrEqual(LABEL_GAP);
      expect(a.rect[2]).toBeLessThanOrEqual(v.W - LABEL_GAP);
      expect(a.rect[3]).toBeLessThanOrEqual(v.H - LABEL_GAP);
      expect(a.rect[2] - a.rect[0]).toBeCloseTo(a.mode === 'full' ? a.w : a.wTag, 9);
      for (const b of placed.slice(i + 1)) expect(disjoint(a.rect, b.rect)).toBe(true);
    }
  });
});

test('crowdedNote lists the labels with no room, the first 20, then how many more', () => {
  expect(crowdedNote([])).toBe('');
  const items = Array.from({ length: 23 }, (_, i) => ({ note: `${i + 1} chair` }));
  expect(crowdedNote(items.slice(0, 2))).toBe('No room for: 1 chair, 2 chair (zoom in to show)');
  expect(crowdedNote(items)).toBe(`No room for: ${items.slice(0, 20).map((o) => o.note).join(', ')} …and 3 more (zoom in to show)`);
  expect(crowdedNote(items, 22)).toEndWith('22 chair …and 1 more (zoom in to show)');
});
