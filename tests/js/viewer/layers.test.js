// lib/layers.js and lib/viewer.js: the independent layers of spec §2.5 (point cloud, segmentation,
// camera poses, labels, oriented boxes), each drawn by a group of its own; and the viewpoint change
// that redraws the on-demand view.
import { expect, test } from 'bun:test';
import * as THREE from 'three';
import { LAYERS } from '../../../src/oh_my_slam/viewer/static/lib/layers.js';
import { differs, GROUPS } from '../../../src/oh_my_slam/viewer/static/lib/viewer.js';

test('one toggle per layer, each with a name and what it shows', () => {
  expect(LAYERS.map(([key]) => key)).toEqual(['points', 'segments', 'cameras', 'labels', 'obbs']);
  expect(LAYERS.map(([, name]) => name)).toEqual(['Point cloud', 'Segmentation', 'Camera poses', 'Labels', 'Oriented boxes']);
  for (const [, , help] of LAYERS) expect(help.length).toBeGreaterThan(10);
});

test('every layer toggles a group of the view of its own', () => {
  expect([...GROUPS].sort()).toEqual(LAYERS.map(([key]) => key).sort());
});

test('a viewpoint change beyond 1 µm or 1 µrad redraws; one below it does not', () => {
  const a = new THREE.Matrix4().makeRotationZ(0.3).setPosition(1, 2, 3);
  expect(differs(a, a.clone())).toBe(false);
  for (const i of [0, 7, 12, 15]) {
    const b = a.clone();
    b.elements[i] += 5e-7;
    expect(differs(a, b)).toBe(false);
    b.elements[i] += 1e-6;
    expect(differs(a, b)).toBe(true);
  }
});
