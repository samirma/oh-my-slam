// lib/obbs.js: the objects of an OpenLABEL 1.0.0 scene (spec §3) and their oriented boxes; their
// colours are read from the scene, never computed, and reach the screen as the same sRGB triple
// (spec §2.4 colour contract).
import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { buildObb, inkFor, obbCorners, OBB_WIDTH_PX, sceneObjects, srgb } from '../../../src/oh_my_slam/viewer/static/lib/obbs.js';
import { LEVEL_UPRIGHT } from '../../../src/oh_my_slam/viewer/static/lib/cloudview.js';
import { objectEntry, PALETTE_HEX, rgbOf, UNSEGMENTED_HEX } from '../helpers.js';

describe('sceneObjects', () => {
  test('every object with its id, label, colour, box, score, volume and dimensions', () => {
    const doc = { openlabel: { metadata: {}, objects: {
      2: objectEntry(2, 'chair', [0.5, 0.2, 0.3, 0, 0, 0, 1, 0.4, 0.3, 0.6]),
      13: objectEntry(13, 'cup', null, { score: 0.5 }),
    } } };
    const [chair, cup] = sceneObjects(doc);
    expect(chair).toEqual({ id: 2, label: 'chair', hex: '#3cb44b', rgb: [60, 180, 75],
      cuboid: [0.5, 0.2, 0.3, 0, 0, 0, 1, 0.4, 0.3, 0.6], score: 0.9, volume: 0.4 * 0.3 * 0.6, dims: [0.4, 0.3, 0.6] });
    expect(cup).toEqual({ id: 13, label: 'cup', hex: '#9a6324', rgb: [154, 99, 36], cuboid: null, score: 0.5, volume: 0, dims: [0, 0, 0] });
  });

  test('missing parts read as absent: no objects, no data, no attributes', () => {
    expect(sceneObjects({})).toEqual([]);
    expect(sceneObjects({ openlabel: {} })).toEqual([]);
    const [o] = sceneObjects({ openlabel: { objects: { 1: { type: 'box', object_data: { cuboid: [{ val: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1] }] } } } } });
    expect(o).toEqual({ id: 1, label: 'box', hex: null, rgb: null, cuboid: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1], score: null, volume: 0, dims: [0, 0, 0] });
    expect(sceneObjects({ openlabel: { objects: { 5: { type: 'x' } } } })[0]).toMatchObject({ id: 5, hex: null, cuboid: null });
  });
});

test('colours reach the screen as the scene\'s sRGB triple (renderer output in sRGB, no tone mapping)', () => {
  for (const hex of [...PALETTE_HEX, UNSEGMENTED_HEX, '#000000', '#ffffff']) {
    const c = srgb(hex);
    expect(`#${c.getHexString(THREE.SRGBColorSpace)}`).toBe(hex);
    // three.js keeps it linear for lighting; the sRGB output converts it back exactly
    const u = rgbOf(hex)[0] / 255;
    expect(c.r).toBeCloseTo(u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4, 6);
  }
});

describe('inkFor: the tag text that contrasts more with the object colour', () => {
  const lum = (rgb) => {
    const lin = rgb.map((v) => { const u = v / 255; return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  };
  test('black on light colours, white on dark ones', () => {
    expect(inkFor([255, 255, 255])).toBe('#000');
    expect(inkFor([0, 0, 0])).toBe('#fff');
    expect(inkFor([255, 225, 25])).toBe('#000');  // palette yellow
    expect(inkFor([67, 99, 216])).toBe('#fff');   // palette blue
    expect(inkFor([128, 128, 128])).toBe('#000');
  });
  test('for every palette colour, the WCAG contrast of the chosen ink is the larger one', () => {
    for (const hex of PALETTE_HEX) {
      const y = lum(rgbOf(hex));
      const black = (y + 0.05) / 0.05, white = 1.05 / (y + 0.05);
      expect(inkFor(rgbOf(hex))).toBe(black >= white ? '#000' : '#fff');
      expect(Math.max(black, white)).toBeGreaterThanOrEqual(4.5);  // WCAG AA for the tag's text
    }
  });
});

describe('a cuboid: x, y, z, qx, qy, qz, qw, sx, sy, sz (quaternion scalar last)', () => {
  const sorted = (pts) => pts.map((p) => p.toArray().map((v) => +v.toFixed(9) + 0)).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);

  test('its 8 corners: the centre plus or minus half of each size, rotated by the quaternion', () => {
    expect(sorted(obbCorners([1, 2, 3, 0, 0, 0, 1, 2, 4, 6]))).toEqual(sorted(
      [[0, 0, 0], [0, 0, 6], [0, 4, 0], [0, 4, 6], [2, 0, 0], [2, 0, 6], [2, 4, 0], [2, 4, 6]].map((p) => new THREE.Vector3(...p))));
    const s = Math.SQRT1_2;  // 90° about z: the x size lies along y
    expect(sorted(obbCorners([0, 0, 0, 0, 0, s, s, 2, 4, 6]))).toEqual(sorted(
      [-1, 1].flatMap((y) => [-2, 2].flatMap((x) => [-3, 3].map((z) => new THREE.Vector3(x, y, z))))));
  });

  test('its box: 12 edges in the object\'s colour, anchored at the top face\'s centre', () => {
    const obj = { id: 7, hex: '#f58231', cuboid: [0, 0, 1, 0, 0, 0, 1, 2, 4, 6] };
    const { line, anchor, size } = buildObb(obj, new THREE.Matrix4());
    expect(line.userData.id).toBe(7);
    expect(`#${line.material.color.getHexString(THREE.SRGBColorSpace)}`).toBe('#f58231');
    expect(line.material.linewidth).toBe(OBB_WIDTH_PX);
    expect(line.material.worldUnits).toBe(false);  // a width in pixels at any distance
    const start = line.geometry.attributes.instanceStart, end = line.geometry.attributes.instanceEnd;
    expect(start.count).toBe(12);
    const lengths = [];
    for (let i = 0; i < 12; i++) {
      lengths.push(new THREE.Vector3().fromBufferAttribute(start, i).distanceTo(new THREE.Vector3().fromBufferAttribute(end, i)));
    }
    expect(lengths.sort((a, b) => a - b).map((l) => +l.toFixed(6))).toEqual([2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6]);
    expect(anchor.toArray()).toEqual([0, 0, 4]);  // the top face: z = 1 + 6 / 2
    expect(size).toBeCloseTo(Math.cbrt(48), 9);
  });

  test('the top is the top of the display frame; a degenerate box still has a size', () => {
    // an image's camera frame (y down) shown upright: the camera's -y is up
    const upright = new THREE.Matrix4().fromArray(LEVEL_UPRIGHT.flat()).transpose();
    const { anchor, size } = buildObb({ id: 1, hex: '#e6194b', cuboid: [0, 0, 5, 0, 0, 0, 1, 1, 2, 1] }, upright);
    expect(anchor.toArray().map((v) => +v.toFixed(9) + 0)).toEqual([0, 5, 1]);  // y = -1 in the camera frame
    expect(buildObb({ id: 1, hex: '#e6194b', cuboid: [0, 0, 0, 0, 0, 0, 1, 1, 0, 1] }, upright).size).toBeCloseTo(1e-3, 12);
    expect(size).toBeCloseTo(Math.cbrt(2), 9);
  });
});
