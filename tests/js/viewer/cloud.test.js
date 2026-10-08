// lib/cloud.js: the point-cloud layer in the colours the cloud carries, and the segmentation layer
// (the segmented points of the same subset in their objects' colours, spec §2.5); color=segment
// colours reach the screen exactly (spec §2.4 colour contract).
import { describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { buildCloud, isExact, pointMaterial, POINT_SIZE_CM, robustBox } from '../../../src/oh_my_slam/viewer/static/lib/cloud.js';
import { PALETTE_HEX, rgbOf } from '../helpers.js';

test('a cloud is exact (no shading) only with color=segment', () => {
  expect(isExact({ attrs: 'color=segment,voxel=0,normals=on' })).toBe(true);
  expect(isExact({ attrs: 'normals=on,color=segment' })).toBe(true);
  expect(isExact({ attrs: 'color=rgb,normals=on' })).toBe(false);
  expect(isExact({ attrs: 'color=segmentation' })).toBe(false);
  expect(isExact({ attrs: '' })).toBe(false);
  expect(isExact({})).toBe(false);
});

describe('pointMaterial', () => {
  test('the attributes it draws, its point size and its focal length', () => {
    const m = pointMaterial({ color: true, normal: true, exact: false, focal: 812 });
    expect(m).toBeInstanceOf(THREE.ShaderMaterial);
    expect(m.defines).toEqual({ HAS_COLOR: '', HAS_NORMAL: '' });
    expect(m.uniforms.size.value).toBe(POINT_SIZE_CM);
    expect(m.uniforms.focal.value).toBe(812);
    expect(m.uniforms.shade.value).toBe(1);  // shaded by its normals
    expect(pointMaterial({ focal: 1 }).defines).toEqual({});
  });

  test('exact colours are written as sent: no shading and no colour-space conversion', () => {
    const m = pointMaterial({ color: true, normal: true, exact: true, focal: 1 });
    expect(m.uniforms.shade.value).toBe(0);
    expect(m.fragmentShader).toContain('gl_FragColor = vec4(vColor, 1.0);');
    expect(m.fragmentShader).not.toContain('colorspace_fragment');
  });
});

describe('buildCloud', () => {
  const n = 6;
  const position = Float32Array.from({ length: 3 * n }, (_, i) => i / 10);
  const color = Uint8Array.from({ length: 3 * n }, (_, i) => 200 - i);
  const normal = Float32Array.from({ length: 3 * n }, (_, i) => (i % 3 === 2 ? 1 : 0));
  // ids: unsegmented (0), objects 1, 2 and 9 (9 is in no scene: no colour)
  const label = new Int32Array([0, 1, 2, 0, 9, 1]);
  const objectRgb = new Map([[1, rgbOf(PALETTE_HEX[0])], [2, rgbOf(PALETTE_HEX[1])]]);

  test('the point-cloud layer draws the arrays as sent', () => {
    const { points, segments } = buildCloud({ header: { count: n, attrs: 'color=rgb,normals=on' }, arrays: { position, color, normal } }, objectRgb, 900);
    expect(segments).toBeNull();  // no object ids: no segmentation layer
    expect(points).toBeInstanceOf(THREE.Points);
    expect(points.name).toBe('points');
    expect(points.frustumCulled).toBe(false);
    const g = points.geometry.attributes;
    expect(g.position.array).toBe(position);  // no copy
    expect(g.rgb.array).toBe(color);
    expect(g.rgb.normalized).toBe(true);  // sRGB bytes as 0..1
    expect(g.normal.array).toBe(normal);
    expect(points.material.defines).toEqual({ HAS_COLOR: '', HAS_NORMAL: '' });
    expect(points.material.uniforms.shade.value).toBe(1);
    expect(points.material.uniforms.focal.value).toBe(900);
  });

  test('a cloud without colours or normals draws neither', () => {
    const { points } = buildCloud({ header: { count: n, attrs: 'color=none' }, arrays: { position } }, objectRgb, 1);
    expect(Object.keys(points.geometry.attributes)).toEqual(['position']);
    expect(points.material.defines).toEqual({});
  });

  test('the segmentation layer: the same points, only the segmented ones, in their objects\' colours', () => {
    const { points, segments } = buildCloud({ header: { count: n, attrs: 'color=segment' }, arrays: { position, color, label } }, objectRgb, 1);
    expect(points.material.uniforms.shade.value).toBe(0);  // color=segment: exact
    expect(segments.name).toBe('segments');
    expect(segments.geometry.attributes.position).toBe(points.geometry.attributes.position);  // the same subset
    expect([...segments.geometry.index.array]).toEqual([1, 2, 5]);  // id 0 and the id with no colour are left out
    const rgb = segments.geometry.attributes.rgb.array;
    for (const i of [1, 2, 5]) expect([...rgb.slice(3 * i, 3 * i + 3)]).toEqual(objectRgb.get(label[i]));
    expect(segments.material.uniforms.shade.value).toBe(0);
    expect(segments.material.defines).toEqual({ HAS_COLOR: '' });
    expect(segments.renderOrder).toBe(1);  // drawn over the cloud
    expect(segments.frustumCulled).toBe(false);
  });
});

describe('robustBox: the 2nd-98th percentile bounds of the positions in the display frame', () => {
  test('far outliers do not widen it', () => {
    const pts = [];
    for (let i = 0; i < 100; i++) pts.push(i, 2 * i, -i);
    pts.push(1e6, -1e6, 1e6);  // a window, the sky
    const box = robustBox(new Float32Array(pts), new THREE.Matrix4());
    expect(box.min.toArray()).toEqual([2, 2, -97]);
    expect(box.max.toArray()).toEqual([98, 194, -1]);
  });

  test('it is taken in the frame the matrix maps to, without non-finite points', () => {
    const pts = [];
    for (let i = 0; i < 100; i++) pts.push(i, 0, 0);
    pts.push(NaN, 0, 0, Infinity, 1, 1);
    const box = robustBox(new Float32Array(pts), new THREE.Matrix4().makeTranslation(10, 20, 30));
    expect(box.min.toArray()).toEqual([12, 20, 30]);
    expect(box.max.toArray()).toEqual([108, 20, 30]);
    expect(robustBox(new Float32Array([NaN, 0, 0, 1, Infinity, 0]), new THREE.Matrix4()).isEmpty()).toBe(true);
    expect(robustBox(new Float32Array(0), new THREE.Matrix4()).isEmpty()).toBe(true);
  });

  test('a large cloud is sampled', () => {
    const n = 200_000;
    const pts = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) pts[3 * i] = i;
    const box = robustBox(pts, new THREE.Matrix4());
    expect(box.min.x).toBeCloseTo(0.02 * n, -2);
    expect(box.max.x).toBeCloseTo(0.98 * n, -2);
  });
});
