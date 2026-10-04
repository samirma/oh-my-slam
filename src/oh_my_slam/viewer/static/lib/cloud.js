// Drawing a cloud (data.js structure): the point-cloud layer in the colours it carries and the
// segmentation layer, the segmented points of the same subset in their objects' colours.
import * as THREE from 'three';

export const POINT_SIZE_CM = 2.0;

// Points are drawn in their sRGB bytes as they are; with normals they are shaded by them
// (headlight), so that the attribute shows, except when `exact` (color=segment), whose object
// colours and unsegmented grey must reach the screen exactly (§2.4 colour contract).
export function pointMaterial({ color, normal, exact, focal }) {
  const defines = {};
  if (color) defines.HAS_COLOR = '';
  if (normal) defines.HAS_NORMAL = '';
  return new THREE.ShaderMaterial({
    defines,
    uniforms: {
      size: { value: POINT_SIZE_CM },               // point diameter in centimetres
      shade: { value: exact ? 0 : 1 },              // 0: colours exactly as sent
      focal: { value: focal },                      // viewport focal length in pixels
      base: { value: new THREE.Vector3(0.80, 0.82, 0.86) },  // no colour
    },
    vertexShader: `
      #ifdef HAS_COLOR
      attribute vec3 rgb;
      #endif
      uniform float size; uniform float focal; uniform int shade; uniform vec3 base;
      varying vec3 vColor;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        #ifdef HAS_COLOR
        vec3 c = rgb;
        #else
        vec3 c = base;
        #endif
        #ifdef HAS_NORMAL
        if (shade == 1) {
          vec3 n = normalize(normalMatrix * normal);
          c *= 0.3 + 0.7 * abs(dot(n, normalize(-mv.xyz)));
        }
        #endif
        vColor = c;
        gl_PointSize = clamp(size * 0.01 * focal / max(-mv.z, 0.05), 1.0, 24.0);
        gl_Position = projectionMatrix * mv;
      }`,
    // No colour-space conversion: the bytes are sRGB already and are written as they are.
    fragmentShader: `
      varying vec3 vColor;
      void main() { gl_FragColor = vec4(vColor, 1.0); }`,
  });
}

export function isExact(header) { return (header.attrs || '').split(',').includes('color=segment'); }

// { points, segments }: THREE.Points of the cloud and, when it carries object ids, of its
// segmented points (id != 0 with a colour in `objectRgb`: id -> [r, g, b], the scene JSON's
// colours), drawn over the cloud at the same positions.
export function buildCloud({ header, arrays }, objectRgb, focal) {
  const n = header.count;
  const position = new THREE.BufferAttribute(arrays.position, 3);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', position);
  if (arrays.color) g.setAttribute('rgb', new THREE.BufferAttribute(arrays.color, 3, true));
  if (arrays.normal) g.setAttribute('normal', new THREE.BufferAttribute(arrays.normal, 3));
  const points = new THREE.Points(g, pointMaterial({
    color: !!arrays.color, normal: !!arrays.normal, exact: isExact(header), focal }));
  points.name = 'points';
  points.frustumCulled = false;
  let segments = null;
  if (arrays.label) {
    const idx = new Uint32Array(n);
    const col = new Uint8Array(n * 3);
    let m = 0;
    for (let i = 0; i < n; i++) {
      const rgb = arrays.label[i] > 0 ? objectRgb.get(arrays.label[i]) : undefined;
      if (rgb) { idx[m++] = i; col[3 * i] = rgb[0]; col[3 * i + 1] = rgb[1]; col[3 * i + 2] = rgb[2]; }
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', position);
    sg.setAttribute('rgb', new THREE.BufferAttribute(col, 3, true));
    sg.setIndex(new THREE.BufferAttribute(idx.slice(0, m), 1));
    segments = new THREE.Points(sg, pointMaterial({ color: true, exact: true, focal }));
    segments.name = 'segments';
    segments.frustumCulled = false;
    segments.renderOrder = 1;  // after the cloud: same positions, depth test passes (less-or-equal)
  }
  return { points, segments };
}

// 2nd-98th percentile bounds of a sample of the positions, in the frame `matrix` maps them to: far
// outliers (windows, sky) do not shrink the view.
export function robustBox(positions, matrix) {
  const n = positions.length / 3;
  const step = Math.max(1, Math.floor(n / 50000));
  const xs = [], ys = [], zs = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    v.set(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]).applyMatrix4(matrix);
    xs.push(v.x); ys.push(v.y); zs.push(v.z);
  }
  const q = (a, f) => { a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(f * a.length))]; };
  return new THREE.Box3(new THREE.Vector3(q(xs, 0.02), q(ys, 0.02), q(zs, 0.02)),
    new THREE.Vector3(q(xs, 0.98), q(ys, 0.98), q(zs, 0.98)));
}
