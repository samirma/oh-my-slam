// lib/controls.js: the live controls of the §2.2 point-cloud attributes (their values are the -p
// strings), and the display-budget notice of spec §2.5.
import { describe, expect, test } from 'bun:test';
import { budgetNote, DISPLAY_POINT_BUDGET, formatValue, sliderPosition, sliderValue } from '../../../src/oh_my_slam/viewer/static/lib/controls.js';

// the controls /api/meta lists for an image (viewer/bundle.py controls: core.cloud_attrs defaults
// and _SLIDERS), the depth extent being 7.35 m
const CONTROLS = {
  color: { key: 'color', kind: 'choice', options: ['rgb', 'segment', 'height', 'none'], default: 'rgb' },
  stride: { key: 'stride', kind: 'int', min: 1, max: 16, step: 1, unit: 'px', off: null, default: '1' },
  'min-depth': { key: 'min-depth', kind: 'float', min: 0, max: 7.35, step: 0.05, unit: 'm', off: '0', default: '0' },
  'max-depth': { key: 'max-depth', kind: 'float', min: 0.05, max: 7.35, step: 0.05, unit: 'm', off: 'inf', default: 'inf' },
  edge: { key: 'edge', kind: 'float', min: 0, max: 0.5, step: 0.005, unit: '', off: '0', default: '0.04' },
  voxel: { key: 'voxel', kind: 'float', min: 0, max: 0.5, step: 0.005, unit: 'm', off: '0', default: '0' },
  normals: { key: 'normals', kind: 'toggle', default: 'off' },
};
const SLIDERS = Object.values(CONTROLS).filter((c) => c.kind === 'int' || c.kind === 'float');

describe('formatValue: what a control shows of its value', () => {
  test('a choice or a switch shows its value', () => {
    expect(formatValue(CONTROLS.color, 'segment')).toBe('segment');
    expect(formatValue(CONTROLS.normals, 'on')).toBe('on');
  });
  test('a number with its step\'s decimals and its unit', () => {
    expect(formatValue(CONTROLS.stride, '4')).toBe('4 px');
    expect(formatValue(CONTROLS['min-depth'], '0.5')).toBe('0.50 m');
    expect(formatValue(CONTROLS.voxel, '0.01')).toBe('0.010 m');
    expect(formatValue(CONTROLS.edge, '0.04')).toBe('0.040');  // a ratio: no unit
  });
  test('the value that turns a filter off says so', () => {
    expect(formatValue(CONTROLS['max-depth'], 'inf')).toBe('∞');
    expect(formatValue(CONTROLS['min-depth'], '0')).toBe('off');
    expect(formatValue(CONTROLS.voxel, '0.000')).toBe('off');
    expect(formatValue(CONTROLS.edge, 0)).toBe('off');
    expect(formatValue(CONTROLS.stride, '1')).toBe('1 px');  // stride has no off value
  });
});

describe('a slider\'s -p value', () => {
  test('at its maximum, a depth limit is off (inf); elsewhere its position, to its step', () => {
    const max = CONTROLS['max-depth'];
    expect(sliderValue(max, 7.35)).toBe('inf');
    expect(sliderValue(max, 7.3)).toBe('7.3');
    expect(sliderValue(max, 0.1 + 0.2)).toBe('0.3');  // no float noise in the -p value
    expect(sliderValue(CONTROLS.edge, 0.5)).toBe('0.5');  // only an inf off value is a maximum's
    expect(sliderValue(CONTROLS.voxel, 0.0049999)).toBe('0.005');
    expect(sliderValue(CONTROLS.stride, 3.6)).toBe('4');
  });
  test('a value sets the slider where it reads the same value back', () => {
    expect(sliderPosition(CONTROLS['max-depth'], 'inf')).toBe(7.35);
    expect(sliderPosition(CONTROLS.voxel, '0.25')).toBe(0.25);
    for (const c of SLIDERS) {
      for (const v of [c.default, String(c.min), c.off === 'inf' ? 'inf' : String(c.max)]) {
        expect(sliderValue(c, sliderPosition(c, v))).toBe(v === 'inf' ? 'inf' : String(Number(v)));
      }
    }
  });
});

describe('the display-budget notice (spec §2.5)', () => {
  const COMPLETE = '(display budget; PLY outputs and the map stay complete).';
  test('none when every point is shown', () => {
    expect(budgetNote({ count: 1200, total: 1200, voxel: 0 })).toBe('');
    expect(budgetNote({ count: 0, total: 0, voxel: 0 })).toBe('');
  });
  test('"Showing X of Y points" with the voxel edge, in a readable unit', () => {
    const note = (voxel) => budgetNote({ count: 15_999_000, total: 42_000_000, voxel });
    expect(note(0.0123)).toBe(`Showing 15,999,000 of 42,000,000 points: one per voxel of 1.2 cm edge ${COMPLETE}`);
    expect(note(1.5)).toBe(`Showing 15,999,000 of 42,000,000 points: one per voxel of 1.50 m edge ${COMPLETE}`);
    expect(note(1)).toContain('of 1.00 m edge');
    expect(note(0.01)).toContain('of 1.0 cm edge');
    expect(note(0.00412)).toContain('of 4.12 mm edge');
  });
  test('with no voxel grid, both reasons a point can be omitted', () => {
    expect(budgetNote({ count: 5, total: 7, voxel: 0 })).toBe('Showing 5 of 7 points: no voxel grid (edge 0); '
      + `each omitted point has a non-finite coordinate or repeats a shown position ${COMPLETE}`);
  });
  test('a PLY read in the page shows its points evenly spaced in the file\'s order', () => {
    expect(budgetNote({ count: 1000, total: 2503, voxel: 0, step: 2.503 }))
      .toBe(`Showing 1,000 of 2,503 points: evenly spaced in the file's order, read in this page ${COMPLETE}`);
  });
  test('the budget is 16 000 000 points', () => {
    expect(DISPLAY_POINT_BUDGET).toBe(16_000_000);
  });
});
