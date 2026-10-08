// js/cloudresult.js: a point-cloud result drawn in the page (spec §2.6 "Image": drawn in 3D with the
// colours and attributes it carries; rotate, pan and zoom), with its text alternative, its buttons
// and the display budget of the §2.5 viewer.
import { afterEach, describe, expect, test } from 'bun:test';
import { budgetOf, drawnText, factRows, joinWords, points, TOOLS } from '../../../src/oh_my_slam/web/static/js/cloudresult.js';
import * as viewer from '../../../src/oh_my_slam/viewer/static/lib/cloudview.js';
import { stubGlobals } from '../helpers.js';

describe('the display budget', () => {
  let restore = () => {};
  afterEach(() => restore());
  test('the viewer\'s, unless a test sets a smaller one', () => {
    restore = stubGlobals({ window: {} });
    expect(budgetOf(viewer)).toBe(16_000_000);
    for (const [set, budget] of [[5000, 5000], ['250', 250], [0, 16_000_000], [-3, 16_000_000], ['many', 16_000_000]]) {
      window.__cloudBudget = set;
      expect(budgetOf(viewer)).toBe(budget);
    }
  });
});

test('its points, and when fewer are drawn, of how many', () => {
  expect(points({ count: 12345, total: 12345 })).toBe('12,345');
  expect(points({ count: 1000, total: 2503 })).toBe('1,000 drawn of 2,503');
  expect(drawnText({ count: 12345, total: 12345 })).toBe('Drawn: all 12,345 points.');
  expect(drawnText({ count: 1000, total: 2503 })).toBe('Drawn: 1,000 of its 2,503 points (see below).');
});

test('a list in words', () => {
  expect(joinWords([])).toBe('');
  expect(joinWords(['position'])).toBe('position');
  expect(joinWords(['position', 'normal'])).toBe('position and normal');
  expect(joinWords(['position', 'colour (color=segment)', 'normal', 'object id'])).toBe('position, colour (color=segment), normal and object id');
});

test('its caption: points, what each carries, the recorded attributes, the frame and the format', () => {
  const facts = { count: 2, total: 5, carries: ['position', 'colour (color=segment)'], attrs: 'color=segment,normals=on',
    frame: viewer.IMAGE_FRAME, format: 'binary_little_endian 1.0', upright: true };
  expect(factRows(facts)).toEqual([
    ['Points', '2 drawn of 5'], ['Each point carries', 'position and colour (color=segment)'],
    ['Recorded attributes', 'color=segment, normals=on'],
    ['Frame', `${viewer.IMAGE_FRAME}; shown upright, as view.sh shows an image (level camera)`],
    ['PLY format', 'binary_little_endian 1.0'],
  ]);
  expect(factRows({ ...facts, frame: 'oh-my-slam map frame (z up), metres', upright: false })[3])
    .toEqual(['Frame', 'oh-my-slam map frame (z up), metres']);
  expect(factRows({ count: 3, total: 3, carries: ['position'], attrs: '', frame: '', format: '', upright: false }))
    .toEqual([['Points', '3'], ['Each point carries', 'position']]);
});

test('the buttons make the moves of the viewer\'s keys, in labelled groups', () => {
  expect(TOOLS.map(([group, buttons]) => [group, buttons.map(([, name]) => name)])).toEqual([
    ['Rotate', ['Rotate left', 'Rotate right', 'Rotate up', 'Rotate down']],
    ['Pan', ['Pan left', 'Pan right', 'Pan up', 'Pan down']],
    ['Zoom', ['Zoom in', 'Zoom out']],
  ]);
  const KEYS = {
    'Rotate left': { key: 'ArrowLeft' }, 'Rotate right': { key: 'ArrowRight' }, 'Rotate up': { key: 'ArrowUp' },
    'Rotate down': { key: 'ArrowDown' }, 'Pan left': { key: 'ArrowLeft', shiftKey: true }, 'Pan right': { key: 'ArrowRight', shiftKey: true },
    'Pan up': { key: 'ArrowUp', shiftKey: true }, 'Pan down': { key: 'ArrowDown', shiftKey: true }, 'Zoom in': { key: '+' }, 'Zoom out': { key: '-' },
  };
  const DEFAULTS = { rotate: [0, 0], pan: [0, 0], zoom: [1] };  // rotate(left, up = 0), pan(right, up = 0)
  for (const [, buttons] of TOOLS) {
    for (const [symbol, name, move] of buttons) {
      expect(symbol.length).toBe(1);
      let made;
      const spy = Object.fromEntries(Object.keys(DEFAULTS).map((m) => [m, (...args) => { made = [m, ...DEFAULTS[m].map((d, i) => args[i] ?? d)]; }]));
      move(spy, viewer);
      const [m, ...args] = viewer.keyMove(KEYS[name]);
      expect(made).toEqual([m, ...args.map((a) => a + 0)]);
    }
  }
});
