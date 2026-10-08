// js/pages/maps.js and js/pages/image.js: the workspace's maps as cards with their summary figures
// and a filter, a map's update history with timings (spec §2.6 "Maps", from map.json); the name a
// result is downloaded under.
import { describe, expect, test } from 'bun:test';
import { cardFigures, count, figures, mapMatches, updateStages } from '../../../src/oh_my_slam/web/static/js/pages/maps.js';
import { stem } from '../../../src/oh_my_slam/web/static/js/pages/image.js';
import { fmtDate } from '../../../src/oh_my_slam/web/static/js/dom.js';

// a map summary as /api/maps lists it
const SUMMARY = {
  name: 'office', path: 'maps/office', frames: 79, objects: 12, update_count: 2, updated_at: 1700000000,
  last_update: { total_s: 125.6, kind: 'images', inputs: ['a.jpg', 'b.jpg'] },
  next_object_id: 13, scale_m: 1.25, closed: false, keyframes: [{ id: 1 }, { id: 2 }, { id: 3 }],
  tags: ['indoor', 'desk'], notes: null, many: [1, 2, 3, 4, 5], empty: [], ok: true,
};

describe('figures: a map\'s metadata, readable', () => {
  test('dates, durations, counts, yes/no; the store\'s own bookkeeping left out', () => {
    expect(figures(SUMMARY)).toEqual([
      ['Frames', '79'], ['Objects', '12'], ['Update count', '2'], ['Updated', fmtDate(1700000000)],
      ['Last update total', '2 min 6 s'], ['Last update kind', 'images'], ['Last update inputs', 'a.jpg, b.jpg'],
      ['Scale m', '1.25'], ['Closed', 'no'], ['Keyframes', '3'], ['Tags', 'indoor, desk'], ['Notes', '—'],
      ['Many', '5'], ['Empty', '—'], ['Ok', 'yes'],
    ]);
  });
  test('a record one level down only; other names hidden on request', () => {
    expect(figures({ a: { b: { c: 1 }, d: [{ x: 1 }], at: 0 }, b_s: 'soon' }, new Set())).toEqual([
      ['A d', '1'], ['A at', fmtDate(0)], ['B', 'soon']]);
    expect(figures(null)).toEqual([]);
    expect(figures({ id: 3, kind: 'video' }, new Set(['id']))).toEqual([['Kind', 'video']]);
  });
});

test('a card shows the main figures and how long the last update took, else the first four', () => {
  expect(cardFigures(SUMMARY)).toEqual([['Frames', '79'], ['Objects', '12'], ['Updates', '2'], ['Updated', fmtDate(1700000000)],
    ['Last update took', '2 min 6 s']]);
  expect(cardFigures({ name: 'x', frames: 3, last_update: { kind: 'images' } })).toEqual([['Frames', '3']]);
  expect(cardFigures({ name: 'x', a: 1, b: 2, c: 3, d: 4, e: 5 })).toEqual([['A', '1'], ['B', '2'], ['C', '3'], ['D', '4']]);
});

test('the filter matches a map\'s name or any of its figures', () => {
  expect(mapMatches(SUMMARY, '')).toBe(true);
  expect(mapMatches(SUMMARY, 'off')).toBe(true);
  expect(mapMatches(SUMMARY, 'frames 79')).toBe(true);
  expect(mapMatches(SUMMARY, 'desk')).toBe(true);
  expect(mapMatches(SUMMARY, 'street')).toBe(false);
  expect(mapMatches({ name: 'Office' }, 'office')).toBe(true);  // the filter is lower case
});

describe('the update history', () => {
  test('a list counts its entries, any other figure reads as on the cards', () => {
    expect(count(['a.jpg', 'b.jpg', 'c.jpg'])).toBe('3');
    expect(count(1234)).toBe((1234).toLocaleString());
    expect(count(null)).toBe('—');
    expect(count('ok')).toBe('ok');
  });
  test('an update\'s stages in milliseconds, its total last', () => {
    expect(updateStages({ stages_s: { sfm: 12.5, depth: 0.25 }, total_s: 20 })).toEqual([
      { name: 'sfm', ms: 12500 }, { name: 'depth', ms: 250 }, { name: 'total', ms: 20000 }]);
    expect(updateStages({ total_s: 0 })).toEqual([{ name: 'total', ms: 0 }]);
    expect(updateStages({})).toEqual([]);
  });
});

test('a result is downloaded under its input\'s name, without folders or suffix', () => {
  expect(stem('uploads/u1/IMG_0042.final.jpg')).toBe('IMG_0042.final');
  expect(stem('photo')).toBe('photo');
  expect(stem(null)).toBe('');
  expect(stem(undefined)).toBe('');
});
