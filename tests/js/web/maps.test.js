// js/pages/maps.js and js/pages/image.js: the workspace's maps as cards with their summary figures
// and a filter, a map's update history with timings (spec §2.6 "Maps", from map.json); the name a
// result is downloaded under.
import { describe, expect, test } from 'bun:test';
import { autoRuns, cardFigures, count, figures, imageLinks, linkHref, mapMatches, mapsLead, oneOf, producesText, removeQuestion,
  resultsText, TRACKED_TITLE, trackedCounts, trackedLines, updateStages, viewerStatus } from '../../../src/oh_my_slam/web/static/js/pages/maps.js';
import { Operation } from '../../../src/oh_my_slam/web/static/js/store.js';
import { openapiDocument } from '../helpers.js';
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
  test('the detections an update tracked, confirmed or not: never called the objects the map lists', () => {
    expect(trackedCounts({ objects: { total: 30, confirmed: 8, unconfirmed: 22 } })).toEqual({ confirmed: 8, unconfirmed: 22, total: 30 });
    expect(trackedCounts({ objects: { total: 7 } })).toEqual({ confirmed: null, unconfirmed: null, total: 7 });  // an older record
    expect(trackedCounts({ objects: 'x' })).toEqual({ confirmed: null, unconfirmed: null, total: null });
    expect(trackedCounts(null)).toEqual({ confirmed: null, unconfirmed: null, total: null });
    expect(trackedLines({ confirmed: 8, unconfirmed: 22, total: 30 })).toEqual(['8 confirmed', '+ 22 unconfirmed']);
    expect(trackedLines({ confirmed: 8, unconfirmed: null, total: 8 })).toEqual(['8 confirmed']);
    expect(trackedLines({ confirmed: null, unconfirmed: null, total: 7 })).toEqual(['7']);
    expect(trackedLines({ confirmed: null, unconfirmed: null, total: null })).toEqual(['—']);
    expect(TRACKED_TITLE).toContain('confirmed or not');
    expect(TRACKED_TITLE).toContain("summary's Objects");
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

test('what an operation that takes a map gives, from its definition', () => {
  const doc = openapiDocument();
  const op = (id) => new Operation(id, doc.paths[`/api/ops/${id}`].post);
  expect(producesText(op('mapper-locate'))).toBe('mapper.sh locate: Camera pose of images in an existing map (read-only). Its result: '
    + "the scene description (JSON), the point cloud (PLY) or the bird's-eye image of the map (PNG), by its format.");
  const one = op('segment');
  one.x = { ...one.x, outputs: [{ via: 'stdout', format: 'json', name: 'result', when: [] }] };
  expect(producesText(one)).toBe('segment.sh: Instance segmentation of one image. Its result: result (JSON).');
  // the maps page, in plain words: what a map's page gives
  expect(mapsLead([op('mapper-locate'), one])).toBe('The maps of this workspace. Open one to see its summary and history, and to get '
    + "the scene description, the point cloud, the bird's-eye image of the map or result.");
  expect(mapsLead([])).toBe('The maps of this workspace. Open one to see its summary and history.');
  // a card's links straight to an image result of an operation that needs nothing but the map
  const links = imageLinks([op('mapper-locate'), op('mapper-update')]);
  expect(links).toEqual([{ text: "Bird's-eye image of the map", op: 'mapper-locate', query: { format: 'png' } }]);
  expect(linkHref('my map', links[0])).toBe('#/maps/my%20map?op=mapper-locate&format=png&run=1');  // run at once
  // run unasked only when its check passed, it needs no inference server and writes nothing else
  const locate = op('mapper-locate');
  expect(autoRuns(locate, { valid: true, inference: false })).toBe(true);
  expect(autoRuns(locate, { valid: true, inference: true })).toBe(false);
  expect(autoRuns(locate, { valid: true, inference: null })).toBe(false);
  expect(autoRuns(locate, { valid: false, inference: null })).toBe(false);
  expect(autoRuns(locate, null)).toBe(false);
  expect(autoRuns(op('mapper-update'), { valid: true, inference: false })).toBe(false);  // it writes a map
  const needy = op('mapper-locate');
  needy.params = needy.params.map((p) => (p.name === 'inputs' ? { ...p, required: true } : p));
  expect(imageLinks([needy])).toEqual([]);  // an image it gives only with more than the map
  const untitled = op('mapper-locate');
  untitled.x = { ...untitled.x, outputs: [{ via: 'stdout', format: 'png', name: 'result', media_type: 'image/png', when: [{ option: 'format', in: ['png'] }] }] };
  expect(imageLinks([untitled])[0].text).toBe('Result');
  expect(resultsText(op('reconstruct'))).toBe('Its result: the scene description (JSON), the 16-bit depth image (PNG) '
    + 'or the point cloud (PLY), by its format.');
  expect([oneOf([]), oneOf(['a']), oneOf(['a', 'b'])]).toEqual(['', 'a', 'a or b']);
});

describe('a map\'s viewer and its removal', () => {
  test('the viewer\'s state in words: where its page can be opened', () => {
    expect(viewerStatus({ running: true, url: 'http://127.0.0.1:5000/' })).toBe(
      'The viewer runs. Its page is served on the machine that runs the service only: open it there.');
    expect(viewerStatus({ running: false, url: null })).toBe('The viewer is stopped.');
    expect(viewerStatus(null)).toBe('The viewer is stopped.');
  });
  test('the removal asks first and says it is for good (and that a running viewer stops)', () => {
    expect(removeQuestion('office', { running: false })).toEqual({
      title: 'Remove map office for good?',
      body: ['Map office is deleted from this workspace for good: its keyframes, objects and update history cannot be recovered.'],
      yes: 'Remove the map for good', no: 'Keep the map', danger: true,
    });
    expect(removeQuestion('office', { running: true }).body[1]).toBe('Its viewer is stopped first.');
    expect(removeQuestion('office', null).body).toHaveLength(1);
  });
  test('the viewer\'s state is no figure of the map', () => {
    expect(figures({ frames: 3, viewer: { running: true, url: 'http://127.0.0.1:5000/' } })).toEqual([['Frames', '3']]);
  });
});
