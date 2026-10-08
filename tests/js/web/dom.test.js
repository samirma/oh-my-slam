// js/dom.js: the web application's formatting of figures (spec §2.6 "Web application": summaries,
// timings, sizes) and the one form of colour it puts in a style attribute.
import { describe, expect, test } from 'bun:test';
import { fmtBytes, fmtClock, fmtDate, fmtNumber, fmtSeconds, hexColor, humanize, nextId } from '../../../src/oh_my_slam/web/static/js/dom.js';
import { PALETTE_HEX } from '../helpers.js';

test('a date from Unix seconds (numbers or numeric text); nothing for a missing or bad one', () => {
  // bun test runs in UTC
  expect(fmtDate(0)).toMatch(/^Jan 1, 1970.*12:00\s?AM$/);
  expect(fmtDate('86400')).toMatch(/^Jan 2, 1970/);
  expect(fmtDate(1700000000.5)).toMatch(/^Nov 14, 2023.*10:13\s?PM$/);
  for (const bad of [null, undefined, 'soon', NaN, Infinity]) expect(fmtDate(bad)).toBe('');
});

test('a duration in the unit that reads best', () => {
  expect(fmtSeconds(0.0123)).toBe('12 ms');
  expect(fmtSeconds(0)).toBe('0 ms');
  expect(fmtSeconds(1)).toBe('1.0 s');
  expect(fmtSeconds('59.94')).toBe('59.9 s');
  expect(fmtSeconds(60)).toBe('1 min 0 s');
  expect(fmtSeconds(125.6)).toBe('2 min 6 s');
  expect(fmtSeconds(3600)).toBe('1 h 0 min');
  expect(fmtSeconds(7322)).toBe('2 h 2 min');
  for (const bad of [null, undefined, 'long', NaN]) expect(fmtSeconds(bad)).toBe('');
});

test('a running clock', () => {
  expect(fmtClock(7)).toBe('0:07');
  expect(fmtClock(65.9)).toBe('1:05');
  expect(fmtClock(3723)).toBe('1:02:03');
  expect(fmtClock(-2)).toBe('0:00');
});

test('a size in bytes, binary units up to GB', () => {
  expect(fmtBytes(null)).toBe('');
  expect(fmtBytes(0)).toBe('0 B');
  expect(fmtBytes(1023)).toBe('1023 B');
  expect(fmtBytes(1024)).toBe('1.0 KB');
  expect(fmtBytes(1536)).toBe('1.5 KB');
  expect(fmtBytes(5 * 1024 ** 2)).toBe('5.0 MB');
  expect(fmtBytes(3.25 * 1024 ** 3)).toBe('3.3 GB');
  expect(fmtBytes(2048 * 1024 ** 3)).toBe('2048.0 GB');
});

test('a figure: integers grouped, fractions to 3 decimals, anything else as text', () => {
  expect(fmtNumber(1234567)).toBe((1234567).toLocaleString());
  expect(fmtNumber(1234567)).toMatch(/^1.234.567$/);
  expect(fmtNumber(3.14159)).toBe((3.142).toLocaleString());
  expect(fmtNumber('79')).toBe('79');
  expect(fmtNumber(null)).toBe('null');
});

test('a parameter or figure name as words', () => {
  expect(humanize('min_score')).toBe('Min score');
  expect(humanize('images_or_video')).toBe('Images or video');
  expect(humanize('fps')).toBe('Fps');
  expect(humanize('')).toBe('');
});

describe('hexColor: only #rrggbb reaches a style attribute', () => {
  test('the palette\'s colours', () => {
    for (const hex of PALETTE_HEX) expect(hexColor(hex)).toBe(hex);
    expect(hexColor('#E6194B')).toBe('#E6194B');
  });
  test('anything else is refused', () => {
    for (const bad of ['#fff', 'red', '#12345g', '#1234567', 'e6194b', '#e6194b; background: url(x)', ' #e6194b', null, 0x808080]) {
      expect(hexColor(bad)).toBeNull();
    }
  });
});

test('ids are unique within the page', () => {
  const a = nextId('cloud'), b = nextId('cloud');
  expect(a).toMatch(/^cloud-\d+$/);
  expect(b).not.toBe(a);
  expect(nextId()).toMatch(/^f-\d+$/);
});
