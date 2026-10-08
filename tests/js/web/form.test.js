// js/form.js: forms generated from the API description (spec §2.6 "Forms"): each field shows its
// default from the command's option definition, fields that do not apply to the current choices
// are hidden and not sent, and the values sent are the API's.
import { describe, expect, test } from 'bun:test';
import { attrsValue, defaultText, fileState, holds, mapNote, numberValue } from '../../../src/oh_my_slam/web/static/js/form.js';

test('the default a field shows', () => {
  expect(defaultText({ kind: 'enum', default: 'json' })).toBe('json');
  expect(defaultText({ kind: 'number', default: 2 })).toBe('2');
  expect(defaultText({ kind: 'number', default: 0 })).toBe('0');
  expect(defaultText({ kind: 'flag', default: true })).toBe('on');
  expect(defaultText({ kind: 'flag', default: false })).toBe('off');
  expect(defaultText({ kind: 'text', default: null })).toBeNull();
  expect(defaultText({ kind: 'text' })).toBeNull();
  // the point-cloud attributes show a default per attribute instead
  expect(defaultText({ kind: 'attrs', default: 'color=rgb,voxel=0' })).toBeNull();
});

describe('holds: whether a field applies to the current choices', () => {
  const values = { format: 'ply', fps: '', inputs: undefined, flag: false, given: 'x' };
  const names = { inputs: ['clip.MP4'], images: ['a.jpg', 'b.jpg'], none: [] };
  const check = (w) => holds(w, (n) => values[n], (n) => names[n] || []);
  const VIDEO = ['.mp4', '.mov'];
  test('an option among values', () => {
    expect(check({ option: 'format', in: ['ply'] })).toBe(true);
    expect(check({ option: 'format', in: ['json', 'depth'] })).toBe(false);
  });
  test('an option given', () => {
    expect(check({ option: 'given', is: 'given' })).toBe(true);
    for (const option of ['fps', 'inputs', 'flag', 'missing']) expect(check({ option, is: 'given' })).toBe(false);
  });
  test('exactly one video, by its suffix in any case', () => {
    expect(check({ option: 'inputs', is: 'video', suffixes: VIDEO })).toBe(true);
    expect(check({ option: 'images', is: 'video', suffixes: VIDEO })).toBe(false);
    expect(check({ option: 'none', is: 'video', suffixes: VIDEO })).toBe(false);
    expect(check({ option: 'inputs', is: 'video' })).toBe(false);
    expect(holds({ option: 'i', is: 'video', suffixes: VIDEO }, () => undefined, () => ['movie'])).toBe(false);  // no suffix
  });
  test('a condition this page does not know does not hide the field', () => {
    expect(check({ option: 'format', is: 'something new' })).toBe(true);
  });
});

test('a number field sends a number when its text is one, else the text for the command to refuse', () => {
  for (const [text, value] of [['2', 2], [' -1.5e3 ', -1500], ['.5', 0.5], ['5.', 5], ['+3', 3], ['0', 0]]) expect(numberValue(text)).toBe(value);
  for (const text of ['abc', '0x10', '1e400', 'Infinity', '1,5', '2 fps', 'NaN']) expect(numberValue(text)).toBe(text.trim());
  expect(numberValue('')).toBeUndefined();
  expect(numberValue('   ')).toBeUndefined();
});

test('the point-cloud attributes send only the ones changed from their defaults', () => {
  expect(attrsValue([['color', 'segment', 'rgb'], ['voxel', ' 0.01 ', '0'], ['normals', 'off', 'off']])).toBe('color=segment,voxel=0.01');
  expect(attrsValue([['stride', '1', 1], ['edge', '', '0.04']])).toBeUndefined();  // defaults and empty fields: nothing
  expect(attrsValue([])).toBeUndefined();
});

test('a map name for the mode that writes maps says what it will do', () => {
  const maps = [{ name: 'office' }, { name: 'street' }];
  expect(mapNote(' office ', maps)).toBe('Map office exists: the inputs extend it.');
  expect(mapNote('kitchen', maps)).toBe('A new map kitchen will be created.');
  expect(mapNote('  ', maps)).toBe('');
  expect(mapNote('office', [])).toBe('A new map office will be created.');
});

test('the state of a chosen file', () => {
  expect(fileState({ state: 'uploading', progress: 0.426 })).toBe('uploading 43 %');
  expect(fileState({ state: 'uploading' })).toBe('uploading 0 %');
  expect(fileState({ state: 'failed', error: 'the upload exceeds 4 GiB' })).toBe('upload failed: the upload exceeds 4 GiB');
  expect(fileState({ state: 'path' })).toBe('workspace path');
  expect(fileState({ state: 'ready', size: 1536 })).toBe('uploaded, 1.5 KB');
  expect(fileState({ state: 'ready', size: 0 })).toBe('uploaded, 0 B');
  expect(fileState({ state: 'ready' })).toBe('uploaded');
});
