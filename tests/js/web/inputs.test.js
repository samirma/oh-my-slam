// js/inputs.js: the inputs of a request as the user chose them, so that a result or a warning names a
// file as the user knows it (its name and its number among several), not by its upload's path.
import { expect, test } from 'bun:test';
import { cameraTitle, chosenFor, inputLabel, warningText } from '../../../src/oh_my_slam/web/static/js/inputs.js';

const CHOSEN = new Map([
  ['uploads/u1/IMG_1.jpg', { name: 'IMG_1.jpg', index: 0, multiple: true, parameter: 'inputs' }],
  ['uploads/u2/IMG_2.jpg', { name: 'IMG_2.jpg', index: 1, multiple: true, parameter: 'inputs' }],
  ['photo.jpg', { name: 'photo.jpg', index: 0, multiple: false, parameter: 'image' }],
]);

test('an input\'s name, with its number among several', () => {
  expect(inputLabel(CHOSEN.get('uploads/u2/IMG_2.jpg'))).toBe('IMG_2.jpg (input 2)');
  expect(inputLabel(CHOSEN.get('photo.jpg'))).toBe('photo.jpg');
});

test('a camera by the input it placed: one numbering everywhere (from 1, as a bird\'s-eye image tags it)', () => {
  expect(cameraTitle('located_1', CHOSEN.get('uploads/u2/IMG_2.jpg'))).toBe('input 2 (located_1)');
  expect(cameraTitle('last_known', null)).toBe('last_known');
});

test('the input a path names: as the request sent it, or the absolute path the command was given', () => {
  expect(chosenFor('uploads/u1/IMG_1.jpg', CHOSEN).index).toBe(0);
  expect(chosenFor('/Users/me/oh-my-slam-data/uploads/u2/IMG_2.jpg', CHOSEN).name).toBe('IMG_2.jpg');
  expect(chosenFor('/x/myphoto.jpg', CHOSEN)).toBeNull();  // a path ends with it only after a "/"
  expect(chosenFor('frames/f000003.jpg', CHOSEN)).toBeNull();
  expect(chosenFor('', CHOSEN)).toBeNull();
  expect(chosenFor('photo.jpg', null)).toBeNull();
});

test('a warning names each input it names as the user chose it', () => {
  const w = { message: 'uploads/u2/IMG_2.jpg: not located — 9 inliers; uploads/u2/IMG_2.jpg again',
    inputs: [{ parameter: 'inputs', index: 1, path: 'uploads/u2/IMG_2.jpg' }, { parameter: 'inputs', index: 5, path: 'inputs/x.jpg' }] };
  expect(warningText(w, CHOSEN)).toBe('IMG_2.jpg (input 2): not located — 9 inliers; IMG_2.jpg (input 2) again');
  expect(warningText({ message: '-fps applies to video input only; ignored for images' }, CHOSEN))
    .toBe('-fps applies to video input only; ignored for images');
  expect(warningText({}, null)).toBe('');
});
