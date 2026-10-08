// js/request.js and js/runpanel.js: a request runs within its page (spec §2.6 "Requests"); the page
// shows whether it runs or waits for its turn, and for how long ("Responsive feedback"), and states
// the consequence before running or interrupting it ("Confirmation", "Interruption").
import { describe, expect, test } from 'bun:test';
import { interruptConsequence, ownEntry, sameCommand, waitedNote } from '../../../src/oh_my_slam/web/static/js/request.js';
import { consequence } from '../../../src/oh_my_slam/web/static/js/runpanel.js';
import { Operation } from '../../../src/oh_my_slam/web/static/js/store.js';
import { openapiDocument } from '../helpers.js';

const DOC = openapiDocument();
const op = (id) => new Operation(id, DOC.paths[`/api/ops/${id}`].post);

test('what interrupting a request leaves behind, in the commands\' terms', () => {
  expect(interruptConsequence(op('mapper-update'))).toBe('mapper.sh update stops as Ctrl-C would stop it: the map is left exactly '
    + 'as it was before this update, and there is no result.');
  expect(interruptConsequence(op('segment'))).toBe('segment.sh stops as Ctrl-C would stop it, and there is no result.');
});

test('what running an operation does, stated before it runs', () => {
  const runs = 'Runs mapper.sh locate within this page\'s request and shows its result here: only for retrieval in maps of more '
    + 'keyframes than are matched exhaustively.';
  const queue = 'Requests that use the inference server run one at a time, in arrival order: it may wait for its turn.';
  expect(consequence(op('mapper-locate'))).toBe(`${runs} ${queue}`);
  expect(consequence(op('mapper-update'))).toBe('Runs mapper.sh update within this page\'s request and shows its result here: '
    + `infers depth and objects of every new keyframe. ${queue} It is the only operation that changes a map, and it can take many minutes.`);
  const never = op('segment');
  never.x = { ...never.x, inference: 'never', inference_text: 'needs no inference server' };
  expect(consequence(never)).toBe('Runs segment.sh within this page\'s request and shows its result here: needs no inference server.');
});

describe('this page\'s request among the service\'s requests in progress', () => {
  const mine = ['segment.sh', '-i', 'uploads/u1/a.jpg'];
  test('the same command line', () => {
    expect(sameCommand(mine, [...mine])).toBe(true);
    expect(sameCommand(mine, ['segment.sh', '-i', 'uploads/u2/a.jpg'])).toBe(false);
    expect(sameCommand(mine, mine.slice(0, 2))).toBe(false);
    expect(sameCommand(null, mine)).toBe(false);
    expect(sameCommand(mine, 'segment.sh -i uploads/u1/a.jpg')).toBe(false);
  });
  test('of identical requests, the one that arrived nearest to when this one was sent', () => {
    const list = [
      { command: ['reconstruct.sh', '-i', 'b.jpg'], arrived_at: 100.0, state: 'running' },
      { command: mine, arrived_at: 95.0, state: 'waiting' },
      { command: mine, arrived_at: 100.2, state: 'waiting' },
      { command: mine, arrived_at: 101.0, state: 'waiting' },
    ];
    expect(ownEntry(list, mine, 100.3)).toBe(list[2]);
    expect(ownEntry(list, mine, 90)).toBe(list[1]);
    expect(ownEntry(list, ['mapper.sh', 'update'], 100)).toBeNull();
    expect(ownEntry([], mine, 100)).toBeNull();
  });
});

test('how long an ended request waited for its turn', () => {
  expect(waitedNote(0.6, true)).toBe('');  // below a second: not worth saying
  expect(waitedNote(5, true)).toBe('including 5.0 s waiting for its turn');
  expect(waitedNote(75, false)).toBe('including at least 1 min 15 s waiting for its turn');  // seen waiting, start unknown
});
