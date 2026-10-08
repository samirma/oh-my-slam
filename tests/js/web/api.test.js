// js/api.js: the web application talks to server.sh's public API only (spec §2.6 "API",
// "Requests"); an operation runs within its request, its stages come in Server-Timing, and every
// error carries the command's message and the code of its exit status ("Actionable errors").
import { afterEach, describe, expect, test } from 'bun:test';
import { ApiError, discardUpload, enc, getJson, parseServerTiming, postJson, runOperation, upload } from '../../../src/oh_my_slam/web/static/js/api.js';
import { fakeFetch, json, stubGlobals } from '../helpers.js';

let restore = () => {};
afterEach(() => restore());
function net(handler) { const f = fakeFetch(handler); restore = f.restore; return f.calls; }

describe('ApiError: the service\'s error shape', () => {
  test('the command\'s message, the code of its exit status, its problems by parameter', () => {
    const e = new ApiError(400, { error: { code: 'usage', message: 'argument -fps: must be > 0', http_status: 400,
      problems: [{ message: 'argument -fps: must be > 0', parameters: ['fps'] }], by_parameter: { fps: ['must be > 0'] } } });
    expect(e).toBeInstanceOf(Error);
    expect([e.message, e.status, e.code]).toEqual(['argument -fps: must be > 0', 400, 'usage']);
    expect(e.problems).toEqual([{ message: 'argument -fps: must be > 0', parameters: ['fps'] }]);
    expect(e.byParameter).toEqual({ fps: ['must be > 0'] });
  });
  test('whatever else came back still says what happened', () => {
    expect(new ApiError(500, { error: 'boom' }).message).toBe('boom');
    const bare = new ApiError(503, null);
    expect([bare.message, bare.code, bare.problems, bare.byParameter]).toEqual(['HTTP 503', null, [], {}]);
    expect(new ApiError(0, {}).message).toBe('the service did not answer');
    expect(new ApiError(404, { error: { code: 'not_found' } }).message).toBe('HTTP 404');
  });
});

test('parseServerTiming: the command\'s stages, in its order, in milliseconds', () => {
  expect(parseServerTiming(null)).toEqual([]);
  expect(parseServerTiming('')).toEqual([]);
  expect(parseServerTiming('depth;dur=812.5, segment;dur=120,total;dur=1000.25')).toEqual([
    { name: 'depth', ms: 812.5 }, { name: 'segment', ms: 120 }, { name: 'total', ms: 1000.25 }]);
  expect(parseServerTiming('cache;desc="hit";dur=3, warm, , ;dur=1')).toEqual([
    { name: 'cache', ms: 3 }, { name: 'warm', ms: null }]);
});

test('names in a URL are encoded', () => {
  expect(enc('my map/2?x')).toBe('my%20map%2F2%3Fx');
});

describe('getJson and postJson', () => {
  test('an answer is its JSON (null when empty), never from a cache', async () => {
    const calls = net((url) => (url === '/api/health' ? json({ status: 'ok' }) : new Response('')));
    const ctl = new AbortController();
    expect(await getJson('/api/health', ctl.signal)).toEqual({ status: 'ok' });
    expect(calls[0].init).toEqual({ signal: ctl.signal, headers: { Accept: 'application/json' }, cache: 'no-store' });
    expect(await getJson('/api/maps')).toBeNull();
  });

  test('a refusal throws the service\'s error; a body that is not JSON is the message', async () => {
    const answers = [json({ error: { code: 'not_found', message: 'no map named office' } }, 404),
      new Response('Bad Gateway', { status: 502 })];
    net(() => answers.shift());
    const e = await getJson('/api/maps/office').catch((err) => err);
    expect([e instanceof ApiError, e.status, e.code, e.message]).toEqual([true, 404, 'not_found', 'no map named office']);
    await expect(getJson('/api/maps')).rejects.toMatchObject({ status: 502, message: 'Bad Gateway' });
  });

  test('no answer at all: the service is not running; an abort stays an abort', async () => {
    net(() => { throw new TypeError('fetch failed'); });
    await expect(getJson('/api/health')).rejects.toMatchObject({ status: 0, message: 'the service did not answer; is server.sh running?' });
    restore();
    net(() => { throw new DOMException('aborted', 'AbortError'); });
    await expect(getJson('/api/health')).rejects.toMatchObject({ name: 'AbortError' });
  });

  test('postJson sends JSON and reads JSON', async () => {
    const answers = [json({ valid: true, command: ['segment.sh', '-i', 'a.jpg'] }), json({ error: { message: 'bad' } }, 400)];
    const calls = net(() => answers.shift());
    expect(await postJson('/api/ops/segment/validate', { image: 'a.jpg' })).toEqual({ valid: true, command: ['segment.sh', '-i', 'a.jpg'] });
    expect(calls[0].init).toEqual({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"image":"a.jpg"}', signal: undefined });
    await expect(postJson('/api/ops/segment/validate')).rejects.toMatchObject({ status: 400, message: 'bad' });
    expect(calls[1].init.body).toBe('{}');
  });
});

describe('runOperation: an operation within this request', () => {
  test('its result: the response body as received, its media type and its stages', async () => {
    const ply = new Uint8Array([112, 108, 121, 10, 0, 255]);
    const calls = net(() => new Response(ply, { headers: { 'Content-Type': 'application/x-ply; charset=binary', 'Server-Timing': 'depth;dur=5, total;dur=9' } }));
    const ctl = new AbortController();
    const res = await runOperation('mapper locate', { map: 'office' }, ctl.signal);
    expect(calls[0]).toEqual({ url: '/api/ops/mapper%20locate', init: { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: '{"map":"office"}', signal: ctl.signal } });
    expect(new Uint8Array(await res.blob.arrayBuffer())).toEqual(ply);  // byte for byte
    expect(res.mediaType).toBe('application/x-ply');
    expect(res.stages).toEqual([{ name: 'depth', ms: 5 }, { name: 'total', ms: 9 }]);
  });

  test('a body without a media type is octet-stream, without timings no stages', async () => {
    net(() => { const r = new Response(new Uint8Array([1])); r.headers.delete('Content-Type'); return r; });
    const res = await runOperation('x', {});
    expect([res.mediaType, res.stages]).toEqual(['application/octet-stream', []]);
  });

  test('a failed command throws its error; a lost connection says the command was interrupted', async () => {
    net(() => json({ error: { code: 'inference_unavailable', message: 'inference server is not running — start it with ./start_inference_server.sh' } }, 503));
    await expect(runOperation('segment', {})).rejects.toMatchObject({ status: 503, code: 'inference_unavailable' });
    restore();
    net(() => { throw new TypeError('network connection was lost'); });
    const e = await runOperation('segment', {}).catch((err) => err);
    expect([e.status, e.message]).toEqual([0, 'the connection to the service was lost before the answer: the command was interrupted. '
      + 'Check that server.sh still runs, then run it again.']);
    restore();
    net(() => { throw new DOMException('aborted', 'AbortError'); });
    await expect(runOperation('segment', {})).rejects.toMatchObject({ name: 'AbortError' });  // the user interrupted it
  });
});

describe('upload: one file as the raw request body', () => {
  let xhrs;
  function fakeXhr() {
    xhrs = [];
    restore = stubGlobals({
      XMLHttpRequest: class {
        constructor() { this.headers = {}; this.upload = {}; this.aborted = false; xhrs.push(this); }
        open(method, url) { Object.assign(this, { method, url }); }
        setRequestHeader(k, v) { this.headers[k] = v; }
        send(body) { this.body = body; }
        abort() { this.aborted = true; this.onabort(); }
        answer(status, text) { Object.assign(this, { status, responseText: text }); this.onload(); }
      },
    });
  }

  test('with its own media type and its progress; resolves to the upload', async () => {
    fakeXhr();
    const progress = [];
    const file = new File(['abc'], 'my photo.jpg', { type: 'image/jpeg' });
    const p = upload(file, (f) => progress.push(f));
    const x = xhrs[0];
    expect([x.method, x.url, x.headers, x.body]).toEqual(['POST', '/api/uploads?name=my%20photo.jpg', { 'Content-Type': 'image/jpeg' }, file]);
    x.upload.onprogress({ lengthComputable: true, loaded: 1, total: 4 });
    x.upload.onprogress({ lengthComputable: false, loaded: 2, total: 0 });
    x.answer(201, '{"id": "u1", "name": "my photo.jpg", "size": 3, "path": "uploads/u1/my photo.jpg"}');
    expect(await p).toEqual({ id: 'u1', name: 'my photo.jpg', size: 3, path: 'uploads/u1/my photo.jpg' });
    expect(progress).toEqual([0.25]);
  });

  test('a type a browser could send cross-site without asking goes as octet-stream', () => {
    fakeXhr();
    for (const type of ['text/plain', 'text/plain;charset=utf-8', 'application/x-www-form-urlencoded', 'multipart/form-data', '']) {
      upload(new File(['x'], 'f', { type }));
      expect(xhrs.at(-1).headers['Content-Type']).toBe('application/octet-stream');
    }
    upload(new File(['x'], 'clip.mp4', { type: 'video/mp4' }));
    expect(xhrs.at(-1).headers['Content-Type']).toBe('video/mp4');
  });

  test('a refusal, an interruption or a cancellation rejects with what to do', async () => {
    fakeXhr();
    const refused = upload(new File(['x'], 'f.jpg'));
    xhrs[0].upload.onprogress({ lengthComputable: true, loaded: 1, total: 1 });  // no progress callback: nothing to call
    xhrs[0].answer(413, '{"error": {"code": "upload_too_large", "message": "the upload exceeds 4 GiB"}}');
    await expect(refused).rejects.toMatchObject({ status: 413, code: 'upload_too_large', message: 'the upload exceeds 4 GiB' });
    const proxy = upload(new File(['x'], 'f.jpg'));
    xhrs[1].answer(502, '<html>Bad Gateway</html>');
    await expect(proxy).rejects.toMatchObject({ status: 502, message: '<html>Bad Gateway</html>' });
    const lost = upload(new File(['x'], 'f.jpg'));
    xhrs[2].onerror();
    await expect(lost).rejects.toMatchObject({ status: 0, message: 'the upload was interrupted; choose the file again' });
    const cancelled = upload(new File(['x'], 'f.jpg'));
    cancelled.abort();
    expect(xhrs[3].aborted).toBe(true);
    await expect(cancelled).rejects.toMatchObject({ status: 0, message: 'the upload was cancelled' });
  });
});

test('discardUpload deletes an upload no request consumed, even as the page goes away', async () => {
  const calls = net(() => new Response(null, { status: 204 }));
  await discardUpload('u 1');
  expect(calls).toEqual([{ url: '/api/uploads/u%201', init: { method: 'DELETE', keepalive: true } }]);
  restore();
  net(() => { throw new TypeError('gone'); });
  expect(await discardUpload('u2')).toBeUndefined();  // nothing to report
});
