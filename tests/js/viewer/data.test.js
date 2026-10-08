// lib/data.js: the viewer's data, from the routes view.sh serves (viewer/routes.py), resolved
// against the page's own URL; the binary cloud document of api/cloud.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { DataSource, parseCloudDocument } from '../../../src/oh_my_slam/viewer/static/lib/data.js';
import { cloudDocument, fakeFetch, json, stubGlobals } from '../helpers.js';

const position = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
const color = new Uint8Array([255, 0, 0, 128, 128, 128, 1, 2, 3]);
const label = new Int32Array([4, 0, 12]);
const normal = new Float32Array([0, 0, 1, 0, 1, 0, 1, 0, 0]);
const HEAD = { count: 3, total: 1200, voxel: 0.02, attrs: 'color=segment,voxel=0,normals=on' };

test('the cloud document: its header, and each buffer as a typed view of the bytes', () => {
  const buffer = cloudDocument(HEAD, { position, color, label, normal });
  const { header, arrays } = parseCloudDocument(buffer);
  expect(header).toMatchObject(HEAD);
  expect(arrays.position).toBeInstanceOf(Float32Array);
  expect(arrays.color).toBeInstanceOf(Uint8Array);
  expect(arrays.label).toBeInstanceOf(Int32Array);
  expect(arrays).toEqual({ position, color, label, normal });
  for (const a of Object.values(arrays)) expect(a.buffer).toBe(buffer);  // views, no copies
});

test('a document without colours (color=none) has no colour array', () => {
  const { arrays } = parseCloudDocument(cloudDocument({ ...HEAD, attrs: 'color=none' }, { position }));
  expect(Object.keys(arrays)).toEqual(['position']);
});

describe('DataSource', () => {
  let net;
  let restore;
  beforeEach(() => {
    // a page served under a prefix: every route resolves against the page's own URL
    restore = stubGlobals({ document: { baseURI: 'http://127.0.0.1:8123/view/abc/index.html' } });
  });
  afterEach(() => { net?.restore(); restore(); });

  test('JSON routes relative to the page', async () => {
    net = fakeFetch((url) => json({ url }));
    const data = new DataSource();
    expect(data.url('api/meta')).toBe('http://127.0.0.1:8123/view/abc/api/meta');
    expect(await data.meta()).toEqual({ url: 'http://127.0.0.1:8123/view/abc/api/meta' });
    expect(await data.scene()).toEqual({ url: 'http://127.0.0.1:8123/view/abc/api/scene' });
    expect(await data.catalog()).toEqual({ url: 'http://127.0.0.1:8123/view/abc/api/catalog' });
    expect(data.segmentedUrl()).toBe('http://127.0.0.1:8123/view/abc/api/segmented.png');
    expect(new DataSource('../').url('api/meta')).toBe('http://127.0.0.1:8123/view/api/meta');
  });

  test('a cloud for the attributes, as -p values in the query, cancellable', async () => {
    const body = cloudDocument(HEAD, { position, label });
    net = fakeFetch(() => new Response(body));
    const ctl = new AbortController();
    const cloud = await new DataSource().cloud([['color', 'segment'], ['voxel', '0'], ['max-depth', 'inf']], ctl.signal);
    expect(net.calls[0].url).toBe('http://127.0.0.1:8123/view/abc/api/cloud?color=segment&voxel=0&max-depth=inf');
    expect(net.calls[0].init.signal).toBe(ctl.signal);
    expect(cloud.header.count).toBe(3);
    expect(cloud.arrays.label).toEqual(label);
    await new DataSource().cloud({ stride: '2' });
    expect(net.calls[1].url).toEndWith('api/cloud?stride=2');
  });

  test('a refused request throws the server\'s message, else its HTTP status', async () => {
    const answers = [json({ error: 'stride=0: an integer >= 1' }, 400), json({}, 404), new Response('<html>', { status: 502 })];
    net = fakeFetch(() => answers.shift());
    const data = new DataSource();
    await expect(data.cloud({ stride: '0' })).rejects.toThrow('stride=0: an integer >= 1');
    await expect(data.meta()).rejects.toThrow('HTTP 404');
    await expect(data.scene()).rejects.toThrow('HTTP 502');
  });
});
