// js/url.js: every page of the web application has a stable URL (spec §2.6 "Structure"), its hash
// route, whose query keeps the page's state, changed in place.
import { afterEach, describe, expect, test } from 'bun:test';
import { parseRoute, setQuery, withQuery } from '../../../src/oh_my_slam/web/static/js/url.js';
import { stubGlobals } from '../helpers.js';

describe('parseRoute: the page a URL names', () => {
  const route = (hash) => { const r = parseRoute(hash); return { parts: r.parts, path: r.path, query: Object.fromEntries(r.query) }; };
  test('the routes of the application', () => {
    expect(route('')).toEqual({ parts: [], path: '', query: {} });
    expect(route('#/')).toEqual({ parts: [], path: '', query: {} });
    expect(route('#/image?op=segment')).toEqual({ parts: ['image'], path: 'image', query: { op: 'segment' } });
    expect(route('#/maps?filter=office%20walk')).toEqual({ parts: ['maps'], path: 'maps', query: { filter: 'office walk' } });
    expect(route('#/maps/new')).toEqual({ parts: ['maps', 'new'], path: 'maps/new', query: {} });
    expect(route('#/maps/my%20map%2F2?op=mapper-locate')).toEqual({ parts: ['maps', 'my map/2'], path: 'maps/my map/2', query: { op: 'mapper-locate' } });
    expect(route('#/maps/office/update')).toEqual({ parts: ['maps', 'office', 'update'], path: 'maps/office/update', query: {} });
  });
  test('a sloppy URL still names its page', () => {
    expect(route('#maps//office/')).toEqual({ parts: ['maps', 'office'], path: 'maps/office', query: {} });
    expect(route('#/maps/100%')).toEqual({ parts: ['maps', '100%'], path: 'maps/100%', query: {} });  // not decodable: as written
  });
});

describe('the page\'s state in its URL', () => {
  test('withQuery sets and removes keys, keeping the others and the page', () => {
    expect(withQuery('#/maps', { filter: 'office' })).toBe('#/maps?filter=office');
    expect(withQuery('#/maps?filter=office', { filter: '' })).toBe('#/maps');
    expect(withQuery('#/maps/a?op=x&keep=1', { op: null, n: 3 })).toBe('#/maps/a?keep=1&n=3');
    expect(withQuery('#/maps/a?op=x', { op: undefined })).toBe('#/maps/a');
    expect(withQuery('', { op: 'segment' })).toBe('#/?op=segment');
    expect(withQuery('#/maps', { filter: 'a b&c' })).toBe('#/maps?filter=a+b%26c');
  });

  let restore = () => {};
  afterEach(() => restore());
  test('setQuery replaces the URL in place, and only when it changes', () => {
    const replaced = [];
    restore = stubGlobals({ location: { hash: '#/maps?filter=a' }, history: { replaceState: (...a) => replaced.push(a) } });
    setQuery({ filter: 'b' });
    expect(replaced).toEqual([[null, '', '#/maps?filter=b']]);
    location.hash = '#/maps?filter=b';
    setQuery({ filter: 'b' });
    expect(replaced).toHaveLength(1);
  });
});
