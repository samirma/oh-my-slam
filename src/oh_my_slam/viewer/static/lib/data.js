// Where the viewer's data comes from: the routes of viewer/routes.py, which view.sh serves, resolved
// against the page's own URL.
//
// A cloud, as every drawing function takes it (from api/cloud):
//   { header: { count, total, voxel, attrs }, arrays: { position, color?, label?, normal? } }
// position Float32Array (x, y, z per point), color Uint8Array (sRGB), label Int32Array (object id,
// 0 = unsegmented), normal Float32Array; `total` points were derived and `count` are shown, one per
// voxel of edge `voxel` metres when that is > 0 (spec §2.5 display budget); `attrs` is the
// "key=value,…" of the point-cloud attributes.

export class DataSource {
  constructor(base = './') {
    this.base = new URL(base, document.baseURI);
  }

  url(path) { return new URL(path, this.base).href; }

  async buffer(path, signal) {
    const res = await fetch(this.url(path), { signal });
    if (!res.ok) throw new Error(await errorOf(res));
    return res.arrayBuffer();
  }

  async json(path) { return JSON.parse(new TextDecoder().decode(await this.buffer(path))); }

  meta() { return this.json('api/meta'); }
  scene() { return this.json('api/scene'); }
  catalog() { return this.json('api/catalog'); }
  segmentedUrl() { return this.url('api/segmented.png'); }

  // the cloud derived with these point-cloud attributes ({key: value} or a query string)
  async cloud(attrs, signal) {
    const query = new URLSearchParams(attrs).toString();
    return parseCloudDocument(await this.buffer(`api/cloud?${query}`, signal));
  }
}

async function errorOf(res) {
  try { return (await res.json()).error || `HTTP ${res.status}`; } catch { return `HTTP ${res.status}`; }
}

// The binary document of /api/cloud (viewer/routes.py cloud_document): uint32 LE header length J,
// the JSON header, then 4-byte aligned buffers.
const TYPED = { float32: Float32Array, uint8: Uint8Array, int32: Int32Array };
export function parseCloudDocument(buffer) {
  const hlen = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, hlen)));
  const base = 4 + hlen;
  const arrays = {};
  for (const b of header.buffers) {
    const T = TYPED[b.type];
    arrays[b.name] = new T(buffer, base + b.offset, b.bytes / T.BYTES_PER_ELEMENT);
  }
  return { header, arrays };
}
