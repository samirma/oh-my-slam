// A JSON Schema draft-07 validator for the browser, small and faithful to the draft's validation
// vocabulary (no build step, no dependency): type, enum, const, the numeric, string, array and
// object keywords, properties / patternProperties / additionalProperties, dependencies,
// propertyNames, allOf / anyOf / oneOf / not, if / then / else, boolean schemas and local $ref
// (JSON pointers into the same document). `format` is an annotation only, as in the Python
// validator the commands use (jsonschema.Draft7Validator without a format checker).
//
// validate(schema, instance) → [{ path: 'a/0/b', message }] (empty: valid).

const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

function typeOf(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number';
  return typeof v;  // string, boolean, object
}

function isType(v, t) {
  const actual = typeOf(v);
  if (t === 'number') return actual === 'number' || actual === 'integer';
  return actual === t;
}

function equal(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null || typeof a !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) return a.length === b.length && a.every((x, i) => equal(x, b[i]));
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => hasOwn(b, k) && equal(a[k], b[k]));
}

function pointer(root, ref) {
  if (ref === '#') return root;
  if (!ref.startsWith('#/')) throw new Error(`unsupported $ref ${ref} (only local references)`);
  let node = root;
  for (const raw of ref.slice(2).split('/')) {
    const key = decodeURIComponent(raw).replaceAll('~1', '/').replaceAll('~0', '~');
    if (node === null || typeof node !== 'object' || !hasOwn(node, key)) throw new Error(`unresolvable $ref ${ref}`);
    node = node[key];
  }
  return node;
}

const show = (v) => { const s = JSON.stringify(v); return s && s.length > 60 ? `${s.slice(0, 57)}…` : s; };

export function validate(schema, instance) {
  const errors = [];
  const regex = new Map();
  const re = (p) => { if (!regex.has(p)) regex.set(p, new RegExp(p, 'u')); return regex.get(p); };

  function check(s, v, path, out) {
    if (s === true) return;
    if (s === false) { out.push({ path, message: 'no value is allowed here' }); return; }
    if (hasOwn(s, '$ref')) { check(pointer(schema, s.$ref), v, path, out); return; }  // draft-07: $ref alone
    const err = (message) => out.push({ path, message });
    if (hasOwn(s, 'type')) {
      const types = [].concat(s.type);
      if (!types.some((t) => isType(v, t))) err(`${show(v)} is not of type ${types.map((t) => `'${t}'`).join(', ')}`);
    }
    if (hasOwn(s, 'enum') && !s.enum.some((x) => equal(x, v))) err(`${show(v)} is not one of ${show(s.enum)}`);
    if (hasOwn(s, 'const') && !equal(s.const, v)) err(`${show(s.const)} was expected`);
    if (typeof v === 'number') {
      if (hasOwn(s, 'minimum') && v < s.minimum) err(`${v} is less than the minimum of ${s.minimum}`);
      if (hasOwn(s, 'maximum') && v > s.maximum) err(`${v} is greater than the maximum of ${s.maximum}`);
      if (hasOwn(s, 'exclusiveMinimum') && v <= s.exclusiveMinimum) err(`${v} is less than or equal to the minimum of ${s.exclusiveMinimum}`);
      if (hasOwn(s, 'exclusiveMaximum') && v >= s.exclusiveMaximum) err(`${v} is greater than or equal to the maximum of ${s.exclusiveMaximum}`);
      if (hasOwn(s, 'multipleOf')) { const q = v / s.multipleOf; if (Math.abs(q - Math.round(q)) > 1e-9) err(`${v} is not a multiple of ${s.multipleOf}`); }
    }
    if (typeof v === 'string') {
      const n = [...v].length;
      if (hasOwn(s, 'minLength') && n < s.minLength) err(`${show(v)} is too short`);
      if (hasOwn(s, 'maxLength') && n > s.maxLength) err(`${show(v)} is too long`);
      if (hasOwn(s, 'pattern') && !re(s.pattern).test(v)) err(`${show(v)} does not match ${show(s.pattern)}`);
    }
    if (Array.isArray(v)) {
      if (hasOwn(s, 'minItems') && v.length < s.minItems) err(`${show(v)} is too short (at least ${s.minItems} items)`);
      if (hasOwn(s, 'maxItems') && v.length > s.maxItems) err(`${show(v)} is too long (at most ${s.maxItems} items)`);
      if (s.uniqueItems && v.some((x, i) => v.findIndex((y) => equal(x, y)) !== i)) err(`${show(v)} has non-unique elements`);
      if (Array.isArray(s.items)) {
        s.items.forEach((it, i) => { if (i < v.length) check(it, v[i], `${path}/${i}`, out); });
        if (hasOwn(s, 'additionalItems')) for (let i = s.items.length; i < v.length; i++) check(s.additionalItems, v[i], `${path}/${i}`, out);
      } else if (hasOwn(s, 'items')) {
        v.forEach((x, i) => check(s.items, x, `${path}/${i}`, out));
      }
      if (hasOwn(s, 'contains') && !v.some((x) => { const o = []; check(s.contains, x, path, o); return !o.length; })) err('none of the items is valid under the given schema');
    }
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      const keys = Object.keys(v);
      if (hasOwn(s, 'required')) for (const k of s.required) if (!hasOwn(v, k)) err(`'${k}' is a required property`);
      if (hasOwn(s, 'minProperties') && keys.length < s.minProperties) err(`${show(v)} does not have enough properties`);
      if (hasOwn(s, 'maxProperties') && keys.length > s.maxProperties) err(`${show(v)} has too many properties`);
      const props = s.properties || {};
      const patterns = Object.keys(s.patternProperties || {});
      const extra = [];
      for (const k of keys) {
        let matched = false;
        if (hasOwn(props, k)) { matched = true; check(props[k], v[k], `${path}/${k}`, out); }
        for (const p of patterns) if (re(p).test(k)) { matched = true; check(s.patternProperties[p], v[k], `${path}/${k}`, out); }
        if (!matched) extra.push(k);
      }
      if (hasOwn(s, 'additionalProperties')) {
        if (s.additionalProperties === false) {
          if (extra.length) err(`Additional properties are not allowed (${extra.map((k) => `'${k}'`).join(', ')} ${extra.length > 1 ? 'were' : 'was'} unexpected)`);
        } else for (const k of extra) check(s.additionalProperties, v[k], `${path}/${k}`, out);
      }
      if (hasOwn(s, 'propertyNames')) for (const k of keys) check(s.propertyNames, k, path, out);
      for (const [k, dep] of Object.entries(s.dependencies || {})) {
        if (!hasOwn(v, k)) continue;
        if (Array.isArray(dep)) { for (const d of dep) if (!hasOwn(v, d)) err(`'${d}' is a dependency of '${k}'`); } else check(dep, v, path, out);
      }
    }
    if (hasOwn(s, 'allOf')) for (const sub of s.allOf) check(sub, v, path, out);
    const passes = (sub) => { const o = []; check(sub, v, path, o); return o; };
    if (hasOwn(s, 'anyOf')) {
      const results = s.anyOf.map(passes);
      if (!results.some((o) => !o.length)) err(`${show(v)} is not valid under any of the given schemas`);
    }
    if (hasOwn(s, 'oneOf')) {
      const ok = s.oneOf.map(passes).filter((o) => !o.length).length;
      if (ok === 0) err(`${show(v)} is not valid under any of the given schemas`);
      else if (ok > 1) err(`${show(v)} is valid under each of ${ok} of the given schemas`);
    }
    if (hasOwn(s, 'not') && !passes(s.not).length) err(`${show(v)} should not be valid under ${show(s.not)}`);
    if (hasOwn(s, 'if')) {
      if (!passes(s.if).length) { if (hasOwn(s, 'then')) check(s.then, v, path, out); } else if (hasOwn(s, 'else')) check(s.else, v, path, out);
    }
  }

  check(schema, instance, '', errors);
  return errors.map((e) => ({ path: e.path.replace(/^\//, ''), message: e.message }));
}
