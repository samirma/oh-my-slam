// A DOM element with attributes and children (attributes that are null, undefined or false are
// left out; true gives an empty attribute; "class" sets the class name).
export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  e.append(...children);
  return e;
}
