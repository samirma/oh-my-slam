// The inputs of a request as the user chose them (OpForm.chosen: the path each value sent names →
// { name: the file's name, index: its position among the parameter's values, multiple: whether the
// parameter takes several, parameter }), so that a result or a warning names a file the way the
// user knows it, not by the path of its upload.

// An input as the user chose it: its file's name, and its number among the parameter's values
// when it takes several ("input 2": the second, from 1, as a bird's-eye image tags its camera).
export function inputLabel(c) { return c.multiple ? `${c.name} (input ${c.index + 1})` : c.name; }

// A camera of the result by the input it placed, when it placed one the user chose ("input 2
// (located_1)"), else by its name.
export function cameraTitle(name, input) { return input ? `input ${input.index + 1} (${name})` : name; }

// The chosen input a path names: a path the request sent (a workspace path), or the absolute path
// the command was given for it (which ends with it); null when it names none.
export function chosenFor(path, chosen) {
  if (!path || !chosen) return null;
  if (chosen.has(path)) return chosen.get(path);
  for (const [p, c] of chosen) if (String(path).endsWith(`/${p}`)) return c;
  return null;
}

// A warning of the command ({message, inputs: [{path}]}, X-Oms-Warnings) with each input it
// names named as the user chose it.
export function warningText(w, chosen) {
  let text = String(w.message || '');
  for (const i of w.inputs || []) {
    const c = chosenFor(i.path, chosen);
    if (c) text = text.split(i.path).join(inputLabel(c));
  }
  return text;
}
