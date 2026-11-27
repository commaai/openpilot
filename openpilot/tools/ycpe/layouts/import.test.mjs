// node --test layouts/import.test.mjs: jotpluggler custom series, and custom series names made unique across a layout
import assert from "node:assert/strict";
import {test} from "node:test";
import {importLayout, uniqueCustoms} from "./import.js";

const custom = (name, code, inputs) => ({path: `/custom/${name}`, label: name, custom: {inputs, globals: "", code}});
const layout = (...plots) => ({active: 0, tabs: [{name: "t", root: {split: "col", sizes: plots.map(() => 1 / plots.length),
  children: plots.map((curves) => ({pane: "plot", state: {curves}}))}}]});
const curvesOf = (l) => l.tabs[0].root.children.map((n) => n.state.curves);

test("a reader follows the definition in its own plot", () => {
  const l = uniqueCustoms(layout(
    [custom("x", "return value", ["/carState/vEgo"]), custom("y", "return value * 2", ["/custom/x"])],
    [custom("x", "return -value", ["/carState/vEgo"]), {path: "/custom/x"}]));
  const [a, b] = curvesOf(l);
  assert.equal(a[0].path, "/custom/x");
  assert.deepEqual(a[1].custom.inputs, ["/custom/x"]);
  assert.equal(b[0].path, "/custom/x_2");
  assert.equal(b[0].label, "x_2");
  assert.equal(b[1].path, "/custom/x_2");
});

test("elsewhere, a reader follows the layout's only definition", () => {
  const taken = new Map([["/custom/x", "other code"]]);
  const l = uniqueCustoms(layout([custom("x", "return value", ["/carState/vEgo"])], [{path: "/custom/x"}]), taken);
  const [a, b] = curvesOf(l);
  assert.equal(a[0].path, "/custom/x_2");
  assert.equal(b[0].path, "/custom/x_2");
});

test("a jotpluggler Python series keeps its inputs and name, its code commented out", () => {
  const curve = {name: "/a/b", custom_python: {linked_source: "/carState/vEgo", additional_sources: ["/carState/__valid"], globals_code: "",
    function_code: "def f(time, value):\n    return value"}};
  const {layout, warnings} = importLayout(JSON.stringify({tabs: [{name: "t", root: {curves: [curve]}}]}));
  const [c] = layout.tabs[0].root.state.curves;
  assert.equal(c.path, "/custom/_a_b");
  assert.deepEqual(c.custom, {inputs: ["/carState/vEgo", "/carState/valid"], globals: "", code: "// def f(time, value):\n//     return value", untrusted: true});
  assert.equal(warnings.length, 1);
});
