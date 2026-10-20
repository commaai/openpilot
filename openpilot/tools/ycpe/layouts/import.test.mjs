// node --test layouts/import.test.mjs: custom series names made unique across a layout
import assert from "node:assert/strict";
import {test} from "node:test";
import {uniqueCustoms} from "./import.js";

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
