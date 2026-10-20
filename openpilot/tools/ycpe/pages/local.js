// /local: open logs that aren't in the API, from files or a folder on this machine (or drop them anywhere)
import {h} from "../dom.js";

export function localPage(open) {
  const picker = (label, attrs) => {
    const input = h("input", {type: "file", multiple: true, hidden: true, ...attrs, onchange: () => open([...input.files])});
    return h("label", {class: "button"}, label, input);
  };
  return h("div", {class: "page-msg"},
    h("h2", {}, "Open local logs"),
    h("p", {class: "muted"}, "rlog, qlog and camera files, or a folder of segments. You can also drop them anywhere on the page."),
    h("p", {}, picker("Choose files", {accept: ".zst,.bz2,.ts,.hevc"}), picker("Choose a folder", {webkitdirectory: true})));
}
