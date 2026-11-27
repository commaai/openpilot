// file exports for the plot panes
import {download} from "/dom.js";

// the plot's canvases flattened onto its background, with the legend drawn in (curve swatch, name, value)
export function exportPng(canvases, legendRows, bg, fg, font, name) {
  const [first] = canvases, r = devicePixelRatio || 1;
  const out = Object.assign(document.createElement("canvas"), {width: first.width, height: first.height});
  const g = out.getContext("2d");
  g.fillStyle = bg;
  g.fillRect(0, 0, out.width, out.height);
  for (const cv of canvases) g.drawImage(cv, 0, 0);
  g.scale(r, r);
  g.font = font;
  g.textBaseline = "middle";
  const lines = legendRows.map((row) => [row.color, `${row.name}  ${row.value}`]);
  const w = Math.max(0, ...lines.map(([, s]) => g.measureText(s).width)) + 26, x = first.width / r - w - 12;
  lines.forEach(([color, text], k) => {
    const y = 16 + k * 16;
    g.fillStyle = color;
    g.fillRect(x, y - 1, 10, 3);
    g.fillStyle = fg;
    g.fillText(text, x + 16, y);
  });
  out.toBlob((blob) => download(blob, `${name}.png`));
}
