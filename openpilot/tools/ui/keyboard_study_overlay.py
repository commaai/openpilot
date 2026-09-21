"""Render measured touch density over a stock-keyboard image, with an offline viewer."""
import base64
import json
from pathlib import Path

from matplotlib import colormaps, image as mpl_image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from openpilot.tools.ui.keyboard_study_quality import flagged_gesture

WIDTH, HEIGHT = 536, 240
KEYBOARD_TOP = 70
SMOOTHING_PIXELS = 3


def keyboard_points(sessions):
  points = []
  for session in sessions:
    for record in session['records']:
      if record['type'] != 'gesture' or flagged_gesture(record, session['quality']):
        continue
      # Numeric/symbol layers are not visually interchangeable with this image.
      if not any(key['char'] in ('q', 'Q') for key in record['geometry']):
        continue
      labelled = record.get('target') is not None and len(record.get('committed', '')) == 1
      points.append({'x': record['samples'][0][1], 'y': record['samples'][0][2],
                     'technique': session['metadata'].get('technique'), 'session': session['metadata']['id'],
                     'expected': record['expected'].lower() if labelled else None,
                     'wrong': record['committed'] != record['expected'] if labelled else None})
  return points


def make_keyboard_heatmaps(sessions, background_path: Path, output_dir: Path):
  background = mpl_image.imread(background_path)
  assert abs(background.shape[1] / background.shape[0] - WIDTH / HEIGHT) < 0.01, 'Expected a full 536:240 keyboard image'
  points = keyboard_points(sessions)
  titles = {'all': 'All sessions', 'one_index_finger': 'One index finger', 'two_thumbs': 'Two thumbs'}
  filenames = []
  cmap = colormaps.get_cmap('viridis')
  for technique, title in titles.items():
    selected = [point for point in points if technique == 'all' or point['technique'] == technique]
    if not selected:
      continue
    counts, _, _ = np.histogram2d([point['y'] for point in selected], [point['x'] for point in selected],
                                 bins=(HEIGHT, WIDTH), range=((0, HEIGHT), (0, WIDTH)))
    coordinates = np.arange(-3 * SMOOTHING_PIXELS, 3 * SMOOTHING_PIXELS + 1)
    kernel = np.exp(-(coordinates ** 2) / (2 * SMOOTHING_PIXELS ** 2))
    kernel /= kernel.sum()
    density = np.apply_along_axis(np.convolve, 0, counts, kernel, mode='same')
    density = np.apply_along_axis(np.convolve, 1, density, kernel, mode='same')
    relative = density / max(float(density.max()), 1e-12)
    rgba = cmap(relative)
    # Transparency is required here so the requested underlying keys remain visible.
    rgba[..., 3] = np.where(relative > 0.015, 0.82 * np.sqrt(relative), 0)
    figure = Figure(figsize=(14, 5.5), layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.imshow(background, extent=(0, WIDTH, HEIGHT, 0))
    axis.imshow(rgba, extent=(0, WIDTH, HEIGHT, 0), interpolation='bilinear')
    axis.set(xlim=(0, WIDTH), ylim=(HEIGHT, KEYBOARD_TOP), xlabel='screen x (native px)', ylabel='screen y (native px)',
             title=f'{title}: {len(selected):,} first-contact taps over the keyboard')
    figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, label='Relative density within this panel')
    figure.supxlabel('Letter layers only; uppercase/lowercase pooled. Gaussian smoothing: 3 native px. Flagged bursts omitted.', fontsize=10)
    filename = f'keyboard-heatmap-{technique}.png'
    figure.savefig(output_dir / filename, dpi=300)
    filenames.append(filename)
  encoded = base64.b64encode(background_path.read_bytes()).decode()
  palette = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
  payload = json.dumps({'points': points, 'palette': palette}, separators=(',', ':'))
  html = VIEWER.replace('__BACKGROUND__', encoded).replace('__DATA__', payload)
  (output_dir / 'keyboard-heatmap.html').write_text(html)
  (output_dir / 'keyboard-heatmap-points.json').write_text(json.dumps(points, indent=2))
  return filenames


VIEWER = '''<!doctype html><meta charset="utf-8"><title>Keyboard tap heatmap</title>
<style>body{font:17px system-ui;margin:24px;max-width:1400px;background:#161616;color:#eee}
label{display:inline-block;margin:8px 18px 8px 0}select,button{font:inherit;padding:6px}
canvas{display:block;width:100%;height:auto;background:black}p{max-width:1100px;line-height:1.5}a{color:#8ecbff}</style>
<h1>Where taps landed</h1>
<p>Keyboard layout with first-contact touch density. Letter layers only; uppercase and lowercase share positions.
The full session is retained. Automatically flagged rapid out-of-prompt bursts are omitted from these plots.</p>
<label>Technique <select id="technique"><option value="all">Everyone</option>
<option value="one_index_finger">One index finger</option><option value="two_thumbs">Two thumbs</option></select></label>
<label>Intended key <select id="key"><option value="all">All presses (including controls/unlabelled)</option>
<option value="labelled">All inferred character taps</option></select></label>
<label><input type="checkbox" id="dots"> Show exact tap positions</label>
<button id="download">Download current view as PNG</button>
<p id="count"></p><canvas id="view" width="2144" height="680"></canvas>
<p>Density is normalized within the selected view; yellow is densest. Smoothing: 3 native pixels.
Selecting an intended key includes its wrong-letter selections. Intent is inferred only while entered text matches the prompt prefix;
uncertain taps remain in “All presses” but cannot be assigned to a particular intended key.</p>
<p><a href="keyboard-heatmap-all.png">High-resolution overview (4200 px wide)</a> · <a href="report.html">Session metrics and other heatmaps</a></p>
<script>
const data=__DATA__, background=new Image(), canvas=document.getElementById('view'), ctx=canvas.getContext('2d');
const technique=document.getElementById('technique'), key=document.getElementById('key'), dots=document.getElementById('dots');
for (const char of 'abcdefghijklmnopqrstuvwxyz ') {const option=document.createElement('option');
 option.value=char;option.textContent=char===' '?'space':char;key.appendChild(option);}
function render(){
 const points=data.points.filter(p=>(technique.value==='all'||p.technique===technique.value)&&
   (key.value==='all'||(key.value==='labelled'?p.expected!==null:p.expected===key.value)));
 const density=new Float64Array(536*240), sigma=3, radius=9;
 for(const p of points){const x=Math.floor(p.x), y=Math.floor(p.y);
  for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){
   const px=x+dx,py=y+dy;if(px>=0&&px<536&&py>=0&&py<240)density[py*536+px]+=Math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma));}}
 let maximum=0;for(const value of density)maximum=Math.max(maximum,value);
 const heat=document.createElement('canvas');heat.width=536;heat.height=240;
 const hctx=heat.getContext('2d'), pixels=hctx.createImageData(536,240);
 for(let i=0;i<density.length;i++){const value=maximum?density[i]/maximum:0,color=data.palette[Math.min(255,Math.floor(value*255))];
  pixels.data[i*4]=color[0];pixels.data[i*4+1]=color[1];pixels.data[i*4+2]=color[2];pixels.data[i*4+3]=value>0.015?Math.round(255*0.82*Math.sqrt(value)):0;}
 hctx.putImageData(pixels,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.drawImage(background,0,background.naturalHeight*70/240,background.naturalWidth,background.naturalHeight*170/240,0,0,2144,680);ctx.drawImage(heat,0,70,536,170,0,0,2144,680);
 if(dots.checked){ctx.fillStyle='white';for(const p of points){ctx.beginPath();ctx.arc(p.x*4,(p.y-70)*4,3,0,Math.PI*2);ctx.fill();}}
 const labelled=points.filter(p=>p.expected!==null), wrong=labelled.filter(p=>p.wrong).length;
 document.getElementById('count').textContent=`${points.length} taps across ${new Set(points.map(p=>p.session)).size} sessions. ` +
 `${labelled.length} inferred character targets; ${wrong} selected a different character.`;
}
technique.onchange=key.onchange=dots.onchange=render;
document.getElementById('download').onclick=()=>{const link=document.createElement('a');link.download='keyboard-tap-heatmap.png';
 link.href=canvas.toDataURL();link.click();};
background.onload=render;background.src='data:image/png;base64,__BACKGROUND__';
</script>'''
