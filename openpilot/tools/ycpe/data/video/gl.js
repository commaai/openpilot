// draws decoded frames with WebGL: a VideoFrame uploads as-is, I420 planes from the software decoder become
// RGB in the shader (BT.601, limited range, as the device encodes)
const VERT = `#version 300 es
in vec2 p;
out vec2 uv;
void main() { uv = vec2(p.x, -p.y) * .5 + .5; gl_Position = vec4(p, 0, 1); }`;

const FRAG = `#version 300 es
precision mediump float;
uniform sampler2D ty, tu, tv;
uniform bool yuv;
in vec2 uv;
out vec4 color;
void main() {
  if (!yuv) { color = texture(ty, uv); return; }
  float y = 1.164 * (texture(ty, uv).r - .0625), u = texture(tu, uv).r - .5, v = texture(tv, uv).r - .5;
  color = vec4(y + 1.596 * v, y - .391 * u - .813 * v, y + 2.018 * u, 1);
}`;

// restored() runs once a lost context (a GPU process reset) is back, to draw the frame again
export function frameRenderer(canvas, restored) {
  const gl = canvas.getContext("webgl2", {alpha: false, antialias: false, depth: false});
  let yuv, size = "";
  canvas.addEventListener("webglcontextlost", (e) => e.preventDefault());
  canvas.addEventListener("webglcontextrestored", () => { setup(); restored(); });
  setup();

  function setup() {
    const prog = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VERT], [gl.FRAGMENT_SHADER, FRAG]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      gl.attachShader(prog, s);
    }
    gl.linkProgram(prog);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    ["ty", "tu", "tv"].forEach((name, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(gl.getUniformLocation(prog, name), i);
    });
    yuv = gl.getUniformLocation(prog, "yuv");
    size = "";  // new textures need allocating
  }

  // frame: a VideoFrame, or {width, height, data} with tightly packed I420 planes
  return function draw(frame) {
    const w = frame.displayWidth ?? frame.width, h = frame.displayHeight ?? frame.height;
    const resized = size !== `${w}x${h}`;
    if (resized) [canvas.width, canvas.height, size] = [w, h, `${w}x${h}`];
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0);
    if (frame.data) {
      const cw = (w + 1) >> 1, ch = (h + 1) >> 1;
      const planes = [[w, h, 0], [cw, ch, w * h], [cw, ch, w * h + cw * ch]];
      planes.forEach(([pw, ph, off], i) => {
        gl.activeTexture(gl.TEXTURE0 + i);
        if (resized) gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, pw, ph, 0, gl.RED, gl.UNSIGNED_BYTE, frame.data, off);
        else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, pw, ph, gl.RED, gl.UNSIGNED_BYTE, frame.data, off);
      });
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    }
    gl.uniform1i(yuv, frame.data ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
}
