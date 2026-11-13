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

// restored() runs once a lost context (a GPU process reset) is back, to draw the frame again. The program links in
// the background (KHR_parallel_shader_compile): anything that waits on it would block the main thread for 100-200 ms
// on a cold shader cache, so draw() returns false until it's linked, and the caller tries again next frame.
export function frameRenderer(canvas, restored) {
  const gl = canvas.getContext("webgl2", {alpha: false, antialias: false, depth: false});
  let prog, linked, yuv, size = "", parallel;
  canvas.addEventListener("webglcontextlost", (e) => e.preventDefault());
  canvas.addEventListener("webglcontextrestored", () => { setup(); restored(); });
  setup();

  function setup() {
    parallel = gl.getExtension("KHR_parallel_shader_compile");
    prog = gl.createProgram();
    linked = false;
    for (const [type, src] of [[gl.VERTEX_SHADER, VERT], [gl.FRAGMENT_SHADER, FRAG]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      gl.attachShader(prog, s);
    }
    gl.bindAttribLocation(prog, 0, "p");
    gl.linkProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    for (let i = 0; i < 3; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    size = "";  // new textures need allocating
  }

  function link() {
    if (parallel && !gl.getProgramParameter(prog, parallel.COMPLETION_STATUS_KHR)) return false;
    gl.useProgram(prog);
    ["ty", "tu", "tv"].forEach((name, i) => gl.uniform1i(gl.getUniformLocation(prog, name), i));
    yuv = gl.getUniformLocation(prog, "yuv");
    return linked = true;
  }

  // frame: a VideoFrame, or {width, height, data} with tightly packed I420 planes; false while the program links
  return function draw(frame) {
    if (!linked && !link()) return false;
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
    return true;
  };
}
