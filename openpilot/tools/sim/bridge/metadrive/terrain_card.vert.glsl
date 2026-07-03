#version 330

// Trivial vertex shader for the flat terrain card used in CI software
// rendering (issue #30693). Replaces terrain.vert.glsl, which exists to
// support panda3d's chunked ShaderTerrainMesh with per-vertex heightfield
// displacement — unnecessary for the flat PG-map terrain.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 terrain_uv;
out vec3 vtx_pos;
out vec4 projecteds[1];

void main() {
  terrain_uv = p3d_MultiTexCoord0;
  vtx_pos = (p3d_ModelMatrix * p3d_Vertex).xyz;
  projecteds[0] = vec4(0.0);
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
