#version 330

uniform sampler2D attribute_tex;
uniform float elevation_texture_ratio;

in vec2 terrain_uv;
in vec3 vtx_pos;
in vec4 projecteds[1];

out vec4 color;

void main() {
  float r_min = (1.0 - 1.0 / elevation_texture_ratio) / 2.0;
  float r_max = r_min + 1.0 / elevation_texture_ratio;
  vec4 attri;
  if (abs(elevation_texture_ratio - 1.0) < 0.001) {
    attri = texture(attribute_tex, terrain_uv);
  } else {
    // The semantic texture describes the smaller map region centered inside
    // the full terrain card. Remap that centered UV interval to [0, 1].
    attri = texture(attribute_tex, terrain_uv * elevation_texture_ratio
                                      - (elevation_texture_ratio - 1.0) / 2.0);
  }

  // Keep the fast CI renderer independent of terrain texture state. The
  // semantic map is enough to provide stable lane/road structure; sampling
  // optional asset textures can fall back to solid white on llvmpipe.
  vec3 diffuse = vec3(0.12, 0.14, 0.12);
  if ((attri.r > 0.01) && terrain_uv.x >= r_min && terrain_uv.y >= r_min && terrain_uv.x <= r_max && terrain_uv.y <= r_max) {
    float value = attri.r;
    if (value < 0.11) {
      diffuse = vec3(0.95, 0.75, 0.05);
    } else if (value < 0.21) {
      diffuse = vec3(0.22, 0.23, 0.25);
    } else if (value < 0.31) {
      diffuse = vec3(0.95);
    } else if (value > 0.3999 && value < 0.760001) {
      diffuse = vec3(0.78);
    }
  }
  color = vec4(diffuse * 0.85, 1.0);
}

