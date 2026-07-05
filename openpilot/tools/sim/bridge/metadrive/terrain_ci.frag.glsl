#version 330

uniform sampler2D yellow_tex;
uniform sampler2D white_tex;
uniform sampler2D road_tex;
uniform sampler2D crosswalk_tex;
uniform sampler2D grass_tex;
uniform float grass_tex_ratio;
uniform sampler2D attribute_tex;
uniform float elevation_texture_ratio;

in vec2 terrain_uv;
in vec3 vtx_pos;
in vec4 projecteds[1];

out vec4 color;

void main() {
  float road_tex_ratio = 128.0;
  float r_min = (1.0 - 1.0 / elevation_texture_ratio) / 2.0;
  float r_max = r_min + 1.0 / elevation_texture_ratio;

  vec4 attri;
  if (abs(elevation_texture_ratio - 1.0) < 0.001) {
    attri = texture(attribute_tex, terrain_uv);
  } else {
    attri = texture(attribute_tex, terrain_uv * elevation_texture_ratio + 0.5);
  }

  vec3 diffuse;
  if ((attri.r > 0.01) && terrain_uv.x >= r_min && terrain_uv.y >= r_min && terrain_uv.x <= r_max && terrain_uv.y <= r_max) {
    float value = attri.r;
    if (value < 0.11) {
      diffuse = texture(yellow_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value < 0.21) {
      diffuse = texture(road_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value < 0.31) {
      diffuse = texture(white_tex, terrain_uv * road_tex_ratio).rgb;
    } else if (value > 0.3999 && value < 0.760001) {
      float theta = (value - 0.39999) * 1000.0 / 180.0 * 3.1415926535;
      vec2 uv2 = vec2(cos(theta) * terrain_uv.x - sin(theta) * terrain_uv.y,
                      sin(theta) * terrain_uv.x + cos(theta) * terrain_uv.y);
      diffuse = texture(crosswalk_tex, uv2 * road_tex_ratio).rgb;
    } else {
      diffuse = texture(white_tex, terrain_uv * road_tex_ratio).rgb;
    }
  } else {
    diffuse = texture(grass_tex, terrain_uv * grass_tex_ratio * 4.0).rgb;
  }

  color = vec4(diffuse * 0.85, 1.0);
}
