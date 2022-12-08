#pragma once

typedef struct vec3 {
	float v[3];
} vec3;

typedef struct vec4 {
  float v[4];
} vec4;

typedef struct mat3 {
	float v[3*3];
} mat3;

typedef struct mat4 {
  float v[4*4];
} mat4;

static inline mat3 matmul3(const mat3 &a, const mat3 &b) {
  mat3 ret = {{0.0}};
  for (int r=0; r<3; r++) {
    for (int c=0; c<3; c++) {
      float v = 0.0;
      for (int k=0; k<3; k++) {
        v += a.v[r*3+k] * b.v[k*3+c];
      }
      ret.v[r*3+c] = v;
    }
  }
  return ret;
}

static inline vec3 matvecmul3(const mat3 &a, const vec3 &b) {
  vec3 ret = {{0.0}};
  for (int r=0; r<3; r++) {
    for (int c=0; c<3; c++) {
      ret.v[r] += a.v[r*3+c] * b.v[c];
    }
  }
  return ret;
}

static inline mat4 matmul(const mat4 &a, const mat4 &b) {
  mat4 ret = {{0.0}};
  for (int r=0; r<4; r++) {
    for (int c=0; c<4; c++) {
      float v = 0.0;
      for (int k=0; k<4; k++) {
        v += a.v[r*4+k] * b.v[k*4+c];
      }
      ret.v[r*4+c] = v;
    }
  }
  return ret;
}

static inline vec4 matvecmul(const mat4 &a, const vec4 &b) {
  vec4 ret = {{0.0}};
  for (int r=0; r<4; r++) {
    for (int c=0; c<4; c++) {
      ret.v[r] += a.v[r*4+c] * b.v[c];
    }
  }
  return ret;
}

// scales the input and output space of a transformation matrix
// that assumes pixel-center origin.
static inline mat3 transform_scale_buffer(const mat3 &in, float s) {
  // in_pt = ( transform(out_pt/s + 0.5) - 0.5) * s

  mat3 transform_out = {{
    1.0f/s, 0.0f, 0.5f,
    0.0f, 1.0f/s, 0.5f,
    0.0f, 0.0f, 1.0f,
  }};

  mat3 transform_in = {{
    s,  0.0f, -0.5f*s,
    0.0f, s, -0.5f*s,
    0.0f, 0.0f, 1.0f,
  }};

  return matmul3(transform_in, matmul3(in, transform_out));
}
