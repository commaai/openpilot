__kernel void to_image(
    write_only image2d_t out,
    __global const float4 *in,
    int stride) {
  int2 l;
  l.y = get_global_id(1);
  l.x = get_global_id(0);
  //write_imagef(out, l, in[l.y*stride + l.x]);
}