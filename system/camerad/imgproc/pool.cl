// calculate variance in each subregion
__kernel void var_pool(
  const __global char * input,
  __global ushort * output // should not be larger than 128*128 so uint16
)
{
  const int xidx = get_global_id(0) + ROI_X_MIN;
  const int yidx = get_global_id(1) + ROI_Y_MIN;

  const int size = X_PITCH * Y_PITCH;

  float fsum = 0;
  char mean, max;

  for (int i = 0; i < size; i++) {
    int x_offset = i % X_PITCH;
    int y_offset = i / X_PITCH;
    fsum += input[xidx*X_PITCH + yidx*Y_PITCH*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X];
    max = input[xidx*X_PITCH + yidx*Y_PITCH*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]>max ? input[xidx*X_PITCH + yidx*Y_PITCH*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X]:max;
  }

  mean = convert_char_rte(fsum / size);

  float fvar = 0;
  for (int i = 0; i < size; i++) {
    int x_offset = i % X_PITCH;
    int y_offset = i / X_PITCH;
    fvar += (input[xidx*X_PITCH + yidx*Y_PITCH*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean) * (input[xidx*X_PITCH + yidx*Y_PITCH*FULL_STRIDE_X + x_offset + y_offset*FULL_STRIDE_X] - mean);
  }

  fvar = fvar / size;

  output[(xidx-ROI_X_MIN)+(yidx-ROI_Y_MIN)*(ROI_X_MAX-ROI_X_MIN+1)] = convert_ushort_rte(5 * fvar + convert_float_rte(max));
}