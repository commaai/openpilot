// const __constant float3 rgb_weights = (0.299, 0.587, 0.114); // opencv rgb2gray weights
// const __constant float3 bgr_weights = (0.114, 0.587, 0.299); // bgr2gray weights

// convert input rgb image to single channel then conv
__kernel void rgb2gray_conv2d(
  const __global uchar * input, 
  __global char * output,
  __constant char * filter,
  __local char3 * cached
)
{
  const int rowOffset = get_global_id(1) * IMAGE_W;
  const int my = get_global_id(0) + rowOffset;

  const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);
  const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen;
  const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;

  // cache local pixels
  cached[ myLocal ].x = convert_char_rte(input[ my * 3 ]/2); // r
  cached[ myLocal ].y = convert_char_rte(input[ my * 3 + 1]/2); // g
  cached[ myLocal ].z = convert_char_rte(input[ my * 3 + 2]/2); // b

  // pad
  if (
    get_global_id(0) < HALF_FILTER_SIZE       || 
    get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1   || 
    get_global_id(1) < HALF_FILTER_SIZE     ||
    get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
  )
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  }
  else 
  {
    int localColOffset = -1;
    int globalColOffset = -1;
    
    // cache extra
    if ( get_local_id(0) < HALF_FILTER_SIZE )
    {
      localColOffset = get_local_id(0);
      globalColOffset = -HALF_FILTER_SIZE;

      cached[ localRowOffset + get_local_id(0) ].x = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE * 3 ]/2);
      cached[ localRowOffset + get_local_id(0) ].y = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE * 3 + 1]/2);
      cached[ localRowOffset + get_local_id(0) ].z = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE * 3 + 2]/2);
    }
    else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
    {
      localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
      globalColOffset = HALF_FILTER_SIZE;
      
      cached[ myLocal + HALF_FILTER_SIZE ].x = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE * 3 ]/2);
      cached[ myLocal + HALF_FILTER_SIZE ].y = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE * 3 + 1]/2);
      cached[ myLocal + HALF_FILTER_SIZE ].z = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE * 3 + 2]/2);
    }


    if ( get_local_id(1) < HALF_FILTER_SIZE )
    {
      cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ].x = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 ]/2);
      cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ].y = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 + 1]/2);
      cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ].z = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 + 2]/2);
      if (localColOffset > 0)
      {
        cached[ get_local_id(1) * localRowLen + localColOffset ].x = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3]/2);
        cached[ get_local_id(1) * localRowLen + localColOffset ].y = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3 + 1]/2);
        cached[ get_local_id(1) * localRowLen + localColOffset ].z = convert_char_rte(input[ my * 3 - HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3 + 2]/2);
      }
    }
    else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
    {
      int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
      cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ].x = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 ]/2);
      cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ].y = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 + 1]/2);
      cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ].z = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 + 2]/2);
      if (localColOffset > 0)
      {
        cached[ offset + localColOffset ].x = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3]/2);
        cached[ offset + localColOffset ].y = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3 + 1]/2);
        cached[ offset + localColOffset ].z = convert_char_rte(input[ my * 3 + HALF_FILTER_SIZE_IMAGE_W * 3 + globalColOffset * 3 + 2]/2);
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform convolution
    int fIndex = 0;
    char sum = 0;

    for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
    {
      int curRow = r * localRowLen;
      for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)
      {
        if (!FLIP_RB){
          // sum += dot(rgb_weights, cached[ myLocal + curRow + c ]) * filter[ fIndex ];
          sum += (cached[ myLocal + curRow + c ].x / 3 + cached[ myLocal + curRow + c ].y / 2 + cached[ myLocal + curRow + c ].z / 9) / 4 * filter[ fIndex ];
        } else {
          // sum += dot(bgr_weights, cached[ myLocal + curRow + c ]) * filter[ fIndex ];
          sum += (cached[ myLocal + curRow + c ].x / 9 + cached[ myLocal + curRow + c ].y / 2 + cached[ myLocal + curRow + c ].z / 3) / 4 * filter[ fIndex ];
        }
      }
    }
    output[my] = sum;
  }
}