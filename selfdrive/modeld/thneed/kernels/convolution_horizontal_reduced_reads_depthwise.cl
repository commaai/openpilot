__kernel void convolution_horizontal_reduced_reads_depthwise(
    read_only image2d_t input,
    short totalNumPackedChannels,
    read_only image2d_t weights, __constant float *biases,
    short filterSizeX, short filterSizeY,
    write_only image2d_t output,
    short paddingX, short paddingY, short strideX, short strideY,
    short dilationX, short dilationY,
    short neuron, float a, float b, float min_clamp, float max_clamp,
    short numOutputColumns) {

  // init
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  short packedChannel = get_global_id(0);
  short startOutputColumn = mul24((short)get_global_id(1), 4);
  short outputRow = get_global_id(2);
  short startXForChannel = mad24(mad24(startOutputColumn, strideX, -paddingX),
                                 totalNumPackedChannels, packedChannel);
  short strideWithChannels = mul24(strideX, totalNumPackedChannels);

  float4 outputValues[4];
  for (short i = 0; i < 4; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  int2 inputLocation;
  inputLocation.y = mad24(outputRow, strideY, -paddingY);

  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedChannel;

  // convolution
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {
      short dilatedStepX = mul24(totalNumPackedChannels, dilationX);
      inputLocation.x = mad24(rfColumn, dilatedStepX, startXForChannel);
      float4 inputValues[4];
      for (short i = 0; i < 4; ++i) {
        inputValues[i] = read_imagef(input, smp, inputLocation);
        inputLocation.x += strideWithChannels;
      }
      float4 weightValues = read_imagef(weights, smp, weightLocation);
      ++weightLocation.x;
      outputValues[0] += inputValues[0] * weightValues;
      outputValues[1] += inputValues[1] * weightValues;
      outputValues[2] += inputValues[2] * weightValues;
      outputValues[3] += inputValues[3] * weightValues;
    }
    inputLocation.y += dilationY;
  }

  // bias
  short outputChannel = mul24(packedChannel, 4);
  float4 biasValues = vload4(0, biases + outputChannel);
  for (short i = 0; i < 4; ++i) {
    outputValues[i] += biasValues;
  }

  // activation
  switch (neuron) {
  case 1:
    for (short i = 0; i < 4; ++i) {
      outputValues[i] = max(outputValues[i], 0.0f);
    }
    break;
  case 2:
    for (short i = 0; i < 4; ++i) {
      outputValues[i] = a * tanh(b * outputValues[i]);
    }
    break;
  case 3:
    for (short i = 0; i < 4; ++i) {
      outputValues[i] = native_recip(1.0f + native_exp(-a * outputValues[i] + b));
    }
    break;
  case 4:
    for (short i = 0; i < 4; ++i) {
      outputValues[i] = max(outputValues[i], min_clamp);
      outputValues[i] = min(outputValues[i], max_clamp);
    }
    break;
  case 5:
    for (short i = 0; i < 4; ++i) {
      outputValues[i] = max(outputValues[i], 0.0f) + a * (native_exp(min(outputValues[i], 0.0f)) - 1.0f);
    }
    break;
  }

  // output
  int2 outputLocation;
  short outputColumn = startOutputColumn;
  outputLocation.y = outputRow;
  for (short i = 0; i < 4; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedChannels, packedChannel);
    if (outputColumn < numOutputColumns) {
      write_imagef(output, outputLocation, outputValues[i]);
    }
    ++outputColumn;
  }
}
