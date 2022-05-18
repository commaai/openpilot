    read_only image2d_t input,
#ifndef DEPTHWISE
    short startPackedInputChannel,
    short numPackedInputChannelsForGroup, short totalNumPackedInputChannels,
    // typo required for API compatibility
    short packedOuputChannelOffset, short totalNumPackedOutputChannels,
#else
    short totalNumPackedChannels,
#endif
    read_only image2d_t weights, __constant float *biases,
    short filterSizeX, short filterSizeY,
    write_only image2d_t output,
    short paddingX, short paddingY, short strideX, short strideY,
#ifdef SUPPORT_DILATION
    short dilationX, short dilationY,
#endif
    short neuron, float a, float b, float min_clamp, float max_clamp,
#ifndef DEPTHWISE
    // note: these are not supported
    __constant float *parameters, __constant float *batchNormBiases,
#endif
    short numOutputColumns
#ifdef SUPPORT_ACCUMULATION
    , short doAccumulate, read_only image2d_t accumulator
#endif
    ) {

#ifndef NUM_OUTPUTS
  #define NUM_OUTPUTS 4
#endif

  // init
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  short packedOutputChannel = get_global_id(0);
  short startOutputColumn = mul24((short)get_global_id(1), NUM_OUTPUTS);
  short outputRow = get_global_id(2);

#ifdef DEPTHWISE
  short totalNumPackedInputChannels = totalNumPackedChannels;
  short totalNumPackedOutputChannels = totalNumPackedChannels;
  short startPackedInputChannel = packedOutputChannel;
#endif

  short startX = mad24(mad24(startOutputColumn, strideX, -paddingX), totalNumPackedInputChannels, startPackedInputChannel);
  short strideWithChannels = mul24(strideX, totalNumPackedInputChannels);

  float4 outputValues[NUM_OUTPUTS];
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  int2 inputLocation;
  inputLocation.y = mad24(outputRow, strideY, -paddingY);

  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedOutputChannel;

#ifdef DEPTHWISE

#ifdef SUPPORT_DILATION

  // depthwise convolution
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {
      short dilatedStepX = mul24(totalNumPackedChannels, dilationX);
      inputLocation.x = mad24(rfColumn, dilatedStepX, startX);
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

#else

  // depthwise unstrided convolution
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    float4 inputValues[4];
    inputLocation.x = startX;
    for (short i = 1; i < 4; ++i) {
      inputValues[i] = read_imagef(input, smp, inputLocation);
      inputLocation.x += totalNumPackedOutputChannels;
    }
    for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {
      inputValues[0] = inputValues[1];
      inputValues[1] = inputValues[2];
      inputValues[2] = inputValues[3];
      inputValues[3] = read_imagef(input, smp, inputLocation);
      inputLocation.x += totalNumPackedChannels;
      float4 weightValues = read_imagef(weights, smp, weightLocation);
      ++weightLocation.x;
      outputValues[0] += inputValues[0] * weightValues;
      outputValues[1] += inputValues[1] * weightValues;
      outputValues[2] += inputValues[2] * weightValues;
      outputValues[3] += inputValues[3] * weightValues;
    }
    ++inputLocation.y;
  }

#endif

#elif defined(ONLY_1X1_CONV)

  // 1x1 convolution
  short endPackedInputChannel = startPackedInputChannel + numPackedInputChannelsForGroup;
  for (short packedInputChannel = startPackedInputChannel; packedInputChannel < endPackedInputChannel; ++packedInputChannel) {
    float4 weightValues[4];
    for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
      weightValues[outChIdx] = read_imagef(weights, smp, weightLocation);
      ++weightLocation.x;
    }

    inputLocation.x = startX + packedInputChannel;
    float4 inputValues[NUM_OUTPUTS];
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      inputValues[i] = read_imagef(input, smp, inputLocation);
      inputLocation.x += strideWithChannels;
    }

    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      float4 curOutputValues = outputValues[i];
      curOutputValues.x += inputValues[i].x * weightValues[0].x;
      curOutputValues.x += inputValues[i].y * weightValues[0].y;
      curOutputValues.x += inputValues[i].z * weightValues[0].z;
      curOutputValues.x += inputValues[i].w * weightValues[0].w;
      curOutputValues.y += inputValues[i].x * weightValues[1].x;
      curOutputValues.y += inputValues[i].y * weightValues[1].y;
      curOutputValues.y += inputValues[i].z * weightValues[1].z;
      curOutputValues.y += inputValues[i].w * weightValues[1].w;
      curOutputValues.z += inputValues[i].x * weightValues[2].x;
      curOutputValues.z += inputValues[i].y * weightValues[2].y;
      curOutputValues.z += inputValues[i].z * weightValues[2].z;
      curOutputValues.z += inputValues[i].w * weightValues[2].w;
      curOutputValues.w += inputValues[i].x * weightValues[3].x;
      curOutputValues.w += inputValues[i].y * weightValues[3].y;
      curOutputValues.w += inputValues[i].z * weightValues[3].z;
      curOutputValues.w += inputValues[i].w * weightValues[3].w;
      outputValues[i] = curOutputValues;
    }
  }
  packedOutputChannel += packedOuputChannelOffset;

#else

  // normal convolution
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
      short startXForChannel = startX + packedInputChannel;
      for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {

        float4 weightValues[4];
        for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
          weightValues[outChIdx] = read_imagef(weights, smp, weightLocation);
          ++weightLocation.x;
        }

#ifdef SUPPORT_DILATION
        short dilatedStepX = mul24(totalNumPackedInputChannels, dilationX);
        inputLocation.x = mad24(rfColumn, dilatedStepX, startXForChannel);
#else
        inputLocation.x = mad24(rfColumn, totalNumPackedInputChannels, startXForChannel);
#endif
        float4 inputValues[NUM_OUTPUTS];
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          inputValues[i] = read_imagef(input, smp, inputLocation);
          inputLocation.x += strideWithChannels;
        }
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          float4 curOutputValues = outputValues[i];
          curOutputValues.x += inputValues[i].x * weightValues[0].x;
          curOutputValues.x += inputValues[i].y * weightValues[0].y;
          curOutputValues.x += inputValues[i].z * weightValues[0].z;
          curOutputValues.x += inputValues[i].w * weightValues[0].w;
          curOutputValues.y += inputValues[i].x * weightValues[1].x;
          curOutputValues.y += inputValues[i].y * weightValues[1].y;
          curOutputValues.y += inputValues[i].z * weightValues[1].z;
          curOutputValues.y += inputValues[i].w * weightValues[1].w;
          curOutputValues.z += inputValues[i].x * weightValues[2].x;
          curOutputValues.z += inputValues[i].y * weightValues[2].y;
          curOutputValues.z += inputValues[i].z * weightValues[2].z;
          curOutputValues.z += inputValues[i].w * weightValues[2].w;
          curOutputValues.w += inputValues[i].x * weightValues[3].x;
          curOutputValues.w += inputValues[i].y * weightValues[3].y;
          curOutputValues.w += inputValues[i].z * weightValues[3].z;
          curOutputValues.w += inputValues[i].w * weightValues[3].w;
          outputValues[i] = curOutputValues;
        }
      }
    }
#ifdef SUPPORT_DILATION
    inputLocation.y += dilationY;
#else
    ++inputLocation.y;
#endif
  }
  packedOutputChannel += packedOuputChannelOffset;
#endif

  // bias
  short outputChannel = mul24(packedOutputChannel, 4);
  float4 biasValues = vload4(0, biases + outputChannel);
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] += biasValues;
  }

#ifdef SUPPORT_ACCUMULATION
  // accumulate
  if (doAccumulate) {
    int2 outputLocation;
    short outputColumn = startOutputColumn;
    outputLocation.y = outputRow;
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
      if (outputColumn < numOutputColumns) {
        outputValues[i] += read_imagef(accumulator, smp, outputLocation);
      }
      ++outputColumn;
    }
  }
#endif

  // activation
  switch (neuron) {
  case 1:
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputValues[i] = max(outputValues[i], 0.0f);
    }
    break;
  case 2:
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputValues[i] = a * tanh(b * outputValues[i]);
    }
    break;
  case 3:
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputValues[i] = native_recip(1.0f + native_exp(-a * outputValues[i] + b));
    }
    break;
  case 4:
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputValues[i] = max(outputValues[i], min_clamp);
      outputValues[i] = min(outputValues[i], max_clamp);
    }
    break;
  case 5:
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      outputValues[i] = max(outputValues[i], 0.0f) + a * (native_exp(min(outputValues[i], 0.0f)) - 1.0f);
    }
    break;
  }

  // output
  int2 outputLocation;
  short outputColumn = startOutputColumn;
  outputLocation.y = outputRow;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    if (outputColumn < numOutputColumns) {
      write_imagef(output, outputLocation, outputValues[i]);
    }
    ++outputColumn;
  }
}
