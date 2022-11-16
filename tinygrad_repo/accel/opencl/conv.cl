//PREFIX

__kernel void image_conv(
  write_only image2d_t output,
  read_only image2d_t input,
  read_only image2d_t weights
#ifndef NOARGS
  ,short numPackedInputChannelsForGroup,
  short totalNumPackedInputChannels,
  short numPackedOutputChannelsForGroup,
  short totalNumPackedOutputChannels,
  short numOutputColumns,
  short numOutputRows, short numInputRows
#endif
  /*short filterSizeX, short filterSizeY,
  short paddingX, short paddingY,
  short strideX, short strideY,
  short dilationX, short dilationY*/
  //ARGS
  ) {

  //SHORTS

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  float4 outputValues[NUM_OUTPUTS];
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  short packedOutputChannel = get_global_id(0);
  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedOutputChannel;

  short groupNum = (packedOutputChannel / numPackedOutputChannelsForGroup);
  short startPackedInputChannel = mul24(groupNum, numPackedInputChannelsForGroup);
  short startOutputColumn = mul24((short)get_global_id(1), NUM_OUTPUTS);
  short startX = mad24(mad24(startOutputColumn, strideX, -paddingX), totalNumPackedInputChannels, startPackedInputChannel);
  short strideWithChannels = mul24(strideX, totalNumPackedInputChannels);

  int outputRow = get_global_id(2);
  int2 inputLocation;

#ifdef BATCH
  // TODO: this doesn't work with y padding
  inputLocation.y = mad24(outputRow % numOutputRows, strideY, -paddingY);
  int batchOffset = (outputRow / numOutputRows) * numInputRows;
  inputLocation.y += batchOffset;
#else
  inputLocation.y = mad24(outputRow, strideY, -paddingY);
#endif

#ifdef DEPTHWISE_UNSTRIDED
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    float4 inputValues[4];
    inputLocation.x = startX;
    for (short i = 1; i < 4; ++i) {
      inputValues[i] = read_imagef(input, smp, INPUT_LOCATION);
      inputLocation.x += totalNumPackedOutputChannels;
    }
    for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {
      inputValues[0] = inputValues[1];
      inputValues[1] = inputValues[2];
      inputValues[2] = inputValues[3];
      inputValues[3] = read_imagef(input, smp, INPUT_LOCATION);
      inputLocation.x += totalNumPackedInputChannels;
      float4 weightValues = read_imagef(weights, smp, WEIGHT_LOCATION);
      ++weightLocation.x;
      outputValues[0] += inputValues[0] * weightValues;
      outputValues[1] += inputValues[1] * weightValues;
      outputValues[2] += inputValues[2] * weightValues;
      outputValues[3] += inputValues[3] * weightValues;
    }
    ++inputLocation.y;
  }
#else

  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    // numPackedInputChannelsForGroup is 1 in depthwise
    for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
      short startXForChannel = startX + packedInputChannel;
      for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {

        short dilatedStepX = mul24(totalNumPackedInputChannels, dilationX);
        inputLocation.x = mad24(rfColumn, dilatedStepX, startXForChannel);
        float4 inputValues[NUM_OUTPUTS];
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          inputValues[i] = read_imagef(input, smp, INPUT_LOCATION);
          inputLocation.x += strideWithChannels;
        }

#ifdef DEPTHWISE
        float4 weightValues = read_imagef(weights, smp, WEIGHT_LOCATION);
        ++weightLocation.x;
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          outputValues[i] += inputValues[i] * weightValues;
        }
#else
        float4 weightValues[4];
        for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
          weightValues[outChIdx] = read_imagef(weights, smp, WEIGHT_LOCATION);
          ++weightLocation.x;
        }

        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          // this is marginally faster than using dot
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
#endif
      }
    }
    inputLocation.y += dilationY;
  }
#endif

  int2 outputLocation;
  outputLocation.y = outputRow;

  // do binops
  short outputColumn = startOutputColumn;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    //BINOP
    ++outputColumn;
  }

  // output to memory
  outputColumn = startOutputColumn;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    if (outputColumn < numOutputColumns) {
      write_imagef(output, OUTPUT_LOCATION, outputValues[i]);
    }
    ++outputColumn;
  }
}
