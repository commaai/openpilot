
#define PIX_PER_WI_X 1
#define PIX_PER_WI_Y 1

#define scn 3
#define bidx 2
#define uidx 0

#define R_COMP x
#define G_COMP y
#define B_COMP z

__constant float c_RGB2YUVCoeffs_420[8] = { 0.256999969f, 0.50399971f, 0.09799957f, -0.1479988098f, -0.2909994125f,
                                            0.438999176f, -0.3679990768f, -0.0709991455f };

__kernel void RGB2YUV_YV12_IYUV(__global const uchar* srcptr, int src_step, int src_offset,
                                __global uchar* dstptr, int dst_step, int dst_offset,
                                int rows, int cols)
{
    int x = get_global_id(0) * PIX_PER_WI_X;
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols/2)
    {
        int src_index  = mad24(y << 1, src_step, mad24(x << 1, scn, src_offset));
        int ydst_index = mad24(y << 1, dst_step, (x << 1) + dst_offset);
        int y_rows = rows / 3 * 2;
        int vsteps[2] = { cols >> 1, dst_step - (cols >> 1)};
        __constant float* coeffs = c_RGB2YUVCoeffs_420;

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows / 3)
            {
                __global const uchar* src1 = srcptr + src_index;
                __global const uchar* src2 = src1 + src_step;
                __global uchar* ydst1 = dstptr + ydst_index;
                __global uchar* ydst2 = ydst1 + dst_step;

                __global uchar* udst = dstptr + mad24(y_rows + (y>>1), dst_step, dst_offset + (y%2)*(cols >> 1) + x);
                __global uchar* vdst = udst + mad24(y_rows >> 2, dst_step, y_rows % 4 ? vsteps[y%2] : 0);

#if PIX_PER_WI_X == 2
                int s11 = *((__global const int*) src1);
                int s12 = *((__global const int*) src1 + 1);
                int s13 = *((__global const int*) src1 + 2);
#if scn == 4
                int s14 = *((__global const int*) src1 + 3);
#endif
                int s21 = *((__global const int*) src2);
                int s22 = *((__global const int*) src2 + 1);
                int s23 = *((__global const int*) src2 + 2);
#if scn == 4
                int s24 = *((__global const int*) src2 + 3);
#endif
                float src_pix1[scn * 4], src_pix2[scn * 4];

                *((float4*) src_pix1)     = convert_float4(as_uchar4(s11));
                *((float4*) src_pix1 + 1) = convert_float4(as_uchar4(s12));
                *((float4*) src_pix1 + 2) = convert_float4(as_uchar4(s13));
#if scn == 4
                *((float4*) src_pix1 + 3) = convert_float4(as_uchar4(s14));
#endif
                *((float4*) src_pix2)     = convert_float4(as_uchar4(s21));
                *((float4*) src_pix2 + 1) = convert_float4(as_uchar4(s22));
                *((float4*) src_pix2 + 2) = convert_float4(as_uchar4(s23));
#if scn == 4
                *((float4*) src_pix2 + 3) = convert_float4(as_uchar4(s24));
#endif
                uchar4 y1, y2;
                y1.x = convert_uchar_sat(fma(coeffs[0], src_pix1[      2-bidx], fma(coeffs[1], src_pix1[      1], fma(coeffs[2], src_pix1[      bidx], 16.5f))));
                y1.y = convert_uchar_sat(fma(coeffs[0], src_pix1[  scn+2-bidx], fma(coeffs[1], src_pix1[  scn+1], fma(coeffs[2], src_pix1[  scn+bidx], 16.5f))));
                y1.z = convert_uchar_sat(fma(coeffs[0], src_pix1[2*scn+2-bidx], fma(coeffs[1], src_pix1[2*scn+1], fma(coeffs[2], src_pix1[2*scn+bidx], 16.5f))));
                y1.w = convert_uchar_sat(fma(coeffs[0], src_pix1[3*scn+2-bidx], fma(coeffs[1], src_pix1[3*scn+1], fma(coeffs[2], src_pix1[3*scn+bidx], 16.5f))));
                y2.x = convert_uchar_sat(fma(coeffs[0], src_pix2[      2-bidx], fma(coeffs[1], src_pix2[      1], fma(coeffs[2], src_pix2[      bidx], 16.5f))));
                y2.y = convert_uchar_sat(fma(coeffs[0], src_pix2[  scn+2-bidx], fma(coeffs[1], src_pix2[  scn+1], fma(coeffs[2], src_pix2[  scn+bidx], 16.5f))));
                y2.z = convert_uchar_sat(fma(coeffs[0], src_pix2[2*scn+2-bidx], fma(coeffs[1], src_pix2[2*scn+1], fma(coeffs[2], src_pix2[2*scn+bidx], 16.5f))));
                y2.w = convert_uchar_sat(fma(coeffs[0], src_pix2[3*scn+2-bidx], fma(coeffs[1], src_pix2[3*scn+1], fma(coeffs[2], src_pix2[3*scn+bidx], 16.5f))));

                *((__global int*) ydst1) = as_int(y1);
                *((__global int*) ydst2) = as_int(y2);

                float uv[4] = { fma(coeffs[3], src_pix1[      2-bidx], fma(coeffs[4], src_pix1[      1], fma(coeffs[5], src_pix1[      bidx], 128.5f))),
                                fma(coeffs[5], src_pix1[      2-bidx], fma(coeffs[6], src_pix1[      1], fma(coeffs[7], src_pix1[      bidx], 128.5f))),
                                fma(coeffs[3], src_pix1[2*scn+2-bidx], fma(coeffs[4], src_pix1[2*scn+1], fma(coeffs[5], src_pix1[2*scn+bidx], 128.5f))),
                                fma(coeffs[5], src_pix1[2*scn+2-bidx], fma(coeffs[6], src_pix1[2*scn+1], fma(coeffs[7], src_pix1[2*scn+bidx], 128.5f))) };

                udst[0] = convert_uchar_sat(uv[uidx]    );
                vdst[0] = convert_uchar_sat(uv[1 - uidx]);
                udst[1] = convert_uchar_sat(uv[2 + uidx]);
                vdst[1] = convert_uchar_sat(uv[3 - uidx]);
#else
                float4 src_pix1 = convert_float4(vload4(0, src1));
                float4 src_pix2 = convert_float4(vload4(0, src1+scn));
                float4 src_pix3 = convert_float4(vload4(0, src2));
                float4 src_pix4 = convert_float4(vload4(0, src2+scn));

                ydst1[0] = convert_uchar_sat(fma(coeffs[0], src_pix1.R_COMP, fma(coeffs[1], src_pix1.G_COMP, fma(coeffs[2], src_pix1.B_COMP, 16.5f))));
                ydst1[1] = convert_uchar_sat(fma(coeffs[0], src_pix2.R_COMP, fma(coeffs[1], src_pix2.G_COMP, fma(coeffs[2], src_pix2.B_COMP, 16.5f))));
                ydst2[0] = convert_uchar_sat(fma(coeffs[0], src_pix3.R_COMP, fma(coeffs[1], src_pix3.G_COMP, fma(coeffs[2], src_pix3.B_COMP, 16.5f))));
                ydst2[1] = convert_uchar_sat(fma(coeffs[0], src_pix4.R_COMP, fma(coeffs[1], src_pix4.G_COMP, fma(coeffs[2], src_pix4.B_COMP, 16.5f))));

                float uv[2] = { fma(coeffs[3], src_pix1.R_COMP, fma(coeffs[4], src_pix1.G_COMP, fma(coeffs[5], src_pix1.B_COMP, 128.5f))),
                                fma(coeffs[5], src_pix1.R_COMP, fma(coeffs[6], src_pix1.G_COMP, fma(coeffs[7], src_pix1.B_COMP, 128.5f))) };

                udst[0] = convert_uchar_sat(uv[uidx]  );
                vdst[0] = convert_uchar_sat(uv[1-uidx]);
#endif
                ++y;
                src_index += 2*src_step;
                ydst_index += 2*dst_step;
            }
        }
    }
}
