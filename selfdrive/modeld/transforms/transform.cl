#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

__kernel void warpPerspective(__global const uchar * src,
                              int src_row_stride, int src_px_stride, int src_offset, int src_rows, int src_cols,
                              __global uchar * dst,
                              int dst_row_stride, int dst_offset, int dst_rows, int dst_cols,
                              __constant float * M)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);

        int sx = convert_short_sat(X >> INTER_BITS);
        int sy = convert_short_sat(Y >> INTER_BITS);

        short sx_clamp = clamp(sx, 0, src_cols - 1);
        short sx_p1_clamp = clamp(sx + 1, 0, src_cols - 1);
        short sy_clamp = clamp(sy, 0, src_rows - 1);
        short sy_p1_clamp = clamp(sy + 1, 0, src_rows - 1);
        int v0 = convert_int(src[mad24(sy_clamp, src_row_stride, src_offset + sx_clamp*src_px_stride)]);
        int v1 = convert_int(src[mad24(sy_clamp, src_row_stride, src_offset + sx_p1_clamp*src_px_stride)]);
        int v2 = convert_int(src[mad24(sy_p1_clamp, src_row_stride, src_offset + sx_clamp*src_px_stride)]);
        int v3 = convert_int(src[mad24(sy_p1_clamp, src_row_stride, src_offset + sx_p1_clamp*src_px_stride)]);

        short ay = (short)(Y & (INTER_TAB_SIZE - 1));
        short ax = (short)(X & (INTER_TAB_SIZE - 1));
        float taby = 1.f/INTER_TAB_SIZE*ay;
        float tabx = 1.f/INTER_TAB_SIZE*ax;

        int dst_index = mad24(dy, dst_row_stride, dst_offset + dx);

        int itab0 = convert_short_sat_rte( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab1 = convert_short_sat_rte( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE );
        int itab2 = convert_short_sat_rte( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab3 = convert_short_sat_rte( taby*tabx * INTER_REMAP_COEF_SCALE );

        int val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;

        uchar pix = convert_uchar_sat((val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS);
        dst[dst_index] = pix;
    }
}
