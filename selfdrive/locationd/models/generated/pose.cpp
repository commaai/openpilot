#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.13.2                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_8287156320329551527) {
   out_8287156320329551527[0] = delta_x[0] + nom_x[0];
   out_8287156320329551527[1] = delta_x[1] + nom_x[1];
   out_8287156320329551527[2] = delta_x[2] + nom_x[2];
   out_8287156320329551527[3] = delta_x[3] + nom_x[3];
   out_8287156320329551527[4] = delta_x[4] + nom_x[4];
   out_8287156320329551527[5] = delta_x[5] + nom_x[5];
   out_8287156320329551527[6] = delta_x[6] + nom_x[6];
   out_8287156320329551527[7] = delta_x[7] + nom_x[7];
   out_8287156320329551527[8] = delta_x[8] + nom_x[8];
   out_8287156320329551527[9] = delta_x[9] + nom_x[9];
   out_8287156320329551527[10] = delta_x[10] + nom_x[10];
   out_8287156320329551527[11] = delta_x[11] + nom_x[11];
   out_8287156320329551527[12] = delta_x[12] + nom_x[12];
   out_8287156320329551527[13] = delta_x[13] + nom_x[13];
   out_8287156320329551527[14] = delta_x[14] + nom_x[14];
   out_8287156320329551527[15] = delta_x[15] + nom_x[15];
   out_8287156320329551527[16] = delta_x[16] + nom_x[16];
   out_8287156320329551527[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_152026319407410727) {
   out_152026319407410727[0] = -nom_x[0] + true_x[0];
   out_152026319407410727[1] = -nom_x[1] + true_x[1];
   out_152026319407410727[2] = -nom_x[2] + true_x[2];
   out_152026319407410727[3] = -nom_x[3] + true_x[3];
   out_152026319407410727[4] = -nom_x[4] + true_x[4];
   out_152026319407410727[5] = -nom_x[5] + true_x[5];
   out_152026319407410727[6] = -nom_x[6] + true_x[6];
   out_152026319407410727[7] = -nom_x[7] + true_x[7];
   out_152026319407410727[8] = -nom_x[8] + true_x[8];
   out_152026319407410727[9] = -nom_x[9] + true_x[9];
   out_152026319407410727[10] = -nom_x[10] + true_x[10];
   out_152026319407410727[11] = -nom_x[11] + true_x[11];
   out_152026319407410727[12] = -nom_x[12] + true_x[12];
   out_152026319407410727[13] = -nom_x[13] + true_x[13];
   out_152026319407410727[14] = -nom_x[14] + true_x[14];
   out_152026319407410727[15] = -nom_x[15] + true_x[15];
   out_152026319407410727[16] = -nom_x[16] + true_x[16];
   out_152026319407410727[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3515917231783269738) {
   out_3515917231783269738[0] = 1.0;
   out_3515917231783269738[1] = 0.0;
   out_3515917231783269738[2] = 0.0;
   out_3515917231783269738[3] = 0.0;
   out_3515917231783269738[4] = 0.0;
   out_3515917231783269738[5] = 0.0;
   out_3515917231783269738[6] = 0.0;
   out_3515917231783269738[7] = 0.0;
   out_3515917231783269738[8] = 0.0;
   out_3515917231783269738[9] = 0.0;
   out_3515917231783269738[10] = 0.0;
   out_3515917231783269738[11] = 0.0;
   out_3515917231783269738[12] = 0.0;
   out_3515917231783269738[13] = 0.0;
   out_3515917231783269738[14] = 0.0;
   out_3515917231783269738[15] = 0.0;
   out_3515917231783269738[16] = 0.0;
   out_3515917231783269738[17] = 0.0;
   out_3515917231783269738[18] = 0.0;
   out_3515917231783269738[19] = 1.0;
   out_3515917231783269738[20] = 0.0;
   out_3515917231783269738[21] = 0.0;
   out_3515917231783269738[22] = 0.0;
   out_3515917231783269738[23] = 0.0;
   out_3515917231783269738[24] = 0.0;
   out_3515917231783269738[25] = 0.0;
   out_3515917231783269738[26] = 0.0;
   out_3515917231783269738[27] = 0.0;
   out_3515917231783269738[28] = 0.0;
   out_3515917231783269738[29] = 0.0;
   out_3515917231783269738[30] = 0.0;
   out_3515917231783269738[31] = 0.0;
   out_3515917231783269738[32] = 0.0;
   out_3515917231783269738[33] = 0.0;
   out_3515917231783269738[34] = 0.0;
   out_3515917231783269738[35] = 0.0;
   out_3515917231783269738[36] = 0.0;
   out_3515917231783269738[37] = 0.0;
   out_3515917231783269738[38] = 1.0;
   out_3515917231783269738[39] = 0.0;
   out_3515917231783269738[40] = 0.0;
   out_3515917231783269738[41] = 0.0;
   out_3515917231783269738[42] = 0.0;
   out_3515917231783269738[43] = 0.0;
   out_3515917231783269738[44] = 0.0;
   out_3515917231783269738[45] = 0.0;
   out_3515917231783269738[46] = 0.0;
   out_3515917231783269738[47] = 0.0;
   out_3515917231783269738[48] = 0.0;
   out_3515917231783269738[49] = 0.0;
   out_3515917231783269738[50] = 0.0;
   out_3515917231783269738[51] = 0.0;
   out_3515917231783269738[52] = 0.0;
   out_3515917231783269738[53] = 0.0;
   out_3515917231783269738[54] = 0.0;
   out_3515917231783269738[55] = 0.0;
   out_3515917231783269738[56] = 0.0;
   out_3515917231783269738[57] = 1.0;
   out_3515917231783269738[58] = 0.0;
   out_3515917231783269738[59] = 0.0;
   out_3515917231783269738[60] = 0.0;
   out_3515917231783269738[61] = 0.0;
   out_3515917231783269738[62] = 0.0;
   out_3515917231783269738[63] = 0.0;
   out_3515917231783269738[64] = 0.0;
   out_3515917231783269738[65] = 0.0;
   out_3515917231783269738[66] = 0.0;
   out_3515917231783269738[67] = 0.0;
   out_3515917231783269738[68] = 0.0;
   out_3515917231783269738[69] = 0.0;
   out_3515917231783269738[70] = 0.0;
   out_3515917231783269738[71] = 0.0;
   out_3515917231783269738[72] = 0.0;
   out_3515917231783269738[73] = 0.0;
   out_3515917231783269738[74] = 0.0;
   out_3515917231783269738[75] = 0.0;
   out_3515917231783269738[76] = 1.0;
   out_3515917231783269738[77] = 0.0;
   out_3515917231783269738[78] = 0.0;
   out_3515917231783269738[79] = 0.0;
   out_3515917231783269738[80] = 0.0;
   out_3515917231783269738[81] = 0.0;
   out_3515917231783269738[82] = 0.0;
   out_3515917231783269738[83] = 0.0;
   out_3515917231783269738[84] = 0.0;
   out_3515917231783269738[85] = 0.0;
   out_3515917231783269738[86] = 0.0;
   out_3515917231783269738[87] = 0.0;
   out_3515917231783269738[88] = 0.0;
   out_3515917231783269738[89] = 0.0;
   out_3515917231783269738[90] = 0.0;
   out_3515917231783269738[91] = 0.0;
   out_3515917231783269738[92] = 0.0;
   out_3515917231783269738[93] = 0.0;
   out_3515917231783269738[94] = 0.0;
   out_3515917231783269738[95] = 1.0;
   out_3515917231783269738[96] = 0.0;
   out_3515917231783269738[97] = 0.0;
   out_3515917231783269738[98] = 0.0;
   out_3515917231783269738[99] = 0.0;
   out_3515917231783269738[100] = 0.0;
   out_3515917231783269738[101] = 0.0;
   out_3515917231783269738[102] = 0.0;
   out_3515917231783269738[103] = 0.0;
   out_3515917231783269738[104] = 0.0;
   out_3515917231783269738[105] = 0.0;
   out_3515917231783269738[106] = 0.0;
   out_3515917231783269738[107] = 0.0;
   out_3515917231783269738[108] = 0.0;
   out_3515917231783269738[109] = 0.0;
   out_3515917231783269738[110] = 0.0;
   out_3515917231783269738[111] = 0.0;
   out_3515917231783269738[112] = 0.0;
   out_3515917231783269738[113] = 0.0;
   out_3515917231783269738[114] = 1.0;
   out_3515917231783269738[115] = 0.0;
   out_3515917231783269738[116] = 0.0;
   out_3515917231783269738[117] = 0.0;
   out_3515917231783269738[118] = 0.0;
   out_3515917231783269738[119] = 0.0;
   out_3515917231783269738[120] = 0.0;
   out_3515917231783269738[121] = 0.0;
   out_3515917231783269738[122] = 0.0;
   out_3515917231783269738[123] = 0.0;
   out_3515917231783269738[124] = 0.0;
   out_3515917231783269738[125] = 0.0;
   out_3515917231783269738[126] = 0.0;
   out_3515917231783269738[127] = 0.0;
   out_3515917231783269738[128] = 0.0;
   out_3515917231783269738[129] = 0.0;
   out_3515917231783269738[130] = 0.0;
   out_3515917231783269738[131] = 0.0;
   out_3515917231783269738[132] = 0.0;
   out_3515917231783269738[133] = 1.0;
   out_3515917231783269738[134] = 0.0;
   out_3515917231783269738[135] = 0.0;
   out_3515917231783269738[136] = 0.0;
   out_3515917231783269738[137] = 0.0;
   out_3515917231783269738[138] = 0.0;
   out_3515917231783269738[139] = 0.0;
   out_3515917231783269738[140] = 0.0;
   out_3515917231783269738[141] = 0.0;
   out_3515917231783269738[142] = 0.0;
   out_3515917231783269738[143] = 0.0;
   out_3515917231783269738[144] = 0.0;
   out_3515917231783269738[145] = 0.0;
   out_3515917231783269738[146] = 0.0;
   out_3515917231783269738[147] = 0.0;
   out_3515917231783269738[148] = 0.0;
   out_3515917231783269738[149] = 0.0;
   out_3515917231783269738[150] = 0.0;
   out_3515917231783269738[151] = 0.0;
   out_3515917231783269738[152] = 1.0;
   out_3515917231783269738[153] = 0.0;
   out_3515917231783269738[154] = 0.0;
   out_3515917231783269738[155] = 0.0;
   out_3515917231783269738[156] = 0.0;
   out_3515917231783269738[157] = 0.0;
   out_3515917231783269738[158] = 0.0;
   out_3515917231783269738[159] = 0.0;
   out_3515917231783269738[160] = 0.0;
   out_3515917231783269738[161] = 0.0;
   out_3515917231783269738[162] = 0.0;
   out_3515917231783269738[163] = 0.0;
   out_3515917231783269738[164] = 0.0;
   out_3515917231783269738[165] = 0.0;
   out_3515917231783269738[166] = 0.0;
   out_3515917231783269738[167] = 0.0;
   out_3515917231783269738[168] = 0.0;
   out_3515917231783269738[169] = 0.0;
   out_3515917231783269738[170] = 0.0;
   out_3515917231783269738[171] = 1.0;
   out_3515917231783269738[172] = 0.0;
   out_3515917231783269738[173] = 0.0;
   out_3515917231783269738[174] = 0.0;
   out_3515917231783269738[175] = 0.0;
   out_3515917231783269738[176] = 0.0;
   out_3515917231783269738[177] = 0.0;
   out_3515917231783269738[178] = 0.0;
   out_3515917231783269738[179] = 0.0;
   out_3515917231783269738[180] = 0.0;
   out_3515917231783269738[181] = 0.0;
   out_3515917231783269738[182] = 0.0;
   out_3515917231783269738[183] = 0.0;
   out_3515917231783269738[184] = 0.0;
   out_3515917231783269738[185] = 0.0;
   out_3515917231783269738[186] = 0.0;
   out_3515917231783269738[187] = 0.0;
   out_3515917231783269738[188] = 0.0;
   out_3515917231783269738[189] = 0.0;
   out_3515917231783269738[190] = 1.0;
   out_3515917231783269738[191] = 0.0;
   out_3515917231783269738[192] = 0.0;
   out_3515917231783269738[193] = 0.0;
   out_3515917231783269738[194] = 0.0;
   out_3515917231783269738[195] = 0.0;
   out_3515917231783269738[196] = 0.0;
   out_3515917231783269738[197] = 0.0;
   out_3515917231783269738[198] = 0.0;
   out_3515917231783269738[199] = 0.0;
   out_3515917231783269738[200] = 0.0;
   out_3515917231783269738[201] = 0.0;
   out_3515917231783269738[202] = 0.0;
   out_3515917231783269738[203] = 0.0;
   out_3515917231783269738[204] = 0.0;
   out_3515917231783269738[205] = 0.0;
   out_3515917231783269738[206] = 0.0;
   out_3515917231783269738[207] = 0.0;
   out_3515917231783269738[208] = 0.0;
   out_3515917231783269738[209] = 1.0;
   out_3515917231783269738[210] = 0.0;
   out_3515917231783269738[211] = 0.0;
   out_3515917231783269738[212] = 0.0;
   out_3515917231783269738[213] = 0.0;
   out_3515917231783269738[214] = 0.0;
   out_3515917231783269738[215] = 0.0;
   out_3515917231783269738[216] = 0.0;
   out_3515917231783269738[217] = 0.0;
   out_3515917231783269738[218] = 0.0;
   out_3515917231783269738[219] = 0.0;
   out_3515917231783269738[220] = 0.0;
   out_3515917231783269738[221] = 0.0;
   out_3515917231783269738[222] = 0.0;
   out_3515917231783269738[223] = 0.0;
   out_3515917231783269738[224] = 0.0;
   out_3515917231783269738[225] = 0.0;
   out_3515917231783269738[226] = 0.0;
   out_3515917231783269738[227] = 0.0;
   out_3515917231783269738[228] = 1.0;
   out_3515917231783269738[229] = 0.0;
   out_3515917231783269738[230] = 0.0;
   out_3515917231783269738[231] = 0.0;
   out_3515917231783269738[232] = 0.0;
   out_3515917231783269738[233] = 0.0;
   out_3515917231783269738[234] = 0.0;
   out_3515917231783269738[235] = 0.0;
   out_3515917231783269738[236] = 0.0;
   out_3515917231783269738[237] = 0.0;
   out_3515917231783269738[238] = 0.0;
   out_3515917231783269738[239] = 0.0;
   out_3515917231783269738[240] = 0.0;
   out_3515917231783269738[241] = 0.0;
   out_3515917231783269738[242] = 0.0;
   out_3515917231783269738[243] = 0.0;
   out_3515917231783269738[244] = 0.0;
   out_3515917231783269738[245] = 0.0;
   out_3515917231783269738[246] = 0.0;
   out_3515917231783269738[247] = 1.0;
   out_3515917231783269738[248] = 0.0;
   out_3515917231783269738[249] = 0.0;
   out_3515917231783269738[250] = 0.0;
   out_3515917231783269738[251] = 0.0;
   out_3515917231783269738[252] = 0.0;
   out_3515917231783269738[253] = 0.0;
   out_3515917231783269738[254] = 0.0;
   out_3515917231783269738[255] = 0.0;
   out_3515917231783269738[256] = 0.0;
   out_3515917231783269738[257] = 0.0;
   out_3515917231783269738[258] = 0.0;
   out_3515917231783269738[259] = 0.0;
   out_3515917231783269738[260] = 0.0;
   out_3515917231783269738[261] = 0.0;
   out_3515917231783269738[262] = 0.0;
   out_3515917231783269738[263] = 0.0;
   out_3515917231783269738[264] = 0.0;
   out_3515917231783269738[265] = 0.0;
   out_3515917231783269738[266] = 1.0;
   out_3515917231783269738[267] = 0.0;
   out_3515917231783269738[268] = 0.0;
   out_3515917231783269738[269] = 0.0;
   out_3515917231783269738[270] = 0.0;
   out_3515917231783269738[271] = 0.0;
   out_3515917231783269738[272] = 0.0;
   out_3515917231783269738[273] = 0.0;
   out_3515917231783269738[274] = 0.0;
   out_3515917231783269738[275] = 0.0;
   out_3515917231783269738[276] = 0.0;
   out_3515917231783269738[277] = 0.0;
   out_3515917231783269738[278] = 0.0;
   out_3515917231783269738[279] = 0.0;
   out_3515917231783269738[280] = 0.0;
   out_3515917231783269738[281] = 0.0;
   out_3515917231783269738[282] = 0.0;
   out_3515917231783269738[283] = 0.0;
   out_3515917231783269738[284] = 0.0;
   out_3515917231783269738[285] = 1.0;
   out_3515917231783269738[286] = 0.0;
   out_3515917231783269738[287] = 0.0;
   out_3515917231783269738[288] = 0.0;
   out_3515917231783269738[289] = 0.0;
   out_3515917231783269738[290] = 0.0;
   out_3515917231783269738[291] = 0.0;
   out_3515917231783269738[292] = 0.0;
   out_3515917231783269738[293] = 0.0;
   out_3515917231783269738[294] = 0.0;
   out_3515917231783269738[295] = 0.0;
   out_3515917231783269738[296] = 0.0;
   out_3515917231783269738[297] = 0.0;
   out_3515917231783269738[298] = 0.0;
   out_3515917231783269738[299] = 0.0;
   out_3515917231783269738[300] = 0.0;
   out_3515917231783269738[301] = 0.0;
   out_3515917231783269738[302] = 0.0;
   out_3515917231783269738[303] = 0.0;
   out_3515917231783269738[304] = 1.0;
   out_3515917231783269738[305] = 0.0;
   out_3515917231783269738[306] = 0.0;
   out_3515917231783269738[307] = 0.0;
   out_3515917231783269738[308] = 0.0;
   out_3515917231783269738[309] = 0.0;
   out_3515917231783269738[310] = 0.0;
   out_3515917231783269738[311] = 0.0;
   out_3515917231783269738[312] = 0.0;
   out_3515917231783269738[313] = 0.0;
   out_3515917231783269738[314] = 0.0;
   out_3515917231783269738[315] = 0.0;
   out_3515917231783269738[316] = 0.0;
   out_3515917231783269738[317] = 0.0;
   out_3515917231783269738[318] = 0.0;
   out_3515917231783269738[319] = 0.0;
   out_3515917231783269738[320] = 0.0;
   out_3515917231783269738[321] = 0.0;
   out_3515917231783269738[322] = 0.0;
   out_3515917231783269738[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8327036891744297669) {
   out_8327036891744297669[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8327036891744297669[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8327036891744297669[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8327036891744297669[3] = dt*state[12] + state[3];
   out_8327036891744297669[4] = dt*state[13] + state[4];
   out_8327036891744297669[5] = dt*state[14] + state[5];
   out_8327036891744297669[6] = state[6];
   out_8327036891744297669[7] = state[7];
   out_8327036891744297669[8] = state[8];
   out_8327036891744297669[9] = state[9];
   out_8327036891744297669[10] = state[10];
   out_8327036891744297669[11] = state[11];
   out_8327036891744297669[12] = state[12];
   out_8327036891744297669[13] = state[13];
   out_8327036891744297669[14] = state[14];
   out_8327036891744297669[15] = state[15];
   out_8327036891744297669[16] = state[16];
   out_8327036891744297669[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6376990126270885036) {
   out_6376990126270885036[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6376990126270885036[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6376990126270885036[2] = 0;
   out_6376990126270885036[3] = 0;
   out_6376990126270885036[4] = 0;
   out_6376990126270885036[5] = 0;
   out_6376990126270885036[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6376990126270885036[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6376990126270885036[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6376990126270885036[9] = 0;
   out_6376990126270885036[10] = 0;
   out_6376990126270885036[11] = 0;
   out_6376990126270885036[12] = 0;
   out_6376990126270885036[13] = 0;
   out_6376990126270885036[14] = 0;
   out_6376990126270885036[15] = 0;
   out_6376990126270885036[16] = 0;
   out_6376990126270885036[17] = 0;
   out_6376990126270885036[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6376990126270885036[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6376990126270885036[20] = 0;
   out_6376990126270885036[21] = 0;
   out_6376990126270885036[22] = 0;
   out_6376990126270885036[23] = 0;
   out_6376990126270885036[24] = 0;
   out_6376990126270885036[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6376990126270885036[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6376990126270885036[27] = 0;
   out_6376990126270885036[28] = 0;
   out_6376990126270885036[29] = 0;
   out_6376990126270885036[30] = 0;
   out_6376990126270885036[31] = 0;
   out_6376990126270885036[32] = 0;
   out_6376990126270885036[33] = 0;
   out_6376990126270885036[34] = 0;
   out_6376990126270885036[35] = 0;
   out_6376990126270885036[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6376990126270885036[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6376990126270885036[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6376990126270885036[39] = 0;
   out_6376990126270885036[40] = 0;
   out_6376990126270885036[41] = 0;
   out_6376990126270885036[42] = 0;
   out_6376990126270885036[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6376990126270885036[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6376990126270885036[45] = 0;
   out_6376990126270885036[46] = 0;
   out_6376990126270885036[47] = 0;
   out_6376990126270885036[48] = 0;
   out_6376990126270885036[49] = 0;
   out_6376990126270885036[50] = 0;
   out_6376990126270885036[51] = 0;
   out_6376990126270885036[52] = 0;
   out_6376990126270885036[53] = 0;
   out_6376990126270885036[54] = 0;
   out_6376990126270885036[55] = 0;
   out_6376990126270885036[56] = 0;
   out_6376990126270885036[57] = 1;
   out_6376990126270885036[58] = 0;
   out_6376990126270885036[59] = 0;
   out_6376990126270885036[60] = 0;
   out_6376990126270885036[61] = 0;
   out_6376990126270885036[62] = 0;
   out_6376990126270885036[63] = 0;
   out_6376990126270885036[64] = 0;
   out_6376990126270885036[65] = 0;
   out_6376990126270885036[66] = dt;
   out_6376990126270885036[67] = 0;
   out_6376990126270885036[68] = 0;
   out_6376990126270885036[69] = 0;
   out_6376990126270885036[70] = 0;
   out_6376990126270885036[71] = 0;
   out_6376990126270885036[72] = 0;
   out_6376990126270885036[73] = 0;
   out_6376990126270885036[74] = 0;
   out_6376990126270885036[75] = 0;
   out_6376990126270885036[76] = 1;
   out_6376990126270885036[77] = 0;
   out_6376990126270885036[78] = 0;
   out_6376990126270885036[79] = 0;
   out_6376990126270885036[80] = 0;
   out_6376990126270885036[81] = 0;
   out_6376990126270885036[82] = 0;
   out_6376990126270885036[83] = 0;
   out_6376990126270885036[84] = 0;
   out_6376990126270885036[85] = dt;
   out_6376990126270885036[86] = 0;
   out_6376990126270885036[87] = 0;
   out_6376990126270885036[88] = 0;
   out_6376990126270885036[89] = 0;
   out_6376990126270885036[90] = 0;
   out_6376990126270885036[91] = 0;
   out_6376990126270885036[92] = 0;
   out_6376990126270885036[93] = 0;
   out_6376990126270885036[94] = 0;
   out_6376990126270885036[95] = 1;
   out_6376990126270885036[96] = 0;
   out_6376990126270885036[97] = 0;
   out_6376990126270885036[98] = 0;
   out_6376990126270885036[99] = 0;
   out_6376990126270885036[100] = 0;
   out_6376990126270885036[101] = 0;
   out_6376990126270885036[102] = 0;
   out_6376990126270885036[103] = 0;
   out_6376990126270885036[104] = dt;
   out_6376990126270885036[105] = 0;
   out_6376990126270885036[106] = 0;
   out_6376990126270885036[107] = 0;
   out_6376990126270885036[108] = 0;
   out_6376990126270885036[109] = 0;
   out_6376990126270885036[110] = 0;
   out_6376990126270885036[111] = 0;
   out_6376990126270885036[112] = 0;
   out_6376990126270885036[113] = 0;
   out_6376990126270885036[114] = 1;
   out_6376990126270885036[115] = 0;
   out_6376990126270885036[116] = 0;
   out_6376990126270885036[117] = 0;
   out_6376990126270885036[118] = 0;
   out_6376990126270885036[119] = 0;
   out_6376990126270885036[120] = 0;
   out_6376990126270885036[121] = 0;
   out_6376990126270885036[122] = 0;
   out_6376990126270885036[123] = 0;
   out_6376990126270885036[124] = 0;
   out_6376990126270885036[125] = 0;
   out_6376990126270885036[126] = 0;
   out_6376990126270885036[127] = 0;
   out_6376990126270885036[128] = 0;
   out_6376990126270885036[129] = 0;
   out_6376990126270885036[130] = 0;
   out_6376990126270885036[131] = 0;
   out_6376990126270885036[132] = 0;
   out_6376990126270885036[133] = 1;
   out_6376990126270885036[134] = 0;
   out_6376990126270885036[135] = 0;
   out_6376990126270885036[136] = 0;
   out_6376990126270885036[137] = 0;
   out_6376990126270885036[138] = 0;
   out_6376990126270885036[139] = 0;
   out_6376990126270885036[140] = 0;
   out_6376990126270885036[141] = 0;
   out_6376990126270885036[142] = 0;
   out_6376990126270885036[143] = 0;
   out_6376990126270885036[144] = 0;
   out_6376990126270885036[145] = 0;
   out_6376990126270885036[146] = 0;
   out_6376990126270885036[147] = 0;
   out_6376990126270885036[148] = 0;
   out_6376990126270885036[149] = 0;
   out_6376990126270885036[150] = 0;
   out_6376990126270885036[151] = 0;
   out_6376990126270885036[152] = 1;
   out_6376990126270885036[153] = 0;
   out_6376990126270885036[154] = 0;
   out_6376990126270885036[155] = 0;
   out_6376990126270885036[156] = 0;
   out_6376990126270885036[157] = 0;
   out_6376990126270885036[158] = 0;
   out_6376990126270885036[159] = 0;
   out_6376990126270885036[160] = 0;
   out_6376990126270885036[161] = 0;
   out_6376990126270885036[162] = 0;
   out_6376990126270885036[163] = 0;
   out_6376990126270885036[164] = 0;
   out_6376990126270885036[165] = 0;
   out_6376990126270885036[166] = 0;
   out_6376990126270885036[167] = 0;
   out_6376990126270885036[168] = 0;
   out_6376990126270885036[169] = 0;
   out_6376990126270885036[170] = 0;
   out_6376990126270885036[171] = 1;
   out_6376990126270885036[172] = 0;
   out_6376990126270885036[173] = 0;
   out_6376990126270885036[174] = 0;
   out_6376990126270885036[175] = 0;
   out_6376990126270885036[176] = 0;
   out_6376990126270885036[177] = 0;
   out_6376990126270885036[178] = 0;
   out_6376990126270885036[179] = 0;
   out_6376990126270885036[180] = 0;
   out_6376990126270885036[181] = 0;
   out_6376990126270885036[182] = 0;
   out_6376990126270885036[183] = 0;
   out_6376990126270885036[184] = 0;
   out_6376990126270885036[185] = 0;
   out_6376990126270885036[186] = 0;
   out_6376990126270885036[187] = 0;
   out_6376990126270885036[188] = 0;
   out_6376990126270885036[189] = 0;
   out_6376990126270885036[190] = 1;
   out_6376990126270885036[191] = 0;
   out_6376990126270885036[192] = 0;
   out_6376990126270885036[193] = 0;
   out_6376990126270885036[194] = 0;
   out_6376990126270885036[195] = 0;
   out_6376990126270885036[196] = 0;
   out_6376990126270885036[197] = 0;
   out_6376990126270885036[198] = 0;
   out_6376990126270885036[199] = 0;
   out_6376990126270885036[200] = 0;
   out_6376990126270885036[201] = 0;
   out_6376990126270885036[202] = 0;
   out_6376990126270885036[203] = 0;
   out_6376990126270885036[204] = 0;
   out_6376990126270885036[205] = 0;
   out_6376990126270885036[206] = 0;
   out_6376990126270885036[207] = 0;
   out_6376990126270885036[208] = 0;
   out_6376990126270885036[209] = 1;
   out_6376990126270885036[210] = 0;
   out_6376990126270885036[211] = 0;
   out_6376990126270885036[212] = 0;
   out_6376990126270885036[213] = 0;
   out_6376990126270885036[214] = 0;
   out_6376990126270885036[215] = 0;
   out_6376990126270885036[216] = 0;
   out_6376990126270885036[217] = 0;
   out_6376990126270885036[218] = 0;
   out_6376990126270885036[219] = 0;
   out_6376990126270885036[220] = 0;
   out_6376990126270885036[221] = 0;
   out_6376990126270885036[222] = 0;
   out_6376990126270885036[223] = 0;
   out_6376990126270885036[224] = 0;
   out_6376990126270885036[225] = 0;
   out_6376990126270885036[226] = 0;
   out_6376990126270885036[227] = 0;
   out_6376990126270885036[228] = 1;
   out_6376990126270885036[229] = 0;
   out_6376990126270885036[230] = 0;
   out_6376990126270885036[231] = 0;
   out_6376990126270885036[232] = 0;
   out_6376990126270885036[233] = 0;
   out_6376990126270885036[234] = 0;
   out_6376990126270885036[235] = 0;
   out_6376990126270885036[236] = 0;
   out_6376990126270885036[237] = 0;
   out_6376990126270885036[238] = 0;
   out_6376990126270885036[239] = 0;
   out_6376990126270885036[240] = 0;
   out_6376990126270885036[241] = 0;
   out_6376990126270885036[242] = 0;
   out_6376990126270885036[243] = 0;
   out_6376990126270885036[244] = 0;
   out_6376990126270885036[245] = 0;
   out_6376990126270885036[246] = 0;
   out_6376990126270885036[247] = 1;
   out_6376990126270885036[248] = 0;
   out_6376990126270885036[249] = 0;
   out_6376990126270885036[250] = 0;
   out_6376990126270885036[251] = 0;
   out_6376990126270885036[252] = 0;
   out_6376990126270885036[253] = 0;
   out_6376990126270885036[254] = 0;
   out_6376990126270885036[255] = 0;
   out_6376990126270885036[256] = 0;
   out_6376990126270885036[257] = 0;
   out_6376990126270885036[258] = 0;
   out_6376990126270885036[259] = 0;
   out_6376990126270885036[260] = 0;
   out_6376990126270885036[261] = 0;
   out_6376990126270885036[262] = 0;
   out_6376990126270885036[263] = 0;
   out_6376990126270885036[264] = 0;
   out_6376990126270885036[265] = 0;
   out_6376990126270885036[266] = 1;
   out_6376990126270885036[267] = 0;
   out_6376990126270885036[268] = 0;
   out_6376990126270885036[269] = 0;
   out_6376990126270885036[270] = 0;
   out_6376990126270885036[271] = 0;
   out_6376990126270885036[272] = 0;
   out_6376990126270885036[273] = 0;
   out_6376990126270885036[274] = 0;
   out_6376990126270885036[275] = 0;
   out_6376990126270885036[276] = 0;
   out_6376990126270885036[277] = 0;
   out_6376990126270885036[278] = 0;
   out_6376990126270885036[279] = 0;
   out_6376990126270885036[280] = 0;
   out_6376990126270885036[281] = 0;
   out_6376990126270885036[282] = 0;
   out_6376990126270885036[283] = 0;
   out_6376990126270885036[284] = 0;
   out_6376990126270885036[285] = 1;
   out_6376990126270885036[286] = 0;
   out_6376990126270885036[287] = 0;
   out_6376990126270885036[288] = 0;
   out_6376990126270885036[289] = 0;
   out_6376990126270885036[290] = 0;
   out_6376990126270885036[291] = 0;
   out_6376990126270885036[292] = 0;
   out_6376990126270885036[293] = 0;
   out_6376990126270885036[294] = 0;
   out_6376990126270885036[295] = 0;
   out_6376990126270885036[296] = 0;
   out_6376990126270885036[297] = 0;
   out_6376990126270885036[298] = 0;
   out_6376990126270885036[299] = 0;
   out_6376990126270885036[300] = 0;
   out_6376990126270885036[301] = 0;
   out_6376990126270885036[302] = 0;
   out_6376990126270885036[303] = 0;
   out_6376990126270885036[304] = 1;
   out_6376990126270885036[305] = 0;
   out_6376990126270885036[306] = 0;
   out_6376990126270885036[307] = 0;
   out_6376990126270885036[308] = 0;
   out_6376990126270885036[309] = 0;
   out_6376990126270885036[310] = 0;
   out_6376990126270885036[311] = 0;
   out_6376990126270885036[312] = 0;
   out_6376990126270885036[313] = 0;
   out_6376990126270885036[314] = 0;
   out_6376990126270885036[315] = 0;
   out_6376990126270885036[316] = 0;
   out_6376990126270885036[317] = 0;
   out_6376990126270885036[318] = 0;
   out_6376990126270885036[319] = 0;
   out_6376990126270885036[320] = 0;
   out_6376990126270885036[321] = 0;
   out_6376990126270885036[322] = 0;
   out_6376990126270885036[323] = 1;
}
void h_4(double *state, double *unused, double *out_1784339471610928199) {
   out_1784339471610928199[0] = state[6] + state[9];
   out_1784339471610928199[1] = state[7] + state[10];
   out_1784339471610928199[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4713864713666167842) {
   out_4713864713666167842[0] = 0;
   out_4713864713666167842[1] = 0;
   out_4713864713666167842[2] = 0;
   out_4713864713666167842[3] = 0;
   out_4713864713666167842[4] = 0;
   out_4713864713666167842[5] = 0;
   out_4713864713666167842[6] = 1;
   out_4713864713666167842[7] = 0;
   out_4713864713666167842[8] = 0;
   out_4713864713666167842[9] = 1;
   out_4713864713666167842[10] = 0;
   out_4713864713666167842[11] = 0;
   out_4713864713666167842[12] = 0;
   out_4713864713666167842[13] = 0;
   out_4713864713666167842[14] = 0;
   out_4713864713666167842[15] = 0;
   out_4713864713666167842[16] = 0;
   out_4713864713666167842[17] = 0;
   out_4713864713666167842[18] = 0;
   out_4713864713666167842[19] = 0;
   out_4713864713666167842[20] = 0;
   out_4713864713666167842[21] = 0;
   out_4713864713666167842[22] = 0;
   out_4713864713666167842[23] = 0;
   out_4713864713666167842[24] = 0;
   out_4713864713666167842[25] = 1;
   out_4713864713666167842[26] = 0;
   out_4713864713666167842[27] = 0;
   out_4713864713666167842[28] = 1;
   out_4713864713666167842[29] = 0;
   out_4713864713666167842[30] = 0;
   out_4713864713666167842[31] = 0;
   out_4713864713666167842[32] = 0;
   out_4713864713666167842[33] = 0;
   out_4713864713666167842[34] = 0;
   out_4713864713666167842[35] = 0;
   out_4713864713666167842[36] = 0;
   out_4713864713666167842[37] = 0;
   out_4713864713666167842[38] = 0;
   out_4713864713666167842[39] = 0;
   out_4713864713666167842[40] = 0;
   out_4713864713666167842[41] = 0;
   out_4713864713666167842[42] = 0;
   out_4713864713666167842[43] = 0;
   out_4713864713666167842[44] = 1;
   out_4713864713666167842[45] = 0;
   out_4713864713666167842[46] = 0;
   out_4713864713666167842[47] = 1;
   out_4713864713666167842[48] = 0;
   out_4713864713666167842[49] = 0;
   out_4713864713666167842[50] = 0;
   out_4713864713666167842[51] = 0;
   out_4713864713666167842[52] = 0;
   out_4713864713666167842[53] = 0;
}
void h_10(double *state, double *unused, double *out_7218205989387104649) {
   out_7218205989387104649[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_7218205989387104649[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_7218205989387104649[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_664999596844092490) {
   out_664999596844092490[0] = 0;
   out_664999596844092490[1] = 9.8100000000000005*cos(state[1]);
   out_664999596844092490[2] = 0;
   out_664999596844092490[3] = 0;
   out_664999596844092490[4] = -state[8];
   out_664999596844092490[5] = state[7];
   out_664999596844092490[6] = 0;
   out_664999596844092490[7] = state[5];
   out_664999596844092490[8] = -state[4];
   out_664999596844092490[9] = 0;
   out_664999596844092490[10] = 0;
   out_664999596844092490[11] = 0;
   out_664999596844092490[12] = 1;
   out_664999596844092490[13] = 0;
   out_664999596844092490[14] = 0;
   out_664999596844092490[15] = 1;
   out_664999596844092490[16] = 0;
   out_664999596844092490[17] = 0;
   out_664999596844092490[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_664999596844092490[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_664999596844092490[20] = 0;
   out_664999596844092490[21] = state[8];
   out_664999596844092490[22] = 0;
   out_664999596844092490[23] = -state[6];
   out_664999596844092490[24] = -state[5];
   out_664999596844092490[25] = 0;
   out_664999596844092490[26] = state[3];
   out_664999596844092490[27] = 0;
   out_664999596844092490[28] = 0;
   out_664999596844092490[29] = 0;
   out_664999596844092490[30] = 0;
   out_664999596844092490[31] = 1;
   out_664999596844092490[32] = 0;
   out_664999596844092490[33] = 0;
   out_664999596844092490[34] = 1;
   out_664999596844092490[35] = 0;
   out_664999596844092490[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_664999596844092490[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_664999596844092490[38] = 0;
   out_664999596844092490[39] = -state[7];
   out_664999596844092490[40] = state[6];
   out_664999596844092490[41] = 0;
   out_664999596844092490[42] = state[4];
   out_664999596844092490[43] = -state[3];
   out_664999596844092490[44] = 0;
   out_664999596844092490[45] = 0;
   out_664999596844092490[46] = 0;
   out_664999596844092490[47] = 0;
   out_664999596844092490[48] = 0;
   out_664999596844092490[49] = 0;
   out_664999596844092490[50] = 1;
   out_664999596844092490[51] = 0;
   out_664999596844092490[52] = 0;
   out_664999596844092490[53] = 1;
}
void h_13(double *state, double *unused, double *out_4772014765743314712) {
   out_4772014765743314712[0] = state[3];
   out_4772014765743314712[1] = state[4];
   out_4772014765743314712[2] = state[5];
}
void H_13(double *state, double *unused, double *out_880109250363643818) {
   out_880109250363643818[0] = 0;
   out_880109250363643818[1] = 0;
   out_880109250363643818[2] = 0;
   out_880109250363643818[3] = 1;
   out_880109250363643818[4] = 0;
   out_880109250363643818[5] = 0;
   out_880109250363643818[6] = 0;
   out_880109250363643818[7] = 0;
   out_880109250363643818[8] = 0;
   out_880109250363643818[9] = 0;
   out_880109250363643818[10] = 0;
   out_880109250363643818[11] = 0;
   out_880109250363643818[12] = 0;
   out_880109250363643818[13] = 0;
   out_880109250363643818[14] = 0;
   out_880109250363643818[15] = 0;
   out_880109250363643818[16] = 0;
   out_880109250363643818[17] = 0;
   out_880109250363643818[18] = 0;
   out_880109250363643818[19] = 0;
   out_880109250363643818[20] = 0;
   out_880109250363643818[21] = 0;
   out_880109250363643818[22] = 1;
   out_880109250363643818[23] = 0;
   out_880109250363643818[24] = 0;
   out_880109250363643818[25] = 0;
   out_880109250363643818[26] = 0;
   out_880109250363643818[27] = 0;
   out_880109250363643818[28] = 0;
   out_880109250363643818[29] = 0;
   out_880109250363643818[30] = 0;
   out_880109250363643818[31] = 0;
   out_880109250363643818[32] = 0;
   out_880109250363643818[33] = 0;
   out_880109250363643818[34] = 0;
   out_880109250363643818[35] = 0;
   out_880109250363643818[36] = 0;
   out_880109250363643818[37] = 0;
   out_880109250363643818[38] = 0;
   out_880109250363643818[39] = 0;
   out_880109250363643818[40] = 0;
   out_880109250363643818[41] = 1;
   out_880109250363643818[42] = 0;
   out_880109250363643818[43] = 0;
   out_880109250363643818[44] = 0;
   out_880109250363643818[45] = 0;
   out_880109250363643818[46] = 0;
   out_880109250363643818[47] = 0;
   out_880109250363643818[48] = 0;
   out_880109250363643818[49] = 0;
   out_880109250363643818[50] = 0;
   out_880109250363643818[51] = 0;
   out_880109250363643818[52] = 0;
   out_880109250363643818[53] = 0;
}
void h_14(double *state, double *unused, double *out_7459610224657432703) {
   out_7459610224657432703[0] = state[6];
   out_7459610224657432703[1] = state[7];
   out_7459610224657432703[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1631076281370795546) {
   out_1631076281370795546[0] = 0;
   out_1631076281370795546[1] = 0;
   out_1631076281370795546[2] = 0;
   out_1631076281370795546[3] = 0;
   out_1631076281370795546[4] = 0;
   out_1631076281370795546[5] = 0;
   out_1631076281370795546[6] = 1;
   out_1631076281370795546[7] = 0;
   out_1631076281370795546[8] = 0;
   out_1631076281370795546[9] = 0;
   out_1631076281370795546[10] = 0;
   out_1631076281370795546[11] = 0;
   out_1631076281370795546[12] = 0;
   out_1631076281370795546[13] = 0;
   out_1631076281370795546[14] = 0;
   out_1631076281370795546[15] = 0;
   out_1631076281370795546[16] = 0;
   out_1631076281370795546[17] = 0;
   out_1631076281370795546[18] = 0;
   out_1631076281370795546[19] = 0;
   out_1631076281370795546[20] = 0;
   out_1631076281370795546[21] = 0;
   out_1631076281370795546[22] = 0;
   out_1631076281370795546[23] = 0;
   out_1631076281370795546[24] = 0;
   out_1631076281370795546[25] = 1;
   out_1631076281370795546[26] = 0;
   out_1631076281370795546[27] = 0;
   out_1631076281370795546[28] = 0;
   out_1631076281370795546[29] = 0;
   out_1631076281370795546[30] = 0;
   out_1631076281370795546[31] = 0;
   out_1631076281370795546[32] = 0;
   out_1631076281370795546[33] = 0;
   out_1631076281370795546[34] = 0;
   out_1631076281370795546[35] = 0;
   out_1631076281370795546[36] = 0;
   out_1631076281370795546[37] = 0;
   out_1631076281370795546[38] = 0;
   out_1631076281370795546[39] = 0;
   out_1631076281370795546[40] = 0;
   out_1631076281370795546[41] = 0;
   out_1631076281370795546[42] = 0;
   out_1631076281370795546[43] = 0;
   out_1631076281370795546[44] = 1;
   out_1631076281370795546[45] = 0;
   out_1631076281370795546[46] = 0;
   out_1631076281370795546[47] = 0;
   out_1631076281370795546[48] = 0;
   out_1631076281370795546[49] = 0;
   out_1631076281370795546[50] = 0;
   out_1631076281370795546[51] = 0;
   out_1631076281370795546[52] = 0;
   out_1631076281370795546[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_8287156320329551527) {
  err_fun(nom_x, delta_x, out_8287156320329551527);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_152026319407410727) {
  inv_err_fun(nom_x, true_x, out_152026319407410727);
}
void pose_H_mod_fun(double *state, double *out_3515917231783269738) {
  H_mod_fun(state, out_3515917231783269738);
}
void pose_f_fun(double *state, double dt, double *out_8327036891744297669) {
  f_fun(state,  dt, out_8327036891744297669);
}
void pose_F_fun(double *state, double dt, double *out_6376990126270885036) {
  F_fun(state,  dt, out_6376990126270885036);
}
void pose_h_4(double *state, double *unused, double *out_1784339471610928199) {
  h_4(state, unused, out_1784339471610928199);
}
void pose_H_4(double *state, double *unused, double *out_4713864713666167842) {
  H_4(state, unused, out_4713864713666167842);
}
void pose_h_10(double *state, double *unused, double *out_7218205989387104649) {
  h_10(state, unused, out_7218205989387104649);
}
void pose_H_10(double *state, double *unused, double *out_664999596844092490) {
  H_10(state, unused, out_664999596844092490);
}
void pose_h_13(double *state, double *unused, double *out_4772014765743314712) {
  h_13(state, unused, out_4772014765743314712);
}
void pose_H_13(double *state, double *unused, double *out_880109250363643818) {
  H_13(state, unused, out_880109250363643818);
}
void pose_h_14(double *state, double *unused, double *out_7459610224657432703) {
  h_14(state, unused, out_7459610224657432703);
}
void pose_H_14(double *state, double *unused, double *out_1631076281370795546) {
  H_14(state, unused, out_1631076281370795546);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
