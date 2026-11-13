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
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_695385451800373050) {
   out_695385451800373050[0] = delta_x[0] + nom_x[0];
   out_695385451800373050[1] = delta_x[1] + nom_x[1];
   out_695385451800373050[2] = delta_x[2] + nom_x[2];
   out_695385451800373050[3] = delta_x[3] + nom_x[3];
   out_695385451800373050[4] = delta_x[4] + nom_x[4];
   out_695385451800373050[5] = delta_x[5] + nom_x[5];
   out_695385451800373050[6] = delta_x[6] + nom_x[6];
   out_695385451800373050[7] = delta_x[7] + nom_x[7];
   out_695385451800373050[8] = delta_x[8] + nom_x[8];
   out_695385451800373050[9] = delta_x[9] + nom_x[9];
   out_695385451800373050[10] = delta_x[10] + nom_x[10];
   out_695385451800373050[11] = delta_x[11] + nom_x[11];
   out_695385451800373050[12] = delta_x[12] + nom_x[12];
   out_695385451800373050[13] = delta_x[13] + nom_x[13];
   out_695385451800373050[14] = delta_x[14] + nom_x[14];
   out_695385451800373050[15] = delta_x[15] + nom_x[15];
   out_695385451800373050[16] = delta_x[16] + nom_x[16];
   out_695385451800373050[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3513419238100027883) {
   out_3513419238100027883[0] = -nom_x[0] + true_x[0];
   out_3513419238100027883[1] = -nom_x[1] + true_x[1];
   out_3513419238100027883[2] = -nom_x[2] + true_x[2];
   out_3513419238100027883[3] = -nom_x[3] + true_x[3];
   out_3513419238100027883[4] = -nom_x[4] + true_x[4];
   out_3513419238100027883[5] = -nom_x[5] + true_x[5];
   out_3513419238100027883[6] = -nom_x[6] + true_x[6];
   out_3513419238100027883[7] = -nom_x[7] + true_x[7];
   out_3513419238100027883[8] = -nom_x[8] + true_x[8];
   out_3513419238100027883[9] = -nom_x[9] + true_x[9];
   out_3513419238100027883[10] = -nom_x[10] + true_x[10];
   out_3513419238100027883[11] = -nom_x[11] + true_x[11];
   out_3513419238100027883[12] = -nom_x[12] + true_x[12];
   out_3513419238100027883[13] = -nom_x[13] + true_x[13];
   out_3513419238100027883[14] = -nom_x[14] + true_x[14];
   out_3513419238100027883[15] = -nom_x[15] + true_x[15];
   out_3513419238100027883[16] = -nom_x[16] + true_x[16];
   out_3513419238100027883[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_6118050737020864777) {
   out_6118050737020864777[0] = 1.0;
   out_6118050737020864777[1] = 0.0;
   out_6118050737020864777[2] = 0.0;
   out_6118050737020864777[3] = 0.0;
   out_6118050737020864777[4] = 0.0;
   out_6118050737020864777[5] = 0.0;
   out_6118050737020864777[6] = 0.0;
   out_6118050737020864777[7] = 0.0;
   out_6118050737020864777[8] = 0.0;
   out_6118050737020864777[9] = 0.0;
   out_6118050737020864777[10] = 0.0;
   out_6118050737020864777[11] = 0.0;
   out_6118050737020864777[12] = 0.0;
   out_6118050737020864777[13] = 0.0;
   out_6118050737020864777[14] = 0.0;
   out_6118050737020864777[15] = 0.0;
   out_6118050737020864777[16] = 0.0;
   out_6118050737020864777[17] = 0.0;
   out_6118050737020864777[18] = 0.0;
   out_6118050737020864777[19] = 1.0;
   out_6118050737020864777[20] = 0.0;
   out_6118050737020864777[21] = 0.0;
   out_6118050737020864777[22] = 0.0;
   out_6118050737020864777[23] = 0.0;
   out_6118050737020864777[24] = 0.0;
   out_6118050737020864777[25] = 0.0;
   out_6118050737020864777[26] = 0.0;
   out_6118050737020864777[27] = 0.0;
   out_6118050737020864777[28] = 0.0;
   out_6118050737020864777[29] = 0.0;
   out_6118050737020864777[30] = 0.0;
   out_6118050737020864777[31] = 0.0;
   out_6118050737020864777[32] = 0.0;
   out_6118050737020864777[33] = 0.0;
   out_6118050737020864777[34] = 0.0;
   out_6118050737020864777[35] = 0.0;
   out_6118050737020864777[36] = 0.0;
   out_6118050737020864777[37] = 0.0;
   out_6118050737020864777[38] = 1.0;
   out_6118050737020864777[39] = 0.0;
   out_6118050737020864777[40] = 0.0;
   out_6118050737020864777[41] = 0.0;
   out_6118050737020864777[42] = 0.0;
   out_6118050737020864777[43] = 0.0;
   out_6118050737020864777[44] = 0.0;
   out_6118050737020864777[45] = 0.0;
   out_6118050737020864777[46] = 0.0;
   out_6118050737020864777[47] = 0.0;
   out_6118050737020864777[48] = 0.0;
   out_6118050737020864777[49] = 0.0;
   out_6118050737020864777[50] = 0.0;
   out_6118050737020864777[51] = 0.0;
   out_6118050737020864777[52] = 0.0;
   out_6118050737020864777[53] = 0.0;
   out_6118050737020864777[54] = 0.0;
   out_6118050737020864777[55] = 0.0;
   out_6118050737020864777[56] = 0.0;
   out_6118050737020864777[57] = 1.0;
   out_6118050737020864777[58] = 0.0;
   out_6118050737020864777[59] = 0.0;
   out_6118050737020864777[60] = 0.0;
   out_6118050737020864777[61] = 0.0;
   out_6118050737020864777[62] = 0.0;
   out_6118050737020864777[63] = 0.0;
   out_6118050737020864777[64] = 0.0;
   out_6118050737020864777[65] = 0.0;
   out_6118050737020864777[66] = 0.0;
   out_6118050737020864777[67] = 0.0;
   out_6118050737020864777[68] = 0.0;
   out_6118050737020864777[69] = 0.0;
   out_6118050737020864777[70] = 0.0;
   out_6118050737020864777[71] = 0.0;
   out_6118050737020864777[72] = 0.0;
   out_6118050737020864777[73] = 0.0;
   out_6118050737020864777[74] = 0.0;
   out_6118050737020864777[75] = 0.0;
   out_6118050737020864777[76] = 1.0;
   out_6118050737020864777[77] = 0.0;
   out_6118050737020864777[78] = 0.0;
   out_6118050737020864777[79] = 0.0;
   out_6118050737020864777[80] = 0.0;
   out_6118050737020864777[81] = 0.0;
   out_6118050737020864777[82] = 0.0;
   out_6118050737020864777[83] = 0.0;
   out_6118050737020864777[84] = 0.0;
   out_6118050737020864777[85] = 0.0;
   out_6118050737020864777[86] = 0.0;
   out_6118050737020864777[87] = 0.0;
   out_6118050737020864777[88] = 0.0;
   out_6118050737020864777[89] = 0.0;
   out_6118050737020864777[90] = 0.0;
   out_6118050737020864777[91] = 0.0;
   out_6118050737020864777[92] = 0.0;
   out_6118050737020864777[93] = 0.0;
   out_6118050737020864777[94] = 0.0;
   out_6118050737020864777[95] = 1.0;
   out_6118050737020864777[96] = 0.0;
   out_6118050737020864777[97] = 0.0;
   out_6118050737020864777[98] = 0.0;
   out_6118050737020864777[99] = 0.0;
   out_6118050737020864777[100] = 0.0;
   out_6118050737020864777[101] = 0.0;
   out_6118050737020864777[102] = 0.0;
   out_6118050737020864777[103] = 0.0;
   out_6118050737020864777[104] = 0.0;
   out_6118050737020864777[105] = 0.0;
   out_6118050737020864777[106] = 0.0;
   out_6118050737020864777[107] = 0.0;
   out_6118050737020864777[108] = 0.0;
   out_6118050737020864777[109] = 0.0;
   out_6118050737020864777[110] = 0.0;
   out_6118050737020864777[111] = 0.0;
   out_6118050737020864777[112] = 0.0;
   out_6118050737020864777[113] = 0.0;
   out_6118050737020864777[114] = 1.0;
   out_6118050737020864777[115] = 0.0;
   out_6118050737020864777[116] = 0.0;
   out_6118050737020864777[117] = 0.0;
   out_6118050737020864777[118] = 0.0;
   out_6118050737020864777[119] = 0.0;
   out_6118050737020864777[120] = 0.0;
   out_6118050737020864777[121] = 0.0;
   out_6118050737020864777[122] = 0.0;
   out_6118050737020864777[123] = 0.0;
   out_6118050737020864777[124] = 0.0;
   out_6118050737020864777[125] = 0.0;
   out_6118050737020864777[126] = 0.0;
   out_6118050737020864777[127] = 0.0;
   out_6118050737020864777[128] = 0.0;
   out_6118050737020864777[129] = 0.0;
   out_6118050737020864777[130] = 0.0;
   out_6118050737020864777[131] = 0.0;
   out_6118050737020864777[132] = 0.0;
   out_6118050737020864777[133] = 1.0;
   out_6118050737020864777[134] = 0.0;
   out_6118050737020864777[135] = 0.0;
   out_6118050737020864777[136] = 0.0;
   out_6118050737020864777[137] = 0.0;
   out_6118050737020864777[138] = 0.0;
   out_6118050737020864777[139] = 0.0;
   out_6118050737020864777[140] = 0.0;
   out_6118050737020864777[141] = 0.0;
   out_6118050737020864777[142] = 0.0;
   out_6118050737020864777[143] = 0.0;
   out_6118050737020864777[144] = 0.0;
   out_6118050737020864777[145] = 0.0;
   out_6118050737020864777[146] = 0.0;
   out_6118050737020864777[147] = 0.0;
   out_6118050737020864777[148] = 0.0;
   out_6118050737020864777[149] = 0.0;
   out_6118050737020864777[150] = 0.0;
   out_6118050737020864777[151] = 0.0;
   out_6118050737020864777[152] = 1.0;
   out_6118050737020864777[153] = 0.0;
   out_6118050737020864777[154] = 0.0;
   out_6118050737020864777[155] = 0.0;
   out_6118050737020864777[156] = 0.0;
   out_6118050737020864777[157] = 0.0;
   out_6118050737020864777[158] = 0.0;
   out_6118050737020864777[159] = 0.0;
   out_6118050737020864777[160] = 0.0;
   out_6118050737020864777[161] = 0.0;
   out_6118050737020864777[162] = 0.0;
   out_6118050737020864777[163] = 0.0;
   out_6118050737020864777[164] = 0.0;
   out_6118050737020864777[165] = 0.0;
   out_6118050737020864777[166] = 0.0;
   out_6118050737020864777[167] = 0.0;
   out_6118050737020864777[168] = 0.0;
   out_6118050737020864777[169] = 0.0;
   out_6118050737020864777[170] = 0.0;
   out_6118050737020864777[171] = 1.0;
   out_6118050737020864777[172] = 0.0;
   out_6118050737020864777[173] = 0.0;
   out_6118050737020864777[174] = 0.0;
   out_6118050737020864777[175] = 0.0;
   out_6118050737020864777[176] = 0.0;
   out_6118050737020864777[177] = 0.0;
   out_6118050737020864777[178] = 0.0;
   out_6118050737020864777[179] = 0.0;
   out_6118050737020864777[180] = 0.0;
   out_6118050737020864777[181] = 0.0;
   out_6118050737020864777[182] = 0.0;
   out_6118050737020864777[183] = 0.0;
   out_6118050737020864777[184] = 0.0;
   out_6118050737020864777[185] = 0.0;
   out_6118050737020864777[186] = 0.0;
   out_6118050737020864777[187] = 0.0;
   out_6118050737020864777[188] = 0.0;
   out_6118050737020864777[189] = 0.0;
   out_6118050737020864777[190] = 1.0;
   out_6118050737020864777[191] = 0.0;
   out_6118050737020864777[192] = 0.0;
   out_6118050737020864777[193] = 0.0;
   out_6118050737020864777[194] = 0.0;
   out_6118050737020864777[195] = 0.0;
   out_6118050737020864777[196] = 0.0;
   out_6118050737020864777[197] = 0.0;
   out_6118050737020864777[198] = 0.0;
   out_6118050737020864777[199] = 0.0;
   out_6118050737020864777[200] = 0.0;
   out_6118050737020864777[201] = 0.0;
   out_6118050737020864777[202] = 0.0;
   out_6118050737020864777[203] = 0.0;
   out_6118050737020864777[204] = 0.0;
   out_6118050737020864777[205] = 0.0;
   out_6118050737020864777[206] = 0.0;
   out_6118050737020864777[207] = 0.0;
   out_6118050737020864777[208] = 0.0;
   out_6118050737020864777[209] = 1.0;
   out_6118050737020864777[210] = 0.0;
   out_6118050737020864777[211] = 0.0;
   out_6118050737020864777[212] = 0.0;
   out_6118050737020864777[213] = 0.0;
   out_6118050737020864777[214] = 0.0;
   out_6118050737020864777[215] = 0.0;
   out_6118050737020864777[216] = 0.0;
   out_6118050737020864777[217] = 0.0;
   out_6118050737020864777[218] = 0.0;
   out_6118050737020864777[219] = 0.0;
   out_6118050737020864777[220] = 0.0;
   out_6118050737020864777[221] = 0.0;
   out_6118050737020864777[222] = 0.0;
   out_6118050737020864777[223] = 0.0;
   out_6118050737020864777[224] = 0.0;
   out_6118050737020864777[225] = 0.0;
   out_6118050737020864777[226] = 0.0;
   out_6118050737020864777[227] = 0.0;
   out_6118050737020864777[228] = 1.0;
   out_6118050737020864777[229] = 0.0;
   out_6118050737020864777[230] = 0.0;
   out_6118050737020864777[231] = 0.0;
   out_6118050737020864777[232] = 0.0;
   out_6118050737020864777[233] = 0.0;
   out_6118050737020864777[234] = 0.0;
   out_6118050737020864777[235] = 0.0;
   out_6118050737020864777[236] = 0.0;
   out_6118050737020864777[237] = 0.0;
   out_6118050737020864777[238] = 0.0;
   out_6118050737020864777[239] = 0.0;
   out_6118050737020864777[240] = 0.0;
   out_6118050737020864777[241] = 0.0;
   out_6118050737020864777[242] = 0.0;
   out_6118050737020864777[243] = 0.0;
   out_6118050737020864777[244] = 0.0;
   out_6118050737020864777[245] = 0.0;
   out_6118050737020864777[246] = 0.0;
   out_6118050737020864777[247] = 1.0;
   out_6118050737020864777[248] = 0.0;
   out_6118050737020864777[249] = 0.0;
   out_6118050737020864777[250] = 0.0;
   out_6118050737020864777[251] = 0.0;
   out_6118050737020864777[252] = 0.0;
   out_6118050737020864777[253] = 0.0;
   out_6118050737020864777[254] = 0.0;
   out_6118050737020864777[255] = 0.0;
   out_6118050737020864777[256] = 0.0;
   out_6118050737020864777[257] = 0.0;
   out_6118050737020864777[258] = 0.0;
   out_6118050737020864777[259] = 0.0;
   out_6118050737020864777[260] = 0.0;
   out_6118050737020864777[261] = 0.0;
   out_6118050737020864777[262] = 0.0;
   out_6118050737020864777[263] = 0.0;
   out_6118050737020864777[264] = 0.0;
   out_6118050737020864777[265] = 0.0;
   out_6118050737020864777[266] = 1.0;
   out_6118050737020864777[267] = 0.0;
   out_6118050737020864777[268] = 0.0;
   out_6118050737020864777[269] = 0.0;
   out_6118050737020864777[270] = 0.0;
   out_6118050737020864777[271] = 0.0;
   out_6118050737020864777[272] = 0.0;
   out_6118050737020864777[273] = 0.0;
   out_6118050737020864777[274] = 0.0;
   out_6118050737020864777[275] = 0.0;
   out_6118050737020864777[276] = 0.0;
   out_6118050737020864777[277] = 0.0;
   out_6118050737020864777[278] = 0.0;
   out_6118050737020864777[279] = 0.0;
   out_6118050737020864777[280] = 0.0;
   out_6118050737020864777[281] = 0.0;
   out_6118050737020864777[282] = 0.0;
   out_6118050737020864777[283] = 0.0;
   out_6118050737020864777[284] = 0.0;
   out_6118050737020864777[285] = 1.0;
   out_6118050737020864777[286] = 0.0;
   out_6118050737020864777[287] = 0.0;
   out_6118050737020864777[288] = 0.0;
   out_6118050737020864777[289] = 0.0;
   out_6118050737020864777[290] = 0.0;
   out_6118050737020864777[291] = 0.0;
   out_6118050737020864777[292] = 0.0;
   out_6118050737020864777[293] = 0.0;
   out_6118050737020864777[294] = 0.0;
   out_6118050737020864777[295] = 0.0;
   out_6118050737020864777[296] = 0.0;
   out_6118050737020864777[297] = 0.0;
   out_6118050737020864777[298] = 0.0;
   out_6118050737020864777[299] = 0.0;
   out_6118050737020864777[300] = 0.0;
   out_6118050737020864777[301] = 0.0;
   out_6118050737020864777[302] = 0.0;
   out_6118050737020864777[303] = 0.0;
   out_6118050737020864777[304] = 1.0;
   out_6118050737020864777[305] = 0.0;
   out_6118050737020864777[306] = 0.0;
   out_6118050737020864777[307] = 0.0;
   out_6118050737020864777[308] = 0.0;
   out_6118050737020864777[309] = 0.0;
   out_6118050737020864777[310] = 0.0;
   out_6118050737020864777[311] = 0.0;
   out_6118050737020864777[312] = 0.0;
   out_6118050737020864777[313] = 0.0;
   out_6118050737020864777[314] = 0.0;
   out_6118050737020864777[315] = 0.0;
   out_6118050737020864777[316] = 0.0;
   out_6118050737020864777[317] = 0.0;
   out_6118050737020864777[318] = 0.0;
   out_6118050737020864777[319] = 0.0;
   out_6118050737020864777[320] = 0.0;
   out_6118050737020864777[321] = 0.0;
   out_6118050737020864777[322] = 0.0;
   out_6118050737020864777[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_4351988193126604514) {
   out_4351988193126604514[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_4351988193126604514[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_4351988193126604514[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_4351988193126604514[3] = dt*state[12] + state[3];
   out_4351988193126604514[4] = dt*state[13] + state[4];
   out_4351988193126604514[5] = dt*state[14] + state[5];
   out_4351988193126604514[6] = state[6];
   out_4351988193126604514[7] = state[7];
   out_4351988193126604514[8] = state[8];
   out_4351988193126604514[9] = state[9];
   out_4351988193126604514[10] = state[10];
   out_4351988193126604514[11] = state[11];
   out_4351988193126604514[12] = state[12];
   out_4351988193126604514[13] = state[13];
   out_4351988193126604514[14] = state[14];
   out_4351988193126604514[15] = state[15];
   out_4351988193126604514[16] = state[16];
   out_4351988193126604514[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2969389070201032483) {
   out_2969389070201032483[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2969389070201032483[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2969389070201032483[2] = 0;
   out_2969389070201032483[3] = 0;
   out_2969389070201032483[4] = 0;
   out_2969389070201032483[5] = 0;
   out_2969389070201032483[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2969389070201032483[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2969389070201032483[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2969389070201032483[9] = 0;
   out_2969389070201032483[10] = 0;
   out_2969389070201032483[11] = 0;
   out_2969389070201032483[12] = 0;
   out_2969389070201032483[13] = 0;
   out_2969389070201032483[14] = 0;
   out_2969389070201032483[15] = 0;
   out_2969389070201032483[16] = 0;
   out_2969389070201032483[17] = 0;
   out_2969389070201032483[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2969389070201032483[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2969389070201032483[20] = 0;
   out_2969389070201032483[21] = 0;
   out_2969389070201032483[22] = 0;
   out_2969389070201032483[23] = 0;
   out_2969389070201032483[24] = 0;
   out_2969389070201032483[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2969389070201032483[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2969389070201032483[27] = 0;
   out_2969389070201032483[28] = 0;
   out_2969389070201032483[29] = 0;
   out_2969389070201032483[30] = 0;
   out_2969389070201032483[31] = 0;
   out_2969389070201032483[32] = 0;
   out_2969389070201032483[33] = 0;
   out_2969389070201032483[34] = 0;
   out_2969389070201032483[35] = 0;
   out_2969389070201032483[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2969389070201032483[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2969389070201032483[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2969389070201032483[39] = 0;
   out_2969389070201032483[40] = 0;
   out_2969389070201032483[41] = 0;
   out_2969389070201032483[42] = 0;
   out_2969389070201032483[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2969389070201032483[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2969389070201032483[45] = 0;
   out_2969389070201032483[46] = 0;
   out_2969389070201032483[47] = 0;
   out_2969389070201032483[48] = 0;
   out_2969389070201032483[49] = 0;
   out_2969389070201032483[50] = 0;
   out_2969389070201032483[51] = 0;
   out_2969389070201032483[52] = 0;
   out_2969389070201032483[53] = 0;
   out_2969389070201032483[54] = 0;
   out_2969389070201032483[55] = 0;
   out_2969389070201032483[56] = 0;
   out_2969389070201032483[57] = 1;
   out_2969389070201032483[58] = 0;
   out_2969389070201032483[59] = 0;
   out_2969389070201032483[60] = 0;
   out_2969389070201032483[61] = 0;
   out_2969389070201032483[62] = 0;
   out_2969389070201032483[63] = 0;
   out_2969389070201032483[64] = 0;
   out_2969389070201032483[65] = 0;
   out_2969389070201032483[66] = dt;
   out_2969389070201032483[67] = 0;
   out_2969389070201032483[68] = 0;
   out_2969389070201032483[69] = 0;
   out_2969389070201032483[70] = 0;
   out_2969389070201032483[71] = 0;
   out_2969389070201032483[72] = 0;
   out_2969389070201032483[73] = 0;
   out_2969389070201032483[74] = 0;
   out_2969389070201032483[75] = 0;
   out_2969389070201032483[76] = 1;
   out_2969389070201032483[77] = 0;
   out_2969389070201032483[78] = 0;
   out_2969389070201032483[79] = 0;
   out_2969389070201032483[80] = 0;
   out_2969389070201032483[81] = 0;
   out_2969389070201032483[82] = 0;
   out_2969389070201032483[83] = 0;
   out_2969389070201032483[84] = 0;
   out_2969389070201032483[85] = dt;
   out_2969389070201032483[86] = 0;
   out_2969389070201032483[87] = 0;
   out_2969389070201032483[88] = 0;
   out_2969389070201032483[89] = 0;
   out_2969389070201032483[90] = 0;
   out_2969389070201032483[91] = 0;
   out_2969389070201032483[92] = 0;
   out_2969389070201032483[93] = 0;
   out_2969389070201032483[94] = 0;
   out_2969389070201032483[95] = 1;
   out_2969389070201032483[96] = 0;
   out_2969389070201032483[97] = 0;
   out_2969389070201032483[98] = 0;
   out_2969389070201032483[99] = 0;
   out_2969389070201032483[100] = 0;
   out_2969389070201032483[101] = 0;
   out_2969389070201032483[102] = 0;
   out_2969389070201032483[103] = 0;
   out_2969389070201032483[104] = dt;
   out_2969389070201032483[105] = 0;
   out_2969389070201032483[106] = 0;
   out_2969389070201032483[107] = 0;
   out_2969389070201032483[108] = 0;
   out_2969389070201032483[109] = 0;
   out_2969389070201032483[110] = 0;
   out_2969389070201032483[111] = 0;
   out_2969389070201032483[112] = 0;
   out_2969389070201032483[113] = 0;
   out_2969389070201032483[114] = 1;
   out_2969389070201032483[115] = 0;
   out_2969389070201032483[116] = 0;
   out_2969389070201032483[117] = 0;
   out_2969389070201032483[118] = 0;
   out_2969389070201032483[119] = 0;
   out_2969389070201032483[120] = 0;
   out_2969389070201032483[121] = 0;
   out_2969389070201032483[122] = 0;
   out_2969389070201032483[123] = 0;
   out_2969389070201032483[124] = 0;
   out_2969389070201032483[125] = 0;
   out_2969389070201032483[126] = 0;
   out_2969389070201032483[127] = 0;
   out_2969389070201032483[128] = 0;
   out_2969389070201032483[129] = 0;
   out_2969389070201032483[130] = 0;
   out_2969389070201032483[131] = 0;
   out_2969389070201032483[132] = 0;
   out_2969389070201032483[133] = 1;
   out_2969389070201032483[134] = 0;
   out_2969389070201032483[135] = 0;
   out_2969389070201032483[136] = 0;
   out_2969389070201032483[137] = 0;
   out_2969389070201032483[138] = 0;
   out_2969389070201032483[139] = 0;
   out_2969389070201032483[140] = 0;
   out_2969389070201032483[141] = 0;
   out_2969389070201032483[142] = 0;
   out_2969389070201032483[143] = 0;
   out_2969389070201032483[144] = 0;
   out_2969389070201032483[145] = 0;
   out_2969389070201032483[146] = 0;
   out_2969389070201032483[147] = 0;
   out_2969389070201032483[148] = 0;
   out_2969389070201032483[149] = 0;
   out_2969389070201032483[150] = 0;
   out_2969389070201032483[151] = 0;
   out_2969389070201032483[152] = 1;
   out_2969389070201032483[153] = 0;
   out_2969389070201032483[154] = 0;
   out_2969389070201032483[155] = 0;
   out_2969389070201032483[156] = 0;
   out_2969389070201032483[157] = 0;
   out_2969389070201032483[158] = 0;
   out_2969389070201032483[159] = 0;
   out_2969389070201032483[160] = 0;
   out_2969389070201032483[161] = 0;
   out_2969389070201032483[162] = 0;
   out_2969389070201032483[163] = 0;
   out_2969389070201032483[164] = 0;
   out_2969389070201032483[165] = 0;
   out_2969389070201032483[166] = 0;
   out_2969389070201032483[167] = 0;
   out_2969389070201032483[168] = 0;
   out_2969389070201032483[169] = 0;
   out_2969389070201032483[170] = 0;
   out_2969389070201032483[171] = 1;
   out_2969389070201032483[172] = 0;
   out_2969389070201032483[173] = 0;
   out_2969389070201032483[174] = 0;
   out_2969389070201032483[175] = 0;
   out_2969389070201032483[176] = 0;
   out_2969389070201032483[177] = 0;
   out_2969389070201032483[178] = 0;
   out_2969389070201032483[179] = 0;
   out_2969389070201032483[180] = 0;
   out_2969389070201032483[181] = 0;
   out_2969389070201032483[182] = 0;
   out_2969389070201032483[183] = 0;
   out_2969389070201032483[184] = 0;
   out_2969389070201032483[185] = 0;
   out_2969389070201032483[186] = 0;
   out_2969389070201032483[187] = 0;
   out_2969389070201032483[188] = 0;
   out_2969389070201032483[189] = 0;
   out_2969389070201032483[190] = 1;
   out_2969389070201032483[191] = 0;
   out_2969389070201032483[192] = 0;
   out_2969389070201032483[193] = 0;
   out_2969389070201032483[194] = 0;
   out_2969389070201032483[195] = 0;
   out_2969389070201032483[196] = 0;
   out_2969389070201032483[197] = 0;
   out_2969389070201032483[198] = 0;
   out_2969389070201032483[199] = 0;
   out_2969389070201032483[200] = 0;
   out_2969389070201032483[201] = 0;
   out_2969389070201032483[202] = 0;
   out_2969389070201032483[203] = 0;
   out_2969389070201032483[204] = 0;
   out_2969389070201032483[205] = 0;
   out_2969389070201032483[206] = 0;
   out_2969389070201032483[207] = 0;
   out_2969389070201032483[208] = 0;
   out_2969389070201032483[209] = 1;
   out_2969389070201032483[210] = 0;
   out_2969389070201032483[211] = 0;
   out_2969389070201032483[212] = 0;
   out_2969389070201032483[213] = 0;
   out_2969389070201032483[214] = 0;
   out_2969389070201032483[215] = 0;
   out_2969389070201032483[216] = 0;
   out_2969389070201032483[217] = 0;
   out_2969389070201032483[218] = 0;
   out_2969389070201032483[219] = 0;
   out_2969389070201032483[220] = 0;
   out_2969389070201032483[221] = 0;
   out_2969389070201032483[222] = 0;
   out_2969389070201032483[223] = 0;
   out_2969389070201032483[224] = 0;
   out_2969389070201032483[225] = 0;
   out_2969389070201032483[226] = 0;
   out_2969389070201032483[227] = 0;
   out_2969389070201032483[228] = 1;
   out_2969389070201032483[229] = 0;
   out_2969389070201032483[230] = 0;
   out_2969389070201032483[231] = 0;
   out_2969389070201032483[232] = 0;
   out_2969389070201032483[233] = 0;
   out_2969389070201032483[234] = 0;
   out_2969389070201032483[235] = 0;
   out_2969389070201032483[236] = 0;
   out_2969389070201032483[237] = 0;
   out_2969389070201032483[238] = 0;
   out_2969389070201032483[239] = 0;
   out_2969389070201032483[240] = 0;
   out_2969389070201032483[241] = 0;
   out_2969389070201032483[242] = 0;
   out_2969389070201032483[243] = 0;
   out_2969389070201032483[244] = 0;
   out_2969389070201032483[245] = 0;
   out_2969389070201032483[246] = 0;
   out_2969389070201032483[247] = 1;
   out_2969389070201032483[248] = 0;
   out_2969389070201032483[249] = 0;
   out_2969389070201032483[250] = 0;
   out_2969389070201032483[251] = 0;
   out_2969389070201032483[252] = 0;
   out_2969389070201032483[253] = 0;
   out_2969389070201032483[254] = 0;
   out_2969389070201032483[255] = 0;
   out_2969389070201032483[256] = 0;
   out_2969389070201032483[257] = 0;
   out_2969389070201032483[258] = 0;
   out_2969389070201032483[259] = 0;
   out_2969389070201032483[260] = 0;
   out_2969389070201032483[261] = 0;
   out_2969389070201032483[262] = 0;
   out_2969389070201032483[263] = 0;
   out_2969389070201032483[264] = 0;
   out_2969389070201032483[265] = 0;
   out_2969389070201032483[266] = 1;
   out_2969389070201032483[267] = 0;
   out_2969389070201032483[268] = 0;
   out_2969389070201032483[269] = 0;
   out_2969389070201032483[270] = 0;
   out_2969389070201032483[271] = 0;
   out_2969389070201032483[272] = 0;
   out_2969389070201032483[273] = 0;
   out_2969389070201032483[274] = 0;
   out_2969389070201032483[275] = 0;
   out_2969389070201032483[276] = 0;
   out_2969389070201032483[277] = 0;
   out_2969389070201032483[278] = 0;
   out_2969389070201032483[279] = 0;
   out_2969389070201032483[280] = 0;
   out_2969389070201032483[281] = 0;
   out_2969389070201032483[282] = 0;
   out_2969389070201032483[283] = 0;
   out_2969389070201032483[284] = 0;
   out_2969389070201032483[285] = 1;
   out_2969389070201032483[286] = 0;
   out_2969389070201032483[287] = 0;
   out_2969389070201032483[288] = 0;
   out_2969389070201032483[289] = 0;
   out_2969389070201032483[290] = 0;
   out_2969389070201032483[291] = 0;
   out_2969389070201032483[292] = 0;
   out_2969389070201032483[293] = 0;
   out_2969389070201032483[294] = 0;
   out_2969389070201032483[295] = 0;
   out_2969389070201032483[296] = 0;
   out_2969389070201032483[297] = 0;
   out_2969389070201032483[298] = 0;
   out_2969389070201032483[299] = 0;
   out_2969389070201032483[300] = 0;
   out_2969389070201032483[301] = 0;
   out_2969389070201032483[302] = 0;
   out_2969389070201032483[303] = 0;
   out_2969389070201032483[304] = 1;
   out_2969389070201032483[305] = 0;
   out_2969389070201032483[306] = 0;
   out_2969389070201032483[307] = 0;
   out_2969389070201032483[308] = 0;
   out_2969389070201032483[309] = 0;
   out_2969389070201032483[310] = 0;
   out_2969389070201032483[311] = 0;
   out_2969389070201032483[312] = 0;
   out_2969389070201032483[313] = 0;
   out_2969389070201032483[314] = 0;
   out_2969389070201032483[315] = 0;
   out_2969389070201032483[316] = 0;
   out_2969389070201032483[317] = 0;
   out_2969389070201032483[318] = 0;
   out_2969389070201032483[319] = 0;
   out_2969389070201032483[320] = 0;
   out_2969389070201032483[321] = 0;
   out_2969389070201032483[322] = 0;
   out_2969389070201032483[323] = 1;
}
void h_4(double *state, double *unused, double *out_1854639727209469544) {
   out_1854639727209469544[0] = state[6] + state[9];
   out_1854639727209469544[1] = state[7] + state[10];
   out_1854639727209469544[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_269968930268906056) {
   out_269968930268906056[0] = 0;
   out_269968930268906056[1] = 0;
   out_269968930268906056[2] = 0;
   out_269968930268906056[3] = 0;
   out_269968930268906056[4] = 0;
   out_269968930268906056[5] = 0;
   out_269968930268906056[6] = 1;
   out_269968930268906056[7] = 0;
   out_269968930268906056[8] = 0;
   out_269968930268906056[9] = 1;
   out_269968930268906056[10] = 0;
   out_269968930268906056[11] = 0;
   out_269968930268906056[12] = 0;
   out_269968930268906056[13] = 0;
   out_269968930268906056[14] = 0;
   out_269968930268906056[15] = 0;
   out_269968930268906056[16] = 0;
   out_269968930268906056[17] = 0;
   out_269968930268906056[18] = 0;
   out_269968930268906056[19] = 0;
   out_269968930268906056[20] = 0;
   out_269968930268906056[21] = 0;
   out_269968930268906056[22] = 0;
   out_269968930268906056[23] = 0;
   out_269968930268906056[24] = 0;
   out_269968930268906056[25] = 1;
   out_269968930268906056[26] = 0;
   out_269968930268906056[27] = 0;
   out_269968930268906056[28] = 1;
   out_269968930268906056[29] = 0;
   out_269968930268906056[30] = 0;
   out_269968930268906056[31] = 0;
   out_269968930268906056[32] = 0;
   out_269968930268906056[33] = 0;
   out_269968930268906056[34] = 0;
   out_269968930268906056[35] = 0;
   out_269968930268906056[36] = 0;
   out_269968930268906056[37] = 0;
   out_269968930268906056[38] = 0;
   out_269968930268906056[39] = 0;
   out_269968930268906056[40] = 0;
   out_269968930268906056[41] = 0;
   out_269968930268906056[42] = 0;
   out_269968930268906056[43] = 0;
   out_269968930268906056[44] = 1;
   out_269968930268906056[45] = 0;
   out_269968930268906056[46] = 0;
   out_269968930268906056[47] = 1;
   out_269968930268906056[48] = 0;
   out_269968930268906056[49] = 0;
   out_269968930268906056[50] = 0;
   out_269968930268906056[51] = 0;
   out_269968930268906056[52] = 0;
   out_269968930268906056[53] = 0;
}
void h_10(double *state, double *unused, double *out_2214826177905303244) {
   out_2214826177905303244[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2214826177905303244[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2214826177905303244[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_8815669168610743905) {
   out_8815669168610743905[0] = 0;
   out_8815669168610743905[1] = 9.8100000000000005*cos(state[1]);
   out_8815669168610743905[2] = 0;
   out_8815669168610743905[3] = 0;
   out_8815669168610743905[4] = -state[8];
   out_8815669168610743905[5] = state[7];
   out_8815669168610743905[6] = 0;
   out_8815669168610743905[7] = state[5];
   out_8815669168610743905[8] = -state[4];
   out_8815669168610743905[9] = 0;
   out_8815669168610743905[10] = 0;
   out_8815669168610743905[11] = 0;
   out_8815669168610743905[12] = 1;
   out_8815669168610743905[13] = 0;
   out_8815669168610743905[14] = 0;
   out_8815669168610743905[15] = 1;
   out_8815669168610743905[16] = 0;
   out_8815669168610743905[17] = 0;
   out_8815669168610743905[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_8815669168610743905[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_8815669168610743905[20] = 0;
   out_8815669168610743905[21] = state[8];
   out_8815669168610743905[22] = 0;
   out_8815669168610743905[23] = -state[6];
   out_8815669168610743905[24] = -state[5];
   out_8815669168610743905[25] = 0;
   out_8815669168610743905[26] = state[3];
   out_8815669168610743905[27] = 0;
   out_8815669168610743905[28] = 0;
   out_8815669168610743905[29] = 0;
   out_8815669168610743905[30] = 0;
   out_8815669168610743905[31] = 1;
   out_8815669168610743905[32] = 0;
   out_8815669168610743905[33] = 0;
   out_8815669168610743905[34] = 1;
   out_8815669168610743905[35] = 0;
   out_8815669168610743905[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_8815669168610743905[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_8815669168610743905[38] = 0;
   out_8815669168610743905[39] = -state[7];
   out_8815669168610743905[40] = state[6];
   out_8815669168610743905[41] = 0;
   out_8815669168610743905[42] = state[4];
   out_8815669168610743905[43] = -state[3];
   out_8815669168610743905[44] = 0;
   out_8815669168610743905[45] = 0;
   out_8815669168610743905[46] = 0;
   out_8815669168610743905[47] = 0;
   out_8815669168610743905[48] = 0;
   out_8815669168610743905[49] = 0;
   out_8815669168610743905[50] = 1;
   out_8815669168610743905[51] = 0;
   out_8815669168610743905[52] = 0;
   out_8815669168610743905[53] = 1;
}
void h_13(double *state, double *unused, double *out_7189848073171455906) {
   out_7189848073171455906[0] = state[3];
   out_7189848073171455906[1] = state[4];
   out_7189848073171455906[2] = state[5];
}
void H_13(double *state, double *unused, double *out_3482242755601238857) {
   out_3482242755601238857[0] = 0;
   out_3482242755601238857[1] = 0;
   out_3482242755601238857[2] = 0;
   out_3482242755601238857[3] = 1;
   out_3482242755601238857[4] = 0;
   out_3482242755601238857[5] = 0;
   out_3482242755601238857[6] = 0;
   out_3482242755601238857[7] = 0;
   out_3482242755601238857[8] = 0;
   out_3482242755601238857[9] = 0;
   out_3482242755601238857[10] = 0;
   out_3482242755601238857[11] = 0;
   out_3482242755601238857[12] = 0;
   out_3482242755601238857[13] = 0;
   out_3482242755601238857[14] = 0;
   out_3482242755601238857[15] = 0;
   out_3482242755601238857[16] = 0;
   out_3482242755601238857[17] = 0;
   out_3482242755601238857[18] = 0;
   out_3482242755601238857[19] = 0;
   out_3482242755601238857[20] = 0;
   out_3482242755601238857[21] = 0;
   out_3482242755601238857[22] = 1;
   out_3482242755601238857[23] = 0;
   out_3482242755601238857[24] = 0;
   out_3482242755601238857[25] = 0;
   out_3482242755601238857[26] = 0;
   out_3482242755601238857[27] = 0;
   out_3482242755601238857[28] = 0;
   out_3482242755601238857[29] = 0;
   out_3482242755601238857[30] = 0;
   out_3482242755601238857[31] = 0;
   out_3482242755601238857[32] = 0;
   out_3482242755601238857[33] = 0;
   out_3482242755601238857[34] = 0;
   out_3482242755601238857[35] = 0;
   out_3482242755601238857[36] = 0;
   out_3482242755601238857[37] = 0;
   out_3482242755601238857[38] = 0;
   out_3482242755601238857[39] = 0;
   out_3482242755601238857[40] = 0;
   out_3482242755601238857[41] = 1;
   out_3482242755601238857[42] = 0;
   out_3482242755601238857[43] = 0;
   out_3482242755601238857[44] = 0;
   out_3482242755601238857[45] = 0;
   out_3482242755601238857[46] = 0;
   out_3482242755601238857[47] = 0;
   out_3482242755601238857[48] = 0;
   out_3482242755601238857[49] = 0;
   out_3482242755601238857[50] = 0;
   out_3482242755601238857[51] = 0;
   out_3482242755601238857[52] = 0;
   out_3482242755601238857[53] = 0;
}
void h_14(double *state, double *unused, double *out_2380654778222354163) {
   out_2380654778222354163[0] = state[6];
   out_2380654778222354163[1] = state[7];
   out_2380654778222354163[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4233209786608390585) {
   out_4233209786608390585[0] = 0;
   out_4233209786608390585[1] = 0;
   out_4233209786608390585[2] = 0;
   out_4233209786608390585[3] = 0;
   out_4233209786608390585[4] = 0;
   out_4233209786608390585[5] = 0;
   out_4233209786608390585[6] = 1;
   out_4233209786608390585[7] = 0;
   out_4233209786608390585[8] = 0;
   out_4233209786608390585[9] = 0;
   out_4233209786608390585[10] = 0;
   out_4233209786608390585[11] = 0;
   out_4233209786608390585[12] = 0;
   out_4233209786608390585[13] = 0;
   out_4233209786608390585[14] = 0;
   out_4233209786608390585[15] = 0;
   out_4233209786608390585[16] = 0;
   out_4233209786608390585[17] = 0;
   out_4233209786608390585[18] = 0;
   out_4233209786608390585[19] = 0;
   out_4233209786608390585[20] = 0;
   out_4233209786608390585[21] = 0;
   out_4233209786608390585[22] = 0;
   out_4233209786608390585[23] = 0;
   out_4233209786608390585[24] = 0;
   out_4233209786608390585[25] = 1;
   out_4233209786608390585[26] = 0;
   out_4233209786608390585[27] = 0;
   out_4233209786608390585[28] = 0;
   out_4233209786608390585[29] = 0;
   out_4233209786608390585[30] = 0;
   out_4233209786608390585[31] = 0;
   out_4233209786608390585[32] = 0;
   out_4233209786608390585[33] = 0;
   out_4233209786608390585[34] = 0;
   out_4233209786608390585[35] = 0;
   out_4233209786608390585[36] = 0;
   out_4233209786608390585[37] = 0;
   out_4233209786608390585[38] = 0;
   out_4233209786608390585[39] = 0;
   out_4233209786608390585[40] = 0;
   out_4233209786608390585[41] = 0;
   out_4233209786608390585[42] = 0;
   out_4233209786608390585[43] = 0;
   out_4233209786608390585[44] = 1;
   out_4233209786608390585[45] = 0;
   out_4233209786608390585[46] = 0;
   out_4233209786608390585[47] = 0;
   out_4233209786608390585[48] = 0;
   out_4233209786608390585[49] = 0;
   out_4233209786608390585[50] = 0;
   out_4233209786608390585[51] = 0;
   out_4233209786608390585[52] = 0;
   out_4233209786608390585[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_695385451800373050) {
  err_fun(nom_x, delta_x, out_695385451800373050);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3513419238100027883) {
  inv_err_fun(nom_x, true_x, out_3513419238100027883);
}
void pose_H_mod_fun(double *state, double *out_6118050737020864777) {
  H_mod_fun(state, out_6118050737020864777);
}
void pose_f_fun(double *state, double dt, double *out_4351988193126604514) {
  f_fun(state,  dt, out_4351988193126604514);
}
void pose_F_fun(double *state, double dt, double *out_2969389070201032483) {
  F_fun(state,  dt, out_2969389070201032483);
}
void pose_h_4(double *state, double *unused, double *out_1854639727209469544) {
  h_4(state, unused, out_1854639727209469544);
}
void pose_H_4(double *state, double *unused, double *out_269968930268906056) {
  H_4(state, unused, out_269968930268906056);
}
void pose_h_10(double *state, double *unused, double *out_2214826177905303244) {
  h_10(state, unused, out_2214826177905303244);
}
void pose_H_10(double *state, double *unused, double *out_8815669168610743905) {
  H_10(state, unused, out_8815669168610743905);
}
void pose_h_13(double *state, double *unused, double *out_7189848073171455906) {
  h_13(state, unused, out_7189848073171455906);
}
void pose_H_13(double *state, double *unused, double *out_3482242755601238857) {
  H_13(state, unused, out_3482242755601238857);
}
void pose_h_14(double *state, double *unused, double *out_2380654778222354163) {
  h_14(state, unused, out_2380654778222354163);
}
void pose_H_14(double *state, double *unused, double *out_4233209786608390585) {
  H_14(state, unused, out_4233209786608390585);
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
