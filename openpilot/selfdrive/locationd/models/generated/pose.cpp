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
void err_fun(double *nom_x, double *delta_x, double *out_716632891583450328) {
   out_716632891583450328[0] = delta_x[0] + nom_x[0];
   out_716632891583450328[1] = delta_x[1] + nom_x[1];
   out_716632891583450328[2] = delta_x[2] + nom_x[2];
   out_716632891583450328[3] = delta_x[3] + nom_x[3];
   out_716632891583450328[4] = delta_x[4] + nom_x[4];
   out_716632891583450328[5] = delta_x[5] + nom_x[5];
   out_716632891583450328[6] = delta_x[6] + nom_x[6];
   out_716632891583450328[7] = delta_x[7] + nom_x[7];
   out_716632891583450328[8] = delta_x[8] + nom_x[8];
   out_716632891583450328[9] = delta_x[9] + nom_x[9];
   out_716632891583450328[10] = delta_x[10] + nom_x[10];
   out_716632891583450328[11] = delta_x[11] + nom_x[11];
   out_716632891583450328[12] = delta_x[12] + nom_x[12];
   out_716632891583450328[13] = delta_x[13] + nom_x[13];
   out_716632891583450328[14] = delta_x[14] + nom_x[14];
   out_716632891583450328[15] = delta_x[15] + nom_x[15];
   out_716632891583450328[16] = delta_x[16] + nom_x[16];
   out_716632891583450328[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3104130222774257522) {
   out_3104130222774257522[0] = -nom_x[0] + true_x[0];
   out_3104130222774257522[1] = -nom_x[1] + true_x[1];
   out_3104130222774257522[2] = -nom_x[2] + true_x[2];
   out_3104130222774257522[3] = -nom_x[3] + true_x[3];
   out_3104130222774257522[4] = -nom_x[4] + true_x[4];
   out_3104130222774257522[5] = -nom_x[5] + true_x[5];
   out_3104130222774257522[6] = -nom_x[6] + true_x[6];
   out_3104130222774257522[7] = -nom_x[7] + true_x[7];
   out_3104130222774257522[8] = -nom_x[8] + true_x[8];
   out_3104130222774257522[9] = -nom_x[9] + true_x[9];
   out_3104130222774257522[10] = -nom_x[10] + true_x[10];
   out_3104130222774257522[11] = -nom_x[11] + true_x[11];
   out_3104130222774257522[12] = -nom_x[12] + true_x[12];
   out_3104130222774257522[13] = -nom_x[13] + true_x[13];
   out_3104130222774257522[14] = -nom_x[14] + true_x[14];
   out_3104130222774257522[15] = -nom_x[15] + true_x[15];
   out_3104130222774257522[16] = -nom_x[16] + true_x[16];
   out_3104130222774257522[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_8665327190051684991) {
   out_8665327190051684991[0] = 1.0;
   out_8665327190051684991[1] = 0.0;
   out_8665327190051684991[2] = 0.0;
   out_8665327190051684991[3] = 0.0;
   out_8665327190051684991[4] = 0.0;
   out_8665327190051684991[5] = 0.0;
   out_8665327190051684991[6] = 0.0;
   out_8665327190051684991[7] = 0.0;
   out_8665327190051684991[8] = 0.0;
   out_8665327190051684991[9] = 0.0;
   out_8665327190051684991[10] = 0.0;
   out_8665327190051684991[11] = 0.0;
   out_8665327190051684991[12] = 0.0;
   out_8665327190051684991[13] = 0.0;
   out_8665327190051684991[14] = 0.0;
   out_8665327190051684991[15] = 0.0;
   out_8665327190051684991[16] = 0.0;
   out_8665327190051684991[17] = 0.0;
   out_8665327190051684991[18] = 0.0;
   out_8665327190051684991[19] = 1.0;
   out_8665327190051684991[20] = 0.0;
   out_8665327190051684991[21] = 0.0;
   out_8665327190051684991[22] = 0.0;
   out_8665327190051684991[23] = 0.0;
   out_8665327190051684991[24] = 0.0;
   out_8665327190051684991[25] = 0.0;
   out_8665327190051684991[26] = 0.0;
   out_8665327190051684991[27] = 0.0;
   out_8665327190051684991[28] = 0.0;
   out_8665327190051684991[29] = 0.0;
   out_8665327190051684991[30] = 0.0;
   out_8665327190051684991[31] = 0.0;
   out_8665327190051684991[32] = 0.0;
   out_8665327190051684991[33] = 0.0;
   out_8665327190051684991[34] = 0.0;
   out_8665327190051684991[35] = 0.0;
   out_8665327190051684991[36] = 0.0;
   out_8665327190051684991[37] = 0.0;
   out_8665327190051684991[38] = 1.0;
   out_8665327190051684991[39] = 0.0;
   out_8665327190051684991[40] = 0.0;
   out_8665327190051684991[41] = 0.0;
   out_8665327190051684991[42] = 0.0;
   out_8665327190051684991[43] = 0.0;
   out_8665327190051684991[44] = 0.0;
   out_8665327190051684991[45] = 0.0;
   out_8665327190051684991[46] = 0.0;
   out_8665327190051684991[47] = 0.0;
   out_8665327190051684991[48] = 0.0;
   out_8665327190051684991[49] = 0.0;
   out_8665327190051684991[50] = 0.0;
   out_8665327190051684991[51] = 0.0;
   out_8665327190051684991[52] = 0.0;
   out_8665327190051684991[53] = 0.0;
   out_8665327190051684991[54] = 0.0;
   out_8665327190051684991[55] = 0.0;
   out_8665327190051684991[56] = 0.0;
   out_8665327190051684991[57] = 1.0;
   out_8665327190051684991[58] = 0.0;
   out_8665327190051684991[59] = 0.0;
   out_8665327190051684991[60] = 0.0;
   out_8665327190051684991[61] = 0.0;
   out_8665327190051684991[62] = 0.0;
   out_8665327190051684991[63] = 0.0;
   out_8665327190051684991[64] = 0.0;
   out_8665327190051684991[65] = 0.0;
   out_8665327190051684991[66] = 0.0;
   out_8665327190051684991[67] = 0.0;
   out_8665327190051684991[68] = 0.0;
   out_8665327190051684991[69] = 0.0;
   out_8665327190051684991[70] = 0.0;
   out_8665327190051684991[71] = 0.0;
   out_8665327190051684991[72] = 0.0;
   out_8665327190051684991[73] = 0.0;
   out_8665327190051684991[74] = 0.0;
   out_8665327190051684991[75] = 0.0;
   out_8665327190051684991[76] = 1.0;
   out_8665327190051684991[77] = 0.0;
   out_8665327190051684991[78] = 0.0;
   out_8665327190051684991[79] = 0.0;
   out_8665327190051684991[80] = 0.0;
   out_8665327190051684991[81] = 0.0;
   out_8665327190051684991[82] = 0.0;
   out_8665327190051684991[83] = 0.0;
   out_8665327190051684991[84] = 0.0;
   out_8665327190051684991[85] = 0.0;
   out_8665327190051684991[86] = 0.0;
   out_8665327190051684991[87] = 0.0;
   out_8665327190051684991[88] = 0.0;
   out_8665327190051684991[89] = 0.0;
   out_8665327190051684991[90] = 0.0;
   out_8665327190051684991[91] = 0.0;
   out_8665327190051684991[92] = 0.0;
   out_8665327190051684991[93] = 0.0;
   out_8665327190051684991[94] = 0.0;
   out_8665327190051684991[95] = 1.0;
   out_8665327190051684991[96] = 0.0;
   out_8665327190051684991[97] = 0.0;
   out_8665327190051684991[98] = 0.0;
   out_8665327190051684991[99] = 0.0;
   out_8665327190051684991[100] = 0.0;
   out_8665327190051684991[101] = 0.0;
   out_8665327190051684991[102] = 0.0;
   out_8665327190051684991[103] = 0.0;
   out_8665327190051684991[104] = 0.0;
   out_8665327190051684991[105] = 0.0;
   out_8665327190051684991[106] = 0.0;
   out_8665327190051684991[107] = 0.0;
   out_8665327190051684991[108] = 0.0;
   out_8665327190051684991[109] = 0.0;
   out_8665327190051684991[110] = 0.0;
   out_8665327190051684991[111] = 0.0;
   out_8665327190051684991[112] = 0.0;
   out_8665327190051684991[113] = 0.0;
   out_8665327190051684991[114] = 1.0;
   out_8665327190051684991[115] = 0.0;
   out_8665327190051684991[116] = 0.0;
   out_8665327190051684991[117] = 0.0;
   out_8665327190051684991[118] = 0.0;
   out_8665327190051684991[119] = 0.0;
   out_8665327190051684991[120] = 0.0;
   out_8665327190051684991[121] = 0.0;
   out_8665327190051684991[122] = 0.0;
   out_8665327190051684991[123] = 0.0;
   out_8665327190051684991[124] = 0.0;
   out_8665327190051684991[125] = 0.0;
   out_8665327190051684991[126] = 0.0;
   out_8665327190051684991[127] = 0.0;
   out_8665327190051684991[128] = 0.0;
   out_8665327190051684991[129] = 0.0;
   out_8665327190051684991[130] = 0.0;
   out_8665327190051684991[131] = 0.0;
   out_8665327190051684991[132] = 0.0;
   out_8665327190051684991[133] = 1.0;
   out_8665327190051684991[134] = 0.0;
   out_8665327190051684991[135] = 0.0;
   out_8665327190051684991[136] = 0.0;
   out_8665327190051684991[137] = 0.0;
   out_8665327190051684991[138] = 0.0;
   out_8665327190051684991[139] = 0.0;
   out_8665327190051684991[140] = 0.0;
   out_8665327190051684991[141] = 0.0;
   out_8665327190051684991[142] = 0.0;
   out_8665327190051684991[143] = 0.0;
   out_8665327190051684991[144] = 0.0;
   out_8665327190051684991[145] = 0.0;
   out_8665327190051684991[146] = 0.0;
   out_8665327190051684991[147] = 0.0;
   out_8665327190051684991[148] = 0.0;
   out_8665327190051684991[149] = 0.0;
   out_8665327190051684991[150] = 0.0;
   out_8665327190051684991[151] = 0.0;
   out_8665327190051684991[152] = 1.0;
   out_8665327190051684991[153] = 0.0;
   out_8665327190051684991[154] = 0.0;
   out_8665327190051684991[155] = 0.0;
   out_8665327190051684991[156] = 0.0;
   out_8665327190051684991[157] = 0.0;
   out_8665327190051684991[158] = 0.0;
   out_8665327190051684991[159] = 0.0;
   out_8665327190051684991[160] = 0.0;
   out_8665327190051684991[161] = 0.0;
   out_8665327190051684991[162] = 0.0;
   out_8665327190051684991[163] = 0.0;
   out_8665327190051684991[164] = 0.0;
   out_8665327190051684991[165] = 0.0;
   out_8665327190051684991[166] = 0.0;
   out_8665327190051684991[167] = 0.0;
   out_8665327190051684991[168] = 0.0;
   out_8665327190051684991[169] = 0.0;
   out_8665327190051684991[170] = 0.0;
   out_8665327190051684991[171] = 1.0;
   out_8665327190051684991[172] = 0.0;
   out_8665327190051684991[173] = 0.0;
   out_8665327190051684991[174] = 0.0;
   out_8665327190051684991[175] = 0.0;
   out_8665327190051684991[176] = 0.0;
   out_8665327190051684991[177] = 0.0;
   out_8665327190051684991[178] = 0.0;
   out_8665327190051684991[179] = 0.0;
   out_8665327190051684991[180] = 0.0;
   out_8665327190051684991[181] = 0.0;
   out_8665327190051684991[182] = 0.0;
   out_8665327190051684991[183] = 0.0;
   out_8665327190051684991[184] = 0.0;
   out_8665327190051684991[185] = 0.0;
   out_8665327190051684991[186] = 0.0;
   out_8665327190051684991[187] = 0.0;
   out_8665327190051684991[188] = 0.0;
   out_8665327190051684991[189] = 0.0;
   out_8665327190051684991[190] = 1.0;
   out_8665327190051684991[191] = 0.0;
   out_8665327190051684991[192] = 0.0;
   out_8665327190051684991[193] = 0.0;
   out_8665327190051684991[194] = 0.0;
   out_8665327190051684991[195] = 0.0;
   out_8665327190051684991[196] = 0.0;
   out_8665327190051684991[197] = 0.0;
   out_8665327190051684991[198] = 0.0;
   out_8665327190051684991[199] = 0.0;
   out_8665327190051684991[200] = 0.0;
   out_8665327190051684991[201] = 0.0;
   out_8665327190051684991[202] = 0.0;
   out_8665327190051684991[203] = 0.0;
   out_8665327190051684991[204] = 0.0;
   out_8665327190051684991[205] = 0.0;
   out_8665327190051684991[206] = 0.0;
   out_8665327190051684991[207] = 0.0;
   out_8665327190051684991[208] = 0.0;
   out_8665327190051684991[209] = 1.0;
   out_8665327190051684991[210] = 0.0;
   out_8665327190051684991[211] = 0.0;
   out_8665327190051684991[212] = 0.0;
   out_8665327190051684991[213] = 0.0;
   out_8665327190051684991[214] = 0.0;
   out_8665327190051684991[215] = 0.0;
   out_8665327190051684991[216] = 0.0;
   out_8665327190051684991[217] = 0.0;
   out_8665327190051684991[218] = 0.0;
   out_8665327190051684991[219] = 0.0;
   out_8665327190051684991[220] = 0.0;
   out_8665327190051684991[221] = 0.0;
   out_8665327190051684991[222] = 0.0;
   out_8665327190051684991[223] = 0.0;
   out_8665327190051684991[224] = 0.0;
   out_8665327190051684991[225] = 0.0;
   out_8665327190051684991[226] = 0.0;
   out_8665327190051684991[227] = 0.0;
   out_8665327190051684991[228] = 1.0;
   out_8665327190051684991[229] = 0.0;
   out_8665327190051684991[230] = 0.0;
   out_8665327190051684991[231] = 0.0;
   out_8665327190051684991[232] = 0.0;
   out_8665327190051684991[233] = 0.0;
   out_8665327190051684991[234] = 0.0;
   out_8665327190051684991[235] = 0.0;
   out_8665327190051684991[236] = 0.0;
   out_8665327190051684991[237] = 0.0;
   out_8665327190051684991[238] = 0.0;
   out_8665327190051684991[239] = 0.0;
   out_8665327190051684991[240] = 0.0;
   out_8665327190051684991[241] = 0.0;
   out_8665327190051684991[242] = 0.0;
   out_8665327190051684991[243] = 0.0;
   out_8665327190051684991[244] = 0.0;
   out_8665327190051684991[245] = 0.0;
   out_8665327190051684991[246] = 0.0;
   out_8665327190051684991[247] = 1.0;
   out_8665327190051684991[248] = 0.0;
   out_8665327190051684991[249] = 0.0;
   out_8665327190051684991[250] = 0.0;
   out_8665327190051684991[251] = 0.0;
   out_8665327190051684991[252] = 0.0;
   out_8665327190051684991[253] = 0.0;
   out_8665327190051684991[254] = 0.0;
   out_8665327190051684991[255] = 0.0;
   out_8665327190051684991[256] = 0.0;
   out_8665327190051684991[257] = 0.0;
   out_8665327190051684991[258] = 0.0;
   out_8665327190051684991[259] = 0.0;
   out_8665327190051684991[260] = 0.0;
   out_8665327190051684991[261] = 0.0;
   out_8665327190051684991[262] = 0.0;
   out_8665327190051684991[263] = 0.0;
   out_8665327190051684991[264] = 0.0;
   out_8665327190051684991[265] = 0.0;
   out_8665327190051684991[266] = 1.0;
   out_8665327190051684991[267] = 0.0;
   out_8665327190051684991[268] = 0.0;
   out_8665327190051684991[269] = 0.0;
   out_8665327190051684991[270] = 0.0;
   out_8665327190051684991[271] = 0.0;
   out_8665327190051684991[272] = 0.0;
   out_8665327190051684991[273] = 0.0;
   out_8665327190051684991[274] = 0.0;
   out_8665327190051684991[275] = 0.0;
   out_8665327190051684991[276] = 0.0;
   out_8665327190051684991[277] = 0.0;
   out_8665327190051684991[278] = 0.0;
   out_8665327190051684991[279] = 0.0;
   out_8665327190051684991[280] = 0.0;
   out_8665327190051684991[281] = 0.0;
   out_8665327190051684991[282] = 0.0;
   out_8665327190051684991[283] = 0.0;
   out_8665327190051684991[284] = 0.0;
   out_8665327190051684991[285] = 1.0;
   out_8665327190051684991[286] = 0.0;
   out_8665327190051684991[287] = 0.0;
   out_8665327190051684991[288] = 0.0;
   out_8665327190051684991[289] = 0.0;
   out_8665327190051684991[290] = 0.0;
   out_8665327190051684991[291] = 0.0;
   out_8665327190051684991[292] = 0.0;
   out_8665327190051684991[293] = 0.0;
   out_8665327190051684991[294] = 0.0;
   out_8665327190051684991[295] = 0.0;
   out_8665327190051684991[296] = 0.0;
   out_8665327190051684991[297] = 0.0;
   out_8665327190051684991[298] = 0.0;
   out_8665327190051684991[299] = 0.0;
   out_8665327190051684991[300] = 0.0;
   out_8665327190051684991[301] = 0.0;
   out_8665327190051684991[302] = 0.0;
   out_8665327190051684991[303] = 0.0;
   out_8665327190051684991[304] = 1.0;
   out_8665327190051684991[305] = 0.0;
   out_8665327190051684991[306] = 0.0;
   out_8665327190051684991[307] = 0.0;
   out_8665327190051684991[308] = 0.0;
   out_8665327190051684991[309] = 0.0;
   out_8665327190051684991[310] = 0.0;
   out_8665327190051684991[311] = 0.0;
   out_8665327190051684991[312] = 0.0;
   out_8665327190051684991[313] = 0.0;
   out_8665327190051684991[314] = 0.0;
   out_8665327190051684991[315] = 0.0;
   out_8665327190051684991[316] = 0.0;
   out_8665327190051684991[317] = 0.0;
   out_8665327190051684991[318] = 0.0;
   out_8665327190051684991[319] = 0.0;
   out_8665327190051684991[320] = 0.0;
   out_8665327190051684991[321] = 0.0;
   out_8665327190051684991[322] = 0.0;
   out_8665327190051684991[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_1781769764659626549) {
   out_1781769764659626549[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_1781769764659626549[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_1781769764659626549[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_1781769764659626549[3] = dt*state[12] + state[3];
   out_1781769764659626549[4] = dt*state[13] + state[4];
   out_1781769764659626549[5] = dt*state[14] + state[5];
   out_1781769764659626549[6] = state[6];
   out_1781769764659626549[7] = state[7];
   out_1781769764659626549[8] = state[8];
   out_1781769764659626549[9] = state[9];
   out_1781769764659626549[10] = state[10];
   out_1781769764659626549[11] = state[11];
   out_1781769764659626549[12] = state[12];
   out_1781769764659626549[13] = state[13];
   out_1781769764659626549[14] = state[14];
   out_1781769764659626549[15] = state[15];
   out_1781769764659626549[16] = state[16];
   out_1781769764659626549[17] = state[17];
}
void F_fun(double *state, double dt, double *out_3937534521909816001) {
   out_3937534521909816001[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3937534521909816001[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3937534521909816001[2] = 0;
   out_3937534521909816001[3] = 0;
   out_3937534521909816001[4] = 0;
   out_3937534521909816001[5] = 0;
   out_3937534521909816001[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3937534521909816001[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3937534521909816001[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3937534521909816001[9] = 0;
   out_3937534521909816001[10] = 0;
   out_3937534521909816001[11] = 0;
   out_3937534521909816001[12] = 0;
   out_3937534521909816001[13] = 0;
   out_3937534521909816001[14] = 0;
   out_3937534521909816001[15] = 0;
   out_3937534521909816001[16] = 0;
   out_3937534521909816001[17] = 0;
   out_3937534521909816001[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3937534521909816001[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3937534521909816001[20] = 0;
   out_3937534521909816001[21] = 0;
   out_3937534521909816001[22] = 0;
   out_3937534521909816001[23] = 0;
   out_3937534521909816001[24] = 0;
   out_3937534521909816001[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3937534521909816001[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3937534521909816001[27] = 0;
   out_3937534521909816001[28] = 0;
   out_3937534521909816001[29] = 0;
   out_3937534521909816001[30] = 0;
   out_3937534521909816001[31] = 0;
   out_3937534521909816001[32] = 0;
   out_3937534521909816001[33] = 0;
   out_3937534521909816001[34] = 0;
   out_3937534521909816001[35] = 0;
   out_3937534521909816001[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3937534521909816001[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3937534521909816001[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3937534521909816001[39] = 0;
   out_3937534521909816001[40] = 0;
   out_3937534521909816001[41] = 0;
   out_3937534521909816001[42] = 0;
   out_3937534521909816001[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3937534521909816001[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3937534521909816001[45] = 0;
   out_3937534521909816001[46] = 0;
   out_3937534521909816001[47] = 0;
   out_3937534521909816001[48] = 0;
   out_3937534521909816001[49] = 0;
   out_3937534521909816001[50] = 0;
   out_3937534521909816001[51] = 0;
   out_3937534521909816001[52] = 0;
   out_3937534521909816001[53] = 0;
   out_3937534521909816001[54] = 0;
   out_3937534521909816001[55] = 0;
   out_3937534521909816001[56] = 0;
   out_3937534521909816001[57] = 1;
   out_3937534521909816001[58] = 0;
   out_3937534521909816001[59] = 0;
   out_3937534521909816001[60] = 0;
   out_3937534521909816001[61] = 0;
   out_3937534521909816001[62] = 0;
   out_3937534521909816001[63] = 0;
   out_3937534521909816001[64] = 0;
   out_3937534521909816001[65] = 0;
   out_3937534521909816001[66] = dt;
   out_3937534521909816001[67] = 0;
   out_3937534521909816001[68] = 0;
   out_3937534521909816001[69] = 0;
   out_3937534521909816001[70] = 0;
   out_3937534521909816001[71] = 0;
   out_3937534521909816001[72] = 0;
   out_3937534521909816001[73] = 0;
   out_3937534521909816001[74] = 0;
   out_3937534521909816001[75] = 0;
   out_3937534521909816001[76] = 1;
   out_3937534521909816001[77] = 0;
   out_3937534521909816001[78] = 0;
   out_3937534521909816001[79] = 0;
   out_3937534521909816001[80] = 0;
   out_3937534521909816001[81] = 0;
   out_3937534521909816001[82] = 0;
   out_3937534521909816001[83] = 0;
   out_3937534521909816001[84] = 0;
   out_3937534521909816001[85] = dt;
   out_3937534521909816001[86] = 0;
   out_3937534521909816001[87] = 0;
   out_3937534521909816001[88] = 0;
   out_3937534521909816001[89] = 0;
   out_3937534521909816001[90] = 0;
   out_3937534521909816001[91] = 0;
   out_3937534521909816001[92] = 0;
   out_3937534521909816001[93] = 0;
   out_3937534521909816001[94] = 0;
   out_3937534521909816001[95] = 1;
   out_3937534521909816001[96] = 0;
   out_3937534521909816001[97] = 0;
   out_3937534521909816001[98] = 0;
   out_3937534521909816001[99] = 0;
   out_3937534521909816001[100] = 0;
   out_3937534521909816001[101] = 0;
   out_3937534521909816001[102] = 0;
   out_3937534521909816001[103] = 0;
   out_3937534521909816001[104] = dt;
   out_3937534521909816001[105] = 0;
   out_3937534521909816001[106] = 0;
   out_3937534521909816001[107] = 0;
   out_3937534521909816001[108] = 0;
   out_3937534521909816001[109] = 0;
   out_3937534521909816001[110] = 0;
   out_3937534521909816001[111] = 0;
   out_3937534521909816001[112] = 0;
   out_3937534521909816001[113] = 0;
   out_3937534521909816001[114] = 1;
   out_3937534521909816001[115] = 0;
   out_3937534521909816001[116] = 0;
   out_3937534521909816001[117] = 0;
   out_3937534521909816001[118] = 0;
   out_3937534521909816001[119] = 0;
   out_3937534521909816001[120] = 0;
   out_3937534521909816001[121] = 0;
   out_3937534521909816001[122] = 0;
   out_3937534521909816001[123] = 0;
   out_3937534521909816001[124] = 0;
   out_3937534521909816001[125] = 0;
   out_3937534521909816001[126] = 0;
   out_3937534521909816001[127] = 0;
   out_3937534521909816001[128] = 0;
   out_3937534521909816001[129] = 0;
   out_3937534521909816001[130] = 0;
   out_3937534521909816001[131] = 0;
   out_3937534521909816001[132] = 0;
   out_3937534521909816001[133] = 1;
   out_3937534521909816001[134] = 0;
   out_3937534521909816001[135] = 0;
   out_3937534521909816001[136] = 0;
   out_3937534521909816001[137] = 0;
   out_3937534521909816001[138] = 0;
   out_3937534521909816001[139] = 0;
   out_3937534521909816001[140] = 0;
   out_3937534521909816001[141] = 0;
   out_3937534521909816001[142] = 0;
   out_3937534521909816001[143] = 0;
   out_3937534521909816001[144] = 0;
   out_3937534521909816001[145] = 0;
   out_3937534521909816001[146] = 0;
   out_3937534521909816001[147] = 0;
   out_3937534521909816001[148] = 0;
   out_3937534521909816001[149] = 0;
   out_3937534521909816001[150] = 0;
   out_3937534521909816001[151] = 0;
   out_3937534521909816001[152] = 1;
   out_3937534521909816001[153] = 0;
   out_3937534521909816001[154] = 0;
   out_3937534521909816001[155] = 0;
   out_3937534521909816001[156] = 0;
   out_3937534521909816001[157] = 0;
   out_3937534521909816001[158] = 0;
   out_3937534521909816001[159] = 0;
   out_3937534521909816001[160] = 0;
   out_3937534521909816001[161] = 0;
   out_3937534521909816001[162] = 0;
   out_3937534521909816001[163] = 0;
   out_3937534521909816001[164] = 0;
   out_3937534521909816001[165] = 0;
   out_3937534521909816001[166] = 0;
   out_3937534521909816001[167] = 0;
   out_3937534521909816001[168] = 0;
   out_3937534521909816001[169] = 0;
   out_3937534521909816001[170] = 0;
   out_3937534521909816001[171] = 1;
   out_3937534521909816001[172] = 0;
   out_3937534521909816001[173] = 0;
   out_3937534521909816001[174] = 0;
   out_3937534521909816001[175] = 0;
   out_3937534521909816001[176] = 0;
   out_3937534521909816001[177] = 0;
   out_3937534521909816001[178] = 0;
   out_3937534521909816001[179] = 0;
   out_3937534521909816001[180] = 0;
   out_3937534521909816001[181] = 0;
   out_3937534521909816001[182] = 0;
   out_3937534521909816001[183] = 0;
   out_3937534521909816001[184] = 0;
   out_3937534521909816001[185] = 0;
   out_3937534521909816001[186] = 0;
   out_3937534521909816001[187] = 0;
   out_3937534521909816001[188] = 0;
   out_3937534521909816001[189] = 0;
   out_3937534521909816001[190] = 1;
   out_3937534521909816001[191] = 0;
   out_3937534521909816001[192] = 0;
   out_3937534521909816001[193] = 0;
   out_3937534521909816001[194] = 0;
   out_3937534521909816001[195] = 0;
   out_3937534521909816001[196] = 0;
   out_3937534521909816001[197] = 0;
   out_3937534521909816001[198] = 0;
   out_3937534521909816001[199] = 0;
   out_3937534521909816001[200] = 0;
   out_3937534521909816001[201] = 0;
   out_3937534521909816001[202] = 0;
   out_3937534521909816001[203] = 0;
   out_3937534521909816001[204] = 0;
   out_3937534521909816001[205] = 0;
   out_3937534521909816001[206] = 0;
   out_3937534521909816001[207] = 0;
   out_3937534521909816001[208] = 0;
   out_3937534521909816001[209] = 1;
   out_3937534521909816001[210] = 0;
   out_3937534521909816001[211] = 0;
   out_3937534521909816001[212] = 0;
   out_3937534521909816001[213] = 0;
   out_3937534521909816001[214] = 0;
   out_3937534521909816001[215] = 0;
   out_3937534521909816001[216] = 0;
   out_3937534521909816001[217] = 0;
   out_3937534521909816001[218] = 0;
   out_3937534521909816001[219] = 0;
   out_3937534521909816001[220] = 0;
   out_3937534521909816001[221] = 0;
   out_3937534521909816001[222] = 0;
   out_3937534521909816001[223] = 0;
   out_3937534521909816001[224] = 0;
   out_3937534521909816001[225] = 0;
   out_3937534521909816001[226] = 0;
   out_3937534521909816001[227] = 0;
   out_3937534521909816001[228] = 1;
   out_3937534521909816001[229] = 0;
   out_3937534521909816001[230] = 0;
   out_3937534521909816001[231] = 0;
   out_3937534521909816001[232] = 0;
   out_3937534521909816001[233] = 0;
   out_3937534521909816001[234] = 0;
   out_3937534521909816001[235] = 0;
   out_3937534521909816001[236] = 0;
   out_3937534521909816001[237] = 0;
   out_3937534521909816001[238] = 0;
   out_3937534521909816001[239] = 0;
   out_3937534521909816001[240] = 0;
   out_3937534521909816001[241] = 0;
   out_3937534521909816001[242] = 0;
   out_3937534521909816001[243] = 0;
   out_3937534521909816001[244] = 0;
   out_3937534521909816001[245] = 0;
   out_3937534521909816001[246] = 0;
   out_3937534521909816001[247] = 1;
   out_3937534521909816001[248] = 0;
   out_3937534521909816001[249] = 0;
   out_3937534521909816001[250] = 0;
   out_3937534521909816001[251] = 0;
   out_3937534521909816001[252] = 0;
   out_3937534521909816001[253] = 0;
   out_3937534521909816001[254] = 0;
   out_3937534521909816001[255] = 0;
   out_3937534521909816001[256] = 0;
   out_3937534521909816001[257] = 0;
   out_3937534521909816001[258] = 0;
   out_3937534521909816001[259] = 0;
   out_3937534521909816001[260] = 0;
   out_3937534521909816001[261] = 0;
   out_3937534521909816001[262] = 0;
   out_3937534521909816001[263] = 0;
   out_3937534521909816001[264] = 0;
   out_3937534521909816001[265] = 0;
   out_3937534521909816001[266] = 1;
   out_3937534521909816001[267] = 0;
   out_3937534521909816001[268] = 0;
   out_3937534521909816001[269] = 0;
   out_3937534521909816001[270] = 0;
   out_3937534521909816001[271] = 0;
   out_3937534521909816001[272] = 0;
   out_3937534521909816001[273] = 0;
   out_3937534521909816001[274] = 0;
   out_3937534521909816001[275] = 0;
   out_3937534521909816001[276] = 0;
   out_3937534521909816001[277] = 0;
   out_3937534521909816001[278] = 0;
   out_3937534521909816001[279] = 0;
   out_3937534521909816001[280] = 0;
   out_3937534521909816001[281] = 0;
   out_3937534521909816001[282] = 0;
   out_3937534521909816001[283] = 0;
   out_3937534521909816001[284] = 0;
   out_3937534521909816001[285] = 1;
   out_3937534521909816001[286] = 0;
   out_3937534521909816001[287] = 0;
   out_3937534521909816001[288] = 0;
   out_3937534521909816001[289] = 0;
   out_3937534521909816001[290] = 0;
   out_3937534521909816001[291] = 0;
   out_3937534521909816001[292] = 0;
   out_3937534521909816001[293] = 0;
   out_3937534521909816001[294] = 0;
   out_3937534521909816001[295] = 0;
   out_3937534521909816001[296] = 0;
   out_3937534521909816001[297] = 0;
   out_3937534521909816001[298] = 0;
   out_3937534521909816001[299] = 0;
   out_3937534521909816001[300] = 0;
   out_3937534521909816001[301] = 0;
   out_3937534521909816001[302] = 0;
   out_3937534521909816001[303] = 0;
   out_3937534521909816001[304] = 1;
   out_3937534521909816001[305] = 0;
   out_3937534521909816001[306] = 0;
   out_3937534521909816001[307] = 0;
   out_3937534521909816001[308] = 0;
   out_3937534521909816001[309] = 0;
   out_3937534521909816001[310] = 0;
   out_3937534521909816001[311] = 0;
   out_3937534521909816001[312] = 0;
   out_3937534521909816001[313] = 0;
   out_3937534521909816001[314] = 0;
   out_3937534521909816001[315] = 0;
   out_3937534521909816001[316] = 0;
   out_3937534521909816001[317] = 0;
   out_3937534521909816001[318] = 0;
   out_3937534521909816001[319] = 0;
   out_3937534521909816001[320] = 0;
   out_3937534521909816001[321] = 0;
   out_3937534521909816001[322] = 0;
   out_3937534521909816001[323] = 1;
}
void h_4(double *state, double *unused, double *out_4373223397940532151) {
   out_4373223397940532151[0] = state[6] + state[9];
   out_4373223397940532151[1] = state[7] + state[10];
   out_4373223397940532151[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7961718157751569764) {
   out_7961718157751569764[0] = 0;
   out_7961718157751569764[1] = 0;
   out_7961718157751569764[2] = 0;
   out_7961718157751569764[3] = 0;
   out_7961718157751569764[4] = 0;
   out_7961718157751569764[5] = 0;
   out_7961718157751569764[6] = 1;
   out_7961718157751569764[7] = 0;
   out_7961718157751569764[8] = 0;
   out_7961718157751569764[9] = 1;
   out_7961718157751569764[10] = 0;
   out_7961718157751569764[11] = 0;
   out_7961718157751569764[12] = 0;
   out_7961718157751569764[13] = 0;
   out_7961718157751569764[14] = 0;
   out_7961718157751569764[15] = 0;
   out_7961718157751569764[16] = 0;
   out_7961718157751569764[17] = 0;
   out_7961718157751569764[18] = 0;
   out_7961718157751569764[19] = 0;
   out_7961718157751569764[20] = 0;
   out_7961718157751569764[21] = 0;
   out_7961718157751569764[22] = 0;
   out_7961718157751569764[23] = 0;
   out_7961718157751569764[24] = 0;
   out_7961718157751569764[25] = 1;
   out_7961718157751569764[26] = 0;
   out_7961718157751569764[27] = 0;
   out_7961718157751569764[28] = 1;
   out_7961718157751569764[29] = 0;
   out_7961718157751569764[30] = 0;
   out_7961718157751569764[31] = 0;
   out_7961718157751569764[32] = 0;
   out_7961718157751569764[33] = 0;
   out_7961718157751569764[34] = 0;
   out_7961718157751569764[35] = 0;
   out_7961718157751569764[36] = 0;
   out_7961718157751569764[37] = 0;
   out_7961718157751569764[38] = 0;
   out_7961718157751569764[39] = 0;
   out_7961718157751569764[40] = 0;
   out_7961718157751569764[41] = 0;
   out_7961718157751569764[42] = 0;
   out_7961718157751569764[43] = 0;
   out_7961718157751569764[44] = 1;
   out_7961718157751569764[45] = 0;
   out_7961718157751569764[46] = 0;
   out_7961718157751569764[47] = 1;
   out_7961718157751569764[48] = 0;
   out_7961718157751569764[49] = 0;
   out_7961718157751569764[50] = 0;
   out_7961718157751569764[51] = 0;
   out_7961718157751569764[52] = 0;
   out_7961718157751569764[53] = 0;
}
void h_10(double *state, double *unused, double *out_6286469033191029799) {
   out_6286469033191029799[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_6286469033191029799[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_6286469033191029799[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_5531647416424197146) {
   out_5531647416424197146[0] = 0;
   out_5531647416424197146[1] = 9.8100000000000005*cos(state[1]);
   out_5531647416424197146[2] = 0;
   out_5531647416424197146[3] = 0;
   out_5531647416424197146[4] = -state[8];
   out_5531647416424197146[5] = state[7];
   out_5531647416424197146[6] = 0;
   out_5531647416424197146[7] = state[5];
   out_5531647416424197146[8] = -state[4];
   out_5531647416424197146[9] = 0;
   out_5531647416424197146[10] = 0;
   out_5531647416424197146[11] = 0;
   out_5531647416424197146[12] = 1;
   out_5531647416424197146[13] = 0;
   out_5531647416424197146[14] = 0;
   out_5531647416424197146[15] = 1;
   out_5531647416424197146[16] = 0;
   out_5531647416424197146[17] = 0;
   out_5531647416424197146[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_5531647416424197146[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_5531647416424197146[20] = 0;
   out_5531647416424197146[21] = state[8];
   out_5531647416424197146[22] = 0;
   out_5531647416424197146[23] = -state[6];
   out_5531647416424197146[24] = -state[5];
   out_5531647416424197146[25] = 0;
   out_5531647416424197146[26] = state[3];
   out_5531647416424197146[27] = 0;
   out_5531647416424197146[28] = 0;
   out_5531647416424197146[29] = 0;
   out_5531647416424197146[30] = 0;
   out_5531647416424197146[31] = 1;
   out_5531647416424197146[32] = 0;
   out_5531647416424197146[33] = 0;
   out_5531647416424197146[34] = 1;
   out_5531647416424197146[35] = 0;
   out_5531647416424197146[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_5531647416424197146[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_5531647416424197146[38] = 0;
   out_5531647416424197146[39] = -state[7];
   out_5531647416424197146[40] = state[6];
   out_5531647416424197146[41] = 0;
   out_5531647416424197146[42] = state[4];
   out_5531647416424197146[43] = -state[3];
   out_5531647416424197146[44] = 0;
   out_5531647416424197146[45] = 0;
   out_5531647416424197146[46] = 0;
   out_5531647416424197146[47] = 0;
   out_5531647416424197146[48] = 0;
   out_5531647416424197146[49] = 0;
   out_5531647416424197146[50] = 1;
   out_5531647416424197146[51] = 0;
   out_5531647416424197146[52] = 0;
   out_5531647416424197146[53] = 1;
}
void h_13(double *state, double *unused, double *out_5115075594500414792) {
   out_5115075594500414792[0] = state[3];
   out_5115075594500414792[1] = state[4];
   out_5115075594500414792[2] = state[5];
}
void H_13(double *state, double *unused, double *out_7272752090625649051) {
   out_7272752090625649051[0] = 0;
   out_7272752090625649051[1] = 0;
   out_7272752090625649051[2] = 0;
   out_7272752090625649051[3] = 1;
   out_7272752090625649051[4] = 0;
   out_7272752090625649051[5] = 0;
   out_7272752090625649051[6] = 0;
   out_7272752090625649051[7] = 0;
   out_7272752090625649051[8] = 0;
   out_7272752090625649051[9] = 0;
   out_7272752090625649051[10] = 0;
   out_7272752090625649051[11] = 0;
   out_7272752090625649051[12] = 0;
   out_7272752090625649051[13] = 0;
   out_7272752090625649051[14] = 0;
   out_7272752090625649051[15] = 0;
   out_7272752090625649051[16] = 0;
   out_7272752090625649051[17] = 0;
   out_7272752090625649051[18] = 0;
   out_7272752090625649051[19] = 0;
   out_7272752090625649051[20] = 0;
   out_7272752090625649051[21] = 0;
   out_7272752090625649051[22] = 1;
   out_7272752090625649051[23] = 0;
   out_7272752090625649051[24] = 0;
   out_7272752090625649051[25] = 0;
   out_7272752090625649051[26] = 0;
   out_7272752090625649051[27] = 0;
   out_7272752090625649051[28] = 0;
   out_7272752090625649051[29] = 0;
   out_7272752090625649051[30] = 0;
   out_7272752090625649051[31] = 0;
   out_7272752090625649051[32] = 0;
   out_7272752090625649051[33] = 0;
   out_7272752090625649051[34] = 0;
   out_7272752090625649051[35] = 0;
   out_7272752090625649051[36] = 0;
   out_7272752090625649051[37] = 0;
   out_7272752090625649051[38] = 0;
   out_7272752090625649051[39] = 0;
   out_7272752090625649051[40] = 0;
   out_7272752090625649051[41] = 1;
   out_7272752090625649051[42] = 0;
   out_7272752090625649051[43] = 0;
   out_7272752090625649051[44] = 0;
   out_7272752090625649051[45] = 0;
   out_7272752090625649051[46] = 0;
   out_7272752090625649051[47] = 0;
   out_7272752090625649051[48] = 0;
   out_7272752090625649051[49] = 0;
   out_7272752090625649051[50] = 0;
   out_7272752090625649051[51] = 0;
   out_7272752090625649051[52] = 0;
   out_7272752090625649051[53] = 0;
}
void h_14(double *state, double *unused, double *out_7724198509017227969) {
   out_7724198509017227969[0] = state[6];
   out_7724198509017227969[1] = state[7];
   out_7724198509017227969[2] = state[8];
}
void H_14(double *state, double *unused, double *out_6521785059618497323) {
   out_6521785059618497323[0] = 0;
   out_6521785059618497323[1] = 0;
   out_6521785059618497323[2] = 0;
   out_6521785059618497323[3] = 0;
   out_6521785059618497323[4] = 0;
   out_6521785059618497323[5] = 0;
   out_6521785059618497323[6] = 1;
   out_6521785059618497323[7] = 0;
   out_6521785059618497323[8] = 0;
   out_6521785059618497323[9] = 0;
   out_6521785059618497323[10] = 0;
   out_6521785059618497323[11] = 0;
   out_6521785059618497323[12] = 0;
   out_6521785059618497323[13] = 0;
   out_6521785059618497323[14] = 0;
   out_6521785059618497323[15] = 0;
   out_6521785059618497323[16] = 0;
   out_6521785059618497323[17] = 0;
   out_6521785059618497323[18] = 0;
   out_6521785059618497323[19] = 0;
   out_6521785059618497323[20] = 0;
   out_6521785059618497323[21] = 0;
   out_6521785059618497323[22] = 0;
   out_6521785059618497323[23] = 0;
   out_6521785059618497323[24] = 0;
   out_6521785059618497323[25] = 1;
   out_6521785059618497323[26] = 0;
   out_6521785059618497323[27] = 0;
   out_6521785059618497323[28] = 0;
   out_6521785059618497323[29] = 0;
   out_6521785059618497323[30] = 0;
   out_6521785059618497323[31] = 0;
   out_6521785059618497323[32] = 0;
   out_6521785059618497323[33] = 0;
   out_6521785059618497323[34] = 0;
   out_6521785059618497323[35] = 0;
   out_6521785059618497323[36] = 0;
   out_6521785059618497323[37] = 0;
   out_6521785059618497323[38] = 0;
   out_6521785059618497323[39] = 0;
   out_6521785059618497323[40] = 0;
   out_6521785059618497323[41] = 0;
   out_6521785059618497323[42] = 0;
   out_6521785059618497323[43] = 0;
   out_6521785059618497323[44] = 1;
   out_6521785059618497323[45] = 0;
   out_6521785059618497323[46] = 0;
   out_6521785059618497323[47] = 0;
   out_6521785059618497323[48] = 0;
   out_6521785059618497323[49] = 0;
   out_6521785059618497323[50] = 0;
   out_6521785059618497323[51] = 0;
   out_6521785059618497323[52] = 0;
   out_6521785059618497323[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_716632891583450328) {
  err_fun(nom_x, delta_x, out_716632891583450328);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3104130222774257522) {
  inv_err_fun(nom_x, true_x, out_3104130222774257522);
}
void pose_H_mod_fun(double *state, double *out_8665327190051684991) {
  H_mod_fun(state, out_8665327190051684991);
}
void pose_f_fun(double *state, double dt, double *out_1781769764659626549) {
  f_fun(state,  dt, out_1781769764659626549);
}
void pose_F_fun(double *state, double dt, double *out_3937534521909816001) {
  F_fun(state,  dt, out_3937534521909816001);
}
void pose_h_4(double *state, double *unused, double *out_4373223397940532151) {
  h_4(state, unused, out_4373223397940532151);
}
void pose_H_4(double *state, double *unused, double *out_7961718157751569764) {
  H_4(state, unused, out_7961718157751569764);
}
void pose_h_10(double *state, double *unused, double *out_6286469033191029799) {
  h_10(state, unused, out_6286469033191029799);
}
void pose_H_10(double *state, double *unused, double *out_5531647416424197146) {
  H_10(state, unused, out_5531647416424197146);
}
void pose_h_13(double *state, double *unused, double *out_5115075594500414792) {
  h_13(state, unused, out_5115075594500414792);
}
void pose_H_13(double *state, double *unused, double *out_7272752090625649051) {
  H_13(state, unused, out_7272752090625649051);
}
void pose_h_14(double *state, double *unused, double *out_7724198509017227969) {
  h_14(state, unused, out_7724198509017227969);
}
void pose_H_14(double *state, double *unused, double *out_6521785059618497323) {
  H_14(state, unused, out_6521785059618497323);
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
