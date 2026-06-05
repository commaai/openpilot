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
void err_fun(double *nom_x, double *delta_x, double *out_1100575581318041579) {
   out_1100575581318041579[0] = delta_x[0] + nom_x[0];
   out_1100575581318041579[1] = delta_x[1] + nom_x[1];
   out_1100575581318041579[2] = delta_x[2] + nom_x[2];
   out_1100575581318041579[3] = delta_x[3] + nom_x[3];
   out_1100575581318041579[4] = delta_x[4] + nom_x[4];
   out_1100575581318041579[5] = delta_x[5] + nom_x[5];
   out_1100575581318041579[6] = delta_x[6] + nom_x[6];
   out_1100575581318041579[7] = delta_x[7] + nom_x[7];
   out_1100575581318041579[8] = delta_x[8] + nom_x[8];
   out_1100575581318041579[9] = delta_x[9] + nom_x[9];
   out_1100575581318041579[10] = delta_x[10] + nom_x[10];
   out_1100575581318041579[11] = delta_x[11] + nom_x[11];
   out_1100575581318041579[12] = delta_x[12] + nom_x[12];
   out_1100575581318041579[13] = delta_x[13] + nom_x[13];
   out_1100575581318041579[14] = delta_x[14] + nom_x[14];
   out_1100575581318041579[15] = delta_x[15] + nom_x[15];
   out_1100575581318041579[16] = delta_x[16] + nom_x[16];
   out_1100575581318041579[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6410724245809315096) {
   out_6410724245809315096[0] = -nom_x[0] + true_x[0];
   out_6410724245809315096[1] = -nom_x[1] + true_x[1];
   out_6410724245809315096[2] = -nom_x[2] + true_x[2];
   out_6410724245809315096[3] = -nom_x[3] + true_x[3];
   out_6410724245809315096[4] = -nom_x[4] + true_x[4];
   out_6410724245809315096[5] = -nom_x[5] + true_x[5];
   out_6410724245809315096[6] = -nom_x[6] + true_x[6];
   out_6410724245809315096[7] = -nom_x[7] + true_x[7];
   out_6410724245809315096[8] = -nom_x[8] + true_x[8];
   out_6410724245809315096[9] = -nom_x[9] + true_x[9];
   out_6410724245809315096[10] = -nom_x[10] + true_x[10];
   out_6410724245809315096[11] = -nom_x[11] + true_x[11];
   out_6410724245809315096[12] = -nom_x[12] + true_x[12];
   out_6410724245809315096[13] = -nom_x[13] + true_x[13];
   out_6410724245809315096[14] = -nom_x[14] + true_x[14];
   out_6410724245809315096[15] = -nom_x[15] + true_x[15];
   out_6410724245809315096[16] = -nom_x[16] + true_x[16];
   out_6410724245809315096[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1238262041853718804) {
   out_1238262041853718804[0] = 1.0;
   out_1238262041853718804[1] = 0.0;
   out_1238262041853718804[2] = 0.0;
   out_1238262041853718804[3] = 0.0;
   out_1238262041853718804[4] = 0.0;
   out_1238262041853718804[5] = 0.0;
   out_1238262041853718804[6] = 0.0;
   out_1238262041853718804[7] = 0.0;
   out_1238262041853718804[8] = 0.0;
   out_1238262041853718804[9] = 0.0;
   out_1238262041853718804[10] = 0.0;
   out_1238262041853718804[11] = 0.0;
   out_1238262041853718804[12] = 0.0;
   out_1238262041853718804[13] = 0.0;
   out_1238262041853718804[14] = 0.0;
   out_1238262041853718804[15] = 0.0;
   out_1238262041853718804[16] = 0.0;
   out_1238262041853718804[17] = 0.0;
   out_1238262041853718804[18] = 0.0;
   out_1238262041853718804[19] = 1.0;
   out_1238262041853718804[20] = 0.0;
   out_1238262041853718804[21] = 0.0;
   out_1238262041853718804[22] = 0.0;
   out_1238262041853718804[23] = 0.0;
   out_1238262041853718804[24] = 0.0;
   out_1238262041853718804[25] = 0.0;
   out_1238262041853718804[26] = 0.0;
   out_1238262041853718804[27] = 0.0;
   out_1238262041853718804[28] = 0.0;
   out_1238262041853718804[29] = 0.0;
   out_1238262041853718804[30] = 0.0;
   out_1238262041853718804[31] = 0.0;
   out_1238262041853718804[32] = 0.0;
   out_1238262041853718804[33] = 0.0;
   out_1238262041853718804[34] = 0.0;
   out_1238262041853718804[35] = 0.0;
   out_1238262041853718804[36] = 0.0;
   out_1238262041853718804[37] = 0.0;
   out_1238262041853718804[38] = 1.0;
   out_1238262041853718804[39] = 0.0;
   out_1238262041853718804[40] = 0.0;
   out_1238262041853718804[41] = 0.0;
   out_1238262041853718804[42] = 0.0;
   out_1238262041853718804[43] = 0.0;
   out_1238262041853718804[44] = 0.0;
   out_1238262041853718804[45] = 0.0;
   out_1238262041853718804[46] = 0.0;
   out_1238262041853718804[47] = 0.0;
   out_1238262041853718804[48] = 0.0;
   out_1238262041853718804[49] = 0.0;
   out_1238262041853718804[50] = 0.0;
   out_1238262041853718804[51] = 0.0;
   out_1238262041853718804[52] = 0.0;
   out_1238262041853718804[53] = 0.0;
   out_1238262041853718804[54] = 0.0;
   out_1238262041853718804[55] = 0.0;
   out_1238262041853718804[56] = 0.0;
   out_1238262041853718804[57] = 1.0;
   out_1238262041853718804[58] = 0.0;
   out_1238262041853718804[59] = 0.0;
   out_1238262041853718804[60] = 0.0;
   out_1238262041853718804[61] = 0.0;
   out_1238262041853718804[62] = 0.0;
   out_1238262041853718804[63] = 0.0;
   out_1238262041853718804[64] = 0.0;
   out_1238262041853718804[65] = 0.0;
   out_1238262041853718804[66] = 0.0;
   out_1238262041853718804[67] = 0.0;
   out_1238262041853718804[68] = 0.0;
   out_1238262041853718804[69] = 0.0;
   out_1238262041853718804[70] = 0.0;
   out_1238262041853718804[71] = 0.0;
   out_1238262041853718804[72] = 0.0;
   out_1238262041853718804[73] = 0.0;
   out_1238262041853718804[74] = 0.0;
   out_1238262041853718804[75] = 0.0;
   out_1238262041853718804[76] = 1.0;
   out_1238262041853718804[77] = 0.0;
   out_1238262041853718804[78] = 0.0;
   out_1238262041853718804[79] = 0.0;
   out_1238262041853718804[80] = 0.0;
   out_1238262041853718804[81] = 0.0;
   out_1238262041853718804[82] = 0.0;
   out_1238262041853718804[83] = 0.0;
   out_1238262041853718804[84] = 0.0;
   out_1238262041853718804[85] = 0.0;
   out_1238262041853718804[86] = 0.0;
   out_1238262041853718804[87] = 0.0;
   out_1238262041853718804[88] = 0.0;
   out_1238262041853718804[89] = 0.0;
   out_1238262041853718804[90] = 0.0;
   out_1238262041853718804[91] = 0.0;
   out_1238262041853718804[92] = 0.0;
   out_1238262041853718804[93] = 0.0;
   out_1238262041853718804[94] = 0.0;
   out_1238262041853718804[95] = 1.0;
   out_1238262041853718804[96] = 0.0;
   out_1238262041853718804[97] = 0.0;
   out_1238262041853718804[98] = 0.0;
   out_1238262041853718804[99] = 0.0;
   out_1238262041853718804[100] = 0.0;
   out_1238262041853718804[101] = 0.0;
   out_1238262041853718804[102] = 0.0;
   out_1238262041853718804[103] = 0.0;
   out_1238262041853718804[104] = 0.0;
   out_1238262041853718804[105] = 0.0;
   out_1238262041853718804[106] = 0.0;
   out_1238262041853718804[107] = 0.0;
   out_1238262041853718804[108] = 0.0;
   out_1238262041853718804[109] = 0.0;
   out_1238262041853718804[110] = 0.0;
   out_1238262041853718804[111] = 0.0;
   out_1238262041853718804[112] = 0.0;
   out_1238262041853718804[113] = 0.0;
   out_1238262041853718804[114] = 1.0;
   out_1238262041853718804[115] = 0.0;
   out_1238262041853718804[116] = 0.0;
   out_1238262041853718804[117] = 0.0;
   out_1238262041853718804[118] = 0.0;
   out_1238262041853718804[119] = 0.0;
   out_1238262041853718804[120] = 0.0;
   out_1238262041853718804[121] = 0.0;
   out_1238262041853718804[122] = 0.0;
   out_1238262041853718804[123] = 0.0;
   out_1238262041853718804[124] = 0.0;
   out_1238262041853718804[125] = 0.0;
   out_1238262041853718804[126] = 0.0;
   out_1238262041853718804[127] = 0.0;
   out_1238262041853718804[128] = 0.0;
   out_1238262041853718804[129] = 0.0;
   out_1238262041853718804[130] = 0.0;
   out_1238262041853718804[131] = 0.0;
   out_1238262041853718804[132] = 0.0;
   out_1238262041853718804[133] = 1.0;
   out_1238262041853718804[134] = 0.0;
   out_1238262041853718804[135] = 0.0;
   out_1238262041853718804[136] = 0.0;
   out_1238262041853718804[137] = 0.0;
   out_1238262041853718804[138] = 0.0;
   out_1238262041853718804[139] = 0.0;
   out_1238262041853718804[140] = 0.0;
   out_1238262041853718804[141] = 0.0;
   out_1238262041853718804[142] = 0.0;
   out_1238262041853718804[143] = 0.0;
   out_1238262041853718804[144] = 0.0;
   out_1238262041853718804[145] = 0.0;
   out_1238262041853718804[146] = 0.0;
   out_1238262041853718804[147] = 0.0;
   out_1238262041853718804[148] = 0.0;
   out_1238262041853718804[149] = 0.0;
   out_1238262041853718804[150] = 0.0;
   out_1238262041853718804[151] = 0.0;
   out_1238262041853718804[152] = 1.0;
   out_1238262041853718804[153] = 0.0;
   out_1238262041853718804[154] = 0.0;
   out_1238262041853718804[155] = 0.0;
   out_1238262041853718804[156] = 0.0;
   out_1238262041853718804[157] = 0.0;
   out_1238262041853718804[158] = 0.0;
   out_1238262041853718804[159] = 0.0;
   out_1238262041853718804[160] = 0.0;
   out_1238262041853718804[161] = 0.0;
   out_1238262041853718804[162] = 0.0;
   out_1238262041853718804[163] = 0.0;
   out_1238262041853718804[164] = 0.0;
   out_1238262041853718804[165] = 0.0;
   out_1238262041853718804[166] = 0.0;
   out_1238262041853718804[167] = 0.0;
   out_1238262041853718804[168] = 0.0;
   out_1238262041853718804[169] = 0.0;
   out_1238262041853718804[170] = 0.0;
   out_1238262041853718804[171] = 1.0;
   out_1238262041853718804[172] = 0.0;
   out_1238262041853718804[173] = 0.0;
   out_1238262041853718804[174] = 0.0;
   out_1238262041853718804[175] = 0.0;
   out_1238262041853718804[176] = 0.0;
   out_1238262041853718804[177] = 0.0;
   out_1238262041853718804[178] = 0.0;
   out_1238262041853718804[179] = 0.0;
   out_1238262041853718804[180] = 0.0;
   out_1238262041853718804[181] = 0.0;
   out_1238262041853718804[182] = 0.0;
   out_1238262041853718804[183] = 0.0;
   out_1238262041853718804[184] = 0.0;
   out_1238262041853718804[185] = 0.0;
   out_1238262041853718804[186] = 0.0;
   out_1238262041853718804[187] = 0.0;
   out_1238262041853718804[188] = 0.0;
   out_1238262041853718804[189] = 0.0;
   out_1238262041853718804[190] = 1.0;
   out_1238262041853718804[191] = 0.0;
   out_1238262041853718804[192] = 0.0;
   out_1238262041853718804[193] = 0.0;
   out_1238262041853718804[194] = 0.0;
   out_1238262041853718804[195] = 0.0;
   out_1238262041853718804[196] = 0.0;
   out_1238262041853718804[197] = 0.0;
   out_1238262041853718804[198] = 0.0;
   out_1238262041853718804[199] = 0.0;
   out_1238262041853718804[200] = 0.0;
   out_1238262041853718804[201] = 0.0;
   out_1238262041853718804[202] = 0.0;
   out_1238262041853718804[203] = 0.0;
   out_1238262041853718804[204] = 0.0;
   out_1238262041853718804[205] = 0.0;
   out_1238262041853718804[206] = 0.0;
   out_1238262041853718804[207] = 0.0;
   out_1238262041853718804[208] = 0.0;
   out_1238262041853718804[209] = 1.0;
   out_1238262041853718804[210] = 0.0;
   out_1238262041853718804[211] = 0.0;
   out_1238262041853718804[212] = 0.0;
   out_1238262041853718804[213] = 0.0;
   out_1238262041853718804[214] = 0.0;
   out_1238262041853718804[215] = 0.0;
   out_1238262041853718804[216] = 0.0;
   out_1238262041853718804[217] = 0.0;
   out_1238262041853718804[218] = 0.0;
   out_1238262041853718804[219] = 0.0;
   out_1238262041853718804[220] = 0.0;
   out_1238262041853718804[221] = 0.0;
   out_1238262041853718804[222] = 0.0;
   out_1238262041853718804[223] = 0.0;
   out_1238262041853718804[224] = 0.0;
   out_1238262041853718804[225] = 0.0;
   out_1238262041853718804[226] = 0.0;
   out_1238262041853718804[227] = 0.0;
   out_1238262041853718804[228] = 1.0;
   out_1238262041853718804[229] = 0.0;
   out_1238262041853718804[230] = 0.0;
   out_1238262041853718804[231] = 0.0;
   out_1238262041853718804[232] = 0.0;
   out_1238262041853718804[233] = 0.0;
   out_1238262041853718804[234] = 0.0;
   out_1238262041853718804[235] = 0.0;
   out_1238262041853718804[236] = 0.0;
   out_1238262041853718804[237] = 0.0;
   out_1238262041853718804[238] = 0.0;
   out_1238262041853718804[239] = 0.0;
   out_1238262041853718804[240] = 0.0;
   out_1238262041853718804[241] = 0.0;
   out_1238262041853718804[242] = 0.0;
   out_1238262041853718804[243] = 0.0;
   out_1238262041853718804[244] = 0.0;
   out_1238262041853718804[245] = 0.0;
   out_1238262041853718804[246] = 0.0;
   out_1238262041853718804[247] = 1.0;
   out_1238262041853718804[248] = 0.0;
   out_1238262041853718804[249] = 0.0;
   out_1238262041853718804[250] = 0.0;
   out_1238262041853718804[251] = 0.0;
   out_1238262041853718804[252] = 0.0;
   out_1238262041853718804[253] = 0.0;
   out_1238262041853718804[254] = 0.0;
   out_1238262041853718804[255] = 0.0;
   out_1238262041853718804[256] = 0.0;
   out_1238262041853718804[257] = 0.0;
   out_1238262041853718804[258] = 0.0;
   out_1238262041853718804[259] = 0.0;
   out_1238262041853718804[260] = 0.0;
   out_1238262041853718804[261] = 0.0;
   out_1238262041853718804[262] = 0.0;
   out_1238262041853718804[263] = 0.0;
   out_1238262041853718804[264] = 0.0;
   out_1238262041853718804[265] = 0.0;
   out_1238262041853718804[266] = 1.0;
   out_1238262041853718804[267] = 0.0;
   out_1238262041853718804[268] = 0.0;
   out_1238262041853718804[269] = 0.0;
   out_1238262041853718804[270] = 0.0;
   out_1238262041853718804[271] = 0.0;
   out_1238262041853718804[272] = 0.0;
   out_1238262041853718804[273] = 0.0;
   out_1238262041853718804[274] = 0.0;
   out_1238262041853718804[275] = 0.0;
   out_1238262041853718804[276] = 0.0;
   out_1238262041853718804[277] = 0.0;
   out_1238262041853718804[278] = 0.0;
   out_1238262041853718804[279] = 0.0;
   out_1238262041853718804[280] = 0.0;
   out_1238262041853718804[281] = 0.0;
   out_1238262041853718804[282] = 0.0;
   out_1238262041853718804[283] = 0.0;
   out_1238262041853718804[284] = 0.0;
   out_1238262041853718804[285] = 1.0;
   out_1238262041853718804[286] = 0.0;
   out_1238262041853718804[287] = 0.0;
   out_1238262041853718804[288] = 0.0;
   out_1238262041853718804[289] = 0.0;
   out_1238262041853718804[290] = 0.0;
   out_1238262041853718804[291] = 0.0;
   out_1238262041853718804[292] = 0.0;
   out_1238262041853718804[293] = 0.0;
   out_1238262041853718804[294] = 0.0;
   out_1238262041853718804[295] = 0.0;
   out_1238262041853718804[296] = 0.0;
   out_1238262041853718804[297] = 0.0;
   out_1238262041853718804[298] = 0.0;
   out_1238262041853718804[299] = 0.0;
   out_1238262041853718804[300] = 0.0;
   out_1238262041853718804[301] = 0.0;
   out_1238262041853718804[302] = 0.0;
   out_1238262041853718804[303] = 0.0;
   out_1238262041853718804[304] = 1.0;
   out_1238262041853718804[305] = 0.0;
   out_1238262041853718804[306] = 0.0;
   out_1238262041853718804[307] = 0.0;
   out_1238262041853718804[308] = 0.0;
   out_1238262041853718804[309] = 0.0;
   out_1238262041853718804[310] = 0.0;
   out_1238262041853718804[311] = 0.0;
   out_1238262041853718804[312] = 0.0;
   out_1238262041853718804[313] = 0.0;
   out_1238262041853718804[314] = 0.0;
   out_1238262041853718804[315] = 0.0;
   out_1238262041853718804[316] = 0.0;
   out_1238262041853718804[317] = 0.0;
   out_1238262041853718804[318] = 0.0;
   out_1238262041853718804[319] = 0.0;
   out_1238262041853718804[320] = 0.0;
   out_1238262041853718804[321] = 0.0;
   out_1238262041853718804[322] = 0.0;
   out_1238262041853718804[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_470659832336319089) {
   out_470659832336319089[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_470659832336319089[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_470659832336319089[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_470659832336319089[3] = dt*state[12] + state[3];
   out_470659832336319089[4] = dt*state[13] + state[4];
   out_470659832336319089[5] = dt*state[14] + state[5];
   out_470659832336319089[6] = state[6];
   out_470659832336319089[7] = state[7];
   out_470659832336319089[8] = state[8];
   out_470659832336319089[9] = state[9];
   out_470659832336319089[10] = state[10];
   out_470659832336319089[11] = state[11];
   out_470659832336319089[12] = state[12];
   out_470659832336319089[13] = state[13];
   out_470659832336319089[14] = state[14];
   out_470659832336319089[15] = state[15];
   out_470659832336319089[16] = state[16];
   out_470659832336319089[17] = state[17];
}
void F_fun(double *state, double dt, double *out_7507724760190486926) {
   out_7507724760190486926[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7507724760190486926[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7507724760190486926[2] = 0;
   out_7507724760190486926[3] = 0;
   out_7507724760190486926[4] = 0;
   out_7507724760190486926[5] = 0;
   out_7507724760190486926[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7507724760190486926[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7507724760190486926[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7507724760190486926[9] = 0;
   out_7507724760190486926[10] = 0;
   out_7507724760190486926[11] = 0;
   out_7507724760190486926[12] = 0;
   out_7507724760190486926[13] = 0;
   out_7507724760190486926[14] = 0;
   out_7507724760190486926[15] = 0;
   out_7507724760190486926[16] = 0;
   out_7507724760190486926[17] = 0;
   out_7507724760190486926[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7507724760190486926[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7507724760190486926[20] = 0;
   out_7507724760190486926[21] = 0;
   out_7507724760190486926[22] = 0;
   out_7507724760190486926[23] = 0;
   out_7507724760190486926[24] = 0;
   out_7507724760190486926[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7507724760190486926[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7507724760190486926[27] = 0;
   out_7507724760190486926[28] = 0;
   out_7507724760190486926[29] = 0;
   out_7507724760190486926[30] = 0;
   out_7507724760190486926[31] = 0;
   out_7507724760190486926[32] = 0;
   out_7507724760190486926[33] = 0;
   out_7507724760190486926[34] = 0;
   out_7507724760190486926[35] = 0;
   out_7507724760190486926[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7507724760190486926[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7507724760190486926[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7507724760190486926[39] = 0;
   out_7507724760190486926[40] = 0;
   out_7507724760190486926[41] = 0;
   out_7507724760190486926[42] = 0;
   out_7507724760190486926[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7507724760190486926[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7507724760190486926[45] = 0;
   out_7507724760190486926[46] = 0;
   out_7507724760190486926[47] = 0;
   out_7507724760190486926[48] = 0;
   out_7507724760190486926[49] = 0;
   out_7507724760190486926[50] = 0;
   out_7507724760190486926[51] = 0;
   out_7507724760190486926[52] = 0;
   out_7507724760190486926[53] = 0;
   out_7507724760190486926[54] = 0;
   out_7507724760190486926[55] = 0;
   out_7507724760190486926[56] = 0;
   out_7507724760190486926[57] = 1;
   out_7507724760190486926[58] = 0;
   out_7507724760190486926[59] = 0;
   out_7507724760190486926[60] = 0;
   out_7507724760190486926[61] = 0;
   out_7507724760190486926[62] = 0;
   out_7507724760190486926[63] = 0;
   out_7507724760190486926[64] = 0;
   out_7507724760190486926[65] = 0;
   out_7507724760190486926[66] = dt;
   out_7507724760190486926[67] = 0;
   out_7507724760190486926[68] = 0;
   out_7507724760190486926[69] = 0;
   out_7507724760190486926[70] = 0;
   out_7507724760190486926[71] = 0;
   out_7507724760190486926[72] = 0;
   out_7507724760190486926[73] = 0;
   out_7507724760190486926[74] = 0;
   out_7507724760190486926[75] = 0;
   out_7507724760190486926[76] = 1;
   out_7507724760190486926[77] = 0;
   out_7507724760190486926[78] = 0;
   out_7507724760190486926[79] = 0;
   out_7507724760190486926[80] = 0;
   out_7507724760190486926[81] = 0;
   out_7507724760190486926[82] = 0;
   out_7507724760190486926[83] = 0;
   out_7507724760190486926[84] = 0;
   out_7507724760190486926[85] = dt;
   out_7507724760190486926[86] = 0;
   out_7507724760190486926[87] = 0;
   out_7507724760190486926[88] = 0;
   out_7507724760190486926[89] = 0;
   out_7507724760190486926[90] = 0;
   out_7507724760190486926[91] = 0;
   out_7507724760190486926[92] = 0;
   out_7507724760190486926[93] = 0;
   out_7507724760190486926[94] = 0;
   out_7507724760190486926[95] = 1;
   out_7507724760190486926[96] = 0;
   out_7507724760190486926[97] = 0;
   out_7507724760190486926[98] = 0;
   out_7507724760190486926[99] = 0;
   out_7507724760190486926[100] = 0;
   out_7507724760190486926[101] = 0;
   out_7507724760190486926[102] = 0;
   out_7507724760190486926[103] = 0;
   out_7507724760190486926[104] = dt;
   out_7507724760190486926[105] = 0;
   out_7507724760190486926[106] = 0;
   out_7507724760190486926[107] = 0;
   out_7507724760190486926[108] = 0;
   out_7507724760190486926[109] = 0;
   out_7507724760190486926[110] = 0;
   out_7507724760190486926[111] = 0;
   out_7507724760190486926[112] = 0;
   out_7507724760190486926[113] = 0;
   out_7507724760190486926[114] = 1;
   out_7507724760190486926[115] = 0;
   out_7507724760190486926[116] = 0;
   out_7507724760190486926[117] = 0;
   out_7507724760190486926[118] = 0;
   out_7507724760190486926[119] = 0;
   out_7507724760190486926[120] = 0;
   out_7507724760190486926[121] = 0;
   out_7507724760190486926[122] = 0;
   out_7507724760190486926[123] = 0;
   out_7507724760190486926[124] = 0;
   out_7507724760190486926[125] = 0;
   out_7507724760190486926[126] = 0;
   out_7507724760190486926[127] = 0;
   out_7507724760190486926[128] = 0;
   out_7507724760190486926[129] = 0;
   out_7507724760190486926[130] = 0;
   out_7507724760190486926[131] = 0;
   out_7507724760190486926[132] = 0;
   out_7507724760190486926[133] = 1;
   out_7507724760190486926[134] = 0;
   out_7507724760190486926[135] = 0;
   out_7507724760190486926[136] = 0;
   out_7507724760190486926[137] = 0;
   out_7507724760190486926[138] = 0;
   out_7507724760190486926[139] = 0;
   out_7507724760190486926[140] = 0;
   out_7507724760190486926[141] = 0;
   out_7507724760190486926[142] = 0;
   out_7507724760190486926[143] = 0;
   out_7507724760190486926[144] = 0;
   out_7507724760190486926[145] = 0;
   out_7507724760190486926[146] = 0;
   out_7507724760190486926[147] = 0;
   out_7507724760190486926[148] = 0;
   out_7507724760190486926[149] = 0;
   out_7507724760190486926[150] = 0;
   out_7507724760190486926[151] = 0;
   out_7507724760190486926[152] = 1;
   out_7507724760190486926[153] = 0;
   out_7507724760190486926[154] = 0;
   out_7507724760190486926[155] = 0;
   out_7507724760190486926[156] = 0;
   out_7507724760190486926[157] = 0;
   out_7507724760190486926[158] = 0;
   out_7507724760190486926[159] = 0;
   out_7507724760190486926[160] = 0;
   out_7507724760190486926[161] = 0;
   out_7507724760190486926[162] = 0;
   out_7507724760190486926[163] = 0;
   out_7507724760190486926[164] = 0;
   out_7507724760190486926[165] = 0;
   out_7507724760190486926[166] = 0;
   out_7507724760190486926[167] = 0;
   out_7507724760190486926[168] = 0;
   out_7507724760190486926[169] = 0;
   out_7507724760190486926[170] = 0;
   out_7507724760190486926[171] = 1;
   out_7507724760190486926[172] = 0;
   out_7507724760190486926[173] = 0;
   out_7507724760190486926[174] = 0;
   out_7507724760190486926[175] = 0;
   out_7507724760190486926[176] = 0;
   out_7507724760190486926[177] = 0;
   out_7507724760190486926[178] = 0;
   out_7507724760190486926[179] = 0;
   out_7507724760190486926[180] = 0;
   out_7507724760190486926[181] = 0;
   out_7507724760190486926[182] = 0;
   out_7507724760190486926[183] = 0;
   out_7507724760190486926[184] = 0;
   out_7507724760190486926[185] = 0;
   out_7507724760190486926[186] = 0;
   out_7507724760190486926[187] = 0;
   out_7507724760190486926[188] = 0;
   out_7507724760190486926[189] = 0;
   out_7507724760190486926[190] = 1;
   out_7507724760190486926[191] = 0;
   out_7507724760190486926[192] = 0;
   out_7507724760190486926[193] = 0;
   out_7507724760190486926[194] = 0;
   out_7507724760190486926[195] = 0;
   out_7507724760190486926[196] = 0;
   out_7507724760190486926[197] = 0;
   out_7507724760190486926[198] = 0;
   out_7507724760190486926[199] = 0;
   out_7507724760190486926[200] = 0;
   out_7507724760190486926[201] = 0;
   out_7507724760190486926[202] = 0;
   out_7507724760190486926[203] = 0;
   out_7507724760190486926[204] = 0;
   out_7507724760190486926[205] = 0;
   out_7507724760190486926[206] = 0;
   out_7507724760190486926[207] = 0;
   out_7507724760190486926[208] = 0;
   out_7507724760190486926[209] = 1;
   out_7507724760190486926[210] = 0;
   out_7507724760190486926[211] = 0;
   out_7507724760190486926[212] = 0;
   out_7507724760190486926[213] = 0;
   out_7507724760190486926[214] = 0;
   out_7507724760190486926[215] = 0;
   out_7507724760190486926[216] = 0;
   out_7507724760190486926[217] = 0;
   out_7507724760190486926[218] = 0;
   out_7507724760190486926[219] = 0;
   out_7507724760190486926[220] = 0;
   out_7507724760190486926[221] = 0;
   out_7507724760190486926[222] = 0;
   out_7507724760190486926[223] = 0;
   out_7507724760190486926[224] = 0;
   out_7507724760190486926[225] = 0;
   out_7507724760190486926[226] = 0;
   out_7507724760190486926[227] = 0;
   out_7507724760190486926[228] = 1;
   out_7507724760190486926[229] = 0;
   out_7507724760190486926[230] = 0;
   out_7507724760190486926[231] = 0;
   out_7507724760190486926[232] = 0;
   out_7507724760190486926[233] = 0;
   out_7507724760190486926[234] = 0;
   out_7507724760190486926[235] = 0;
   out_7507724760190486926[236] = 0;
   out_7507724760190486926[237] = 0;
   out_7507724760190486926[238] = 0;
   out_7507724760190486926[239] = 0;
   out_7507724760190486926[240] = 0;
   out_7507724760190486926[241] = 0;
   out_7507724760190486926[242] = 0;
   out_7507724760190486926[243] = 0;
   out_7507724760190486926[244] = 0;
   out_7507724760190486926[245] = 0;
   out_7507724760190486926[246] = 0;
   out_7507724760190486926[247] = 1;
   out_7507724760190486926[248] = 0;
   out_7507724760190486926[249] = 0;
   out_7507724760190486926[250] = 0;
   out_7507724760190486926[251] = 0;
   out_7507724760190486926[252] = 0;
   out_7507724760190486926[253] = 0;
   out_7507724760190486926[254] = 0;
   out_7507724760190486926[255] = 0;
   out_7507724760190486926[256] = 0;
   out_7507724760190486926[257] = 0;
   out_7507724760190486926[258] = 0;
   out_7507724760190486926[259] = 0;
   out_7507724760190486926[260] = 0;
   out_7507724760190486926[261] = 0;
   out_7507724760190486926[262] = 0;
   out_7507724760190486926[263] = 0;
   out_7507724760190486926[264] = 0;
   out_7507724760190486926[265] = 0;
   out_7507724760190486926[266] = 1;
   out_7507724760190486926[267] = 0;
   out_7507724760190486926[268] = 0;
   out_7507724760190486926[269] = 0;
   out_7507724760190486926[270] = 0;
   out_7507724760190486926[271] = 0;
   out_7507724760190486926[272] = 0;
   out_7507724760190486926[273] = 0;
   out_7507724760190486926[274] = 0;
   out_7507724760190486926[275] = 0;
   out_7507724760190486926[276] = 0;
   out_7507724760190486926[277] = 0;
   out_7507724760190486926[278] = 0;
   out_7507724760190486926[279] = 0;
   out_7507724760190486926[280] = 0;
   out_7507724760190486926[281] = 0;
   out_7507724760190486926[282] = 0;
   out_7507724760190486926[283] = 0;
   out_7507724760190486926[284] = 0;
   out_7507724760190486926[285] = 1;
   out_7507724760190486926[286] = 0;
   out_7507724760190486926[287] = 0;
   out_7507724760190486926[288] = 0;
   out_7507724760190486926[289] = 0;
   out_7507724760190486926[290] = 0;
   out_7507724760190486926[291] = 0;
   out_7507724760190486926[292] = 0;
   out_7507724760190486926[293] = 0;
   out_7507724760190486926[294] = 0;
   out_7507724760190486926[295] = 0;
   out_7507724760190486926[296] = 0;
   out_7507724760190486926[297] = 0;
   out_7507724760190486926[298] = 0;
   out_7507724760190486926[299] = 0;
   out_7507724760190486926[300] = 0;
   out_7507724760190486926[301] = 0;
   out_7507724760190486926[302] = 0;
   out_7507724760190486926[303] = 0;
   out_7507724760190486926[304] = 1;
   out_7507724760190486926[305] = 0;
   out_7507724760190486926[306] = 0;
   out_7507724760190486926[307] = 0;
   out_7507724760190486926[308] = 0;
   out_7507724760190486926[309] = 0;
   out_7507724760190486926[310] = 0;
   out_7507724760190486926[311] = 0;
   out_7507724760190486926[312] = 0;
   out_7507724760190486926[313] = 0;
   out_7507724760190486926[314] = 0;
   out_7507724760190486926[315] = 0;
   out_7507724760190486926[316] = 0;
   out_7507724760190486926[317] = 0;
   out_7507724760190486926[318] = 0;
   out_7507724760190486926[319] = 0;
   out_7507724760190486926[320] = 0;
   out_7507724760190486926[321] = 0;
   out_7507724760190486926[322] = 0;
   out_7507724760190486926[323] = 1;
}
void h_4(double *state, double *unused, double *out_8065105448835972349) {
   out_8065105448835972349[0] = state[6] + state[9];
   out_8065105448835972349[1] = state[7] + state[10];
   out_8065105448835972349[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_1962147859247751220) {
   out_1962147859247751220[0] = 0;
   out_1962147859247751220[1] = 0;
   out_1962147859247751220[2] = 0;
   out_1962147859247751220[3] = 0;
   out_1962147859247751220[4] = 0;
   out_1962147859247751220[5] = 0;
   out_1962147859247751220[6] = 1;
   out_1962147859247751220[7] = 0;
   out_1962147859247751220[8] = 0;
   out_1962147859247751220[9] = 1;
   out_1962147859247751220[10] = 0;
   out_1962147859247751220[11] = 0;
   out_1962147859247751220[12] = 0;
   out_1962147859247751220[13] = 0;
   out_1962147859247751220[14] = 0;
   out_1962147859247751220[15] = 0;
   out_1962147859247751220[16] = 0;
   out_1962147859247751220[17] = 0;
   out_1962147859247751220[18] = 0;
   out_1962147859247751220[19] = 0;
   out_1962147859247751220[20] = 0;
   out_1962147859247751220[21] = 0;
   out_1962147859247751220[22] = 0;
   out_1962147859247751220[23] = 0;
   out_1962147859247751220[24] = 0;
   out_1962147859247751220[25] = 1;
   out_1962147859247751220[26] = 0;
   out_1962147859247751220[27] = 0;
   out_1962147859247751220[28] = 1;
   out_1962147859247751220[29] = 0;
   out_1962147859247751220[30] = 0;
   out_1962147859247751220[31] = 0;
   out_1962147859247751220[32] = 0;
   out_1962147859247751220[33] = 0;
   out_1962147859247751220[34] = 0;
   out_1962147859247751220[35] = 0;
   out_1962147859247751220[36] = 0;
   out_1962147859247751220[37] = 0;
   out_1962147859247751220[38] = 0;
   out_1962147859247751220[39] = 0;
   out_1962147859247751220[40] = 0;
   out_1962147859247751220[41] = 0;
   out_1962147859247751220[42] = 0;
   out_1962147859247751220[43] = 0;
   out_1962147859247751220[44] = 1;
   out_1962147859247751220[45] = 0;
   out_1962147859247751220[46] = 0;
   out_1962147859247751220[47] = 1;
   out_1962147859247751220[48] = 0;
   out_1962147859247751220[49] = 0;
   out_1962147859247751220[50] = 0;
   out_1962147859247751220[51] = 0;
   out_1962147859247751220[52] = 0;
   out_1962147859247751220[53] = 0;
}
void h_10(double *state, double *unused, double *out_365029382432652932) {
   out_365029382432652932[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_365029382432652932[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_365029382432652932[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_6475974887783404571) {
   out_6475974887783404571[0] = 0;
   out_6475974887783404571[1] = 9.8100000000000005*cos(state[1]);
   out_6475974887783404571[2] = 0;
   out_6475974887783404571[3] = 0;
   out_6475974887783404571[4] = -state[8];
   out_6475974887783404571[5] = state[7];
   out_6475974887783404571[6] = 0;
   out_6475974887783404571[7] = state[5];
   out_6475974887783404571[8] = -state[4];
   out_6475974887783404571[9] = 0;
   out_6475974887783404571[10] = 0;
   out_6475974887783404571[11] = 0;
   out_6475974887783404571[12] = 1;
   out_6475974887783404571[13] = 0;
   out_6475974887783404571[14] = 0;
   out_6475974887783404571[15] = 1;
   out_6475974887783404571[16] = 0;
   out_6475974887783404571[17] = 0;
   out_6475974887783404571[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_6475974887783404571[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_6475974887783404571[20] = 0;
   out_6475974887783404571[21] = state[8];
   out_6475974887783404571[22] = 0;
   out_6475974887783404571[23] = -state[6];
   out_6475974887783404571[24] = -state[5];
   out_6475974887783404571[25] = 0;
   out_6475974887783404571[26] = state[3];
   out_6475974887783404571[27] = 0;
   out_6475974887783404571[28] = 0;
   out_6475974887783404571[29] = 0;
   out_6475974887783404571[30] = 0;
   out_6475974887783404571[31] = 1;
   out_6475974887783404571[32] = 0;
   out_6475974887783404571[33] = 0;
   out_6475974887783404571[34] = 1;
   out_6475974887783404571[35] = 0;
   out_6475974887783404571[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_6475974887783404571[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_6475974887783404571[38] = 0;
   out_6475974887783404571[39] = -state[7];
   out_6475974887783404571[40] = state[6];
   out_6475974887783404571[41] = 0;
   out_6475974887783404571[42] = state[4];
   out_6475974887783404571[43] = -state[3];
   out_6475974887783404571[44] = 0;
   out_6475974887783404571[45] = 0;
   out_6475974887783404571[46] = 0;
   out_6475974887783404571[47] = 0;
   out_6475974887783404571[48] = 0;
   out_6475974887783404571[49] = 0;
   out_6475974887783404571[50] = 1;
   out_6475974887783404571[51] = 0;
   out_6475974887783404571[52] = 0;
   out_6475974887783404571[53] = 1;
}
void h_13(double *state, double *unused, double *out_1501704762674320483) {
   out_1501704762674320483[0] = state[3];
   out_1501704762674320483[1] = state[4];
   out_1501704762674320483[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1397545939565907116) {
   out_1397545939565907116[0] = 0;
   out_1397545939565907116[1] = 0;
   out_1397545939565907116[2] = 0;
   out_1397545939565907116[3] = 1;
   out_1397545939565907116[4] = 0;
   out_1397545939565907116[5] = 0;
   out_1397545939565907116[6] = 0;
   out_1397545939565907116[7] = 0;
   out_1397545939565907116[8] = 0;
   out_1397545939565907116[9] = 0;
   out_1397545939565907116[10] = 0;
   out_1397545939565907116[11] = 0;
   out_1397545939565907116[12] = 0;
   out_1397545939565907116[13] = 0;
   out_1397545939565907116[14] = 0;
   out_1397545939565907116[15] = 0;
   out_1397545939565907116[16] = 0;
   out_1397545939565907116[17] = 0;
   out_1397545939565907116[18] = 0;
   out_1397545939565907116[19] = 0;
   out_1397545939565907116[20] = 0;
   out_1397545939565907116[21] = 0;
   out_1397545939565907116[22] = 1;
   out_1397545939565907116[23] = 0;
   out_1397545939565907116[24] = 0;
   out_1397545939565907116[25] = 0;
   out_1397545939565907116[26] = 0;
   out_1397545939565907116[27] = 0;
   out_1397545939565907116[28] = 0;
   out_1397545939565907116[29] = 0;
   out_1397545939565907116[30] = 0;
   out_1397545939565907116[31] = 0;
   out_1397545939565907116[32] = 0;
   out_1397545939565907116[33] = 0;
   out_1397545939565907116[34] = 0;
   out_1397545939565907116[35] = 0;
   out_1397545939565907116[36] = 0;
   out_1397545939565907116[37] = 0;
   out_1397545939565907116[38] = 0;
   out_1397545939565907116[39] = 0;
   out_1397545939565907116[40] = 0;
   out_1397545939565907116[41] = 1;
   out_1397545939565907116[42] = 0;
   out_1397545939565907116[43] = 0;
   out_1397545939565907116[44] = 0;
   out_1397545939565907116[45] = 0;
   out_1397545939565907116[46] = 0;
   out_1397545939565907116[47] = 0;
   out_1397545939565907116[48] = 0;
   out_1397545939565907116[49] = 0;
   out_1397545939565907116[50] = 0;
   out_1397545939565907116[51] = 0;
   out_1397545939565907116[52] = 0;
   out_1397545939565907116[53] = 0;
}
void h_14(double *state, double *unused, double *out_2046062482872930650) {
   out_2046062482872930650[0] = state[6];
   out_2046062482872930650[1] = state[7];
   out_2046062482872930650[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5044936291543123516) {
   out_5044936291543123516[0] = 0;
   out_5044936291543123516[1] = 0;
   out_5044936291543123516[2] = 0;
   out_5044936291543123516[3] = 0;
   out_5044936291543123516[4] = 0;
   out_5044936291543123516[5] = 0;
   out_5044936291543123516[6] = 1;
   out_5044936291543123516[7] = 0;
   out_5044936291543123516[8] = 0;
   out_5044936291543123516[9] = 0;
   out_5044936291543123516[10] = 0;
   out_5044936291543123516[11] = 0;
   out_5044936291543123516[12] = 0;
   out_5044936291543123516[13] = 0;
   out_5044936291543123516[14] = 0;
   out_5044936291543123516[15] = 0;
   out_5044936291543123516[16] = 0;
   out_5044936291543123516[17] = 0;
   out_5044936291543123516[18] = 0;
   out_5044936291543123516[19] = 0;
   out_5044936291543123516[20] = 0;
   out_5044936291543123516[21] = 0;
   out_5044936291543123516[22] = 0;
   out_5044936291543123516[23] = 0;
   out_5044936291543123516[24] = 0;
   out_5044936291543123516[25] = 1;
   out_5044936291543123516[26] = 0;
   out_5044936291543123516[27] = 0;
   out_5044936291543123516[28] = 0;
   out_5044936291543123516[29] = 0;
   out_5044936291543123516[30] = 0;
   out_5044936291543123516[31] = 0;
   out_5044936291543123516[32] = 0;
   out_5044936291543123516[33] = 0;
   out_5044936291543123516[34] = 0;
   out_5044936291543123516[35] = 0;
   out_5044936291543123516[36] = 0;
   out_5044936291543123516[37] = 0;
   out_5044936291543123516[38] = 0;
   out_5044936291543123516[39] = 0;
   out_5044936291543123516[40] = 0;
   out_5044936291543123516[41] = 0;
   out_5044936291543123516[42] = 0;
   out_5044936291543123516[43] = 0;
   out_5044936291543123516[44] = 1;
   out_5044936291543123516[45] = 0;
   out_5044936291543123516[46] = 0;
   out_5044936291543123516[47] = 0;
   out_5044936291543123516[48] = 0;
   out_5044936291543123516[49] = 0;
   out_5044936291543123516[50] = 0;
   out_5044936291543123516[51] = 0;
   out_5044936291543123516[52] = 0;
   out_5044936291543123516[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_1100575581318041579) {
  err_fun(nom_x, delta_x, out_1100575581318041579);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6410724245809315096) {
  inv_err_fun(nom_x, true_x, out_6410724245809315096);
}
void pose_H_mod_fun(double *state, double *out_1238262041853718804) {
  H_mod_fun(state, out_1238262041853718804);
}
void pose_f_fun(double *state, double dt, double *out_470659832336319089) {
  f_fun(state,  dt, out_470659832336319089);
}
void pose_F_fun(double *state, double dt, double *out_7507724760190486926) {
  F_fun(state,  dt, out_7507724760190486926);
}
void pose_h_4(double *state, double *unused, double *out_8065105448835972349) {
  h_4(state, unused, out_8065105448835972349);
}
void pose_H_4(double *state, double *unused, double *out_1962147859247751220) {
  H_4(state, unused, out_1962147859247751220);
}
void pose_h_10(double *state, double *unused, double *out_365029382432652932) {
  h_10(state, unused, out_365029382432652932);
}
void pose_H_10(double *state, double *unused, double *out_6475974887783404571) {
  H_10(state, unused, out_6475974887783404571);
}
void pose_h_13(double *state, double *unused, double *out_1501704762674320483) {
  h_13(state, unused, out_1501704762674320483);
}
void pose_H_13(double *state, double *unused, double *out_1397545939565907116) {
  H_13(state, unused, out_1397545939565907116);
}
void pose_h_14(double *state, double *unused, double *out_2046062482872930650) {
  h_14(state, unused, out_2046062482872930650);
}
void pose_H_14(double *state, double *unused, double *out_5044936291543123516) {
  H_14(state, unused, out_5044936291543123516);
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
