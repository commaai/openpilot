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
void err_fun(double *nom_x, double *delta_x, double *out_1039242481979313707) {
   out_1039242481979313707[0] = delta_x[0] + nom_x[0];
   out_1039242481979313707[1] = delta_x[1] + nom_x[1];
   out_1039242481979313707[2] = delta_x[2] + nom_x[2];
   out_1039242481979313707[3] = delta_x[3] + nom_x[3];
   out_1039242481979313707[4] = delta_x[4] + nom_x[4];
   out_1039242481979313707[5] = delta_x[5] + nom_x[5];
   out_1039242481979313707[6] = delta_x[6] + nom_x[6];
   out_1039242481979313707[7] = delta_x[7] + nom_x[7];
   out_1039242481979313707[8] = delta_x[8] + nom_x[8];
   out_1039242481979313707[9] = delta_x[9] + nom_x[9];
   out_1039242481979313707[10] = delta_x[10] + nom_x[10];
   out_1039242481979313707[11] = delta_x[11] + nom_x[11];
   out_1039242481979313707[12] = delta_x[12] + nom_x[12];
   out_1039242481979313707[13] = delta_x[13] + nom_x[13];
   out_1039242481979313707[14] = delta_x[14] + nom_x[14];
   out_1039242481979313707[15] = delta_x[15] + nom_x[15];
   out_1039242481979313707[16] = delta_x[16] + nom_x[16];
   out_1039242481979313707[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5271632768120927696) {
   out_5271632768120927696[0] = -nom_x[0] + true_x[0];
   out_5271632768120927696[1] = -nom_x[1] + true_x[1];
   out_5271632768120927696[2] = -nom_x[2] + true_x[2];
   out_5271632768120927696[3] = -nom_x[3] + true_x[3];
   out_5271632768120927696[4] = -nom_x[4] + true_x[4];
   out_5271632768120927696[5] = -nom_x[5] + true_x[5];
   out_5271632768120927696[6] = -nom_x[6] + true_x[6];
   out_5271632768120927696[7] = -nom_x[7] + true_x[7];
   out_5271632768120927696[8] = -nom_x[8] + true_x[8];
   out_5271632768120927696[9] = -nom_x[9] + true_x[9];
   out_5271632768120927696[10] = -nom_x[10] + true_x[10];
   out_5271632768120927696[11] = -nom_x[11] + true_x[11];
   out_5271632768120927696[12] = -nom_x[12] + true_x[12];
   out_5271632768120927696[13] = -nom_x[13] + true_x[13];
   out_5271632768120927696[14] = -nom_x[14] + true_x[14];
   out_5271632768120927696[15] = -nom_x[15] + true_x[15];
   out_5271632768120927696[16] = -nom_x[16] + true_x[16];
   out_5271632768120927696[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_7382780514853116996) {
   out_7382780514853116996[0] = 1.0;
   out_7382780514853116996[1] = 0.0;
   out_7382780514853116996[2] = 0.0;
   out_7382780514853116996[3] = 0.0;
   out_7382780514853116996[4] = 0.0;
   out_7382780514853116996[5] = 0.0;
   out_7382780514853116996[6] = 0.0;
   out_7382780514853116996[7] = 0.0;
   out_7382780514853116996[8] = 0.0;
   out_7382780514853116996[9] = 0.0;
   out_7382780514853116996[10] = 0.0;
   out_7382780514853116996[11] = 0.0;
   out_7382780514853116996[12] = 0.0;
   out_7382780514853116996[13] = 0.0;
   out_7382780514853116996[14] = 0.0;
   out_7382780514853116996[15] = 0.0;
   out_7382780514853116996[16] = 0.0;
   out_7382780514853116996[17] = 0.0;
   out_7382780514853116996[18] = 0.0;
   out_7382780514853116996[19] = 1.0;
   out_7382780514853116996[20] = 0.0;
   out_7382780514853116996[21] = 0.0;
   out_7382780514853116996[22] = 0.0;
   out_7382780514853116996[23] = 0.0;
   out_7382780514853116996[24] = 0.0;
   out_7382780514853116996[25] = 0.0;
   out_7382780514853116996[26] = 0.0;
   out_7382780514853116996[27] = 0.0;
   out_7382780514853116996[28] = 0.0;
   out_7382780514853116996[29] = 0.0;
   out_7382780514853116996[30] = 0.0;
   out_7382780514853116996[31] = 0.0;
   out_7382780514853116996[32] = 0.0;
   out_7382780514853116996[33] = 0.0;
   out_7382780514853116996[34] = 0.0;
   out_7382780514853116996[35] = 0.0;
   out_7382780514853116996[36] = 0.0;
   out_7382780514853116996[37] = 0.0;
   out_7382780514853116996[38] = 1.0;
   out_7382780514853116996[39] = 0.0;
   out_7382780514853116996[40] = 0.0;
   out_7382780514853116996[41] = 0.0;
   out_7382780514853116996[42] = 0.0;
   out_7382780514853116996[43] = 0.0;
   out_7382780514853116996[44] = 0.0;
   out_7382780514853116996[45] = 0.0;
   out_7382780514853116996[46] = 0.0;
   out_7382780514853116996[47] = 0.0;
   out_7382780514853116996[48] = 0.0;
   out_7382780514853116996[49] = 0.0;
   out_7382780514853116996[50] = 0.0;
   out_7382780514853116996[51] = 0.0;
   out_7382780514853116996[52] = 0.0;
   out_7382780514853116996[53] = 0.0;
   out_7382780514853116996[54] = 0.0;
   out_7382780514853116996[55] = 0.0;
   out_7382780514853116996[56] = 0.0;
   out_7382780514853116996[57] = 1.0;
   out_7382780514853116996[58] = 0.0;
   out_7382780514853116996[59] = 0.0;
   out_7382780514853116996[60] = 0.0;
   out_7382780514853116996[61] = 0.0;
   out_7382780514853116996[62] = 0.0;
   out_7382780514853116996[63] = 0.0;
   out_7382780514853116996[64] = 0.0;
   out_7382780514853116996[65] = 0.0;
   out_7382780514853116996[66] = 0.0;
   out_7382780514853116996[67] = 0.0;
   out_7382780514853116996[68] = 0.0;
   out_7382780514853116996[69] = 0.0;
   out_7382780514853116996[70] = 0.0;
   out_7382780514853116996[71] = 0.0;
   out_7382780514853116996[72] = 0.0;
   out_7382780514853116996[73] = 0.0;
   out_7382780514853116996[74] = 0.0;
   out_7382780514853116996[75] = 0.0;
   out_7382780514853116996[76] = 1.0;
   out_7382780514853116996[77] = 0.0;
   out_7382780514853116996[78] = 0.0;
   out_7382780514853116996[79] = 0.0;
   out_7382780514853116996[80] = 0.0;
   out_7382780514853116996[81] = 0.0;
   out_7382780514853116996[82] = 0.0;
   out_7382780514853116996[83] = 0.0;
   out_7382780514853116996[84] = 0.0;
   out_7382780514853116996[85] = 0.0;
   out_7382780514853116996[86] = 0.0;
   out_7382780514853116996[87] = 0.0;
   out_7382780514853116996[88] = 0.0;
   out_7382780514853116996[89] = 0.0;
   out_7382780514853116996[90] = 0.0;
   out_7382780514853116996[91] = 0.0;
   out_7382780514853116996[92] = 0.0;
   out_7382780514853116996[93] = 0.0;
   out_7382780514853116996[94] = 0.0;
   out_7382780514853116996[95] = 1.0;
   out_7382780514853116996[96] = 0.0;
   out_7382780514853116996[97] = 0.0;
   out_7382780514853116996[98] = 0.0;
   out_7382780514853116996[99] = 0.0;
   out_7382780514853116996[100] = 0.0;
   out_7382780514853116996[101] = 0.0;
   out_7382780514853116996[102] = 0.0;
   out_7382780514853116996[103] = 0.0;
   out_7382780514853116996[104] = 0.0;
   out_7382780514853116996[105] = 0.0;
   out_7382780514853116996[106] = 0.0;
   out_7382780514853116996[107] = 0.0;
   out_7382780514853116996[108] = 0.0;
   out_7382780514853116996[109] = 0.0;
   out_7382780514853116996[110] = 0.0;
   out_7382780514853116996[111] = 0.0;
   out_7382780514853116996[112] = 0.0;
   out_7382780514853116996[113] = 0.0;
   out_7382780514853116996[114] = 1.0;
   out_7382780514853116996[115] = 0.0;
   out_7382780514853116996[116] = 0.0;
   out_7382780514853116996[117] = 0.0;
   out_7382780514853116996[118] = 0.0;
   out_7382780514853116996[119] = 0.0;
   out_7382780514853116996[120] = 0.0;
   out_7382780514853116996[121] = 0.0;
   out_7382780514853116996[122] = 0.0;
   out_7382780514853116996[123] = 0.0;
   out_7382780514853116996[124] = 0.0;
   out_7382780514853116996[125] = 0.0;
   out_7382780514853116996[126] = 0.0;
   out_7382780514853116996[127] = 0.0;
   out_7382780514853116996[128] = 0.0;
   out_7382780514853116996[129] = 0.0;
   out_7382780514853116996[130] = 0.0;
   out_7382780514853116996[131] = 0.0;
   out_7382780514853116996[132] = 0.0;
   out_7382780514853116996[133] = 1.0;
   out_7382780514853116996[134] = 0.0;
   out_7382780514853116996[135] = 0.0;
   out_7382780514853116996[136] = 0.0;
   out_7382780514853116996[137] = 0.0;
   out_7382780514853116996[138] = 0.0;
   out_7382780514853116996[139] = 0.0;
   out_7382780514853116996[140] = 0.0;
   out_7382780514853116996[141] = 0.0;
   out_7382780514853116996[142] = 0.0;
   out_7382780514853116996[143] = 0.0;
   out_7382780514853116996[144] = 0.0;
   out_7382780514853116996[145] = 0.0;
   out_7382780514853116996[146] = 0.0;
   out_7382780514853116996[147] = 0.0;
   out_7382780514853116996[148] = 0.0;
   out_7382780514853116996[149] = 0.0;
   out_7382780514853116996[150] = 0.0;
   out_7382780514853116996[151] = 0.0;
   out_7382780514853116996[152] = 1.0;
   out_7382780514853116996[153] = 0.0;
   out_7382780514853116996[154] = 0.0;
   out_7382780514853116996[155] = 0.0;
   out_7382780514853116996[156] = 0.0;
   out_7382780514853116996[157] = 0.0;
   out_7382780514853116996[158] = 0.0;
   out_7382780514853116996[159] = 0.0;
   out_7382780514853116996[160] = 0.0;
   out_7382780514853116996[161] = 0.0;
   out_7382780514853116996[162] = 0.0;
   out_7382780514853116996[163] = 0.0;
   out_7382780514853116996[164] = 0.0;
   out_7382780514853116996[165] = 0.0;
   out_7382780514853116996[166] = 0.0;
   out_7382780514853116996[167] = 0.0;
   out_7382780514853116996[168] = 0.0;
   out_7382780514853116996[169] = 0.0;
   out_7382780514853116996[170] = 0.0;
   out_7382780514853116996[171] = 1.0;
   out_7382780514853116996[172] = 0.0;
   out_7382780514853116996[173] = 0.0;
   out_7382780514853116996[174] = 0.0;
   out_7382780514853116996[175] = 0.0;
   out_7382780514853116996[176] = 0.0;
   out_7382780514853116996[177] = 0.0;
   out_7382780514853116996[178] = 0.0;
   out_7382780514853116996[179] = 0.0;
   out_7382780514853116996[180] = 0.0;
   out_7382780514853116996[181] = 0.0;
   out_7382780514853116996[182] = 0.0;
   out_7382780514853116996[183] = 0.0;
   out_7382780514853116996[184] = 0.0;
   out_7382780514853116996[185] = 0.0;
   out_7382780514853116996[186] = 0.0;
   out_7382780514853116996[187] = 0.0;
   out_7382780514853116996[188] = 0.0;
   out_7382780514853116996[189] = 0.0;
   out_7382780514853116996[190] = 1.0;
   out_7382780514853116996[191] = 0.0;
   out_7382780514853116996[192] = 0.0;
   out_7382780514853116996[193] = 0.0;
   out_7382780514853116996[194] = 0.0;
   out_7382780514853116996[195] = 0.0;
   out_7382780514853116996[196] = 0.0;
   out_7382780514853116996[197] = 0.0;
   out_7382780514853116996[198] = 0.0;
   out_7382780514853116996[199] = 0.0;
   out_7382780514853116996[200] = 0.0;
   out_7382780514853116996[201] = 0.0;
   out_7382780514853116996[202] = 0.0;
   out_7382780514853116996[203] = 0.0;
   out_7382780514853116996[204] = 0.0;
   out_7382780514853116996[205] = 0.0;
   out_7382780514853116996[206] = 0.0;
   out_7382780514853116996[207] = 0.0;
   out_7382780514853116996[208] = 0.0;
   out_7382780514853116996[209] = 1.0;
   out_7382780514853116996[210] = 0.0;
   out_7382780514853116996[211] = 0.0;
   out_7382780514853116996[212] = 0.0;
   out_7382780514853116996[213] = 0.0;
   out_7382780514853116996[214] = 0.0;
   out_7382780514853116996[215] = 0.0;
   out_7382780514853116996[216] = 0.0;
   out_7382780514853116996[217] = 0.0;
   out_7382780514853116996[218] = 0.0;
   out_7382780514853116996[219] = 0.0;
   out_7382780514853116996[220] = 0.0;
   out_7382780514853116996[221] = 0.0;
   out_7382780514853116996[222] = 0.0;
   out_7382780514853116996[223] = 0.0;
   out_7382780514853116996[224] = 0.0;
   out_7382780514853116996[225] = 0.0;
   out_7382780514853116996[226] = 0.0;
   out_7382780514853116996[227] = 0.0;
   out_7382780514853116996[228] = 1.0;
   out_7382780514853116996[229] = 0.0;
   out_7382780514853116996[230] = 0.0;
   out_7382780514853116996[231] = 0.0;
   out_7382780514853116996[232] = 0.0;
   out_7382780514853116996[233] = 0.0;
   out_7382780514853116996[234] = 0.0;
   out_7382780514853116996[235] = 0.0;
   out_7382780514853116996[236] = 0.0;
   out_7382780514853116996[237] = 0.0;
   out_7382780514853116996[238] = 0.0;
   out_7382780514853116996[239] = 0.0;
   out_7382780514853116996[240] = 0.0;
   out_7382780514853116996[241] = 0.0;
   out_7382780514853116996[242] = 0.0;
   out_7382780514853116996[243] = 0.0;
   out_7382780514853116996[244] = 0.0;
   out_7382780514853116996[245] = 0.0;
   out_7382780514853116996[246] = 0.0;
   out_7382780514853116996[247] = 1.0;
   out_7382780514853116996[248] = 0.0;
   out_7382780514853116996[249] = 0.0;
   out_7382780514853116996[250] = 0.0;
   out_7382780514853116996[251] = 0.0;
   out_7382780514853116996[252] = 0.0;
   out_7382780514853116996[253] = 0.0;
   out_7382780514853116996[254] = 0.0;
   out_7382780514853116996[255] = 0.0;
   out_7382780514853116996[256] = 0.0;
   out_7382780514853116996[257] = 0.0;
   out_7382780514853116996[258] = 0.0;
   out_7382780514853116996[259] = 0.0;
   out_7382780514853116996[260] = 0.0;
   out_7382780514853116996[261] = 0.0;
   out_7382780514853116996[262] = 0.0;
   out_7382780514853116996[263] = 0.0;
   out_7382780514853116996[264] = 0.0;
   out_7382780514853116996[265] = 0.0;
   out_7382780514853116996[266] = 1.0;
   out_7382780514853116996[267] = 0.0;
   out_7382780514853116996[268] = 0.0;
   out_7382780514853116996[269] = 0.0;
   out_7382780514853116996[270] = 0.0;
   out_7382780514853116996[271] = 0.0;
   out_7382780514853116996[272] = 0.0;
   out_7382780514853116996[273] = 0.0;
   out_7382780514853116996[274] = 0.0;
   out_7382780514853116996[275] = 0.0;
   out_7382780514853116996[276] = 0.0;
   out_7382780514853116996[277] = 0.0;
   out_7382780514853116996[278] = 0.0;
   out_7382780514853116996[279] = 0.0;
   out_7382780514853116996[280] = 0.0;
   out_7382780514853116996[281] = 0.0;
   out_7382780514853116996[282] = 0.0;
   out_7382780514853116996[283] = 0.0;
   out_7382780514853116996[284] = 0.0;
   out_7382780514853116996[285] = 1.0;
   out_7382780514853116996[286] = 0.0;
   out_7382780514853116996[287] = 0.0;
   out_7382780514853116996[288] = 0.0;
   out_7382780514853116996[289] = 0.0;
   out_7382780514853116996[290] = 0.0;
   out_7382780514853116996[291] = 0.0;
   out_7382780514853116996[292] = 0.0;
   out_7382780514853116996[293] = 0.0;
   out_7382780514853116996[294] = 0.0;
   out_7382780514853116996[295] = 0.0;
   out_7382780514853116996[296] = 0.0;
   out_7382780514853116996[297] = 0.0;
   out_7382780514853116996[298] = 0.0;
   out_7382780514853116996[299] = 0.0;
   out_7382780514853116996[300] = 0.0;
   out_7382780514853116996[301] = 0.0;
   out_7382780514853116996[302] = 0.0;
   out_7382780514853116996[303] = 0.0;
   out_7382780514853116996[304] = 1.0;
   out_7382780514853116996[305] = 0.0;
   out_7382780514853116996[306] = 0.0;
   out_7382780514853116996[307] = 0.0;
   out_7382780514853116996[308] = 0.0;
   out_7382780514853116996[309] = 0.0;
   out_7382780514853116996[310] = 0.0;
   out_7382780514853116996[311] = 0.0;
   out_7382780514853116996[312] = 0.0;
   out_7382780514853116996[313] = 0.0;
   out_7382780514853116996[314] = 0.0;
   out_7382780514853116996[315] = 0.0;
   out_7382780514853116996[316] = 0.0;
   out_7382780514853116996[317] = 0.0;
   out_7382780514853116996[318] = 0.0;
   out_7382780514853116996[319] = 0.0;
   out_7382780514853116996[320] = 0.0;
   out_7382780514853116996[321] = 0.0;
   out_7382780514853116996[322] = 0.0;
   out_7382780514853116996[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7524264837216844965) {
   out_7524264837216844965[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7524264837216844965[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7524264837216844965[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7524264837216844965[3] = dt*state[12] + state[3];
   out_7524264837216844965[4] = dt*state[13] + state[4];
   out_7524264837216844965[5] = dt*state[14] + state[5];
   out_7524264837216844965[6] = state[6];
   out_7524264837216844965[7] = state[7];
   out_7524264837216844965[8] = state[8];
   out_7524264837216844965[9] = state[9];
   out_7524264837216844965[10] = state[10];
   out_7524264837216844965[11] = state[11];
   out_7524264837216844965[12] = state[12];
   out_7524264837216844965[13] = state[13];
   out_7524264837216844965[14] = state[14];
   out_7524264837216844965[15] = state[15];
   out_7524264837216844965[16] = state[16];
   out_7524264837216844965[17] = state[17];
}
void F_fun(double *state, double dt, double *out_7682706624390115746) {
   out_7682706624390115746[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7682706624390115746[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7682706624390115746[2] = 0;
   out_7682706624390115746[3] = 0;
   out_7682706624390115746[4] = 0;
   out_7682706624390115746[5] = 0;
   out_7682706624390115746[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7682706624390115746[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7682706624390115746[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7682706624390115746[9] = 0;
   out_7682706624390115746[10] = 0;
   out_7682706624390115746[11] = 0;
   out_7682706624390115746[12] = 0;
   out_7682706624390115746[13] = 0;
   out_7682706624390115746[14] = 0;
   out_7682706624390115746[15] = 0;
   out_7682706624390115746[16] = 0;
   out_7682706624390115746[17] = 0;
   out_7682706624390115746[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7682706624390115746[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7682706624390115746[20] = 0;
   out_7682706624390115746[21] = 0;
   out_7682706624390115746[22] = 0;
   out_7682706624390115746[23] = 0;
   out_7682706624390115746[24] = 0;
   out_7682706624390115746[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7682706624390115746[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7682706624390115746[27] = 0;
   out_7682706624390115746[28] = 0;
   out_7682706624390115746[29] = 0;
   out_7682706624390115746[30] = 0;
   out_7682706624390115746[31] = 0;
   out_7682706624390115746[32] = 0;
   out_7682706624390115746[33] = 0;
   out_7682706624390115746[34] = 0;
   out_7682706624390115746[35] = 0;
   out_7682706624390115746[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7682706624390115746[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7682706624390115746[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7682706624390115746[39] = 0;
   out_7682706624390115746[40] = 0;
   out_7682706624390115746[41] = 0;
   out_7682706624390115746[42] = 0;
   out_7682706624390115746[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7682706624390115746[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7682706624390115746[45] = 0;
   out_7682706624390115746[46] = 0;
   out_7682706624390115746[47] = 0;
   out_7682706624390115746[48] = 0;
   out_7682706624390115746[49] = 0;
   out_7682706624390115746[50] = 0;
   out_7682706624390115746[51] = 0;
   out_7682706624390115746[52] = 0;
   out_7682706624390115746[53] = 0;
   out_7682706624390115746[54] = 0;
   out_7682706624390115746[55] = 0;
   out_7682706624390115746[56] = 0;
   out_7682706624390115746[57] = 1;
   out_7682706624390115746[58] = 0;
   out_7682706624390115746[59] = 0;
   out_7682706624390115746[60] = 0;
   out_7682706624390115746[61] = 0;
   out_7682706624390115746[62] = 0;
   out_7682706624390115746[63] = 0;
   out_7682706624390115746[64] = 0;
   out_7682706624390115746[65] = 0;
   out_7682706624390115746[66] = dt;
   out_7682706624390115746[67] = 0;
   out_7682706624390115746[68] = 0;
   out_7682706624390115746[69] = 0;
   out_7682706624390115746[70] = 0;
   out_7682706624390115746[71] = 0;
   out_7682706624390115746[72] = 0;
   out_7682706624390115746[73] = 0;
   out_7682706624390115746[74] = 0;
   out_7682706624390115746[75] = 0;
   out_7682706624390115746[76] = 1;
   out_7682706624390115746[77] = 0;
   out_7682706624390115746[78] = 0;
   out_7682706624390115746[79] = 0;
   out_7682706624390115746[80] = 0;
   out_7682706624390115746[81] = 0;
   out_7682706624390115746[82] = 0;
   out_7682706624390115746[83] = 0;
   out_7682706624390115746[84] = 0;
   out_7682706624390115746[85] = dt;
   out_7682706624390115746[86] = 0;
   out_7682706624390115746[87] = 0;
   out_7682706624390115746[88] = 0;
   out_7682706624390115746[89] = 0;
   out_7682706624390115746[90] = 0;
   out_7682706624390115746[91] = 0;
   out_7682706624390115746[92] = 0;
   out_7682706624390115746[93] = 0;
   out_7682706624390115746[94] = 0;
   out_7682706624390115746[95] = 1;
   out_7682706624390115746[96] = 0;
   out_7682706624390115746[97] = 0;
   out_7682706624390115746[98] = 0;
   out_7682706624390115746[99] = 0;
   out_7682706624390115746[100] = 0;
   out_7682706624390115746[101] = 0;
   out_7682706624390115746[102] = 0;
   out_7682706624390115746[103] = 0;
   out_7682706624390115746[104] = dt;
   out_7682706624390115746[105] = 0;
   out_7682706624390115746[106] = 0;
   out_7682706624390115746[107] = 0;
   out_7682706624390115746[108] = 0;
   out_7682706624390115746[109] = 0;
   out_7682706624390115746[110] = 0;
   out_7682706624390115746[111] = 0;
   out_7682706624390115746[112] = 0;
   out_7682706624390115746[113] = 0;
   out_7682706624390115746[114] = 1;
   out_7682706624390115746[115] = 0;
   out_7682706624390115746[116] = 0;
   out_7682706624390115746[117] = 0;
   out_7682706624390115746[118] = 0;
   out_7682706624390115746[119] = 0;
   out_7682706624390115746[120] = 0;
   out_7682706624390115746[121] = 0;
   out_7682706624390115746[122] = 0;
   out_7682706624390115746[123] = 0;
   out_7682706624390115746[124] = 0;
   out_7682706624390115746[125] = 0;
   out_7682706624390115746[126] = 0;
   out_7682706624390115746[127] = 0;
   out_7682706624390115746[128] = 0;
   out_7682706624390115746[129] = 0;
   out_7682706624390115746[130] = 0;
   out_7682706624390115746[131] = 0;
   out_7682706624390115746[132] = 0;
   out_7682706624390115746[133] = 1;
   out_7682706624390115746[134] = 0;
   out_7682706624390115746[135] = 0;
   out_7682706624390115746[136] = 0;
   out_7682706624390115746[137] = 0;
   out_7682706624390115746[138] = 0;
   out_7682706624390115746[139] = 0;
   out_7682706624390115746[140] = 0;
   out_7682706624390115746[141] = 0;
   out_7682706624390115746[142] = 0;
   out_7682706624390115746[143] = 0;
   out_7682706624390115746[144] = 0;
   out_7682706624390115746[145] = 0;
   out_7682706624390115746[146] = 0;
   out_7682706624390115746[147] = 0;
   out_7682706624390115746[148] = 0;
   out_7682706624390115746[149] = 0;
   out_7682706624390115746[150] = 0;
   out_7682706624390115746[151] = 0;
   out_7682706624390115746[152] = 1;
   out_7682706624390115746[153] = 0;
   out_7682706624390115746[154] = 0;
   out_7682706624390115746[155] = 0;
   out_7682706624390115746[156] = 0;
   out_7682706624390115746[157] = 0;
   out_7682706624390115746[158] = 0;
   out_7682706624390115746[159] = 0;
   out_7682706624390115746[160] = 0;
   out_7682706624390115746[161] = 0;
   out_7682706624390115746[162] = 0;
   out_7682706624390115746[163] = 0;
   out_7682706624390115746[164] = 0;
   out_7682706624390115746[165] = 0;
   out_7682706624390115746[166] = 0;
   out_7682706624390115746[167] = 0;
   out_7682706624390115746[168] = 0;
   out_7682706624390115746[169] = 0;
   out_7682706624390115746[170] = 0;
   out_7682706624390115746[171] = 1;
   out_7682706624390115746[172] = 0;
   out_7682706624390115746[173] = 0;
   out_7682706624390115746[174] = 0;
   out_7682706624390115746[175] = 0;
   out_7682706624390115746[176] = 0;
   out_7682706624390115746[177] = 0;
   out_7682706624390115746[178] = 0;
   out_7682706624390115746[179] = 0;
   out_7682706624390115746[180] = 0;
   out_7682706624390115746[181] = 0;
   out_7682706624390115746[182] = 0;
   out_7682706624390115746[183] = 0;
   out_7682706624390115746[184] = 0;
   out_7682706624390115746[185] = 0;
   out_7682706624390115746[186] = 0;
   out_7682706624390115746[187] = 0;
   out_7682706624390115746[188] = 0;
   out_7682706624390115746[189] = 0;
   out_7682706624390115746[190] = 1;
   out_7682706624390115746[191] = 0;
   out_7682706624390115746[192] = 0;
   out_7682706624390115746[193] = 0;
   out_7682706624390115746[194] = 0;
   out_7682706624390115746[195] = 0;
   out_7682706624390115746[196] = 0;
   out_7682706624390115746[197] = 0;
   out_7682706624390115746[198] = 0;
   out_7682706624390115746[199] = 0;
   out_7682706624390115746[200] = 0;
   out_7682706624390115746[201] = 0;
   out_7682706624390115746[202] = 0;
   out_7682706624390115746[203] = 0;
   out_7682706624390115746[204] = 0;
   out_7682706624390115746[205] = 0;
   out_7682706624390115746[206] = 0;
   out_7682706624390115746[207] = 0;
   out_7682706624390115746[208] = 0;
   out_7682706624390115746[209] = 1;
   out_7682706624390115746[210] = 0;
   out_7682706624390115746[211] = 0;
   out_7682706624390115746[212] = 0;
   out_7682706624390115746[213] = 0;
   out_7682706624390115746[214] = 0;
   out_7682706624390115746[215] = 0;
   out_7682706624390115746[216] = 0;
   out_7682706624390115746[217] = 0;
   out_7682706624390115746[218] = 0;
   out_7682706624390115746[219] = 0;
   out_7682706624390115746[220] = 0;
   out_7682706624390115746[221] = 0;
   out_7682706624390115746[222] = 0;
   out_7682706624390115746[223] = 0;
   out_7682706624390115746[224] = 0;
   out_7682706624390115746[225] = 0;
   out_7682706624390115746[226] = 0;
   out_7682706624390115746[227] = 0;
   out_7682706624390115746[228] = 1;
   out_7682706624390115746[229] = 0;
   out_7682706624390115746[230] = 0;
   out_7682706624390115746[231] = 0;
   out_7682706624390115746[232] = 0;
   out_7682706624390115746[233] = 0;
   out_7682706624390115746[234] = 0;
   out_7682706624390115746[235] = 0;
   out_7682706624390115746[236] = 0;
   out_7682706624390115746[237] = 0;
   out_7682706624390115746[238] = 0;
   out_7682706624390115746[239] = 0;
   out_7682706624390115746[240] = 0;
   out_7682706624390115746[241] = 0;
   out_7682706624390115746[242] = 0;
   out_7682706624390115746[243] = 0;
   out_7682706624390115746[244] = 0;
   out_7682706624390115746[245] = 0;
   out_7682706624390115746[246] = 0;
   out_7682706624390115746[247] = 1;
   out_7682706624390115746[248] = 0;
   out_7682706624390115746[249] = 0;
   out_7682706624390115746[250] = 0;
   out_7682706624390115746[251] = 0;
   out_7682706624390115746[252] = 0;
   out_7682706624390115746[253] = 0;
   out_7682706624390115746[254] = 0;
   out_7682706624390115746[255] = 0;
   out_7682706624390115746[256] = 0;
   out_7682706624390115746[257] = 0;
   out_7682706624390115746[258] = 0;
   out_7682706624390115746[259] = 0;
   out_7682706624390115746[260] = 0;
   out_7682706624390115746[261] = 0;
   out_7682706624390115746[262] = 0;
   out_7682706624390115746[263] = 0;
   out_7682706624390115746[264] = 0;
   out_7682706624390115746[265] = 0;
   out_7682706624390115746[266] = 1;
   out_7682706624390115746[267] = 0;
   out_7682706624390115746[268] = 0;
   out_7682706624390115746[269] = 0;
   out_7682706624390115746[270] = 0;
   out_7682706624390115746[271] = 0;
   out_7682706624390115746[272] = 0;
   out_7682706624390115746[273] = 0;
   out_7682706624390115746[274] = 0;
   out_7682706624390115746[275] = 0;
   out_7682706624390115746[276] = 0;
   out_7682706624390115746[277] = 0;
   out_7682706624390115746[278] = 0;
   out_7682706624390115746[279] = 0;
   out_7682706624390115746[280] = 0;
   out_7682706624390115746[281] = 0;
   out_7682706624390115746[282] = 0;
   out_7682706624390115746[283] = 0;
   out_7682706624390115746[284] = 0;
   out_7682706624390115746[285] = 1;
   out_7682706624390115746[286] = 0;
   out_7682706624390115746[287] = 0;
   out_7682706624390115746[288] = 0;
   out_7682706624390115746[289] = 0;
   out_7682706624390115746[290] = 0;
   out_7682706624390115746[291] = 0;
   out_7682706624390115746[292] = 0;
   out_7682706624390115746[293] = 0;
   out_7682706624390115746[294] = 0;
   out_7682706624390115746[295] = 0;
   out_7682706624390115746[296] = 0;
   out_7682706624390115746[297] = 0;
   out_7682706624390115746[298] = 0;
   out_7682706624390115746[299] = 0;
   out_7682706624390115746[300] = 0;
   out_7682706624390115746[301] = 0;
   out_7682706624390115746[302] = 0;
   out_7682706624390115746[303] = 0;
   out_7682706624390115746[304] = 1;
   out_7682706624390115746[305] = 0;
   out_7682706624390115746[306] = 0;
   out_7682706624390115746[307] = 0;
   out_7682706624390115746[308] = 0;
   out_7682706624390115746[309] = 0;
   out_7682706624390115746[310] = 0;
   out_7682706624390115746[311] = 0;
   out_7682706624390115746[312] = 0;
   out_7682706624390115746[313] = 0;
   out_7682706624390115746[314] = 0;
   out_7682706624390115746[315] = 0;
   out_7682706624390115746[316] = 0;
   out_7682706624390115746[317] = 0;
   out_7682706624390115746[318] = 0;
   out_7682706624390115746[319] = 0;
   out_7682706624390115746[320] = 0;
   out_7682706624390115746[321] = 0;
   out_7682706624390115746[322] = 0;
   out_7682706624390115746[323] = 1;
}
void h_4(double *state, double *unused, double *out_1857154749449920576) {
   out_1857154749449920576[0] = state[6] + state[9];
   out_1857154749449920576[1] = state[7] + state[10];
   out_1857154749449920576[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_5215881752104475899) {
   out_5215881752104475899[0] = 0;
   out_5215881752104475899[1] = 0;
   out_5215881752104475899[2] = 0;
   out_5215881752104475899[3] = 0;
   out_5215881752104475899[4] = 0;
   out_5215881752104475899[5] = 0;
   out_5215881752104475899[6] = 1;
   out_5215881752104475899[7] = 0;
   out_5215881752104475899[8] = 0;
   out_5215881752104475899[9] = 1;
   out_5215881752104475899[10] = 0;
   out_5215881752104475899[11] = 0;
   out_5215881752104475899[12] = 0;
   out_5215881752104475899[13] = 0;
   out_5215881752104475899[14] = 0;
   out_5215881752104475899[15] = 0;
   out_5215881752104475899[16] = 0;
   out_5215881752104475899[17] = 0;
   out_5215881752104475899[18] = 0;
   out_5215881752104475899[19] = 0;
   out_5215881752104475899[20] = 0;
   out_5215881752104475899[21] = 0;
   out_5215881752104475899[22] = 0;
   out_5215881752104475899[23] = 0;
   out_5215881752104475899[24] = 0;
   out_5215881752104475899[25] = 1;
   out_5215881752104475899[26] = 0;
   out_5215881752104475899[27] = 0;
   out_5215881752104475899[28] = 1;
   out_5215881752104475899[29] = 0;
   out_5215881752104475899[30] = 0;
   out_5215881752104475899[31] = 0;
   out_5215881752104475899[32] = 0;
   out_5215881752104475899[33] = 0;
   out_5215881752104475899[34] = 0;
   out_5215881752104475899[35] = 0;
   out_5215881752104475899[36] = 0;
   out_5215881752104475899[37] = 0;
   out_5215881752104475899[38] = 0;
   out_5215881752104475899[39] = 0;
   out_5215881752104475899[40] = 0;
   out_5215881752104475899[41] = 0;
   out_5215881752104475899[42] = 0;
   out_5215881752104475899[43] = 0;
   out_5215881752104475899[44] = 1;
   out_5215881752104475899[45] = 0;
   out_5215881752104475899[46] = 0;
   out_5215881752104475899[47] = 1;
   out_5215881752104475899[48] = 0;
   out_5215881752104475899[49] = 0;
   out_5215881752104475899[50] = 0;
   out_5215881752104475899[51] = 0;
   out_5215881752104475899[52] = 0;
   out_5215881752104475899[53] = 0;
}
void h_10(double *state, double *unused, double *out_271235286086808204) {
   out_271235286086808204[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_271235286086808204[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_271235286086808204[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2135374553292641286) {
   out_2135374553292641286[0] = 0;
   out_2135374553292641286[1] = 9.8100000000000005*cos(state[1]);
   out_2135374553292641286[2] = 0;
   out_2135374553292641286[3] = 0;
   out_2135374553292641286[4] = -state[8];
   out_2135374553292641286[5] = state[7];
   out_2135374553292641286[6] = 0;
   out_2135374553292641286[7] = state[5];
   out_2135374553292641286[8] = -state[4];
   out_2135374553292641286[9] = 0;
   out_2135374553292641286[10] = 0;
   out_2135374553292641286[11] = 0;
   out_2135374553292641286[12] = 1;
   out_2135374553292641286[13] = 0;
   out_2135374553292641286[14] = 0;
   out_2135374553292641286[15] = 1;
   out_2135374553292641286[16] = 0;
   out_2135374553292641286[17] = 0;
   out_2135374553292641286[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2135374553292641286[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2135374553292641286[20] = 0;
   out_2135374553292641286[21] = state[8];
   out_2135374553292641286[22] = 0;
   out_2135374553292641286[23] = -state[6];
   out_2135374553292641286[24] = -state[5];
   out_2135374553292641286[25] = 0;
   out_2135374553292641286[26] = state[3];
   out_2135374553292641286[27] = 0;
   out_2135374553292641286[28] = 0;
   out_2135374553292641286[29] = 0;
   out_2135374553292641286[30] = 0;
   out_2135374553292641286[31] = 1;
   out_2135374553292641286[32] = 0;
   out_2135374553292641286[33] = 0;
   out_2135374553292641286[34] = 1;
   out_2135374553292641286[35] = 0;
   out_2135374553292641286[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2135374553292641286[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2135374553292641286[38] = 0;
   out_2135374553292641286[39] = -state[7];
   out_2135374553292641286[40] = state[6];
   out_2135374553292641286[41] = 0;
   out_2135374553292641286[42] = state[4];
   out_2135374553292641286[43] = -state[3];
   out_2135374553292641286[44] = 0;
   out_2135374553292641286[45] = 0;
   out_2135374553292641286[46] = 0;
   out_2135374553292641286[47] = 0;
   out_2135374553292641286[48] = 0;
   out_2135374553292641286[49] = 0;
   out_2135374553292641286[50] = 1;
   out_2135374553292641286[51] = 0;
   out_2135374553292641286[52] = 0;
   out_2135374553292641286[53] = 1;
}
void h_13(double *state, double *unused, double *out_9186029837407148995) {
   out_9186029837407148995[0] = state[3];
   out_9186029837407148995[1] = state[4];
   out_9186029837407148995[2] = state[5];
}
void H_13(double *state, double *unused, double *out_8428155577436808700) {
   out_8428155577436808700[0] = 0;
   out_8428155577436808700[1] = 0;
   out_8428155577436808700[2] = 0;
   out_8428155577436808700[3] = 1;
   out_8428155577436808700[4] = 0;
   out_8428155577436808700[5] = 0;
   out_8428155577436808700[6] = 0;
   out_8428155577436808700[7] = 0;
   out_8428155577436808700[8] = 0;
   out_8428155577436808700[9] = 0;
   out_8428155577436808700[10] = 0;
   out_8428155577436808700[11] = 0;
   out_8428155577436808700[12] = 0;
   out_8428155577436808700[13] = 0;
   out_8428155577436808700[14] = 0;
   out_8428155577436808700[15] = 0;
   out_8428155577436808700[16] = 0;
   out_8428155577436808700[17] = 0;
   out_8428155577436808700[18] = 0;
   out_8428155577436808700[19] = 0;
   out_8428155577436808700[20] = 0;
   out_8428155577436808700[21] = 0;
   out_8428155577436808700[22] = 1;
   out_8428155577436808700[23] = 0;
   out_8428155577436808700[24] = 0;
   out_8428155577436808700[25] = 0;
   out_8428155577436808700[26] = 0;
   out_8428155577436808700[27] = 0;
   out_8428155577436808700[28] = 0;
   out_8428155577436808700[29] = 0;
   out_8428155577436808700[30] = 0;
   out_8428155577436808700[31] = 0;
   out_8428155577436808700[32] = 0;
   out_8428155577436808700[33] = 0;
   out_8428155577436808700[34] = 0;
   out_8428155577436808700[35] = 0;
   out_8428155577436808700[36] = 0;
   out_8428155577436808700[37] = 0;
   out_8428155577436808700[38] = 0;
   out_8428155577436808700[39] = 0;
   out_8428155577436808700[40] = 0;
   out_8428155577436808700[41] = 1;
   out_8428155577436808700[42] = 0;
   out_8428155577436808700[43] = 0;
   out_8428155577436808700[44] = 0;
   out_8428155577436808700[45] = 0;
   out_8428155577436808700[46] = 0;
   out_8428155577436808700[47] = 0;
   out_8428155577436808700[48] = 0;
   out_8428155577436808700[49] = 0;
   out_8428155577436808700[50] = 0;
   out_8428155577436808700[51] = 0;
   out_8428155577436808700[52] = 0;
   out_8428155577436808700[53] = 0;
}
void h_14(double *state, double *unused, double *out_2185818957137124734) {
   out_2185818957137124734[0] = state[6];
   out_2185818957137124734[1] = state[7];
   out_2185818957137124734[2] = state[8];
}
void H_14(double *state, double *unused, double *out_9179122608443960428) {
   out_9179122608443960428[0] = 0;
   out_9179122608443960428[1] = 0;
   out_9179122608443960428[2] = 0;
   out_9179122608443960428[3] = 0;
   out_9179122608443960428[4] = 0;
   out_9179122608443960428[5] = 0;
   out_9179122608443960428[6] = 1;
   out_9179122608443960428[7] = 0;
   out_9179122608443960428[8] = 0;
   out_9179122608443960428[9] = 0;
   out_9179122608443960428[10] = 0;
   out_9179122608443960428[11] = 0;
   out_9179122608443960428[12] = 0;
   out_9179122608443960428[13] = 0;
   out_9179122608443960428[14] = 0;
   out_9179122608443960428[15] = 0;
   out_9179122608443960428[16] = 0;
   out_9179122608443960428[17] = 0;
   out_9179122608443960428[18] = 0;
   out_9179122608443960428[19] = 0;
   out_9179122608443960428[20] = 0;
   out_9179122608443960428[21] = 0;
   out_9179122608443960428[22] = 0;
   out_9179122608443960428[23] = 0;
   out_9179122608443960428[24] = 0;
   out_9179122608443960428[25] = 1;
   out_9179122608443960428[26] = 0;
   out_9179122608443960428[27] = 0;
   out_9179122608443960428[28] = 0;
   out_9179122608443960428[29] = 0;
   out_9179122608443960428[30] = 0;
   out_9179122608443960428[31] = 0;
   out_9179122608443960428[32] = 0;
   out_9179122608443960428[33] = 0;
   out_9179122608443960428[34] = 0;
   out_9179122608443960428[35] = 0;
   out_9179122608443960428[36] = 0;
   out_9179122608443960428[37] = 0;
   out_9179122608443960428[38] = 0;
   out_9179122608443960428[39] = 0;
   out_9179122608443960428[40] = 0;
   out_9179122608443960428[41] = 0;
   out_9179122608443960428[42] = 0;
   out_9179122608443960428[43] = 0;
   out_9179122608443960428[44] = 1;
   out_9179122608443960428[45] = 0;
   out_9179122608443960428[46] = 0;
   out_9179122608443960428[47] = 0;
   out_9179122608443960428[48] = 0;
   out_9179122608443960428[49] = 0;
   out_9179122608443960428[50] = 0;
   out_9179122608443960428[51] = 0;
   out_9179122608443960428[52] = 0;
   out_9179122608443960428[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_1039242481979313707) {
  err_fun(nom_x, delta_x, out_1039242481979313707);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5271632768120927696) {
  inv_err_fun(nom_x, true_x, out_5271632768120927696);
}
void pose_H_mod_fun(double *state, double *out_7382780514853116996) {
  H_mod_fun(state, out_7382780514853116996);
}
void pose_f_fun(double *state, double dt, double *out_7524264837216844965) {
  f_fun(state,  dt, out_7524264837216844965);
}
void pose_F_fun(double *state, double dt, double *out_7682706624390115746) {
  F_fun(state,  dt, out_7682706624390115746);
}
void pose_h_4(double *state, double *unused, double *out_1857154749449920576) {
  h_4(state, unused, out_1857154749449920576);
}
void pose_H_4(double *state, double *unused, double *out_5215881752104475899) {
  H_4(state, unused, out_5215881752104475899);
}
void pose_h_10(double *state, double *unused, double *out_271235286086808204) {
  h_10(state, unused, out_271235286086808204);
}
void pose_H_10(double *state, double *unused, double *out_2135374553292641286) {
  H_10(state, unused, out_2135374553292641286);
}
void pose_h_13(double *state, double *unused, double *out_9186029837407148995) {
  h_13(state, unused, out_9186029837407148995);
}
void pose_H_13(double *state, double *unused, double *out_8428155577436808700) {
  H_13(state, unused, out_8428155577436808700);
}
void pose_h_14(double *state, double *unused, double *out_2185818957137124734) {
  h_14(state, unused, out_2185818957137124734);
}
void pose_H_14(double *state, double *unused, double *out_9179122608443960428) {
  H_14(state, unused, out_9179122608443960428);
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
