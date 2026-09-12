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
void err_fun(double *nom_x, double *delta_x, double *out_7210081177362107870) {
   out_7210081177362107870[0] = delta_x[0] + nom_x[0];
   out_7210081177362107870[1] = delta_x[1] + nom_x[1];
   out_7210081177362107870[2] = delta_x[2] + nom_x[2];
   out_7210081177362107870[3] = delta_x[3] + nom_x[3];
   out_7210081177362107870[4] = delta_x[4] + nom_x[4];
   out_7210081177362107870[5] = delta_x[5] + nom_x[5];
   out_7210081177362107870[6] = delta_x[6] + nom_x[6];
   out_7210081177362107870[7] = delta_x[7] + nom_x[7];
   out_7210081177362107870[8] = delta_x[8] + nom_x[8];
   out_7210081177362107870[9] = delta_x[9] + nom_x[9];
   out_7210081177362107870[10] = delta_x[10] + nom_x[10];
   out_7210081177362107870[11] = delta_x[11] + nom_x[11];
   out_7210081177362107870[12] = delta_x[12] + nom_x[12];
   out_7210081177362107870[13] = delta_x[13] + nom_x[13];
   out_7210081177362107870[14] = delta_x[14] + nom_x[14];
   out_7210081177362107870[15] = delta_x[15] + nom_x[15];
   out_7210081177362107870[16] = delta_x[16] + nom_x[16];
   out_7210081177362107870[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8354364799378035034) {
   out_8354364799378035034[0] = -nom_x[0] + true_x[0];
   out_8354364799378035034[1] = -nom_x[1] + true_x[1];
   out_8354364799378035034[2] = -nom_x[2] + true_x[2];
   out_8354364799378035034[3] = -nom_x[3] + true_x[3];
   out_8354364799378035034[4] = -nom_x[4] + true_x[4];
   out_8354364799378035034[5] = -nom_x[5] + true_x[5];
   out_8354364799378035034[6] = -nom_x[6] + true_x[6];
   out_8354364799378035034[7] = -nom_x[7] + true_x[7];
   out_8354364799378035034[8] = -nom_x[8] + true_x[8];
   out_8354364799378035034[9] = -nom_x[9] + true_x[9];
   out_8354364799378035034[10] = -nom_x[10] + true_x[10];
   out_8354364799378035034[11] = -nom_x[11] + true_x[11];
   out_8354364799378035034[12] = -nom_x[12] + true_x[12];
   out_8354364799378035034[13] = -nom_x[13] + true_x[13];
   out_8354364799378035034[14] = -nom_x[14] + true_x[14];
   out_8354364799378035034[15] = -nom_x[15] + true_x[15];
   out_8354364799378035034[16] = -nom_x[16] + true_x[16];
   out_8354364799378035034[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1145354702242097965) {
   out_1145354702242097965[0] = 1.0;
   out_1145354702242097965[1] = 0.0;
   out_1145354702242097965[2] = 0.0;
   out_1145354702242097965[3] = 0.0;
   out_1145354702242097965[4] = 0.0;
   out_1145354702242097965[5] = 0.0;
   out_1145354702242097965[6] = 0.0;
   out_1145354702242097965[7] = 0.0;
   out_1145354702242097965[8] = 0.0;
   out_1145354702242097965[9] = 0.0;
   out_1145354702242097965[10] = 0.0;
   out_1145354702242097965[11] = 0.0;
   out_1145354702242097965[12] = 0.0;
   out_1145354702242097965[13] = 0.0;
   out_1145354702242097965[14] = 0.0;
   out_1145354702242097965[15] = 0.0;
   out_1145354702242097965[16] = 0.0;
   out_1145354702242097965[17] = 0.0;
   out_1145354702242097965[18] = 0.0;
   out_1145354702242097965[19] = 1.0;
   out_1145354702242097965[20] = 0.0;
   out_1145354702242097965[21] = 0.0;
   out_1145354702242097965[22] = 0.0;
   out_1145354702242097965[23] = 0.0;
   out_1145354702242097965[24] = 0.0;
   out_1145354702242097965[25] = 0.0;
   out_1145354702242097965[26] = 0.0;
   out_1145354702242097965[27] = 0.0;
   out_1145354702242097965[28] = 0.0;
   out_1145354702242097965[29] = 0.0;
   out_1145354702242097965[30] = 0.0;
   out_1145354702242097965[31] = 0.0;
   out_1145354702242097965[32] = 0.0;
   out_1145354702242097965[33] = 0.0;
   out_1145354702242097965[34] = 0.0;
   out_1145354702242097965[35] = 0.0;
   out_1145354702242097965[36] = 0.0;
   out_1145354702242097965[37] = 0.0;
   out_1145354702242097965[38] = 1.0;
   out_1145354702242097965[39] = 0.0;
   out_1145354702242097965[40] = 0.0;
   out_1145354702242097965[41] = 0.0;
   out_1145354702242097965[42] = 0.0;
   out_1145354702242097965[43] = 0.0;
   out_1145354702242097965[44] = 0.0;
   out_1145354702242097965[45] = 0.0;
   out_1145354702242097965[46] = 0.0;
   out_1145354702242097965[47] = 0.0;
   out_1145354702242097965[48] = 0.0;
   out_1145354702242097965[49] = 0.0;
   out_1145354702242097965[50] = 0.0;
   out_1145354702242097965[51] = 0.0;
   out_1145354702242097965[52] = 0.0;
   out_1145354702242097965[53] = 0.0;
   out_1145354702242097965[54] = 0.0;
   out_1145354702242097965[55] = 0.0;
   out_1145354702242097965[56] = 0.0;
   out_1145354702242097965[57] = 1.0;
   out_1145354702242097965[58] = 0.0;
   out_1145354702242097965[59] = 0.0;
   out_1145354702242097965[60] = 0.0;
   out_1145354702242097965[61] = 0.0;
   out_1145354702242097965[62] = 0.0;
   out_1145354702242097965[63] = 0.0;
   out_1145354702242097965[64] = 0.0;
   out_1145354702242097965[65] = 0.0;
   out_1145354702242097965[66] = 0.0;
   out_1145354702242097965[67] = 0.0;
   out_1145354702242097965[68] = 0.0;
   out_1145354702242097965[69] = 0.0;
   out_1145354702242097965[70] = 0.0;
   out_1145354702242097965[71] = 0.0;
   out_1145354702242097965[72] = 0.0;
   out_1145354702242097965[73] = 0.0;
   out_1145354702242097965[74] = 0.0;
   out_1145354702242097965[75] = 0.0;
   out_1145354702242097965[76] = 1.0;
   out_1145354702242097965[77] = 0.0;
   out_1145354702242097965[78] = 0.0;
   out_1145354702242097965[79] = 0.0;
   out_1145354702242097965[80] = 0.0;
   out_1145354702242097965[81] = 0.0;
   out_1145354702242097965[82] = 0.0;
   out_1145354702242097965[83] = 0.0;
   out_1145354702242097965[84] = 0.0;
   out_1145354702242097965[85] = 0.0;
   out_1145354702242097965[86] = 0.0;
   out_1145354702242097965[87] = 0.0;
   out_1145354702242097965[88] = 0.0;
   out_1145354702242097965[89] = 0.0;
   out_1145354702242097965[90] = 0.0;
   out_1145354702242097965[91] = 0.0;
   out_1145354702242097965[92] = 0.0;
   out_1145354702242097965[93] = 0.0;
   out_1145354702242097965[94] = 0.0;
   out_1145354702242097965[95] = 1.0;
   out_1145354702242097965[96] = 0.0;
   out_1145354702242097965[97] = 0.0;
   out_1145354702242097965[98] = 0.0;
   out_1145354702242097965[99] = 0.0;
   out_1145354702242097965[100] = 0.0;
   out_1145354702242097965[101] = 0.0;
   out_1145354702242097965[102] = 0.0;
   out_1145354702242097965[103] = 0.0;
   out_1145354702242097965[104] = 0.0;
   out_1145354702242097965[105] = 0.0;
   out_1145354702242097965[106] = 0.0;
   out_1145354702242097965[107] = 0.0;
   out_1145354702242097965[108] = 0.0;
   out_1145354702242097965[109] = 0.0;
   out_1145354702242097965[110] = 0.0;
   out_1145354702242097965[111] = 0.0;
   out_1145354702242097965[112] = 0.0;
   out_1145354702242097965[113] = 0.0;
   out_1145354702242097965[114] = 1.0;
   out_1145354702242097965[115] = 0.0;
   out_1145354702242097965[116] = 0.0;
   out_1145354702242097965[117] = 0.0;
   out_1145354702242097965[118] = 0.0;
   out_1145354702242097965[119] = 0.0;
   out_1145354702242097965[120] = 0.0;
   out_1145354702242097965[121] = 0.0;
   out_1145354702242097965[122] = 0.0;
   out_1145354702242097965[123] = 0.0;
   out_1145354702242097965[124] = 0.0;
   out_1145354702242097965[125] = 0.0;
   out_1145354702242097965[126] = 0.0;
   out_1145354702242097965[127] = 0.0;
   out_1145354702242097965[128] = 0.0;
   out_1145354702242097965[129] = 0.0;
   out_1145354702242097965[130] = 0.0;
   out_1145354702242097965[131] = 0.0;
   out_1145354702242097965[132] = 0.0;
   out_1145354702242097965[133] = 1.0;
   out_1145354702242097965[134] = 0.0;
   out_1145354702242097965[135] = 0.0;
   out_1145354702242097965[136] = 0.0;
   out_1145354702242097965[137] = 0.0;
   out_1145354702242097965[138] = 0.0;
   out_1145354702242097965[139] = 0.0;
   out_1145354702242097965[140] = 0.0;
   out_1145354702242097965[141] = 0.0;
   out_1145354702242097965[142] = 0.0;
   out_1145354702242097965[143] = 0.0;
   out_1145354702242097965[144] = 0.0;
   out_1145354702242097965[145] = 0.0;
   out_1145354702242097965[146] = 0.0;
   out_1145354702242097965[147] = 0.0;
   out_1145354702242097965[148] = 0.0;
   out_1145354702242097965[149] = 0.0;
   out_1145354702242097965[150] = 0.0;
   out_1145354702242097965[151] = 0.0;
   out_1145354702242097965[152] = 1.0;
   out_1145354702242097965[153] = 0.0;
   out_1145354702242097965[154] = 0.0;
   out_1145354702242097965[155] = 0.0;
   out_1145354702242097965[156] = 0.0;
   out_1145354702242097965[157] = 0.0;
   out_1145354702242097965[158] = 0.0;
   out_1145354702242097965[159] = 0.0;
   out_1145354702242097965[160] = 0.0;
   out_1145354702242097965[161] = 0.0;
   out_1145354702242097965[162] = 0.0;
   out_1145354702242097965[163] = 0.0;
   out_1145354702242097965[164] = 0.0;
   out_1145354702242097965[165] = 0.0;
   out_1145354702242097965[166] = 0.0;
   out_1145354702242097965[167] = 0.0;
   out_1145354702242097965[168] = 0.0;
   out_1145354702242097965[169] = 0.0;
   out_1145354702242097965[170] = 0.0;
   out_1145354702242097965[171] = 1.0;
   out_1145354702242097965[172] = 0.0;
   out_1145354702242097965[173] = 0.0;
   out_1145354702242097965[174] = 0.0;
   out_1145354702242097965[175] = 0.0;
   out_1145354702242097965[176] = 0.0;
   out_1145354702242097965[177] = 0.0;
   out_1145354702242097965[178] = 0.0;
   out_1145354702242097965[179] = 0.0;
   out_1145354702242097965[180] = 0.0;
   out_1145354702242097965[181] = 0.0;
   out_1145354702242097965[182] = 0.0;
   out_1145354702242097965[183] = 0.0;
   out_1145354702242097965[184] = 0.0;
   out_1145354702242097965[185] = 0.0;
   out_1145354702242097965[186] = 0.0;
   out_1145354702242097965[187] = 0.0;
   out_1145354702242097965[188] = 0.0;
   out_1145354702242097965[189] = 0.0;
   out_1145354702242097965[190] = 1.0;
   out_1145354702242097965[191] = 0.0;
   out_1145354702242097965[192] = 0.0;
   out_1145354702242097965[193] = 0.0;
   out_1145354702242097965[194] = 0.0;
   out_1145354702242097965[195] = 0.0;
   out_1145354702242097965[196] = 0.0;
   out_1145354702242097965[197] = 0.0;
   out_1145354702242097965[198] = 0.0;
   out_1145354702242097965[199] = 0.0;
   out_1145354702242097965[200] = 0.0;
   out_1145354702242097965[201] = 0.0;
   out_1145354702242097965[202] = 0.0;
   out_1145354702242097965[203] = 0.0;
   out_1145354702242097965[204] = 0.0;
   out_1145354702242097965[205] = 0.0;
   out_1145354702242097965[206] = 0.0;
   out_1145354702242097965[207] = 0.0;
   out_1145354702242097965[208] = 0.0;
   out_1145354702242097965[209] = 1.0;
   out_1145354702242097965[210] = 0.0;
   out_1145354702242097965[211] = 0.0;
   out_1145354702242097965[212] = 0.0;
   out_1145354702242097965[213] = 0.0;
   out_1145354702242097965[214] = 0.0;
   out_1145354702242097965[215] = 0.0;
   out_1145354702242097965[216] = 0.0;
   out_1145354702242097965[217] = 0.0;
   out_1145354702242097965[218] = 0.0;
   out_1145354702242097965[219] = 0.0;
   out_1145354702242097965[220] = 0.0;
   out_1145354702242097965[221] = 0.0;
   out_1145354702242097965[222] = 0.0;
   out_1145354702242097965[223] = 0.0;
   out_1145354702242097965[224] = 0.0;
   out_1145354702242097965[225] = 0.0;
   out_1145354702242097965[226] = 0.0;
   out_1145354702242097965[227] = 0.0;
   out_1145354702242097965[228] = 1.0;
   out_1145354702242097965[229] = 0.0;
   out_1145354702242097965[230] = 0.0;
   out_1145354702242097965[231] = 0.0;
   out_1145354702242097965[232] = 0.0;
   out_1145354702242097965[233] = 0.0;
   out_1145354702242097965[234] = 0.0;
   out_1145354702242097965[235] = 0.0;
   out_1145354702242097965[236] = 0.0;
   out_1145354702242097965[237] = 0.0;
   out_1145354702242097965[238] = 0.0;
   out_1145354702242097965[239] = 0.0;
   out_1145354702242097965[240] = 0.0;
   out_1145354702242097965[241] = 0.0;
   out_1145354702242097965[242] = 0.0;
   out_1145354702242097965[243] = 0.0;
   out_1145354702242097965[244] = 0.0;
   out_1145354702242097965[245] = 0.0;
   out_1145354702242097965[246] = 0.0;
   out_1145354702242097965[247] = 1.0;
   out_1145354702242097965[248] = 0.0;
   out_1145354702242097965[249] = 0.0;
   out_1145354702242097965[250] = 0.0;
   out_1145354702242097965[251] = 0.0;
   out_1145354702242097965[252] = 0.0;
   out_1145354702242097965[253] = 0.0;
   out_1145354702242097965[254] = 0.0;
   out_1145354702242097965[255] = 0.0;
   out_1145354702242097965[256] = 0.0;
   out_1145354702242097965[257] = 0.0;
   out_1145354702242097965[258] = 0.0;
   out_1145354702242097965[259] = 0.0;
   out_1145354702242097965[260] = 0.0;
   out_1145354702242097965[261] = 0.0;
   out_1145354702242097965[262] = 0.0;
   out_1145354702242097965[263] = 0.0;
   out_1145354702242097965[264] = 0.0;
   out_1145354702242097965[265] = 0.0;
   out_1145354702242097965[266] = 1.0;
   out_1145354702242097965[267] = 0.0;
   out_1145354702242097965[268] = 0.0;
   out_1145354702242097965[269] = 0.0;
   out_1145354702242097965[270] = 0.0;
   out_1145354702242097965[271] = 0.0;
   out_1145354702242097965[272] = 0.0;
   out_1145354702242097965[273] = 0.0;
   out_1145354702242097965[274] = 0.0;
   out_1145354702242097965[275] = 0.0;
   out_1145354702242097965[276] = 0.0;
   out_1145354702242097965[277] = 0.0;
   out_1145354702242097965[278] = 0.0;
   out_1145354702242097965[279] = 0.0;
   out_1145354702242097965[280] = 0.0;
   out_1145354702242097965[281] = 0.0;
   out_1145354702242097965[282] = 0.0;
   out_1145354702242097965[283] = 0.0;
   out_1145354702242097965[284] = 0.0;
   out_1145354702242097965[285] = 1.0;
   out_1145354702242097965[286] = 0.0;
   out_1145354702242097965[287] = 0.0;
   out_1145354702242097965[288] = 0.0;
   out_1145354702242097965[289] = 0.0;
   out_1145354702242097965[290] = 0.0;
   out_1145354702242097965[291] = 0.0;
   out_1145354702242097965[292] = 0.0;
   out_1145354702242097965[293] = 0.0;
   out_1145354702242097965[294] = 0.0;
   out_1145354702242097965[295] = 0.0;
   out_1145354702242097965[296] = 0.0;
   out_1145354702242097965[297] = 0.0;
   out_1145354702242097965[298] = 0.0;
   out_1145354702242097965[299] = 0.0;
   out_1145354702242097965[300] = 0.0;
   out_1145354702242097965[301] = 0.0;
   out_1145354702242097965[302] = 0.0;
   out_1145354702242097965[303] = 0.0;
   out_1145354702242097965[304] = 1.0;
   out_1145354702242097965[305] = 0.0;
   out_1145354702242097965[306] = 0.0;
   out_1145354702242097965[307] = 0.0;
   out_1145354702242097965[308] = 0.0;
   out_1145354702242097965[309] = 0.0;
   out_1145354702242097965[310] = 0.0;
   out_1145354702242097965[311] = 0.0;
   out_1145354702242097965[312] = 0.0;
   out_1145354702242097965[313] = 0.0;
   out_1145354702242097965[314] = 0.0;
   out_1145354702242097965[315] = 0.0;
   out_1145354702242097965[316] = 0.0;
   out_1145354702242097965[317] = 0.0;
   out_1145354702242097965[318] = 0.0;
   out_1145354702242097965[319] = 0.0;
   out_1145354702242097965[320] = 0.0;
   out_1145354702242097965[321] = 0.0;
   out_1145354702242097965[322] = 0.0;
   out_1145354702242097965[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7691519941823953160) {
   out_7691519941823953160[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7691519941823953160[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7691519941823953160[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7691519941823953160[3] = dt*state[12] + state[3];
   out_7691519941823953160[4] = dt*state[13] + state[4];
   out_7691519941823953160[5] = dt*state[14] + state[5];
   out_7691519941823953160[6] = state[6];
   out_7691519941823953160[7] = state[7];
   out_7691519941823953160[8] = state[8];
   out_7691519941823953160[9] = state[9];
   out_7691519941823953160[10] = state[10];
   out_7691519941823953160[11] = state[11];
   out_7691519941823953160[12] = state[12];
   out_7691519941823953160[13] = state[13];
   out_7691519941823953160[14] = state[14];
   out_7691519941823953160[15] = state[15];
   out_7691519941823953160[16] = state[16];
   out_7691519941823953160[17] = state[17];
}
void F_fun(double *state, double dt, double *out_1204100135177654472) {
   out_1204100135177654472[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1204100135177654472[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1204100135177654472[2] = 0;
   out_1204100135177654472[3] = 0;
   out_1204100135177654472[4] = 0;
   out_1204100135177654472[5] = 0;
   out_1204100135177654472[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1204100135177654472[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1204100135177654472[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1204100135177654472[9] = 0;
   out_1204100135177654472[10] = 0;
   out_1204100135177654472[11] = 0;
   out_1204100135177654472[12] = 0;
   out_1204100135177654472[13] = 0;
   out_1204100135177654472[14] = 0;
   out_1204100135177654472[15] = 0;
   out_1204100135177654472[16] = 0;
   out_1204100135177654472[17] = 0;
   out_1204100135177654472[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1204100135177654472[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1204100135177654472[20] = 0;
   out_1204100135177654472[21] = 0;
   out_1204100135177654472[22] = 0;
   out_1204100135177654472[23] = 0;
   out_1204100135177654472[24] = 0;
   out_1204100135177654472[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1204100135177654472[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1204100135177654472[27] = 0;
   out_1204100135177654472[28] = 0;
   out_1204100135177654472[29] = 0;
   out_1204100135177654472[30] = 0;
   out_1204100135177654472[31] = 0;
   out_1204100135177654472[32] = 0;
   out_1204100135177654472[33] = 0;
   out_1204100135177654472[34] = 0;
   out_1204100135177654472[35] = 0;
   out_1204100135177654472[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1204100135177654472[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1204100135177654472[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1204100135177654472[39] = 0;
   out_1204100135177654472[40] = 0;
   out_1204100135177654472[41] = 0;
   out_1204100135177654472[42] = 0;
   out_1204100135177654472[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1204100135177654472[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1204100135177654472[45] = 0;
   out_1204100135177654472[46] = 0;
   out_1204100135177654472[47] = 0;
   out_1204100135177654472[48] = 0;
   out_1204100135177654472[49] = 0;
   out_1204100135177654472[50] = 0;
   out_1204100135177654472[51] = 0;
   out_1204100135177654472[52] = 0;
   out_1204100135177654472[53] = 0;
   out_1204100135177654472[54] = 0;
   out_1204100135177654472[55] = 0;
   out_1204100135177654472[56] = 0;
   out_1204100135177654472[57] = 1;
   out_1204100135177654472[58] = 0;
   out_1204100135177654472[59] = 0;
   out_1204100135177654472[60] = 0;
   out_1204100135177654472[61] = 0;
   out_1204100135177654472[62] = 0;
   out_1204100135177654472[63] = 0;
   out_1204100135177654472[64] = 0;
   out_1204100135177654472[65] = 0;
   out_1204100135177654472[66] = dt;
   out_1204100135177654472[67] = 0;
   out_1204100135177654472[68] = 0;
   out_1204100135177654472[69] = 0;
   out_1204100135177654472[70] = 0;
   out_1204100135177654472[71] = 0;
   out_1204100135177654472[72] = 0;
   out_1204100135177654472[73] = 0;
   out_1204100135177654472[74] = 0;
   out_1204100135177654472[75] = 0;
   out_1204100135177654472[76] = 1;
   out_1204100135177654472[77] = 0;
   out_1204100135177654472[78] = 0;
   out_1204100135177654472[79] = 0;
   out_1204100135177654472[80] = 0;
   out_1204100135177654472[81] = 0;
   out_1204100135177654472[82] = 0;
   out_1204100135177654472[83] = 0;
   out_1204100135177654472[84] = 0;
   out_1204100135177654472[85] = dt;
   out_1204100135177654472[86] = 0;
   out_1204100135177654472[87] = 0;
   out_1204100135177654472[88] = 0;
   out_1204100135177654472[89] = 0;
   out_1204100135177654472[90] = 0;
   out_1204100135177654472[91] = 0;
   out_1204100135177654472[92] = 0;
   out_1204100135177654472[93] = 0;
   out_1204100135177654472[94] = 0;
   out_1204100135177654472[95] = 1;
   out_1204100135177654472[96] = 0;
   out_1204100135177654472[97] = 0;
   out_1204100135177654472[98] = 0;
   out_1204100135177654472[99] = 0;
   out_1204100135177654472[100] = 0;
   out_1204100135177654472[101] = 0;
   out_1204100135177654472[102] = 0;
   out_1204100135177654472[103] = 0;
   out_1204100135177654472[104] = dt;
   out_1204100135177654472[105] = 0;
   out_1204100135177654472[106] = 0;
   out_1204100135177654472[107] = 0;
   out_1204100135177654472[108] = 0;
   out_1204100135177654472[109] = 0;
   out_1204100135177654472[110] = 0;
   out_1204100135177654472[111] = 0;
   out_1204100135177654472[112] = 0;
   out_1204100135177654472[113] = 0;
   out_1204100135177654472[114] = 1;
   out_1204100135177654472[115] = 0;
   out_1204100135177654472[116] = 0;
   out_1204100135177654472[117] = 0;
   out_1204100135177654472[118] = 0;
   out_1204100135177654472[119] = 0;
   out_1204100135177654472[120] = 0;
   out_1204100135177654472[121] = 0;
   out_1204100135177654472[122] = 0;
   out_1204100135177654472[123] = 0;
   out_1204100135177654472[124] = 0;
   out_1204100135177654472[125] = 0;
   out_1204100135177654472[126] = 0;
   out_1204100135177654472[127] = 0;
   out_1204100135177654472[128] = 0;
   out_1204100135177654472[129] = 0;
   out_1204100135177654472[130] = 0;
   out_1204100135177654472[131] = 0;
   out_1204100135177654472[132] = 0;
   out_1204100135177654472[133] = 1;
   out_1204100135177654472[134] = 0;
   out_1204100135177654472[135] = 0;
   out_1204100135177654472[136] = 0;
   out_1204100135177654472[137] = 0;
   out_1204100135177654472[138] = 0;
   out_1204100135177654472[139] = 0;
   out_1204100135177654472[140] = 0;
   out_1204100135177654472[141] = 0;
   out_1204100135177654472[142] = 0;
   out_1204100135177654472[143] = 0;
   out_1204100135177654472[144] = 0;
   out_1204100135177654472[145] = 0;
   out_1204100135177654472[146] = 0;
   out_1204100135177654472[147] = 0;
   out_1204100135177654472[148] = 0;
   out_1204100135177654472[149] = 0;
   out_1204100135177654472[150] = 0;
   out_1204100135177654472[151] = 0;
   out_1204100135177654472[152] = 1;
   out_1204100135177654472[153] = 0;
   out_1204100135177654472[154] = 0;
   out_1204100135177654472[155] = 0;
   out_1204100135177654472[156] = 0;
   out_1204100135177654472[157] = 0;
   out_1204100135177654472[158] = 0;
   out_1204100135177654472[159] = 0;
   out_1204100135177654472[160] = 0;
   out_1204100135177654472[161] = 0;
   out_1204100135177654472[162] = 0;
   out_1204100135177654472[163] = 0;
   out_1204100135177654472[164] = 0;
   out_1204100135177654472[165] = 0;
   out_1204100135177654472[166] = 0;
   out_1204100135177654472[167] = 0;
   out_1204100135177654472[168] = 0;
   out_1204100135177654472[169] = 0;
   out_1204100135177654472[170] = 0;
   out_1204100135177654472[171] = 1;
   out_1204100135177654472[172] = 0;
   out_1204100135177654472[173] = 0;
   out_1204100135177654472[174] = 0;
   out_1204100135177654472[175] = 0;
   out_1204100135177654472[176] = 0;
   out_1204100135177654472[177] = 0;
   out_1204100135177654472[178] = 0;
   out_1204100135177654472[179] = 0;
   out_1204100135177654472[180] = 0;
   out_1204100135177654472[181] = 0;
   out_1204100135177654472[182] = 0;
   out_1204100135177654472[183] = 0;
   out_1204100135177654472[184] = 0;
   out_1204100135177654472[185] = 0;
   out_1204100135177654472[186] = 0;
   out_1204100135177654472[187] = 0;
   out_1204100135177654472[188] = 0;
   out_1204100135177654472[189] = 0;
   out_1204100135177654472[190] = 1;
   out_1204100135177654472[191] = 0;
   out_1204100135177654472[192] = 0;
   out_1204100135177654472[193] = 0;
   out_1204100135177654472[194] = 0;
   out_1204100135177654472[195] = 0;
   out_1204100135177654472[196] = 0;
   out_1204100135177654472[197] = 0;
   out_1204100135177654472[198] = 0;
   out_1204100135177654472[199] = 0;
   out_1204100135177654472[200] = 0;
   out_1204100135177654472[201] = 0;
   out_1204100135177654472[202] = 0;
   out_1204100135177654472[203] = 0;
   out_1204100135177654472[204] = 0;
   out_1204100135177654472[205] = 0;
   out_1204100135177654472[206] = 0;
   out_1204100135177654472[207] = 0;
   out_1204100135177654472[208] = 0;
   out_1204100135177654472[209] = 1;
   out_1204100135177654472[210] = 0;
   out_1204100135177654472[211] = 0;
   out_1204100135177654472[212] = 0;
   out_1204100135177654472[213] = 0;
   out_1204100135177654472[214] = 0;
   out_1204100135177654472[215] = 0;
   out_1204100135177654472[216] = 0;
   out_1204100135177654472[217] = 0;
   out_1204100135177654472[218] = 0;
   out_1204100135177654472[219] = 0;
   out_1204100135177654472[220] = 0;
   out_1204100135177654472[221] = 0;
   out_1204100135177654472[222] = 0;
   out_1204100135177654472[223] = 0;
   out_1204100135177654472[224] = 0;
   out_1204100135177654472[225] = 0;
   out_1204100135177654472[226] = 0;
   out_1204100135177654472[227] = 0;
   out_1204100135177654472[228] = 1;
   out_1204100135177654472[229] = 0;
   out_1204100135177654472[230] = 0;
   out_1204100135177654472[231] = 0;
   out_1204100135177654472[232] = 0;
   out_1204100135177654472[233] = 0;
   out_1204100135177654472[234] = 0;
   out_1204100135177654472[235] = 0;
   out_1204100135177654472[236] = 0;
   out_1204100135177654472[237] = 0;
   out_1204100135177654472[238] = 0;
   out_1204100135177654472[239] = 0;
   out_1204100135177654472[240] = 0;
   out_1204100135177654472[241] = 0;
   out_1204100135177654472[242] = 0;
   out_1204100135177654472[243] = 0;
   out_1204100135177654472[244] = 0;
   out_1204100135177654472[245] = 0;
   out_1204100135177654472[246] = 0;
   out_1204100135177654472[247] = 1;
   out_1204100135177654472[248] = 0;
   out_1204100135177654472[249] = 0;
   out_1204100135177654472[250] = 0;
   out_1204100135177654472[251] = 0;
   out_1204100135177654472[252] = 0;
   out_1204100135177654472[253] = 0;
   out_1204100135177654472[254] = 0;
   out_1204100135177654472[255] = 0;
   out_1204100135177654472[256] = 0;
   out_1204100135177654472[257] = 0;
   out_1204100135177654472[258] = 0;
   out_1204100135177654472[259] = 0;
   out_1204100135177654472[260] = 0;
   out_1204100135177654472[261] = 0;
   out_1204100135177654472[262] = 0;
   out_1204100135177654472[263] = 0;
   out_1204100135177654472[264] = 0;
   out_1204100135177654472[265] = 0;
   out_1204100135177654472[266] = 1;
   out_1204100135177654472[267] = 0;
   out_1204100135177654472[268] = 0;
   out_1204100135177654472[269] = 0;
   out_1204100135177654472[270] = 0;
   out_1204100135177654472[271] = 0;
   out_1204100135177654472[272] = 0;
   out_1204100135177654472[273] = 0;
   out_1204100135177654472[274] = 0;
   out_1204100135177654472[275] = 0;
   out_1204100135177654472[276] = 0;
   out_1204100135177654472[277] = 0;
   out_1204100135177654472[278] = 0;
   out_1204100135177654472[279] = 0;
   out_1204100135177654472[280] = 0;
   out_1204100135177654472[281] = 0;
   out_1204100135177654472[282] = 0;
   out_1204100135177654472[283] = 0;
   out_1204100135177654472[284] = 0;
   out_1204100135177654472[285] = 1;
   out_1204100135177654472[286] = 0;
   out_1204100135177654472[287] = 0;
   out_1204100135177654472[288] = 0;
   out_1204100135177654472[289] = 0;
   out_1204100135177654472[290] = 0;
   out_1204100135177654472[291] = 0;
   out_1204100135177654472[292] = 0;
   out_1204100135177654472[293] = 0;
   out_1204100135177654472[294] = 0;
   out_1204100135177654472[295] = 0;
   out_1204100135177654472[296] = 0;
   out_1204100135177654472[297] = 0;
   out_1204100135177654472[298] = 0;
   out_1204100135177654472[299] = 0;
   out_1204100135177654472[300] = 0;
   out_1204100135177654472[301] = 0;
   out_1204100135177654472[302] = 0;
   out_1204100135177654472[303] = 0;
   out_1204100135177654472[304] = 1;
   out_1204100135177654472[305] = 0;
   out_1204100135177654472[306] = 0;
   out_1204100135177654472[307] = 0;
   out_1204100135177654472[308] = 0;
   out_1204100135177654472[309] = 0;
   out_1204100135177654472[310] = 0;
   out_1204100135177654472[311] = 0;
   out_1204100135177654472[312] = 0;
   out_1204100135177654472[313] = 0;
   out_1204100135177654472[314] = 0;
   out_1204100135177654472[315] = 0;
   out_1204100135177654472[316] = 0;
   out_1204100135177654472[317] = 0;
   out_1204100135177654472[318] = 0;
   out_1204100135177654472[319] = 0;
   out_1204100135177654472[320] = 0;
   out_1204100135177654472[321] = 0;
   out_1204100135177654472[322] = 0;
   out_1204100135177654472[323] = 1;
}
void h_4(double *state, double *unused, double *out_7251927051155261244) {
   out_7251927051155261244[0] = state[6] + state[9];
   out_7251927051155261244[1] = state[7] + state[10];
   out_7251927051155261244[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7799006963815723377) {
   out_7799006963815723377[0] = 0;
   out_7799006963815723377[1] = 0;
   out_7799006963815723377[2] = 0;
   out_7799006963815723377[3] = 0;
   out_7799006963815723377[4] = 0;
   out_7799006963815723377[5] = 0;
   out_7799006963815723377[6] = 1;
   out_7799006963815723377[7] = 0;
   out_7799006963815723377[8] = 0;
   out_7799006963815723377[9] = 1;
   out_7799006963815723377[10] = 0;
   out_7799006963815723377[11] = 0;
   out_7799006963815723377[12] = 0;
   out_7799006963815723377[13] = 0;
   out_7799006963815723377[14] = 0;
   out_7799006963815723377[15] = 0;
   out_7799006963815723377[16] = 0;
   out_7799006963815723377[17] = 0;
   out_7799006963815723377[18] = 0;
   out_7799006963815723377[19] = 0;
   out_7799006963815723377[20] = 0;
   out_7799006963815723377[21] = 0;
   out_7799006963815723377[22] = 0;
   out_7799006963815723377[23] = 0;
   out_7799006963815723377[24] = 0;
   out_7799006963815723377[25] = 1;
   out_7799006963815723377[26] = 0;
   out_7799006963815723377[27] = 0;
   out_7799006963815723377[28] = 1;
   out_7799006963815723377[29] = 0;
   out_7799006963815723377[30] = 0;
   out_7799006963815723377[31] = 0;
   out_7799006963815723377[32] = 0;
   out_7799006963815723377[33] = 0;
   out_7799006963815723377[34] = 0;
   out_7799006963815723377[35] = 0;
   out_7799006963815723377[36] = 0;
   out_7799006963815723377[37] = 0;
   out_7799006963815723377[38] = 0;
   out_7799006963815723377[39] = 0;
   out_7799006963815723377[40] = 0;
   out_7799006963815723377[41] = 0;
   out_7799006963815723377[42] = 0;
   out_7799006963815723377[43] = 0;
   out_7799006963815723377[44] = 1;
   out_7799006963815723377[45] = 0;
   out_7799006963815723377[46] = 0;
   out_7799006963815723377[47] = 1;
   out_7799006963815723377[48] = 0;
   out_7799006963815723377[49] = 0;
   out_7799006963815723377[50] = 0;
   out_7799006963815723377[51] = 0;
   out_7799006963815723377[52] = 0;
   out_7799006963815723377[53] = 0;
}
void h_10(double *state, double *unused, double *out_3161621592790876862) {
   out_3161621592790876862[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_3161621592790876862[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_3161621592790876862[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_8738726503938101473) {
   out_8738726503938101473[0] = 0;
   out_8738726503938101473[1] = 9.8100000000000005*cos(state[1]);
   out_8738726503938101473[2] = 0;
   out_8738726503938101473[3] = 0;
   out_8738726503938101473[4] = -state[8];
   out_8738726503938101473[5] = state[7];
   out_8738726503938101473[6] = 0;
   out_8738726503938101473[7] = state[5];
   out_8738726503938101473[8] = -state[4];
   out_8738726503938101473[9] = 0;
   out_8738726503938101473[10] = 0;
   out_8738726503938101473[11] = 0;
   out_8738726503938101473[12] = 1;
   out_8738726503938101473[13] = 0;
   out_8738726503938101473[14] = 0;
   out_8738726503938101473[15] = 1;
   out_8738726503938101473[16] = 0;
   out_8738726503938101473[17] = 0;
   out_8738726503938101473[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_8738726503938101473[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_8738726503938101473[20] = 0;
   out_8738726503938101473[21] = state[8];
   out_8738726503938101473[22] = 0;
   out_8738726503938101473[23] = -state[6];
   out_8738726503938101473[24] = -state[5];
   out_8738726503938101473[25] = 0;
   out_8738726503938101473[26] = state[3];
   out_8738726503938101473[27] = 0;
   out_8738726503938101473[28] = 0;
   out_8738726503938101473[29] = 0;
   out_8738726503938101473[30] = 0;
   out_8738726503938101473[31] = 1;
   out_8738726503938101473[32] = 0;
   out_8738726503938101473[33] = 0;
   out_8738726503938101473[34] = 1;
   out_8738726503938101473[35] = 0;
   out_8738726503938101473[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_8738726503938101473[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_8738726503938101473[38] = 0;
   out_8738726503938101473[39] = -state[7];
   out_8738726503938101473[40] = state[6];
   out_8738726503938101473[41] = 0;
   out_8738726503938101473[42] = state[4];
   out_8738726503938101473[43] = -state[3];
   out_8738726503938101473[44] = 0;
   out_8738726503938101473[45] = 0;
   out_8738726503938101473[46] = 0;
   out_8738726503938101473[47] = 0;
   out_8738726503938101473[48] = 0;
   out_8738726503938101473[49] = 0;
   out_8738726503938101473[50] = 1;
   out_8738726503938101473[51] = 0;
   out_8738726503938101473[52] = 0;
   out_8738726503938101473[53] = 1;
}
void h_13(double *state, double *unused, double *out_5099159631779018063) {
   out_5099159631779018063[0] = state[3];
   out_5099159631779018063[1] = state[4];
   out_5099159631779018063[2] = state[5];
}
void H_13(double *state, double *unused, double *out_4586733138483390576) {
   out_4586733138483390576[0] = 0;
   out_4586733138483390576[1] = 0;
   out_4586733138483390576[2] = 0;
   out_4586733138483390576[3] = 1;
   out_4586733138483390576[4] = 0;
   out_4586733138483390576[5] = 0;
   out_4586733138483390576[6] = 0;
   out_4586733138483390576[7] = 0;
   out_4586733138483390576[8] = 0;
   out_4586733138483390576[9] = 0;
   out_4586733138483390576[10] = 0;
   out_4586733138483390576[11] = 0;
   out_4586733138483390576[12] = 0;
   out_4586733138483390576[13] = 0;
   out_4586733138483390576[14] = 0;
   out_4586733138483390576[15] = 0;
   out_4586733138483390576[16] = 0;
   out_4586733138483390576[17] = 0;
   out_4586733138483390576[18] = 0;
   out_4586733138483390576[19] = 0;
   out_4586733138483390576[20] = 0;
   out_4586733138483390576[21] = 0;
   out_4586733138483390576[22] = 1;
   out_4586733138483390576[23] = 0;
   out_4586733138483390576[24] = 0;
   out_4586733138483390576[25] = 0;
   out_4586733138483390576[26] = 0;
   out_4586733138483390576[27] = 0;
   out_4586733138483390576[28] = 0;
   out_4586733138483390576[29] = 0;
   out_4586733138483390576[30] = 0;
   out_4586733138483390576[31] = 0;
   out_4586733138483390576[32] = 0;
   out_4586733138483390576[33] = 0;
   out_4586733138483390576[34] = 0;
   out_4586733138483390576[35] = 0;
   out_4586733138483390576[36] = 0;
   out_4586733138483390576[37] = 0;
   out_4586733138483390576[38] = 0;
   out_4586733138483390576[39] = 0;
   out_4586733138483390576[40] = 0;
   out_4586733138483390576[41] = 1;
   out_4586733138483390576[42] = 0;
   out_4586733138483390576[43] = 0;
   out_4586733138483390576[44] = 0;
   out_4586733138483390576[45] = 0;
   out_4586733138483390576[46] = 0;
   out_4586733138483390576[47] = 0;
   out_4586733138483390576[48] = 0;
   out_4586733138483390576[49] = 0;
   out_4586733138483390576[50] = 0;
   out_4586733138483390576[51] = 0;
   out_4586733138483390576[52] = 0;
   out_4586733138483390576[53] = 0;
}
void h_14(double *state, double *unused, double *out_204192348592699604) {
   out_204192348592699604[0] = state[6];
   out_204192348592699604[1] = state[7];
   out_204192348592699604[2] = state[8];
}
void H_14(double *state, double *unused, double *out_3835766107476238848) {
   out_3835766107476238848[0] = 0;
   out_3835766107476238848[1] = 0;
   out_3835766107476238848[2] = 0;
   out_3835766107476238848[3] = 0;
   out_3835766107476238848[4] = 0;
   out_3835766107476238848[5] = 0;
   out_3835766107476238848[6] = 1;
   out_3835766107476238848[7] = 0;
   out_3835766107476238848[8] = 0;
   out_3835766107476238848[9] = 0;
   out_3835766107476238848[10] = 0;
   out_3835766107476238848[11] = 0;
   out_3835766107476238848[12] = 0;
   out_3835766107476238848[13] = 0;
   out_3835766107476238848[14] = 0;
   out_3835766107476238848[15] = 0;
   out_3835766107476238848[16] = 0;
   out_3835766107476238848[17] = 0;
   out_3835766107476238848[18] = 0;
   out_3835766107476238848[19] = 0;
   out_3835766107476238848[20] = 0;
   out_3835766107476238848[21] = 0;
   out_3835766107476238848[22] = 0;
   out_3835766107476238848[23] = 0;
   out_3835766107476238848[24] = 0;
   out_3835766107476238848[25] = 1;
   out_3835766107476238848[26] = 0;
   out_3835766107476238848[27] = 0;
   out_3835766107476238848[28] = 0;
   out_3835766107476238848[29] = 0;
   out_3835766107476238848[30] = 0;
   out_3835766107476238848[31] = 0;
   out_3835766107476238848[32] = 0;
   out_3835766107476238848[33] = 0;
   out_3835766107476238848[34] = 0;
   out_3835766107476238848[35] = 0;
   out_3835766107476238848[36] = 0;
   out_3835766107476238848[37] = 0;
   out_3835766107476238848[38] = 0;
   out_3835766107476238848[39] = 0;
   out_3835766107476238848[40] = 0;
   out_3835766107476238848[41] = 0;
   out_3835766107476238848[42] = 0;
   out_3835766107476238848[43] = 0;
   out_3835766107476238848[44] = 1;
   out_3835766107476238848[45] = 0;
   out_3835766107476238848[46] = 0;
   out_3835766107476238848[47] = 0;
   out_3835766107476238848[48] = 0;
   out_3835766107476238848[49] = 0;
   out_3835766107476238848[50] = 0;
   out_3835766107476238848[51] = 0;
   out_3835766107476238848[52] = 0;
   out_3835766107476238848[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_7210081177362107870) {
  err_fun(nom_x, delta_x, out_7210081177362107870);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8354364799378035034) {
  inv_err_fun(nom_x, true_x, out_8354364799378035034);
}
void pose_H_mod_fun(double *state, double *out_1145354702242097965) {
  H_mod_fun(state, out_1145354702242097965);
}
void pose_f_fun(double *state, double dt, double *out_7691519941823953160) {
  f_fun(state,  dt, out_7691519941823953160);
}
void pose_F_fun(double *state, double dt, double *out_1204100135177654472) {
  F_fun(state,  dt, out_1204100135177654472);
}
void pose_h_4(double *state, double *unused, double *out_7251927051155261244) {
  h_4(state, unused, out_7251927051155261244);
}
void pose_H_4(double *state, double *unused, double *out_7799006963815723377) {
  H_4(state, unused, out_7799006963815723377);
}
void pose_h_10(double *state, double *unused, double *out_3161621592790876862) {
  h_10(state, unused, out_3161621592790876862);
}
void pose_H_10(double *state, double *unused, double *out_8738726503938101473) {
  H_10(state, unused, out_8738726503938101473);
}
void pose_h_13(double *state, double *unused, double *out_5099159631779018063) {
  h_13(state, unused, out_5099159631779018063);
}
void pose_H_13(double *state, double *unused, double *out_4586733138483390576) {
  H_13(state, unused, out_4586733138483390576);
}
void pose_h_14(double *state, double *unused, double *out_204192348592699604) {
  h_14(state, unused, out_204192348592699604);
}
void pose_H_14(double *state, double *unused, double *out_3835766107476238848) {
  H_14(state, unused, out_3835766107476238848);
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
