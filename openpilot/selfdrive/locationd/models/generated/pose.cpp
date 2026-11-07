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
void err_fun(double *nom_x, double *delta_x, double *out_3110422523440551038) {
   out_3110422523440551038[0] = delta_x[0] + nom_x[0];
   out_3110422523440551038[1] = delta_x[1] + nom_x[1];
   out_3110422523440551038[2] = delta_x[2] + nom_x[2];
   out_3110422523440551038[3] = delta_x[3] + nom_x[3];
   out_3110422523440551038[4] = delta_x[4] + nom_x[4];
   out_3110422523440551038[5] = delta_x[5] + nom_x[5];
   out_3110422523440551038[6] = delta_x[6] + nom_x[6];
   out_3110422523440551038[7] = delta_x[7] + nom_x[7];
   out_3110422523440551038[8] = delta_x[8] + nom_x[8];
   out_3110422523440551038[9] = delta_x[9] + nom_x[9];
   out_3110422523440551038[10] = delta_x[10] + nom_x[10];
   out_3110422523440551038[11] = delta_x[11] + nom_x[11];
   out_3110422523440551038[12] = delta_x[12] + nom_x[12];
   out_3110422523440551038[13] = delta_x[13] + nom_x[13];
   out_3110422523440551038[14] = delta_x[14] + nom_x[14];
   out_3110422523440551038[15] = delta_x[15] + nom_x[15];
   out_3110422523440551038[16] = delta_x[16] + nom_x[16];
   out_3110422523440551038[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2741580585373072420) {
   out_2741580585373072420[0] = -nom_x[0] + true_x[0];
   out_2741580585373072420[1] = -nom_x[1] + true_x[1];
   out_2741580585373072420[2] = -nom_x[2] + true_x[2];
   out_2741580585373072420[3] = -nom_x[3] + true_x[3];
   out_2741580585373072420[4] = -nom_x[4] + true_x[4];
   out_2741580585373072420[5] = -nom_x[5] + true_x[5];
   out_2741580585373072420[6] = -nom_x[6] + true_x[6];
   out_2741580585373072420[7] = -nom_x[7] + true_x[7];
   out_2741580585373072420[8] = -nom_x[8] + true_x[8];
   out_2741580585373072420[9] = -nom_x[9] + true_x[9];
   out_2741580585373072420[10] = -nom_x[10] + true_x[10];
   out_2741580585373072420[11] = -nom_x[11] + true_x[11];
   out_2741580585373072420[12] = -nom_x[12] + true_x[12];
   out_2741580585373072420[13] = -nom_x[13] + true_x[13];
   out_2741580585373072420[14] = -nom_x[14] + true_x[14];
   out_2741580585373072420[15] = -nom_x[15] + true_x[15];
   out_2741580585373072420[16] = -nom_x[16] + true_x[16];
   out_2741580585373072420[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3055547229087302731) {
   out_3055547229087302731[0] = 1.0;
   out_3055547229087302731[1] = 0.0;
   out_3055547229087302731[2] = 0.0;
   out_3055547229087302731[3] = 0.0;
   out_3055547229087302731[4] = 0.0;
   out_3055547229087302731[5] = 0.0;
   out_3055547229087302731[6] = 0.0;
   out_3055547229087302731[7] = 0.0;
   out_3055547229087302731[8] = 0.0;
   out_3055547229087302731[9] = 0.0;
   out_3055547229087302731[10] = 0.0;
   out_3055547229087302731[11] = 0.0;
   out_3055547229087302731[12] = 0.0;
   out_3055547229087302731[13] = 0.0;
   out_3055547229087302731[14] = 0.0;
   out_3055547229087302731[15] = 0.0;
   out_3055547229087302731[16] = 0.0;
   out_3055547229087302731[17] = 0.0;
   out_3055547229087302731[18] = 0.0;
   out_3055547229087302731[19] = 1.0;
   out_3055547229087302731[20] = 0.0;
   out_3055547229087302731[21] = 0.0;
   out_3055547229087302731[22] = 0.0;
   out_3055547229087302731[23] = 0.0;
   out_3055547229087302731[24] = 0.0;
   out_3055547229087302731[25] = 0.0;
   out_3055547229087302731[26] = 0.0;
   out_3055547229087302731[27] = 0.0;
   out_3055547229087302731[28] = 0.0;
   out_3055547229087302731[29] = 0.0;
   out_3055547229087302731[30] = 0.0;
   out_3055547229087302731[31] = 0.0;
   out_3055547229087302731[32] = 0.0;
   out_3055547229087302731[33] = 0.0;
   out_3055547229087302731[34] = 0.0;
   out_3055547229087302731[35] = 0.0;
   out_3055547229087302731[36] = 0.0;
   out_3055547229087302731[37] = 0.0;
   out_3055547229087302731[38] = 1.0;
   out_3055547229087302731[39] = 0.0;
   out_3055547229087302731[40] = 0.0;
   out_3055547229087302731[41] = 0.0;
   out_3055547229087302731[42] = 0.0;
   out_3055547229087302731[43] = 0.0;
   out_3055547229087302731[44] = 0.0;
   out_3055547229087302731[45] = 0.0;
   out_3055547229087302731[46] = 0.0;
   out_3055547229087302731[47] = 0.0;
   out_3055547229087302731[48] = 0.0;
   out_3055547229087302731[49] = 0.0;
   out_3055547229087302731[50] = 0.0;
   out_3055547229087302731[51] = 0.0;
   out_3055547229087302731[52] = 0.0;
   out_3055547229087302731[53] = 0.0;
   out_3055547229087302731[54] = 0.0;
   out_3055547229087302731[55] = 0.0;
   out_3055547229087302731[56] = 0.0;
   out_3055547229087302731[57] = 1.0;
   out_3055547229087302731[58] = 0.0;
   out_3055547229087302731[59] = 0.0;
   out_3055547229087302731[60] = 0.0;
   out_3055547229087302731[61] = 0.0;
   out_3055547229087302731[62] = 0.0;
   out_3055547229087302731[63] = 0.0;
   out_3055547229087302731[64] = 0.0;
   out_3055547229087302731[65] = 0.0;
   out_3055547229087302731[66] = 0.0;
   out_3055547229087302731[67] = 0.0;
   out_3055547229087302731[68] = 0.0;
   out_3055547229087302731[69] = 0.0;
   out_3055547229087302731[70] = 0.0;
   out_3055547229087302731[71] = 0.0;
   out_3055547229087302731[72] = 0.0;
   out_3055547229087302731[73] = 0.0;
   out_3055547229087302731[74] = 0.0;
   out_3055547229087302731[75] = 0.0;
   out_3055547229087302731[76] = 1.0;
   out_3055547229087302731[77] = 0.0;
   out_3055547229087302731[78] = 0.0;
   out_3055547229087302731[79] = 0.0;
   out_3055547229087302731[80] = 0.0;
   out_3055547229087302731[81] = 0.0;
   out_3055547229087302731[82] = 0.0;
   out_3055547229087302731[83] = 0.0;
   out_3055547229087302731[84] = 0.0;
   out_3055547229087302731[85] = 0.0;
   out_3055547229087302731[86] = 0.0;
   out_3055547229087302731[87] = 0.0;
   out_3055547229087302731[88] = 0.0;
   out_3055547229087302731[89] = 0.0;
   out_3055547229087302731[90] = 0.0;
   out_3055547229087302731[91] = 0.0;
   out_3055547229087302731[92] = 0.0;
   out_3055547229087302731[93] = 0.0;
   out_3055547229087302731[94] = 0.0;
   out_3055547229087302731[95] = 1.0;
   out_3055547229087302731[96] = 0.0;
   out_3055547229087302731[97] = 0.0;
   out_3055547229087302731[98] = 0.0;
   out_3055547229087302731[99] = 0.0;
   out_3055547229087302731[100] = 0.0;
   out_3055547229087302731[101] = 0.0;
   out_3055547229087302731[102] = 0.0;
   out_3055547229087302731[103] = 0.0;
   out_3055547229087302731[104] = 0.0;
   out_3055547229087302731[105] = 0.0;
   out_3055547229087302731[106] = 0.0;
   out_3055547229087302731[107] = 0.0;
   out_3055547229087302731[108] = 0.0;
   out_3055547229087302731[109] = 0.0;
   out_3055547229087302731[110] = 0.0;
   out_3055547229087302731[111] = 0.0;
   out_3055547229087302731[112] = 0.0;
   out_3055547229087302731[113] = 0.0;
   out_3055547229087302731[114] = 1.0;
   out_3055547229087302731[115] = 0.0;
   out_3055547229087302731[116] = 0.0;
   out_3055547229087302731[117] = 0.0;
   out_3055547229087302731[118] = 0.0;
   out_3055547229087302731[119] = 0.0;
   out_3055547229087302731[120] = 0.0;
   out_3055547229087302731[121] = 0.0;
   out_3055547229087302731[122] = 0.0;
   out_3055547229087302731[123] = 0.0;
   out_3055547229087302731[124] = 0.0;
   out_3055547229087302731[125] = 0.0;
   out_3055547229087302731[126] = 0.0;
   out_3055547229087302731[127] = 0.0;
   out_3055547229087302731[128] = 0.0;
   out_3055547229087302731[129] = 0.0;
   out_3055547229087302731[130] = 0.0;
   out_3055547229087302731[131] = 0.0;
   out_3055547229087302731[132] = 0.0;
   out_3055547229087302731[133] = 1.0;
   out_3055547229087302731[134] = 0.0;
   out_3055547229087302731[135] = 0.0;
   out_3055547229087302731[136] = 0.0;
   out_3055547229087302731[137] = 0.0;
   out_3055547229087302731[138] = 0.0;
   out_3055547229087302731[139] = 0.0;
   out_3055547229087302731[140] = 0.0;
   out_3055547229087302731[141] = 0.0;
   out_3055547229087302731[142] = 0.0;
   out_3055547229087302731[143] = 0.0;
   out_3055547229087302731[144] = 0.0;
   out_3055547229087302731[145] = 0.0;
   out_3055547229087302731[146] = 0.0;
   out_3055547229087302731[147] = 0.0;
   out_3055547229087302731[148] = 0.0;
   out_3055547229087302731[149] = 0.0;
   out_3055547229087302731[150] = 0.0;
   out_3055547229087302731[151] = 0.0;
   out_3055547229087302731[152] = 1.0;
   out_3055547229087302731[153] = 0.0;
   out_3055547229087302731[154] = 0.0;
   out_3055547229087302731[155] = 0.0;
   out_3055547229087302731[156] = 0.0;
   out_3055547229087302731[157] = 0.0;
   out_3055547229087302731[158] = 0.0;
   out_3055547229087302731[159] = 0.0;
   out_3055547229087302731[160] = 0.0;
   out_3055547229087302731[161] = 0.0;
   out_3055547229087302731[162] = 0.0;
   out_3055547229087302731[163] = 0.0;
   out_3055547229087302731[164] = 0.0;
   out_3055547229087302731[165] = 0.0;
   out_3055547229087302731[166] = 0.0;
   out_3055547229087302731[167] = 0.0;
   out_3055547229087302731[168] = 0.0;
   out_3055547229087302731[169] = 0.0;
   out_3055547229087302731[170] = 0.0;
   out_3055547229087302731[171] = 1.0;
   out_3055547229087302731[172] = 0.0;
   out_3055547229087302731[173] = 0.0;
   out_3055547229087302731[174] = 0.0;
   out_3055547229087302731[175] = 0.0;
   out_3055547229087302731[176] = 0.0;
   out_3055547229087302731[177] = 0.0;
   out_3055547229087302731[178] = 0.0;
   out_3055547229087302731[179] = 0.0;
   out_3055547229087302731[180] = 0.0;
   out_3055547229087302731[181] = 0.0;
   out_3055547229087302731[182] = 0.0;
   out_3055547229087302731[183] = 0.0;
   out_3055547229087302731[184] = 0.0;
   out_3055547229087302731[185] = 0.0;
   out_3055547229087302731[186] = 0.0;
   out_3055547229087302731[187] = 0.0;
   out_3055547229087302731[188] = 0.0;
   out_3055547229087302731[189] = 0.0;
   out_3055547229087302731[190] = 1.0;
   out_3055547229087302731[191] = 0.0;
   out_3055547229087302731[192] = 0.0;
   out_3055547229087302731[193] = 0.0;
   out_3055547229087302731[194] = 0.0;
   out_3055547229087302731[195] = 0.0;
   out_3055547229087302731[196] = 0.0;
   out_3055547229087302731[197] = 0.0;
   out_3055547229087302731[198] = 0.0;
   out_3055547229087302731[199] = 0.0;
   out_3055547229087302731[200] = 0.0;
   out_3055547229087302731[201] = 0.0;
   out_3055547229087302731[202] = 0.0;
   out_3055547229087302731[203] = 0.0;
   out_3055547229087302731[204] = 0.0;
   out_3055547229087302731[205] = 0.0;
   out_3055547229087302731[206] = 0.0;
   out_3055547229087302731[207] = 0.0;
   out_3055547229087302731[208] = 0.0;
   out_3055547229087302731[209] = 1.0;
   out_3055547229087302731[210] = 0.0;
   out_3055547229087302731[211] = 0.0;
   out_3055547229087302731[212] = 0.0;
   out_3055547229087302731[213] = 0.0;
   out_3055547229087302731[214] = 0.0;
   out_3055547229087302731[215] = 0.0;
   out_3055547229087302731[216] = 0.0;
   out_3055547229087302731[217] = 0.0;
   out_3055547229087302731[218] = 0.0;
   out_3055547229087302731[219] = 0.0;
   out_3055547229087302731[220] = 0.0;
   out_3055547229087302731[221] = 0.0;
   out_3055547229087302731[222] = 0.0;
   out_3055547229087302731[223] = 0.0;
   out_3055547229087302731[224] = 0.0;
   out_3055547229087302731[225] = 0.0;
   out_3055547229087302731[226] = 0.0;
   out_3055547229087302731[227] = 0.0;
   out_3055547229087302731[228] = 1.0;
   out_3055547229087302731[229] = 0.0;
   out_3055547229087302731[230] = 0.0;
   out_3055547229087302731[231] = 0.0;
   out_3055547229087302731[232] = 0.0;
   out_3055547229087302731[233] = 0.0;
   out_3055547229087302731[234] = 0.0;
   out_3055547229087302731[235] = 0.0;
   out_3055547229087302731[236] = 0.0;
   out_3055547229087302731[237] = 0.0;
   out_3055547229087302731[238] = 0.0;
   out_3055547229087302731[239] = 0.0;
   out_3055547229087302731[240] = 0.0;
   out_3055547229087302731[241] = 0.0;
   out_3055547229087302731[242] = 0.0;
   out_3055547229087302731[243] = 0.0;
   out_3055547229087302731[244] = 0.0;
   out_3055547229087302731[245] = 0.0;
   out_3055547229087302731[246] = 0.0;
   out_3055547229087302731[247] = 1.0;
   out_3055547229087302731[248] = 0.0;
   out_3055547229087302731[249] = 0.0;
   out_3055547229087302731[250] = 0.0;
   out_3055547229087302731[251] = 0.0;
   out_3055547229087302731[252] = 0.0;
   out_3055547229087302731[253] = 0.0;
   out_3055547229087302731[254] = 0.0;
   out_3055547229087302731[255] = 0.0;
   out_3055547229087302731[256] = 0.0;
   out_3055547229087302731[257] = 0.0;
   out_3055547229087302731[258] = 0.0;
   out_3055547229087302731[259] = 0.0;
   out_3055547229087302731[260] = 0.0;
   out_3055547229087302731[261] = 0.0;
   out_3055547229087302731[262] = 0.0;
   out_3055547229087302731[263] = 0.0;
   out_3055547229087302731[264] = 0.0;
   out_3055547229087302731[265] = 0.0;
   out_3055547229087302731[266] = 1.0;
   out_3055547229087302731[267] = 0.0;
   out_3055547229087302731[268] = 0.0;
   out_3055547229087302731[269] = 0.0;
   out_3055547229087302731[270] = 0.0;
   out_3055547229087302731[271] = 0.0;
   out_3055547229087302731[272] = 0.0;
   out_3055547229087302731[273] = 0.0;
   out_3055547229087302731[274] = 0.0;
   out_3055547229087302731[275] = 0.0;
   out_3055547229087302731[276] = 0.0;
   out_3055547229087302731[277] = 0.0;
   out_3055547229087302731[278] = 0.0;
   out_3055547229087302731[279] = 0.0;
   out_3055547229087302731[280] = 0.0;
   out_3055547229087302731[281] = 0.0;
   out_3055547229087302731[282] = 0.0;
   out_3055547229087302731[283] = 0.0;
   out_3055547229087302731[284] = 0.0;
   out_3055547229087302731[285] = 1.0;
   out_3055547229087302731[286] = 0.0;
   out_3055547229087302731[287] = 0.0;
   out_3055547229087302731[288] = 0.0;
   out_3055547229087302731[289] = 0.0;
   out_3055547229087302731[290] = 0.0;
   out_3055547229087302731[291] = 0.0;
   out_3055547229087302731[292] = 0.0;
   out_3055547229087302731[293] = 0.0;
   out_3055547229087302731[294] = 0.0;
   out_3055547229087302731[295] = 0.0;
   out_3055547229087302731[296] = 0.0;
   out_3055547229087302731[297] = 0.0;
   out_3055547229087302731[298] = 0.0;
   out_3055547229087302731[299] = 0.0;
   out_3055547229087302731[300] = 0.0;
   out_3055547229087302731[301] = 0.0;
   out_3055547229087302731[302] = 0.0;
   out_3055547229087302731[303] = 0.0;
   out_3055547229087302731[304] = 1.0;
   out_3055547229087302731[305] = 0.0;
   out_3055547229087302731[306] = 0.0;
   out_3055547229087302731[307] = 0.0;
   out_3055547229087302731[308] = 0.0;
   out_3055547229087302731[309] = 0.0;
   out_3055547229087302731[310] = 0.0;
   out_3055547229087302731[311] = 0.0;
   out_3055547229087302731[312] = 0.0;
   out_3055547229087302731[313] = 0.0;
   out_3055547229087302731[314] = 0.0;
   out_3055547229087302731[315] = 0.0;
   out_3055547229087302731[316] = 0.0;
   out_3055547229087302731[317] = 0.0;
   out_3055547229087302731[318] = 0.0;
   out_3055547229087302731[319] = 0.0;
   out_3055547229087302731[320] = 0.0;
   out_3055547229087302731[321] = 0.0;
   out_3055547229087302731[322] = 0.0;
   out_3055547229087302731[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_2068541479278321289) {
   out_2068541479278321289[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_2068541479278321289[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_2068541479278321289[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_2068541479278321289[3] = dt*state[12] + state[3];
   out_2068541479278321289[4] = dt*state[13] + state[4];
   out_2068541479278321289[5] = dt*state[14] + state[5];
   out_2068541479278321289[6] = state[6];
   out_2068541479278321289[7] = state[7];
   out_2068541479278321289[8] = state[8];
   out_2068541479278321289[9] = state[9];
   out_2068541479278321289[10] = state[10];
   out_2068541479278321289[11] = state[11];
   out_2068541479278321289[12] = state[12];
   out_2068541479278321289[13] = state[13];
   out_2068541479278321289[14] = state[14];
   out_2068541479278321289[15] = state[15];
   out_2068541479278321289[16] = state[16];
   out_2068541479278321289[17] = state[17];
}
void F_fun(double *state, double dt, double *out_1433385751187260512) {
   out_1433385751187260512[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1433385751187260512[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1433385751187260512[2] = 0;
   out_1433385751187260512[3] = 0;
   out_1433385751187260512[4] = 0;
   out_1433385751187260512[5] = 0;
   out_1433385751187260512[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1433385751187260512[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1433385751187260512[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1433385751187260512[9] = 0;
   out_1433385751187260512[10] = 0;
   out_1433385751187260512[11] = 0;
   out_1433385751187260512[12] = 0;
   out_1433385751187260512[13] = 0;
   out_1433385751187260512[14] = 0;
   out_1433385751187260512[15] = 0;
   out_1433385751187260512[16] = 0;
   out_1433385751187260512[17] = 0;
   out_1433385751187260512[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1433385751187260512[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1433385751187260512[20] = 0;
   out_1433385751187260512[21] = 0;
   out_1433385751187260512[22] = 0;
   out_1433385751187260512[23] = 0;
   out_1433385751187260512[24] = 0;
   out_1433385751187260512[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1433385751187260512[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1433385751187260512[27] = 0;
   out_1433385751187260512[28] = 0;
   out_1433385751187260512[29] = 0;
   out_1433385751187260512[30] = 0;
   out_1433385751187260512[31] = 0;
   out_1433385751187260512[32] = 0;
   out_1433385751187260512[33] = 0;
   out_1433385751187260512[34] = 0;
   out_1433385751187260512[35] = 0;
   out_1433385751187260512[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1433385751187260512[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1433385751187260512[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1433385751187260512[39] = 0;
   out_1433385751187260512[40] = 0;
   out_1433385751187260512[41] = 0;
   out_1433385751187260512[42] = 0;
   out_1433385751187260512[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1433385751187260512[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1433385751187260512[45] = 0;
   out_1433385751187260512[46] = 0;
   out_1433385751187260512[47] = 0;
   out_1433385751187260512[48] = 0;
   out_1433385751187260512[49] = 0;
   out_1433385751187260512[50] = 0;
   out_1433385751187260512[51] = 0;
   out_1433385751187260512[52] = 0;
   out_1433385751187260512[53] = 0;
   out_1433385751187260512[54] = 0;
   out_1433385751187260512[55] = 0;
   out_1433385751187260512[56] = 0;
   out_1433385751187260512[57] = 1;
   out_1433385751187260512[58] = 0;
   out_1433385751187260512[59] = 0;
   out_1433385751187260512[60] = 0;
   out_1433385751187260512[61] = 0;
   out_1433385751187260512[62] = 0;
   out_1433385751187260512[63] = 0;
   out_1433385751187260512[64] = 0;
   out_1433385751187260512[65] = 0;
   out_1433385751187260512[66] = dt;
   out_1433385751187260512[67] = 0;
   out_1433385751187260512[68] = 0;
   out_1433385751187260512[69] = 0;
   out_1433385751187260512[70] = 0;
   out_1433385751187260512[71] = 0;
   out_1433385751187260512[72] = 0;
   out_1433385751187260512[73] = 0;
   out_1433385751187260512[74] = 0;
   out_1433385751187260512[75] = 0;
   out_1433385751187260512[76] = 1;
   out_1433385751187260512[77] = 0;
   out_1433385751187260512[78] = 0;
   out_1433385751187260512[79] = 0;
   out_1433385751187260512[80] = 0;
   out_1433385751187260512[81] = 0;
   out_1433385751187260512[82] = 0;
   out_1433385751187260512[83] = 0;
   out_1433385751187260512[84] = 0;
   out_1433385751187260512[85] = dt;
   out_1433385751187260512[86] = 0;
   out_1433385751187260512[87] = 0;
   out_1433385751187260512[88] = 0;
   out_1433385751187260512[89] = 0;
   out_1433385751187260512[90] = 0;
   out_1433385751187260512[91] = 0;
   out_1433385751187260512[92] = 0;
   out_1433385751187260512[93] = 0;
   out_1433385751187260512[94] = 0;
   out_1433385751187260512[95] = 1;
   out_1433385751187260512[96] = 0;
   out_1433385751187260512[97] = 0;
   out_1433385751187260512[98] = 0;
   out_1433385751187260512[99] = 0;
   out_1433385751187260512[100] = 0;
   out_1433385751187260512[101] = 0;
   out_1433385751187260512[102] = 0;
   out_1433385751187260512[103] = 0;
   out_1433385751187260512[104] = dt;
   out_1433385751187260512[105] = 0;
   out_1433385751187260512[106] = 0;
   out_1433385751187260512[107] = 0;
   out_1433385751187260512[108] = 0;
   out_1433385751187260512[109] = 0;
   out_1433385751187260512[110] = 0;
   out_1433385751187260512[111] = 0;
   out_1433385751187260512[112] = 0;
   out_1433385751187260512[113] = 0;
   out_1433385751187260512[114] = 1;
   out_1433385751187260512[115] = 0;
   out_1433385751187260512[116] = 0;
   out_1433385751187260512[117] = 0;
   out_1433385751187260512[118] = 0;
   out_1433385751187260512[119] = 0;
   out_1433385751187260512[120] = 0;
   out_1433385751187260512[121] = 0;
   out_1433385751187260512[122] = 0;
   out_1433385751187260512[123] = 0;
   out_1433385751187260512[124] = 0;
   out_1433385751187260512[125] = 0;
   out_1433385751187260512[126] = 0;
   out_1433385751187260512[127] = 0;
   out_1433385751187260512[128] = 0;
   out_1433385751187260512[129] = 0;
   out_1433385751187260512[130] = 0;
   out_1433385751187260512[131] = 0;
   out_1433385751187260512[132] = 0;
   out_1433385751187260512[133] = 1;
   out_1433385751187260512[134] = 0;
   out_1433385751187260512[135] = 0;
   out_1433385751187260512[136] = 0;
   out_1433385751187260512[137] = 0;
   out_1433385751187260512[138] = 0;
   out_1433385751187260512[139] = 0;
   out_1433385751187260512[140] = 0;
   out_1433385751187260512[141] = 0;
   out_1433385751187260512[142] = 0;
   out_1433385751187260512[143] = 0;
   out_1433385751187260512[144] = 0;
   out_1433385751187260512[145] = 0;
   out_1433385751187260512[146] = 0;
   out_1433385751187260512[147] = 0;
   out_1433385751187260512[148] = 0;
   out_1433385751187260512[149] = 0;
   out_1433385751187260512[150] = 0;
   out_1433385751187260512[151] = 0;
   out_1433385751187260512[152] = 1;
   out_1433385751187260512[153] = 0;
   out_1433385751187260512[154] = 0;
   out_1433385751187260512[155] = 0;
   out_1433385751187260512[156] = 0;
   out_1433385751187260512[157] = 0;
   out_1433385751187260512[158] = 0;
   out_1433385751187260512[159] = 0;
   out_1433385751187260512[160] = 0;
   out_1433385751187260512[161] = 0;
   out_1433385751187260512[162] = 0;
   out_1433385751187260512[163] = 0;
   out_1433385751187260512[164] = 0;
   out_1433385751187260512[165] = 0;
   out_1433385751187260512[166] = 0;
   out_1433385751187260512[167] = 0;
   out_1433385751187260512[168] = 0;
   out_1433385751187260512[169] = 0;
   out_1433385751187260512[170] = 0;
   out_1433385751187260512[171] = 1;
   out_1433385751187260512[172] = 0;
   out_1433385751187260512[173] = 0;
   out_1433385751187260512[174] = 0;
   out_1433385751187260512[175] = 0;
   out_1433385751187260512[176] = 0;
   out_1433385751187260512[177] = 0;
   out_1433385751187260512[178] = 0;
   out_1433385751187260512[179] = 0;
   out_1433385751187260512[180] = 0;
   out_1433385751187260512[181] = 0;
   out_1433385751187260512[182] = 0;
   out_1433385751187260512[183] = 0;
   out_1433385751187260512[184] = 0;
   out_1433385751187260512[185] = 0;
   out_1433385751187260512[186] = 0;
   out_1433385751187260512[187] = 0;
   out_1433385751187260512[188] = 0;
   out_1433385751187260512[189] = 0;
   out_1433385751187260512[190] = 1;
   out_1433385751187260512[191] = 0;
   out_1433385751187260512[192] = 0;
   out_1433385751187260512[193] = 0;
   out_1433385751187260512[194] = 0;
   out_1433385751187260512[195] = 0;
   out_1433385751187260512[196] = 0;
   out_1433385751187260512[197] = 0;
   out_1433385751187260512[198] = 0;
   out_1433385751187260512[199] = 0;
   out_1433385751187260512[200] = 0;
   out_1433385751187260512[201] = 0;
   out_1433385751187260512[202] = 0;
   out_1433385751187260512[203] = 0;
   out_1433385751187260512[204] = 0;
   out_1433385751187260512[205] = 0;
   out_1433385751187260512[206] = 0;
   out_1433385751187260512[207] = 0;
   out_1433385751187260512[208] = 0;
   out_1433385751187260512[209] = 1;
   out_1433385751187260512[210] = 0;
   out_1433385751187260512[211] = 0;
   out_1433385751187260512[212] = 0;
   out_1433385751187260512[213] = 0;
   out_1433385751187260512[214] = 0;
   out_1433385751187260512[215] = 0;
   out_1433385751187260512[216] = 0;
   out_1433385751187260512[217] = 0;
   out_1433385751187260512[218] = 0;
   out_1433385751187260512[219] = 0;
   out_1433385751187260512[220] = 0;
   out_1433385751187260512[221] = 0;
   out_1433385751187260512[222] = 0;
   out_1433385751187260512[223] = 0;
   out_1433385751187260512[224] = 0;
   out_1433385751187260512[225] = 0;
   out_1433385751187260512[226] = 0;
   out_1433385751187260512[227] = 0;
   out_1433385751187260512[228] = 1;
   out_1433385751187260512[229] = 0;
   out_1433385751187260512[230] = 0;
   out_1433385751187260512[231] = 0;
   out_1433385751187260512[232] = 0;
   out_1433385751187260512[233] = 0;
   out_1433385751187260512[234] = 0;
   out_1433385751187260512[235] = 0;
   out_1433385751187260512[236] = 0;
   out_1433385751187260512[237] = 0;
   out_1433385751187260512[238] = 0;
   out_1433385751187260512[239] = 0;
   out_1433385751187260512[240] = 0;
   out_1433385751187260512[241] = 0;
   out_1433385751187260512[242] = 0;
   out_1433385751187260512[243] = 0;
   out_1433385751187260512[244] = 0;
   out_1433385751187260512[245] = 0;
   out_1433385751187260512[246] = 0;
   out_1433385751187260512[247] = 1;
   out_1433385751187260512[248] = 0;
   out_1433385751187260512[249] = 0;
   out_1433385751187260512[250] = 0;
   out_1433385751187260512[251] = 0;
   out_1433385751187260512[252] = 0;
   out_1433385751187260512[253] = 0;
   out_1433385751187260512[254] = 0;
   out_1433385751187260512[255] = 0;
   out_1433385751187260512[256] = 0;
   out_1433385751187260512[257] = 0;
   out_1433385751187260512[258] = 0;
   out_1433385751187260512[259] = 0;
   out_1433385751187260512[260] = 0;
   out_1433385751187260512[261] = 0;
   out_1433385751187260512[262] = 0;
   out_1433385751187260512[263] = 0;
   out_1433385751187260512[264] = 0;
   out_1433385751187260512[265] = 0;
   out_1433385751187260512[266] = 1;
   out_1433385751187260512[267] = 0;
   out_1433385751187260512[268] = 0;
   out_1433385751187260512[269] = 0;
   out_1433385751187260512[270] = 0;
   out_1433385751187260512[271] = 0;
   out_1433385751187260512[272] = 0;
   out_1433385751187260512[273] = 0;
   out_1433385751187260512[274] = 0;
   out_1433385751187260512[275] = 0;
   out_1433385751187260512[276] = 0;
   out_1433385751187260512[277] = 0;
   out_1433385751187260512[278] = 0;
   out_1433385751187260512[279] = 0;
   out_1433385751187260512[280] = 0;
   out_1433385751187260512[281] = 0;
   out_1433385751187260512[282] = 0;
   out_1433385751187260512[283] = 0;
   out_1433385751187260512[284] = 0;
   out_1433385751187260512[285] = 1;
   out_1433385751187260512[286] = 0;
   out_1433385751187260512[287] = 0;
   out_1433385751187260512[288] = 0;
   out_1433385751187260512[289] = 0;
   out_1433385751187260512[290] = 0;
   out_1433385751187260512[291] = 0;
   out_1433385751187260512[292] = 0;
   out_1433385751187260512[293] = 0;
   out_1433385751187260512[294] = 0;
   out_1433385751187260512[295] = 0;
   out_1433385751187260512[296] = 0;
   out_1433385751187260512[297] = 0;
   out_1433385751187260512[298] = 0;
   out_1433385751187260512[299] = 0;
   out_1433385751187260512[300] = 0;
   out_1433385751187260512[301] = 0;
   out_1433385751187260512[302] = 0;
   out_1433385751187260512[303] = 0;
   out_1433385751187260512[304] = 1;
   out_1433385751187260512[305] = 0;
   out_1433385751187260512[306] = 0;
   out_1433385751187260512[307] = 0;
   out_1433385751187260512[308] = 0;
   out_1433385751187260512[309] = 0;
   out_1433385751187260512[310] = 0;
   out_1433385751187260512[311] = 0;
   out_1433385751187260512[312] = 0;
   out_1433385751187260512[313] = 0;
   out_1433385751187260512[314] = 0;
   out_1433385751187260512[315] = 0;
   out_1433385751187260512[316] = 0;
   out_1433385751187260512[317] = 0;
   out_1433385751187260512[318] = 0;
   out_1433385751187260512[319] = 0;
   out_1433385751187260512[320] = 0;
   out_1433385751187260512[321] = 0;
   out_1433385751187260512[322] = 0;
   out_1433385751187260512[323] = 1;
}
void h_4(double *state, double *unused, double *out_5960296184483738033) {
   out_5960296184483738033[0] = state[6] + state[9];
   out_5960296184483738033[1] = state[7] + state[10];
   out_5960296184483738033[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_8903629035839261452) {
   out_8903629035839261452[0] = 0;
   out_8903629035839261452[1] = 0;
   out_8903629035839261452[2] = 0;
   out_8903629035839261452[3] = 0;
   out_8903629035839261452[4] = 0;
   out_8903629035839261452[5] = 0;
   out_8903629035839261452[6] = 1;
   out_8903629035839261452[7] = 0;
   out_8903629035839261452[8] = 0;
   out_8903629035839261452[9] = 1;
   out_8903629035839261452[10] = 0;
   out_8903629035839261452[11] = 0;
   out_8903629035839261452[12] = 0;
   out_8903629035839261452[13] = 0;
   out_8903629035839261452[14] = 0;
   out_8903629035839261452[15] = 0;
   out_8903629035839261452[16] = 0;
   out_8903629035839261452[17] = 0;
   out_8903629035839261452[18] = 0;
   out_8903629035839261452[19] = 0;
   out_8903629035839261452[20] = 0;
   out_8903629035839261452[21] = 0;
   out_8903629035839261452[22] = 0;
   out_8903629035839261452[23] = 0;
   out_8903629035839261452[24] = 0;
   out_8903629035839261452[25] = 1;
   out_8903629035839261452[26] = 0;
   out_8903629035839261452[27] = 0;
   out_8903629035839261452[28] = 1;
   out_8903629035839261452[29] = 0;
   out_8903629035839261452[30] = 0;
   out_8903629035839261452[31] = 0;
   out_8903629035839261452[32] = 0;
   out_8903629035839261452[33] = 0;
   out_8903629035839261452[34] = 0;
   out_8903629035839261452[35] = 0;
   out_8903629035839261452[36] = 0;
   out_8903629035839261452[37] = 0;
   out_8903629035839261452[38] = 0;
   out_8903629035839261452[39] = 0;
   out_8903629035839261452[40] = 0;
   out_8903629035839261452[41] = 0;
   out_8903629035839261452[42] = 0;
   out_8903629035839261452[43] = 0;
   out_8903629035839261452[44] = 1;
   out_8903629035839261452[45] = 0;
   out_8903629035839261452[46] = 0;
   out_8903629035839261452[47] = 1;
   out_8903629035839261452[48] = 0;
   out_8903629035839261452[49] = 0;
   out_8903629035839261452[50] = 0;
   out_8903629035839261452[51] = 0;
   out_8903629035839261452[52] = 0;
   out_8903629035839261452[53] = 0;
}
void h_10(double *state, double *unused, double *out_2371720150447351138) {
   out_2371720150447351138[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2371720150447351138[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2371720150447351138[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_8795871100057175473) {
   out_8795871100057175473[0] = 0;
   out_8795871100057175473[1] = 9.8100000000000005*cos(state[1]);
   out_8795871100057175473[2] = 0;
   out_8795871100057175473[3] = 0;
   out_8795871100057175473[4] = -state[8];
   out_8795871100057175473[5] = state[7];
   out_8795871100057175473[6] = 0;
   out_8795871100057175473[7] = state[5];
   out_8795871100057175473[8] = -state[4];
   out_8795871100057175473[9] = 0;
   out_8795871100057175473[10] = 0;
   out_8795871100057175473[11] = 0;
   out_8795871100057175473[12] = 1;
   out_8795871100057175473[13] = 0;
   out_8795871100057175473[14] = 0;
   out_8795871100057175473[15] = 1;
   out_8795871100057175473[16] = 0;
   out_8795871100057175473[17] = 0;
   out_8795871100057175473[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_8795871100057175473[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_8795871100057175473[20] = 0;
   out_8795871100057175473[21] = state[8];
   out_8795871100057175473[22] = 0;
   out_8795871100057175473[23] = -state[6];
   out_8795871100057175473[24] = -state[5];
   out_8795871100057175473[25] = 0;
   out_8795871100057175473[26] = state[3];
   out_8795871100057175473[27] = 0;
   out_8795871100057175473[28] = 0;
   out_8795871100057175473[29] = 0;
   out_8795871100057175473[30] = 0;
   out_8795871100057175473[31] = 1;
   out_8795871100057175473[32] = 0;
   out_8795871100057175473[33] = 0;
   out_8795871100057175473[34] = 1;
   out_8795871100057175473[35] = 0;
   out_8795871100057175473[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_8795871100057175473[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_8795871100057175473[38] = 0;
   out_8795871100057175473[39] = -state[7];
   out_8795871100057175473[40] = state[6];
   out_8795871100057175473[41] = 0;
   out_8795871100057175473[42] = state[4];
   out_8795871100057175473[43] = -state[3];
   out_8795871100057175473[44] = 0;
   out_8795871100057175473[45] = 0;
   out_8795871100057175473[46] = 0;
   out_8795871100057175473[47] = 0;
   out_8795871100057175473[48] = 0;
   out_8795871100057175473[49] = 0;
   out_8795871100057175473[50] = 1;
   out_8795871100057175473[51] = 0;
   out_8795871100057175473[52] = 0;
   out_8795871100057175473[53] = 1;
}
void h_13(double *state, double *unused, double *out_790049311483924192) {
   out_790049311483924192[0] = state[3];
   out_790049311483924192[1] = state[4];
   out_790049311483924192[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5691355210506928651) {
   out_5691355210506928651[0] = 0;
   out_5691355210506928651[1] = 0;
   out_5691355210506928651[2] = 0;
   out_5691355210506928651[3] = 1;
   out_5691355210506928651[4] = 0;
   out_5691355210506928651[5] = 0;
   out_5691355210506928651[6] = 0;
   out_5691355210506928651[7] = 0;
   out_5691355210506928651[8] = 0;
   out_5691355210506928651[9] = 0;
   out_5691355210506928651[10] = 0;
   out_5691355210506928651[11] = 0;
   out_5691355210506928651[12] = 0;
   out_5691355210506928651[13] = 0;
   out_5691355210506928651[14] = 0;
   out_5691355210506928651[15] = 0;
   out_5691355210506928651[16] = 0;
   out_5691355210506928651[17] = 0;
   out_5691355210506928651[18] = 0;
   out_5691355210506928651[19] = 0;
   out_5691355210506928651[20] = 0;
   out_5691355210506928651[21] = 0;
   out_5691355210506928651[22] = 1;
   out_5691355210506928651[23] = 0;
   out_5691355210506928651[24] = 0;
   out_5691355210506928651[25] = 0;
   out_5691355210506928651[26] = 0;
   out_5691355210506928651[27] = 0;
   out_5691355210506928651[28] = 0;
   out_5691355210506928651[29] = 0;
   out_5691355210506928651[30] = 0;
   out_5691355210506928651[31] = 0;
   out_5691355210506928651[32] = 0;
   out_5691355210506928651[33] = 0;
   out_5691355210506928651[34] = 0;
   out_5691355210506928651[35] = 0;
   out_5691355210506928651[36] = 0;
   out_5691355210506928651[37] = 0;
   out_5691355210506928651[38] = 0;
   out_5691355210506928651[39] = 0;
   out_5691355210506928651[40] = 0;
   out_5691355210506928651[41] = 1;
   out_5691355210506928651[42] = 0;
   out_5691355210506928651[43] = 0;
   out_5691355210506928651[44] = 0;
   out_5691355210506928651[45] = 0;
   out_5691355210506928651[46] = 0;
   out_5691355210506928651[47] = 0;
   out_5691355210506928651[48] = 0;
   out_5691355210506928651[49] = 0;
   out_5691355210506928651[50] = 0;
   out_5691355210506928651[51] = 0;
   out_5691355210506928651[52] = 0;
   out_5691355210506928651[53] = 0;
}
void h_14(double *state, double *unused, double *out_7499051543095232089) {
   out_7499051543095232089[0] = state[6];
   out_7499051543095232089[1] = state[7];
   out_7499051543095232089[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4940388179499776923) {
   out_4940388179499776923[0] = 0;
   out_4940388179499776923[1] = 0;
   out_4940388179499776923[2] = 0;
   out_4940388179499776923[3] = 0;
   out_4940388179499776923[4] = 0;
   out_4940388179499776923[5] = 0;
   out_4940388179499776923[6] = 1;
   out_4940388179499776923[7] = 0;
   out_4940388179499776923[8] = 0;
   out_4940388179499776923[9] = 0;
   out_4940388179499776923[10] = 0;
   out_4940388179499776923[11] = 0;
   out_4940388179499776923[12] = 0;
   out_4940388179499776923[13] = 0;
   out_4940388179499776923[14] = 0;
   out_4940388179499776923[15] = 0;
   out_4940388179499776923[16] = 0;
   out_4940388179499776923[17] = 0;
   out_4940388179499776923[18] = 0;
   out_4940388179499776923[19] = 0;
   out_4940388179499776923[20] = 0;
   out_4940388179499776923[21] = 0;
   out_4940388179499776923[22] = 0;
   out_4940388179499776923[23] = 0;
   out_4940388179499776923[24] = 0;
   out_4940388179499776923[25] = 1;
   out_4940388179499776923[26] = 0;
   out_4940388179499776923[27] = 0;
   out_4940388179499776923[28] = 0;
   out_4940388179499776923[29] = 0;
   out_4940388179499776923[30] = 0;
   out_4940388179499776923[31] = 0;
   out_4940388179499776923[32] = 0;
   out_4940388179499776923[33] = 0;
   out_4940388179499776923[34] = 0;
   out_4940388179499776923[35] = 0;
   out_4940388179499776923[36] = 0;
   out_4940388179499776923[37] = 0;
   out_4940388179499776923[38] = 0;
   out_4940388179499776923[39] = 0;
   out_4940388179499776923[40] = 0;
   out_4940388179499776923[41] = 0;
   out_4940388179499776923[42] = 0;
   out_4940388179499776923[43] = 0;
   out_4940388179499776923[44] = 1;
   out_4940388179499776923[45] = 0;
   out_4940388179499776923[46] = 0;
   out_4940388179499776923[47] = 0;
   out_4940388179499776923[48] = 0;
   out_4940388179499776923[49] = 0;
   out_4940388179499776923[50] = 0;
   out_4940388179499776923[51] = 0;
   out_4940388179499776923[52] = 0;
   out_4940388179499776923[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_3110422523440551038) {
  err_fun(nom_x, delta_x, out_3110422523440551038);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2741580585373072420) {
  inv_err_fun(nom_x, true_x, out_2741580585373072420);
}
void pose_H_mod_fun(double *state, double *out_3055547229087302731) {
  H_mod_fun(state, out_3055547229087302731);
}
void pose_f_fun(double *state, double dt, double *out_2068541479278321289) {
  f_fun(state,  dt, out_2068541479278321289);
}
void pose_F_fun(double *state, double dt, double *out_1433385751187260512) {
  F_fun(state,  dt, out_1433385751187260512);
}
void pose_h_4(double *state, double *unused, double *out_5960296184483738033) {
  h_4(state, unused, out_5960296184483738033);
}
void pose_H_4(double *state, double *unused, double *out_8903629035839261452) {
  H_4(state, unused, out_8903629035839261452);
}
void pose_h_10(double *state, double *unused, double *out_2371720150447351138) {
  h_10(state, unused, out_2371720150447351138);
}
void pose_H_10(double *state, double *unused, double *out_8795871100057175473) {
  H_10(state, unused, out_8795871100057175473);
}
void pose_h_13(double *state, double *unused, double *out_790049311483924192) {
  h_13(state, unused, out_790049311483924192);
}
void pose_H_13(double *state, double *unused, double *out_5691355210506928651) {
  H_13(state, unused, out_5691355210506928651);
}
void pose_h_14(double *state, double *unused, double *out_7499051543095232089) {
  h_14(state, unused, out_7499051543095232089);
}
void pose_H_14(double *state, double *unused, double *out_4940388179499776923) {
  H_14(state, unused, out_4940388179499776923);
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
