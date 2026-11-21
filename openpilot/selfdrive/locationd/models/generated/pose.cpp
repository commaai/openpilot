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
void err_fun(double *nom_x, double *delta_x, double *out_4336476086057343998) {
   out_4336476086057343998[0] = delta_x[0] + nom_x[0];
   out_4336476086057343998[1] = delta_x[1] + nom_x[1];
   out_4336476086057343998[2] = delta_x[2] + nom_x[2];
   out_4336476086057343998[3] = delta_x[3] + nom_x[3];
   out_4336476086057343998[4] = delta_x[4] + nom_x[4];
   out_4336476086057343998[5] = delta_x[5] + nom_x[5];
   out_4336476086057343998[6] = delta_x[6] + nom_x[6];
   out_4336476086057343998[7] = delta_x[7] + nom_x[7];
   out_4336476086057343998[8] = delta_x[8] + nom_x[8];
   out_4336476086057343998[9] = delta_x[9] + nom_x[9];
   out_4336476086057343998[10] = delta_x[10] + nom_x[10];
   out_4336476086057343998[11] = delta_x[11] + nom_x[11];
   out_4336476086057343998[12] = delta_x[12] + nom_x[12];
   out_4336476086057343998[13] = delta_x[13] + nom_x[13];
   out_4336476086057343998[14] = delta_x[14] + nom_x[14];
   out_4336476086057343998[15] = delta_x[15] + nom_x[15];
   out_4336476086057343998[16] = delta_x[16] + nom_x[16];
   out_4336476086057343998[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1258106400637610027) {
   out_1258106400637610027[0] = -nom_x[0] + true_x[0];
   out_1258106400637610027[1] = -nom_x[1] + true_x[1];
   out_1258106400637610027[2] = -nom_x[2] + true_x[2];
   out_1258106400637610027[3] = -nom_x[3] + true_x[3];
   out_1258106400637610027[4] = -nom_x[4] + true_x[4];
   out_1258106400637610027[5] = -nom_x[5] + true_x[5];
   out_1258106400637610027[6] = -nom_x[6] + true_x[6];
   out_1258106400637610027[7] = -nom_x[7] + true_x[7];
   out_1258106400637610027[8] = -nom_x[8] + true_x[8];
   out_1258106400637610027[9] = -nom_x[9] + true_x[9];
   out_1258106400637610027[10] = -nom_x[10] + true_x[10];
   out_1258106400637610027[11] = -nom_x[11] + true_x[11];
   out_1258106400637610027[12] = -nom_x[12] + true_x[12];
   out_1258106400637610027[13] = -nom_x[13] + true_x[13];
   out_1258106400637610027[14] = -nom_x[14] + true_x[14];
   out_1258106400637610027[15] = -nom_x[15] + true_x[15];
   out_1258106400637610027[16] = -nom_x[16] + true_x[16];
   out_1258106400637610027[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_6527990562688054560) {
   out_6527990562688054560[0] = 1.0;
   out_6527990562688054560[1] = 0.0;
   out_6527990562688054560[2] = 0.0;
   out_6527990562688054560[3] = 0.0;
   out_6527990562688054560[4] = 0.0;
   out_6527990562688054560[5] = 0.0;
   out_6527990562688054560[6] = 0.0;
   out_6527990562688054560[7] = 0.0;
   out_6527990562688054560[8] = 0.0;
   out_6527990562688054560[9] = 0.0;
   out_6527990562688054560[10] = 0.0;
   out_6527990562688054560[11] = 0.0;
   out_6527990562688054560[12] = 0.0;
   out_6527990562688054560[13] = 0.0;
   out_6527990562688054560[14] = 0.0;
   out_6527990562688054560[15] = 0.0;
   out_6527990562688054560[16] = 0.0;
   out_6527990562688054560[17] = 0.0;
   out_6527990562688054560[18] = 0.0;
   out_6527990562688054560[19] = 1.0;
   out_6527990562688054560[20] = 0.0;
   out_6527990562688054560[21] = 0.0;
   out_6527990562688054560[22] = 0.0;
   out_6527990562688054560[23] = 0.0;
   out_6527990562688054560[24] = 0.0;
   out_6527990562688054560[25] = 0.0;
   out_6527990562688054560[26] = 0.0;
   out_6527990562688054560[27] = 0.0;
   out_6527990562688054560[28] = 0.0;
   out_6527990562688054560[29] = 0.0;
   out_6527990562688054560[30] = 0.0;
   out_6527990562688054560[31] = 0.0;
   out_6527990562688054560[32] = 0.0;
   out_6527990562688054560[33] = 0.0;
   out_6527990562688054560[34] = 0.0;
   out_6527990562688054560[35] = 0.0;
   out_6527990562688054560[36] = 0.0;
   out_6527990562688054560[37] = 0.0;
   out_6527990562688054560[38] = 1.0;
   out_6527990562688054560[39] = 0.0;
   out_6527990562688054560[40] = 0.0;
   out_6527990562688054560[41] = 0.0;
   out_6527990562688054560[42] = 0.0;
   out_6527990562688054560[43] = 0.0;
   out_6527990562688054560[44] = 0.0;
   out_6527990562688054560[45] = 0.0;
   out_6527990562688054560[46] = 0.0;
   out_6527990562688054560[47] = 0.0;
   out_6527990562688054560[48] = 0.0;
   out_6527990562688054560[49] = 0.0;
   out_6527990562688054560[50] = 0.0;
   out_6527990562688054560[51] = 0.0;
   out_6527990562688054560[52] = 0.0;
   out_6527990562688054560[53] = 0.0;
   out_6527990562688054560[54] = 0.0;
   out_6527990562688054560[55] = 0.0;
   out_6527990562688054560[56] = 0.0;
   out_6527990562688054560[57] = 1.0;
   out_6527990562688054560[58] = 0.0;
   out_6527990562688054560[59] = 0.0;
   out_6527990562688054560[60] = 0.0;
   out_6527990562688054560[61] = 0.0;
   out_6527990562688054560[62] = 0.0;
   out_6527990562688054560[63] = 0.0;
   out_6527990562688054560[64] = 0.0;
   out_6527990562688054560[65] = 0.0;
   out_6527990562688054560[66] = 0.0;
   out_6527990562688054560[67] = 0.0;
   out_6527990562688054560[68] = 0.0;
   out_6527990562688054560[69] = 0.0;
   out_6527990562688054560[70] = 0.0;
   out_6527990562688054560[71] = 0.0;
   out_6527990562688054560[72] = 0.0;
   out_6527990562688054560[73] = 0.0;
   out_6527990562688054560[74] = 0.0;
   out_6527990562688054560[75] = 0.0;
   out_6527990562688054560[76] = 1.0;
   out_6527990562688054560[77] = 0.0;
   out_6527990562688054560[78] = 0.0;
   out_6527990562688054560[79] = 0.0;
   out_6527990562688054560[80] = 0.0;
   out_6527990562688054560[81] = 0.0;
   out_6527990562688054560[82] = 0.0;
   out_6527990562688054560[83] = 0.0;
   out_6527990562688054560[84] = 0.0;
   out_6527990562688054560[85] = 0.0;
   out_6527990562688054560[86] = 0.0;
   out_6527990562688054560[87] = 0.0;
   out_6527990562688054560[88] = 0.0;
   out_6527990562688054560[89] = 0.0;
   out_6527990562688054560[90] = 0.0;
   out_6527990562688054560[91] = 0.0;
   out_6527990562688054560[92] = 0.0;
   out_6527990562688054560[93] = 0.0;
   out_6527990562688054560[94] = 0.0;
   out_6527990562688054560[95] = 1.0;
   out_6527990562688054560[96] = 0.0;
   out_6527990562688054560[97] = 0.0;
   out_6527990562688054560[98] = 0.0;
   out_6527990562688054560[99] = 0.0;
   out_6527990562688054560[100] = 0.0;
   out_6527990562688054560[101] = 0.0;
   out_6527990562688054560[102] = 0.0;
   out_6527990562688054560[103] = 0.0;
   out_6527990562688054560[104] = 0.0;
   out_6527990562688054560[105] = 0.0;
   out_6527990562688054560[106] = 0.0;
   out_6527990562688054560[107] = 0.0;
   out_6527990562688054560[108] = 0.0;
   out_6527990562688054560[109] = 0.0;
   out_6527990562688054560[110] = 0.0;
   out_6527990562688054560[111] = 0.0;
   out_6527990562688054560[112] = 0.0;
   out_6527990562688054560[113] = 0.0;
   out_6527990562688054560[114] = 1.0;
   out_6527990562688054560[115] = 0.0;
   out_6527990562688054560[116] = 0.0;
   out_6527990562688054560[117] = 0.0;
   out_6527990562688054560[118] = 0.0;
   out_6527990562688054560[119] = 0.0;
   out_6527990562688054560[120] = 0.0;
   out_6527990562688054560[121] = 0.0;
   out_6527990562688054560[122] = 0.0;
   out_6527990562688054560[123] = 0.0;
   out_6527990562688054560[124] = 0.0;
   out_6527990562688054560[125] = 0.0;
   out_6527990562688054560[126] = 0.0;
   out_6527990562688054560[127] = 0.0;
   out_6527990562688054560[128] = 0.0;
   out_6527990562688054560[129] = 0.0;
   out_6527990562688054560[130] = 0.0;
   out_6527990562688054560[131] = 0.0;
   out_6527990562688054560[132] = 0.0;
   out_6527990562688054560[133] = 1.0;
   out_6527990562688054560[134] = 0.0;
   out_6527990562688054560[135] = 0.0;
   out_6527990562688054560[136] = 0.0;
   out_6527990562688054560[137] = 0.0;
   out_6527990562688054560[138] = 0.0;
   out_6527990562688054560[139] = 0.0;
   out_6527990562688054560[140] = 0.0;
   out_6527990562688054560[141] = 0.0;
   out_6527990562688054560[142] = 0.0;
   out_6527990562688054560[143] = 0.0;
   out_6527990562688054560[144] = 0.0;
   out_6527990562688054560[145] = 0.0;
   out_6527990562688054560[146] = 0.0;
   out_6527990562688054560[147] = 0.0;
   out_6527990562688054560[148] = 0.0;
   out_6527990562688054560[149] = 0.0;
   out_6527990562688054560[150] = 0.0;
   out_6527990562688054560[151] = 0.0;
   out_6527990562688054560[152] = 1.0;
   out_6527990562688054560[153] = 0.0;
   out_6527990562688054560[154] = 0.0;
   out_6527990562688054560[155] = 0.0;
   out_6527990562688054560[156] = 0.0;
   out_6527990562688054560[157] = 0.0;
   out_6527990562688054560[158] = 0.0;
   out_6527990562688054560[159] = 0.0;
   out_6527990562688054560[160] = 0.0;
   out_6527990562688054560[161] = 0.0;
   out_6527990562688054560[162] = 0.0;
   out_6527990562688054560[163] = 0.0;
   out_6527990562688054560[164] = 0.0;
   out_6527990562688054560[165] = 0.0;
   out_6527990562688054560[166] = 0.0;
   out_6527990562688054560[167] = 0.0;
   out_6527990562688054560[168] = 0.0;
   out_6527990562688054560[169] = 0.0;
   out_6527990562688054560[170] = 0.0;
   out_6527990562688054560[171] = 1.0;
   out_6527990562688054560[172] = 0.0;
   out_6527990562688054560[173] = 0.0;
   out_6527990562688054560[174] = 0.0;
   out_6527990562688054560[175] = 0.0;
   out_6527990562688054560[176] = 0.0;
   out_6527990562688054560[177] = 0.0;
   out_6527990562688054560[178] = 0.0;
   out_6527990562688054560[179] = 0.0;
   out_6527990562688054560[180] = 0.0;
   out_6527990562688054560[181] = 0.0;
   out_6527990562688054560[182] = 0.0;
   out_6527990562688054560[183] = 0.0;
   out_6527990562688054560[184] = 0.0;
   out_6527990562688054560[185] = 0.0;
   out_6527990562688054560[186] = 0.0;
   out_6527990562688054560[187] = 0.0;
   out_6527990562688054560[188] = 0.0;
   out_6527990562688054560[189] = 0.0;
   out_6527990562688054560[190] = 1.0;
   out_6527990562688054560[191] = 0.0;
   out_6527990562688054560[192] = 0.0;
   out_6527990562688054560[193] = 0.0;
   out_6527990562688054560[194] = 0.0;
   out_6527990562688054560[195] = 0.0;
   out_6527990562688054560[196] = 0.0;
   out_6527990562688054560[197] = 0.0;
   out_6527990562688054560[198] = 0.0;
   out_6527990562688054560[199] = 0.0;
   out_6527990562688054560[200] = 0.0;
   out_6527990562688054560[201] = 0.0;
   out_6527990562688054560[202] = 0.0;
   out_6527990562688054560[203] = 0.0;
   out_6527990562688054560[204] = 0.0;
   out_6527990562688054560[205] = 0.0;
   out_6527990562688054560[206] = 0.0;
   out_6527990562688054560[207] = 0.0;
   out_6527990562688054560[208] = 0.0;
   out_6527990562688054560[209] = 1.0;
   out_6527990562688054560[210] = 0.0;
   out_6527990562688054560[211] = 0.0;
   out_6527990562688054560[212] = 0.0;
   out_6527990562688054560[213] = 0.0;
   out_6527990562688054560[214] = 0.0;
   out_6527990562688054560[215] = 0.0;
   out_6527990562688054560[216] = 0.0;
   out_6527990562688054560[217] = 0.0;
   out_6527990562688054560[218] = 0.0;
   out_6527990562688054560[219] = 0.0;
   out_6527990562688054560[220] = 0.0;
   out_6527990562688054560[221] = 0.0;
   out_6527990562688054560[222] = 0.0;
   out_6527990562688054560[223] = 0.0;
   out_6527990562688054560[224] = 0.0;
   out_6527990562688054560[225] = 0.0;
   out_6527990562688054560[226] = 0.0;
   out_6527990562688054560[227] = 0.0;
   out_6527990562688054560[228] = 1.0;
   out_6527990562688054560[229] = 0.0;
   out_6527990562688054560[230] = 0.0;
   out_6527990562688054560[231] = 0.0;
   out_6527990562688054560[232] = 0.0;
   out_6527990562688054560[233] = 0.0;
   out_6527990562688054560[234] = 0.0;
   out_6527990562688054560[235] = 0.0;
   out_6527990562688054560[236] = 0.0;
   out_6527990562688054560[237] = 0.0;
   out_6527990562688054560[238] = 0.0;
   out_6527990562688054560[239] = 0.0;
   out_6527990562688054560[240] = 0.0;
   out_6527990562688054560[241] = 0.0;
   out_6527990562688054560[242] = 0.0;
   out_6527990562688054560[243] = 0.0;
   out_6527990562688054560[244] = 0.0;
   out_6527990562688054560[245] = 0.0;
   out_6527990562688054560[246] = 0.0;
   out_6527990562688054560[247] = 1.0;
   out_6527990562688054560[248] = 0.0;
   out_6527990562688054560[249] = 0.0;
   out_6527990562688054560[250] = 0.0;
   out_6527990562688054560[251] = 0.0;
   out_6527990562688054560[252] = 0.0;
   out_6527990562688054560[253] = 0.0;
   out_6527990562688054560[254] = 0.0;
   out_6527990562688054560[255] = 0.0;
   out_6527990562688054560[256] = 0.0;
   out_6527990562688054560[257] = 0.0;
   out_6527990562688054560[258] = 0.0;
   out_6527990562688054560[259] = 0.0;
   out_6527990562688054560[260] = 0.0;
   out_6527990562688054560[261] = 0.0;
   out_6527990562688054560[262] = 0.0;
   out_6527990562688054560[263] = 0.0;
   out_6527990562688054560[264] = 0.0;
   out_6527990562688054560[265] = 0.0;
   out_6527990562688054560[266] = 1.0;
   out_6527990562688054560[267] = 0.0;
   out_6527990562688054560[268] = 0.0;
   out_6527990562688054560[269] = 0.0;
   out_6527990562688054560[270] = 0.0;
   out_6527990562688054560[271] = 0.0;
   out_6527990562688054560[272] = 0.0;
   out_6527990562688054560[273] = 0.0;
   out_6527990562688054560[274] = 0.0;
   out_6527990562688054560[275] = 0.0;
   out_6527990562688054560[276] = 0.0;
   out_6527990562688054560[277] = 0.0;
   out_6527990562688054560[278] = 0.0;
   out_6527990562688054560[279] = 0.0;
   out_6527990562688054560[280] = 0.0;
   out_6527990562688054560[281] = 0.0;
   out_6527990562688054560[282] = 0.0;
   out_6527990562688054560[283] = 0.0;
   out_6527990562688054560[284] = 0.0;
   out_6527990562688054560[285] = 1.0;
   out_6527990562688054560[286] = 0.0;
   out_6527990562688054560[287] = 0.0;
   out_6527990562688054560[288] = 0.0;
   out_6527990562688054560[289] = 0.0;
   out_6527990562688054560[290] = 0.0;
   out_6527990562688054560[291] = 0.0;
   out_6527990562688054560[292] = 0.0;
   out_6527990562688054560[293] = 0.0;
   out_6527990562688054560[294] = 0.0;
   out_6527990562688054560[295] = 0.0;
   out_6527990562688054560[296] = 0.0;
   out_6527990562688054560[297] = 0.0;
   out_6527990562688054560[298] = 0.0;
   out_6527990562688054560[299] = 0.0;
   out_6527990562688054560[300] = 0.0;
   out_6527990562688054560[301] = 0.0;
   out_6527990562688054560[302] = 0.0;
   out_6527990562688054560[303] = 0.0;
   out_6527990562688054560[304] = 1.0;
   out_6527990562688054560[305] = 0.0;
   out_6527990562688054560[306] = 0.0;
   out_6527990562688054560[307] = 0.0;
   out_6527990562688054560[308] = 0.0;
   out_6527990562688054560[309] = 0.0;
   out_6527990562688054560[310] = 0.0;
   out_6527990562688054560[311] = 0.0;
   out_6527990562688054560[312] = 0.0;
   out_6527990562688054560[313] = 0.0;
   out_6527990562688054560[314] = 0.0;
   out_6527990562688054560[315] = 0.0;
   out_6527990562688054560[316] = 0.0;
   out_6527990562688054560[317] = 0.0;
   out_6527990562688054560[318] = 0.0;
   out_6527990562688054560[319] = 0.0;
   out_6527990562688054560[320] = 0.0;
   out_6527990562688054560[321] = 0.0;
   out_6527990562688054560[322] = 0.0;
   out_6527990562688054560[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8412414231100200223) {
   out_8412414231100200223[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8412414231100200223[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8412414231100200223[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8412414231100200223[3] = dt*state[12] + state[3];
   out_8412414231100200223[4] = dt*state[13] + state[4];
   out_8412414231100200223[5] = dt*state[14] + state[5];
   out_8412414231100200223[6] = state[6];
   out_8412414231100200223[7] = state[7];
   out_8412414231100200223[8] = state[8];
   out_8412414231100200223[9] = state[9];
   out_8412414231100200223[10] = state[10];
   out_8412414231100200223[11] = state[11];
   out_8412414231100200223[12] = state[12];
   out_8412414231100200223[13] = state[13];
   out_8412414231100200223[14] = state[14];
   out_8412414231100200223[15] = state[15];
   out_8412414231100200223[16] = state[16];
   out_8412414231100200223[17] = state[17];
}
void F_fun(double *state, double dt, double *out_5891562599511662399) {
   out_5891562599511662399[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5891562599511662399[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5891562599511662399[2] = 0;
   out_5891562599511662399[3] = 0;
   out_5891562599511662399[4] = 0;
   out_5891562599511662399[5] = 0;
   out_5891562599511662399[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5891562599511662399[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5891562599511662399[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5891562599511662399[9] = 0;
   out_5891562599511662399[10] = 0;
   out_5891562599511662399[11] = 0;
   out_5891562599511662399[12] = 0;
   out_5891562599511662399[13] = 0;
   out_5891562599511662399[14] = 0;
   out_5891562599511662399[15] = 0;
   out_5891562599511662399[16] = 0;
   out_5891562599511662399[17] = 0;
   out_5891562599511662399[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5891562599511662399[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5891562599511662399[20] = 0;
   out_5891562599511662399[21] = 0;
   out_5891562599511662399[22] = 0;
   out_5891562599511662399[23] = 0;
   out_5891562599511662399[24] = 0;
   out_5891562599511662399[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5891562599511662399[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5891562599511662399[27] = 0;
   out_5891562599511662399[28] = 0;
   out_5891562599511662399[29] = 0;
   out_5891562599511662399[30] = 0;
   out_5891562599511662399[31] = 0;
   out_5891562599511662399[32] = 0;
   out_5891562599511662399[33] = 0;
   out_5891562599511662399[34] = 0;
   out_5891562599511662399[35] = 0;
   out_5891562599511662399[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5891562599511662399[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5891562599511662399[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5891562599511662399[39] = 0;
   out_5891562599511662399[40] = 0;
   out_5891562599511662399[41] = 0;
   out_5891562599511662399[42] = 0;
   out_5891562599511662399[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5891562599511662399[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5891562599511662399[45] = 0;
   out_5891562599511662399[46] = 0;
   out_5891562599511662399[47] = 0;
   out_5891562599511662399[48] = 0;
   out_5891562599511662399[49] = 0;
   out_5891562599511662399[50] = 0;
   out_5891562599511662399[51] = 0;
   out_5891562599511662399[52] = 0;
   out_5891562599511662399[53] = 0;
   out_5891562599511662399[54] = 0;
   out_5891562599511662399[55] = 0;
   out_5891562599511662399[56] = 0;
   out_5891562599511662399[57] = 1;
   out_5891562599511662399[58] = 0;
   out_5891562599511662399[59] = 0;
   out_5891562599511662399[60] = 0;
   out_5891562599511662399[61] = 0;
   out_5891562599511662399[62] = 0;
   out_5891562599511662399[63] = 0;
   out_5891562599511662399[64] = 0;
   out_5891562599511662399[65] = 0;
   out_5891562599511662399[66] = dt;
   out_5891562599511662399[67] = 0;
   out_5891562599511662399[68] = 0;
   out_5891562599511662399[69] = 0;
   out_5891562599511662399[70] = 0;
   out_5891562599511662399[71] = 0;
   out_5891562599511662399[72] = 0;
   out_5891562599511662399[73] = 0;
   out_5891562599511662399[74] = 0;
   out_5891562599511662399[75] = 0;
   out_5891562599511662399[76] = 1;
   out_5891562599511662399[77] = 0;
   out_5891562599511662399[78] = 0;
   out_5891562599511662399[79] = 0;
   out_5891562599511662399[80] = 0;
   out_5891562599511662399[81] = 0;
   out_5891562599511662399[82] = 0;
   out_5891562599511662399[83] = 0;
   out_5891562599511662399[84] = 0;
   out_5891562599511662399[85] = dt;
   out_5891562599511662399[86] = 0;
   out_5891562599511662399[87] = 0;
   out_5891562599511662399[88] = 0;
   out_5891562599511662399[89] = 0;
   out_5891562599511662399[90] = 0;
   out_5891562599511662399[91] = 0;
   out_5891562599511662399[92] = 0;
   out_5891562599511662399[93] = 0;
   out_5891562599511662399[94] = 0;
   out_5891562599511662399[95] = 1;
   out_5891562599511662399[96] = 0;
   out_5891562599511662399[97] = 0;
   out_5891562599511662399[98] = 0;
   out_5891562599511662399[99] = 0;
   out_5891562599511662399[100] = 0;
   out_5891562599511662399[101] = 0;
   out_5891562599511662399[102] = 0;
   out_5891562599511662399[103] = 0;
   out_5891562599511662399[104] = dt;
   out_5891562599511662399[105] = 0;
   out_5891562599511662399[106] = 0;
   out_5891562599511662399[107] = 0;
   out_5891562599511662399[108] = 0;
   out_5891562599511662399[109] = 0;
   out_5891562599511662399[110] = 0;
   out_5891562599511662399[111] = 0;
   out_5891562599511662399[112] = 0;
   out_5891562599511662399[113] = 0;
   out_5891562599511662399[114] = 1;
   out_5891562599511662399[115] = 0;
   out_5891562599511662399[116] = 0;
   out_5891562599511662399[117] = 0;
   out_5891562599511662399[118] = 0;
   out_5891562599511662399[119] = 0;
   out_5891562599511662399[120] = 0;
   out_5891562599511662399[121] = 0;
   out_5891562599511662399[122] = 0;
   out_5891562599511662399[123] = 0;
   out_5891562599511662399[124] = 0;
   out_5891562599511662399[125] = 0;
   out_5891562599511662399[126] = 0;
   out_5891562599511662399[127] = 0;
   out_5891562599511662399[128] = 0;
   out_5891562599511662399[129] = 0;
   out_5891562599511662399[130] = 0;
   out_5891562599511662399[131] = 0;
   out_5891562599511662399[132] = 0;
   out_5891562599511662399[133] = 1;
   out_5891562599511662399[134] = 0;
   out_5891562599511662399[135] = 0;
   out_5891562599511662399[136] = 0;
   out_5891562599511662399[137] = 0;
   out_5891562599511662399[138] = 0;
   out_5891562599511662399[139] = 0;
   out_5891562599511662399[140] = 0;
   out_5891562599511662399[141] = 0;
   out_5891562599511662399[142] = 0;
   out_5891562599511662399[143] = 0;
   out_5891562599511662399[144] = 0;
   out_5891562599511662399[145] = 0;
   out_5891562599511662399[146] = 0;
   out_5891562599511662399[147] = 0;
   out_5891562599511662399[148] = 0;
   out_5891562599511662399[149] = 0;
   out_5891562599511662399[150] = 0;
   out_5891562599511662399[151] = 0;
   out_5891562599511662399[152] = 1;
   out_5891562599511662399[153] = 0;
   out_5891562599511662399[154] = 0;
   out_5891562599511662399[155] = 0;
   out_5891562599511662399[156] = 0;
   out_5891562599511662399[157] = 0;
   out_5891562599511662399[158] = 0;
   out_5891562599511662399[159] = 0;
   out_5891562599511662399[160] = 0;
   out_5891562599511662399[161] = 0;
   out_5891562599511662399[162] = 0;
   out_5891562599511662399[163] = 0;
   out_5891562599511662399[164] = 0;
   out_5891562599511662399[165] = 0;
   out_5891562599511662399[166] = 0;
   out_5891562599511662399[167] = 0;
   out_5891562599511662399[168] = 0;
   out_5891562599511662399[169] = 0;
   out_5891562599511662399[170] = 0;
   out_5891562599511662399[171] = 1;
   out_5891562599511662399[172] = 0;
   out_5891562599511662399[173] = 0;
   out_5891562599511662399[174] = 0;
   out_5891562599511662399[175] = 0;
   out_5891562599511662399[176] = 0;
   out_5891562599511662399[177] = 0;
   out_5891562599511662399[178] = 0;
   out_5891562599511662399[179] = 0;
   out_5891562599511662399[180] = 0;
   out_5891562599511662399[181] = 0;
   out_5891562599511662399[182] = 0;
   out_5891562599511662399[183] = 0;
   out_5891562599511662399[184] = 0;
   out_5891562599511662399[185] = 0;
   out_5891562599511662399[186] = 0;
   out_5891562599511662399[187] = 0;
   out_5891562599511662399[188] = 0;
   out_5891562599511662399[189] = 0;
   out_5891562599511662399[190] = 1;
   out_5891562599511662399[191] = 0;
   out_5891562599511662399[192] = 0;
   out_5891562599511662399[193] = 0;
   out_5891562599511662399[194] = 0;
   out_5891562599511662399[195] = 0;
   out_5891562599511662399[196] = 0;
   out_5891562599511662399[197] = 0;
   out_5891562599511662399[198] = 0;
   out_5891562599511662399[199] = 0;
   out_5891562599511662399[200] = 0;
   out_5891562599511662399[201] = 0;
   out_5891562599511662399[202] = 0;
   out_5891562599511662399[203] = 0;
   out_5891562599511662399[204] = 0;
   out_5891562599511662399[205] = 0;
   out_5891562599511662399[206] = 0;
   out_5891562599511662399[207] = 0;
   out_5891562599511662399[208] = 0;
   out_5891562599511662399[209] = 1;
   out_5891562599511662399[210] = 0;
   out_5891562599511662399[211] = 0;
   out_5891562599511662399[212] = 0;
   out_5891562599511662399[213] = 0;
   out_5891562599511662399[214] = 0;
   out_5891562599511662399[215] = 0;
   out_5891562599511662399[216] = 0;
   out_5891562599511662399[217] = 0;
   out_5891562599511662399[218] = 0;
   out_5891562599511662399[219] = 0;
   out_5891562599511662399[220] = 0;
   out_5891562599511662399[221] = 0;
   out_5891562599511662399[222] = 0;
   out_5891562599511662399[223] = 0;
   out_5891562599511662399[224] = 0;
   out_5891562599511662399[225] = 0;
   out_5891562599511662399[226] = 0;
   out_5891562599511662399[227] = 0;
   out_5891562599511662399[228] = 1;
   out_5891562599511662399[229] = 0;
   out_5891562599511662399[230] = 0;
   out_5891562599511662399[231] = 0;
   out_5891562599511662399[232] = 0;
   out_5891562599511662399[233] = 0;
   out_5891562599511662399[234] = 0;
   out_5891562599511662399[235] = 0;
   out_5891562599511662399[236] = 0;
   out_5891562599511662399[237] = 0;
   out_5891562599511662399[238] = 0;
   out_5891562599511662399[239] = 0;
   out_5891562599511662399[240] = 0;
   out_5891562599511662399[241] = 0;
   out_5891562599511662399[242] = 0;
   out_5891562599511662399[243] = 0;
   out_5891562599511662399[244] = 0;
   out_5891562599511662399[245] = 0;
   out_5891562599511662399[246] = 0;
   out_5891562599511662399[247] = 1;
   out_5891562599511662399[248] = 0;
   out_5891562599511662399[249] = 0;
   out_5891562599511662399[250] = 0;
   out_5891562599511662399[251] = 0;
   out_5891562599511662399[252] = 0;
   out_5891562599511662399[253] = 0;
   out_5891562599511662399[254] = 0;
   out_5891562599511662399[255] = 0;
   out_5891562599511662399[256] = 0;
   out_5891562599511662399[257] = 0;
   out_5891562599511662399[258] = 0;
   out_5891562599511662399[259] = 0;
   out_5891562599511662399[260] = 0;
   out_5891562599511662399[261] = 0;
   out_5891562599511662399[262] = 0;
   out_5891562599511662399[263] = 0;
   out_5891562599511662399[264] = 0;
   out_5891562599511662399[265] = 0;
   out_5891562599511662399[266] = 1;
   out_5891562599511662399[267] = 0;
   out_5891562599511662399[268] = 0;
   out_5891562599511662399[269] = 0;
   out_5891562599511662399[270] = 0;
   out_5891562599511662399[271] = 0;
   out_5891562599511662399[272] = 0;
   out_5891562599511662399[273] = 0;
   out_5891562599511662399[274] = 0;
   out_5891562599511662399[275] = 0;
   out_5891562599511662399[276] = 0;
   out_5891562599511662399[277] = 0;
   out_5891562599511662399[278] = 0;
   out_5891562599511662399[279] = 0;
   out_5891562599511662399[280] = 0;
   out_5891562599511662399[281] = 0;
   out_5891562599511662399[282] = 0;
   out_5891562599511662399[283] = 0;
   out_5891562599511662399[284] = 0;
   out_5891562599511662399[285] = 1;
   out_5891562599511662399[286] = 0;
   out_5891562599511662399[287] = 0;
   out_5891562599511662399[288] = 0;
   out_5891562599511662399[289] = 0;
   out_5891562599511662399[290] = 0;
   out_5891562599511662399[291] = 0;
   out_5891562599511662399[292] = 0;
   out_5891562599511662399[293] = 0;
   out_5891562599511662399[294] = 0;
   out_5891562599511662399[295] = 0;
   out_5891562599511662399[296] = 0;
   out_5891562599511662399[297] = 0;
   out_5891562599511662399[298] = 0;
   out_5891562599511662399[299] = 0;
   out_5891562599511662399[300] = 0;
   out_5891562599511662399[301] = 0;
   out_5891562599511662399[302] = 0;
   out_5891562599511662399[303] = 0;
   out_5891562599511662399[304] = 1;
   out_5891562599511662399[305] = 0;
   out_5891562599511662399[306] = 0;
   out_5891562599511662399[307] = 0;
   out_5891562599511662399[308] = 0;
   out_5891562599511662399[309] = 0;
   out_5891562599511662399[310] = 0;
   out_5891562599511662399[311] = 0;
   out_5891562599511662399[312] = 0;
   out_5891562599511662399[313] = 0;
   out_5891562599511662399[314] = 0;
   out_5891562599511662399[315] = 0;
   out_5891562599511662399[316] = 0;
   out_5891562599511662399[317] = 0;
   out_5891562599511662399[318] = 0;
   out_5891562599511662399[319] = 0;
   out_5891562599511662399[320] = 0;
   out_5891562599511662399[321] = 0;
   out_5891562599511662399[322] = 0;
   out_5891562599511662399[323] = 1;
}
void h_4(double *state, double *unused, double *out_5872972277401777544) {
   out_5872972277401777544[0] = state[6] + state[9];
   out_5872972277401777544[1] = state[7] + state[10];
   out_5872972277401777544[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_2756183352844146988) {
   out_2756183352844146988[0] = 0;
   out_2756183352844146988[1] = 0;
   out_2756183352844146988[2] = 0;
   out_2756183352844146988[3] = 0;
   out_2756183352844146988[4] = 0;
   out_2756183352844146988[5] = 0;
   out_2756183352844146988[6] = 1;
   out_2756183352844146988[7] = 0;
   out_2756183352844146988[8] = 0;
   out_2756183352844146988[9] = 1;
   out_2756183352844146988[10] = 0;
   out_2756183352844146988[11] = 0;
   out_2756183352844146988[12] = 0;
   out_2756183352844146988[13] = 0;
   out_2756183352844146988[14] = 0;
   out_2756183352844146988[15] = 0;
   out_2756183352844146988[16] = 0;
   out_2756183352844146988[17] = 0;
   out_2756183352844146988[18] = 0;
   out_2756183352844146988[19] = 0;
   out_2756183352844146988[20] = 0;
   out_2756183352844146988[21] = 0;
   out_2756183352844146988[22] = 0;
   out_2756183352844146988[23] = 0;
   out_2756183352844146988[24] = 0;
   out_2756183352844146988[25] = 1;
   out_2756183352844146988[26] = 0;
   out_2756183352844146988[27] = 0;
   out_2756183352844146988[28] = 1;
   out_2756183352844146988[29] = 0;
   out_2756183352844146988[30] = 0;
   out_2756183352844146988[31] = 0;
   out_2756183352844146988[32] = 0;
   out_2756183352844146988[33] = 0;
   out_2756183352844146988[34] = 0;
   out_2756183352844146988[35] = 0;
   out_2756183352844146988[36] = 0;
   out_2756183352844146988[37] = 0;
   out_2756183352844146988[38] = 0;
   out_2756183352844146988[39] = 0;
   out_2756183352844146988[40] = 0;
   out_2756183352844146988[41] = 0;
   out_2756183352844146988[42] = 0;
   out_2756183352844146988[43] = 0;
   out_2756183352844146988[44] = 1;
   out_2756183352844146988[45] = 0;
   out_2756183352844146988[46] = 0;
   out_2756183352844146988[47] = 1;
   out_2756183352844146988[48] = 0;
   out_2756183352844146988[49] = 0;
   out_2756183352844146988[50] = 0;
   out_2756183352844146988[51] = 0;
   out_2756183352844146988[52] = 0;
   out_2756183352844146988[53] = 0;
}
void h_10(double *state, double *unused, double *out_2437929224325462368) {
   out_2437929224325462368[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2437929224325462368[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2437929224325462368[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2308350212782797437) {
   out_2308350212782797437[0] = 0;
   out_2308350212782797437[1] = 9.8100000000000005*cos(state[1]);
   out_2308350212782797437[2] = 0;
   out_2308350212782797437[3] = 0;
   out_2308350212782797437[4] = -state[8];
   out_2308350212782797437[5] = state[7];
   out_2308350212782797437[6] = 0;
   out_2308350212782797437[7] = state[5];
   out_2308350212782797437[8] = -state[4];
   out_2308350212782797437[9] = 0;
   out_2308350212782797437[10] = 0;
   out_2308350212782797437[11] = 0;
   out_2308350212782797437[12] = 1;
   out_2308350212782797437[13] = 0;
   out_2308350212782797437[14] = 0;
   out_2308350212782797437[15] = 1;
   out_2308350212782797437[16] = 0;
   out_2308350212782797437[17] = 0;
   out_2308350212782797437[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2308350212782797437[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2308350212782797437[20] = 0;
   out_2308350212782797437[21] = state[8];
   out_2308350212782797437[22] = 0;
   out_2308350212782797437[23] = -state[6];
   out_2308350212782797437[24] = -state[5];
   out_2308350212782797437[25] = 0;
   out_2308350212782797437[26] = state[3];
   out_2308350212782797437[27] = 0;
   out_2308350212782797437[28] = 0;
   out_2308350212782797437[29] = 0;
   out_2308350212782797437[30] = 0;
   out_2308350212782797437[31] = 1;
   out_2308350212782797437[32] = 0;
   out_2308350212782797437[33] = 0;
   out_2308350212782797437[34] = 1;
   out_2308350212782797437[35] = 0;
   out_2308350212782797437[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2308350212782797437[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2308350212782797437[38] = 0;
   out_2308350212782797437[39] = -state[7];
   out_2308350212782797437[40] = state[6];
   out_2308350212782797437[41] = 0;
   out_2308350212782797437[42] = state[4];
   out_2308350212782797437[43] = -state[3];
   out_2308350212782797437[44] = 0;
   out_2308350212782797437[45] = 0;
   out_2308350212782797437[46] = 0;
   out_2308350212782797437[47] = 0;
   out_2308350212782797437[48] = 0;
   out_2308350212782797437[49] = 0;
   out_2308350212782797437[50] = 1;
   out_2308350212782797437[51] = 0;
   out_2308350212782797437[52] = 0;
   out_2308350212782797437[53] = 1;
}
void h_13(double *state, double *unused, double *out_5303446754481693594) {
   out_5303446754481693594[0] = state[3];
   out_5303446754481693594[1] = state[4];
   out_5303446754481693594[2] = state[5];
}
void H_13(double *state, double *unused, double *out_4854447855472553941) {
   out_4854447855472553941[0] = 0;
   out_4854447855472553941[1] = 0;
   out_4854447855472553941[2] = 0;
   out_4854447855472553941[3] = 1;
   out_4854447855472553941[4] = 0;
   out_4854447855472553941[5] = 0;
   out_4854447855472553941[6] = 0;
   out_4854447855472553941[7] = 0;
   out_4854447855472553941[8] = 0;
   out_4854447855472553941[9] = 0;
   out_4854447855472553941[10] = 0;
   out_4854447855472553941[11] = 0;
   out_4854447855472553941[12] = 0;
   out_4854447855472553941[13] = 0;
   out_4854447855472553941[14] = 0;
   out_4854447855472553941[15] = 0;
   out_4854447855472553941[16] = 0;
   out_4854447855472553941[17] = 0;
   out_4854447855472553941[18] = 0;
   out_4854447855472553941[19] = 0;
   out_4854447855472553941[20] = 0;
   out_4854447855472553941[21] = 0;
   out_4854447855472553941[22] = 1;
   out_4854447855472553941[23] = 0;
   out_4854447855472553941[24] = 0;
   out_4854447855472553941[25] = 0;
   out_4854447855472553941[26] = 0;
   out_4854447855472553941[27] = 0;
   out_4854447855472553941[28] = 0;
   out_4854447855472553941[29] = 0;
   out_4854447855472553941[30] = 0;
   out_4854447855472553941[31] = 0;
   out_4854447855472553941[32] = 0;
   out_4854447855472553941[33] = 0;
   out_4854447855472553941[34] = 0;
   out_4854447855472553941[35] = 0;
   out_4854447855472553941[36] = 0;
   out_4854447855472553941[37] = 0;
   out_4854447855472553941[38] = 0;
   out_4854447855472553941[39] = 0;
   out_4854447855472553941[40] = 0;
   out_4854447855472553941[41] = 1;
   out_4854447855472553941[42] = 0;
   out_4854447855472553941[43] = 0;
   out_4854447855472553941[44] = 0;
   out_4854447855472553941[45] = 0;
   out_4854447855472553941[46] = 0;
   out_4854447855472553941[47] = 0;
   out_4854447855472553941[48] = 0;
   out_4854447855472553941[49] = 0;
   out_4854447855472553941[50] = 0;
   out_4854447855472553941[51] = 0;
   out_4854447855472553941[52] = 0;
   out_4854447855472553941[53] = 0;
}
void h_14(double *state, double *unused, double *out_7614535523248366498) {
   out_7614535523248366498[0] = state[6];
   out_7614535523248366498[1] = state[7];
   out_7614535523248366498[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1207057503495337541) {
   out_1207057503495337541[0] = 0;
   out_1207057503495337541[1] = 0;
   out_1207057503495337541[2] = 0;
   out_1207057503495337541[3] = 0;
   out_1207057503495337541[4] = 0;
   out_1207057503495337541[5] = 0;
   out_1207057503495337541[6] = 1;
   out_1207057503495337541[7] = 0;
   out_1207057503495337541[8] = 0;
   out_1207057503495337541[9] = 0;
   out_1207057503495337541[10] = 0;
   out_1207057503495337541[11] = 0;
   out_1207057503495337541[12] = 0;
   out_1207057503495337541[13] = 0;
   out_1207057503495337541[14] = 0;
   out_1207057503495337541[15] = 0;
   out_1207057503495337541[16] = 0;
   out_1207057503495337541[17] = 0;
   out_1207057503495337541[18] = 0;
   out_1207057503495337541[19] = 0;
   out_1207057503495337541[20] = 0;
   out_1207057503495337541[21] = 0;
   out_1207057503495337541[22] = 0;
   out_1207057503495337541[23] = 0;
   out_1207057503495337541[24] = 0;
   out_1207057503495337541[25] = 1;
   out_1207057503495337541[26] = 0;
   out_1207057503495337541[27] = 0;
   out_1207057503495337541[28] = 0;
   out_1207057503495337541[29] = 0;
   out_1207057503495337541[30] = 0;
   out_1207057503495337541[31] = 0;
   out_1207057503495337541[32] = 0;
   out_1207057503495337541[33] = 0;
   out_1207057503495337541[34] = 0;
   out_1207057503495337541[35] = 0;
   out_1207057503495337541[36] = 0;
   out_1207057503495337541[37] = 0;
   out_1207057503495337541[38] = 0;
   out_1207057503495337541[39] = 0;
   out_1207057503495337541[40] = 0;
   out_1207057503495337541[41] = 0;
   out_1207057503495337541[42] = 0;
   out_1207057503495337541[43] = 0;
   out_1207057503495337541[44] = 1;
   out_1207057503495337541[45] = 0;
   out_1207057503495337541[46] = 0;
   out_1207057503495337541[47] = 0;
   out_1207057503495337541[48] = 0;
   out_1207057503495337541[49] = 0;
   out_1207057503495337541[50] = 0;
   out_1207057503495337541[51] = 0;
   out_1207057503495337541[52] = 0;
   out_1207057503495337541[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_4336476086057343998) {
  err_fun(nom_x, delta_x, out_4336476086057343998);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1258106400637610027) {
  inv_err_fun(nom_x, true_x, out_1258106400637610027);
}
void pose_H_mod_fun(double *state, double *out_6527990562688054560) {
  H_mod_fun(state, out_6527990562688054560);
}
void pose_f_fun(double *state, double dt, double *out_8412414231100200223) {
  f_fun(state,  dt, out_8412414231100200223);
}
void pose_F_fun(double *state, double dt, double *out_5891562599511662399) {
  F_fun(state,  dt, out_5891562599511662399);
}
void pose_h_4(double *state, double *unused, double *out_5872972277401777544) {
  h_4(state, unused, out_5872972277401777544);
}
void pose_H_4(double *state, double *unused, double *out_2756183352844146988) {
  H_4(state, unused, out_2756183352844146988);
}
void pose_h_10(double *state, double *unused, double *out_2437929224325462368) {
  h_10(state, unused, out_2437929224325462368);
}
void pose_H_10(double *state, double *unused, double *out_2308350212782797437) {
  H_10(state, unused, out_2308350212782797437);
}
void pose_h_13(double *state, double *unused, double *out_5303446754481693594) {
  h_13(state, unused, out_5303446754481693594);
}
void pose_H_13(double *state, double *unused, double *out_4854447855472553941) {
  H_13(state, unused, out_4854447855472553941);
}
void pose_h_14(double *state, double *unused, double *out_7614535523248366498) {
  h_14(state, unused, out_7614535523248366498);
}
void pose_H_14(double *state, double *unused, double *out_1207057503495337541) {
  H_14(state, unused, out_1207057503495337541);
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
