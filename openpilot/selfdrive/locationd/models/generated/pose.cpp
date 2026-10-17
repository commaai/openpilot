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
void err_fun(double *nom_x, double *delta_x, double *out_3812274923922881427) {
   out_3812274923922881427[0] = delta_x[0] + nom_x[0];
   out_3812274923922881427[1] = delta_x[1] + nom_x[1];
   out_3812274923922881427[2] = delta_x[2] + nom_x[2];
   out_3812274923922881427[3] = delta_x[3] + nom_x[3];
   out_3812274923922881427[4] = delta_x[4] + nom_x[4];
   out_3812274923922881427[5] = delta_x[5] + nom_x[5];
   out_3812274923922881427[6] = delta_x[6] + nom_x[6];
   out_3812274923922881427[7] = delta_x[7] + nom_x[7];
   out_3812274923922881427[8] = delta_x[8] + nom_x[8];
   out_3812274923922881427[9] = delta_x[9] + nom_x[9];
   out_3812274923922881427[10] = delta_x[10] + nom_x[10];
   out_3812274923922881427[11] = delta_x[11] + nom_x[11];
   out_3812274923922881427[12] = delta_x[12] + nom_x[12];
   out_3812274923922881427[13] = delta_x[13] + nom_x[13];
   out_3812274923922881427[14] = delta_x[14] + nom_x[14];
   out_3812274923922881427[15] = delta_x[15] + nom_x[15];
   out_3812274923922881427[16] = delta_x[16] + nom_x[16];
   out_3812274923922881427[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3419763620583365922) {
   out_3419763620583365922[0] = -nom_x[0] + true_x[0];
   out_3419763620583365922[1] = -nom_x[1] + true_x[1];
   out_3419763620583365922[2] = -nom_x[2] + true_x[2];
   out_3419763620583365922[3] = -nom_x[3] + true_x[3];
   out_3419763620583365922[4] = -nom_x[4] + true_x[4];
   out_3419763620583365922[5] = -nom_x[5] + true_x[5];
   out_3419763620583365922[6] = -nom_x[6] + true_x[6];
   out_3419763620583365922[7] = -nom_x[7] + true_x[7];
   out_3419763620583365922[8] = -nom_x[8] + true_x[8];
   out_3419763620583365922[9] = -nom_x[9] + true_x[9];
   out_3419763620583365922[10] = -nom_x[10] + true_x[10];
   out_3419763620583365922[11] = -nom_x[11] + true_x[11];
   out_3419763620583365922[12] = -nom_x[12] + true_x[12];
   out_3419763620583365922[13] = -nom_x[13] + true_x[13];
   out_3419763620583365922[14] = -nom_x[14] + true_x[14];
   out_3419763620583365922[15] = -nom_x[15] + true_x[15];
   out_3419763620583365922[16] = -nom_x[16] + true_x[16];
   out_3419763620583365922[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_6689702553421276651) {
   out_6689702553421276651[0] = 1.0;
   out_6689702553421276651[1] = 0.0;
   out_6689702553421276651[2] = 0.0;
   out_6689702553421276651[3] = 0.0;
   out_6689702553421276651[4] = 0.0;
   out_6689702553421276651[5] = 0.0;
   out_6689702553421276651[6] = 0.0;
   out_6689702553421276651[7] = 0.0;
   out_6689702553421276651[8] = 0.0;
   out_6689702553421276651[9] = 0.0;
   out_6689702553421276651[10] = 0.0;
   out_6689702553421276651[11] = 0.0;
   out_6689702553421276651[12] = 0.0;
   out_6689702553421276651[13] = 0.0;
   out_6689702553421276651[14] = 0.0;
   out_6689702553421276651[15] = 0.0;
   out_6689702553421276651[16] = 0.0;
   out_6689702553421276651[17] = 0.0;
   out_6689702553421276651[18] = 0.0;
   out_6689702553421276651[19] = 1.0;
   out_6689702553421276651[20] = 0.0;
   out_6689702553421276651[21] = 0.0;
   out_6689702553421276651[22] = 0.0;
   out_6689702553421276651[23] = 0.0;
   out_6689702553421276651[24] = 0.0;
   out_6689702553421276651[25] = 0.0;
   out_6689702553421276651[26] = 0.0;
   out_6689702553421276651[27] = 0.0;
   out_6689702553421276651[28] = 0.0;
   out_6689702553421276651[29] = 0.0;
   out_6689702553421276651[30] = 0.0;
   out_6689702553421276651[31] = 0.0;
   out_6689702553421276651[32] = 0.0;
   out_6689702553421276651[33] = 0.0;
   out_6689702553421276651[34] = 0.0;
   out_6689702553421276651[35] = 0.0;
   out_6689702553421276651[36] = 0.0;
   out_6689702553421276651[37] = 0.0;
   out_6689702553421276651[38] = 1.0;
   out_6689702553421276651[39] = 0.0;
   out_6689702553421276651[40] = 0.0;
   out_6689702553421276651[41] = 0.0;
   out_6689702553421276651[42] = 0.0;
   out_6689702553421276651[43] = 0.0;
   out_6689702553421276651[44] = 0.0;
   out_6689702553421276651[45] = 0.0;
   out_6689702553421276651[46] = 0.0;
   out_6689702553421276651[47] = 0.0;
   out_6689702553421276651[48] = 0.0;
   out_6689702553421276651[49] = 0.0;
   out_6689702553421276651[50] = 0.0;
   out_6689702553421276651[51] = 0.0;
   out_6689702553421276651[52] = 0.0;
   out_6689702553421276651[53] = 0.0;
   out_6689702553421276651[54] = 0.0;
   out_6689702553421276651[55] = 0.0;
   out_6689702553421276651[56] = 0.0;
   out_6689702553421276651[57] = 1.0;
   out_6689702553421276651[58] = 0.0;
   out_6689702553421276651[59] = 0.0;
   out_6689702553421276651[60] = 0.0;
   out_6689702553421276651[61] = 0.0;
   out_6689702553421276651[62] = 0.0;
   out_6689702553421276651[63] = 0.0;
   out_6689702553421276651[64] = 0.0;
   out_6689702553421276651[65] = 0.0;
   out_6689702553421276651[66] = 0.0;
   out_6689702553421276651[67] = 0.0;
   out_6689702553421276651[68] = 0.0;
   out_6689702553421276651[69] = 0.0;
   out_6689702553421276651[70] = 0.0;
   out_6689702553421276651[71] = 0.0;
   out_6689702553421276651[72] = 0.0;
   out_6689702553421276651[73] = 0.0;
   out_6689702553421276651[74] = 0.0;
   out_6689702553421276651[75] = 0.0;
   out_6689702553421276651[76] = 1.0;
   out_6689702553421276651[77] = 0.0;
   out_6689702553421276651[78] = 0.0;
   out_6689702553421276651[79] = 0.0;
   out_6689702553421276651[80] = 0.0;
   out_6689702553421276651[81] = 0.0;
   out_6689702553421276651[82] = 0.0;
   out_6689702553421276651[83] = 0.0;
   out_6689702553421276651[84] = 0.0;
   out_6689702553421276651[85] = 0.0;
   out_6689702553421276651[86] = 0.0;
   out_6689702553421276651[87] = 0.0;
   out_6689702553421276651[88] = 0.0;
   out_6689702553421276651[89] = 0.0;
   out_6689702553421276651[90] = 0.0;
   out_6689702553421276651[91] = 0.0;
   out_6689702553421276651[92] = 0.0;
   out_6689702553421276651[93] = 0.0;
   out_6689702553421276651[94] = 0.0;
   out_6689702553421276651[95] = 1.0;
   out_6689702553421276651[96] = 0.0;
   out_6689702553421276651[97] = 0.0;
   out_6689702553421276651[98] = 0.0;
   out_6689702553421276651[99] = 0.0;
   out_6689702553421276651[100] = 0.0;
   out_6689702553421276651[101] = 0.0;
   out_6689702553421276651[102] = 0.0;
   out_6689702553421276651[103] = 0.0;
   out_6689702553421276651[104] = 0.0;
   out_6689702553421276651[105] = 0.0;
   out_6689702553421276651[106] = 0.0;
   out_6689702553421276651[107] = 0.0;
   out_6689702553421276651[108] = 0.0;
   out_6689702553421276651[109] = 0.0;
   out_6689702553421276651[110] = 0.0;
   out_6689702553421276651[111] = 0.0;
   out_6689702553421276651[112] = 0.0;
   out_6689702553421276651[113] = 0.0;
   out_6689702553421276651[114] = 1.0;
   out_6689702553421276651[115] = 0.0;
   out_6689702553421276651[116] = 0.0;
   out_6689702553421276651[117] = 0.0;
   out_6689702553421276651[118] = 0.0;
   out_6689702553421276651[119] = 0.0;
   out_6689702553421276651[120] = 0.0;
   out_6689702553421276651[121] = 0.0;
   out_6689702553421276651[122] = 0.0;
   out_6689702553421276651[123] = 0.0;
   out_6689702553421276651[124] = 0.0;
   out_6689702553421276651[125] = 0.0;
   out_6689702553421276651[126] = 0.0;
   out_6689702553421276651[127] = 0.0;
   out_6689702553421276651[128] = 0.0;
   out_6689702553421276651[129] = 0.0;
   out_6689702553421276651[130] = 0.0;
   out_6689702553421276651[131] = 0.0;
   out_6689702553421276651[132] = 0.0;
   out_6689702553421276651[133] = 1.0;
   out_6689702553421276651[134] = 0.0;
   out_6689702553421276651[135] = 0.0;
   out_6689702553421276651[136] = 0.0;
   out_6689702553421276651[137] = 0.0;
   out_6689702553421276651[138] = 0.0;
   out_6689702553421276651[139] = 0.0;
   out_6689702553421276651[140] = 0.0;
   out_6689702553421276651[141] = 0.0;
   out_6689702553421276651[142] = 0.0;
   out_6689702553421276651[143] = 0.0;
   out_6689702553421276651[144] = 0.0;
   out_6689702553421276651[145] = 0.0;
   out_6689702553421276651[146] = 0.0;
   out_6689702553421276651[147] = 0.0;
   out_6689702553421276651[148] = 0.0;
   out_6689702553421276651[149] = 0.0;
   out_6689702553421276651[150] = 0.0;
   out_6689702553421276651[151] = 0.0;
   out_6689702553421276651[152] = 1.0;
   out_6689702553421276651[153] = 0.0;
   out_6689702553421276651[154] = 0.0;
   out_6689702553421276651[155] = 0.0;
   out_6689702553421276651[156] = 0.0;
   out_6689702553421276651[157] = 0.0;
   out_6689702553421276651[158] = 0.0;
   out_6689702553421276651[159] = 0.0;
   out_6689702553421276651[160] = 0.0;
   out_6689702553421276651[161] = 0.0;
   out_6689702553421276651[162] = 0.0;
   out_6689702553421276651[163] = 0.0;
   out_6689702553421276651[164] = 0.0;
   out_6689702553421276651[165] = 0.0;
   out_6689702553421276651[166] = 0.0;
   out_6689702553421276651[167] = 0.0;
   out_6689702553421276651[168] = 0.0;
   out_6689702553421276651[169] = 0.0;
   out_6689702553421276651[170] = 0.0;
   out_6689702553421276651[171] = 1.0;
   out_6689702553421276651[172] = 0.0;
   out_6689702553421276651[173] = 0.0;
   out_6689702553421276651[174] = 0.0;
   out_6689702553421276651[175] = 0.0;
   out_6689702553421276651[176] = 0.0;
   out_6689702553421276651[177] = 0.0;
   out_6689702553421276651[178] = 0.0;
   out_6689702553421276651[179] = 0.0;
   out_6689702553421276651[180] = 0.0;
   out_6689702553421276651[181] = 0.0;
   out_6689702553421276651[182] = 0.0;
   out_6689702553421276651[183] = 0.0;
   out_6689702553421276651[184] = 0.0;
   out_6689702553421276651[185] = 0.0;
   out_6689702553421276651[186] = 0.0;
   out_6689702553421276651[187] = 0.0;
   out_6689702553421276651[188] = 0.0;
   out_6689702553421276651[189] = 0.0;
   out_6689702553421276651[190] = 1.0;
   out_6689702553421276651[191] = 0.0;
   out_6689702553421276651[192] = 0.0;
   out_6689702553421276651[193] = 0.0;
   out_6689702553421276651[194] = 0.0;
   out_6689702553421276651[195] = 0.0;
   out_6689702553421276651[196] = 0.0;
   out_6689702553421276651[197] = 0.0;
   out_6689702553421276651[198] = 0.0;
   out_6689702553421276651[199] = 0.0;
   out_6689702553421276651[200] = 0.0;
   out_6689702553421276651[201] = 0.0;
   out_6689702553421276651[202] = 0.0;
   out_6689702553421276651[203] = 0.0;
   out_6689702553421276651[204] = 0.0;
   out_6689702553421276651[205] = 0.0;
   out_6689702553421276651[206] = 0.0;
   out_6689702553421276651[207] = 0.0;
   out_6689702553421276651[208] = 0.0;
   out_6689702553421276651[209] = 1.0;
   out_6689702553421276651[210] = 0.0;
   out_6689702553421276651[211] = 0.0;
   out_6689702553421276651[212] = 0.0;
   out_6689702553421276651[213] = 0.0;
   out_6689702553421276651[214] = 0.0;
   out_6689702553421276651[215] = 0.0;
   out_6689702553421276651[216] = 0.0;
   out_6689702553421276651[217] = 0.0;
   out_6689702553421276651[218] = 0.0;
   out_6689702553421276651[219] = 0.0;
   out_6689702553421276651[220] = 0.0;
   out_6689702553421276651[221] = 0.0;
   out_6689702553421276651[222] = 0.0;
   out_6689702553421276651[223] = 0.0;
   out_6689702553421276651[224] = 0.0;
   out_6689702553421276651[225] = 0.0;
   out_6689702553421276651[226] = 0.0;
   out_6689702553421276651[227] = 0.0;
   out_6689702553421276651[228] = 1.0;
   out_6689702553421276651[229] = 0.0;
   out_6689702553421276651[230] = 0.0;
   out_6689702553421276651[231] = 0.0;
   out_6689702553421276651[232] = 0.0;
   out_6689702553421276651[233] = 0.0;
   out_6689702553421276651[234] = 0.0;
   out_6689702553421276651[235] = 0.0;
   out_6689702553421276651[236] = 0.0;
   out_6689702553421276651[237] = 0.0;
   out_6689702553421276651[238] = 0.0;
   out_6689702553421276651[239] = 0.0;
   out_6689702553421276651[240] = 0.0;
   out_6689702553421276651[241] = 0.0;
   out_6689702553421276651[242] = 0.0;
   out_6689702553421276651[243] = 0.0;
   out_6689702553421276651[244] = 0.0;
   out_6689702553421276651[245] = 0.0;
   out_6689702553421276651[246] = 0.0;
   out_6689702553421276651[247] = 1.0;
   out_6689702553421276651[248] = 0.0;
   out_6689702553421276651[249] = 0.0;
   out_6689702553421276651[250] = 0.0;
   out_6689702553421276651[251] = 0.0;
   out_6689702553421276651[252] = 0.0;
   out_6689702553421276651[253] = 0.0;
   out_6689702553421276651[254] = 0.0;
   out_6689702553421276651[255] = 0.0;
   out_6689702553421276651[256] = 0.0;
   out_6689702553421276651[257] = 0.0;
   out_6689702553421276651[258] = 0.0;
   out_6689702553421276651[259] = 0.0;
   out_6689702553421276651[260] = 0.0;
   out_6689702553421276651[261] = 0.0;
   out_6689702553421276651[262] = 0.0;
   out_6689702553421276651[263] = 0.0;
   out_6689702553421276651[264] = 0.0;
   out_6689702553421276651[265] = 0.0;
   out_6689702553421276651[266] = 1.0;
   out_6689702553421276651[267] = 0.0;
   out_6689702553421276651[268] = 0.0;
   out_6689702553421276651[269] = 0.0;
   out_6689702553421276651[270] = 0.0;
   out_6689702553421276651[271] = 0.0;
   out_6689702553421276651[272] = 0.0;
   out_6689702553421276651[273] = 0.0;
   out_6689702553421276651[274] = 0.0;
   out_6689702553421276651[275] = 0.0;
   out_6689702553421276651[276] = 0.0;
   out_6689702553421276651[277] = 0.0;
   out_6689702553421276651[278] = 0.0;
   out_6689702553421276651[279] = 0.0;
   out_6689702553421276651[280] = 0.0;
   out_6689702553421276651[281] = 0.0;
   out_6689702553421276651[282] = 0.0;
   out_6689702553421276651[283] = 0.0;
   out_6689702553421276651[284] = 0.0;
   out_6689702553421276651[285] = 1.0;
   out_6689702553421276651[286] = 0.0;
   out_6689702553421276651[287] = 0.0;
   out_6689702553421276651[288] = 0.0;
   out_6689702553421276651[289] = 0.0;
   out_6689702553421276651[290] = 0.0;
   out_6689702553421276651[291] = 0.0;
   out_6689702553421276651[292] = 0.0;
   out_6689702553421276651[293] = 0.0;
   out_6689702553421276651[294] = 0.0;
   out_6689702553421276651[295] = 0.0;
   out_6689702553421276651[296] = 0.0;
   out_6689702553421276651[297] = 0.0;
   out_6689702553421276651[298] = 0.0;
   out_6689702553421276651[299] = 0.0;
   out_6689702553421276651[300] = 0.0;
   out_6689702553421276651[301] = 0.0;
   out_6689702553421276651[302] = 0.0;
   out_6689702553421276651[303] = 0.0;
   out_6689702553421276651[304] = 1.0;
   out_6689702553421276651[305] = 0.0;
   out_6689702553421276651[306] = 0.0;
   out_6689702553421276651[307] = 0.0;
   out_6689702553421276651[308] = 0.0;
   out_6689702553421276651[309] = 0.0;
   out_6689702553421276651[310] = 0.0;
   out_6689702553421276651[311] = 0.0;
   out_6689702553421276651[312] = 0.0;
   out_6689702553421276651[313] = 0.0;
   out_6689702553421276651[314] = 0.0;
   out_6689702553421276651[315] = 0.0;
   out_6689702553421276651[316] = 0.0;
   out_6689702553421276651[317] = 0.0;
   out_6689702553421276651[318] = 0.0;
   out_6689702553421276651[319] = 0.0;
   out_6689702553421276651[320] = 0.0;
   out_6689702553421276651[321] = 0.0;
   out_6689702553421276651[322] = 0.0;
   out_6689702553421276651[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_446039340896779864) {
   out_446039340896779864[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_446039340896779864[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_446039340896779864[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_446039340896779864[3] = dt*state[12] + state[3];
   out_446039340896779864[4] = dt*state[13] + state[4];
   out_446039340896779864[5] = dt*state[14] + state[5];
   out_446039340896779864[6] = state[6];
   out_446039340896779864[7] = state[7];
   out_446039340896779864[8] = state[8];
   out_446039340896779864[9] = state[9];
   out_446039340896779864[10] = state[10];
   out_446039340896779864[11] = state[11];
   out_446039340896779864[12] = state[12];
   out_446039340896779864[13] = state[13];
   out_446039340896779864[14] = state[14];
   out_446039340896779864[15] = state[15];
   out_446039340896779864[16] = state[16];
   out_446039340896779864[17] = state[17];
}
void F_fun(double *state, double dt, double *out_642758158368098798) {
   out_642758158368098798[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_642758158368098798[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_642758158368098798[2] = 0;
   out_642758158368098798[3] = 0;
   out_642758158368098798[4] = 0;
   out_642758158368098798[5] = 0;
   out_642758158368098798[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_642758158368098798[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_642758158368098798[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_642758158368098798[9] = 0;
   out_642758158368098798[10] = 0;
   out_642758158368098798[11] = 0;
   out_642758158368098798[12] = 0;
   out_642758158368098798[13] = 0;
   out_642758158368098798[14] = 0;
   out_642758158368098798[15] = 0;
   out_642758158368098798[16] = 0;
   out_642758158368098798[17] = 0;
   out_642758158368098798[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_642758158368098798[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_642758158368098798[20] = 0;
   out_642758158368098798[21] = 0;
   out_642758158368098798[22] = 0;
   out_642758158368098798[23] = 0;
   out_642758158368098798[24] = 0;
   out_642758158368098798[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_642758158368098798[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_642758158368098798[27] = 0;
   out_642758158368098798[28] = 0;
   out_642758158368098798[29] = 0;
   out_642758158368098798[30] = 0;
   out_642758158368098798[31] = 0;
   out_642758158368098798[32] = 0;
   out_642758158368098798[33] = 0;
   out_642758158368098798[34] = 0;
   out_642758158368098798[35] = 0;
   out_642758158368098798[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_642758158368098798[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_642758158368098798[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_642758158368098798[39] = 0;
   out_642758158368098798[40] = 0;
   out_642758158368098798[41] = 0;
   out_642758158368098798[42] = 0;
   out_642758158368098798[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_642758158368098798[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_642758158368098798[45] = 0;
   out_642758158368098798[46] = 0;
   out_642758158368098798[47] = 0;
   out_642758158368098798[48] = 0;
   out_642758158368098798[49] = 0;
   out_642758158368098798[50] = 0;
   out_642758158368098798[51] = 0;
   out_642758158368098798[52] = 0;
   out_642758158368098798[53] = 0;
   out_642758158368098798[54] = 0;
   out_642758158368098798[55] = 0;
   out_642758158368098798[56] = 0;
   out_642758158368098798[57] = 1;
   out_642758158368098798[58] = 0;
   out_642758158368098798[59] = 0;
   out_642758158368098798[60] = 0;
   out_642758158368098798[61] = 0;
   out_642758158368098798[62] = 0;
   out_642758158368098798[63] = 0;
   out_642758158368098798[64] = 0;
   out_642758158368098798[65] = 0;
   out_642758158368098798[66] = dt;
   out_642758158368098798[67] = 0;
   out_642758158368098798[68] = 0;
   out_642758158368098798[69] = 0;
   out_642758158368098798[70] = 0;
   out_642758158368098798[71] = 0;
   out_642758158368098798[72] = 0;
   out_642758158368098798[73] = 0;
   out_642758158368098798[74] = 0;
   out_642758158368098798[75] = 0;
   out_642758158368098798[76] = 1;
   out_642758158368098798[77] = 0;
   out_642758158368098798[78] = 0;
   out_642758158368098798[79] = 0;
   out_642758158368098798[80] = 0;
   out_642758158368098798[81] = 0;
   out_642758158368098798[82] = 0;
   out_642758158368098798[83] = 0;
   out_642758158368098798[84] = 0;
   out_642758158368098798[85] = dt;
   out_642758158368098798[86] = 0;
   out_642758158368098798[87] = 0;
   out_642758158368098798[88] = 0;
   out_642758158368098798[89] = 0;
   out_642758158368098798[90] = 0;
   out_642758158368098798[91] = 0;
   out_642758158368098798[92] = 0;
   out_642758158368098798[93] = 0;
   out_642758158368098798[94] = 0;
   out_642758158368098798[95] = 1;
   out_642758158368098798[96] = 0;
   out_642758158368098798[97] = 0;
   out_642758158368098798[98] = 0;
   out_642758158368098798[99] = 0;
   out_642758158368098798[100] = 0;
   out_642758158368098798[101] = 0;
   out_642758158368098798[102] = 0;
   out_642758158368098798[103] = 0;
   out_642758158368098798[104] = dt;
   out_642758158368098798[105] = 0;
   out_642758158368098798[106] = 0;
   out_642758158368098798[107] = 0;
   out_642758158368098798[108] = 0;
   out_642758158368098798[109] = 0;
   out_642758158368098798[110] = 0;
   out_642758158368098798[111] = 0;
   out_642758158368098798[112] = 0;
   out_642758158368098798[113] = 0;
   out_642758158368098798[114] = 1;
   out_642758158368098798[115] = 0;
   out_642758158368098798[116] = 0;
   out_642758158368098798[117] = 0;
   out_642758158368098798[118] = 0;
   out_642758158368098798[119] = 0;
   out_642758158368098798[120] = 0;
   out_642758158368098798[121] = 0;
   out_642758158368098798[122] = 0;
   out_642758158368098798[123] = 0;
   out_642758158368098798[124] = 0;
   out_642758158368098798[125] = 0;
   out_642758158368098798[126] = 0;
   out_642758158368098798[127] = 0;
   out_642758158368098798[128] = 0;
   out_642758158368098798[129] = 0;
   out_642758158368098798[130] = 0;
   out_642758158368098798[131] = 0;
   out_642758158368098798[132] = 0;
   out_642758158368098798[133] = 1;
   out_642758158368098798[134] = 0;
   out_642758158368098798[135] = 0;
   out_642758158368098798[136] = 0;
   out_642758158368098798[137] = 0;
   out_642758158368098798[138] = 0;
   out_642758158368098798[139] = 0;
   out_642758158368098798[140] = 0;
   out_642758158368098798[141] = 0;
   out_642758158368098798[142] = 0;
   out_642758158368098798[143] = 0;
   out_642758158368098798[144] = 0;
   out_642758158368098798[145] = 0;
   out_642758158368098798[146] = 0;
   out_642758158368098798[147] = 0;
   out_642758158368098798[148] = 0;
   out_642758158368098798[149] = 0;
   out_642758158368098798[150] = 0;
   out_642758158368098798[151] = 0;
   out_642758158368098798[152] = 1;
   out_642758158368098798[153] = 0;
   out_642758158368098798[154] = 0;
   out_642758158368098798[155] = 0;
   out_642758158368098798[156] = 0;
   out_642758158368098798[157] = 0;
   out_642758158368098798[158] = 0;
   out_642758158368098798[159] = 0;
   out_642758158368098798[160] = 0;
   out_642758158368098798[161] = 0;
   out_642758158368098798[162] = 0;
   out_642758158368098798[163] = 0;
   out_642758158368098798[164] = 0;
   out_642758158368098798[165] = 0;
   out_642758158368098798[166] = 0;
   out_642758158368098798[167] = 0;
   out_642758158368098798[168] = 0;
   out_642758158368098798[169] = 0;
   out_642758158368098798[170] = 0;
   out_642758158368098798[171] = 1;
   out_642758158368098798[172] = 0;
   out_642758158368098798[173] = 0;
   out_642758158368098798[174] = 0;
   out_642758158368098798[175] = 0;
   out_642758158368098798[176] = 0;
   out_642758158368098798[177] = 0;
   out_642758158368098798[178] = 0;
   out_642758158368098798[179] = 0;
   out_642758158368098798[180] = 0;
   out_642758158368098798[181] = 0;
   out_642758158368098798[182] = 0;
   out_642758158368098798[183] = 0;
   out_642758158368098798[184] = 0;
   out_642758158368098798[185] = 0;
   out_642758158368098798[186] = 0;
   out_642758158368098798[187] = 0;
   out_642758158368098798[188] = 0;
   out_642758158368098798[189] = 0;
   out_642758158368098798[190] = 1;
   out_642758158368098798[191] = 0;
   out_642758158368098798[192] = 0;
   out_642758158368098798[193] = 0;
   out_642758158368098798[194] = 0;
   out_642758158368098798[195] = 0;
   out_642758158368098798[196] = 0;
   out_642758158368098798[197] = 0;
   out_642758158368098798[198] = 0;
   out_642758158368098798[199] = 0;
   out_642758158368098798[200] = 0;
   out_642758158368098798[201] = 0;
   out_642758158368098798[202] = 0;
   out_642758158368098798[203] = 0;
   out_642758158368098798[204] = 0;
   out_642758158368098798[205] = 0;
   out_642758158368098798[206] = 0;
   out_642758158368098798[207] = 0;
   out_642758158368098798[208] = 0;
   out_642758158368098798[209] = 1;
   out_642758158368098798[210] = 0;
   out_642758158368098798[211] = 0;
   out_642758158368098798[212] = 0;
   out_642758158368098798[213] = 0;
   out_642758158368098798[214] = 0;
   out_642758158368098798[215] = 0;
   out_642758158368098798[216] = 0;
   out_642758158368098798[217] = 0;
   out_642758158368098798[218] = 0;
   out_642758158368098798[219] = 0;
   out_642758158368098798[220] = 0;
   out_642758158368098798[221] = 0;
   out_642758158368098798[222] = 0;
   out_642758158368098798[223] = 0;
   out_642758158368098798[224] = 0;
   out_642758158368098798[225] = 0;
   out_642758158368098798[226] = 0;
   out_642758158368098798[227] = 0;
   out_642758158368098798[228] = 1;
   out_642758158368098798[229] = 0;
   out_642758158368098798[230] = 0;
   out_642758158368098798[231] = 0;
   out_642758158368098798[232] = 0;
   out_642758158368098798[233] = 0;
   out_642758158368098798[234] = 0;
   out_642758158368098798[235] = 0;
   out_642758158368098798[236] = 0;
   out_642758158368098798[237] = 0;
   out_642758158368098798[238] = 0;
   out_642758158368098798[239] = 0;
   out_642758158368098798[240] = 0;
   out_642758158368098798[241] = 0;
   out_642758158368098798[242] = 0;
   out_642758158368098798[243] = 0;
   out_642758158368098798[244] = 0;
   out_642758158368098798[245] = 0;
   out_642758158368098798[246] = 0;
   out_642758158368098798[247] = 1;
   out_642758158368098798[248] = 0;
   out_642758158368098798[249] = 0;
   out_642758158368098798[250] = 0;
   out_642758158368098798[251] = 0;
   out_642758158368098798[252] = 0;
   out_642758158368098798[253] = 0;
   out_642758158368098798[254] = 0;
   out_642758158368098798[255] = 0;
   out_642758158368098798[256] = 0;
   out_642758158368098798[257] = 0;
   out_642758158368098798[258] = 0;
   out_642758158368098798[259] = 0;
   out_642758158368098798[260] = 0;
   out_642758158368098798[261] = 0;
   out_642758158368098798[262] = 0;
   out_642758158368098798[263] = 0;
   out_642758158368098798[264] = 0;
   out_642758158368098798[265] = 0;
   out_642758158368098798[266] = 1;
   out_642758158368098798[267] = 0;
   out_642758158368098798[268] = 0;
   out_642758158368098798[269] = 0;
   out_642758158368098798[270] = 0;
   out_642758158368098798[271] = 0;
   out_642758158368098798[272] = 0;
   out_642758158368098798[273] = 0;
   out_642758158368098798[274] = 0;
   out_642758158368098798[275] = 0;
   out_642758158368098798[276] = 0;
   out_642758158368098798[277] = 0;
   out_642758158368098798[278] = 0;
   out_642758158368098798[279] = 0;
   out_642758158368098798[280] = 0;
   out_642758158368098798[281] = 0;
   out_642758158368098798[282] = 0;
   out_642758158368098798[283] = 0;
   out_642758158368098798[284] = 0;
   out_642758158368098798[285] = 1;
   out_642758158368098798[286] = 0;
   out_642758158368098798[287] = 0;
   out_642758158368098798[288] = 0;
   out_642758158368098798[289] = 0;
   out_642758158368098798[290] = 0;
   out_642758158368098798[291] = 0;
   out_642758158368098798[292] = 0;
   out_642758158368098798[293] = 0;
   out_642758158368098798[294] = 0;
   out_642758158368098798[295] = 0;
   out_642758158368098798[296] = 0;
   out_642758158368098798[297] = 0;
   out_642758158368098798[298] = 0;
   out_642758158368098798[299] = 0;
   out_642758158368098798[300] = 0;
   out_642758158368098798[301] = 0;
   out_642758158368098798[302] = 0;
   out_642758158368098798[303] = 0;
   out_642758158368098798[304] = 1;
   out_642758158368098798[305] = 0;
   out_642758158368098798[306] = 0;
   out_642758158368098798[307] = 0;
   out_642758158368098798[308] = 0;
   out_642758158368098798[309] = 0;
   out_642758158368098798[310] = 0;
   out_642758158368098798[311] = 0;
   out_642758158368098798[312] = 0;
   out_642758158368098798[313] = 0;
   out_642758158368098798[314] = 0;
   out_642758158368098798[315] = 0;
   out_642758158368098798[316] = 0;
   out_642758158368098798[317] = 0;
   out_642758158368098798[318] = 0;
   out_642758158368098798[319] = 0;
   out_642758158368098798[320] = 0;
   out_642758158368098798[321] = 0;
   out_642758158368098798[322] = 0;
   out_642758158368098798[323] = 1;
}
void h_4(double *state, double *unused, double *out_7695501649787619215) {
   out_7695501649787619215[0] = state[6] + state[9];
   out_7695501649787619215[1] = state[7] + state[10];
   out_7695501649787619215[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_2075075516491960287) {
   out_2075075516491960287[0] = 0;
   out_2075075516491960287[1] = 0;
   out_2075075516491960287[2] = 0;
   out_2075075516491960287[3] = 0;
   out_2075075516491960287[4] = 0;
   out_2075075516491960287[5] = 0;
   out_2075075516491960287[6] = 1;
   out_2075075516491960287[7] = 0;
   out_2075075516491960287[8] = 0;
   out_2075075516491960287[9] = 1;
   out_2075075516491960287[10] = 0;
   out_2075075516491960287[11] = 0;
   out_2075075516491960287[12] = 0;
   out_2075075516491960287[13] = 0;
   out_2075075516491960287[14] = 0;
   out_2075075516491960287[15] = 0;
   out_2075075516491960287[16] = 0;
   out_2075075516491960287[17] = 0;
   out_2075075516491960287[18] = 0;
   out_2075075516491960287[19] = 0;
   out_2075075516491960287[20] = 0;
   out_2075075516491960287[21] = 0;
   out_2075075516491960287[22] = 0;
   out_2075075516491960287[23] = 0;
   out_2075075516491960287[24] = 0;
   out_2075075516491960287[25] = 1;
   out_2075075516491960287[26] = 0;
   out_2075075516491960287[27] = 0;
   out_2075075516491960287[28] = 1;
   out_2075075516491960287[29] = 0;
   out_2075075516491960287[30] = 0;
   out_2075075516491960287[31] = 0;
   out_2075075516491960287[32] = 0;
   out_2075075516491960287[33] = 0;
   out_2075075516491960287[34] = 0;
   out_2075075516491960287[35] = 0;
   out_2075075516491960287[36] = 0;
   out_2075075516491960287[37] = 0;
   out_2075075516491960287[38] = 0;
   out_2075075516491960287[39] = 0;
   out_2075075516491960287[40] = 0;
   out_2075075516491960287[41] = 0;
   out_2075075516491960287[42] = 0;
   out_2075075516491960287[43] = 0;
   out_2075075516491960287[44] = 1;
   out_2075075516491960287[45] = 0;
   out_2075075516491960287[46] = 0;
   out_2075075516491960287[47] = 1;
   out_2075075516491960287[48] = 0;
   out_2075075516491960287[49] = 0;
   out_2075075516491960287[50] = 0;
   out_2075075516491960287[51] = 0;
   out_2075075516491960287[52] = 0;
   out_2075075516491960287[53] = 0;
}
void h_10(double *state, double *unused, double *out_289310091903726368) {
   out_289310091903726368[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_289310091903726368[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_289310091903726368[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2508005545533897472) {
   out_2508005545533897472[0] = 0;
   out_2508005545533897472[1] = 9.8100000000000005*cos(state[1]);
   out_2508005545533897472[2] = 0;
   out_2508005545533897472[3] = 0;
   out_2508005545533897472[4] = -state[8];
   out_2508005545533897472[5] = state[7];
   out_2508005545533897472[6] = 0;
   out_2508005545533897472[7] = state[5];
   out_2508005545533897472[8] = -state[4];
   out_2508005545533897472[9] = 0;
   out_2508005545533897472[10] = 0;
   out_2508005545533897472[11] = 0;
   out_2508005545533897472[12] = 1;
   out_2508005545533897472[13] = 0;
   out_2508005545533897472[14] = 0;
   out_2508005545533897472[15] = 1;
   out_2508005545533897472[16] = 0;
   out_2508005545533897472[17] = 0;
   out_2508005545533897472[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2508005545533897472[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2508005545533897472[20] = 0;
   out_2508005545533897472[21] = state[8];
   out_2508005545533897472[22] = 0;
   out_2508005545533897472[23] = -state[6];
   out_2508005545533897472[24] = -state[5];
   out_2508005545533897472[25] = 0;
   out_2508005545533897472[26] = state[3];
   out_2508005545533897472[27] = 0;
   out_2508005545533897472[28] = 0;
   out_2508005545533897472[29] = 0;
   out_2508005545533897472[30] = 0;
   out_2508005545533897472[31] = 1;
   out_2508005545533897472[32] = 0;
   out_2508005545533897472[33] = 0;
   out_2508005545533897472[34] = 1;
   out_2508005545533897472[35] = 0;
   out_2508005545533897472[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2508005545533897472[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2508005545533897472[38] = 0;
   out_2508005545533897472[39] = -state[7];
   out_2508005545533897472[40] = state[6];
   out_2508005545533897472[41] = 0;
   out_2508005545533897472[42] = state[4];
   out_2508005545533897472[43] = -state[3];
   out_2508005545533897472[44] = 0;
   out_2508005545533897472[45] = 0;
   out_2508005545533897472[46] = 0;
   out_2508005545533897472[47] = 0;
   out_2508005545533897472[48] = 0;
   out_2508005545533897472[49] = 0;
   out_2508005545533897472[50] = 1;
   out_2508005545533897472[51] = 0;
   out_2508005545533897472[52] = 0;
   out_2508005545533897472[53] = 1;
}
void h_13(double *state, double *unused, double *out_2751864404123375741) {
   out_2751864404123375741[0] = state[3];
   out_2751864404123375741[1] = state[4];
   out_2751864404123375741[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1137198308840372514) {
   out_1137198308840372514[0] = 0;
   out_1137198308840372514[1] = 0;
   out_1137198308840372514[2] = 0;
   out_1137198308840372514[3] = 1;
   out_1137198308840372514[4] = 0;
   out_1137198308840372514[5] = 0;
   out_1137198308840372514[6] = 0;
   out_1137198308840372514[7] = 0;
   out_1137198308840372514[8] = 0;
   out_1137198308840372514[9] = 0;
   out_1137198308840372514[10] = 0;
   out_1137198308840372514[11] = 0;
   out_1137198308840372514[12] = 0;
   out_1137198308840372514[13] = 0;
   out_1137198308840372514[14] = 0;
   out_1137198308840372514[15] = 0;
   out_1137198308840372514[16] = 0;
   out_1137198308840372514[17] = 0;
   out_1137198308840372514[18] = 0;
   out_1137198308840372514[19] = 0;
   out_1137198308840372514[20] = 0;
   out_1137198308840372514[21] = 0;
   out_1137198308840372514[22] = 1;
   out_1137198308840372514[23] = 0;
   out_1137198308840372514[24] = 0;
   out_1137198308840372514[25] = 0;
   out_1137198308840372514[26] = 0;
   out_1137198308840372514[27] = 0;
   out_1137198308840372514[28] = 0;
   out_1137198308840372514[29] = 0;
   out_1137198308840372514[30] = 0;
   out_1137198308840372514[31] = 0;
   out_1137198308840372514[32] = 0;
   out_1137198308840372514[33] = 0;
   out_1137198308840372514[34] = 0;
   out_1137198308840372514[35] = 0;
   out_1137198308840372514[36] = 0;
   out_1137198308840372514[37] = 0;
   out_1137198308840372514[38] = 0;
   out_1137198308840372514[39] = 0;
   out_1137198308840372514[40] = 0;
   out_1137198308840372514[41] = 1;
   out_1137198308840372514[42] = 0;
   out_1137198308840372514[43] = 0;
   out_1137198308840372514[44] = 0;
   out_1137198308840372514[45] = 0;
   out_1137198308840372514[46] = 0;
   out_1137198308840372514[47] = 0;
   out_1137198308840372514[48] = 0;
   out_1137198308840372514[49] = 0;
   out_1137198308840372514[50] = 0;
   out_1137198308840372514[51] = 0;
   out_1137198308840372514[52] = 0;
   out_1137198308840372514[53] = 0;
}
void h_14(double *state, double *unused, double *out_2963815555134069202) {
   out_2963815555134069202[0] = state[6];
   out_2963815555134069202[1] = state[7];
   out_2963815555134069202[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1888165339847524242) {
   out_1888165339847524242[0] = 0;
   out_1888165339847524242[1] = 0;
   out_1888165339847524242[2] = 0;
   out_1888165339847524242[3] = 0;
   out_1888165339847524242[4] = 0;
   out_1888165339847524242[5] = 0;
   out_1888165339847524242[6] = 1;
   out_1888165339847524242[7] = 0;
   out_1888165339847524242[8] = 0;
   out_1888165339847524242[9] = 0;
   out_1888165339847524242[10] = 0;
   out_1888165339847524242[11] = 0;
   out_1888165339847524242[12] = 0;
   out_1888165339847524242[13] = 0;
   out_1888165339847524242[14] = 0;
   out_1888165339847524242[15] = 0;
   out_1888165339847524242[16] = 0;
   out_1888165339847524242[17] = 0;
   out_1888165339847524242[18] = 0;
   out_1888165339847524242[19] = 0;
   out_1888165339847524242[20] = 0;
   out_1888165339847524242[21] = 0;
   out_1888165339847524242[22] = 0;
   out_1888165339847524242[23] = 0;
   out_1888165339847524242[24] = 0;
   out_1888165339847524242[25] = 1;
   out_1888165339847524242[26] = 0;
   out_1888165339847524242[27] = 0;
   out_1888165339847524242[28] = 0;
   out_1888165339847524242[29] = 0;
   out_1888165339847524242[30] = 0;
   out_1888165339847524242[31] = 0;
   out_1888165339847524242[32] = 0;
   out_1888165339847524242[33] = 0;
   out_1888165339847524242[34] = 0;
   out_1888165339847524242[35] = 0;
   out_1888165339847524242[36] = 0;
   out_1888165339847524242[37] = 0;
   out_1888165339847524242[38] = 0;
   out_1888165339847524242[39] = 0;
   out_1888165339847524242[40] = 0;
   out_1888165339847524242[41] = 0;
   out_1888165339847524242[42] = 0;
   out_1888165339847524242[43] = 0;
   out_1888165339847524242[44] = 1;
   out_1888165339847524242[45] = 0;
   out_1888165339847524242[46] = 0;
   out_1888165339847524242[47] = 0;
   out_1888165339847524242[48] = 0;
   out_1888165339847524242[49] = 0;
   out_1888165339847524242[50] = 0;
   out_1888165339847524242[51] = 0;
   out_1888165339847524242[52] = 0;
   out_1888165339847524242[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_3812274923922881427) {
  err_fun(nom_x, delta_x, out_3812274923922881427);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3419763620583365922) {
  inv_err_fun(nom_x, true_x, out_3419763620583365922);
}
void pose_H_mod_fun(double *state, double *out_6689702553421276651) {
  H_mod_fun(state, out_6689702553421276651);
}
void pose_f_fun(double *state, double dt, double *out_446039340896779864) {
  f_fun(state,  dt, out_446039340896779864);
}
void pose_F_fun(double *state, double dt, double *out_642758158368098798) {
  F_fun(state,  dt, out_642758158368098798);
}
void pose_h_4(double *state, double *unused, double *out_7695501649787619215) {
  h_4(state, unused, out_7695501649787619215);
}
void pose_H_4(double *state, double *unused, double *out_2075075516491960287) {
  H_4(state, unused, out_2075075516491960287);
}
void pose_h_10(double *state, double *unused, double *out_289310091903726368) {
  h_10(state, unused, out_289310091903726368);
}
void pose_H_10(double *state, double *unused, double *out_2508005545533897472) {
  H_10(state, unused, out_2508005545533897472);
}
void pose_h_13(double *state, double *unused, double *out_2751864404123375741) {
  h_13(state, unused, out_2751864404123375741);
}
void pose_H_13(double *state, double *unused, double *out_1137198308840372514) {
  H_13(state, unused, out_1137198308840372514);
}
void pose_h_14(double *state, double *unused, double *out_2963815555134069202) {
  h_14(state, unused, out_2963815555134069202);
}
void pose_H_14(double *state, double *unused, double *out_1888165339847524242) {
  H_14(state, unused, out_1888165339847524242);
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
