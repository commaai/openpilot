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
void err_fun(double *nom_x, double *delta_x, double *out_409617722116612880) {
   out_409617722116612880[0] = delta_x[0] + nom_x[0];
   out_409617722116612880[1] = delta_x[1] + nom_x[1];
   out_409617722116612880[2] = delta_x[2] + nom_x[2];
   out_409617722116612880[3] = delta_x[3] + nom_x[3];
   out_409617722116612880[4] = delta_x[4] + nom_x[4];
   out_409617722116612880[5] = delta_x[5] + nom_x[5];
   out_409617722116612880[6] = delta_x[6] + nom_x[6];
   out_409617722116612880[7] = delta_x[7] + nom_x[7];
   out_409617722116612880[8] = delta_x[8] + nom_x[8];
   out_409617722116612880[9] = delta_x[9] + nom_x[9];
   out_409617722116612880[10] = delta_x[10] + nom_x[10];
   out_409617722116612880[11] = delta_x[11] + nom_x[11];
   out_409617722116612880[12] = delta_x[12] + nom_x[12];
   out_409617722116612880[13] = delta_x[13] + nom_x[13];
   out_409617722116612880[14] = delta_x[14] + nom_x[14];
   out_409617722116612880[15] = delta_x[15] + nom_x[15];
   out_409617722116612880[16] = delta_x[16] + nom_x[16];
   out_409617722116612880[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2261331234596504125) {
   out_2261331234596504125[0] = -nom_x[0] + true_x[0];
   out_2261331234596504125[1] = -nom_x[1] + true_x[1];
   out_2261331234596504125[2] = -nom_x[2] + true_x[2];
   out_2261331234596504125[3] = -nom_x[3] + true_x[3];
   out_2261331234596504125[4] = -nom_x[4] + true_x[4];
   out_2261331234596504125[5] = -nom_x[5] + true_x[5];
   out_2261331234596504125[6] = -nom_x[6] + true_x[6];
   out_2261331234596504125[7] = -nom_x[7] + true_x[7];
   out_2261331234596504125[8] = -nom_x[8] + true_x[8];
   out_2261331234596504125[9] = -nom_x[9] + true_x[9];
   out_2261331234596504125[10] = -nom_x[10] + true_x[10];
   out_2261331234596504125[11] = -nom_x[11] + true_x[11];
   out_2261331234596504125[12] = -nom_x[12] + true_x[12];
   out_2261331234596504125[13] = -nom_x[13] + true_x[13];
   out_2261331234596504125[14] = -nom_x[14] + true_x[14];
   out_2261331234596504125[15] = -nom_x[15] + true_x[15];
   out_2261331234596504125[16] = -nom_x[16] + true_x[16];
   out_2261331234596504125[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_8946862073519308870) {
   out_8946862073519308870[0] = 1.0;
   out_8946862073519308870[1] = 0.0;
   out_8946862073519308870[2] = 0.0;
   out_8946862073519308870[3] = 0.0;
   out_8946862073519308870[4] = 0.0;
   out_8946862073519308870[5] = 0.0;
   out_8946862073519308870[6] = 0.0;
   out_8946862073519308870[7] = 0.0;
   out_8946862073519308870[8] = 0.0;
   out_8946862073519308870[9] = 0.0;
   out_8946862073519308870[10] = 0.0;
   out_8946862073519308870[11] = 0.0;
   out_8946862073519308870[12] = 0.0;
   out_8946862073519308870[13] = 0.0;
   out_8946862073519308870[14] = 0.0;
   out_8946862073519308870[15] = 0.0;
   out_8946862073519308870[16] = 0.0;
   out_8946862073519308870[17] = 0.0;
   out_8946862073519308870[18] = 0.0;
   out_8946862073519308870[19] = 1.0;
   out_8946862073519308870[20] = 0.0;
   out_8946862073519308870[21] = 0.0;
   out_8946862073519308870[22] = 0.0;
   out_8946862073519308870[23] = 0.0;
   out_8946862073519308870[24] = 0.0;
   out_8946862073519308870[25] = 0.0;
   out_8946862073519308870[26] = 0.0;
   out_8946862073519308870[27] = 0.0;
   out_8946862073519308870[28] = 0.0;
   out_8946862073519308870[29] = 0.0;
   out_8946862073519308870[30] = 0.0;
   out_8946862073519308870[31] = 0.0;
   out_8946862073519308870[32] = 0.0;
   out_8946862073519308870[33] = 0.0;
   out_8946862073519308870[34] = 0.0;
   out_8946862073519308870[35] = 0.0;
   out_8946862073519308870[36] = 0.0;
   out_8946862073519308870[37] = 0.0;
   out_8946862073519308870[38] = 1.0;
   out_8946862073519308870[39] = 0.0;
   out_8946862073519308870[40] = 0.0;
   out_8946862073519308870[41] = 0.0;
   out_8946862073519308870[42] = 0.0;
   out_8946862073519308870[43] = 0.0;
   out_8946862073519308870[44] = 0.0;
   out_8946862073519308870[45] = 0.0;
   out_8946862073519308870[46] = 0.0;
   out_8946862073519308870[47] = 0.0;
   out_8946862073519308870[48] = 0.0;
   out_8946862073519308870[49] = 0.0;
   out_8946862073519308870[50] = 0.0;
   out_8946862073519308870[51] = 0.0;
   out_8946862073519308870[52] = 0.0;
   out_8946862073519308870[53] = 0.0;
   out_8946862073519308870[54] = 0.0;
   out_8946862073519308870[55] = 0.0;
   out_8946862073519308870[56] = 0.0;
   out_8946862073519308870[57] = 1.0;
   out_8946862073519308870[58] = 0.0;
   out_8946862073519308870[59] = 0.0;
   out_8946862073519308870[60] = 0.0;
   out_8946862073519308870[61] = 0.0;
   out_8946862073519308870[62] = 0.0;
   out_8946862073519308870[63] = 0.0;
   out_8946862073519308870[64] = 0.0;
   out_8946862073519308870[65] = 0.0;
   out_8946862073519308870[66] = 0.0;
   out_8946862073519308870[67] = 0.0;
   out_8946862073519308870[68] = 0.0;
   out_8946862073519308870[69] = 0.0;
   out_8946862073519308870[70] = 0.0;
   out_8946862073519308870[71] = 0.0;
   out_8946862073519308870[72] = 0.0;
   out_8946862073519308870[73] = 0.0;
   out_8946862073519308870[74] = 0.0;
   out_8946862073519308870[75] = 0.0;
   out_8946862073519308870[76] = 1.0;
   out_8946862073519308870[77] = 0.0;
   out_8946862073519308870[78] = 0.0;
   out_8946862073519308870[79] = 0.0;
   out_8946862073519308870[80] = 0.0;
   out_8946862073519308870[81] = 0.0;
   out_8946862073519308870[82] = 0.0;
   out_8946862073519308870[83] = 0.0;
   out_8946862073519308870[84] = 0.0;
   out_8946862073519308870[85] = 0.0;
   out_8946862073519308870[86] = 0.0;
   out_8946862073519308870[87] = 0.0;
   out_8946862073519308870[88] = 0.0;
   out_8946862073519308870[89] = 0.0;
   out_8946862073519308870[90] = 0.0;
   out_8946862073519308870[91] = 0.0;
   out_8946862073519308870[92] = 0.0;
   out_8946862073519308870[93] = 0.0;
   out_8946862073519308870[94] = 0.0;
   out_8946862073519308870[95] = 1.0;
   out_8946862073519308870[96] = 0.0;
   out_8946862073519308870[97] = 0.0;
   out_8946862073519308870[98] = 0.0;
   out_8946862073519308870[99] = 0.0;
   out_8946862073519308870[100] = 0.0;
   out_8946862073519308870[101] = 0.0;
   out_8946862073519308870[102] = 0.0;
   out_8946862073519308870[103] = 0.0;
   out_8946862073519308870[104] = 0.0;
   out_8946862073519308870[105] = 0.0;
   out_8946862073519308870[106] = 0.0;
   out_8946862073519308870[107] = 0.0;
   out_8946862073519308870[108] = 0.0;
   out_8946862073519308870[109] = 0.0;
   out_8946862073519308870[110] = 0.0;
   out_8946862073519308870[111] = 0.0;
   out_8946862073519308870[112] = 0.0;
   out_8946862073519308870[113] = 0.0;
   out_8946862073519308870[114] = 1.0;
   out_8946862073519308870[115] = 0.0;
   out_8946862073519308870[116] = 0.0;
   out_8946862073519308870[117] = 0.0;
   out_8946862073519308870[118] = 0.0;
   out_8946862073519308870[119] = 0.0;
   out_8946862073519308870[120] = 0.0;
   out_8946862073519308870[121] = 0.0;
   out_8946862073519308870[122] = 0.0;
   out_8946862073519308870[123] = 0.0;
   out_8946862073519308870[124] = 0.0;
   out_8946862073519308870[125] = 0.0;
   out_8946862073519308870[126] = 0.0;
   out_8946862073519308870[127] = 0.0;
   out_8946862073519308870[128] = 0.0;
   out_8946862073519308870[129] = 0.0;
   out_8946862073519308870[130] = 0.0;
   out_8946862073519308870[131] = 0.0;
   out_8946862073519308870[132] = 0.0;
   out_8946862073519308870[133] = 1.0;
   out_8946862073519308870[134] = 0.0;
   out_8946862073519308870[135] = 0.0;
   out_8946862073519308870[136] = 0.0;
   out_8946862073519308870[137] = 0.0;
   out_8946862073519308870[138] = 0.0;
   out_8946862073519308870[139] = 0.0;
   out_8946862073519308870[140] = 0.0;
   out_8946862073519308870[141] = 0.0;
   out_8946862073519308870[142] = 0.0;
   out_8946862073519308870[143] = 0.0;
   out_8946862073519308870[144] = 0.0;
   out_8946862073519308870[145] = 0.0;
   out_8946862073519308870[146] = 0.0;
   out_8946862073519308870[147] = 0.0;
   out_8946862073519308870[148] = 0.0;
   out_8946862073519308870[149] = 0.0;
   out_8946862073519308870[150] = 0.0;
   out_8946862073519308870[151] = 0.0;
   out_8946862073519308870[152] = 1.0;
   out_8946862073519308870[153] = 0.0;
   out_8946862073519308870[154] = 0.0;
   out_8946862073519308870[155] = 0.0;
   out_8946862073519308870[156] = 0.0;
   out_8946862073519308870[157] = 0.0;
   out_8946862073519308870[158] = 0.0;
   out_8946862073519308870[159] = 0.0;
   out_8946862073519308870[160] = 0.0;
   out_8946862073519308870[161] = 0.0;
   out_8946862073519308870[162] = 0.0;
   out_8946862073519308870[163] = 0.0;
   out_8946862073519308870[164] = 0.0;
   out_8946862073519308870[165] = 0.0;
   out_8946862073519308870[166] = 0.0;
   out_8946862073519308870[167] = 0.0;
   out_8946862073519308870[168] = 0.0;
   out_8946862073519308870[169] = 0.0;
   out_8946862073519308870[170] = 0.0;
   out_8946862073519308870[171] = 1.0;
   out_8946862073519308870[172] = 0.0;
   out_8946862073519308870[173] = 0.0;
   out_8946862073519308870[174] = 0.0;
   out_8946862073519308870[175] = 0.0;
   out_8946862073519308870[176] = 0.0;
   out_8946862073519308870[177] = 0.0;
   out_8946862073519308870[178] = 0.0;
   out_8946862073519308870[179] = 0.0;
   out_8946862073519308870[180] = 0.0;
   out_8946862073519308870[181] = 0.0;
   out_8946862073519308870[182] = 0.0;
   out_8946862073519308870[183] = 0.0;
   out_8946862073519308870[184] = 0.0;
   out_8946862073519308870[185] = 0.0;
   out_8946862073519308870[186] = 0.0;
   out_8946862073519308870[187] = 0.0;
   out_8946862073519308870[188] = 0.0;
   out_8946862073519308870[189] = 0.0;
   out_8946862073519308870[190] = 1.0;
   out_8946862073519308870[191] = 0.0;
   out_8946862073519308870[192] = 0.0;
   out_8946862073519308870[193] = 0.0;
   out_8946862073519308870[194] = 0.0;
   out_8946862073519308870[195] = 0.0;
   out_8946862073519308870[196] = 0.0;
   out_8946862073519308870[197] = 0.0;
   out_8946862073519308870[198] = 0.0;
   out_8946862073519308870[199] = 0.0;
   out_8946862073519308870[200] = 0.0;
   out_8946862073519308870[201] = 0.0;
   out_8946862073519308870[202] = 0.0;
   out_8946862073519308870[203] = 0.0;
   out_8946862073519308870[204] = 0.0;
   out_8946862073519308870[205] = 0.0;
   out_8946862073519308870[206] = 0.0;
   out_8946862073519308870[207] = 0.0;
   out_8946862073519308870[208] = 0.0;
   out_8946862073519308870[209] = 1.0;
   out_8946862073519308870[210] = 0.0;
   out_8946862073519308870[211] = 0.0;
   out_8946862073519308870[212] = 0.0;
   out_8946862073519308870[213] = 0.0;
   out_8946862073519308870[214] = 0.0;
   out_8946862073519308870[215] = 0.0;
   out_8946862073519308870[216] = 0.0;
   out_8946862073519308870[217] = 0.0;
   out_8946862073519308870[218] = 0.0;
   out_8946862073519308870[219] = 0.0;
   out_8946862073519308870[220] = 0.0;
   out_8946862073519308870[221] = 0.0;
   out_8946862073519308870[222] = 0.0;
   out_8946862073519308870[223] = 0.0;
   out_8946862073519308870[224] = 0.0;
   out_8946862073519308870[225] = 0.0;
   out_8946862073519308870[226] = 0.0;
   out_8946862073519308870[227] = 0.0;
   out_8946862073519308870[228] = 1.0;
   out_8946862073519308870[229] = 0.0;
   out_8946862073519308870[230] = 0.0;
   out_8946862073519308870[231] = 0.0;
   out_8946862073519308870[232] = 0.0;
   out_8946862073519308870[233] = 0.0;
   out_8946862073519308870[234] = 0.0;
   out_8946862073519308870[235] = 0.0;
   out_8946862073519308870[236] = 0.0;
   out_8946862073519308870[237] = 0.0;
   out_8946862073519308870[238] = 0.0;
   out_8946862073519308870[239] = 0.0;
   out_8946862073519308870[240] = 0.0;
   out_8946862073519308870[241] = 0.0;
   out_8946862073519308870[242] = 0.0;
   out_8946862073519308870[243] = 0.0;
   out_8946862073519308870[244] = 0.0;
   out_8946862073519308870[245] = 0.0;
   out_8946862073519308870[246] = 0.0;
   out_8946862073519308870[247] = 1.0;
   out_8946862073519308870[248] = 0.0;
   out_8946862073519308870[249] = 0.0;
   out_8946862073519308870[250] = 0.0;
   out_8946862073519308870[251] = 0.0;
   out_8946862073519308870[252] = 0.0;
   out_8946862073519308870[253] = 0.0;
   out_8946862073519308870[254] = 0.0;
   out_8946862073519308870[255] = 0.0;
   out_8946862073519308870[256] = 0.0;
   out_8946862073519308870[257] = 0.0;
   out_8946862073519308870[258] = 0.0;
   out_8946862073519308870[259] = 0.0;
   out_8946862073519308870[260] = 0.0;
   out_8946862073519308870[261] = 0.0;
   out_8946862073519308870[262] = 0.0;
   out_8946862073519308870[263] = 0.0;
   out_8946862073519308870[264] = 0.0;
   out_8946862073519308870[265] = 0.0;
   out_8946862073519308870[266] = 1.0;
   out_8946862073519308870[267] = 0.0;
   out_8946862073519308870[268] = 0.0;
   out_8946862073519308870[269] = 0.0;
   out_8946862073519308870[270] = 0.0;
   out_8946862073519308870[271] = 0.0;
   out_8946862073519308870[272] = 0.0;
   out_8946862073519308870[273] = 0.0;
   out_8946862073519308870[274] = 0.0;
   out_8946862073519308870[275] = 0.0;
   out_8946862073519308870[276] = 0.0;
   out_8946862073519308870[277] = 0.0;
   out_8946862073519308870[278] = 0.0;
   out_8946862073519308870[279] = 0.0;
   out_8946862073519308870[280] = 0.0;
   out_8946862073519308870[281] = 0.0;
   out_8946862073519308870[282] = 0.0;
   out_8946862073519308870[283] = 0.0;
   out_8946862073519308870[284] = 0.0;
   out_8946862073519308870[285] = 1.0;
   out_8946862073519308870[286] = 0.0;
   out_8946862073519308870[287] = 0.0;
   out_8946862073519308870[288] = 0.0;
   out_8946862073519308870[289] = 0.0;
   out_8946862073519308870[290] = 0.0;
   out_8946862073519308870[291] = 0.0;
   out_8946862073519308870[292] = 0.0;
   out_8946862073519308870[293] = 0.0;
   out_8946862073519308870[294] = 0.0;
   out_8946862073519308870[295] = 0.0;
   out_8946862073519308870[296] = 0.0;
   out_8946862073519308870[297] = 0.0;
   out_8946862073519308870[298] = 0.0;
   out_8946862073519308870[299] = 0.0;
   out_8946862073519308870[300] = 0.0;
   out_8946862073519308870[301] = 0.0;
   out_8946862073519308870[302] = 0.0;
   out_8946862073519308870[303] = 0.0;
   out_8946862073519308870[304] = 1.0;
   out_8946862073519308870[305] = 0.0;
   out_8946862073519308870[306] = 0.0;
   out_8946862073519308870[307] = 0.0;
   out_8946862073519308870[308] = 0.0;
   out_8946862073519308870[309] = 0.0;
   out_8946862073519308870[310] = 0.0;
   out_8946862073519308870[311] = 0.0;
   out_8946862073519308870[312] = 0.0;
   out_8946862073519308870[313] = 0.0;
   out_8946862073519308870[314] = 0.0;
   out_8946862073519308870[315] = 0.0;
   out_8946862073519308870[316] = 0.0;
   out_8946862073519308870[317] = 0.0;
   out_8946862073519308870[318] = 0.0;
   out_8946862073519308870[319] = 0.0;
   out_8946862073519308870[320] = 0.0;
   out_8946862073519308870[321] = 0.0;
   out_8946862073519308870[322] = 0.0;
   out_8946862073519308870[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5554883369010406028) {
   out_5554883369010406028[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5554883369010406028[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5554883369010406028[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5554883369010406028[3] = dt*state[12] + state[3];
   out_5554883369010406028[4] = dt*state[13] + state[4];
   out_5554883369010406028[5] = dt*state[14] + state[5];
   out_5554883369010406028[6] = state[6];
   out_5554883369010406028[7] = state[7];
   out_5554883369010406028[8] = state[8];
   out_5554883369010406028[9] = state[9];
   out_5554883369010406028[10] = state[10];
   out_5554883369010406028[11] = state[11];
   out_5554883369010406028[12] = state[12];
   out_5554883369010406028[13] = state[13];
   out_5554883369010406028[14] = state[14];
   out_5554883369010406028[15] = state[15];
   out_5554883369010406028[16] = state[16];
   out_5554883369010406028[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6041026006058863705) {
   out_6041026006058863705[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6041026006058863705[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6041026006058863705[2] = 0;
   out_6041026006058863705[3] = 0;
   out_6041026006058863705[4] = 0;
   out_6041026006058863705[5] = 0;
   out_6041026006058863705[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6041026006058863705[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6041026006058863705[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6041026006058863705[9] = 0;
   out_6041026006058863705[10] = 0;
   out_6041026006058863705[11] = 0;
   out_6041026006058863705[12] = 0;
   out_6041026006058863705[13] = 0;
   out_6041026006058863705[14] = 0;
   out_6041026006058863705[15] = 0;
   out_6041026006058863705[16] = 0;
   out_6041026006058863705[17] = 0;
   out_6041026006058863705[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6041026006058863705[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6041026006058863705[20] = 0;
   out_6041026006058863705[21] = 0;
   out_6041026006058863705[22] = 0;
   out_6041026006058863705[23] = 0;
   out_6041026006058863705[24] = 0;
   out_6041026006058863705[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6041026006058863705[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6041026006058863705[27] = 0;
   out_6041026006058863705[28] = 0;
   out_6041026006058863705[29] = 0;
   out_6041026006058863705[30] = 0;
   out_6041026006058863705[31] = 0;
   out_6041026006058863705[32] = 0;
   out_6041026006058863705[33] = 0;
   out_6041026006058863705[34] = 0;
   out_6041026006058863705[35] = 0;
   out_6041026006058863705[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6041026006058863705[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6041026006058863705[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6041026006058863705[39] = 0;
   out_6041026006058863705[40] = 0;
   out_6041026006058863705[41] = 0;
   out_6041026006058863705[42] = 0;
   out_6041026006058863705[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6041026006058863705[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6041026006058863705[45] = 0;
   out_6041026006058863705[46] = 0;
   out_6041026006058863705[47] = 0;
   out_6041026006058863705[48] = 0;
   out_6041026006058863705[49] = 0;
   out_6041026006058863705[50] = 0;
   out_6041026006058863705[51] = 0;
   out_6041026006058863705[52] = 0;
   out_6041026006058863705[53] = 0;
   out_6041026006058863705[54] = 0;
   out_6041026006058863705[55] = 0;
   out_6041026006058863705[56] = 0;
   out_6041026006058863705[57] = 1;
   out_6041026006058863705[58] = 0;
   out_6041026006058863705[59] = 0;
   out_6041026006058863705[60] = 0;
   out_6041026006058863705[61] = 0;
   out_6041026006058863705[62] = 0;
   out_6041026006058863705[63] = 0;
   out_6041026006058863705[64] = 0;
   out_6041026006058863705[65] = 0;
   out_6041026006058863705[66] = dt;
   out_6041026006058863705[67] = 0;
   out_6041026006058863705[68] = 0;
   out_6041026006058863705[69] = 0;
   out_6041026006058863705[70] = 0;
   out_6041026006058863705[71] = 0;
   out_6041026006058863705[72] = 0;
   out_6041026006058863705[73] = 0;
   out_6041026006058863705[74] = 0;
   out_6041026006058863705[75] = 0;
   out_6041026006058863705[76] = 1;
   out_6041026006058863705[77] = 0;
   out_6041026006058863705[78] = 0;
   out_6041026006058863705[79] = 0;
   out_6041026006058863705[80] = 0;
   out_6041026006058863705[81] = 0;
   out_6041026006058863705[82] = 0;
   out_6041026006058863705[83] = 0;
   out_6041026006058863705[84] = 0;
   out_6041026006058863705[85] = dt;
   out_6041026006058863705[86] = 0;
   out_6041026006058863705[87] = 0;
   out_6041026006058863705[88] = 0;
   out_6041026006058863705[89] = 0;
   out_6041026006058863705[90] = 0;
   out_6041026006058863705[91] = 0;
   out_6041026006058863705[92] = 0;
   out_6041026006058863705[93] = 0;
   out_6041026006058863705[94] = 0;
   out_6041026006058863705[95] = 1;
   out_6041026006058863705[96] = 0;
   out_6041026006058863705[97] = 0;
   out_6041026006058863705[98] = 0;
   out_6041026006058863705[99] = 0;
   out_6041026006058863705[100] = 0;
   out_6041026006058863705[101] = 0;
   out_6041026006058863705[102] = 0;
   out_6041026006058863705[103] = 0;
   out_6041026006058863705[104] = dt;
   out_6041026006058863705[105] = 0;
   out_6041026006058863705[106] = 0;
   out_6041026006058863705[107] = 0;
   out_6041026006058863705[108] = 0;
   out_6041026006058863705[109] = 0;
   out_6041026006058863705[110] = 0;
   out_6041026006058863705[111] = 0;
   out_6041026006058863705[112] = 0;
   out_6041026006058863705[113] = 0;
   out_6041026006058863705[114] = 1;
   out_6041026006058863705[115] = 0;
   out_6041026006058863705[116] = 0;
   out_6041026006058863705[117] = 0;
   out_6041026006058863705[118] = 0;
   out_6041026006058863705[119] = 0;
   out_6041026006058863705[120] = 0;
   out_6041026006058863705[121] = 0;
   out_6041026006058863705[122] = 0;
   out_6041026006058863705[123] = 0;
   out_6041026006058863705[124] = 0;
   out_6041026006058863705[125] = 0;
   out_6041026006058863705[126] = 0;
   out_6041026006058863705[127] = 0;
   out_6041026006058863705[128] = 0;
   out_6041026006058863705[129] = 0;
   out_6041026006058863705[130] = 0;
   out_6041026006058863705[131] = 0;
   out_6041026006058863705[132] = 0;
   out_6041026006058863705[133] = 1;
   out_6041026006058863705[134] = 0;
   out_6041026006058863705[135] = 0;
   out_6041026006058863705[136] = 0;
   out_6041026006058863705[137] = 0;
   out_6041026006058863705[138] = 0;
   out_6041026006058863705[139] = 0;
   out_6041026006058863705[140] = 0;
   out_6041026006058863705[141] = 0;
   out_6041026006058863705[142] = 0;
   out_6041026006058863705[143] = 0;
   out_6041026006058863705[144] = 0;
   out_6041026006058863705[145] = 0;
   out_6041026006058863705[146] = 0;
   out_6041026006058863705[147] = 0;
   out_6041026006058863705[148] = 0;
   out_6041026006058863705[149] = 0;
   out_6041026006058863705[150] = 0;
   out_6041026006058863705[151] = 0;
   out_6041026006058863705[152] = 1;
   out_6041026006058863705[153] = 0;
   out_6041026006058863705[154] = 0;
   out_6041026006058863705[155] = 0;
   out_6041026006058863705[156] = 0;
   out_6041026006058863705[157] = 0;
   out_6041026006058863705[158] = 0;
   out_6041026006058863705[159] = 0;
   out_6041026006058863705[160] = 0;
   out_6041026006058863705[161] = 0;
   out_6041026006058863705[162] = 0;
   out_6041026006058863705[163] = 0;
   out_6041026006058863705[164] = 0;
   out_6041026006058863705[165] = 0;
   out_6041026006058863705[166] = 0;
   out_6041026006058863705[167] = 0;
   out_6041026006058863705[168] = 0;
   out_6041026006058863705[169] = 0;
   out_6041026006058863705[170] = 0;
   out_6041026006058863705[171] = 1;
   out_6041026006058863705[172] = 0;
   out_6041026006058863705[173] = 0;
   out_6041026006058863705[174] = 0;
   out_6041026006058863705[175] = 0;
   out_6041026006058863705[176] = 0;
   out_6041026006058863705[177] = 0;
   out_6041026006058863705[178] = 0;
   out_6041026006058863705[179] = 0;
   out_6041026006058863705[180] = 0;
   out_6041026006058863705[181] = 0;
   out_6041026006058863705[182] = 0;
   out_6041026006058863705[183] = 0;
   out_6041026006058863705[184] = 0;
   out_6041026006058863705[185] = 0;
   out_6041026006058863705[186] = 0;
   out_6041026006058863705[187] = 0;
   out_6041026006058863705[188] = 0;
   out_6041026006058863705[189] = 0;
   out_6041026006058863705[190] = 1;
   out_6041026006058863705[191] = 0;
   out_6041026006058863705[192] = 0;
   out_6041026006058863705[193] = 0;
   out_6041026006058863705[194] = 0;
   out_6041026006058863705[195] = 0;
   out_6041026006058863705[196] = 0;
   out_6041026006058863705[197] = 0;
   out_6041026006058863705[198] = 0;
   out_6041026006058863705[199] = 0;
   out_6041026006058863705[200] = 0;
   out_6041026006058863705[201] = 0;
   out_6041026006058863705[202] = 0;
   out_6041026006058863705[203] = 0;
   out_6041026006058863705[204] = 0;
   out_6041026006058863705[205] = 0;
   out_6041026006058863705[206] = 0;
   out_6041026006058863705[207] = 0;
   out_6041026006058863705[208] = 0;
   out_6041026006058863705[209] = 1;
   out_6041026006058863705[210] = 0;
   out_6041026006058863705[211] = 0;
   out_6041026006058863705[212] = 0;
   out_6041026006058863705[213] = 0;
   out_6041026006058863705[214] = 0;
   out_6041026006058863705[215] = 0;
   out_6041026006058863705[216] = 0;
   out_6041026006058863705[217] = 0;
   out_6041026006058863705[218] = 0;
   out_6041026006058863705[219] = 0;
   out_6041026006058863705[220] = 0;
   out_6041026006058863705[221] = 0;
   out_6041026006058863705[222] = 0;
   out_6041026006058863705[223] = 0;
   out_6041026006058863705[224] = 0;
   out_6041026006058863705[225] = 0;
   out_6041026006058863705[226] = 0;
   out_6041026006058863705[227] = 0;
   out_6041026006058863705[228] = 1;
   out_6041026006058863705[229] = 0;
   out_6041026006058863705[230] = 0;
   out_6041026006058863705[231] = 0;
   out_6041026006058863705[232] = 0;
   out_6041026006058863705[233] = 0;
   out_6041026006058863705[234] = 0;
   out_6041026006058863705[235] = 0;
   out_6041026006058863705[236] = 0;
   out_6041026006058863705[237] = 0;
   out_6041026006058863705[238] = 0;
   out_6041026006058863705[239] = 0;
   out_6041026006058863705[240] = 0;
   out_6041026006058863705[241] = 0;
   out_6041026006058863705[242] = 0;
   out_6041026006058863705[243] = 0;
   out_6041026006058863705[244] = 0;
   out_6041026006058863705[245] = 0;
   out_6041026006058863705[246] = 0;
   out_6041026006058863705[247] = 1;
   out_6041026006058863705[248] = 0;
   out_6041026006058863705[249] = 0;
   out_6041026006058863705[250] = 0;
   out_6041026006058863705[251] = 0;
   out_6041026006058863705[252] = 0;
   out_6041026006058863705[253] = 0;
   out_6041026006058863705[254] = 0;
   out_6041026006058863705[255] = 0;
   out_6041026006058863705[256] = 0;
   out_6041026006058863705[257] = 0;
   out_6041026006058863705[258] = 0;
   out_6041026006058863705[259] = 0;
   out_6041026006058863705[260] = 0;
   out_6041026006058863705[261] = 0;
   out_6041026006058863705[262] = 0;
   out_6041026006058863705[263] = 0;
   out_6041026006058863705[264] = 0;
   out_6041026006058863705[265] = 0;
   out_6041026006058863705[266] = 1;
   out_6041026006058863705[267] = 0;
   out_6041026006058863705[268] = 0;
   out_6041026006058863705[269] = 0;
   out_6041026006058863705[270] = 0;
   out_6041026006058863705[271] = 0;
   out_6041026006058863705[272] = 0;
   out_6041026006058863705[273] = 0;
   out_6041026006058863705[274] = 0;
   out_6041026006058863705[275] = 0;
   out_6041026006058863705[276] = 0;
   out_6041026006058863705[277] = 0;
   out_6041026006058863705[278] = 0;
   out_6041026006058863705[279] = 0;
   out_6041026006058863705[280] = 0;
   out_6041026006058863705[281] = 0;
   out_6041026006058863705[282] = 0;
   out_6041026006058863705[283] = 0;
   out_6041026006058863705[284] = 0;
   out_6041026006058863705[285] = 1;
   out_6041026006058863705[286] = 0;
   out_6041026006058863705[287] = 0;
   out_6041026006058863705[288] = 0;
   out_6041026006058863705[289] = 0;
   out_6041026006058863705[290] = 0;
   out_6041026006058863705[291] = 0;
   out_6041026006058863705[292] = 0;
   out_6041026006058863705[293] = 0;
   out_6041026006058863705[294] = 0;
   out_6041026006058863705[295] = 0;
   out_6041026006058863705[296] = 0;
   out_6041026006058863705[297] = 0;
   out_6041026006058863705[298] = 0;
   out_6041026006058863705[299] = 0;
   out_6041026006058863705[300] = 0;
   out_6041026006058863705[301] = 0;
   out_6041026006058863705[302] = 0;
   out_6041026006058863705[303] = 0;
   out_6041026006058863705[304] = 1;
   out_6041026006058863705[305] = 0;
   out_6041026006058863705[306] = 0;
   out_6041026006058863705[307] = 0;
   out_6041026006058863705[308] = 0;
   out_6041026006058863705[309] = 0;
   out_6041026006058863705[310] = 0;
   out_6041026006058863705[311] = 0;
   out_6041026006058863705[312] = 0;
   out_6041026006058863705[313] = 0;
   out_6041026006058863705[314] = 0;
   out_6041026006058863705[315] = 0;
   out_6041026006058863705[316] = 0;
   out_6041026006058863705[317] = 0;
   out_6041026006058863705[318] = 0;
   out_6041026006058863705[319] = 0;
   out_6041026006058863705[320] = 0;
   out_6041026006058863705[321] = 0;
   out_6041026006058863705[322] = 0;
   out_6041026006058863705[323] = 1;
}
void h_4(double *state, double *unused, double *out_2384878000485610448) {
   out_2384878000485610448[0] = state[6] + state[9];
   out_2384878000485610448[1] = state[7] + state[10];
   out_2384878000485610448[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_761685768107253608) {
   out_761685768107253608[0] = 0;
   out_761685768107253608[1] = 0;
   out_761685768107253608[2] = 0;
   out_761685768107253608[3] = 0;
   out_761685768107253608[4] = 0;
   out_761685768107253608[5] = 0;
   out_761685768107253608[6] = 1;
   out_761685768107253608[7] = 0;
   out_761685768107253608[8] = 0;
   out_761685768107253608[9] = 1;
   out_761685768107253608[10] = 0;
   out_761685768107253608[11] = 0;
   out_761685768107253608[12] = 0;
   out_761685768107253608[13] = 0;
   out_761685768107253608[14] = 0;
   out_761685768107253608[15] = 0;
   out_761685768107253608[16] = 0;
   out_761685768107253608[17] = 0;
   out_761685768107253608[18] = 0;
   out_761685768107253608[19] = 0;
   out_761685768107253608[20] = 0;
   out_761685768107253608[21] = 0;
   out_761685768107253608[22] = 0;
   out_761685768107253608[23] = 0;
   out_761685768107253608[24] = 0;
   out_761685768107253608[25] = 1;
   out_761685768107253608[26] = 0;
   out_761685768107253608[27] = 0;
   out_761685768107253608[28] = 1;
   out_761685768107253608[29] = 0;
   out_761685768107253608[30] = 0;
   out_761685768107253608[31] = 0;
   out_761685768107253608[32] = 0;
   out_761685768107253608[33] = 0;
   out_761685768107253608[34] = 0;
   out_761685768107253608[35] = 0;
   out_761685768107253608[36] = 0;
   out_761685768107253608[37] = 0;
   out_761685768107253608[38] = 0;
   out_761685768107253608[39] = 0;
   out_761685768107253608[40] = 0;
   out_761685768107253608[41] = 0;
   out_761685768107253608[42] = 0;
   out_761685768107253608[43] = 0;
   out_761685768107253608[44] = 1;
   out_761685768107253608[45] = 0;
   out_761685768107253608[46] = 0;
   out_761685768107253608[47] = 1;
   out_761685768107253608[48] = 0;
   out_761685768107253608[49] = 0;
   out_761685768107253608[50] = 0;
   out_761685768107253608[51] = 0;
   out_761685768107253608[52] = 0;
   out_761685768107253608[53] = 0;
}
void h_10(double *state, double *unused, double *out_1692252871197538134) {
   out_1692252871197538134[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_1692252871197538134[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_1692252871197538134[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4918497349089969484) {
   out_4918497349089969484[0] = 0;
   out_4918497349089969484[1] = 9.8100000000000005*cos(state[1]);
   out_4918497349089969484[2] = 0;
   out_4918497349089969484[3] = 0;
   out_4918497349089969484[4] = -state[8];
   out_4918497349089969484[5] = state[7];
   out_4918497349089969484[6] = 0;
   out_4918497349089969484[7] = state[5];
   out_4918497349089969484[8] = -state[4];
   out_4918497349089969484[9] = 0;
   out_4918497349089969484[10] = 0;
   out_4918497349089969484[11] = 0;
   out_4918497349089969484[12] = 1;
   out_4918497349089969484[13] = 0;
   out_4918497349089969484[14] = 0;
   out_4918497349089969484[15] = 1;
   out_4918497349089969484[16] = 0;
   out_4918497349089969484[17] = 0;
   out_4918497349089969484[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4918497349089969484[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4918497349089969484[20] = 0;
   out_4918497349089969484[21] = state[8];
   out_4918497349089969484[22] = 0;
   out_4918497349089969484[23] = -state[6];
   out_4918497349089969484[24] = -state[5];
   out_4918497349089969484[25] = 0;
   out_4918497349089969484[26] = state[3];
   out_4918497349089969484[27] = 0;
   out_4918497349089969484[28] = 0;
   out_4918497349089969484[29] = 0;
   out_4918497349089969484[30] = 0;
   out_4918497349089969484[31] = 1;
   out_4918497349089969484[32] = 0;
   out_4918497349089969484[33] = 0;
   out_4918497349089969484[34] = 1;
   out_4918497349089969484[35] = 0;
   out_4918497349089969484[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4918497349089969484[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4918497349089969484[38] = 0;
   out_4918497349089969484[39] = -state[7];
   out_4918497349089969484[40] = state[6];
   out_4918497349089969484[41] = 0;
   out_4918497349089969484[42] = state[4];
   out_4918497349089969484[43] = -state[3];
   out_4918497349089969484[44] = 0;
   out_4918497349089969484[45] = 0;
   out_4918497349089969484[46] = 0;
   out_4918497349089969484[47] = 0;
   out_4918497349089969484[48] = 0;
   out_4918497349089969484[49] = 0;
   out_4918497349089969484[50] = 1;
   out_4918497349089969484[51] = 0;
   out_4918497349089969484[52] = 0;
   out_4918497349089969484[53] = 1;
}
void h_13(double *state, double *unused, double *out_3349492577274348733) {
   out_3349492577274348733[0] = state[3];
   out_3349492577274348733[1] = state[4];
   out_3349492577274348733[2] = state[5];
}
void H_13(double *state, double *unused, double *out_2450588057225079193) {
   out_2450588057225079193[0] = 0;
   out_2450588057225079193[1] = 0;
   out_2450588057225079193[2] = 0;
   out_2450588057225079193[3] = 1;
   out_2450588057225079193[4] = 0;
   out_2450588057225079193[5] = 0;
   out_2450588057225079193[6] = 0;
   out_2450588057225079193[7] = 0;
   out_2450588057225079193[8] = 0;
   out_2450588057225079193[9] = 0;
   out_2450588057225079193[10] = 0;
   out_2450588057225079193[11] = 0;
   out_2450588057225079193[12] = 0;
   out_2450588057225079193[13] = 0;
   out_2450588057225079193[14] = 0;
   out_2450588057225079193[15] = 0;
   out_2450588057225079193[16] = 0;
   out_2450588057225079193[17] = 0;
   out_2450588057225079193[18] = 0;
   out_2450588057225079193[19] = 0;
   out_2450588057225079193[20] = 0;
   out_2450588057225079193[21] = 0;
   out_2450588057225079193[22] = 1;
   out_2450588057225079193[23] = 0;
   out_2450588057225079193[24] = 0;
   out_2450588057225079193[25] = 0;
   out_2450588057225079193[26] = 0;
   out_2450588057225079193[27] = 0;
   out_2450588057225079193[28] = 0;
   out_2450588057225079193[29] = 0;
   out_2450588057225079193[30] = 0;
   out_2450588057225079193[31] = 0;
   out_2450588057225079193[32] = 0;
   out_2450588057225079193[33] = 0;
   out_2450588057225079193[34] = 0;
   out_2450588057225079193[35] = 0;
   out_2450588057225079193[36] = 0;
   out_2450588057225079193[37] = 0;
   out_2450588057225079193[38] = 0;
   out_2450588057225079193[39] = 0;
   out_2450588057225079193[40] = 0;
   out_2450588057225079193[41] = 1;
   out_2450588057225079193[42] = 0;
   out_2450588057225079193[43] = 0;
   out_2450588057225079193[44] = 0;
   out_2450588057225079193[45] = 0;
   out_2450588057225079193[46] = 0;
   out_2450588057225079193[47] = 0;
   out_2450588057225079193[48] = 0;
   out_2450588057225079193[49] = 0;
   out_2450588057225079193[50] = 0;
   out_2450588057225079193[51] = 0;
   out_2450588057225079193[52] = 0;
   out_2450588057225079193[53] = 0;
}
void h_14(double *state, double *unused, double *out_5910242476345995022) {
   out_5910242476345995022[0] = state[6];
   out_5910242476345995022[1] = state[7];
   out_5910242476345995022[2] = state[8];
}
void H_14(double *state, double *unused, double *out_8242831583386994032) {
   out_8242831583386994032[0] = 0;
   out_8242831583386994032[1] = 0;
   out_8242831583386994032[2] = 0;
   out_8242831583386994032[3] = 0;
   out_8242831583386994032[4] = 0;
   out_8242831583386994032[5] = 0;
   out_8242831583386994032[6] = 1;
   out_8242831583386994032[7] = 0;
   out_8242831583386994032[8] = 0;
   out_8242831583386994032[9] = 0;
   out_8242831583386994032[10] = 0;
   out_8242831583386994032[11] = 0;
   out_8242831583386994032[12] = 0;
   out_8242831583386994032[13] = 0;
   out_8242831583386994032[14] = 0;
   out_8242831583386994032[15] = 0;
   out_8242831583386994032[16] = 0;
   out_8242831583386994032[17] = 0;
   out_8242831583386994032[18] = 0;
   out_8242831583386994032[19] = 0;
   out_8242831583386994032[20] = 0;
   out_8242831583386994032[21] = 0;
   out_8242831583386994032[22] = 0;
   out_8242831583386994032[23] = 0;
   out_8242831583386994032[24] = 0;
   out_8242831583386994032[25] = 1;
   out_8242831583386994032[26] = 0;
   out_8242831583386994032[27] = 0;
   out_8242831583386994032[28] = 0;
   out_8242831583386994032[29] = 0;
   out_8242831583386994032[30] = 0;
   out_8242831583386994032[31] = 0;
   out_8242831583386994032[32] = 0;
   out_8242831583386994032[33] = 0;
   out_8242831583386994032[34] = 0;
   out_8242831583386994032[35] = 0;
   out_8242831583386994032[36] = 0;
   out_8242831583386994032[37] = 0;
   out_8242831583386994032[38] = 0;
   out_8242831583386994032[39] = 0;
   out_8242831583386994032[40] = 0;
   out_8242831583386994032[41] = 0;
   out_8242831583386994032[42] = 0;
   out_8242831583386994032[43] = 0;
   out_8242831583386994032[44] = 1;
   out_8242831583386994032[45] = 0;
   out_8242831583386994032[46] = 0;
   out_8242831583386994032[47] = 0;
   out_8242831583386994032[48] = 0;
   out_8242831583386994032[49] = 0;
   out_8242831583386994032[50] = 0;
   out_8242831583386994032[51] = 0;
   out_8242831583386994032[52] = 0;
   out_8242831583386994032[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_409617722116612880) {
  err_fun(nom_x, delta_x, out_409617722116612880);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2261331234596504125) {
  inv_err_fun(nom_x, true_x, out_2261331234596504125);
}
void pose_H_mod_fun(double *state, double *out_8946862073519308870) {
  H_mod_fun(state, out_8946862073519308870);
}
void pose_f_fun(double *state, double dt, double *out_5554883369010406028) {
  f_fun(state,  dt, out_5554883369010406028);
}
void pose_F_fun(double *state, double dt, double *out_6041026006058863705) {
  F_fun(state,  dt, out_6041026006058863705);
}
void pose_h_4(double *state, double *unused, double *out_2384878000485610448) {
  h_4(state, unused, out_2384878000485610448);
}
void pose_H_4(double *state, double *unused, double *out_761685768107253608) {
  H_4(state, unused, out_761685768107253608);
}
void pose_h_10(double *state, double *unused, double *out_1692252871197538134) {
  h_10(state, unused, out_1692252871197538134);
}
void pose_H_10(double *state, double *unused, double *out_4918497349089969484) {
  H_10(state, unused, out_4918497349089969484);
}
void pose_h_13(double *state, double *unused, double *out_3349492577274348733) {
  h_13(state, unused, out_3349492577274348733);
}
void pose_H_13(double *state, double *unused, double *out_2450588057225079193) {
  H_13(state, unused, out_2450588057225079193);
}
void pose_h_14(double *state, double *unused, double *out_5910242476345995022) {
  h_14(state, unused, out_5910242476345995022);
}
void pose_H_14(double *state, double *unused, double *out_8242831583386994032) {
  H_14(state, unused, out_8242831583386994032);
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
