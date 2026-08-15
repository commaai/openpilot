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
void err_fun(double *nom_x, double *delta_x, double *out_1364756071930907291) {
   out_1364756071930907291[0] = delta_x[0] + nom_x[0];
   out_1364756071930907291[1] = delta_x[1] + nom_x[1];
   out_1364756071930907291[2] = delta_x[2] + nom_x[2];
   out_1364756071930907291[3] = delta_x[3] + nom_x[3];
   out_1364756071930907291[4] = delta_x[4] + nom_x[4];
   out_1364756071930907291[5] = delta_x[5] + nom_x[5];
   out_1364756071930907291[6] = delta_x[6] + nom_x[6];
   out_1364756071930907291[7] = delta_x[7] + nom_x[7];
   out_1364756071930907291[8] = delta_x[8] + nom_x[8];
   out_1364756071930907291[9] = delta_x[9] + nom_x[9];
   out_1364756071930907291[10] = delta_x[10] + nom_x[10];
   out_1364756071930907291[11] = delta_x[11] + nom_x[11];
   out_1364756071930907291[12] = delta_x[12] + nom_x[12];
   out_1364756071930907291[13] = delta_x[13] + nom_x[13];
   out_1364756071930907291[14] = delta_x[14] + nom_x[14];
   out_1364756071930907291[15] = delta_x[15] + nom_x[15];
   out_1364756071930907291[16] = delta_x[16] + nom_x[16];
   out_1364756071930907291[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8303365725516605546) {
   out_8303365725516605546[0] = -nom_x[0] + true_x[0];
   out_8303365725516605546[1] = -nom_x[1] + true_x[1];
   out_8303365725516605546[2] = -nom_x[2] + true_x[2];
   out_8303365725516605546[3] = -nom_x[3] + true_x[3];
   out_8303365725516605546[4] = -nom_x[4] + true_x[4];
   out_8303365725516605546[5] = -nom_x[5] + true_x[5];
   out_8303365725516605546[6] = -nom_x[6] + true_x[6];
   out_8303365725516605546[7] = -nom_x[7] + true_x[7];
   out_8303365725516605546[8] = -nom_x[8] + true_x[8];
   out_8303365725516605546[9] = -nom_x[9] + true_x[9];
   out_8303365725516605546[10] = -nom_x[10] + true_x[10];
   out_8303365725516605546[11] = -nom_x[11] + true_x[11];
   out_8303365725516605546[12] = -nom_x[12] + true_x[12];
   out_8303365725516605546[13] = -nom_x[13] + true_x[13];
   out_8303365725516605546[14] = -nom_x[14] + true_x[14];
   out_8303365725516605546[15] = -nom_x[15] + true_x[15];
   out_8303365725516605546[16] = -nom_x[16] + true_x[16];
   out_8303365725516605546[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3322466727120633125) {
   out_3322466727120633125[0] = 1.0;
   out_3322466727120633125[1] = 0.0;
   out_3322466727120633125[2] = 0.0;
   out_3322466727120633125[3] = 0.0;
   out_3322466727120633125[4] = 0.0;
   out_3322466727120633125[5] = 0.0;
   out_3322466727120633125[6] = 0.0;
   out_3322466727120633125[7] = 0.0;
   out_3322466727120633125[8] = 0.0;
   out_3322466727120633125[9] = 0.0;
   out_3322466727120633125[10] = 0.0;
   out_3322466727120633125[11] = 0.0;
   out_3322466727120633125[12] = 0.0;
   out_3322466727120633125[13] = 0.0;
   out_3322466727120633125[14] = 0.0;
   out_3322466727120633125[15] = 0.0;
   out_3322466727120633125[16] = 0.0;
   out_3322466727120633125[17] = 0.0;
   out_3322466727120633125[18] = 0.0;
   out_3322466727120633125[19] = 1.0;
   out_3322466727120633125[20] = 0.0;
   out_3322466727120633125[21] = 0.0;
   out_3322466727120633125[22] = 0.0;
   out_3322466727120633125[23] = 0.0;
   out_3322466727120633125[24] = 0.0;
   out_3322466727120633125[25] = 0.0;
   out_3322466727120633125[26] = 0.0;
   out_3322466727120633125[27] = 0.0;
   out_3322466727120633125[28] = 0.0;
   out_3322466727120633125[29] = 0.0;
   out_3322466727120633125[30] = 0.0;
   out_3322466727120633125[31] = 0.0;
   out_3322466727120633125[32] = 0.0;
   out_3322466727120633125[33] = 0.0;
   out_3322466727120633125[34] = 0.0;
   out_3322466727120633125[35] = 0.0;
   out_3322466727120633125[36] = 0.0;
   out_3322466727120633125[37] = 0.0;
   out_3322466727120633125[38] = 1.0;
   out_3322466727120633125[39] = 0.0;
   out_3322466727120633125[40] = 0.0;
   out_3322466727120633125[41] = 0.0;
   out_3322466727120633125[42] = 0.0;
   out_3322466727120633125[43] = 0.0;
   out_3322466727120633125[44] = 0.0;
   out_3322466727120633125[45] = 0.0;
   out_3322466727120633125[46] = 0.0;
   out_3322466727120633125[47] = 0.0;
   out_3322466727120633125[48] = 0.0;
   out_3322466727120633125[49] = 0.0;
   out_3322466727120633125[50] = 0.0;
   out_3322466727120633125[51] = 0.0;
   out_3322466727120633125[52] = 0.0;
   out_3322466727120633125[53] = 0.0;
   out_3322466727120633125[54] = 0.0;
   out_3322466727120633125[55] = 0.0;
   out_3322466727120633125[56] = 0.0;
   out_3322466727120633125[57] = 1.0;
   out_3322466727120633125[58] = 0.0;
   out_3322466727120633125[59] = 0.0;
   out_3322466727120633125[60] = 0.0;
   out_3322466727120633125[61] = 0.0;
   out_3322466727120633125[62] = 0.0;
   out_3322466727120633125[63] = 0.0;
   out_3322466727120633125[64] = 0.0;
   out_3322466727120633125[65] = 0.0;
   out_3322466727120633125[66] = 0.0;
   out_3322466727120633125[67] = 0.0;
   out_3322466727120633125[68] = 0.0;
   out_3322466727120633125[69] = 0.0;
   out_3322466727120633125[70] = 0.0;
   out_3322466727120633125[71] = 0.0;
   out_3322466727120633125[72] = 0.0;
   out_3322466727120633125[73] = 0.0;
   out_3322466727120633125[74] = 0.0;
   out_3322466727120633125[75] = 0.0;
   out_3322466727120633125[76] = 1.0;
   out_3322466727120633125[77] = 0.0;
   out_3322466727120633125[78] = 0.0;
   out_3322466727120633125[79] = 0.0;
   out_3322466727120633125[80] = 0.0;
   out_3322466727120633125[81] = 0.0;
   out_3322466727120633125[82] = 0.0;
   out_3322466727120633125[83] = 0.0;
   out_3322466727120633125[84] = 0.0;
   out_3322466727120633125[85] = 0.0;
   out_3322466727120633125[86] = 0.0;
   out_3322466727120633125[87] = 0.0;
   out_3322466727120633125[88] = 0.0;
   out_3322466727120633125[89] = 0.0;
   out_3322466727120633125[90] = 0.0;
   out_3322466727120633125[91] = 0.0;
   out_3322466727120633125[92] = 0.0;
   out_3322466727120633125[93] = 0.0;
   out_3322466727120633125[94] = 0.0;
   out_3322466727120633125[95] = 1.0;
   out_3322466727120633125[96] = 0.0;
   out_3322466727120633125[97] = 0.0;
   out_3322466727120633125[98] = 0.0;
   out_3322466727120633125[99] = 0.0;
   out_3322466727120633125[100] = 0.0;
   out_3322466727120633125[101] = 0.0;
   out_3322466727120633125[102] = 0.0;
   out_3322466727120633125[103] = 0.0;
   out_3322466727120633125[104] = 0.0;
   out_3322466727120633125[105] = 0.0;
   out_3322466727120633125[106] = 0.0;
   out_3322466727120633125[107] = 0.0;
   out_3322466727120633125[108] = 0.0;
   out_3322466727120633125[109] = 0.0;
   out_3322466727120633125[110] = 0.0;
   out_3322466727120633125[111] = 0.0;
   out_3322466727120633125[112] = 0.0;
   out_3322466727120633125[113] = 0.0;
   out_3322466727120633125[114] = 1.0;
   out_3322466727120633125[115] = 0.0;
   out_3322466727120633125[116] = 0.0;
   out_3322466727120633125[117] = 0.0;
   out_3322466727120633125[118] = 0.0;
   out_3322466727120633125[119] = 0.0;
   out_3322466727120633125[120] = 0.0;
   out_3322466727120633125[121] = 0.0;
   out_3322466727120633125[122] = 0.0;
   out_3322466727120633125[123] = 0.0;
   out_3322466727120633125[124] = 0.0;
   out_3322466727120633125[125] = 0.0;
   out_3322466727120633125[126] = 0.0;
   out_3322466727120633125[127] = 0.0;
   out_3322466727120633125[128] = 0.0;
   out_3322466727120633125[129] = 0.0;
   out_3322466727120633125[130] = 0.0;
   out_3322466727120633125[131] = 0.0;
   out_3322466727120633125[132] = 0.0;
   out_3322466727120633125[133] = 1.0;
   out_3322466727120633125[134] = 0.0;
   out_3322466727120633125[135] = 0.0;
   out_3322466727120633125[136] = 0.0;
   out_3322466727120633125[137] = 0.0;
   out_3322466727120633125[138] = 0.0;
   out_3322466727120633125[139] = 0.0;
   out_3322466727120633125[140] = 0.0;
   out_3322466727120633125[141] = 0.0;
   out_3322466727120633125[142] = 0.0;
   out_3322466727120633125[143] = 0.0;
   out_3322466727120633125[144] = 0.0;
   out_3322466727120633125[145] = 0.0;
   out_3322466727120633125[146] = 0.0;
   out_3322466727120633125[147] = 0.0;
   out_3322466727120633125[148] = 0.0;
   out_3322466727120633125[149] = 0.0;
   out_3322466727120633125[150] = 0.0;
   out_3322466727120633125[151] = 0.0;
   out_3322466727120633125[152] = 1.0;
   out_3322466727120633125[153] = 0.0;
   out_3322466727120633125[154] = 0.0;
   out_3322466727120633125[155] = 0.0;
   out_3322466727120633125[156] = 0.0;
   out_3322466727120633125[157] = 0.0;
   out_3322466727120633125[158] = 0.0;
   out_3322466727120633125[159] = 0.0;
   out_3322466727120633125[160] = 0.0;
   out_3322466727120633125[161] = 0.0;
   out_3322466727120633125[162] = 0.0;
   out_3322466727120633125[163] = 0.0;
   out_3322466727120633125[164] = 0.0;
   out_3322466727120633125[165] = 0.0;
   out_3322466727120633125[166] = 0.0;
   out_3322466727120633125[167] = 0.0;
   out_3322466727120633125[168] = 0.0;
   out_3322466727120633125[169] = 0.0;
   out_3322466727120633125[170] = 0.0;
   out_3322466727120633125[171] = 1.0;
   out_3322466727120633125[172] = 0.0;
   out_3322466727120633125[173] = 0.0;
   out_3322466727120633125[174] = 0.0;
   out_3322466727120633125[175] = 0.0;
   out_3322466727120633125[176] = 0.0;
   out_3322466727120633125[177] = 0.0;
   out_3322466727120633125[178] = 0.0;
   out_3322466727120633125[179] = 0.0;
   out_3322466727120633125[180] = 0.0;
   out_3322466727120633125[181] = 0.0;
   out_3322466727120633125[182] = 0.0;
   out_3322466727120633125[183] = 0.0;
   out_3322466727120633125[184] = 0.0;
   out_3322466727120633125[185] = 0.0;
   out_3322466727120633125[186] = 0.0;
   out_3322466727120633125[187] = 0.0;
   out_3322466727120633125[188] = 0.0;
   out_3322466727120633125[189] = 0.0;
   out_3322466727120633125[190] = 1.0;
   out_3322466727120633125[191] = 0.0;
   out_3322466727120633125[192] = 0.0;
   out_3322466727120633125[193] = 0.0;
   out_3322466727120633125[194] = 0.0;
   out_3322466727120633125[195] = 0.0;
   out_3322466727120633125[196] = 0.0;
   out_3322466727120633125[197] = 0.0;
   out_3322466727120633125[198] = 0.0;
   out_3322466727120633125[199] = 0.0;
   out_3322466727120633125[200] = 0.0;
   out_3322466727120633125[201] = 0.0;
   out_3322466727120633125[202] = 0.0;
   out_3322466727120633125[203] = 0.0;
   out_3322466727120633125[204] = 0.0;
   out_3322466727120633125[205] = 0.0;
   out_3322466727120633125[206] = 0.0;
   out_3322466727120633125[207] = 0.0;
   out_3322466727120633125[208] = 0.0;
   out_3322466727120633125[209] = 1.0;
   out_3322466727120633125[210] = 0.0;
   out_3322466727120633125[211] = 0.0;
   out_3322466727120633125[212] = 0.0;
   out_3322466727120633125[213] = 0.0;
   out_3322466727120633125[214] = 0.0;
   out_3322466727120633125[215] = 0.0;
   out_3322466727120633125[216] = 0.0;
   out_3322466727120633125[217] = 0.0;
   out_3322466727120633125[218] = 0.0;
   out_3322466727120633125[219] = 0.0;
   out_3322466727120633125[220] = 0.0;
   out_3322466727120633125[221] = 0.0;
   out_3322466727120633125[222] = 0.0;
   out_3322466727120633125[223] = 0.0;
   out_3322466727120633125[224] = 0.0;
   out_3322466727120633125[225] = 0.0;
   out_3322466727120633125[226] = 0.0;
   out_3322466727120633125[227] = 0.0;
   out_3322466727120633125[228] = 1.0;
   out_3322466727120633125[229] = 0.0;
   out_3322466727120633125[230] = 0.0;
   out_3322466727120633125[231] = 0.0;
   out_3322466727120633125[232] = 0.0;
   out_3322466727120633125[233] = 0.0;
   out_3322466727120633125[234] = 0.0;
   out_3322466727120633125[235] = 0.0;
   out_3322466727120633125[236] = 0.0;
   out_3322466727120633125[237] = 0.0;
   out_3322466727120633125[238] = 0.0;
   out_3322466727120633125[239] = 0.0;
   out_3322466727120633125[240] = 0.0;
   out_3322466727120633125[241] = 0.0;
   out_3322466727120633125[242] = 0.0;
   out_3322466727120633125[243] = 0.0;
   out_3322466727120633125[244] = 0.0;
   out_3322466727120633125[245] = 0.0;
   out_3322466727120633125[246] = 0.0;
   out_3322466727120633125[247] = 1.0;
   out_3322466727120633125[248] = 0.0;
   out_3322466727120633125[249] = 0.0;
   out_3322466727120633125[250] = 0.0;
   out_3322466727120633125[251] = 0.0;
   out_3322466727120633125[252] = 0.0;
   out_3322466727120633125[253] = 0.0;
   out_3322466727120633125[254] = 0.0;
   out_3322466727120633125[255] = 0.0;
   out_3322466727120633125[256] = 0.0;
   out_3322466727120633125[257] = 0.0;
   out_3322466727120633125[258] = 0.0;
   out_3322466727120633125[259] = 0.0;
   out_3322466727120633125[260] = 0.0;
   out_3322466727120633125[261] = 0.0;
   out_3322466727120633125[262] = 0.0;
   out_3322466727120633125[263] = 0.0;
   out_3322466727120633125[264] = 0.0;
   out_3322466727120633125[265] = 0.0;
   out_3322466727120633125[266] = 1.0;
   out_3322466727120633125[267] = 0.0;
   out_3322466727120633125[268] = 0.0;
   out_3322466727120633125[269] = 0.0;
   out_3322466727120633125[270] = 0.0;
   out_3322466727120633125[271] = 0.0;
   out_3322466727120633125[272] = 0.0;
   out_3322466727120633125[273] = 0.0;
   out_3322466727120633125[274] = 0.0;
   out_3322466727120633125[275] = 0.0;
   out_3322466727120633125[276] = 0.0;
   out_3322466727120633125[277] = 0.0;
   out_3322466727120633125[278] = 0.0;
   out_3322466727120633125[279] = 0.0;
   out_3322466727120633125[280] = 0.0;
   out_3322466727120633125[281] = 0.0;
   out_3322466727120633125[282] = 0.0;
   out_3322466727120633125[283] = 0.0;
   out_3322466727120633125[284] = 0.0;
   out_3322466727120633125[285] = 1.0;
   out_3322466727120633125[286] = 0.0;
   out_3322466727120633125[287] = 0.0;
   out_3322466727120633125[288] = 0.0;
   out_3322466727120633125[289] = 0.0;
   out_3322466727120633125[290] = 0.0;
   out_3322466727120633125[291] = 0.0;
   out_3322466727120633125[292] = 0.0;
   out_3322466727120633125[293] = 0.0;
   out_3322466727120633125[294] = 0.0;
   out_3322466727120633125[295] = 0.0;
   out_3322466727120633125[296] = 0.0;
   out_3322466727120633125[297] = 0.0;
   out_3322466727120633125[298] = 0.0;
   out_3322466727120633125[299] = 0.0;
   out_3322466727120633125[300] = 0.0;
   out_3322466727120633125[301] = 0.0;
   out_3322466727120633125[302] = 0.0;
   out_3322466727120633125[303] = 0.0;
   out_3322466727120633125[304] = 1.0;
   out_3322466727120633125[305] = 0.0;
   out_3322466727120633125[306] = 0.0;
   out_3322466727120633125[307] = 0.0;
   out_3322466727120633125[308] = 0.0;
   out_3322466727120633125[309] = 0.0;
   out_3322466727120633125[310] = 0.0;
   out_3322466727120633125[311] = 0.0;
   out_3322466727120633125[312] = 0.0;
   out_3322466727120633125[313] = 0.0;
   out_3322466727120633125[314] = 0.0;
   out_3322466727120633125[315] = 0.0;
   out_3322466727120633125[316] = 0.0;
   out_3322466727120633125[317] = 0.0;
   out_3322466727120633125[318] = 0.0;
   out_3322466727120633125[319] = 0.0;
   out_3322466727120633125[320] = 0.0;
   out_3322466727120633125[321] = 0.0;
   out_3322466727120633125[322] = 0.0;
   out_3322466727120633125[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_4385019422503781050) {
   out_4385019422503781050[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_4385019422503781050[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_4385019422503781050[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_4385019422503781050[3] = dt*state[12] + state[3];
   out_4385019422503781050[4] = dt*state[13] + state[4];
   out_4385019422503781050[5] = dt*state[14] + state[5];
   out_4385019422503781050[6] = state[6];
   out_4385019422503781050[7] = state[7];
   out_4385019422503781050[8] = state[8];
   out_4385019422503781050[9] = state[9];
   out_4385019422503781050[10] = state[10];
   out_4385019422503781050[11] = state[11];
   out_4385019422503781050[12] = state[12];
   out_4385019422503781050[13] = state[13];
   out_4385019422503781050[14] = state[14];
   out_4385019422503781050[15] = state[15];
   out_4385019422503781050[16] = state[16];
   out_4385019422503781050[17] = state[17];
}
void F_fun(double *state, double dt, double *out_3364185213448184257) {
   out_3364185213448184257[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3364185213448184257[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3364185213448184257[2] = 0;
   out_3364185213448184257[3] = 0;
   out_3364185213448184257[4] = 0;
   out_3364185213448184257[5] = 0;
   out_3364185213448184257[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3364185213448184257[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3364185213448184257[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3364185213448184257[9] = 0;
   out_3364185213448184257[10] = 0;
   out_3364185213448184257[11] = 0;
   out_3364185213448184257[12] = 0;
   out_3364185213448184257[13] = 0;
   out_3364185213448184257[14] = 0;
   out_3364185213448184257[15] = 0;
   out_3364185213448184257[16] = 0;
   out_3364185213448184257[17] = 0;
   out_3364185213448184257[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3364185213448184257[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3364185213448184257[20] = 0;
   out_3364185213448184257[21] = 0;
   out_3364185213448184257[22] = 0;
   out_3364185213448184257[23] = 0;
   out_3364185213448184257[24] = 0;
   out_3364185213448184257[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3364185213448184257[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3364185213448184257[27] = 0;
   out_3364185213448184257[28] = 0;
   out_3364185213448184257[29] = 0;
   out_3364185213448184257[30] = 0;
   out_3364185213448184257[31] = 0;
   out_3364185213448184257[32] = 0;
   out_3364185213448184257[33] = 0;
   out_3364185213448184257[34] = 0;
   out_3364185213448184257[35] = 0;
   out_3364185213448184257[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3364185213448184257[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3364185213448184257[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3364185213448184257[39] = 0;
   out_3364185213448184257[40] = 0;
   out_3364185213448184257[41] = 0;
   out_3364185213448184257[42] = 0;
   out_3364185213448184257[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3364185213448184257[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3364185213448184257[45] = 0;
   out_3364185213448184257[46] = 0;
   out_3364185213448184257[47] = 0;
   out_3364185213448184257[48] = 0;
   out_3364185213448184257[49] = 0;
   out_3364185213448184257[50] = 0;
   out_3364185213448184257[51] = 0;
   out_3364185213448184257[52] = 0;
   out_3364185213448184257[53] = 0;
   out_3364185213448184257[54] = 0;
   out_3364185213448184257[55] = 0;
   out_3364185213448184257[56] = 0;
   out_3364185213448184257[57] = 1;
   out_3364185213448184257[58] = 0;
   out_3364185213448184257[59] = 0;
   out_3364185213448184257[60] = 0;
   out_3364185213448184257[61] = 0;
   out_3364185213448184257[62] = 0;
   out_3364185213448184257[63] = 0;
   out_3364185213448184257[64] = 0;
   out_3364185213448184257[65] = 0;
   out_3364185213448184257[66] = dt;
   out_3364185213448184257[67] = 0;
   out_3364185213448184257[68] = 0;
   out_3364185213448184257[69] = 0;
   out_3364185213448184257[70] = 0;
   out_3364185213448184257[71] = 0;
   out_3364185213448184257[72] = 0;
   out_3364185213448184257[73] = 0;
   out_3364185213448184257[74] = 0;
   out_3364185213448184257[75] = 0;
   out_3364185213448184257[76] = 1;
   out_3364185213448184257[77] = 0;
   out_3364185213448184257[78] = 0;
   out_3364185213448184257[79] = 0;
   out_3364185213448184257[80] = 0;
   out_3364185213448184257[81] = 0;
   out_3364185213448184257[82] = 0;
   out_3364185213448184257[83] = 0;
   out_3364185213448184257[84] = 0;
   out_3364185213448184257[85] = dt;
   out_3364185213448184257[86] = 0;
   out_3364185213448184257[87] = 0;
   out_3364185213448184257[88] = 0;
   out_3364185213448184257[89] = 0;
   out_3364185213448184257[90] = 0;
   out_3364185213448184257[91] = 0;
   out_3364185213448184257[92] = 0;
   out_3364185213448184257[93] = 0;
   out_3364185213448184257[94] = 0;
   out_3364185213448184257[95] = 1;
   out_3364185213448184257[96] = 0;
   out_3364185213448184257[97] = 0;
   out_3364185213448184257[98] = 0;
   out_3364185213448184257[99] = 0;
   out_3364185213448184257[100] = 0;
   out_3364185213448184257[101] = 0;
   out_3364185213448184257[102] = 0;
   out_3364185213448184257[103] = 0;
   out_3364185213448184257[104] = dt;
   out_3364185213448184257[105] = 0;
   out_3364185213448184257[106] = 0;
   out_3364185213448184257[107] = 0;
   out_3364185213448184257[108] = 0;
   out_3364185213448184257[109] = 0;
   out_3364185213448184257[110] = 0;
   out_3364185213448184257[111] = 0;
   out_3364185213448184257[112] = 0;
   out_3364185213448184257[113] = 0;
   out_3364185213448184257[114] = 1;
   out_3364185213448184257[115] = 0;
   out_3364185213448184257[116] = 0;
   out_3364185213448184257[117] = 0;
   out_3364185213448184257[118] = 0;
   out_3364185213448184257[119] = 0;
   out_3364185213448184257[120] = 0;
   out_3364185213448184257[121] = 0;
   out_3364185213448184257[122] = 0;
   out_3364185213448184257[123] = 0;
   out_3364185213448184257[124] = 0;
   out_3364185213448184257[125] = 0;
   out_3364185213448184257[126] = 0;
   out_3364185213448184257[127] = 0;
   out_3364185213448184257[128] = 0;
   out_3364185213448184257[129] = 0;
   out_3364185213448184257[130] = 0;
   out_3364185213448184257[131] = 0;
   out_3364185213448184257[132] = 0;
   out_3364185213448184257[133] = 1;
   out_3364185213448184257[134] = 0;
   out_3364185213448184257[135] = 0;
   out_3364185213448184257[136] = 0;
   out_3364185213448184257[137] = 0;
   out_3364185213448184257[138] = 0;
   out_3364185213448184257[139] = 0;
   out_3364185213448184257[140] = 0;
   out_3364185213448184257[141] = 0;
   out_3364185213448184257[142] = 0;
   out_3364185213448184257[143] = 0;
   out_3364185213448184257[144] = 0;
   out_3364185213448184257[145] = 0;
   out_3364185213448184257[146] = 0;
   out_3364185213448184257[147] = 0;
   out_3364185213448184257[148] = 0;
   out_3364185213448184257[149] = 0;
   out_3364185213448184257[150] = 0;
   out_3364185213448184257[151] = 0;
   out_3364185213448184257[152] = 1;
   out_3364185213448184257[153] = 0;
   out_3364185213448184257[154] = 0;
   out_3364185213448184257[155] = 0;
   out_3364185213448184257[156] = 0;
   out_3364185213448184257[157] = 0;
   out_3364185213448184257[158] = 0;
   out_3364185213448184257[159] = 0;
   out_3364185213448184257[160] = 0;
   out_3364185213448184257[161] = 0;
   out_3364185213448184257[162] = 0;
   out_3364185213448184257[163] = 0;
   out_3364185213448184257[164] = 0;
   out_3364185213448184257[165] = 0;
   out_3364185213448184257[166] = 0;
   out_3364185213448184257[167] = 0;
   out_3364185213448184257[168] = 0;
   out_3364185213448184257[169] = 0;
   out_3364185213448184257[170] = 0;
   out_3364185213448184257[171] = 1;
   out_3364185213448184257[172] = 0;
   out_3364185213448184257[173] = 0;
   out_3364185213448184257[174] = 0;
   out_3364185213448184257[175] = 0;
   out_3364185213448184257[176] = 0;
   out_3364185213448184257[177] = 0;
   out_3364185213448184257[178] = 0;
   out_3364185213448184257[179] = 0;
   out_3364185213448184257[180] = 0;
   out_3364185213448184257[181] = 0;
   out_3364185213448184257[182] = 0;
   out_3364185213448184257[183] = 0;
   out_3364185213448184257[184] = 0;
   out_3364185213448184257[185] = 0;
   out_3364185213448184257[186] = 0;
   out_3364185213448184257[187] = 0;
   out_3364185213448184257[188] = 0;
   out_3364185213448184257[189] = 0;
   out_3364185213448184257[190] = 1;
   out_3364185213448184257[191] = 0;
   out_3364185213448184257[192] = 0;
   out_3364185213448184257[193] = 0;
   out_3364185213448184257[194] = 0;
   out_3364185213448184257[195] = 0;
   out_3364185213448184257[196] = 0;
   out_3364185213448184257[197] = 0;
   out_3364185213448184257[198] = 0;
   out_3364185213448184257[199] = 0;
   out_3364185213448184257[200] = 0;
   out_3364185213448184257[201] = 0;
   out_3364185213448184257[202] = 0;
   out_3364185213448184257[203] = 0;
   out_3364185213448184257[204] = 0;
   out_3364185213448184257[205] = 0;
   out_3364185213448184257[206] = 0;
   out_3364185213448184257[207] = 0;
   out_3364185213448184257[208] = 0;
   out_3364185213448184257[209] = 1;
   out_3364185213448184257[210] = 0;
   out_3364185213448184257[211] = 0;
   out_3364185213448184257[212] = 0;
   out_3364185213448184257[213] = 0;
   out_3364185213448184257[214] = 0;
   out_3364185213448184257[215] = 0;
   out_3364185213448184257[216] = 0;
   out_3364185213448184257[217] = 0;
   out_3364185213448184257[218] = 0;
   out_3364185213448184257[219] = 0;
   out_3364185213448184257[220] = 0;
   out_3364185213448184257[221] = 0;
   out_3364185213448184257[222] = 0;
   out_3364185213448184257[223] = 0;
   out_3364185213448184257[224] = 0;
   out_3364185213448184257[225] = 0;
   out_3364185213448184257[226] = 0;
   out_3364185213448184257[227] = 0;
   out_3364185213448184257[228] = 1;
   out_3364185213448184257[229] = 0;
   out_3364185213448184257[230] = 0;
   out_3364185213448184257[231] = 0;
   out_3364185213448184257[232] = 0;
   out_3364185213448184257[233] = 0;
   out_3364185213448184257[234] = 0;
   out_3364185213448184257[235] = 0;
   out_3364185213448184257[236] = 0;
   out_3364185213448184257[237] = 0;
   out_3364185213448184257[238] = 0;
   out_3364185213448184257[239] = 0;
   out_3364185213448184257[240] = 0;
   out_3364185213448184257[241] = 0;
   out_3364185213448184257[242] = 0;
   out_3364185213448184257[243] = 0;
   out_3364185213448184257[244] = 0;
   out_3364185213448184257[245] = 0;
   out_3364185213448184257[246] = 0;
   out_3364185213448184257[247] = 1;
   out_3364185213448184257[248] = 0;
   out_3364185213448184257[249] = 0;
   out_3364185213448184257[250] = 0;
   out_3364185213448184257[251] = 0;
   out_3364185213448184257[252] = 0;
   out_3364185213448184257[253] = 0;
   out_3364185213448184257[254] = 0;
   out_3364185213448184257[255] = 0;
   out_3364185213448184257[256] = 0;
   out_3364185213448184257[257] = 0;
   out_3364185213448184257[258] = 0;
   out_3364185213448184257[259] = 0;
   out_3364185213448184257[260] = 0;
   out_3364185213448184257[261] = 0;
   out_3364185213448184257[262] = 0;
   out_3364185213448184257[263] = 0;
   out_3364185213448184257[264] = 0;
   out_3364185213448184257[265] = 0;
   out_3364185213448184257[266] = 1;
   out_3364185213448184257[267] = 0;
   out_3364185213448184257[268] = 0;
   out_3364185213448184257[269] = 0;
   out_3364185213448184257[270] = 0;
   out_3364185213448184257[271] = 0;
   out_3364185213448184257[272] = 0;
   out_3364185213448184257[273] = 0;
   out_3364185213448184257[274] = 0;
   out_3364185213448184257[275] = 0;
   out_3364185213448184257[276] = 0;
   out_3364185213448184257[277] = 0;
   out_3364185213448184257[278] = 0;
   out_3364185213448184257[279] = 0;
   out_3364185213448184257[280] = 0;
   out_3364185213448184257[281] = 0;
   out_3364185213448184257[282] = 0;
   out_3364185213448184257[283] = 0;
   out_3364185213448184257[284] = 0;
   out_3364185213448184257[285] = 1;
   out_3364185213448184257[286] = 0;
   out_3364185213448184257[287] = 0;
   out_3364185213448184257[288] = 0;
   out_3364185213448184257[289] = 0;
   out_3364185213448184257[290] = 0;
   out_3364185213448184257[291] = 0;
   out_3364185213448184257[292] = 0;
   out_3364185213448184257[293] = 0;
   out_3364185213448184257[294] = 0;
   out_3364185213448184257[295] = 0;
   out_3364185213448184257[296] = 0;
   out_3364185213448184257[297] = 0;
   out_3364185213448184257[298] = 0;
   out_3364185213448184257[299] = 0;
   out_3364185213448184257[300] = 0;
   out_3364185213448184257[301] = 0;
   out_3364185213448184257[302] = 0;
   out_3364185213448184257[303] = 0;
   out_3364185213448184257[304] = 1;
   out_3364185213448184257[305] = 0;
   out_3364185213448184257[306] = 0;
   out_3364185213448184257[307] = 0;
   out_3364185213448184257[308] = 0;
   out_3364185213448184257[309] = 0;
   out_3364185213448184257[310] = 0;
   out_3364185213448184257[311] = 0;
   out_3364185213448184257[312] = 0;
   out_3364185213448184257[313] = 0;
   out_3364185213448184257[314] = 0;
   out_3364185213448184257[315] = 0;
   out_3364185213448184257[316] = 0;
   out_3364185213448184257[317] = 0;
   out_3364185213448184257[318] = 0;
   out_3364185213448184257[319] = 0;
   out_3364185213448184257[320] = 0;
   out_3364185213448184257[321] = 0;
   out_3364185213448184257[322] = 0;
   out_3364185213448184257[323] = 1;
}
void h_4(double *state, double *unused, double *out_277067005022287991) {
   out_277067005022287991[0] = state[6] + state[9];
   out_277067005022287991[1] = state[7] + state[10];
   out_277067005022287991[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_6474383267850384560) {
   out_6474383267850384560[0] = 0;
   out_6474383267850384560[1] = 0;
   out_6474383267850384560[2] = 0;
   out_6474383267850384560[3] = 0;
   out_6474383267850384560[4] = 0;
   out_6474383267850384560[5] = 0;
   out_6474383267850384560[6] = 1;
   out_6474383267850384560[7] = 0;
   out_6474383267850384560[8] = 0;
   out_6474383267850384560[9] = 1;
   out_6474383267850384560[10] = 0;
   out_6474383267850384560[11] = 0;
   out_6474383267850384560[12] = 0;
   out_6474383267850384560[13] = 0;
   out_6474383267850384560[14] = 0;
   out_6474383267850384560[15] = 0;
   out_6474383267850384560[16] = 0;
   out_6474383267850384560[17] = 0;
   out_6474383267850384560[18] = 0;
   out_6474383267850384560[19] = 0;
   out_6474383267850384560[20] = 0;
   out_6474383267850384560[21] = 0;
   out_6474383267850384560[22] = 0;
   out_6474383267850384560[23] = 0;
   out_6474383267850384560[24] = 0;
   out_6474383267850384560[25] = 1;
   out_6474383267850384560[26] = 0;
   out_6474383267850384560[27] = 0;
   out_6474383267850384560[28] = 1;
   out_6474383267850384560[29] = 0;
   out_6474383267850384560[30] = 0;
   out_6474383267850384560[31] = 0;
   out_6474383267850384560[32] = 0;
   out_6474383267850384560[33] = 0;
   out_6474383267850384560[34] = 0;
   out_6474383267850384560[35] = 0;
   out_6474383267850384560[36] = 0;
   out_6474383267850384560[37] = 0;
   out_6474383267850384560[38] = 0;
   out_6474383267850384560[39] = 0;
   out_6474383267850384560[40] = 0;
   out_6474383267850384560[41] = 0;
   out_6474383267850384560[42] = 0;
   out_6474383267850384560[43] = 0;
   out_6474383267850384560[44] = 1;
   out_6474383267850384560[45] = 0;
   out_6474383267850384560[46] = 0;
   out_6474383267850384560[47] = 1;
   out_6474383267850384560[48] = 0;
   out_6474383267850384560[49] = 0;
   out_6474383267850384560[50] = 0;
   out_6474383267850384560[51] = 0;
   out_6474383267850384560[52] = 0;
   out_6474383267850384560[53] = 0;
}
void h_10(double *state, double *unused, double *out_439254518827768136) {
   out_439254518827768136[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_439254518827768136[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_439254518827768136[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4997374602163059806) {
   out_4997374602163059806[0] = 0;
   out_4997374602163059806[1] = 9.8100000000000005*cos(state[1]);
   out_4997374602163059806[2] = 0;
   out_4997374602163059806[3] = 0;
   out_4997374602163059806[4] = -state[8];
   out_4997374602163059806[5] = state[7];
   out_4997374602163059806[6] = 0;
   out_4997374602163059806[7] = state[5];
   out_4997374602163059806[8] = -state[4];
   out_4997374602163059806[9] = 0;
   out_4997374602163059806[10] = 0;
   out_4997374602163059806[11] = 0;
   out_4997374602163059806[12] = 1;
   out_4997374602163059806[13] = 0;
   out_4997374602163059806[14] = 0;
   out_4997374602163059806[15] = 1;
   out_4997374602163059806[16] = 0;
   out_4997374602163059806[17] = 0;
   out_4997374602163059806[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4997374602163059806[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4997374602163059806[20] = 0;
   out_4997374602163059806[21] = state[8];
   out_4997374602163059806[22] = 0;
   out_4997374602163059806[23] = -state[6];
   out_4997374602163059806[24] = -state[5];
   out_4997374602163059806[25] = 0;
   out_4997374602163059806[26] = state[3];
   out_4997374602163059806[27] = 0;
   out_4997374602163059806[28] = 0;
   out_4997374602163059806[29] = 0;
   out_4997374602163059806[30] = 0;
   out_4997374602163059806[31] = 1;
   out_4997374602163059806[32] = 0;
   out_4997374602163059806[33] = 0;
   out_4997374602163059806[34] = 1;
   out_4997374602163059806[35] = 0;
   out_4997374602163059806[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4997374602163059806[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4997374602163059806[38] = 0;
   out_4997374602163059806[39] = -state[7];
   out_4997374602163059806[40] = state[6];
   out_4997374602163059806[41] = 0;
   out_4997374602163059806[42] = state[4];
   out_4997374602163059806[43] = -state[3];
   out_4997374602163059806[44] = 0;
   out_4997374602163059806[45] = 0;
   out_4997374602163059806[46] = 0;
   out_4997374602163059806[47] = 0;
   out_4997374602163059806[48] = 0;
   out_4997374602163059806[49] = 0;
   out_4997374602163059806[50] = 1;
   out_4997374602163059806[51] = 0;
   out_4997374602163059806[52] = 0;
   out_4997374602163059806[53] = 1;
}
void h_13(double *state, double *unused, double *out_5459722292722030835) {
   out_5459722292722030835[0] = state[3];
   out_5459722292722030835[1] = state[4];
   out_5459722292722030835[2] = state[5];
}
void H_13(double *state, double *unused, double *out_3262109442518051759) {
   out_3262109442518051759[0] = 0;
   out_3262109442518051759[1] = 0;
   out_3262109442518051759[2] = 0;
   out_3262109442518051759[3] = 1;
   out_3262109442518051759[4] = 0;
   out_3262109442518051759[5] = 0;
   out_3262109442518051759[6] = 0;
   out_3262109442518051759[7] = 0;
   out_3262109442518051759[8] = 0;
   out_3262109442518051759[9] = 0;
   out_3262109442518051759[10] = 0;
   out_3262109442518051759[11] = 0;
   out_3262109442518051759[12] = 0;
   out_3262109442518051759[13] = 0;
   out_3262109442518051759[14] = 0;
   out_3262109442518051759[15] = 0;
   out_3262109442518051759[16] = 0;
   out_3262109442518051759[17] = 0;
   out_3262109442518051759[18] = 0;
   out_3262109442518051759[19] = 0;
   out_3262109442518051759[20] = 0;
   out_3262109442518051759[21] = 0;
   out_3262109442518051759[22] = 1;
   out_3262109442518051759[23] = 0;
   out_3262109442518051759[24] = 0;
   out_3262109442518051759[25] = 0;
   out_3262109442518051759[26] = 0;
   out_3262109442518051759[27] = 0;
   out_3262109442518051759[28] = 0;
   out_3262109442518051759[29] = 0;
   out_3262109442518051759[30] = 0;
   out_3262109442518051759[31] = 0;
   out_3262109442518051759[32] = 0;
   out_3262109442518051759[33] = 0;
   out_3262109442518051759[34] = 0;
   out_3262109442518051759[35] = 0;
   out_3262109442518051759[36] = 0;
   out_3262109442518051759[37] = 0;
   out_3262109442518051759[38] = 0;
   out_3262109442518051759[39] = 0;
   out_3262109442518051759[40] = 0;
   out_3262109442518051759[41] = 1;
   out_3262109442518051759[42] = 0;
   out_3262109442518051759[43] = 0;
   out_3262109442518051759[44] = 0;
   out_3262109442518051759[45] = 0;
   out_3262109442518051759[46] = 0;
   out_3262109442518051759[47] = 0;
   out_3262109442518051759[48] = 0;
   out_3262109442518051759[49] = 0;
   out_3262109442518051759[50] = 0;
   out_3262109442518051759[51] = 0;
   out_3262109442518051759[52] = 0;
   out_3262109442518051759[53] = 0;
}
void h_14(double *state, double *unused, double *out_3151095188830224548) {
   out_3151095188830224548[0] = state[6];
   out_3151095188830224548[1] = state[7];
   out_3151095188830224548[2] = state[8];
}
void H_14(double *state, double *unused, double *out_6909499794495268159) {
   out_6909499794495268159[0] = 0;
   out_6909499794495268159[1] = 0;
   out_6909499794495268159[2] = 0;
   out_6909499794495268159[3] = 0;
   out_6909499794495268159[4] = 0;
   out_6909499794495268159[5] = 0;
   out_6909499794495268159[6] = 1;
   out_6909499794495268159[7] = 0;
   out_6909499794495268159[8] = 0;
   out_6909499794495268159[9] = 0;
   out_6909499794495268159[10] = 0;
   out_6909499794495268159[11] = 0;
   out_6909499794495268159[12] = 0;
   out_6909499794495268159[13] = 0;
   out_6909499794495268159[14] = 0;
   out_6909499794495268159[15] = 0;
   out_6909499794495268159[16] = 0;
   out_6909499794495268159[17] = 0;
   out_6909499794495268159[18] = 0;
   out_6909499794495268159[19] = 0;
   out_6909499794495268159[20] = 0;
   out_6909499794495268159[21] = 0;
   out_6909499794495268159[22] = 0;
   out_6909499794495268159[23] = 0;
   out_6909499794495268159[24] = 0;
   out_6909499794495268159[25] = 1;
   out_6909499794495268159[26] = 0;
   out_6909499794495268159[27] = 0;
   out_6909499794495268159[28] = 0;
   out_6909499794495268159[29] = 0;
   out_6909499794495268159[30] = 0;
   out_6909499794495268159[31] = 0;
   out_6909499794495268159[32] = 0;
   out_6909499794495268159[33] = 0;
   out_6909499794495268159[34] = 0;
   out_6909499794495268159[35] = 0;
   out_6909499794495268159[36] = 0;
   out_6909499794495268159[37] = 0;
   out_6909499794495268159[38] = 0;
   out_6909499794495268159[39] = 0;
   out_6909499794495268159[40] = 0;
   out_6909499794495268159[41] = 0;
   out_6909499794495268159[42] = 0;
   out_6909499794495268159[43] = 0;
   out_6909499794495268159[44] = 1;
   out_6909499794495268159[45] = 0;
   out_6909499794495268159[46] = 0;
   out_6909499794495268159[47] = 0;
   out_6909499794495268159[48] = 0;
   out_6909499794495268159[49] = 0;
   out_6909499794495268159[50] = 0;
   out_6909499794495268159[51] = 0;
   out_6909499794495268159[52] = 0;
   out_6909499794495268159[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_1364756071930907291) {
  err_fun(nom_x, delta_x, out_1364756071930907291);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8303365725516605546) {
  inv_err_fun(nom_x, true_x, out_8303365725516605546);
}
void pose_H_mod_fun(double *state, double *out_3322466727120633125) {
  H_mod_fun(state, out_3322466727120633125);
}
void pose_f_fun(double *state, double dt, double *out_4385019422503781050) {
  f_fun(state,  dt, out_4385019422503781050);
}
void pose_F_fun(double *state, double dt, double *out_3364185213448184257) {
  F_fun(state,  dt, out_3364185213448184257);
}
void pose_h_4(double *state, double *unused, double *out_277067005022287991) {
  h_4(state, unused, out_277067005022287991);
}
void pose_H_4(double *state, double *unused, double *out_6474383267850384560) {
  H_4(state, unused, out_6474383267850384560);
}
void pose_h_10(double *state, double *unused, double *out_439254518827768136) {
  h_10(state, unused, out_439254518827768136);
}
void pose_H_10(double *state, double *unused, double *out_4997374602163059806) {
  H_10(state, unused, out_4997374602163059806);
}
void pose_h_13(double *state, double *unused, double *out_5459722292722030835) {
  h_13(state, unused, out_5459722292722030835);
}
void pose_H_13(double *state, double *unused, double *out_3262109442518051759) {
  H_13(state, unused, out_3262109442518051759);
}
void pose_h_14(double *state, double *unused, double *out_3151095188830224548) {
  h_14(state, unused, out_3151095188830224548);
}
void pose_H_14(double *state, double *unused, double *out_6909499794495268159) {
  H_14(state, unused, out_6909499794495268159);
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
