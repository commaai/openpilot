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
void err_fun(double *nom_x, double *delta_x, double *out_5727667037901716489) {
   out_5727667037901716489[0] = delta_x[0] + nom_x[0];
   out_5727667037901716489[1] = delta_x[1] + nom_x[1];
   out_5727667037901716489[2] = delta_x[2] + nom_x[2];
   out_5727667037901716489[3] = delta_x[3] + nom_x[3];
   out_5727667037901716489[4] = delta_x[4] + nom_x[4];
   out_5727667037901716489[5] = delta_x[5] + nom_x[5];
   out_5727667037901716489[6] = delta_x[6] + nom_x[6];
   out_5727667037901716489[7] = delta_x[7] + nom_x[7];
   out_5727667037901716489[8] = delta_x[8] + nom_x[8];
   out_5727667037901716489[9] = delta_x[9] + nom_x[9];
   out_5727667037901716489[10] = delta_x[10] + nom_x[10];
   out_5727667037901716489[11] = delta_x[11] + nom_x[11];
   out_5727667037901716489[12] = delta_x[12] + nom_x[12];
   out_5727667037901716489[13] = delta_x[13] + nom_x[13];
   out_5727667037901716489[14] = delta_x[14] + nom_x[14];
   out_5727667037901716489[15] = delta_x[15] + nom_x[15];
   out_5727667037901716489[16] = delta_x[16] + nom_x[16];
   out_5727667037901716489[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6198627751216921880) {
   out_6198627751216921880[0] = -nom_x[0] + true_x[0];
   out_6198627751216921880[1] = -nom_x[1] + true_x[1];
   out_6198627751216921880[2] = -nom_x[2] + true_x[2];
   out_6198627751216921880[3] = -nom_x[3] + true_x[3];
   out_6198627751216921880[4] = -nom_x[4] + true_x[4];
   out_6198627751216921880[5] = -nom_x[5] + true_x[5];
   out_6198627751216921880[6] = -nom_x[6] + true_x[6];
   out_6198627751216921880[7] = -nom_x[7] + true_x[7];
   out_6198627751216921880[8] = -nom_x[8] + true_x[8];
   out_6198627751216921880[9] = -nom_x[9] + true_x[9];
   out_6198627751216921880[10] = -nom_x[10] + true_x[10];
   out_6198627751216921880[11] = -nom_x[11] + true_x[11];
   out_6198627751216921880[12] = -nom_x[12] + true_x[12];
   out_6198627751216921880[13] = -nom_x[13] + true_x[13];
   out_6198627751216921880[14] = -nom_x[14] + true_x[14];
   out_6198627751216921880[15] = -nom_x[15] + true_x[15];
   out_6198627751216921880[16] = -nom_x[16] + true_x[16];
   out_6198627751216921880[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_8253589554410785672) {
   out_8253589554410785672[0] = 1.0;
   out_8253589554410785672[1] = 0.0;
   out_8253589554410785672[2] = 0.0;
   out_8253589554410785672[3] = 0.0;
   out_8253589554410785672[4] = 0.0;
   out_8253589554410785672[5] = 0.0;
   out_8253589554410785672[6] = 0.0;
   out_8253589554410785672[7] = 0.0;
   out_8253589554410785672[8] = 0.0;
   out_8253589554410785672[9] = 0.0;
   out_8253589554410785672[10] = 0.0;
   out_8253589554410785672[11] = 0.0;
   out_8253589554410785672[12] = 0.0;
   out_8253589554410785672[13] = 0.0;
   out_8253589554410785672[14] = 0.0;
   out_8253589554410785672[15] = 0.0;
   out_8253589554410785672[16] = 0.0;
   out_8253589554410785672[17] = 0.0;
   out_8253589554410785672[18] = 0.0;
   out_8253589554410785672[19] = 1.0;
   out_8253589554410785672[20] = 0.0;
   out_8253589554410785672[21] = 0.0;
   out_8253589554410785672[22] = 0.0;
   out_8253589554410785672[23] = 0.0;
   out_8253589554410785672[24] = 0.0;
   out_8253589554410785672[25] = 0.0;
   out_8253589554410785672[26] = 0.0;
   out_8253589554410785672[27] = 0.0;
   out_8253589554410785672[28] = 0.0;
   out_8253589554410785672[29] = 0.0;
   out_8253589554410785672[30] = 0.0;
   out_8253589554410785672[31] = 0.0;
   out_8253589554410785672[32] = 0.0;
   out_8253589554410785672[33] = 0.0;
   out_8253589554410785672[34] = 0.0;
   out_8253589554410785672[35] = 0.0;
   out_8253589554410785672[36] = 0.0;
   out_8253589554410785672[37] = 0.0;
   out_8253589554410785672[38] = 1.0;
   out_8253589554410785672[39] = 0.0;
   out_8253589554410785672[40] = 0.0;
   out_8253589554410785672[41] = 0.0;
   out_8253589554410785672[42] = 0.0;
   out_8253589554410785672[43] = 0.0;
   out_8253589554410785672[44] = 0.0;
   out_8253589554410785672[45] = 0.0;
   out_8253589554410785672[46] = 0.0;
   out_8253589554410785672[47] = 0.0;
   out_8253589554410785672[48] = 0.0;
   out_8253589554410785672[49] = 0.0;
   out_8253589554410785672[50] = 0.0;
   out_8253589554410785672[51] = 0.0;
   out_8253589554410785672[52] = 0.0;
   out_8253589554410785672[53] = 0.0;
   out_8253589554410785672[54] = 0.0;
   out_8253589554410785672[55] = 0.0;
   out_8253589554410785672[56] = 0.0;
   out_8253589554410785672[57] = 1.0;
   out_8253589554410785672[58] = 0.0;
   out_8253589554410785672[59] = 0.0;
   out_8253589554410785672[60] = 0.0;
   out_8253589554410785672[61] = 0.0;
   out_8253589554410785672[62] = 0.0;
   out_8253589554410785672[63] = 0.0;
   out_8253589554410785672[64] = 0.0;
   out_8253589554410785672[65] = 0.0;
   out_8253589554410785672[66] = 0.0;
   out_8253589554410785672[67] = 0.0;
   out_8253589554410785672[68] = 0.0;
   out_8253589554410785672[69] = 0.0;
   out_8253589554410785672[70] = 0.0;
   out_8253589554410785672[71] = 0.0;
   out_8253589554410785672[72] = 0.0;
   out_8253589554410785672[73] = 0.0;
   out_8253589554410785672[74] = 0.0;
   out_8253589554410785672[75] = 0.0;
   out_8253589554410785672[76] = 1.0;
   out_8253589554410785672[77] = 0.0;
   out_8253589554410785672[78] = 0.0;
   out_8253589554410785672[79] = 0.0;
   out_8253589554410785672[80] = 0.0;
   out_8253589554410785672[81] = 0.0;
   out_8253589554410785672[82] = 0.0;
   out_8253589554410785672[83] = 0.0;
   out_8253589554410785672[84] = 0.0;
   out_8253589554410785672[85] = 0.0;
   out_8253589554410785672[86] = 0.0;
   out_8253589554410785672[87] = 0.0;
   out_8253589554410785672[88] = 0.0;
   out_8253589554410785672[89] = 0.0;
   out_8253589554410785672[90] = 0.0;
   out_8253589554410785672[91] = 0.0;
   out_8253589554410785672[92] = 0.0;
   out_8253589554410785672[93] = 0.0;
   out_8253589554410785672[94] = 0.0;
   out_8253589554410785672[95] = 1.0;
   out_8253589554410785672[96] = 0.0;
   out_8253589554410785672[97] = 0.0;
   out_8253589554410785672[98] = 0.0;
   out_8253589554410785672[99] = 0.0;
   out_8253589554410785672[100] = 0.0;
   out_8253589554410785672[101] = 0.0;
   out_8253589554410785672[102] = 0.0;
   out_8253589554410785672[103] = 0.0;
   out_8253589554410785672[104] = 0.0;
   out_8253589554410785672[105] = 0.0;
   out_8253589554410785672[106] = 0.0;
   out_8253589554410785672[107] = 0.0;
   out_8253589554410785672[108] = 0.0;
   out_8253589554410785672[109] = 0.0;
   out_8253589554410785672[110] = 0.0;
   out_8253589554410785672[111] = 0.0;
   out_8253589554410785672[112] = 0.0;
   out_8253589554410785672[113] = 0.0;
   out_8253589554410785672[114] = 1.0;
   out_8253589554410785672[115] = 0.0;
   out_8253589554410785672[116] = 0.0;
   out_8253589554410785672[117] = 0.0;
   out_8253589554410785672[118] = 0.0;
   out_8253589554410785672[119] = 0.0;
   out_8253589554410785672[120] = 0.0;
   out_8253589554410785672[121] = 0.0;
   out_8253589554410785672[122] = 0.0;
   out_8253589554410785672[123] = 0.0;
   out_8253589554410785672[124] = 0.0;
   out_8253589554410785672[125] = 0.0;
   out_8253589554410785672[126] = 0.0;
   out_8253589554410785672[127] = 0.0;
   out_8253589554410785672[128] = 0.0;
   out_8253589554410785672[129] = 0.0;
   out_8253589554410785672[130] = 0.0;
   out_8253589554410785672[131] = 0.0;
   out_8253589554410785672[132] = 0.0;
   out_8253589554410785672[133] = 1.0;
   out_8253589554410785672[134] = 0.0;
   out_8253589554410785672[135] = 0.0;
   out_8253589554410785672[136] = 0.0;
   out_8253589554410785672[137] = 0.0;
   out_8253589554410785672[138] = 0.0;
   out_8253589554410785672[139] = 0.0;
   out_8253589554410785672[140] = 0.0;
   out_8253589554410785672[141] = 0.0;
   out_8253589554410785672[142] = 0.0;
   out_8253589554410785672[143] = 0.0;
   out_8253589554410785672[144] = 0.0;
   out_8253589554410785672[145] = 0.0;
   out_8253589554410785672[146] = 0.0;
   out_8253589554410785672[147] = 0.0;
   out_8253589554410785672[148] = 0.0;
   out_8253589554410785672[149] = 0.0;
   out_8253589554410785672[150] = 0.0;
   out_8253589554410785672[151] = 0.0;
   out_8253589554410785672[152] = 1.0;
   out_8253589554410785672[153] = 0.0;
   out_8253589554410785672[154] = 0.0;
   out_8253589554410785672[155] = 0.0;
   out_8253589554410785672[156] = 0.0;
   out_8253589554410785672[157] = 0.0;
   out_8253589554410785672[158] = 0.0;
   out_8253589554410785672[159] = 0.0;
   out_8253589554410785672[160] = 0.0;
   out_8253589554410785672[161] = 0.0;
   out_8253589554410785672[162] = 0.0;
   out_8253589554410785672[163] = 0.0;
   out_8253589554410785672[164] = 0.0;
   out_8253589554410785672[165] = 0.0;
   out_8253589554410785672[166] = 0.0;
   out_8253589554410785672[167] = 0.0;
   out_8253589554410785672[168] = 0.0;
   out_8253589554410785672[169] = 0.0;
   out_8253589554410785672[170] = 0.0;
   out_8253589554410785672[171] = 1.0;
   out_8253589554410785672[172] = 0.0;
   out_8253589554410785672[173] = 0.0;
   out_8253589554410785672[174] = 0.0;
   out_8253589554410785672[175] = 0.0;
   out_8253589554410785672[176] = 0.0;
   out_8253589554410785672[177] = 0.0;
   out_8253589554410785672[178] = 0.0;
   out_8253589554410785672[179] = 0.0;
   out_8253589554410785672[180] = 0.0;
   out_8253589554410785672[181] = 0.0;
   out_8253589554410785672[182] = 0.0;
   out_8253589554410785672[183] = 0.0;
   out_8253589554410785672[184] = 0.0;
   out_8253589554410785672[185] = 0.0;
   out_8253589554410785672[186] = 0.0;
   out_8253589554410785672[187] = 0.0;
   out_8253589554410785672[188] = 0.0;
   out_8253589554410785672[189] = 0.0;
   out_8253589554410785672[190] = 1.0;
   out_8253589554410785672[191] = 0.0;
   out_8253589554410785672[192] = 0.0;
   out_8253589554410785672[193] = 0.0;
   out_8253589554410785672[194] = 0.0;
   out_8253589554410785672[195] = 0.0;
   out_8253589554410785672[196] = 0.0;
   out_8253589554410785672[197] = 0.0;
   out_8253589554410785672[198] = 0.0;
   out_8253589554410785672[199] = 0.0;
   out_8253589554410785672[200] = 0.0;
   out_8253589554410785672[201] = 0.0;
   out_8253589554410785672[202] = 0.0;
   out_8253589554410785672[203] = 0.0;
   out_8253589554410785672[204] = 0.0;
   out_8253589554410785672[205] = 0.0;
   out_8253589554410785672[206] = 0.0;
   out_8253589554410785672[207] = 0.0;
   out_8253589554410785672[208] = 0.0;
   out_8253589554410785672[209] = 1.0;
   out_8253589554410785672[210] = 0.0;
   out_8253589554410785672[211] = 0.0;
   out_8253589554410785672[212] = 0.0;
   out_8253589554410785672[213] = 0.0;
   out_8253589554410785672[214] = 0.0;
   out_8253589554410785672[215] = 0.0;
   out_8253589554410785672[216] = 0.0;
   out_8253589554410785672[217] = 0.0;
   out_8253589554410785672[218] = 0.0;
   out_8253589554410785672[219] = 0.0;
   out_8253589554410785672[220] = 0.0;
   out_8253589554410785672[221] = 0.0;
   out_8253589554410785672[222] = 0.0;
   out_8253589554410785672[223] = 0.0;
   out_8253589554410785672[224] = 0.0;
   out_8253589554410785672[225] = 0.0;
   out_8253589554410785672[226] = 0.0;
   out_8253589554410785672[227] = 0.0;
   out_8253589554410785672[228] = 1.0;
   out_8253589554410785672[229] = 0.0;
   out_8253589554410785672[230] = 0.0;
   out_8253589554410785672[231] = 0.0;
   out_8253589554410785672[232] = 0.0;
   out_8253589554410785672[233] = 0.0;
   out_8253589554410785672[234] = 0.0;
   out_8253589554410785672[235] = 0.0;
   out_8253589554410785672[236] = 0.0;
   out_8253589554410785672[237] = 0.0;
   out_8253589554410785672[238] = 0.0;
   out_8253589554410785672[239] = 0.0;
   out_8253589554410785672[240] = 0.0;
   out_8253589554410785672[241] = 0.0;
   out_8253589554410785672[242] = 0.0;
   out_8253589554410785672[243] = 0.0;
   out_8253589554410785672[244] = 0.0;
   out_8253589554410785672[245] = 0.0;
   out_8253589554410785672[246] = 0.0;
   out_8253589554410785672[247] = 1.0;
   out_8253589554410785672[248] = 0.0;
   out_8253589554410785672[249] = 0.0;
   out_8253589554410785672[250] = 0.0;
   out_8253589554410785672[251] = 0.0;
   out_8253589554410785672[252] = 0.0;
   out_8253589554410785672[253] = 0.0;
   out_8253589554410785672[254] = 0.0;
   out_8253589554410785672[255] = 0.0;
   out_8253589554410785672[256] = 0.0;
   out_8253589554410785672[257] = 0.0;
   out_8253589554410785672[258] = 0.0;
   out_8253589554410785672[259] = 0.0;
   out_8253589554410785672[260] = 0.0;
   out_8253589554410785672[261] = 0.0;
   out_8253589554410785672[262] = 0.0;
   out_8253589554410785672[263] = 0.0;
   out_8253589554410785672[264] = 0.0;
   out_8253589554410785672[265] = 0.0;
   out_8253589554410785672[266] = 1.0;
   out_8253589554410785672[267] = 0.0;
   out_8253589554410785672[268] = 0.0;
   out_8253589554410785672[269] = 0.0;
   out_8253589554410785672[270] = 0.0;
   out_8253589554410785672[271] = 0.0;
   out_8253589554410785672[272] = 0.0;
   out_8253589554410785672[273] = 0.0;
   out_8253589554410785672[274] = 0.0;
   out_8253589554410785672[275] = 0.0;
   out_8253589554410785672[276] = 0.0;
   out_8253589554410785672[277] = 0.0;
   out_8253589554410785672[278] = 0.0;
   out_8253589554410785672[279] = 0.0;
   out_8253589554410785672[280] = 0.0;
   out_8253589554410785672[281] = 0.0;
   out_8253589554410785672[282] = 0.0;
   out_8253589554410785672[283] = 0.0;
   out_8253589554410785672[284] = 0.0;
   out_8253589554410785672[285] = 1.0;
   out_8253589554410785672[286] = 0.0;
   out_8253589554410785672[287] = 0.0;
   out_8253589554410785672[288] = 0.0;
   out_8253589554410785672[289] = 0.0;
   out_8253589554410785672[290] = 0.0;
   out_8253589554410785672[291] = 0.0;
   out_8253589554410785672[292] = 0.0;
   out_8253589554410785672[293] = 0.0;
   out_8253589554410785672[294] = 0.0;
   out_8253589554410785672[295] = 0.0;
   out_8253589554410785672[296] = 0.0;
   out_8253589554410785672[297] = 0.0;
   out_8253589554410785672[298] = 0.0;
   out_8253589554410785672[299] = 0.0;
   out_8253589554410785672[300] = 0.0;
   out_8253589554410785672[301] = 0.0;
   out_8253589554410785672[302] = 0.0;
   out_8253589554410785672[303] = 0.0;
   out_8253589554410785672[304] = 1.0;
   out_8253589554410785672[305] = 0.0;
   out_8253589554410785672[306] = 0.0;
   out_8253589554410785672[307] = 0.0;
   out_8253589554410785672[308] = 0.0;
   out_8253589554410785672[309] = 0.0;
   out_8253589554410785672[310] = 0.0;
   out_8253589554410785672[311] = 0.0;
   out_8253589554410785672[312] = 0.0;
   out_8253589554410785672[313] = 0.0;
   out_8253589554410785672[314] = 0.0;
   out_8253589554410785672[315] = 0.0;
   out_8253589554410785672[316] = 0.0;
   out_8253589554410785672[317] = 0.0;
   out_8253589554410785672[318] = 0.0;
   out_8253589554410785672[319] = 0.0;
   out_8253589554410785672[320] = 0.0;
   out_8253589554410785672[321] = 0.0;
   out_8253589554410785672[322] = 0.0;
   out_8253589554410785672[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_6523760126628443948) {
   out_6523760126628443948[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_6523760126628443948[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_6523760126628443948[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_6523760126628443948[3] = dt*state[12] + state[3];
   out_6523760126628443948[4] = dt*state[13] + state[4];
   out_6523760126628443948[5] = dt*state[14] + state[5];
   out_6523760126628443948[6] = state[6];
   out_6523760126628443948[7] = state[7];
   out_6523760126628443948[8] = state[8];
   out_6523760126628443948[9] = state[9];
   out_6523760126628443948[10] = state[10];
   out_6523760126628443948[11] = state[11];
   out_6523760126628443948[12] = state[12];
   out_6523760126628443948[13] = state[13];
   out_6523760126628443948[14] = state[14];
   out_6523760126628443948[15] = state[15];
   out_6523760126628443948[16] = state[16];
   out_6523760126628443948[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2397429393732395161) {
   out_2397429393732395161[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2397429393732395161[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2397429393732395161[2] = 0;
   out_2397429393732395161[3] = 0;
   out_2397429393732395161[4] = 0;
   out_2397429393732395161[5] = 0;
   out_2397429393732395161[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2397429393732395161[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2397429393732395161[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2397429393732395161[9] = 0;
   out_2397429393732395161[10] = 0;
   out_2397429393732395161[11] = 0;
   out_2397429393732395161[12] = 0;
   out_2397429393732395161[13] = 0;
   out_2397429393732395161[14] = 0;
   out_2397429393732395161[15] = 0;
   out_2397429393732395161[16] = 0;
   out_2397429393732395161[17] = 0;
   out_2397429393732395161[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2397429393732395161[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2397429393732395161[20] = 0;
   out_2397429393732395161[21] = 0;
   out_2397429393732395161[22] = 0;
   out_2397429393732395161[23] = 0;
   out_2397429393732395161[24] = 0;
   out_2397429393732395161[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2397429393732395161[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2397429393732395161[27] = 0;
   out_2397429393732395161[28] = 0;
   out_2397429393732395161[29] = 0;
   out_2397429393732395161[30] = 0;
   out_2397429393732395161[31] = 0;
   out_2397429393732395161[32] = 0;
   out_2397429393732395161[33] = 0;
   out_2397429393732395161[34] = 0;
   out_2397429393732395161[35] = 0;
   out_2397429393732395161[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2397429393732395161[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2397429393732395161[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2397429393732395161[39] = 0;
   out_2397429393732395161[40] = 0;
   out_2397429393732395161[41] = 0;
   out_2397429393732395161[42] = 0;
   out_2397429393732395161[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2397429393732395161[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2397429393732395161[45] = 0;
   out_2397429393732395161[46] = 0;
   out_2397429393732395161[47] = 0;
   out_2397429393732395161[48] = 0;
   out_2397429393732395161[49] = 0;
   out_2397429393732395161[50] = 0;
   out_2397429393732395161[51] = 0;
   out_2397429393732395161[52] = 0;
   out_2397429393732395161[53] = 0;
   out_2397429393732395161[54] = 0;
   out_2397429393732395161[55] = 0;
   out_2397429393732395161[56] = 0;
   out_2397429393732395161[57] = 1;
   out_2397429393732395161[58] = 0;
   out_2397429393732395161[59] = 0;
   out_2397429393732395161[60] = 0;
   out_2397429393732395161[61] = 0;
   out_2397429393732395161[62] = 0;
   out_2397429393732395161[63] = 0;
   out_2397429393732395161[64] = 0;
   out_2397429393732395161[65] = 0;
   out_2397429393732395161[66] = dt;
   out_2397429393732395161[67] = 0;
   out_2397429393732395161[68] = 0;
   out_2397429393732395161[69] = 0;
   out_2397429393732395161[70] = 0;
   out_2397429393732395161[71] = 0;
   out_2397429393732395161[72] = 0;
   out_2397429393732395161[73] = 0;
   out_2397429393732395161[74] = 0;
   out_2397429393732395161[75] = 0;
   out_2397429393732395161[76] = 1;
   out_2397429393732395161[77] = 0;
   out_2397429393732395161[78] = 0;
   out_2397429393732395161[79] = 0;
   out_2397429393732395161[80] = 0;
   out_2397429393732395161[81] = 0;
   out_2397429393732395161[82] = 0;
   out_2397429393732395161[83] = 0;
   out_2397429393732395161[84] = 0;
   out_2397429393732395161[85] = dt;
   out_2397429393732395161[86] = 0;
   out_2397429393732395161[87] = 0;
   out_2397429393732395161[88] = 0;
   out_2397429393732395161[89] = 0;
   out_2397429393732395161[90] = 0;
   out_2397429393732395161[91] = 0;
   out_2397429393732395161[92] = 0;
   out_2397429393732395161[93] = 0;
   out_2397429393732395161[94] = 0;
   out_2397429393732395161[95] = 1;
   out_2397429393732395161[96] = 0;
   out_2397429393732395161[97] = 0;
   out_2397429393732395161[98] = 0;
   out_2397429393732395161[99] = 0;
   out_2397429393732395161[100] = 0;
   out_2397429393732395161[101] = 0;
   out_2397429393732395161[102] = 0;
   out_2397429393732395161[103] = 0;
   out_2397429393732395161[104] = dt;
   out_2397429393732395161[105] = 0;
   out_2397429393732395161[106] = 0;
   out_2397429393732395161[107] = 0;
   out_2397429393732395161[108] = 0;
   out_2397429393732395161[109] = 0;
   out_2397429393732395161[110] = 0;
   out_2397429393732395161[111] = 0;
   out_2397429393732395161[112] = 0;
   out_2397429393732395161[113] = 0;
   out_2397429393732395161[114] = 1;
   out_2397429393732395161[115] = 0;
   out_2397429393732395161[116] = 0;
   out_2397429393732395161[117] = 0;
   out_2397429393732395161[118] = 0;
   out_2397429393732395161[119] = 0;
   out_2397429393732395161[120] = 0;
   out_2397429393732395161[121] = 0;
   out_2397429393732395161[122] = 0;
   out_2397429393732395161[123] = 0;
   out_2397429393732395161[124] = 0;
   out_2397429393732395161[125] = 0;
   out_2397429393732395161[126] = 0;
   out_2397429393732395161[127] = 0;
   out_2397429393732395161[128] = 0;
   out_2397429393732395161[129] = 0;
   out_2397429393732395161[130] = 0;
   out_2397429393732395161[131] = 0;
   out_2397429393732395161[132] = 0;
   out_2397429393732395161[133] = 1;
   out_2397429393732395161[134] = 0;
   out_2397429393732395161[135] = 0;
   out_2397429393732395161[136] = 0;
   out_2397429393732395161[137] = 0;
   out_2397429393732395161[138] = 0;
   out_2397429393732395161[139] = 0;
   out_2397429393732395161[140] = 0;
   out_2397429393732395161[141] = 0;
   out_2397429393732395161[142] = 0;
   out_2397429393732395161[143] = 0;
   out_2397429393732395161[144] = 0;
   out_2397429393732395161[145] = 0;
   out_2397429393732395161[146] = 0;
   out_2397429393732395161[147] = 0;
   out_2397429393732395161[148] = 0;
   out_2397429393732395161[149] = 0;
   out_2397429393732395161[150] = 0;
   out_2397429393732395161[151] = 0;
   out_2397429393732395161[152] = 1;
   out_2397429393732395161[153] = 0;
   out_2397429393732395161[154] = 0;
   out_2397429393732395161[155] = 0;
   out_2397429393732395161[156] = 0;
   out_2397429393732395161[157] = 0;
   out_2397429393732395161[158] = 0;
   out_2397429393732395161[159] = 0;
   out_2397429393732395161[160] = 0;
   out_2397429393732395161[161] = 0;
   out_2397429393732395161[162] = 0;
   out_2397429393732395161[163] = 0;
   out_2397429393732395161[164] = 0;
   out_2397429393732395161[165] = 0;
   out_2397429393732395161[166] = 0;
   out_2397429393732395161[167] = 0;
   out_2397429393732395161[168] = 0;
   out_2397429393732395161[169] = 0;
   out_2397429393732395161[170] = 0;
   out_2397429393732395161[171] = 1;
   out_2397429393732395161[172] = 0;
   out_2397429393732395161[173] = 0;
   out_2397429393732395161[174] = 0;
   out_2397429393732395161[175] = 0;
   out_2397429393732395161[176] = 0;
   out_2397429393732395161[177] = 0;
   out_2397429393732395161[178] = 0;
   out_2397429393732395161[179] = 0;
   out_2397429393732395161[180] = 0;
   out_2397429393732395161[181] = 0;
   out_2397429393732395161[182] = 0;
   out_2397429393732395161[183] = 0;
   out_2397429393732395161[184] = 0;
   out_2397429393732395161[185] = 0;
   out_2397429393732395161[186] = 0;
   out_2397429393732395161[187] = 0;
   out_2397429393732395161[188] = 0;
   out_2397429393732395161[189] = 0;
   out_2397429393732395161[190] = 1;
   out_2397429393732395161[191] = 0;
   out_2397429393732395161[192] = 0;
   out_2397429393732395161[193] = 0;
   out_2397429393732395161[194] = 0;
   out_2397429393732395161[195] = 0;
   out_2397429393732395161[196] = 0;
   out_2397429393732395161[197] = 0;
   out_2397429393732395161[198] = 0;
   out_2397429393732395161[199] = 0;
   out_2397429393732395161[200] = 0;
   out_2397429393732395161[201] = 0;
   out_2397429393732395161[202] = 0;
   out_2397429393732395161[203] = 0;
   out_2397429393732395161[204] = 0;
   out_2397429393732395161[205] = 0;
   out_2397429393732395161[206] = 0;
   out_2397429393732395161[207] = 0;
   out_2397429393732395161[208] = 0;
   out_2397429393732395161[209] = 1;
   out_2397429393732395161[210] = 0;
   out_2397429393732395161[211] = 0;
   out_2397429393732395161[212] = 0;
   out_2397429393732395161[213] = 0;
   out_2397429393732395161[214] = 0;
   out_2397429393732395161[215] = 0;
   out_2397429393732395161[216] = 0;
   out_2397429393732395161[217] = 0;
   out_2397429393732395161[218] = 0;
   out_2397429393732395161[219] = 0;
   out_2397429393732395161[220] = 0;
   out_2397429393732395161[221] = 0;
   out_2397429393732395161[222] = 0;
   out_2397429393732395161[223] = 0;
   out_2397429393732395161[224] = 0;
   out_2397429393732395161[225] = 0;
   out_2397429393732395161[226] = 0;
   out_2397429393732395161[227] = 0;
   out_2397429393732395161[228] = 1;
   out_2397429393732395161[229] = 0;
   out_2397429393732395161[230] = 0;
   out_2397429393732395161[231] = 0;
   out_2397429393732395161[232] = 0;
   out_2397429393732395161[233] = 0;
   out_2397429393732395161[234] = 0;
   out_2397429393732395161[235] = 0;
   out_2397429393732395161[236] = 0;
   out_2397429393732395161[237] = 0;
   out_2397429393732395161[238] = 0;
   out_2397429393732395161[239] = 0;
   out_2397429393732395161[240] = 0;
   out_2397429393732395161[241] = 0;
   out_2397429393732395161[242] = 0;
   out_2397429393732395161[243] = 0;
   out_2397429393732395161[244] = 0;
   out_2397429393732395161[245] = 0;
   out_2397429393732395161[246] = 0;
   out_2397429393732395161[247] = 1;
   out_2397429393732395161[248] = 0;
   out_2397429393732395161[249] = 0;
   out_2397429393732395161[250] = 0;
   out_2397429393732395161[251] = 0;
   out_2397429393732395161[252] = 0;
   out_2397429393732395161[253] = 0;
   out_2397429393732395161[254] = 0;
   out_2397429393732395161[255] = 0;
   out_2397429393732395161[256] = 0;
   out_2397429393732395161[257] = 0;
   out_2397429393732395161[258] = 0;
   out_2397429393732395161[259] = 0;
   out_2397429393732395161[260] = 0;
   out_2397429393732395161[261] = 0;
   out_2397429393732395161[262] = 0;
   out_2397429393732395161[263] = 0;
   out_2397429393732395161[264] = 0;
   out_2397429393732395161[265] = 0;
   out_2397429393732395161[266] = 1;
   out_2397429393732395161[267] = 0;
   out_2397429393732395161[268] = 0;
   out_2397429393732395161[269] = 0;
   out_2397429393732395161[270] = 0;
   out_2397429393732395161[271] = 0;
   out_2397429393732395161[272] = 0;
   out_2397429393732395161[273] = 0;
   out_2397429393732395161[274] = 0;
   out_2397429393732395161[275] = 0;
   out_2397429393732395161[276] = 0;
   out_2397429393732395161[277] = 0;
   out_2397429393732395161[278] = 0;
   out_2397429393732395161[279] = 0;
   out_2397429393732395161[280] = 0;
   out_2397429393732395161[281] = 0;
   out_2397429393732395161[282] = 0;
   out_2397429393732395161[283] = 0;
   out_2397429393732395161[284] = 0;
   out_2397429393732395161[285] = 1;
   out_2397429393732395161[286] = 0;
   out_2397429393732395161[287] = 0;
   out_2397429393732395161[288] = 0;
   out_2397429393732395161[289] = 0;
   out_2397429393732395161[290] = 0;
   out_2397429393732395161[291] = 0;
   out_2397429393732395161[292] = 0;
   out_2397429393732395161[293] = 0;
   out_2397429393732395161[294] = 0;
   out_2397429393732395161[295] = 0;
   out_2397429393732395161[296] = 0;
   out_2397429393732395161[297] = 0;
   out_2397429393732395161[298] = 0;
   out_2397429393732395161[299] = 0;
   out_2397429393732395161[300] = 0;
   out_2397429393732395161[301] = 0;
   out_2397429393732395161[302] = 0;
   out_2397429393732395161[303] = 0;
   out_2397429393732395161[304] = 1;
   out_2397429393732395161[305] = 0;
   out_2397429393732395161[306] = 0;
   out_2397429393732395161[307] = 0;
   out_2397429393732395161[308] = 0;
   out_2397429393732395161[309] = 0;
   out_2397429393732395161[310] = 0;
   out_2397429393732395161[311] = 0;
   out_2397429393732395161[312] = 0;
   out_2397429393732395161[313] = 0;
   out_2397429393732395161[314] = 0;
   out_2397429393732395161[315] = 0;
   out_2397429393732395161[316] = 0;
   out_2397429393732395161[317] = 0;
   out_2397429393732395161[318] = 0;
   out_2397429393732395161[319] = 0;
   out_2397429393732395161[320] = 0;
   out_2397429393732395161[321] = 0;
   out_2397429393732395161[322] = 0;
   out_2397429393732395161[323] = 1;
}
void h_4(double *state, double *unused, double *out_8382275560088012236) {
   out_8382275560088012236[0] = state[6] + state[9];
   out_8382275560088012236[1] = state[7] + state[10];
   out_8382275560088012236[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_6433890828504488811) {
   out_6433890828504488811[0] = 0;
   out_6433890828504488811[1] = 0;
   out_6433890828504488811[2] = 0;
   out_6433890828504488811[3] = 0;
   out_6433890828504488811[4] = 0;
   out_6433890828504488811[5] = 0;
   out_6433890828504488811[6] = 1;
   out_6433890828504488811[7] = 0;
   out_6433890828504488811[8] = 0;
   out_6433890828504488811[9] = 1;
   out_6433890828504488811[10] = 0;
   out_6433890828504488811[11] = 0;
   out_6433890828504488811[12] = 0;
   out_6433890828504488811[13] = 0;
   out_6433890828504488811[14] = 0;
   out_6433890828504488811[15] = 0;
   out_6433890828504488811[16] = 0;
   out_6433890828504488811[17] = 0;
   out_6433890828504488811[18] = 0;
   out_6433890828504488811[19] = 0;
   out_6433890828504488811[20] = 0;
   out_6433890828504488811[21] = 0;
   out_6433890828504488811[22] = 0;
   out_6433890828504488811[23] = 0;
   out_6433890828504488811[24] = 0;
   out_6433890828504488811[25] = 1;
   out_6433890828504488811[26] = 0;
   out_6433890828504488811[27] = 0;
   out_6433890828504488811[28] = 1;
   out_6433890828504488811[29] = 0;
   out_6433890828504488811[30] = 0;
   out_6433890828504488811[31] = 0;
   out_6433890828504488811[32] = 0;
   out_6433890828504488811[33] = 0;
   out_6433890828504488811[34] = 0;
   out_6433890828504488811[35] = 0;
   out_6433890828504488811[36] = 0;
   out_6433890828504488811[37] = 0;
   out_6433890828504488811[38] = 0;
   out_6433890828504488811[39] = 0;
   out_6433890828504488811[40] = 0;
   out_6433890828504488811[41] = 0;
   out_6433890828504488811[42] = 0;
   out_6433890828504488811[43] = 0;
   out_6433890828504488811[44] = 1;
   out_6433890828504488811[45] = 0;
   out_6433890828504488811[46] = 0;
   out_6433890828504488811[47] = 1;
   out_6433890828504488811[48] = 0;
   out_6433890828504488811[49] = 0;
   out_6433890828504488811[50] = 0;
   out_6433890828504488811[51] = 0;
   out_6433890828504488811[52] = 0;
   out_6433890828504488811[53] = 0;
}
void h_10(double *state, double *unused, double *out_3474458810352115326) {
   out_3474458810352115326[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_3474458810352115326[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_3474458810352115326[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_3190334320092666102) {
   out_3190334320092666102[0] = 0;
   out_3190334320092666102[1] = 9.8100000000000005*cos(state[1]);
   out_3190334320092666102[2] = 0;
   out_3190334320092666102[3] = 0;
   out_3190334320092666102[4] = -state[8];
   out_3190334320092666102[5] = state[7];
   out_3190334320092666102[6] = 0;
   out_3190334320092666102[7] = state[5];
   out_3190334320092666102[8] = -state[4];
   out_3190334320092666102[9] = 0;
   out_3190334320092666102[10] = 0;
   out_3190334320092666102[11] = 0;
   out_3190334320092666102[12] = 1;
   out_3190334320092666102[13] = 0;
   out_3190334320092666102[14] = 0;
   out_3190334320092666102[15] = 1;
   out_3190334320092666102[16] = 0;
   out_3190334320092666102[17] = 0;
   out_3190334320092666102[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_3190334320092666102[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_3190334320092666102[20] = 0;
   out_3190334320092666102[21] = state[8];
   out_3190334320092666102[22] = 0;
   out_3190334320092666102[23] = -state[6];
   out_3190334320092666102[24] = -state[5];
   out_3190334320092666102[25] = 0;
   out_3190334320092666102[26] = state[3];
   out_3190334320092666102[27] = 0;
   out_3190334320092666102[28] = 0;
   out_3190334320092666102[29] = 0;
   out_3190334320092666102[30] = 0;
   out_3190334320092666102[31] = 1;
   out_3190334320092666102[32] = 0;
   out_3190334320092666102[33] = 0;
   out_3190334320092666102[34] = 1;
   out_3190334320092666102[35] = 0;
   out_3190334320092666102[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_3190334320092666102[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_3190334320092666102[38] = 0;
   out_3190334320092666102[39] = -state[7];
   out_3190334320092666102[40] = state[6];
   out_3190334320092666102[41] = 0;
   out_3190334320092666102[42] = state[4];
   out_3190334320092666102[43] = -state[3];
   out_3190334320092666102[44] = 0;
   out_3190334320092666102[45] = 0;
   out_3190334320092666102[46] = 0;
   out_3190334320092666102[47] = 0;
   out_3190334320092666102[48] = 0;
   out_3190334320092666102[49] = 0;
   out_3190334320092666102[50] = 1;
   out_3190334320092666102[51] = 0;
   out_3190334320092666102[52] = 0;
   out_3190334320092666102[53] = 1;
}
void h_13(double *state, double *unused, double *out_1070052259346719453) {
   out_1070052259346719453[0] = state[3];
   out_1070052259346719453[1] = state[4];
   out_1070052259346719453[2] = state[5];
}
void H_13(double *state, double *unused, double *out_8800579419872730004) {
   out_8800579419872730004[0] = 0;
   out_8800579419872730004[1] = 0;
   out_8800579419872730004[2] = 0;
   out_8800579419872730004[3] = 1;
   out_8800579419872730004[4] = 0;
   out_8800579419872730004[5] = 0;
   out_8800579419872730004[6] = 0;
   out_8800579419872730004[7] = 0;
   out_8800579419872730004[8] = 0;
   out_8800579419872730004[9] = 0;
   out_8800579419872730004[10] = 0;
   out_8800579419872730004[11] = 0;
   out_8800579419872730004[12] = 0;
   out_8800579419872730004[13] = 0;
   out_8800579419872730004[14] = 0;
   out_8800579419872730004[15] = 0;
   out_8800579419872730004[16] = 0;
   out_8800579419872730004[17] = 0;
   out_8800579419872730004[18] = 0;
   out_8800579419872730004[19] = 0;
   out_8800579419872730004[20] = 0;
   out_8800579419872730004[21] = 0;
   out_8800579419872730004[22] = 1;
   out_8800579419872730004[23] = 0;
   out_8800579419872730004[24] = 0;
   out_8800579419872730004[25] = 0;
   out_8800579419872730004[26] = 0;
   out_8800579419872730004[27] = 0;
   out_8800579419872730004[28] = 0;
   out_8800579419872730004[29] = 0;
   out_8800579419872730004[30] = 0;
   out_8800579419872730004[31] = 0;
   out_8800579419872730004[32] = 0;
   out_8800579419872730004[33] = 0;
   out_8800579419872730004[34] = 0;
   out_8800579419872730004[35] = 0;
   out_8800579419872730004[36] = 0;
   out_8800579419872730004[37] = 0;
   out_8800579419872730004[38] = 0;
   out_8800579419872730004[39] = 0;
   out_8800579419872730004[40] = 0;
   out_8800579419872730004[41] = 1;
   out_8800579419872730004[42] = 0;
   out_8800579419872730004[43] = 0;
   out_8800579419872730004[44] = 0;
   out_8800579419872730004[45] = 0;
   out_8800579419872730004[46] = 0;
   out_8800579419872730004[47] = 0;
   out_8800579419872730004[48] = 0;
   out_8800579419872730004[49] = 0;
   out_8800579419872730004[50] = 0;
   out_8800579419872730004[51] = 0;
   out_8800579419872730004[52] = 0;
   out_8800579419872730004[53] = 0;
}
void h_14(double *state, double *unused, double *out_3428767572042643384) {
   out_3428767572042643384[0] = state[6];
   out_3428767572042643384[1] = state[7];
   out_3428767572042643384[2] = state[8];
}
void H_14(double *state, double *unused, double *out_8049612388865578276) {
   out_8049612388865578276[0] = 0;
   out_8049612388865578276[1] = 0;
   out_8049612388865578276[2] = 0;
   out_8049612388865578276[3] = 0;
   out_8049612388865578276[4] = 0;
   out_8049612388865578276[5] = 0;
   out_8049612388865578276[6] = 1;
   out_8049612388865578276[7] = 0;
   out_8049612388865578276[8] = 0;
   out_8049612388865578276[9] = 0;
   out_8049612388865578276[10] = 0;
   out_8049612388865578276[11] = 0;
   out_8049612388865578276[12] = 0;
   out_8049612388865578276[13] = 0;
   out_8049612388865578276[14] = 0;
   out_8049612388865578276[15] = 0;
   out_8049612388865578276[16] = 0;
   out_8049612388865578276[17] = 0;
   out_8049612388865578276[18] = 0;
   out_8049612388865578276[19] = 0;
   out_8049612388865578276[20] = 0;
   out_8049612388865578276[21] = 0;
   out_8049612388865578276[22] = 0;
   out_8049612388865578276[23] = 0;
   out_8049612388865578276[24] = 0;
   out_8049612388865578276[25] = 1;
   out_8049612388865578276[26] = 0;
   out_8049612388865578276[27] = 0;
   out_8049612388865578276[28] = 0;
   out_8049612388865578276[29] = 0;
   out_8049612388865578276[30] = 0;
   out_8049612388865578276[31] = 0;
   out_8049612388865578276[32] = 0;
   out_8049612388865578276[33] = 0;
   out_8049612388865578276[34] = 0;
   out_8049612388865578276[35] = 0;
   out_8049612388865578276[36] = 0;
   out_8049612388865578276[37] = 0;
   out_8049612388865578276[38] = 0;
   out_8049612388865578276[39] = 0;
   out_8049612388865578276[40] = 0;
   out_8049612388865578276[41] = 0;
   out_8049612388865578276[42] = 0;
   out_8049612388865578276[43] = 0;
   out_8049612388865578276[44] = 1;
   out_8049612388865578276[45] = 0;
   out_8049612388865578276[46] = 0;
   out_8049612388865578276[47] = 0;
   out_8049612388865578276[48] = 0;
   out_8049612388865578276[49] = 0;
   out_8049612388865578276[50] = 0;
   out_8049612388865578276[51] = 0;
   out_8049612388865578276[52] = 0;
   out_8049612388865578276[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_5727667037901716489) {
  err_fun(nom_x, delta_x, out_5727667037901716489);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6198627751216921880) {
  inv_err_fun(nom_x, true_x, out_6198627751216921880);
}
void pose_H_mod_fun(double *state, double *out_8253589554410785672) {
  H_mod_fun(state, out_8253589554410785672);
}
void pose_f_fun(double *state, double dt, double *out_6523760126628443948) {
  f_fun(state,  dt, out_6523760126628443948);
}
void pose_F_fun(double *state, double dt, double *out_2397429393732395161) {
  F_fun(state,  dt, out_2397429393732395161);
}
void pose_h_4(double *state, double *unused, double *out_8382275560088012236) {
  h_4(state, unused, out_8382275560088012236);
}
void pose_H_4(double *state, double *unused, double *out_6433890828504488811) {
  H_4(state, unused, out_6433890828504488811);
}
void pose_h_10(double *state, double *unused, double *out_3474458810352115326) {
  h_10(state, unused, out_3474458810352115326);
}
void pose_H_10(double *state, double *unused, double *out_3190334320092666102) {
  H_10(state, unused, out_3190334320092666102);
}
void pose_h_13(double *state, double *unused, double *out_1070052259346719453) {
  h_13(state, unused, out_1070052259346719453);
}
void pose_H_13(double *state, double *unused, double *out_8800579419872730004) {
  H_13(state, unused, out_8800579419872730004);
}
void pose_h_14(double *state, double *unused, double *out_3428767572042643384) {
  h_14(state, unused, out_3428767572042643384);
}
void pose_H_14(double *state, double *unused, double *out_8049612388865578276) {
  H_14(state, unused, out_8049612388865578276);
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
