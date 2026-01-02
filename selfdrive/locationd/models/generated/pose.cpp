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
void err_fun(double *nom_x, double *delta_x, double *out_3495798777798121399) {
   out_3495798777798121399[0] = delta_x[0] + nom_x[0];
   out_3495798777798121399[1] = delta_x[1] + nom_x[1];
   out_3495798777798121399[2] = delta_x[2] + nom_x[2];
   out_3495798777798121399[3] = delta_x[3] + nom_x[3];
   out_3495798777798121399[4] = delta_x[4] + nom_x[4];
   out_3495798777798121399[5] = delta_x[5] + nom_x[5];
   out_3495798777798121399[6] = delta_x[6] + nom_x[6];
   out_3495798777798121399[7] = delta_x[7] + nom_x[7];
   out_3495798777798121399[8] = delta_x[8] + nom_x[8];
   out_3495798777798121399[9] = delta_x[9] + nom_x[9];
   out_3495798777798121399[10] = delta_x[10] + nom_x[10];
   out_3495798777798121399[11] = delta_x[11] + nom_x[11];
   out_3495798777798121399[12] = delta_x[12] + nom_x[12];
   out_3495798777798121399[13] = delta_x[13] + nom_x[13];
   out_3495798777798121399[14] = delta_x[14] + nom_x[14];
   out_3495798777798121399[15] = delta_x[15] + nom_x[15];
   out_3495798777798121399[16] = delta_x[16] + nom_x[16];
   out_3495798777798121399[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6491638533999781681) {
   out_6491638533999781681[0] = -nom_x[0] + true_x[0];
   out_6491638533999781681[1] = -nom_x[1] + true_x[1];
   out_6491638533999781681[2] = -nom_x[2] + true_x[2];
   out_6491638533999781681[3] = -nom_x[3] + true_x[3];
   out_6491638533999781681[4] = -nom_x[4] + true_x[4];
   out_6491638533999781681[5] = -nom_x[5] + true_x[5];
   out_6491638533999781681[6] = -nom_x[6] + true_x[6];
   out_6491638533999781681[7] = -nom_x[7] + true_x[7];
   out_6491638533999781681[8] = -nom_x[8] + true_x[8];
   out_6491638533999781681[9] = -nom_x[9] + true_x[9];
   out_6491638533999781681[10] = -nom_x[10] + true_x[10];
   out_6491638533999781681[11] = -nom_x[11] + true_x[11];
   out_6491638533999781681[12] = -nom_x[12] + true_x[12];
   out_6491638533999781681[13] = -nom_x[13] + true_x[13];
   out_6491638533999781681[14] = -nom_x[14] + true_x[14];
   out_6491638533999781681[15] = -nom_x[15] + true_x[15];
   out_6491638533999781681[16] = -nom_x[16] + true_x[16];
   out_6491638533999781681[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1193861252693843199) {
   out_1193861252693843199[0] = 1.0;
   out_1193861252693843199[1] = 0.0;
   out_1193861252693843199[2] = 0.0;
   out_1193861252693843199[3] = 0.0;
   out_1193861252693843199[4] = 0.0;
   out_1193861252693843199[5] = 0.0;
   out_1193861252693843199[6] = 0.0;
   out_1193861252693843199[7] = 0.0;
   out_1193861252693843199[8] = 0.0;
   out_1193861252693843199[9] = 0.0;
   out_1193861252693843199[10] = 0.0;
   out_1193861252693843199[11] = 0.0;
   out_1193861252693843199[12] = 0.0;
   out_1193861252693843199[13] = 0.0;
   out_1193861252693843199[14] = 0.0;
   out_1193861252693843199[15] = 0.0;
   out_1193861252693843199[16] = 0.0;
   out_1193861252693843199[17] = 0.0;
   out_1193861252693843199[18] = 0.0;
   out_1193861252693843199[19] = 1.0;
   out_1193861252693843199[20] = 0.0;
   out_1193861252693843199[21] = 0.0;
   out_1193861252693843199[22] = 0.0;
   out_1193861252693843199[23] = 0.0;
   out_1193861252693843199[24] = 0.0;
   out_1193861252693843199[25] = 0.0;
   out_1193861252693843199[26] = 0.0;
   out_1193861252693843199[27] = 0.0;
   out_1193861252693843199[28] = 0.0;
   out_1193861252693843199[29] = 0.0;
   out_1193861252693843199[30] = 0.0;
   out_1193861252693843199[31] = 0.0;
   out_1193861252693843199[32] = 0.0;
   out_1193861252693843199[33] = 0.0;
   out_1193861252693843199[34] = 0.0;
   out_1193861252693843199[35] = 0.0;
   out_1193861252693843199[36] = 0.0;
   out_1193861252693843199[37] = 0.0;
   out_1193861252693843199[38] = 1.0;
   out_1193861252693843199[39] = 0.0;
   out_1193861252693843199[40] = 0.0;
   out_1193861252693843199[41] = 0.0;
   out_1193861252693843199[42] = 0.0;
   out_1193861252693843199[43] = 0.0;
   out_1193861252693843199[44] = 0.0;
   out_1193861252693843199[45] = 0.0;
   out_1193861252693843199[46] = 0.0;
   out_1193861252693843199[47] = 0.0;
   out_1193861252693843199[48] = 0.0;
   out_1193861252693843199[49] = 0.0;
   out_1193861252693843199[50] = 0.0;
   out_1193861252693843199[51] = 0.0;
   out_1193861252693843199[52] = 0.0;
   out_1193861252693843199[53] = 0.0;
   out_1193861252693843199[54] = 0.0;
   out_1193861252693843199[55] = 0.0;
   out_1193861252693843199[56] = 0.0;
   out_1193861252693843199[57] = 1.0;
   out_1193861252693843199[58] = 0.0;
   out_1193861252693843199[59] = 0.0;
   out_1193861252693843199[60] = 0.0;
   out_1193861252693843199[61] = 0.0;
   out_1193861252693843199[62] = 0.0;
   out_1193861252693843199[63] = 0.0;
   out_1193861252693843199[64] = 0.0;
   out_1193861252693843199[65] = 0.0;
   out_1193861252693843199[66] = 0.0;
   out_1193861252693843199[67] = 0.0;
   out_1193861252693843199[68] = 0.0;
   out_1193861252693843199[69] = 0.0;
   out_1193861252693843199[70] = 0.0;
   out_1193861252693843199[71] = 0.0;
   out_1193861252693843199[72] = 0.0;
   out_1193861252693843199[73] = 0.0;
   out_1193861252693843199[74] = 0.0;
   out_1193861252693843199[75] = 0.0;
   out_1193861252693843199[76] = 1.0;
   out_1193861252693843199[77] = 0.0;
   out_1193861252693843199[78] = 0.0;
   out_1193861252693843199[79] = 0.0;
   out_1193861252693843199[80] = 0.0;
   out_1193861252693843199[81] = 0.0;
   out_1193861252693843199[82] = 0.0;
   out_1193861252693843199[83] = 0.0;
   out_1193861252693843199[84] = 0.0;
   out_1193861252693843199[85] = 0.0;
   out_1193861252693843199[86] = 0.0;
   out_1193861252693843199[87] = 0.0;
   out_1193861252693843199[88] = 0.0;
   out_1193861252693843199[89] = 0.0;
   out_1193861252693843199[90] = 0.0;
   out_1193861252693843199[91] = 0.0;
   out_1193861252693843199[92] = 0.0;
   out_1193861252693843199[93] = 0.0;
   out_1193861252693843199[94] = 0.0;
   out_1193861252693843199[95] = 1.0;
   out_1193861252693843199[96] = 0.0;
   out_1193861252693843199[97] = 0.0;
   out_1193861252693843199[98] = 0.0;
   out_1193861252693843199[99] = 0.0;
   out_1193861252693843199[100] = 0.0;
   out_1193861252693843199[101] = 0.0;
   out_1193861252693843199[102] = 0.0;
   out_1193861252693843199[103] = 0.0;
   out_1193861252693843199[104] = 0.0;
   out_1193861252693843199[105] = 0.0;
   out_1193861252693843199[106] = 0.0;
   out_1193861252693843199[107] = 0.0;
   out_1193861252693843199[108] = 0.0;
   out_1193861252693843199[109] = 0.0;
   out_1193861252693843199[110] = 0.0;
   out_1193861252693843199[111] = 0.0;
   out_1193861252693843199[112] = 0.0;
   out_1193861252693843199[113] = 0.0;
   out_1193861252693843199[114] = 1.0;
   out_1193861252693843199[115] = 0.0;
   out_1193861252693843199[116] = 0.0;
   out_1193861252693843199[117] = 0.0;
   out_1193861252693843199[118] = 0.0;
   out_1193861252693843199[119] = 0.0;
   out_1193861252693843199[120] = 0.0;
   out_1193861252693843199[121] = 0.0;
   out_1193861252693843199[122] = 0.0;
   out_1193861252693843199[123] = 0.0;
   out_1193861252693843199[124] = 0.0;
   out_1193861252693843199[125] = 0.0;
   out_1193861252693843199[126] = 0.0;
   out_1193861252693843199[127] = 0.0;
   out_1193861252693843199[128] = 0.0;
   out_1193861252693843199[129] = 0.0;
   out_1193861252693843199[130] = 0.0;
   out_1193861252693843199[131] = 0.0;
   out_1193861252693843199[132] = 0.0;
   out_1193861252693843199[133] = 1.0;
   out_1193861252693843199[134] = 0.0;
   out_1193861252693843199[135] = 0.0;
   out_1193861252693843199[136] = 0.0;
   out_1193861252693843199[137] = 0.0;
   out_1193861252693843199[138] = 0.0;
   out_1193861252693843199[139] = 0.0;
   out_1193861252693843199[140] = 0.0;
   out_1193861252693843199[141] = 0.0;
   out_1193861252693843199[142] = 0.0;
   out_1193861252693843199[143] = 0.0;
   out_1193861252693843199[144] = 0.0;
   out_1193861252693843199[145] = 0.0;
   out_1193861252693843199[146] = 0.0;
   out_1193861252693843199[147] = 0.0;
   out_1193861252693843199[148] = 0.0;
   out_1193861252693843199[149] = 0.0;
   out_1193861252693843199[150] = 0.0;
   out_1193861252693843199[151] = 0.0;
   out_1193861252693843199[152] = 1.0;
   out_1193861252693843199[153] = 0.0;
   out_1193861252693843199[154] = 0.0;
   out_1193861252693843199[155] = 0.0;
   out_1193861252693843199[156] = 0.0;
   out_1193861252693843199[157] = 0.0;
   out_1193861252693843199[158] = 0.0;
   out_1193861252693843199[159] = 0.0;
   out_1193861252693843199[160] = 0.0;
   out_1193861252693843199[161] = 0.0;
   out_1193861252693843199[162] = 0.0;
   out_1193861252693843199[163] = 0.0;
   out_1193861252693843199[164] = 0.0;
   out_1193861252693843199[165] = 0.0;
   out_1193861252693843199[166] = 0.0;
   out_1193861252693843199[167] = 0.0;
   out_1193861252693843199[168] = 0.0;
   out_1193861252693843199[169] = 0.0;
   out_1193861252693843199[170] = 0.0;
   out_1193861252693843199[171] = 1.0;
   out_1193861252693843199[172] = 0.0;
   out_1193861252693843199[173] = 0.0;
   out_1193861252693843199[174] = 0.0;
   out_1193861252693843199[175] = 0.0;
   out_1193861252693843199[176] = 0.0;
   out_1193861252693843199[177] = 0.0;
   out_1193861252693843199[178] = 0.0;
   out_1193861252693843199[179] = 0.0;
   out_1193861252693843199[180] = 0.0;
   out_1193861252693843199[181] = 0.0;
   out_1193861252693843199[182] = 0.0;
   out_1193861252693843199[183] = 0.0;
   out_1193861252693843199[184] = 0.0;
   out_1193861252693843199[185] = 0.0;
   out_1193861252693843199[186] = 0.0;
   out_1193861252693843199[187] = 0.0;
   out_1193861252693843199[188] = 0.0;
   out_1193861252693843199[189] = 0.0;
   out_1193861252693843199[190] = 1.0;
   out_1193861252693843199[191] = 0.0;
   out_1193861252693843199[192] = 0.0;
   out_1193861252693843199[193] = 0.0;
   out_1193861252693843199[194] = 0.0;
   out_1193861252693843199[195] = 0.0;
   out_1193861252693843199[196] = 0.0;
   out_1193861252693843199[197] = 0.0;
   out_1193861252693843199[198] = 0.0;
   out_1193861252693843199[199] = 0.0;
   out_1193861252693843199[200] = 0.0;
   out_1193861252693843199[201] = 0.0;
   out_1193861252693843199[202] = 0.0;
   out_1193861252693843199[203] = 0.0;
   out_1193861252693843199[204] = 0.0;
   out_1193861252693843199[205] = 0.0;
   out_1193861252693843199[206] = 0.0;
   out_1193861252693843199[207] = 0.0;
   out_1193861252693843199[208] = 0.0;
   out_1193861252693843199[209] = 1.0;
   out_1193861252693843199[210] = 0.0;
   out_1193861252693843199[211] = 0.0;
   out_1193861252693843199[212] = 0.0;
   out_1193861252693843199[213] = 0.0;
   out_1193861252693843199[214] = 0.0;
   out_1193861252693843199[215] = 0.0;
   out_1193861252693843199[216] = 0.0;
   out_1193861252693843199[217] = 0.0;
   out_1193861252693843199[218] = 0.0;
   out_1193861252693843199[219] = 0.0;
   out_1193861252693843199[220] = 0.0;
   out_1193861252693843199[221] = 0.0;
   out_1193861252693843199[222] = 0.0;
   out_1193861252693843199[223] = 0.0;
   out_1193861252693843199[224] = 0.0;
   out_1193861252693843199[225] = 0.0;
   out_1193861252693843199[226] = 0.0;
   out_1193861252693843199[227] = 0.0;
   out_1193861252693843199[228] = 1.0;
   out_1193861252693843199[229] = 0.0;
   out_1193861252693843199[230] = 0.0;
   out_1193861252693843199[231] = 0.0;
   out_1193861252693843199[232] = 0.0;
   out_1193861252693843199[233] = 0.0;
   out_1193861252693843199[234] = 0.0;
   out_1193861252693843199[235] = 0.0;
   out_1193861252693843199[236] = 0.0;
   out_1193861252693843199[237] = 0.0;
   out_1193861252693843199[238] = 0.0;
   out_1193861252693843199[239] = 0.0;
   out_1193861252693843199[240] = 0.0;
   out_1193861252693843199[241] = 0.0;
   out_1193861252693843199[242] = 0.0;
   out_1193861252693843199[243] = 0.0;
   out_1193861252693843199[244] = 0.0;
   out_1193861252693843199[245] = 0.0;
   out_1193861252693843199[246] = 0.0;
   out_1193861252693843199[247] = 1.0;
   out_1193861252693843199[248] = 0.0;
   out_1193861252693843199[249] = 0.0;
   out_1193861252693843199[250] = 0.0;
   out_1193861252693843199[251] = 0.0;
   out_1193861252693843199[252] = 0.0;
   out_1193861252693843199[253] = 0.0;
   out_1193861252693843199[254] = 0.0;
   out_1193861252693843199[255] = 0.0;
   out_1193861252693843199[256] = 0.0;
   out_1193861252693843199[257] = 0.0;
   out_1193861252693843199[258] = 0.0;
   out_1193861252693843199[259] = 0.0;
   out_1193861252693843199[260] = 0.0;
   out_1193861252693843199[261] = 0.0;
   out_1193861252693843199[262] = 0.0;
   out_1193861252693843199[263] = 0.0;
   out_1193861252693843199[264] = 0.0;
   out_1193861252693843199[265] = 0.0;
   out_1193861252693843199[266] = 1.0;
   out_1193861252693843199[267] = 0.0;
   out_1193861252693843199[268] = 0.0;
   out_1193861252693843199[269] = 0.0;
   out_1193861252693843199[270] = 0.0;
   out_1193861252693843199[271] = 0.0;
   out_1193861252693843199[272] = 0.0;
   out_1193861252693843199[273] = 0.0;
   out_1193861252693843199[274] = 0.0;
   out_1193861252693843199[275] = 0.0;
   out_1193861252693843199[276] = 0.0;
   out_1193861252693843199[277] = 0.0;
   out_1193861252693843199[278] = 0.0;
   out_1193861252693843199[279] = 0.0;
   out_1193861252693843199[280] = 0.0;
   out_1193861252693843199[281] = 0.0;
   out_1193861252693843199[282] = 0.0;
   out_1193861252693843199[283] = 0.0;
   out_1193861252693843199[284] = 0.0;
   out_1193861252693843199[285] = 1.0;
   out_1193861252693843199[286] = 0.0;
   out_1193861252693843199[287] = 0.0;
   out_1193861252693843199[288] = 0.0;
   out_1193861252693843199[289] = 0.0;
   out_1193861252693843199[290] = 0.0;
   out_1193861252693843199[291] = 0.0;
   out_1193861252693843199[292] = 0.0;
   out_1193861252693843199[293] = 0.0;
   out_1193861252693843199[294] = 0.0;
   out_1193861252693843199[295] = 0.0;
   out_1193861252693843199[296] = 0.0;
   out_1193861252693843199[297] = 0.0;
   out_1193861252693843199[298] = 0.0;
   out_1193861252693843199[299] = 0.0;
   out_1193861252693843199[300] = 0.0;
   out_1193861252693843199[301] = 0.0;
   out_1193861252693843199[302] = 0.0;
   out_1193861252693843199[303] = 0.0;
   out_1193861252693843199[304] = 1.0;
   out_1193861252693843199[305] = 0.0;
   out_1193861252693843199[306] = 0.0;
   out_1193861252693843199[307] = 0.0;
   out_1193861252693843199[308] = 0.0;
   out_1193861252693843199[309] = 0.0;
   out_1193861252693843199[310] = 0.0;
   out_1193861252693843199[311] = 0.0;
   out_1193861252693843199[312] = 0.0;
   out_1193861252693843199[313] = 0.0;
   out_1193861252693843199[314] = 0.0;
   out_1193861252693843199[315] = 0.0;
   out_1193861252693843199[316] = 0.0;
   out_1193861252693843199[317] = 0.0;
   out_1193861252693843199[318] = 0.0;
   out_1193861252693843199[319] = 0.0;
   out_1193861252693843199[320] = 0.0;
   out_1193861252693843199[321] = 0.0;
   out_1193861252693843199[322] = 0.0;
   out_1193861252693843199[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_2162222754272876291) {
   out_2162222754272876291[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_2162222754272876291[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_2162222754272876291[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_2162222754272876291[3] = dt*state[12] + state[3];
   out_2162222754272876291[4] = dt*state[13] + state[4];
   out_2162222754272876291[5] = dt*state[14] + state[5];
   out_2162222754272876291[6] = state[6];
   out_2162222754272876291[7] = state[7];
   out_2162222754272876291[8] = state[8];
   out_2162222754272876291[9] = state[9];
   out_2162222754272876291[10] = state[10];
   out_2162222754272876291[11] = state[11];
   out_2162222754272876291[12] = state[12];
   out_2162222754272876291[13] = state[13];
   out_2162222754272876291[14] = state[14];
   out_2162222754272876291[15] = state[15];
   out_2162222754272876291[16] = state[16];
   out_2162222754272876291[17] = state[17];
}
void F_fun(double *state, double dt, double *out_577996268896955016) {
   out_577996268896955016[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_577996268896955016[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_577996268896955016[2] = 0;
   out_577996268896955016[3] = 0;
   out_577996268896955016[4] = 0;
   out_577996268896955016[5] = 0;
   out_577996268896955016[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_577996268896955016[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_577996268896955016[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_577996268896955016[9] = 0;
   out_577996268896955016[10] = 0;
   out_577996268896955016[11] = 0;
   out_577996268896955016[12] = 0;
   out_577996268896955016[13] = 0;
   out_577996268896955016[14] = 0;
   out_577996268896955016[15] = 0;
   out_577996268896955016[16] = 0;
   out_577996268896955016[17] = 0;
   out_577996268896955016[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_577996268896955016[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_577996268896955016[20] = 0;
   out_577996268896955016[21] = 0;
   out_577996268896955016[22] = 0;
   out_577996268896955016[23] = 0;
   out_577996268896955016[24] = 0;
   out_577996268896955016[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_577996268896955016[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_577996268896955016[27] = 0;
   out_577996268896955016[28] = 0;
   out_577996268896955016[29] = 0;
   out_577996268896955016[30] = 0;
   out_577996268896955016[31] = 0;
   out_577996268896955016[32] = 0;
   out_577996268896955016[33] = 0;
   out_577996268896955016[34] = 0;
   out_577996268896955016[35] = 0;
   out_577996268896955016[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_577996268896955016[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_577996268896955016[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_577996268896955016[39] = 0;
   out_577996268896955016[40] = 0;
   out_577996268896955016[41] = 0;
   out_577996268896955016[42] = 0;
   out_577996268896955016[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_577996268896955016[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_577996268896955016[45] = 0;
   out_577996268896955016[46] = 0;
   out_577996268896955016[47] = 0;
   out_577996268896955016[48] = 0;
   out_577996268896955016[49] = 0;
   out_577996268896955016[50] = 0;
   out_577996268896955016[51] = 0;
   out_577996268896955016[52] = 0;
   out_577996268896955016[53] = 0;
   out_577996268896955016[54] = 0;
   out_577996268896955016[55] = 0;
   out_577996268896955016[56] = 0;
   out_577996268896955016[57] = 1;
   out_577996268896955016[58] = 0;
   out_577996268896955016[59] = 0;
   out_577996268896955016[60] = 0;
   out_577996268896955016[61] = 0;
   out_577996268896955016[62] = 0;
   out_577996268896955016[63] = 0;
   out_577996268896955016[64] = 0;
   out_577996268896955016[65] = 0;
   out_577996268896955016[66] = dt;
   out_577996268896955016[67] = 0;
   out_577996268896955016[68] = 0;
   out_577996268896955016[69] = 0;
   out_577996268896955016[70] = 0;
   out_577996268896955016[71] = 0;
   out_577996268896955016[72] = 0;
   out_577996268896955016[73] = 0;
   out_577996268896955016[74] = 0;
   out_577996268896955016[75] = 0;
   out_577996268896955016[76] = 1;
   out_577996268896955016[77] = 0;
   out_577996268896955016[78] = 0;
   out_577996268896955016[79] = 0;
   out_577996268896955016[80] = 0;
   out_577996268896955016[81] = 0;
   out_577996268896955016[82] = 0;
   out_577996268896955016[83] = 0;
   out_577996268896955016[84] = 0;
   out_577996268896955016[85] = dt;
   out_577996268896955016[86] = 0;
   out_577996268896955016[87] = 0;
   out_577996268896955016[88] = 0;
   out_577996268896955016[89] = 0;
   out_577996268896955016[90] = 0;
   out_577996268896955016[91] = 0;
   out_577996268896955016[92] = 0;
   out_577996268896955016[93] = 0;
   out_577996268896955016[94] = 0;
   out_577996268896955016[95] = 1;
   out_577996268896955016[96] = 0;
   out_577996268896955016[97] = 0;
   out_577996268896955016[98] = 0;
   out_577996268896955016[99] = 0;
   out_577996268896955016[100] = 0;
   out_577996268896955016[101] = 0;
   out_577996268896955016[102] = 0;
   out_577996268896955016[103] = 0;
   out_577996268896955016[104] = dt;
   out_577996268896955016[105] = 0;
   out_577996268896955016[106] = 0;
   out_577996268896955016[107] = 0;
   out_577996268896955016[108] = 0;
   out_577996268896955016[109] = 0;
   out_577996268896955016[110] = 0;
   out_577996268896955016[111] = 0;
   out_577996268896955016[112] = 0;
   out_577996268896955016[113] = 0;
   out_577996268896955016[114] = 1;
   out_577996268896955016[115] = 0;
   out_577996268896955016[116] = 0;
   out_577996268896955016[117] = 0;
   out_577996268896955016[118] = 0;
   out_577996268896955016[119] = 0;
   out_577996268896955016[120] = 0;
   out_577996268896955016[121] = 0;
   out_577996268896955016[122] = 0;
   out_577996268896955016[123] = 0;
   out_577996268896955016[124] = 0;
   out_577996268896955016[125] = 0;
   out_577996268896955016[126] = 0;
   out_577996268896955016[127] = 0;
   out_577996268896955016[128] = 0;
   out_577996268896955016[129] = 0;
   out_577996268896955016[130] = 0;
   out_577996268896955016[131] = 0;
   out_577996268896955016[132] = 0;
   out_577996268896955016[133] = 1;
   out_577996268896955016[134] = 0;
   out_577996268896955016[135] = 0;
   out_577996268896955016[136] = 0;
   out_577996268896955016[137] = 0;
   out_577996268896955016[138] = 0;
   out_577996268896955016[139] = 0;
   out_577996268896955016[140] = 0;
   out_577996268896955016[141] = 0;
   out_577996268896955016[142] = 0;
   out_577996268896955016[143] = 0;
   out_577996268896955016[144] = 0;
   out_577996268896955016[145] = 0;
   out_577996268896955016[146] = 0;
   out_577996268896955016[147] = 0;
   out_577996268896955016[148] = 0;
   out_577996268896955016[149] = 0;
   out_577996268896955016[150] = 0;
   out_577996268896955016[151] = 0;
   out_577996268896955016[152] = 1;
   out_577996268896955016[153] = 0;
   out_577996268896955016[154] = 0;
   out_577996268896955016[155] = 0;
   out_577996268896955016[156] = 0;
   out_577996268896955016[157] = 0;
   out_577996268896955016[158] = 0;
   out_577996268896955016[159] = 0;
   out_577996268896955016[160] = 0;
   out_577996268896955016[161] = 0;
   out_577996268896955016[162] = 0;
   out_577996268896955016[163] = 0;
   out_577996268896955016[164] = 0;
   out_577996268896955016[165] = 0;
   out_577996268896955016[166] = 0;
   out_577996268896955016[167] = 0;
   out_577996268896955016[168] = 0;
   out_577996268896955016[169] = 0;
   out_577996268896955016[170] = 0;
   out_577996268896955016[171] = 1;
   out_577996268896955016[172] = 0;
   out_577996268896955016[173] = 0;
   out_577996268896955016[174] = 0;
   out_577996268896955016[175] = 0;
   out_577996268896955016[176] = 0;
   out_577996268896955016[177] = 0;
   out_577996268896955016[178] = 0;
   out_577996268896955016[179] = 0;
   out_577996268896955016[180] = 0;
   out_577996268896955016[181] = 0;
   out_577996268896955016[182] = 0;
   out_577996268896955016[183] = 0;
   out_577996268896955016[184] = 0;
   out_577996268896955016[185] = 0;
   out_577996268896955016[186] = 0;
   out_577996268896955016[187] = 0;
   out_577996268896955016[188] = 0;
   out_577996268896955016[189] = 0;
   out_577996268896955016[190] = 1;
   out_577996268896955016[191] = 0;
   out_577996268896955016[192] = 0;
   out_577996268896955016[193] = 0;
   out_577996268896955016[194] = 0;
   out_577996268896955016[195] = 0;
   out_577996268896955016[196] = 0;
   out_577996268896955016[197] = 0;
   out_577996268896955016[198] = 0;
   out_577996268896955016[199] = 0;
   out_577996268896955016[200] = 0;
   out_577996268896955016[201] = 0;
   out_577996268896955016[202] = 0;
   out_577996268896955016[203] = 0;
   out_577996268896955016[204] = 0;
   out_577996268896955016[205] = 0;
   out_577996268896955016[206] = 0;
   out_577996268896955016[207] = 0;
   out_577996268896955016[208] = 0;
   out_577996268896955016[209] = 1;
   out_577996268896955016[210] = 0;
   out_577996268896955016[211] = 0;
   out_577996268896955016[212] = 0;
   out_577996268896955016[213] = 0;
   out_577996268896955016[214] = 0;
   out_577996268896955016[215] = 0;
   out_577996268896955016[216] = 0;
   out_577996268896955016[217] = 0;
   out_577996268896955016[218] = 0;
   out_577996268896955016[219] = 0;
   out_577996268896955016[220] = 0;
   out_577996268896955016[221] = 0;
   out_577996268896955016[222] = 0;
   out_577996268896955016[223] = 0;
   out_577996268896955016[224] = 0;
   out_577996268896955016[225] = 0;
   out_577996268896955016[226] = 0;
   out_577996268896955016[227] = 0;
   out_577996268896955016[228] = 1;
   out_577996268896955016[229] = 0;
   out_577996268896955016[230] = 0;
   out_577996268896955016[231] = 0;
   out_577996268896955016[232] = 0;
   out_577996268896955016[233] = 0;
   out_577996268896955016[234] = 0;
   out_577996268896955016[235] = 0;
   out_577996268896955016[236] = 0;
   out_577996268896955016[237] = 0;
   out_577996268896955016[238] = 0;
   out_577996268896955016[239] = 0;
   out_577996268896955016[240] = 0;
   out_577996268896955016[241] = 0;
   out_577996268896955016[242] = 0;
   out_577996268896955016[243] = 0;
   out_577996268896955016[244] = 0;
   out_577996268896955016[245] = 0;
   out_577996268896955016[246] = 0;
   out_577996268896955016[247] = 1;
   out_577996268896955016[248] = 0;
   out_577996268896955016[249] = 0;
   out_577996268896955016[250] = 0;
   out_577996268896955016[251] = 0;
   out_577996268896955016[252] = 0;
   out_577996268896955016[253] = 0;
   out_577996268896955016[254] = 0;
   out_577996268896955016[255] = 0;
   out_577996268896955016[256] = 0;
   out_577996268896955016[257] = 0;
   out_577996268896955016[258] = 0;
   out_577996268896955016[259] = 0;
   out_577996268896955016[260] = 0;
   out_577996268896955016[261] = 0;
   out_577996268896955016[262] = 0;
   out_577996268896955016[263] = 0;
   out_577996268896955016[264] = 0;
   out_577996268896955016[265] = 0;
   out_577996268896955016[266] = 1;
   out_577996268896955016[267] = 0;
   out_577996268896955016[268] = 0;
   out_577996268896955016[269] = 0;
   out_577996268896955016[270] = 0;
   out_577996268896955016[271] = 0;
   out_577996268896955016[272] = 0;
   out_577996268896955016[273] = 0;
   out_577996268896955016[274] = 0;
   out_577996268896955016[275] = 0;
   out_577996268896955016[276] = 0;
   out_577996268896955016[277] = 0;
   out_577996268896955016[278] = 0;
   out_577996268896955016[279] = 0;
   out_577996268896955016[280] = 0;
   out_577996268896955016[281] = 0;
   out_577996268896955016[282] = 0;
   out_577996268896955016[283] = 0;
   out_577996268896955016[284] = 0;
   out_577996268896955016[285] = 1;
   out_577996268896955016[286] = 0;
   out_577996268896955016[287] = 0;
   out_577996268896955016[288] = 0;
   out_577996268896955016[289] = 0;
   out_577996268896955016[290] = 0;
   out_577996268896955016[291] = 0;
   out_577996268896955016[292] = 0;
   out_577996268896955016[293] = 0;
   out_577996268896955016[294] = 0;
   out_577996268896955016[295] = 0;
   out_577996268896955016[296] = 0;
   out_577996268896955016[297] = 0;
   out_577996268896955016[298] = 0;
   out_577996268896955016[299] = 0;
   out_577996268896955016[300] = 0;
   out_577996268896955016[301] = 0;
   out_577996268896955016[302] = 0;
   out_577996268896955016[303] = 0;
   out_577996268896955016[304] = 1;
   out_577996268896955016[305] = 0;
   out_577996268896955016[306] = 0;
   out_577996268896955016[307] = 0;
   out_577996268896955016[308] = 0;
   out_577996268896955016[309] = 0;
   out_577996268896955016[310] = 0;
   out_577996268896955016[311] = 0;
   out_577996268896955016[312] = 0;
   out_577996268896955016[313] = 0;
   out_577996268896955016[314] = 0;
   out_577996268896955016[315] = 0;
   out_577996268896955016[316] = 0;
   out_577996268896955016[317] = 0;
   out_577996268896955016[318] = 0;
   out_577996268896955016[319] = 0;
   out_577996268896955016[320] = 0;
   out_577996268896955016[321] = 0;
   out_577996268896955016[322] = 0;
   out_577996268896955016[323] = 1;
}
void h_4(double *state, double *unused, double *out_2480001917824807916) {
   out_2480001917824807916[0] = state[6] + state[9];
   out_2480001917824807916[1] = state[7] + state[10];
   out_2480001917824807916[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4654220554058115522) {
   out_4654220554058115522[0] = 0;
   out_4654220554058115522[1] = 0;
   out_4654220554058115522[2] = 0;
   out_4654220554058115522[3] = 0;
   out_4654220554058115522[4] = 0;
   out_4654220554058115522[5] = 0;
   out_4654220554058115522[6] = 1;
   out_4654220554058115522[7] = 0;
   out_4654220554058115522[8] = 0;
   out_4654220554058115522[9] = 1;
   out_4654220554058115522[10] = 0;
   out_4654220554058115522[11] = 0;
   out_4654220554058115522[12] = 0;
   out_4654220554058115522[13] = 0;
   out_4654220554058115522[14] = 0;
   out_4654220554058115522[15] = 0;
   out_4654220554058115522[16] = 0;
   out_4654220554058115522[17] = 0;
   out_4654220554058115522[18] = 0;
   out_4654220554058115522[19] = 0;
   out_4654220554058115522[20] = 0;
   out_4654220554058115522[21] = 0;
   out_4654220554058115522[22] = 0;
   out_4654220554058115522[23] = 0;
   out_4654220554058115522[24] = 0;
   out_4654220554058115522[25] = 1;
   out_4654220554058115522[26] = 0;
   out_4654220554058115522[27] = 0;
   out_4654220554058115522[28] = 1;
   out_4654220554058115522[29] = 0;
   out_4654220554058115522[30] = 0;
   out_4654220554058115522[31] = 0;
   out_4654220554058115522[32] = 0;
   out_4654220554058115522[33] = 0;
   out_4654220554058115522[34] = 0;
   out_4654220554058115522[35] = 0;
   out_4654220554058115522[36] = 0;
   out_4654220554058115522[37] = 0;
   out_4654220554058115522[38] = 0;
   out_4654220554058115522[39] = 0;
   out_4654220554058115522[40] = 0;
   out_4654220554058115522[41] = 0;
   out_4654220554058115522[42] = 0;
   out_4654220554058115522[43] = 0;
   out_4654220554058115522[44] = 1;
   out_4654220554058115522[45] = 0;
   out_4654220554058115522[46] = 0;
   out_4654220554058115522[47] = 1;
   out_4654220554058115522[48] = 0;
   out_4654220554058115522[49] = 0;
   out_4654220554058115522[50] = 0;
   out_4654220554058115522[51] = 0;
   out_4654220554058115522[52] = 0;
   out_4654220554058115522[53] = 0;
}
void h_10(double *state, double *unused, double *out_6842199753574438663) {
   out_6842199753574438663[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_6842199753574438663[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_6842199753574438663[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_8460558184635204487) {
   out_8460558184635204487[0] = 0;
   out_8460558184635204487[1] = 9.8100000000000005*cos(state[1]);
   out_8460558184635204487[2] = 0;
   out_8460558184635204487[3] = 0;
   out_8460558184635204487[4] = -state[8];
   out_8460558184635204487[5] = state[7];
   out_8460558184635204487[6] = 0;
   out_8460558184635204487[7] = state[5];
   out_8460558184635204487[8] = -state[4];
   out_8460558184635204487[9] = 0;
   out_8460558184635204487[10] = 0;
   out_8460558184635204487[11] = 0;
   out_8460558184635204487[12] = 1;
   out_8460558184635204487[13] = 0;
   out_8460558184635204487[14] = 0;
   out_8460558184635204487[15] = 1;
   out_8460558184635204487[16] = 0;
   out_8460558184635204487[17] = 0;
   out_8460558184635204487[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_8460558184635204487[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_8460558184635204487[20] = 0;
   out_8460558184635204487[21] = state[8];
   out_8460558184635204487[22] = 0;
   out_8460558184635204487[23] = -state[6];
   out_8460558184635204487[24] = -state[5];
   out_8460558184635204487[25] = 0;
   out_8460558184635204487[26] = state[3];
   out_8460558184635204487[27] = 0;
   out_8460558184635204487[28] = 0;
   out_8460558184635204487[29] = 0;
   out_8460558184635204487[30] = 0;
   out_8460558184635204487[31] = 1;
   out_8460558184635204487[32] = 0;
   out_8460558184635204487[33] = 0;
   out_8460558184635204487[34] = 1;
   out_8460558184635204487[35] = 0;
   out_8460558184635204487[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_8460558184635204487[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_8460558184635204487[38] = 0;
   out_8460558184635204487[39] = -state[7];
   out_8460558184635204487[40] = state[6];
   out_8460558184635204487[41] = 0;
   out_8460558184635204487[42] = state[4];
   out_8460558184635204487[43] = -state[3];
   out_8460558184635204487[44] = 0;
   out_8460558184635204487[45] = 0;
   out_8460558184635204487[46] = 0;
   out_8460558184635204487[47] = 0;
   out_8460558184635204487[48] = 0;
   out_8460558184635204487[49] = 0;
   out_8460558184635204487[50] = 1;
   out_8460558184635204487[51] = 0;
   out_8460558184635204487[52] = 0;
   out_8460558184635204487[53] = 1;
}
void h_13(double *state, double *unused, double *out_7103896844424350049) {
   out_7103896844424350049[0] = state[3];
   out_7103896844424350049[1] = state[4];
   out_7103896844424350049[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1441946728725782721) {
   out_1441946728725782721[0] = 0;
   out_1441946728725782721[1] = 0;
   out_1441946728725782721[2] = 0;
   out_1441946728725782721[3] = 1;
   out_1441946728725782721[4] = 0;
   out_1441946728725782721[5] = 0;
   out_1441946728725782721[6] = 0;
   out_1441946728725782721[7] = 0;
   out_1441946728725782721[8] = 0;
   out_1441946728725782721[9] = 0;
   out_1441946728725782721[10] = 0;
   out_1441946728725782721[11] = 0;
   out_1441946728725782721[12] = 0;
   out_1441946728725782721[13] = 0;
   out_1441946728725782721[14] = 0;
   out_1441946728725782721[15] = 0;
   out_1441946728725782721[16] = 0;
   out_1441946728725782721[17] = 0;
   out_1441946728725782721[18] = 0;
   out_1441946728725782721[19] = 0;
   out_1441946728725782721[20] = 0;
   out_1441946728725782721[21] = 0;
   out_1441946728725782721[22] = 1;
   out_1441946728725782721[23] = 0;
   out_1441946728725782721[24] = 0;
   out_1441946728725782721[25] = 0;
   out_1441946728725782721[26] = 0;
   out_1441946728725782721[27] = 0;
   out_1441946728725782721[28] = 0;
   out_1441946728725782721[29] = 0;
   out_1441946728725782721[30] = 0;
   out_1441946728725782721[31] = 0;
   out_1441946728725782721[32] = 0;
   out_1441946728725782721[33] = 0;
   out_1441946728725782721[34] = 0;
   out_1441946728725782721[35] = 0;
   out_1441946728725782721[36] = 0;
   out_1441946728725782721[37] = 0;
   out_1441946728725782721[38] = 0;
   out_1441946728725782721[39] = 0;
   out_1441946728725782721[40] = 0;
   out_1441946728725782721[41] = 1;
   out_1441946728725782721[42] = 0;
   out_1441946728725782721[43] = 0;
   out_1441946728725782721[44] = 0;
   out_1441946728725782721[45] = 0;
   out_1441946728725782721[46] = 0;
   out_1441946728725782721[47] = 0;
   out_1441946728725782721[48] = 0;
   out_1441946728725782721[49] = 0;
   out_1441946728725782721[50] = 0;
   out_1441946728725782721[51] = 0;
   out_1441946728725782721[52] = 0;
   out_1441946728725782721[53] = 0;
}
void h_14(double *state, double *unused, double *out_2985573120985558604) {
   out_2985573120985558604[0] = state[6];
   out_2985573120985558604[1] = state[7];
   out_2985573120985558604[2] = state[8];
}
void H_14(double *state, double *unused, double *out_690979697718630993) {
   out_690979697718630993[0] = 0;
   out_690979697718630993[1] = 0;
   out_690979697718630993[2] = 0;
   out_690979697718630993[3] = 0;
   out_690979697718630993[4] = 0;
   out_690979697718630993[5] = 0;
   out_690979697718630993[6] = 1;
   out_690979697718630993[7] = 0;
   out_690979697718630993[8] = 0;
   out_690979697718630993[9] = 0;
   out_690979697718630993[10] = 0;
   out_690979697718630993[11] = 0;
   out_690979697718630993[12] = 0;
   out_690979697718630993[13] = 0;
   out_690979697718630993[14] = 0;
   out_690979697718630993[15] = 0;
   out_690979697718630993[16] = 0;
   out_690979697718630993[17] = 0;
   out_690979697718630993[18] = 0;
   out_690979697718630993[19] = 0;
   out_690979697718630993[20] = 0;
   out_690979697718630993[21] = 0;
   out_690979697718630993[22] = 0;
   out_690979697718630993[23] = 0;
   out_690979697718630993[24] = 0;
   out_690979697718630993[25] = 1;
   out_690979697718630993[26] = 0;
   out_690979697718630993[27] = 0;
   out_690979697718630993[28] = 0;
   out_690979697718630993[29] = 0;
   out_690979697718630993[30] = 0;
   out_690979697718630993[31] = 0;
   out_690979697718630993[32] = 0;
   out_690979697718630993[33] = 0;
   out_690979697718630993[34] = 0;
   out_690979697718630993[35] = 0;
   out_690979697718630993[36] = 0;
   out_690979697718630993[37] = 0;
   out_690979697718630993[38] = 0;
   out_690979697718630993[39] = 0;
   out_690979697718630993[40] = 0;
   out_690979697718630993[41] = 0;
   out_690979697718630993[42] = 0;
   out_690979697718630993[43] = 0;
   out_690979697718630993[44] = 1;
   out_690979697718630993[45] = 0;
   out_690979697718630993[46] = 0;
   out_690979697718630993[47] = 0;
   out_690979697718630993[48] = 0;
   out_690979697718630993[49] = 0;
   out_690979697718630993[50] = 0;
   out_690979697718630993[51] = 0;
   out_690979697718630993[52] = 0;
   out_690979697718630993[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_3495798777798121399) {
  err_fun(nom_x, delta_x, out_3495798777798121399);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6491638533999781681) {
  inv_err_fun(nom_x, true_x, out_6491638533999781681);
}
void pose_H_mod_fun(double *state, double *out_1193861252693843199) {
  H_mod_fun(state, out_1193861252693843199);
}
void pose_f_fun(double *state, double dt, double *out_2162222754272876291) {
  f_fun(state,  dt, out_2162222754272876291);
}
void pose_F_fun(double *state, double dt, double *out_577996268896955016) {
  F_fun(state,  dt, out_577996268896955016);
}
void pose_h_4(double *state, double *unused, double *out_2480001917824807916) {
  h_4(state, unused, out_2480001917824807916);
}
void pose_H_4(double *state, double *unused, double *out_4654220554058115522) {
  H_4(state, unused, out_4654220554058115522);
}
void pose_h_10(double *state, double *unused, double *out_6842199753574438663) {
  h_10(state, unused, out_6842199753574438663);
}
void pose_H_10(double *state, double *unused, double *out_8460558184635204487) {
  H_10(state, unused, out_8460558184635204487);
}
void pose_h_13(double *state, double *unused, double *out_7103896844424350049) {
  h_13(state, unused, out_7103896844424350049);
}
void pose_H_13(double *state, double *unused, double *out_1441946728725782721) {
  H_13(state, unused, out_1441946728725782721);
}
void pose_h_14(double *state, double *unused, double *out_2985573120985558604) {
  h_14(state, unused, out_2985573120985558604);
}
void pose_H_14(double *state, double *unused, double *out_690979697718630993) {
  H_14(state, unused, out_690979697718630993);
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
