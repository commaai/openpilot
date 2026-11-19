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
void err_fun(double *nom_x, double *delta_x, double *out_4484765240542640643) {
   out_4484765240542640643[0] = delta_x[0] + nom_x[0];
   out_4484765240542640643[1] = delta_x[1] + nom_x[1];
   out_4484765240542640643[2] = delta_x[2] + nom_x[2];
   out_4484765240542640643[3] = delta_x[3] + nom_x[3];
   out_4484765240542640643[4] = delta_x[4] + nom_x[4];
   out_4484765240542640643[5] = delta_x[5] + nom_x[5];
   out_4484765240542640643[6] = delta_x[6] + nom_x[6];
   out_4484765240542640643[7] = delta_x[7] + nom_x[7];
   out_4484765240542640643[8] = delta_x[8] + nom_x[8];
   out_4484765240542640643[9] = delta_x[9] + nom_x[9];
   out_4484765240542640643[10] = delta_x[10] + nom_x[10];
   out_4484765240542640643[11] = delta_x[11] + nom_x[11];
   out_4484765240542640643[12] = delta_x[12] + nom_x[12];
   out_4484765240542640643[13] = delta_x[13] + nom_x[13];
   out_4484765240542640643[14] = delta_x[14] + nom_x[14];
   out_4484765240542640643[15] = delta_x[15] + nom_x[15];
   out_4484765240542640643[16] = delta_x[16] + nom_x[16];
   out_4484765240542640643[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2314530964318079998) {
   out_2314530964318079998[0] = -nom_x[0] + true_x[0];
   out_2314530964318079998[1] = -nom_x[1] + true_x[1];
   out_2314530964318079998[2] = -nom_x[2] + true_x[2];
   out_2314530964318079998[3] = -nom_x[3] + true_x[3];
   out_2314530964318079998[4] = -nom_x[4] + true_x[4];
   out_2314530964318079998[5] = -nom_x[5] + true_x[5];
   out_2314530964318079998[6] = -nom_x[6] + true_x[6];
   out_2314530964318079998[7] = -nom_x[7] + true_x[7];
   out_2314530964318079998[8] = -nom_x[8] + true_x[8];
   out_2314530964318079998[9] = -nom_x[9] + true_x[9];
   out_2314530964318079998[10] = -nom_x[10] + true_x[10];
   out_2314530964318079998[11] = -nom_x[11] + true_x[11];
   out_2314530964318079998[12] = -nom_x[12] + true_x[12];
   out_2314530964318079998[13] = -nom_x[13] + true_x[13];
   out_2314530964318079998[14] = -nom_x[14] + true_x[14];
   out_2314530964318079998[15] = -nom_x[15] + true_x[15];
   out_2314530964318079998[16] = -nom_x[16] + true_x[16];
   out_2314530964318079998[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3727250654126270259) {
   out_3727250654126270259[0] = 1.0;
   out_3727250654126270259[1] = 0.0;
   out_3727250654126270259[2] = 0.0;
   out_3727250654126270259[3] = 0.0;
   out_3727250654126270259[4] = 0.0;
   out_3727250654126270259[5] = 0.0;
   out_3727250654126270259[6] = 0.0;
   out_3727250654126270259[7] = 0.0;
   out_3727250654126270259[8] = 0.0;
   out_3727250654126270259[9] = 0.0;
   out_3727250654126270259[10] = 0.0;
   out_3727250654126270259[11] = 0.0;
   out_3727250654126270259[12] = 0.0;
   out_3727250654126270259[13] = 0.0;
   out_3727250654126270259[14] = 0.0;
   out_3727250654126270259[15] = 0.0;
   out_3727250654126270259[16] = 0.0;
   out_3727250654126270259[17] = 0.0;
   out_3727250654126270259[18] = 0.0;
   out_3727250654126270259[19] = 1.0;
   out_3727250654126270259[20] = 0.0;
   out_3727250654126270259[21] = 0.0;
   out_3727250654126270259[22] = 0.0;
   out_3727250654126270259[23] = 0.0;
   out_3727250654126270259[24] = 0.0;
   out_3727250654126270259[25] = 0.0;
   out_3727250654126270259[26] = 0.0;
   out_3727250654126270259[27] = 0.0;
   out_3727250654126270259[28] = 0.0;
   out_3727250654126270259[29] = 0.0;
   out_3727250654126270259[30] = 0.0;
   out_3727250654126270259[31] = 0.0;
   out_3727250654126270259[32] = 0.0;
   out_3727250654126270259[33] = 0.0;
   out_3727250654126270259[34] = 0.0;
   out_3727250654126270259[35] = 0.0;
   out_3727250654126270259[36] = 0.0;
   out_3727250654126270259[37] = 0.0;
   out_3727250654126270259[38] = 1.0;
   out_3727250654126270259[39] = 0.0;
   out_3727250654126270259[40] = 0.0;
   out_3727250654126270259[41] = 0.0;
   out_3727250654126270259[42] = 0.0;
   out_3727250654126270259[43] = 0.0;
   out_3727250654126270259[44] = 0.0;
   out_3727250654126270259[45] = 0.0;
   out_3727250654126270259[46] = 0.0;
   out_3727250654126270259[47] = 0.0;
   out_3727250654126270259[48] = 0.0;
   out_3727250654126270259[49] = 0.0;
   out_3727250654126270259[50] = 0.0;
   out_3727250654126270259[51] = 0.0;
   out_3727250654126270259[52] = 0.0;
   out_3727250654126270259[53] = 0.0;
   out_3727250654126270259[54] = 0.0;
   out_3727250654126270259[55] = 0.0;
   out_3727250654126270259[56] = 0.0;
   out_3727250654126270259[57] = 1.0;
   out_3727250654126270259[58] = 0.0;
   out_3727250654126270259[59] = 0.0;
   out_3727250654126270259[60] = 0.0;
   out_3727250654126270259[61] = 0.0;
   out_3727250654126270259[62] = 0.0;
   out_3727250654126270259[63] = 0.0;
   out_3727250654126270259[64] = 0.0;
   out_3727250654126270259[65] = 0.0;
   out_3727250654126270259[66] = 0.0;
   out_3727250654126270259[67] = 0.0;
   out_3727250654126270259[68] = 0.0;
   out_3727250654126270259[69] = 0.0;
   out_3727250654126270259[70] = 0.0;
   out_3727250654126270259[71] = 0.0;
   out_3727250654126270259[72] = 0.0;
   out_3727250654126270259[73] = 0.0;
   out_3727250654126270259[74] = 0.0;
   out_3727250654126270259[75] = 0.0;
   out_3727250654126270259[76] = 1.0;
   out_3727250654126270259[77] = 0.0;
   out_3727250654126270259[78] = 0.0;
   out_3727250654126270259[79] = 0.0;
   out_3727250654126270259[80] = 0.0;
   out_3727250654126270259[81] = 0.0;
   out_3727250654126270259[82] = 0.0;
   out_3727250654126270259[83] = 0.0;
   out_3727250654126270259[84] = 0.0;
   out_3727250654126270259[85] = 0.0;
   out_3727250654126270259[86] = 0.0;
   out_3727250654126270259[87] = 0.0;
   out_3727250654126270259[88] = 0.0;
   out_3727250654126270259[89] = 0.0;
   out_3727250654126270259[90] = 0.0;
   out_3727250654126270259[91] = 0.0;
   out_3727250654126270259[92] = 0.0;
   out_3727250654126270259[93] = 0.0;
   out_3727250654126270259[94] = 0.0;
   out_3727250654126270259[95] = 1.0;
   out_3727250654126270259[96] = 0.0;
   out_3727250654126270259[97] = 0.0;
   out_3727250654126270259[98] = 0.0;
   out_3727250654126270259[99] = 0.0;
   out_3727250654126270259[100] = 0.0;
   out_3727250654126270259[101] = 0.0;
   out_3727250654126270259[102] = 0.0;
   out_3727250654126270259[103] = 0.0;
   out_3727250654126270259[104] = 0.0;
   out_3727250654126270259[105] = 0.0;
   out_3727250654126270259[106] = 0.0;
   out_3727250654126270259[107] = 0.0;
   out_3727250654126270259[108] = 0.0;
   out_3727250654126270259[109] = 0.0;
   out_3727250654126270259[110] = 0.0;
   out_3727250654126270259[111] = 0.0;
   out_3727250654126270259[112] = 0.0;
   out_3727250654126270259[113] = 0.0;
   out_3727250654126270259[114] = 1.0;
   out_3727250654126270259[115] = 0.0;
   out_3727250654126270259[116] = 0.0;
   out_3727250654126270259[117] = 0.0;
   out_3727250654126270259[118] = 0.0;
   out_3727250654126270259[119] = 0.0;
   out_3727250654126270259[120] = 0.0;
   out_3727250654126270259[121] = 0.0;
   out_3727250654126270259[122] = 0.0;
   out_3727250654126270259[123] = 0.0;
   out_3727250654126270259[124] = 0.0;
   out_3727250654126270259[125] = 0.0;
   out_3727250654126270259[126] = 0.0;
   out_3727250654126270259[127] = 0.0;
   out_3727250654126270259[128] = 0.0;
   out_3727250654126270259[129] = 0.0;
   out_3727250654126270259[130] = 0.0;
   out_3727250654126270259[131] = 0.0;
   out_3727250654126270259[132] = 0.0;
   out_3727250654126270259[133] = 1.0;
   out_3727250654126270259[134] = 0.0;
   out_3727250654126270259[135] = 0.0;
   out_3727250654126270259[136] = 0.0;
   out_3727250654126270259[137] = 0.0;
   out_3727250654126270259[138] = 0.0;
   out_3727250654126270259[139] = 0.0;
   out_3727250654126270259[140] = 0.0;
   out_3727250654126270259[141] = 0.0;
   out_3727250654126270259[142] = 0.0;
   out_3727250654126270259[143] = 0.0;
   out_3727250654126270259[144] = 0.0;
   out_3727250654126270259[145] = 0.0;
   out_3727250654126270259[146] = 0.0;
   out_3727250654126270259[147] = 0.0;
   out_3727250654126270259[148] = 0.0;
   out_3727250654126270259[149] = 0.0;
   out_3727250654126270259[150] = 0.0;
   out_3727250654126270259[151] = 0.0;
   out_3727250654126270259[152] = 1.0;
   out_3727250654126270259[153] = 0.0;
   out_3727250654126270259[154] = 0.0;
   out_3727250654126270259[155] = 0.0;
   out_3727250654126270259[156] = 0.0;
   out_3727250654126270259[157] = 0.0;
   out_3727250654126270259[158] = 0.0;
   out_3727250654126270259[159] = 0.0;
   out_3727250654126270259[160] = 0.0;
   out_3727250654126270259[161] = 0.0;
   out_3727250654126270259[162] = 0.0;
   out_3727250654126270259[163] = 0.0;
   out_3727250654126270259[164] = 0.0;
   out_3727250654126270259[165] = 0.0;
   out_3727250654126270259[166] = 0.0;
   out_3727250654126270259[167] = 0.0;
   out_3727250654126270259[168] = 0.0;
   out_3727250654126270259[169] = 0.0;
   out_3727250654126270259[170] = 0.0;
   out_3727250654126270259[171] = 1.0;
   out_3727250654126270259[172] = 0.0;
   out_3727250654126270259[173] = 0.0;
   out_3727250654126270259[174] = 0.0;
   out_3727250654126270259[175] = 0.0;
   out_3727250654126270259[176] = 0.0;
   out_3727250654126270259[177] = 0.0;
   out_3727250654126270259[178] = 0.0;
   out_3727250654126270259[179] = 0.0;
   out_3727250654126270259[180] = 0.0;
   out_3727250654126270259[181] = 0.0;
   out_3727250654126270259[182] = 0.0;
   out_3727250654126270259[183] = 0.0;
   out_3727250654126270259[184] = 0.0;
   out_3727250654126270259[185] = 0.0;
   out_3727250654126270259[186] = 0.0;
   out_3727250654126270259[187] = 0.0;
   out_3727250654126270259[188] = 0.0;
   out_3727250654126270259[189] = 0.0;
   out_3727250654126270259[190] = 1.0;
   out_3727250654126270259[191] = 0.0;
   out_3727250654126270259[192] = 0.0;
   out_3727250654126270259[193] = 0.0;
   out_3727250654126270259[194] = 0.0;
   out_3727250654126270259[195] = 0.0;
   out_3727250654126270259[196] = 0.0;
   out_3727250654126270259[197] = 0.0;
   out_3727250654126270259[198] = 0.0;
   out_3727250654126270259[199] = 0.0;
   out_3727250654126270259[200] = 0.0;
   out_3727250654126270259[201] = 0.0;
   out_3727250654126270259[202] = 0.0;
   out_3727250654126270259[203] = 0.0;
   out_3727250654126270259[204] = 0.0;
   out_3727250654126270259[205] = 0.0;
   out_3727250654126270259[206] = 0.0;
   out_3727250654126270259[207] = 0.0;
   out_3727250654126270259[208] = 0.0;
   out_3727250654126270259[209] = 1.0;
   out_3727250654126270259[210] = 0.0;
   out_3727250654126270259[211] = 0.0;
   out_3727250654126270259[212] = 0.0;
   out_3727250654126270259[213] = 0.0;
   out_3727250654126270259[214] = 0.0;
   out_3727250654126270259[215] = 0.0;
   out_3727250654126270259[216] = 0.0;
   out_3727250654126270259[217] = 0.0;
   out_3727250654126270259[218] = 0.0;
   out_3727250654126270259[219] = 0.0;
   out_3727250654126270259[220] = 0.0;
   out_3727250654126270259[221] = 0.0;
   out_3727250654126270259[222] = 0.0;
   out_3727250654126270259[223] = 0.0;
   out_3727250654126270259[224] = 0.0;
   out_3727250654126270259[225] = 0.0;
   out_3727250654126270259[226] = 0.0;
   out_3727250654126270259[227] = 0.0;
   out_3727250654126270259[228] = 1.0;
   out_3727250654126270259[229] = 0.0;
   out_3727250654126270259[230] = 0.0;
   out_3727250654126270259[231] = 0.0;
   out_3727250654126270259[232] = 0.0;
   out_3727250654126270259[233] = 0.0;
   out_3727250654126270259[234] = 0.0;
   out_3727250654126270259[235] = 0.0;
   out_3727250654126270259[236] = 0.0;
   out_3727250654126270259[237] = 0.0;
   out_3727250654126270259[238] = 0.0;
   out_3727250654126270259[239] = 0.0;
   out_3727250654126270259[240] = 0.0;
   out_3727250654126270259[241] = 0.0;
   out_3727250654126270259[242] = 0.0;
   out_3727250654126270259[243] = 0.0;
   out_3727250654126270259[244] = 0.0;
   out_3727250654126270259[245] = 0.0;
   out_3727250654126270259[246] = 0.0;
   out_3727250654126270259[247] = 1.0;
   out_3727250654126270259[248] = 0.0;
   out_3727250654126270259[249] = 0.0;
   out_3727250654126270259[250] = 0.0;
   out_3727250654126270259[251] = 0.0;
   out_3727250654126270259[252] = 0.0;
   out_3727250654126270259[253] = 0.0;
   out_3727250654126270259[254] = 0.0;
   out_3727250654126270259[255] = 0.0;
   out_3727250654126270259[256] = 0.0;
   out_3727250654126270259[257] = 0.0;
   out_3727250654126270259[258] = 0.0;
   out_3727250654126270259[259] = 0.0;
   out_3727250654126270259[260] = 0.0;
   out_3727250654126270259[261] = 0.0;
   out_3727250654126270259[262] = 0.0;
   out_3727250654126270259[263] = 0.0;
   out_3727250654126270259[264] = 0.0;
   out_3727250654126270259[265] = 0.0;
   out_3727250654126270259[266] = 1.0;
   out_3727250654126270259[267] = 0.0;
   out_3727250654126270259[268] = 0.0;
   out_3727250654126270259[269] = 0.0;
   out_3727250654126270259[270] = 0.0;
   out_3727250654126270259[271] = 0.0;
   out_3727250654126270259[272] = 0.0;
   out_3727250654126270259[273] = 0.0;
   out_3727250654126270259[274] = 0.0;
   out_3727250654126270259[275] = 0.0;
   out_3727250654126270259[276] = 0.0;
   out_3727250654126270259[277] = 0.0;
   out_3727250654126270259[278] = 0.0;
   out_3727250654126270259[279] = 0.0;
   out_3727250654126270259[280] = 0.0;
   out_3727250654126270259[281] = 0.0;
   out_3727250654126270259[282] = 0.0;
   out_3727250654126270259[283] = 0.0;
   out_3727250654126270259[284] = 0.0;
   out_3727250654126270259[285] = 1.0;
   out_3727250654126270259[286] = 0.0;
   out_3727250654126270259[287] = 0.0;
   out_3727250654126270259[288] = 0.0;
   out_3727250654126270259[289] = 0.0;
   out_3727250654126270259[290] = 0.0;
   out_3727250654126270259[291] = 0.0;
   out_3727250654126270259[292] = 0.0;
   out_3727250654126270259[293] = 0.0;
   out_3727250654126270259[294] = 0.0;
   out_3727250654126270259[295] = 0.0;
   out_3727250654126270259[296] = 0.0;
   out_3727250654126270259[297] = 0.0;
   out_3727250654126270259[298] = 0.0;
   out_3727250654126270259[299] = 0.0;
   out_3727250654126270259[300] = 0.0;
   out_3727250654126270259[301] = 0.0;
   out_3727250654126270259[302] = 0.0;
   out_3727250654126270259[303] = 0.0;
   out_3727250654126270259[304] = 1.0;
   out_3727250654126270259[305] = 0.0;
   out_3727250654126270259[306] = 0.0;
   out_3727250654126270259[307] = 0.0;
   out_3727250654126270259[308] = 0.0;
   out_3727250654126270259[309] = 0.0;
   out_3727250654126270259[310] = 0.0;
   out_3727250654126270259[311] = 0.0;
   out_3727250654126270259[312] = 0.0;
   out_3727250654126270259[313] = 0.0;
   out_3727250654126270259[314] = 0.0;
   out_3727250654126270259[315] = 0.0;
   out_3727250654126270259[316] = 0.0;
   out_3727250654126270259[317] = 0.0;
   out_3727250654126270259[318] = 0.0;
   out_3727250654126270259[319] = 0.0;
   out_3727250654126270259[320] = 0.0;
   out_3727250654126270259[321] = 0.0;
   out_3727250654126270259[322] = 0.0;
   out_3727250654126270259[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_2032702142868839994) {
   out_2032702142868839994[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_2032702142868839994[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_2032702142868839994[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_2032702142868839994[3] = dt*state[12] + state[3];
   out_2032702142868839994[4] = dt*state[13] + state[4];
   out_2032702142868839994[5] = dt*state[14] + state[5];
   out_2032702142868839994[6] = state[6];
   out_2032702142868839994[7] = state[7];
   out_2032702142868839994[8] = state[8];
   out_2032702142868839994[9] = state[9];
   out_2032702142868839994[10] = state[10];
   out_2032702142868839994[11] = state[11];
   out_2032702142868839994[12] = state[12];
   out_2032702142868839994[13] = state[13];
   out_2032702142868839994[14] = state[14];
   out_2032702142868839994[15] = state[15];
   out_2032702142868839994[16] = state[16];
   out_2032702142868839994[17] = state[17];
}
void F_fun(double *state, double dt, double *out_7670713034263855145) {
   out_7670713034263855145[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7670713034263855145[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7670713034263855145[2] = 0;
   out_7670713034263855145[3] = 0;
   out_7670713034263855145[4] = 0;
   out_7670713034263855145[5] = 0;
   out_7670713034263855145[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7670713034263855145[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7670713034263855145[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_7670713034263855145[9] = 0;
   out_7670713034263855145[10] = 0;
   out_7670713034263855145[11] = 0;
   out_7670713034263855145[12] = 0;
   out_7670713034263855145[13] = 0;
   out_7670713034263855145[14] = 0;
   out_7670713034263855145[15] = 0;
   out_7670713034263855145[16] = 0;
   out_7670713034263855145[17] = 0;
   out_7670713034263855145[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7670713034263855145[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7670713034263855145[20] = 0;
   out_7670713034263855145[21] = 0;
   out_7670713034263855145[22] = 0;
   out_7670713034263855145[23] = 0;
   out_7670713034263855145[24] = 0;
   out_7670713034263855145[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7670713034263855145[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_7670713034263855145[27] = 0;
   out_7670713034263855145[28] = 0;
   out_7670713034263855145[29] = 0;
   out_7670713034263855145[30] = 0;
   out_7670713034263855145[31] = 0;
   out_7670713034263855145[32] = 0;
   out_7670713034263855145[33] = 0;
   out_7670713034263855145[34] = 0;
   out_7670713034263855145[35] = 0;
   out_7670713034263855145[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7670713034263855145[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7670713034263855145[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7670713034263855145[39] = 0;
   out_7670713034263855145[40] = 0;
   out_7670713034263855145[41] = 0;
   out_7670713034263855145[42] = 0;
   out_7670713034263855145[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7670713034263855145[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_7670713034263855145[45] = 0;
   out_7670713034263855145[46] = 0;
   out_7670713034263855145[47] = 0;
   out_7670713034263855145[48] = 0;
   out_7670713034263855145[49] = 0;
   out_7670713034263855145[50] = 0;
   out_7670713034263855145[51] = 0;
   out_7670713034263855145[52] = 0;
   out_7670713034263855145[53] = 0;
   out_7670713034263855145[54] = 0;
   out_7670713034263855145[55] = 0;
   out_7670713034263855145[56] = 0;
   out_7670713034263855145[57] = 1;
   out_7670713034263855145[58] = 0;
   out_7670713034263855145[59] = 0;
   out_7670713034263855145[60] = 0;
   out_7670713034263855145[61] = 0;
   out_7670713034263855145[62] = 0;
   out_7670713034263855145[63] = 0;
   out_7670713034263855145[64] = 0;
   out_7670713034263855145[65] = 0;
   out_7670713034263855145[66] = dt;
   out_7670713034263855145[67] = 0;
   out_7670713034263855145[68] = 0;
   out_7670713034263855145[69] = 0;
   out_7670713034263855145[70] = 0;
   out_7670713034263855145[71] = 0;
   out_7670713034263855145[72] = 0;
   out_7670713034263855145[73] = 0;
   out_7670713034263855145[74] = 0;
   out_7670713034263855145[75] = 0;
   out_7670713034263855145[76] = 1;
   out_7670713034263855145[77] = 0;
   out_7670713034263855145[78] = 0;
   out_7670713034263855145[79] = 0;
   out_7670713034263855145[80] = 0;
   out_7670713034263855145[81] = 0;
   out_7670713034263855145[82] = 0;
   out_7670713034263855145[83] = 0;
   out_7670713034263855145[84] = 0;
   out_7670713034263855145[85] = dt;
   out_7670713034263855145[86] = 0;
   out_7670713034263855145[87] = 0;
   out_7670713034263855145[88] = 0;
   out_7670713034263855145[89] = 0;
   out_7670713034263855145[90] = 0;
   out_7670713034263855145[91] = 0;
   out_7670713034263855145[92] = 0;
   out_7670713034263855145[93] = 0;
   out_7670713034263855145[94] = 0;
   out_7670713034263855145[95] = 1;
   out_7670713034263855145[96] = 0;
   out_7670713034263855145[97] = 0;
   out_7670713034263855145[98] = 0;
   out_7670713034263855145[99] = 0;
   out_7670713034263855145[100] = 0;
   out_7670713034263855145[101] = 0;
   out_7670713034263855145[102] = 0;
   out_7670713034263855145[103] = 0;
   out_7670713034263855145[104] = dt;
   out_7670713034263855145[105] = 0;
   out_7670713034263855145[106] = 0;
   out_7670713034263855145[107] = 0;
   out_7670713034263855145[108] = 0;
   out_7670713034263855145[109] = 0;
   out_7670713034263855145[110] = 0;
   out_7670713034263855145[111] = 0;
   out_7670713034263855145[112] = 0;
   out_7670713034263855145[113] = 0;
   out_7670713034263855145[114] = 1;
   out_7670713034263855145[115] = 0;
   out_7670713034263855145[116] = 0;
   out_7670713034263855145[117] = 0;
   out_7670713034263855145[118] = 0;
   out_7670713034263855145[119] = 0;
   out_7670713034263855145[120] = 0;
   out_7670713034263855145[121] = 0;
   out_7670713034263855145[122] = 0;
   out_7670713034263855145[123] = 0;
   out_7670713034263855145[124] = 0;
   out_7670713034263855145[125] = 0;
   out_7670713034263855145[126] = 0;
   out_7670713034263855145[127] = 0;
   out_7670713034263855145[128] = 0;
   out_7670713034263855145[129] = 0;
   out_7670713034263855145[130] = 0;
   out_7670713034263855145[131] = 0;
   out_7670713034263855145[132] = 0;
   out_7670713034263855145[133] = 1;
   out_7670713034263855145[134] = 0;
   out_7670713034263855145[135] = 0;
   out_7670713034263855145[136] = 0;
   out_7670713034263855145[137] = 0;
   out_7670713034263855145[138] = 0;
   out_7670713034263855145[139] = 0;
   out_7670713034263855145[140] = 0;
   out_7670713034263855145[141] = 0;
   out_7670713034263855145[142] = 0;
   out_7670713034263855145[143] = 0;
   out_7670713034263855145[144] = 0;
   out_7670713034263855145[145] = 0;
   out_7670713034263855145[146] = 0;
   out_7670713034263855145[147] = 0;
   out_7670713034263855145[148] = 0;
   out_7670713034263855145[149] = 0;
   out_7670713034263855145[150] = 0;
   out_7670713034263855145[151] = 0;
   out_7670713034263855145[152] = 1;
   out_7670713034263855145[153] = 0;
   out_7670713034263855145[154] = 0;
   out_7670713034263855145[155] = 0;
   out_7670713034263855145[156] = 0;
   out_7670713034263855145[157] = 0;
   out_7670713034263855145[158] = 0;
   out_7670713034263855145[159] = 0;
   out_7670713034263855145[160] = 0;
   out_7670713034263855145[161] = 0;
   out_7670713034263855145[162] = 0;
   out_7670713034263855145[163] = 0;
   out_7670713034263855145[164] = 0;
   out_7670713034263855145[165] = 0;
   out_7670713034263855145[166] = 0;
   out_7670713034263855145[167] = 0;
   out_7670713034263855145[168] = 0;
   out_7670713034263855145[169] = 0;
   out_7670713034263855145[170] = 0;
   out_7670713034263855145[171] = 1;
   out_7670713034263855145[172] = 0;
   out_7670713034263855145[173] = 0;
   out_7670713034263855145[174] = 0;
   out_7670713034263855145[175] = 0;
   out_7670713034263855145[176] = 0;
   out_7670713034263855145[177] = 0;
   out_7670713034263855145[178] = 0;
   out_7670713034263855145[179] = 0;
   out_7670713034263855145[180] = 0;
   out_7670713034263855145[181] = 0;
   out_7670713034263855145[182] = 0;
   out_7670713034263855145[183] = 0;
   out_7670713034263855145[184] = 0;
   out_7670713034263855145[185] = 0;
   out_7670713034263855145[186] = 0;
   out_7670713034263855145[187] = 0;
   out_7670713034263855145[188] = 0;
   out_7670713034263855145[189] = 0;
   out_7670713034263855145[190] = 1;
   out_7670713034263855145[191] = 0;
   out_7670713034263855145[192] = 0;
   out_7670713034263855145[193] = 0;
   out_7670713034263855145[194] = 0;
   out_7670713034263855145[195] = 0;
   out_7670713034263855145[196] = 0;
   out_7670713034263855145[197] = 0;
   out_7670713034263855145[198] = 0;
   out_7670713034263855145[199] = 0;
   out_7670713034263855145[200] = 0;
   out_7670713034263855145[201] = 0;
   out_7670713034263855145[202] = 0;
   out_7670713034263855145[203] = 0;
   out_7670713034263855145[204] = 0;
   out_7670713034263855145[205] = 0;
   out_7670713034263855145[206] = 0;
   out_7670713034263855145[207] = 0;
   out_7670713034263855145[208] = 0;
   out_7670713034263855145[209] = 1;
   out_7670713034263855145[210] = 0;
   out_7670713034263855145[211] = 0;
   out_7670713034263855145[212] = 0;
   out_7670713034263855145[213] = 0;
   out_7670713034263855145[214] = 0;
   out_7670713034263855145[215] = 0;
   out_7670713034263855145[216] = 0;
   out_7670713034263855145[217] = 0;
   out_7670713034263855145[218] = 0;
   out_7670713034263855145[219] = 0;
   out_7670713034263855145[220] = 0;
   out_7670713034263855145[221] = 0;
   out_7670713034263855145[222] = 0;
   out_7670713034263855145[223] = 0;
   out_7670713034263855145[224] = 0;
   out_7670713034263855145[225] = 0;
   out_7670713034263855145[226] = 0;
   out_7670713034263855145[227] = 0;
   out_7670713034263855145[228] = 1;
   out_7670713034263855145[229] = 0;
   out_7670713034263855145[230] = 0;
   out_7670713034263855145[231] = 0;
   out_7670713034263855145[232] = 0;
   out_7670713034263855145[233] = 0;
   out_7670713034263855145[234] = 0;
   out_7670713034263855145[235] = 0;
   out_7670713034263855145[236] = 0;
   out_7670713034263855145[237] = 0;
   out_7670713034263855145[238] = 0;
   out_7670713034263855145[239] = 0;
   out_7670713034263855145[240] = 0;
   out_7670713034263855145[241] = 0;
   out_7670713034263855145[242] = 0;
   out_7670713034263855145[243] = 0;
   out_7670713034263855145[244] = 0;
   out_7670713034263855145[245] = 0;
   out_7670713034263855145[246] = 0;
   out_7670713034263855145[247] = 1;
   out_7670713034263855145[248] = 0;
   out_7670713034263855145[249] = 0;
   out_7670713034263855145[250] = 0;
   out_7670713034263855145[251] = 0;
   out_7670713034263855145[252] = 0;
   out_7670713034263855145[253] = 0;
   out_7670713034263855145[254] = 0;
   out_7670713034263855145[255] = 0;
   out_7670713034263855145[256] = 0;
   out_7670713034263855145[257] = 0;
   out_7670713034263855145[258] = 0;
   out_7670713034263855145[259] = 0;
   out_7670713034263855145[260] = 0;
   out_7670713034263855145[261] = 0;
   out_7670713034263855145[262] = 0;
   out_7670713034263855145[263] = 0;
   out_7670713034263855145[264] = 0;
   out_7670713034263855145[265] = 0;
   out_7670713034263855145[266] = 1;
   out_7670713034263855145[267] = 0;
   out_7670713034263855145[268] = 0;
   out_7670713034263855145[269] = 0;
   out_7670713034263855145[270] = 0;
   out_7670713034263855145[271] = 0;
   out_7670713034263855145[272] = 0;
   out_7670713034263855145[273] = 0;
   out_7670713034263855145[274] = 0;
   out_7670713034263855145[275] = 0;
   out_7670713034263855145[276] = 0;
   out_7670713034263855145[277] = 0;
   out_7670713034263855145[278] = 0;
   out_7670713034263855145[279] = 0;
   out_7670713034263855145[280] = 0;
   out_7670713034263855145[281] = 0;
   out_7670713034263855145[282] = 0;
   out_7670713034263855145[283] = 0;
   out_7670713034263855145[284] = 0;
   out_7670713034263855145[285] = 1;
   out_7670713034263855145[286] = 0;
   out_7670713034263855145[287] = 0;
   out_7670713034263855145[288] = 0;
   out_7670713034263855145[289] = 0;
   out_7670713034263855145[290] = 0;
   out_7670713034263855145[291] = 0;
   out_7670713034263855145[292] = 0;
   out_7670713034263855145[293] = 0;
   out_7670713034263855145[294] = 0;
   out_7670713034263855145[295] = 0;
   out_7670713034263855145[296] = 0;
   out_7670713034263855145[297] = 0;
   out_7670713034263855145[298] = 0;
   out_7670713034263855145[299] = 0;
   out_7670713034263855145[300] = 0;
   out_7670713034263855145[301] = 0;
   out_7670713034263855145[302] = 0;
   out_7670713034263855145[303] = 0;
   out_7670713034263855145[304] = 1;
   out_7670713034263855145[305] = 0;
   out_7670713034263855145[306] = 0;
   out_7670713034263855145[307] = 0;
   out_7670713034263855145[308] = 0;
   out_7670713034263855145[309] = 0;
   out_7670713034263855145[310] = 0;
   out_7670713034263855145[311] = 0;
   out_7670713034263855145[312] = 0;
   out_7670713034263855145[313] = 0;
   out_7670713034263855145[314] = 0;
   out_7670713034263855145[315] = 0;
   out_7670713034263855145[316] = 0;
   out_7670713034263855145[317] = 0;
   out_7670713034263855145[318] = 0;
   out_7670713034263855145[319] = 0;
   out_7670713034263855145[320] = 0;
   out_7670713034263855145[321] = 0;
   out_7670713034263855145[322] = 0;
   out_7670713034263855145[323] = 1;
}
void h_4(double *state, double *unused, double *out_702391539073330728) {
   out_702391539073330728[0] = state[6] + state[9];
   out_702391539073330728[1] = state[7] + state[10];
   out_702391539073330728[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_5217111011931551083) {
   out_5217111011931551083[0] = 0;
   out_5217111011931551083[1] = 0;
   out_5217111011931551083[2] = 0;
   out_5217111011931551083[3] = 0;
   out_5217111011931551083[4] = 0;
   out_5217111011931551083[5] = 0;
   out_5217111011931551083[6] = 1;
   out_5217111011931551083[7] = 0;
   out_5217111011931551083[8] = 0;
   out_5217111011931551083[9] = 1;
   out_5217111011931551083[10] = 0;
   out_5217111011931551083[11] = 0;
   out_5217111011931551083[12] = 0;
   out_5217111011931551083[13] = 0;
   out_5217111011931551083[14] = 0;
   out_5217111011931551083[15] = 0;
   out_5217111011931551083[16] = 0;
   out_5217111011931551083[17] = 0;
   out_5217111011931551083[18] = 0;
   out_5217111011931551083[19] = 0;
   out_5217111011931551083[20] = 0;
   out_5217111011931551083[21] = 0;
   out_5217111011931551083[22] = 0;
   out_5217111011931551083[23] = 0;
   out_5217111011931551083[24] = 0;
   out_5217111011931551083[25] = 1;
   out_5217111011931551083[26] = 0;
   out_5217111011931551083[27] = 0;
   out_5217111011931551083[28] = 1;
   out_5217111011931551083[29] = 0;
   out_5217111011931551083[30] = 0;
   out_5217111011931551083[31] = 0;
   out_5217111011931551083[32] = 0;
   out_5217111011931551083[33] = 0;
   out_5217111011931551083[34] = 0;
   out_5217111011931551083[35] = 0;
   out_5217111011931551083[36] = 0;
   out_5217111011931551083[37] = 0;
   out_5217111011931551083[38] = 0;
   out_5217111011931551083[39] = 0;
   out_5217111011931551083[40] = 0;
   out_5217111011931551083[41] = 0;
   out_5217111011931551083[42] = 0;
   out_5217111011931551083[43] = 0;
   out_5217111011931551083[44] = 1;
   out_5217111011931551083[45] = 0;
   out_5217111011931551083[46] = 0;
   out_5217111011931551083[47] = 1;
   out_5217111011931551083[48] = 0;
   out_5217111011931551083[49] = 0;
   out_5217111011931551083[50] = 0;
   out_5217111011931551083[51] = 0;
   out_5217111011931551083[52] = 0;
   out_5217111011931551083[53] = 0;
}
void h_10(double *state, double *unused, double *out_4841844565248541973) {
   out_4841844565248541973[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_4841844565248541973[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_4841844565248541973[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2598878753773211359) {
   out_2598878753773211359[0] = 0;
   out_2598878753773211359[1] = 9.8100000000000005*cos(state[1]);
   out_2598878753773211359[2] = 0;
   out_2598878753773211359[3] = 0;
   out_2598878753773211359[4] = -state[8];
   out_2598878753773211359[5] = state[7];
   out_2598878753773211359[6] = 0;
   out_2598878753773211359[7] = state[5];
   out_2598878753773211359[8] = -state[4];
   out_2598878753773211359[9] = 0;
   out_2598878753773211359[10] = 0;
   out_2598878753773211359[11] = 0;
   out_2598878753773211359[12] = 1;
   out_2598878753773211359[13] = 0;
   out_2598878753773211359[14] = 0;
   out_2598878753773211359[15] = 1;
   out_2598878753773211359[16] = 0;
   out_2598878753773211359[17] = 0;
   out_2598878753773211359[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2598878753773211359[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2598878753773211359[20] = 0;
   out_2598878753773211359[21] = state[8];
   out_2598878753773211359[22] = 0;
   out_2598878753773211359[23] = -state[6];
   out_2598878753773211359[24] = -state[5];
   out_2598878753773211359[25] = 0;
   out_2598878753773211359[26] = state[3];
   out_2598878753773211359[27] = 0;
   out_2598878753773211359[28] = 0;
   out_2598878753773211359[29] = 0;
   out_2598878753773211359[30] = 0;
   out_2598878753773211359[31] = 1;
   out_2598878753773211359[32] = 0;
   out_2598878753773211359[33] = 0;
   out_2598878753773211359[34] = 1;
   out_2598878753773211359[35] = 0;
   out_2598878753773211359[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2598878753773211359[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2598878753773211359[38] = 0;
   out_2598878753773211359[39] = -state[7];
   out_2598878753773211359[40] = state[6];
   out_2598878753773211359[41] = 0;
   out_2598878753773211359[42] = state[4];
   out_2598878753773211359[43] = -state[3];
   out_2598878753773211359[44] = 0;
   out_2598878753773211359[45] = 0;
   out_2598878753773211359[46] = 0;
   out_2598878753773211359[47] = 0;
   out_2598878753773211359[48] = 0;
   out_2598878753773211359[49] = 0;
   out_2598878753773211359[50] = 1;
   out_2598878753773211359[51] = 0;
   out_2598878753773211359[52] = 0;
   out_2598878753773211359[53] = 1;
}
void h_13(double *state, double *unused, double *out_3389735097065185273) {
   out_3389735097065185273[0] = state[3];
   out_3389735097065185273[1] = state[4];
   out_3389735097065185273[2] = state[5];
}
void H_13(double *state, double *unused, double *out_9050866475234075107) {
   out_9050866475234075107[0] = 0;
   out_9050866475234075107[1] = 0;
   out_9050866475234075107[2] = 0;
   out_9050866475234075107[3] = 1;
   out_9050866475234075107[4] = 0;
   out_9050866475234075107[5] = 0;
   out_9050866475234075107[6] = 0;
   out_9050866475234075107[7] = 0;
   out_9050866475234075107[8] = 0;
   out_9050866475234075107[9] = 0;
   out_9050866475234075107[10] = 0;
   out_9050866475234075107[11] = 0;
   out_9050866475234075107[12] = 0;
   out_9050866475234075107[13] = 0;
   out_9050866475234075107[14] = 0;
   out_9050866475234075107[15] = 0;
   out_9050866475234075107[16] = 0;
   out_9050866475234075107[17] = 0;
   out_9050866475234075107[18] = 0;
   out_9050866475234075107[19] = 0;
   out_9050866475234075107[20] = 0;
   out_9050866475234075107[21] = 0;
   out_9050866475234075107[22] = 1;
   out_9050866475234075107[23] = 0;
   out_9050866475234075107[24] = 0;
   out_9050866475234075107[25] = 0;
   out_9050866475234075107[26] = 0;
   out_9050866475234075107[27] = 0;
   out_9050866475234075107[28] = 0;
   out_9050866475234075107[29] = 0;
   out_9050866475234075107[30] = 0;
   out_9050866475234075107[31] = 0;
   out_9050866475234075107[32] = 0;
   out_9050866475234075107[33] = 0;
   out_9050866475234075107[34] = 0;
   out_9050866475234075107[35] = 0;
   out_9050866475234075107[36] = 0;
   out_9050866475234075107[37] = 0;
   out_9050866475234075107[38] = 0;
   out_9050866475234075107[39] = 0;
   out_9050866475234075107[40] = 0;
   out_9050866475234075107[41] = 1;
   out_9050866475234075107[42] = 0;
   out_9050866475234075107[43] = 0;
   out_9050866475234075107[44] = 0;
   out_9050866475234075107[45] = 0;
   out_9050866475234075107[46] = 0;
   out_9050866475234075107[47] = 0;
   out_9050866475234075107[48] = 0;
   out_9050866475234075107[49] = 0;
   out_9050866475234075107[50] = 0;
   out_9050866475234075107[51] = 0;
   out_9050866475234075107[52] = 0;
   out_9050866475234075107[53] = 0;
}
void h_14(double *state, double *unused, double *out_293591342554501993) {
   out_293591342554501993[0] = state[6];
   out_293591342554501993[1] = state[7];
   out_293591342554501993[2] = state[8];
}
void H_14(double *state, double *unused, double *out_8299899444226923379) {
   out_8299899444226923379[0] = 0;
   out_8299899444226923379[1] = 0;
   out_8299899444226923379[2] = 0;
   out_8299899444226923379[3] = 0;
   out_8299899444226923379[4] = 0;
   out_8299899444226923379[5] = 0;
   out_8299899444226923379[6] = 1;
   out_8299899444226923379[7] = 0;
   out_8299899444226923379[8] = 0;
   out_8299899444226923379[9] = 0;
   out_8299899444226923379[10] = 0;
   out_8299899444226923379[11] = 0;
   out_8299899444226923379[12] = 0;
   out_8299899444226923379[13] = 0;
   out_8299899444226923379[14] = 0;
   out_8299899444226923379[15] = 0;
   out_8299899444226923379[16] = 0;
   out_8299899444226923379[17] = 0;
   out_8299899444226923379[18] = 0;
   out_8299899444226923379[19] = 0;
   out_8299899444226923379[20] = 0;
   out_8299899444226923379[21] = 0;
   out_8299899444226923379[22] = 0;
   out_8299899444226923379[23] = 0;
   out_8299899444226923379[24] = 0;
   out_8299899444226923379[25] = 1;
   out_8299899444226923379[26] = 0;
   out_8299899444226923379[27] = 0;
   out_8299899444226923379[28] = 0;
   out_8299899444226923379[29] = 0;
   out_8299899444226923379[30] = 0;
   out_8299899444226923379[31] = 0;
   out_8299899444226923379[32] = 0;
   out_8299899444226923379[33] = 0;
   out_8299899444226923379[34] = 0;
   out_8299899444226923379[35] = 0;
   out_8299899444226923379[36] = 0;
   out_8299899444226923379[37] = 0;
   out_8299899444226923379[38] = 0;
   out_8299899444226923379[39] = 0;
   out_8299899444226923379[40] = 0;
   out_8299899444226923379[41] = 0;
   out_8299899444226923379[42] = 0;
   out_8299899444226923379[43] = 0;
   out_8299899444226923379[44] = 1;
   out_8299899444226923379[45] = 0;
   out_8299899444226923379[46] = 0;
   out_8299899444226923379[47] = 0;
   out_8299899444226923379[48] = 0;
   out_8299899444226923379[49] = 0;
   out_8299899444226923379[50] = 0;
   out_8299899444226923379[51] = 0;
   out_8299899444226923379[52] = 0;
   out_8299899444226923379[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_4484765240542640643) {
  err_fun(nom_x, delta_x, out_4484765240542640643);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2314530964318079998) {
  inv_err_fun(nom_x, true_x, out_2314530964318079998);
}
void pose_H_mod_fun(double *state, double *out_3727250654126270259) {
  H_mod_fun(state, out_3727250654126270259);
}
void pose_f_fun(double *state, double dt, double *out_2032702142868839994) {
  f_fun(state,  dt, out_2032702142868839994);
}
void pose_F_fun(double *state, double dt, double *out_7670713034263855145) {
  F_fun(state,  dt, out_7670713034263855145);
}
void pose_h_4(double *state, double *unused, double *out_702391539073330728) {
  h_4(state, unused, out_702391539073330728);
}
void pose_H_4(double *state, double *unused, double *out_5217111011931551083) {
  H_4(state, unused, out_5217111011931551083);
}
void pose_h_10(double *state, double *unused, double *out_4841844565248541973) {
  h_10(state, unused, out_4841844565248541973);
}
void pose_H_10(double *state, double *unused, double *out_2598878753773211359) {
  H_10(state, unused, out_2598878753773211359);
}
void pose_h_13(double *state, double *unused, double *out_3389735097065185273) {
  h_13(state, unused, out_3389735097065185273);
}
void pose_H_13(double *state, double *unused, double *out_9050866475234075107) {
  H_13(state, unused, out_9050866475234075107);
}
void pose_h_14(double *state, double *unused, double *out_293591342554501993) {
  h_14(state, unused, out_293591342554501993);
}
void pose_H_14(double *state, double *unused, double *out_8299899444226923379) {
  H_14(state, unused, out_8299899444226923379);
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
