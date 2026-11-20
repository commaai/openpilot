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
void err_fun(double *nom_x, double *delta_x, double *out_5073279989451287369) {
   out_5073279989451287369[0] = delta_x[0] + nom_x[0];
   out_5073279989451287369[1] = delta_x[1] + nom_x[1];
   out_5073279989451287369[2] = delta_x[2] + nom_x[2];
   out_5073279989451287369[3] = delta_x[3] + nom_x[3];
   out_5073279989451287369[4] = delta_x[4] + nom_x[4];
   out_5073279989451287369[5] = delta_x[5] + nom_x[5];
   out_5073279989451287369[6] = delta_x[6] + nom_x[6];
   out_5073279989451287369[7] = delta_x[7] + nom_x[7];
   out_5073279989451287369[8] = delta_x[8] + nom_x[8];
   out_5073279989451287369[9] = delta_x[9] + nom_x[9];
   out_5073279989451287369[10] = delta_x[10] + nom_x[10];
   out_5073279989451287369[11] = delta_x[11] + nom_x[11];
   out_5073279989451287369[12] = delta_x[12] + nom_x[12];
   out_5073279989451287369[13] = delta_x[13] + nom_x[13];
   out_5073279989451287369[14] = delta_x[14] + nom_x[14];
   out_5073279989451287369[15] = delta_x[15] + nom_x[15];
   out_5073279989451287369[16] = delta_x[16] + nom_x[16];
   out_5073279989451287369[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5177693965196878277) {
   out_5177693965196878277[0] = -nom_x[0] + true_x[0];
   out_5177693965196878277[1] = -nom_x[1] + true_x[1];
   out_5177693965196878277[2] = -nom_x[2] + true_x[2];
   out_5177693965196878277[3] = -nom_x[3] + true_x[3];
   out_5177693965196878277[4] = -nom_x[4] + true_x[4];
   out_5177693965196878277[5] = -nom_x[5] + true_x[5];
   out_5177693965196878277[6] = -nom_x[6] + true_x[6];
   out_5177693965196878277[7] = -nom_x[7] + true_x[7];
   out_5177693965196878277[8] = -nom_x[8] + true_x[8];
   out_5177693965196878277[9] = -nom_x[9] + true_x[9];
   out_5177693965196878277[10] = -nom_x[10] + true_x[10];
   out_5177693965196878277[11] = -nom_x[11] + true_x[11];
   out_5177693965196878277[12] = -nom_x[12] + true_x[12];
   out_5177693965196878277[13] = -nom_x[13] + true_x[13];
   out_5177693965196878277[14] = -nom_x[14] + true_x[14];
   out_5177693965196878277[15] = -nom_x[15] + true_x[15];
   out_5177693965196878277[16] = -nom_x[16] + true_x[16];
   out_5177693965196878277[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_2613280836668770145) {
   out_2613280836668770145[0] = 1.0;
   out_2613280836668770145[1] = 0.0;
   out_2613280836668770145[2] = 0.0;
   out_2613280836668770145[3] = 0.0;
   out_2613280836668770145[4] = 0.0;
   out_2613280836668770145[5] = 0.0;
   out_2613280836668770145[6] = 0.0;
   out_2613280836668770145[7] = 0.0;
   out_2613280836668770145[8] = 0.0;
   out_2613280836668770145[9] = 0.0;
   out_2613280836668770145[10] = 0.0;
   out_2613280836668770145[11] = 0.0;
   out_2613280836668770145[12] = 0.0;
   out_2613280836668770145[13] = 0.0;
   out_2613280836668770145[14] = 0.0;
   out_2613280836668770145[15] = 0.0;
   out_2613280836668770145[16] = 0.0;
   out_2613280836668770145[17] = 0.0;
   out_2613280836668770145[18] = 0.0;
   out_2613280836668770145[19] = 1.0;
   out_2613280836668770145[20] = 0.0;
   out_2613280836668770145[21] = 0.0;
   out_2613280836668770145[22] = 0.0;
   out_2613280836668770145[23] = 0.0;
   out_2613280836668770145[24] = 0.0;
   out_2613280836668770145[25] = 0.0;
   out_2613280836668770145[26] = 0.0;
   out_2613280836668770145[27] = 0.0;
   out_2613280836668770145[28] = 0.0;
   out_2613280836668770145[29] = 0.0;
   out_2613280836668770145[30] = 0.0;
   out_2613280836668770145[31] = 0.0;
   out_2613280836668770145[32] = 0.0;
   out_2613280836668770145[33] = 0.0;
   out_2613280836668770145[34] = 0.0;
   out_2613280836668770145[35] = 0.0;
   out_2613280836668770145[36] = 0.0;
   out_2613280836668770145[37] = 0.0;
   out_2613280836668770145[38] = 1.0;
   out_2613280836668770145[39] = 0.0;
   out_2613280836668770145[40] = 0.0;
   out_2613280836668770145[41] = 0.0;
   out_2613280836668770145[42] = 0.0;
   out_2613280836668770145[43] = 0.0;
   out_2613280836668770145[44] = 0.0;
   out_2613280836668770145[45] = 0.0;
   out_2613280836668770145[46] = 0.0;
   out_2613280836668770145[47] = 0.0;
   out_2613280836668770145[48] = 0.0;
   out_2613280836668770145[49] = 0.0;
   out_2613280836668770145[50] = 0.0;
   out_2613280836668770145[51] = 0.0;
   out_2613280836668770145[52] = 0.0;
   out_2613280836668770145[53] = 0.0;
   out_2613280836668770145[54] = 0.0;
   out_2613280836668770145[55] = 0.0;
   out_2613280836668770145[56] = 0.0;
   out_2613280836668770145[57] = 1.0;
   out_2613280836668770145[58] = 0.0;
   out_2613280836668770145[59] = 0.0;
   out_2613280836668770145[60] = 0.0;
   out_2613280836668770145[61] = 0.0;
   out_2613280836668770145[62] = 0.0;
   out_2613280836668770145[63] = 0.0;
   out_2613280836668770145[64] = 0.0;
   out_2613280836668770145[65] = 0.0;
   out_2613280836668770145[66] = 0.0;
   out_2613280836668770145[67] = 0.0;
   out_2613280836668770145[68] = 0.0;
   out_2613280836668770145[69] = 0.0;
   out_2613280836668770145[70] = 0.0;
   out_2613280836668770145[71] = 0.0;
   out_2613280836668770145[72] = 0.0;
   out_2613280836668770145[73] = 0.0;
   out_2613280836668770145[74] = 0.0;
   out_2613280836668770145[75] = 0.0;
   out_2613280836668770145[76] = 1.0;
   out_2613280836668770145[77] = 0.0;
   out_2613280836668770145[78] = 0.0;
   out_2613280836668770145[79] = 0.0;
   out_2613280836668770145[80] = 0.0;
   out_2613280836668770145[81] = 0.0;
   out_2613280836668770145[82] = 0.0;
   out_2613280836668770145[83] = 0.0;
   out_2613280836668770145[84] = 0.0;
   out_2613280836668770145[85] = 0.0;
   out_2613280836668770145[86] = 0.0;
   out_2613280836668770145[87] = 0.0;
   out_2613280836668770145[88] = 0.0;
   out_2613280836668770145[89] = 0.0;
   out_2613280836668770145[90] = 0.0;
   out_2613280836668770145[91] = 0.0;
   out_2613280836668770145[92] = 0.0;
   out_2613280836668770145[93] = 0.0;
   out_2613280836668770145[94] = 0.0;
   out_2613280836668770145[95] = 1.0;
   out_2613280836668770145[96] = 0.0;
   out_2613280836668770145[97] = 0.0;
   out_2613280836668770145[98] = 0.0;
   out_2613280836668770145[99] = 0.0;
   out_2613280836668770145[100] = 0.0;
   out_2613280836668770145[101] = 0.0;
   out_2613280836668770145[102] = 0.0;
   out_2613280836668770145[103] = 0.0;
   out_2613280836668770145[104] = 0.0;
   out_2613280836668770145[105] = 0.0;
   out_2613280836668770145[106] = 0.0;
   out_2613280836668770145[107] = 0.0;
   out_2613280836668770145[108] = 0.0;
   out_2613280836668770145[109] = 0.0;
   out_2613280836668770145[110] = 0.0;
   out_2613280836668770145[111] = 0.0;
   out_2613280836668770145[112] = 0.0;
   out_2613280836668770145[113] = 0.0;
   out_2613280836668770145[114] = 1.0;
   out_2613280836668770145[115] = 0.0;
   out_2613280836668770145[116] = 0.0;
   out_2613280836668770145[117] = 0.0;
   out_2613280836668770145[118] = 0.0;
   out_2613280836668770145[119] = 0.0;
   out_2613280836668770145[120] = 0.0;
   out_2613280836668770145[121] = 0.0;
   out_2613280836668770145[122] = 0.0;
   out_2613280836668770145[123] = 0.0;
   out_2613280836668770145[124] = 0.0;
   out_2613280836668770145[125] = 0.0;
   out_2613280836668770145[126] = 0.0;
   out_2613280836668770145[127] = 0.0;
   out_2613280836668770145[128] = 0.0;
   out_2613280836668770145[129] = 0.0;
   out_2613280836668770145[130] = 0.0;
   out_2613280836668770145[131] = 0.0;
   out_2613280836668770145[132] = 0.0;
   out_2613280836668770145[133] = 1.0;
   out_2613280836668770145[134] = 0.0;
   out_2613280836668770145[135] = 0.0;
   out_2613280836668770145[136] = 0.0;
   out_2613280836668770145[137] = 0.0;
   out_2613280836668770145[138] = 0.0;
   out_2613280836668770145[139] = 0.0;
   out_2613280836668770145[140] = 0.0;
   out_2613280836668770145[141] = 0.0;
   out_2613280836668770145[142] = 0.0;
   out_2613280836668770145[143] = 0.0;
   out_2613280836668770145[144] = 0.0;
   out_2613280836668770145[145] = 0.0;
   out_2613280836668770145[146] = 0.0;
   out_2613280836668770145[147] = 0.0;
   out_2613280836668770145[148] = 0.0;
   out_2613280836668770145[149] = 0.0;
   out_2613280836668770145[150] = 0.0;
   out_2613280836668770145[151] = 0.0;
   out_2613280836668770145[152] = 1.0;
   out_2613280836668770145[153] = 0.0;
   out_2613280836668770145[154] = 0.0;
   out_2613280836668770145[155] = 0.0;
   out_2613280836668770145[156] = 0.0;
   out_2613280836668770145[157] = 0.0;
   out_2613280836668770145[158] = 0.0;
   out_2613280836668770145[159] = 0.0;
   out_2613280836668770145[160] = 0.0;
   out_2613280836668770145[161] = 0.0;
   out_2613280836668770145[162] = 0.0;
   out_2613280836668770145[163] = 0.0;
   out_2613280836668770145[164] = 0.0;
   out_2613280836668770145[165] = 0.0;
   out_2613280836668770145[166] = 0.0;
   out_2613280836668770145[167] = 0.0;
   out_2613280836668770145[168] = 0.0;
   out_2613280836668770145[169] = 0.0;
   out_2613280836668770145[170] = 0.0;
   out_2613280836668770145[171] = 1.0;
   out_2613280836668770145[172] = 0.0;
   out_2613280836668770145[173] = 0.0;
   out_2613280836668770145[174] = 0.0;
   out_2613280836668770145[175] = 0.0;
   out_2613280836668770145[176] = 0.0;
   out_2613280836668770145[177] = 0.0;
   out_2613280836668770145[178] = 0.0;
   out_2613280836668770145[179] = 0.0;
   out_2613280836668770145[180] = 0.0;
   out_2613280836668770145[181] = 0.0;
   out_2613280836668770145[182] = 0.0;
   out_2613280836668770145[183] = 0.0;
   out_2613280836668770145[184] = 0.0;
   out_2613280836668770145[185] = 0.0;
   out_2613280836668770145[186] = 0.0;
   out_2613280836668770145[187] = 0.0;
   out_2613280836668770145[188] = 0.0;
   out_2613280836668770145[189] = 0.0;
   out_2613280836668770145[190] = 1.0;
   out_2613280836668770145[191] = 0.0;
   out_2613280836668770145[192] = 0.0;
   out_2613280836668770145[193] = 0.0;
   out_2613280836668770145[194] = 0.0;
   out_2613280836668770145[195] = 0.0;
   out_2613280836668770145[196] = 0.0;
   out_2613280836668770145[197] = 0.0;
   out_2613280836668770145[198] = 0.0;
   out_2613280836668770145[199] = 0.0;
   out_2613280836668770145[200] = 0.0;
   out_2613280836668770145[201] = 0.0;
   out_2613280836668770145[202] = 0.0;
   out_2613280836668770145[203] = 0.0;
   out_2613280836668770145[204] = 0.0;
   out_2613280836668770145[205] = 0.0;
   out_2613280836668770145[206] = 0.0;
   out_2613280836668770145[207] = 0.0;
   out_2613280836668770145[208] = 0.0;
   out_2613280836668770145[209] = 1.0;
   out_2613280836668770145[210] = 0.0;
   out_2613280836668770145[211] = 0.0;
   out_2613280836668770145[212] = 0.0;
   out_2613280836668770145[213] = 0.0;
   out_2613280836668770145[214] = 0.0;
   out_2613280836668770145[215] = 0.0;
   out_2613280836668770145[216] = 0.0;
   out_2613280836668770145[217] = 0.0;
   out_2613280836668770145[218] = 0.0;
   out_2613280836668770145[219] = 0.0;
   out_2613280836668770145[220] = 0.0;
   out_2613280836668770145[221] = 0.0;
   out_2613280836668770145[222] = 0.0;
   out_2613280836668770145[223] = 0.0;
   out_2613280836668770145[224] = 0.0;
   out_2613280836668770145[225] = 0.0;
   out_2613280836668770145[226] = 0.0;
   out_2613280836668770145[227] = 0.0;
   out_2613280836668770145[228] = 1.0;
   out_2613280836668770145[229] = 0.0;
   out_2613280836668770145[230] = 0.0;
   out_2613280836668770145[231] = 0.0;
   out_2613280836668770145[232] = 0.0;
   out_2613280836668770145[233] = 0.0;
   out_2613280836668770145[234] = 0.0;
   out_2613280836668770145[235] = 0.0;
   out_2613280836668770145[236] = 0.0;
   out_2613280836668770145[237] = 0.0;
   out_2613280836668770145[238] = 0.0;
   out_2613280836668770145[239] = 0.0;
   out_2613280836668770145[240] = 0.0;
   out_2613280836668770145[241] = 0.0;
   out_2613280836668770145[242] = 0.0;
   out_2613280836668770145[243] = 0.0;
   out_2613280836668770145[244] = 0.0;
   out_2613280836668770145[245] = 0.0;
   out_2613280836668770145[246] = 0.0;
   out_2613280836668770145[247] = 1.0;
   out_2613280836668770145[248] = 0.0;
   out_2613280836668770145[249] = 0.0;
   out_2613280836668770145[250] = 0.0;
   out_2613280836668770145[251] = 0.0;
   out_2613280836668770145[252] = 0.0;
   out_2613280836668770145[253] = 0.0;
   out_2613280836668770145[254] = 0.0;
   out_2613280836668770145[255] = 0.0;
   out_2613280836668770145[256] = 0.0;
   out_2613280836668770145[257] = 0.0;
   out_2613280836668770145[258] = 0.0;
   out_2613280836668770145[259] = 0.0;
   out_2613280836668770145[260] = 0.0;
   out_2613280836668770145[261] = 0.0;
   out_2613280836668770145[262] = 0.0;
   out_2613280836668770145[263] = 0.0;
   out_2613280836668770145[264] = 0.0;
   out_2613280836668770145[265] = 0.0;
   out_2613280836668770145[266] = 1.0;
   out_2613280836668770145[267] = 0.0;
   out_2613280836668770145[268] = 0.0;
   out_2613280836668770145[269] = 0.0;
   out_2613280836668770145[270] = 0.0;
   out_2613280836668770145[271] = 0.0;
   out_2613280836668770145[272] = 0.0;
   out_2613280836668770145[273] = 0.0;
   out_2613280836668770145[274] = 0.0;
   out_2613280836668770145[275] = 0.0;
   out_2613280836668770145[276] = 0.0;
   out_2613280836668770145[277] = 0.0;
   out_2613280836668770145[278] = 0.0;
   out_2613280836668770145[279] = 0.0;
   out_2613280836668770145[280] = 0.0;
   out_2613280836668770145[281] = 0.0;
   out_2613280836668770145[282] = 0.0;
   out_2613280836668770145[283] = 0.0;
   out_2613280836668770145[284] = 0.0;
   out_2613280836668770145[285] = 1.0;
   out_2613280836668770145[286] = 0.0;
   out_2613280836668770145[287] = 0.0;
   out_2613280836668770145[288] = 0.0;
   out_2613280836668770145[289] = 0.0;
   out_2613280836668770145[290] = 0.0;
   out_2613280836668770145[291] = 0.0;
   out_2613280836668770145[292] = 0.0;
   out_2613280836668770145[293] = 0.0;
   out_2613280836668770145[294] = 0.0;
   out_2613280836668770145[295] = 0.0;
   out_2613280836668770145[296] = 0.0;
   out_2613280836668770145[297] = 0.0;
   out_2613280836668770145[298] = 0.0;
   out_2613280836668770145[299] = 0.0;
   out_2613280836668770145[300] = 0.0;
   out_2613280836668770145[301] = 0.0;
   out_2613280836668770145[302] = 0.0;
   out_2613280836668770145[303] = 0.0;
   out_2613280836668770145[304] = 1.0;
   out_2613280836668770145[305] = 0.0;
   out_2613280836668770145[306] = 0.0;
   out_2613280836668770145[307] = 0.0;
   out_2613280836668770145[308] = 0.0;
   out_2613280836668770145[309] = 0.0;
   out_2613280836668770145[310] = 0.0;
   out_2613280836668770145[311] = 0.0;
   out_2613280836668770145[312] = 0.0;
   out_2613280836668770145[313] = 0.0;
   out_2613280836668770145[314] = 0.0;
   out_2613280836668770145[315] = 0.0;
   out_2613280836668770145[316] = 0.0;
   out_2613280836668770145[317] = 0.0;
   out_2613280836668770145[318] = 0.0;
   out_2613280836668770145[319] = 0.0;
   out_2613280836668770145[320] = 0.0;
   out_2613280836668770145[321] = 0.0;
   out_2613280836668770145[322] = 0.0;
   out_2613280836668770145[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3683444243472113040) {
   out_3683444243472113040[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3683444243472113040[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3683444243472113040[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3683444243472113040[3] = dt*state[12] + state[3];
   out_3683444243472113040[4] = dt*state[13] + state[4];
   out_3683444243472113040[5] = dt*state[14] + state[5];
   out_3683444243472113040[6] = state[6];
   out_3683444243472113040[7] = state[7];
   out_3683444243472113040[8] = state[8];
   out_3683444243472113040[9] = state[9];
   out_3683444243472113040[10] = state[10];
   out_3683444243472113040[11] = state[11];
   out_3683444243472113040[12] = state[12];
   out_3683444243472113040[13] = state[13];
   out_3683444243472113040[14] = state[14];
   out_3683444243472113040[15] = state[15];
   out_3683444243472113040[16] = state[16];
   out_3683444243472113040[17] = state[17];
}
void F_fun(double *state, double dt, double *out_3103705100434820586) {
   out_3103705100434820586[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3103705100434820586[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3103705100434820586[2] = 0;
   out_3103705100434820586[3] = 0;
   out_3103705100434820586[4] = 0;
   out_3103705100434820586[5] = 0;
   out_3103705100434820586[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3103705100434820586[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3103705100434820586[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3103705100434820586[9] = 0;
   out_3103705100434820586[10] = 0;
   out_3103705100434820586[11] = 0;
   out_3103705100434820586[12] = 0;
   out_3103705100434820586[13] = 0;
   out_3103705100434820586[14] = 0;
   out_3103705100434820586[15] = 0;
   out_3103705100434820586[16] = 0;
   out_3103705100434820586[17] = 0;
   out_3103705100434820586[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3103705100434820586[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3103705100434820586[20] = 0;
   out_3103705100434820586[21] = 0;
   out_3103705100434820586[22] = 0;
   out_3103705100434820586[23] = 0;
   out_3103705100434820586[24] = 0;
   out_3103705100434820586[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3103705100434820586[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3103705100434820586[27] = 0;
   out_3103705100434820586[28] = 0;
   out_3103705100434820586[29] = 0;
   out_3103705100434820586[30] = 0;
   out_3103705100434820586[31] = 0;
   out_3103705100434820586[32] = 0;
   out_3103705100434820586[33] = 0;
   out_3103705100434820586[34] = 0;
   out_3103705100434820586[35] = 0;
   out_3103705100434820586[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3103705100434820586[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3103705100434820586[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3103705100434820586[39] = 0;
   out_3103705100434820586[40] = 0;
   out_3103705100434820586[41] = 0;
   out_3103705100434820586[42] = 0;
   out_3103705100434820586[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3103705100434820586[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3103705100434820586[45] = 0;
   out_3103705100434820586[46] = 0;
   out_3103705100434820586[47] = 0;
   out_3103705100434820586[48] = 0;
   out_3103705100434820586[49] = 0;
   out_3103705100434820586[50] = 0;
   out_3103705100434820586[51] = 0;
   out_3103705100434820586[52] = 0;
   out_3103705100434820586[53] = 0;
   out_3103705100434820586[54] = 0;
   out_3103705100434820586[55] = 0;
   out_3103705100434820586[56] = 0;
   out_3103705100434820586[57] = 1;
   out_3103705100434820586[58] = 0;
   out_3103705100434820586[59] = 0;
   out_3103705100434820586[60] = 0;
   out_3103705100434820586[61] = 0;
   out_3103705100434820586[62] = 0;
   out_3103705100434820586[63] = 0;
   out_3103705100434820586[64] = 0;
   out_3103705100434820586[65] = 0;
   out_3103705100434820586[66] = dt;
   out_3103705100434820586[67] = 0;
   out_3103705100434820586[68] = 0;
   out_3103705100434820586[69] = 0;
   out_3103705100434820586[70] = 0;
   out_3103705100434820586[71] = 0;
   out_3103705100434820586[72] = 0;
   out_3103705100434820586[73] = 0;
   out_3103705100434820586[74] = 0;
   out_3103705100434820586[75] = 0;
   out_3103705100434820586[76] = 1;
   out_3103705100434820586[77] = 0;
   out_3103705100434820586[78] = 0;
   out_3103705100434820586[79] = 0;
   out_3103705100434820586[80] = 0;
   out_3103705100434820586[81] = 0;
   out_3103705100434820586[82] = 0;
   out_3103705100434820586[83] = 0;
   out_3103705100434820586[84] = 0;
   out_3103705100434820586[85] = dt;
   out_3103705100434820586[86] = 0;
   out_3103705100434820586[87] = 0;
   out_3103705100434820586[88] = 0;
   out_3103705100434820586[89] = 0;
   out_3103705100434820586[90] = 0;
   out_3103705100434820586[91] = 0;
   out_3103705100434820586[92] = 0;
   out_3103705100434820586[93] = 0;
   out_3103705100434820586[94] = 0;
   out_3103705100434820586[95] = 1;
   out_3103705100434820586[96] = 0;
   out_3103705100434820586[97] = 0;
   out_3103705100434820586[98] = 0;
   out_3103705100434820586[99] = 0;
   out_3103705100434820586[100] = 0;
   out_3103705100434820586[101] = 0;
   out_3103705100434820586[102] = 0;
   out_3103705100434820586[103] = 0;
   out_3103705100434820586[104] = dt;
   out_3103705100434820586[105] = 0;
   out_3103705100434820586[106] = 0;
   out_3103705100434820586[107] = 0;
   out_3103705100434820586[108] = 0;
   out_3103705100434820586[109] = 0;
   out_3103705100434820586[110] = 0;
   out_3103705100434820586[111] = 0;
   out_3103705100434820586[112] = 0;
   out_3103705100434820586[113] = 0;
   out_3103705100434820586[114] = 1;
   out_3103705100434820586[115] = 0;
   out_3103705100434820586[116] = 0;
   out_3103705100434820586[117] = 0;
   out_3103705100434820586[118] = 0;
   out_3103705100434820586[119] = 0;
   out_3103705100434820586[120] = 0;
   out_3103705100434820586[121] = 0;
   out_3103705100434820586[122] = 0;
   out_3103705100434820586[123] = 0;
   out_3103705100434820586[124] = 0;
   out_3103705100434820586[125] = 0;
   out_3103705100434820586[126] = 0;
   out_3103705100434820586[127] = 0;
   out_3103705100434820586[128] = 0;
   out_3103705100434820586[129] = 0;
   out_3103705100434820586[130] = 0;
   out_3103705100434820586[131] = 0;
   out_3103705100434820586[132] = 0;
   out_3103705100434820586[133] = 1;
   out_3103705100434820586[134] = 0;
   out_3103705100434820586[135] = 0;
   out_3103705100434820586[136] = 0;
   out_3103705100434820586[137] = 0;
   out_3103705100434820586[138] = 0;
   out_3103705100434820586[139] = 0;
   out_3103705100434820586[140] = 0;
   out_3103705100434820586[141] = 0;
   out_3103705100434820586[142] = 0;
   out_3103705100434820586[143] = 0;
   out_3103705100434820586[144] = 0;
   out_3103705100434820586[145] = 0;
   out_3103705100434820586[146] = 0;
   out_3103705100434820586[147] = 0;
   out_3103705100434820586[148] = 0;
   out_3103705100434820586[149] = 0;
   out_3103705100434820586[150] = 0;
   out_3103705100434820586[151] = 0;
   out_3103705100434820586[152] = 1;
   out_3103705100434820586[153] = 0;
   out_3103705100434820586[154] = 0;
   out_3103705100434820586[155] = 0;
   out_3103705100434820586[156] = 0;
   out_3103705100434820586[157] = 0;
   out_3103705100434820586[158] = 0;
   out_3103705100434820586[159] = 0;
   out_3103705100434820586[160] = 0;
   out_3103705100434820586[161] = 0;
   out_3103705100434820586[162] = 0;
   out_3103705100434820586[163] = 0;
   out_3103705100434820586[164] = 0;
   out_3103705100434820586[165] = 0;
   out_3103705100434820586[166] = 0;
   out_3103705100434820586[167] = 0;
   out_3103705100434820586[168] = 0;
   out_3103705100434820586[169] = 0;
   out_3103705100434820586[170] = 0;
   out_3103705100434820586[171] = 1;
   out_3103705100434820586[172] = 0;
   out_3103705100434820586[173] = 0;
   out_3103705100434820586[174] = 0;
   out_3103705100434820586[175] = 0;
   out_3103705100434820586[176] = 0;
   out_3103705100434820586[177] = 0;
   out_3103705100434820586[178] = 0;
   out_3103705100434820586[179] = 0;
   out_3103705100434820586[180] = 0;
   out_3103705100434820586[181] = 0;
   out_3103705100434820586[182] = 0;
   out_3103705100434820586[183] = 0;
   out_3103705100434820586[184] = 0;
   out_3103705100434820586[185] = 0;
   out_3103705100434820586[186] = 0;
   out_3103705100434820586[187] = 0;
   out_3103705100434820586[188] = 0;
   out_3103705100434820586[189] = 0;
   out_3103705100434820586[190] = 1;
   out_3103705100434820586[191] = 0;
   out_3103705100434820586[192] = 0;
   out_3103705100434820586[193] = 0;
   out_3103705100434820586[194] = 0;
   out_3103705100434820586[195] = 0;
   out_3103705100434820586[196] = 0;
   out_3103705100434820586[197] = 0;
   out_3103705100434820586[198] = 0;
   out_3103705100434820586[199] = 0;
   out_3103705100434820586[200] = 0;
   out_3103705100434820586[201] = 0;
   out_3103705100434820586[202] = 0;
   out_3103705100434820586[203] = 0;
   out_3103705100434820586[204] = 0;
   out_3103705100434820586[205] = 0;
   out_3103705100434820586[206] = 0;
   out_3103705100434820586[207] = 0;
   out_3103705100434820586[208] = 0;
   out_3103705100434820586[209] = 1;
   out_3103705100434820586[210] = 0;
   out_3103705100434820586[211] = 0;
   out_3103705100434820586[212] = 0;
   out_3103705100434820586[213] = 0;
   out_3103705100434820586[214] = 0;
   out_3103705100434820586[215] = 0;
   out_3103705100434820586[216] = 0;
   out_3103705100434820586[217] = 0;
   out_3103705100434820586[218] = 0;
   out_3103705100434820586[219] = 0;
   out_3103705100434820586[220] = 0;
   out_3103705100434820586[221] = 0;
   out_3103705100434820586[222] = 0;
   out_3103705100434820586[223] = 0;
   out_3103705100434820586[224] = 0;
   out_3103705100434820586[225] = 0;
   out_3103705100434820586[226] = 0;
   out_3103705100434820586[227] = 0;
   out_3103705100434820586[228] = 1;
   out_3103705100434820586[229] = 0;
   out_3103705100434820586[230] = 0;
   out_3103705100434820586[231] = 0;
   out_3103705100434820586[232] = 0;
   out_3103705100434820586[233] = 0;
   out_3103705100434820586[234] = 0;
   out_3103705100434820586[235] = 0;
   out_3103705100434820586[236] = 0;
   out_3103705100434820586[237] = 0;
   out_3103705100434820586[238] = 0;
   out_3103705100434820586[239] = 0;
   out_3103705100434820586[240] = 0;
   out_3103705100434820586[241] = 0;
   out_3103705100434820586[242] = 0;
   out_3103705100434820586[243] = 0;
   out_3103705100434820586[244] = 0;
   out_3103705100434820586[245] = 0;
   out_3103705100434820586[246] = 0;
   out_3103705100434820586[247] = 1;
   out_3103705100434820586[248] = 0;
   out_3103705100434820586[249] = 0;
   out_3103705100434820586[250] = 0;
   out_3103705100434820586[251] = 0;
   out_3103705100434820586[252] = 0;
   out_3103705100434820586[253] = 0;
   out_3103705100434820586[254] = 0;
   out_3103705100434820586[255] = 0;
   out_3103705100434820586[256] = 0;
   out_3103705100434820586[257] = 0;
   out_3103705100434820586[258] = 0;
   out_3103705100434820586[259] = 0;
   out_3103705100434820586[260] = 0;
   out_3103705100434820586[261] = 0;
   out_3103705100434820586[262] = 0;
   out_3103705100434820586[263] = 0;
   out_3103705100434820586[264] = 0;
   out_3103705100434820586[265] = 0;
   out_3103705100434820586[266] = 1;
   out_3103705100434820586[267] = 0;
   out_3103705100434820586[268] = 0;
   out_3103705100434820586[269] = 0;
   out_3103705100434820586[270] = 0;
   out_3103705100434820586[271] = 0;
   out_3103705100434820586[272] = 0;
   out_3103705100434820586[273] = 0;
   out_3103705100434820586[274] = 0;
   out_3103705100434820586[275] = 0;
   out_3103705100434820586[276] = 0;
   out_3103705100434820586[277] = 0;
   out_3103705100434820586[278] = 0;
   out_3103705100434820586[279] = 0;
   out_3103705100434820586[280] = 0;
   out_3103705100434820586[281] = 0;
   out_3103705100434820586[282] = 0;
   out_3103705100434820586[283] = 0;
   out_3103705100434820586[284] = 0;
   out_3103705100434820586[285] = 1;
   out_3103705100434820586[286] = 0;
   out_3103705100434820586[287] = 0;
   out_3103705100434820586[288] = 0;
   out_3103705100434820586[289] = 0;
   out_3103705100434820586[290] = 0;
   out_3103705100434820586[291] = 0;
   out_3103705100434820586[292] = 0;
   out_3103705100434820586[293] = 0;
   out_3103705100434820586[294] = 0;
   out_3103705100434820586[295] = 0;
   out_3103705100434820586[296] = 0;
   out_3103705100434820586[297] = 0;
   out_3103705100434820586[298] = 0;
   out_3103705100434820586[299] = 0;
   out_3103705100434820586[300] = 0;
   out_3103705100434820586[301] = 0;
   out_3103705100434820586[302] = 0;
   out_3103705100434820586[303] = 0;
   out_3103705100434820586[304] = 1;
   out_3103705100434820586[305] = 0;
   out_3103705100434820586[306] = 0;
   out_3103705100434820586[307] = 0;
   out_3103705100434820586[308] = 0;
   out_3103705100434820586[309] = 0;
   out_3103705100434820586[310] = 0;
   out_3103705100434820586[311] = 0;
   out_3103705100434820586[312] = 0;
   out_3103705100434820586[313] = 0;
   out_3103705100434820586[314] = 0;
   out_3103705100434820586[315] = 0;
   out_3103705100434820586[316] = 0;
   out_3103705100434820586[317] = 0;
   out_3103705100434820586[318] = 0;
   out_3103705100434820586[319] = 0;
   out_3103705100434820586[320] = 0;
   out_3103705100434820586[321] = 0;
   out_3103705100434820586[322] = 0;
   out_3103705100434820586[323] = 1;
}
void h_4(double *state, double *unused, double *out_5982454180248270517) {
   out_5982454180248270517[0] = state[6] + state[9];
   out_5982454180248270517[1] = state[7] + state[10];
   out_5982454180248270517[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_5813690737770240169) {
   out_5813690737770240169[0] = 0;
   out_5813690737770240169[1] = 0;
   out_5813690737770240169[2] = 0;
   out_5813690737770240169[3] = 0;
   out_5813690737770240169[4] = 0;
   out_5813690737770240169[5] = 0;
   out_5813690737770240169[6] = 1;
   out_5813690737770240169[7] = 0;
   out_5813690737770240169[8] = 0;
   out_5813690737770240169[9] = 1;
   out_5813690737770240169[10] = 0;
   out_5813690737770240169[11] = 0;
   out_5813690737770240169[12] = 0;
   out_5813690737770240169[13] = 0;
   out_5813690737770240169[14] = 0;
   out_5813690737770240169[15] = 0;
   out_5813690737770240169[16] = 0;
   out_5813690737770240169[17] = 0;
   out_5813690737770240169[18] = 0;
   out_5813690737770240169[19] = 0;
   out_5813690737770240169[20] = 0;
   out_5813690737770240169[21] = 0;
   out_5813690737770240169[22] = 0;
   out_5813690737770240169[23] = 0;
   out_5813690737770240169[24] = 0;
   out_5813690737770240169[25] = 1;
   out_5813690737770240169[26] = 0;
   out_5813690737770240169[27] = 0;
   out_5813690737770240169[28] = 1;
   out_5813690737770240169[29] = 0;
   out_5813690737770240169[30] = 0;
   out_5813690737770240169[31] = 0;
   out_5813690737770240169[32] = 0;
   out_5813690737770240169[33] = 0;
   out_5813690737770240169[34] = 0;
   out_5813690737770240169[35] = 0;
   out_5813690737770240169[36] = 0;
   out_5813690737770240169[37] = 0;
   out_5813690737770240169[38] = 0;
   out_5813690737770240169[39] = 0;
   out_5813690737770240169[40] = 0;
   out_5813690737770240169[41] = 0;
   out_5813690737770240169[42] = 0;
   out_5813690737770240169[43] = 0;
   out_5813690737770240169[44] = 1;
   out_5813690737770240169[45] = 0;
   out_5813690737770240169[46] = 0;
   out_5813690737770240169[47] = 1;
   out_5813690737770240169[48] = 0;
   out_5813690737770240169[49] = 0;
   out_5813690737770240169[50] = 0;
   out_5813690737770240169[51] = 0;
   out_5813690737770240169[52] = 0;
   out_5813690737770240169[53] = 0;
}
void h_10(double *state, double *unused, double *out_8845439129170142173) {
   out_8845439129170142173[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_8845439129170142173[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_8845439129170142173[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_5995085404368594946) {
   out_5995085404368594946[0] = 0;
   out_5995085404368594946[1] = 9.8100000000000005*cos(state[1]);
   out_5995085404368594946[2] = 0;
   out_5995085404368594946[3] = 0;
   out_5995085404368594946[4] = -state[8];
   out_5995085404368594946[5] = state[7];
   out_5995085404368594946[6] = 0;
   out_5995085404368594946[7] = state[5];
   out_5995085404368594946[8] = -state[4];
   out_5995085404368594946[9] = 0;
   out_5995085404368594946[10] = 0;
   out_5995085404368594946[11] = 0;
   out_5995085404368594946[12] = 1;
   out_5995085404368594946[13] = 0;
   out_5995085404368594946[14] = 0;
   out_5995085404368594946[15] = 1;
   out_5995085404368594946[16] = 0;
   out_5995085404368594946[17] = 0;
   out_5995085404368594946[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_5995085404368594946[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_5995085404368594946[20] = 0;
   out_5995085404368594946[21] = state[8];
   out_5995085404368594946[22] = 0;
   out_5995085404368594946[23] = -state[6];
   out_5995085404368594946[24] = -state[5];
   out_5995085404368594946[25] = 0;
   out_5995085404368594946[26] = state[3];
   out_5995085404368594946[27] = 0;
   out_5995085404368594946[28] = 0;
   out_5995085404368594946[29] = 0;
   out_5995085404368594946[30] = 0;
   out_5995085404368594946[31] = 1;
   out_5995085404368594946[32] = 0;
   out_5995085404368594946[33] = 0;
   out_5995085404368594946[34] = 1;
   out_5995085404368594946[35] = 0;
   out_5995085404368594946[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_5995085404368594946[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_5995085404368594946[38] = 0;
   out_5995085404368594946[39] = -state[7];
   out_5995085404368594946[40] = state[6];
   out_5995085404368594946[41] = 0;
   out_5995085404368594946[42] = state[4];
   out_5995085404368594946[43] = -state[3];
   out_5995085404368594946[44] = 0;
   out_5995085404368594946[45] = 0;
   out_5995085404368594946[46] = 0;
   out_5995085404368594946[47] = 0;
   out_5995085404368594946[48] = 0;
   out_5995085404368594946[49] = 0;
   out_5995085404368594946[50] = 1;
   out_5995085404368594946[51] = 0;
   out_5995085404368594946[52] = 0;
   out_5995085404368594946[53] = 1;
}
void h_13(double *state, double *unused, double *out_5280177107270464524) {
   out_5280177107270464524[0] = state[3];
   out_5280177107270464524[1] = state[4];
   out_5280177107270464524[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1796940470546460760) {
   out_1796940470546460760[0] = 0;
   out_1796940470546460760[1] = 0;
   out_1796940470546460760[2] = 0;
   out_1796940470546460760[3] = 1;
   out_1796940470546460760[4] = 0;
   out_1796940470546460760[5] = 0;
   out_1796940470546460760[6] = 0;
   out_1796940470546460760[7] = 0;
   out_1796940470546460760[8] = 0;
   out_1796940470546460760[9] = 0;
   out_1796940470546460760[10] = 0;
   out_1796940470546460760[11] = 0;
   out_1796940470546460760[12] = 0;
   out_1796940470546460760[13] = 0;
   out_1796940470546460760[14] = 0;
   out_1796940470546460760[15] = 0;
   out_1796940470546460760[16] = 0;
   out_1796940470546460760[17] = 0;
   out_1796940470546460760[18] = 0;
   out_1796940470546460760[19] = 0;
   out_1796940470546460760[20] = 0;
   out_1796940470546460760[21] = 0;
   out_1796940470546460760[22] = 1;
   out_1796940470546460760[23] = 0;
   out_1796940470546460760[24] = 0;
   out_1796940470546460760[25] = 0;
   out_1796940470546460760[26] = 0;
   out_1796940470546460760[27] = 0;
   out_1796940470546460760[28] = 0;
   out_1796940470546460760[29] = 0;
   out_1796940470546460760[30] = 0;
   out_1796940470546460760[31] = 0;
   out_1796940470546460760[32] = 0;
   out_1796940470546460760[33] = 0;
   out_1796940470546460760[34] = 0;
   out_1796940470546460760[35] = 0;
   out_1796940470546460760[36] = 0;
   out_1796940470546460760[37] = 0;
   out_1796940470546460760[38] = 0;
   out_1796940470546460760[39] = 0;
   out_1796940470546460760[40] = 0;
   out_1796940470546460760[41] = 1;
   out_1796940470546460760[42] = 0;
   out_1796940470546460760[43] = 0;
   out_1796940470546460760[44] = 0;
   out_1796940470546460760[45] = 0;
   out_1796940470546460760[46] = 0;
   out_1796940470546460760[47] = 0;
   out_1796940470546460760[48] = 0;
   out_1796940470546460760[49] = 0;
   out_1796940470546460760[50] = 0;
   out_1796940470546460760[51] = 0;
   out_1796940470546460760[52] = 0;
   out_1796940470546460760[53] = 0;
}
void h_14(double *state, double *unused, double *out_1941905627500734236) {
   out_1941905627500734236[0] = state[6];
   out_1941905627500734236[1] = state[7];
   out_1941905627500734236[2] = state[8];
}
void H_14(double *state, double *unused, double *out_8896479170065612465) {
   out_8896479170065612465[0] = 0;
   out_8896479170065612465[1] = 0;
   out_8896479170065612465[2] = 0;
   out_8896479170065612465[3] = 0;
   out_8896479170065612465[4] = 0;
   out_8896479170065612465[5] = 0;
   out_8896479170065612465[6] = 1;
   out_8896479170065612465[7] = 0;
   out_8896479170065612465[8] = 0;
   out_8896479170065612465[9] = 0;
   out_8896479170065612465[10] = 0;
   out_8896479170065612465[11] = 0;
   out_8896479170065612465[12] = 0;
   out_8896479170065612465[13] = 0;
   out_8896479170065612465[14] = 0;
   out_8896479170065612465[15] = 0;
   out_8896479170065612465[16] = 0;
   out_8896479170065612465[17] = 0;
   out_8896479170065612465[18] = 0;
   out_8896479170065612465[19] = 0;
   out_8896479170065612465[20] = 0;
   out_8896479170065612465[21] = 0;
   out_8896479170065612465[22] = 0;
   out_8896479170065612465[23] = 0;
   out_8896479170065612465[24] = 0;
   out_8896479170065612465[25] = 1;
   out_8896479170065612465[26] = 0;
   out_8896479170065612465[27] = 0;
   out_8896479170065612465[28] = 0;
   out_8896479170065612465[29] = 0;
   out_8896479170065612465[30] = 0;
   out_8896479170065612465[31] = 0;
   out_8896479170065612465[32] = 0;
   out_8896479170065612465[33] = 0;
   out_8896479170065612465[34] = 0;
   out_8896479170065612465[35] = 0;
   out_8896479170065612465[36] = 0;
   out_8896479170065612465[37] = 0;
   out_8896479170065612465[38] = 0;
   out_8896479170065612465[39] = 0;
   out_8896479170065612465[40] = 0;
   out_8896479170065612465[41] = 0;
   out_8896479170065612465[42] = 0;
   out_8896479170065612465[43] = 0;
   out_8896479170065612465[44] = 1;
   out_8896479170065612465[45] = 0;
   out_8896479170065612465[46] = 0;
   out_8896479170065612465[47] = 0;
   out_8896479170065612465[48] = 0;
   out_8896479170065612465[49] = 0;
   out_8896479170065612465[50] = 0;
   out_8896479170065612465[51] = 0;
   out_8896479170065612465[52] = 0;
   out_8896479170065612465[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_5073279989451287369) {
  err_fun(nom_x, delta_x, out_5073279989451287369);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5177693965196878277) {
  inv_err_fun(nom_x, true_x, out_5177693965196878277);
}
void pose_H_mod_fun(double *state, double *out_2613280836668770145) {
  H_mod_fun(state, out_2613280836668770145);
}
void pose_f_fun(double *state, double dt, double *out_3683444243472113040) {
  f_fun(state,  dt, out_3683444243472113040);
}
void pose_F_fun(double *state, double dt, double *out_3103705100434820586) {
  F_fun(state,  dt, out_3103705100434820586);
}
void pose_h_4(double *state, double *unused, double *out_5982454180248270517) {
  h_4(state, unused, out_5982454180248270517);
}
void pose_H_4(double *state, double *unused, double *out_5813690737770240169) {
  H_4(state, unused, out_5813690737770240169);
}
void pose_h_10(double *state, double *unused, double *out_8845439129170142173) {
  h_10(state, unused, out_8845439129170142173);
}
void pose_H_10(double *state, double *unused, double *out_5995085404368594946) {
  H_10(state, unused, out_5995085404368594946);
}
void pose_h_13(double *state, double *unused, double *out_5280177107270464524) {
  h_13(state, unused, out_5280177107270464524);
}
void pose_H_13(double *state, double *unused, double *out_1796940470546460760) {
  H_13(state, unused, out_1796940470546460760);
}
void pose_h_14(double *state, double *unused, double *out_1941905627500734236) {
  h_14(state, unused, out_1941905627500734236);
}
void pose_H_14(double *state, double *unused, double *out_8896479170065612465) {
  H_14(state, unused, out_8896479170065612465);
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
