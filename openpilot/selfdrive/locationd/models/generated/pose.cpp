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
void err_fun(double *nom_x, double *delta_x, double *out_2740203603399450496) {
   out_2740203603399450496[0] = delta_x[0] + nom_x[0];
   out_2740203603399450496[1] = delta_x[1] + nom_x[1];
   out_2740203603399450496[2] = delta_x[2] + nom_x[2];
   out_2740203603399450496[3] = delta_x[3] + nom_x[3];
   out_2740203603399450496[4] = delta_x[4] + nom_x[4];
   out_2740203603399450496[5] = delta_x[5] + nom_x[5];
   out_2740203603399450496[6] = delta_x[6] + nom_x[6];
   out_2740203603399450496[7] = delta_x[7] + nom_x[7];
   out_2740203603399450496[8] = delta_x[8] + nom_x[8];
   out_2740203603399450496[9] = delta_x[9] + nom_x[9];
   out_2740203603399450496[10] = delta_x[10] + nom_x[10];
   out_2740203603399450496[11] = delta_x[11] + nom_x[11];
   out_2740203603399450496[12] = delta_x[12] + nom_x[12];
   out_2740203603399450496[13] = delta_x[13] + nom_x[13];
   out_2740203603399450496[14] = delta_x[14] + nom_x[14];
   out_2740203603399450496[15] = delta_x[15] + nom_x[15];
   out_2740203603399450496[16] = delta_x[16] + nom_x[16];
   out_2740203603399450496[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2944854366871732438) {
   out_2944854366871732438[0] = -nom_x[0] + true_x[0];
   out_2944854366871732438[1] = -nom_x[1] + true_x[1];
   out_2944854366871732438[2] = -nom_x[2] + true_x[2];
   out_2944854366871732438[3] = -nom_x[3] + true_x[3];
   out_2944854366871732438[4] = -nom_x[4] + true_x[4];
   out_2944854366871732438[5] = -nom_x[5] + true_x[5];
   out_2944854366871732438[6] = -nom_x[6] + true_x[6];
   out_2944854366871732438[7] = -nom_x[7] + true_x[7];
   out_2944854366871732438[8] = -nom_x[8] + true_x[8];
   out_2944854366871732438[9] = -nom_x[9] + true_x[9];
   out_2944854366871732438[10] = -nom_x[10] + true_x[10];
   out_2944854366871732438[11] = -nom_x[11] + true_x[11];
   out_2944854366871732438[12] = -nom_x[12] + true_x[12];
   out_2944854366871732438[13] = -nom_x[13] + true_x[13];
   out_2944854366871732438[14] = -nom_x[14] + true_x[14];
   out_2944854366871732438[15] = -nom_x[15] + true_x[15];
   out_2944854366871732438[16] = -nom_x[16] + true_x[16];
   out_2944854366871732438[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1017226251035741065) {
   out_1017226251035741065[0] = 1.0;
   out_1017226251035741065[1] = 0.0;
   out_1017226251035741065[2] = 0.0;
   out_1017226251035741065[3] = 0.0;
   out_1017226251035741065[4] = 0.0;
   out_1017226251035741065[5] = 0.0;
   out_1017226251035741065[6] = 0.0;
   out_1017226251035741065[7] = 0.0;
   out_1017226251035741065[8] = 0.0;
   out_1017226251035741065[9] = 0.0;
   out_1017226251035741065[10] = 0.0;
   out_1017226251035741065[11] = 0.0;
   out_1017226251035741065[12] = 0.0;
   out_1017226251035741065[13] = 0.0;
   out_1017226251035741065[14] = 0.0;
   out_1017226251035741065[15] = 0.0;
   out_1017226251035741065[16] = 0.0;
   out_1017226251035741065[17] = 0.0;
   out_1017226251035741065[18] = 0.0;
   out_1017226251035741065[19] = 1.0;
   out_1017226251035741065[20] = 0.0;
   out_1017226251035741065[21] = 0.0;
   out_1017226251035741065[22] = 0.0;
   out_1017226251035741065[23] = 0.0;
   out_1017226251035741065[24] = 0.0;
   out_1017226251035741065[25] = 0.0;
   out_1017226251035741065[26] = 0.0;
   out_1017226251035741065[27] = 0.0;
   out_1017226251035741065[28] = 0.0;
   out_1017226251035741065[29] = 0.0;
   out_1017226251035741065[30] = 0.0;
   out_1017226251035741065[31] = 0.0;
   out_1017226251035741065[32] = 0.0;
   out_1017226251035741065[33] = 0.0;
   out_1017226251035741065[34] = 0.0;
   out_1017226251035741065[35] = 0.0;
   out_1017226251035741065[36] = 0.0;
   out_1017226251035741065[37] = 0.0;
   out_1017226251035741065[38] = 1.0;
   out_1017226251035741065[39] = 0.0;
   out_1017226251035741065[40] = 0.0;
   out_1017226251035741065[41] = 0.0;
   out_1017226251035741065[42] = 0.0;
   out_1017226251035741065[43] = 0.0;
   out_1017226251035741065[44] = 0.0;
   out_1017226251035741065[45] = 0.0;
   out_1017226251035741065[46] = 0.0;
   out_1017226251035741065[47] = 0.0;
   out_1017226251035741065[48] = 0.0;
   out_1017226251035741065[49] = 0.0;
   out_1017226251035741065[50] = 0.0;
   out_1017226251035741065[51] = 0.0;
   out_1017226251035741065[52] = 0.0;
   out_1017226251035741065[53] = 0.0;
   out_1017226251035741065[54] = 0.0;
   out_1017226251035741065[55] = 0.0;
   out_1017226251035741065[56] = 0.0;
   out_1017226251035741065[57] = 1.0;
   out_1017226251035741065[58] = 0.0;
   out_1017226251035741065[59] = 0.0;
   out_1017226251035741065[60] = 0.0;
   out_1017226251035741065[61] = 0.0;
   out_1017226251035741065[62] = 0.0;
   out_1017226251035741065[63] = 0.0;
   out_1017226251035741065[64] = 0.0;
   out_1017226251035741065[65] = 0.0;
   out_1017226251035741065[66] = 0.0;
   out_1017226251035741065[67] = 0.0;
   out_1017226251035741065[68] = 0.0;
   out_1017226251035741065[69] = 0.0;
   out_1017226251035741065[70] = 0.0;
   out_1017226251035741065[71] = 0.0;
   out_1017226251035741065[72] = 0.0;
   out_1017226251035741065[73] = 0.0;
   out_1017226251035741065[74] = 0.0;
   out_1017226251035741065[75] = 0.0;
   out_1017226251035741065[76] = 1.0;
   out_1017226251035741065[77] = 0.0;
   out_1017226251035741065[78] = 0.0;
   out_1017226251035741065[79] = 0.0;
   out_1017226251035741065[80] = 0.0;
   out_1017226251035741065[81] = 0.0;
   out_1017226251035741065[82] = 0.0;
   out_1017226251035741065[83] = 0.0;
   out_1017226251035741065[84] = 0.0;
   out_1017226251035741065[85] = 0.0;
   out_1017226251035741065[86] = 0.0;
   out_1017226251035741065[87] = 0.0;
   out_1017226251035741065[88] = 0.0;
   out_1017226251035741065[89] = 0.0;
   out_1017226251035741065[90] = 0.0;
   out_1017226251035741065[91] = 0.0;
   out_1017226251035741065[92] = 0.0;
   out_1017226251035741065[93] = 0.0;
   out_1017226251035741065[94] = 0.0;
   out_1017226251035741065[95] = 1.0;
   out_1017226251035741065[96] = 0.0;
   out_1017226251035741065[97] = 0.0;
   out_1017226251035741065[98] = 0.0;
   out_1017226251035741065[99] = 0.0;
   out_1017226251035741065[100] = 0.0;
   out_1017226251035741065[101] = 0.0;
   out_1017226251035741065[102] = 0.0;
   out_1017226251035741065[103] = 0.0;
   out_1017226251035741065[104] = 0.0;
   out_1017226251035741065[105] = 0.0;
   out_1017226251035741065[106] = 0.0;
   out_1017226251035741065[107] = 0.0;
   out_1017226251035741065[108] = 0.0;
   out_1017226251035741065[109] = 0.0;
   out_1017226251035741065[110] = 0.0;
   out_1017226251035741065[111] = 0.0;
   out_1017226251035741065[112] = 0.0;
   out_1017226251035741065[113] = 0.0;
   out_1017226251035741065[114] = 1.0;
   out_1017226251035741065[115] = 0.0;
   out_1017226251035741065[116] = 0.0;
   out_1017226251035741065[117] = 0.0;
   out_1017226251035741065[118] = 0.0;
   out_1017226251035741065[119] = 0.0;
   out_1017226251035741065[120] = 0.0;
   out_1017226251035741065[121] = 0.0;
   out_1017226251035741065[122] = 0.0;
   out_1017226251035741065[123] = 0.0;
   out_1017226251035741065[124] = 0.0;
   out_1017226251035741065[125] = 0.0;
   out_1017226251035741065[126] = 0.0;
   out_1017226251035741065[127] = 0.0;
   out_1017226251035741065[128] = 0.0;
   out_1017226251035741065[129] = 0.0;
   out_1017226251035741065[130] = 0.0;
   out_1017226251035741065[131] = 0.0;
   out_1017226251035741065[132] = 0.0;
   out_1017226251035741065[133] = 1.0;
   out_1017226251035741065[134] = 0.0;
   out_1017226251035741065[135] = 0.0;
   out_1017226251035741065[136] = 0.0;
   out_1017226251035741065[137] = 0.0;
   out_1017226251035741065[138] = 0.0;
   out_1017226251035741065[139] = 0.0;
   out_1017226251035741065[140] = 0.0;
   out_1017226251035741065[141] = 0.0;
   out_1017226251035741065[142] = 0.0;
   out_1017226251035741065[143] = 0.0;
   out_1017226251035741065[144] = 0.0;
   out_1017226251035741065[145] = 0.0;
   out_1017226251035741065[146] = 0.0;
   out_1017226251035741065[147] = 0.0;
   out_1017226251035741065[148] = 0.0;
   out_1017226251035741065[149] = 0.0;
   out_1017226251035741065[150] = 0.0;
   out_1017226251035741065[151] = 0.0;
   out_1017226251035741065[152] = 1.0;
   out_1017226251035741065[153] = 0.0;
   out_1017226251035741065[154] = 0.0;
   out_1017226251035741065[155] = 0.0;
   out_1017226251035741065[156] = 0.0;
   out_1017226251035741065[157] = 0.0;
   out_1017226251035741065[158] = 0.0;
   out_1017226251035741065[159] = 0.0;
   out_1017226251035741065[160] = 0.0;
   out_1017226251035741065[161] = 0.0;
   out_1017226251035741065[162] = 0.0;
   out_1017226251035741065[163] = 0.0;
   out_1017226251035741065[164] = 0.0;
   out_1017226251035741065[165] = 0.0;
   out_1017226251035741065[166] = 0.0;
   out_1017226251035741065[167] = 0.0;
   out_1017226251035741065[168] = 0.0;
   out_1017226251035741065[169] = 0.0;
   out_1017226251035741065[170] = 0.0;
   out_1017226251035741065[171] = 1.0;
   out_1017226251035741065[172] = 0.0;
   out_1017226251035741065[173] = 0.0;
   out_1017226251035741065[174] = 0.0;
   out_1017226251035741065[175] = 0.0;
   out_1017226251035741065[176] = 0.0;
   out_1017226251035741065[177] = 0.0;
   out_1017226251035741065[178] = 0.0;
   out_1017226251035741065[179] = 0.0;
   out_1017226251035741065[180] = 0.0;
   out_1017226251035741065[181] = 0.0;
   out_1017226251035741065[182] = 0.0;
   out_1017226251035741065[183] = 0.0;
   out_1017226251035741065[184] = 0.0;
   out_1017226251035741065[185] = 0.0;
   out_1017226251035741065[186] = 0.0;
   out_1017226251035741065[187] = 0.0;
   out_1017226251035741065[188] = 0.0;
   out_1017226251035741065[189] = 0.0;
   out_1017226251035741065[190] = 1.0;
   out_1017226251035741065[191] = 0.0;
   out_1017226251035741065[192] = 0.0;
   out_1017226251035741065[193] = 0.0;
   out_1017226251035741065[194] = 0.0;
   out_1017226251035741065[195] = 0.0;
   out_1017226251035741065[196] = 0.0;
   out_1017226251035741065[197] = 0.0;
   out_1017226251035741065[198] = 0.0;
   out_1017226251035741065[199] = 0.0;
   out_1017226251035741065[200] = 0.0;
   out_1017226251035741065[201] = 0.0;
   out_1017226251035741065[202] = 0.0;
   out_1017226251035741065[203] = 0.0;
   out_1017226251035741065[204] = 0.0;
   out_1017226251035741065[205] = 0.0;
   out_1017226251035741065[206] = 0.0;
   out_1017226251035741065[207] = 0.0;
   out_1017226251035741065[208] = 0.0;
   out_1017226251035741065[209] = 1.0;
   out_1017226251035741065[210] = 0.0;
   out_1017226251035741065[211] = 0.0;
   out_1017226251035741065[212] = 0.0;
   out_1017226251035741065[213] = 0.0;
   out_1017226251035741065[214] = 0.0;
   out_1017226251035741065[215] = 0.0;
   out_1017226251035741065[216] = 0.0;
   out_1017226251035741065[217] = 0.0;
   out_1017226251035741065[218] = 0.0;
   out_1017226251035741065[219] = 0.0;
   out_1017226251035741065[220] = 0.0;
   out_1017226251035741065[221] = 0.0;
   out_1017226251035741065[222] = 0.0;
   out_1017226251035741065[223] = 0.0;
   out_1017226251035741065[224] = 0.0;
   out_1017226251035741065[225] = 0.0;
   out_1017226251035741065[226] = 0.0;
   out_1017226251035741065[227] = 0.0;
   out_1017226251035741065[228] = 1.0;
   out_1017226251035741065[229] = 0.0;
   out_1017226251035741065[230] = 0.0;
   out_1017226251035741065[231] = 0.0;
   out_1017226251035741065[232] = 0.0;
   out_1017226251035741065[233] = 0.0;
   out_1017226251035741065[234] = 0.0;
   out_1017226251035741065[235] = 0.0;
   out_1017226251035741065[236] = 0.0;
   out_1017226251035741065[237] = 0.0;
   out_1017226251035741065[238] = 0.0;
   out_1017226251035741065[239] = 0.0;
   out_1017226251035741065[240] = 0.0;
   out_1017226251035741065[241] = 0.0;
   out_1017226251035741065[242] = 0.0;
   out_1017226251035741065[243] = 0.0;
   out_1017226251035741065[244] = 0.0;
   out_1017226251035741065[245] = 0.0;
   out_1017226251035741065[246] = 0.0;
   out_1017226251035741065[247] = 1.0;
   out_1017226251035741065[248] = 0.0;
   out_1017226251035741065[249] = 0.0;
   out_1017226251035741065[250] = 0.0;
   out_1017226251035741065[251] = 0.0;
   out_1017226251035741065[252] = 0.0;
   out_1017226251035741065[253] = 0.0;
   out_1017226251035741065[254] = 0.0;
   out_1017226251035741065[255] = 0.0;
   out_1017226251035741065[256] = 0.0;
   out_1017226251035741065[257] = 0.0;
   out_1017226251035741065[258] = 0.0;
   out_1017226251035741065[259] = 0.0;
   out_1017226251035741065[260] = 0.0;
   out_1017226251035741065[261] = 0.0;
   out_1017226251035741065[262] = 0.0;
   out_1017226251035741065[263] = 0.0;
   out_1017226251035741065[264] = 0.0;
   out_1017226251035741065[265] = 0.0;
   out_1017226251035741065[266] = 1.0;
   out_1017226251035741065[267] = 0.0;
   out_1017226251035741065[268] = 0.0;
   out_1017226251035741065[269] = 0.0;
   out_1017226251035741065[270] = 0.0;
   out_1017226251035741065[271] = 0.0;
   out_1017226251035741065[272] = 0.0;
   out_1017226251035741065[273] = 0.0;
   out_1017226251035741065[274] = 0.0;
   out_1017226251035741065[275] = 0.0;
   out_1017226251035741065[276] = 0.0;
   out_1017226251035741065[277] = 0.0;
   out_1017226251035741065[278] = 0.0;
   out_1017226251035741065[279] = 0.0;
   out_1017226251035741065[280] = 0.0;
   out_1017226251035741065[281] = 0.0;
   out_1017226251035741065[282] = 0.0;
   out_1017226251035741065[283] = 0.0;
   out_1017226251035741065[284] = 0.0;
   out_1017226251035741065[285] = 1.0;
   out_1017226251035741065[286] = 0.0;
   out_1017226251035741065[287] = 0.0;
   out_1017226251035741065[288] = 0.0;
   out_1017226251035741065[289] = 0.0;
   out_1017226251035741065[290] = 0.0;
   out_1017226251035741065[291] = 0.0;
   out_1017226251035741065[292] = 0.0;
   out_1017226251035741065[293] = 0.0;
   out_1017226251035741065[294] = 0.0;
   out_1017226251035741065[295] = 0.0;
   out_1017226251035741065[296] = 0.0;
   out_1017226251035741065[297] = 0.0;
   out_1017226251035741065[298] = 0.0;
   out_1017226251035741065[299] = 0.0;
   out_1017226251035741065[300] = 0.0;
   out_1017226251035741065[301] = 0.0;
   out_1017226251035741065[302] = 0.0;
   out_1017226251035741065[303] = 0.0;
   out_1017226251035741065[304] = 1.0;
   out_1017226251035741065[305] = 0.0;
   out_1017226251035741065[306] = 0.0;
   out_1017226251035741065[307] = 0.0;
   out_1017226251035741065[308] = 0.0;
   out_1017226251035741065[309] = 0.0;
   out_1017226251035741065[310] = 0.0;
   out_1017226251035741065[311] = 0.0;
   out_1017226251035741065[312] = 0.0;
   out_1017226251035741065[313] = 0.0;
   out_1017226251035741065[314] = 0.0;
   out_1017226251035741065[315] = 0.0;
   out_1017226251035741065[316] = 0.0;
   out_1017226251035741065[317] = 0.0;
   out_1017226251035741065[318] = 0.0;
   out_1017226251035741065[319] = 0.0;
   out_1017226251035741065[320] = 0.0;
   out_1017226251035741065[321] = 0.0;
   out_1017226251035741065[322] = 0.0;
   out_1017226251035741065[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3838696256166161148) {
   out_3838696256166161148[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3838696256166161148[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3838696256166161148[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3838696256166161148[3] = dt*state[12] + state[3];
   out_3838696256166161148[4] = dt*state[13] + state[4];
   out_3838696256166161148[5] = dt*state[14] + state[5];
   out_3838696256166161148[6] = state[6];
   out_3838696256166161148[7] = state[7];
   out_3838696256166161148[8] = state[8];
   out_3838696256166161148[9] = state[9];
   out_3838696256166161148[10] = state[10];
   out_3838696256166161148[11] = state[11];
   out_3838696256166161148[12] = state[12];
   out_3838696256166161148[13] = state[13];
   out_3838696256166161148[14] = state[14];
   out_3838696256166161148[15] = state[15];
   out_3838696256166161148[16] = state[16];
   out_3838696256166161148[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6662765533021581793) {
   out_6662765533021581793[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6662765533021581793[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6662765533021581793[2] = 0;
   out_6662765533021581793[3] = 0;
   out_6662765533021581793[4] = 0;
   out_6662765533021581793[5] = 0;
   out_6662765533021581793[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6662765533021581793[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6662765533021581793[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6662765533021581793[9] = 0;
   out_6662765533021581793[10] = 0;
   out_6662765533021581793[11] = 0;
   out_6662765533021581793[12] = 0;
   out_6662765533021581793[13] = 0;
   out_6662765533021581793[14] = 0;
   out_6662765533021581793[15] = 0;
   out_6662765533021581793[16] = 0;
   out_6662765533021581793[17] = 0;
   out_6662765533021581793[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6662765533021581793[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6662765533021581793[20] = 0;
   out_6662765533021581793[21] = 0;
   out_6662765533021581793[22] = 0;
   out_6662765533021581793[23] = 0;
   out_6662765533021581793[24] = 0;
   out_6662765533021581793[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6662765533021581793[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6662765533021581793[27] = 0;
   out_6662765533021581793[28] = 0;
   out_6662765533021581793[29] = 0;
   out_6662765533021581793[30] = 0;
   out_6662765533021581793[31] = 0;
   out_6662765533021581793[32] = 0;
   out_6662765533021581793[33] = 0;
   out_6662765533021581793[34] = 0;
   out_6662765533021581793[35] = 0;
   out_6662765533021581793[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6662765533021581793[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6662765533021581793[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6662765533021581793[39] = 0;
   out_6662765533021581793[40] = 0;
   out_6662765533021581793[41] = 0;
   out_6662765533021581793[42] = 0;
   out_6662765533021581793[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6662765533021581793[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6662765533021581793[45] = 0;
   out_6662765533021581793[46] = 0;
   out_6662765533021581793[47] = 0;
   out_6662765533021581793[48] = 0;
   out_6662765533021581793[49] = 0;
   out_6662765533021581793[50] = 0;
   out_6662765533021581793[51] = 0;
   out_6662765533021581793[52] = 0;
   out_6662765533021581793[53] = 0;
   out_6662765533021581793[54] = 0;
   out_6662765533021581793[55] = 0;
   out_6662765533021581793[56] = 0;
   out_6662765533021581793[57] = 1;
   out_6662765533021581793[58] = 0;
   out_6662765533021581793[59] = 0;
   out_6662765533021581793[60] = 0;
   out_6662765533021581793[61] = 0;
   out_6662765533021581793[62] = 0;
   out_6662765533021581793[63] = 0;
   out_6662765533021581793[64] = 0;
   out_6662765533021581793[65] = 0;
   out_6662765533021581793[66] = dt;
   out_6662765533021581793[67] = 0;
   out_6662765533021581793[68] = 0;
   out_6662765533021581793[69] = 0;
   out_6662765533021581793[70] = 0;
   out_6662765533021581793[71] = 0;
   out_6662765533021581793[72] = 0;
   out_6662765533021581793[73] = 0;
   out_6662765533021581793[74] = 0;
   out_6662765533021581793[75] = 0;
   out_6662765533021581793[76] = 1;
   out_6662765533021581793[77] = 0;
   out_6662765533021581793[78] = 0;
   out_6662765533021581793[79] = 0;
   out_6662765533021581793[80] = 0;
   out_6662765533021581793[81] = 0;
   out_6662765533021581793[82] = 0;
   out_6662765533021581793[83] = 0;
   out_6662765533021581793[84] = 0;
   out_6662765533021581793[85] = dt;
   out_6662765533021581793[86] = 0;
   out_6662765533021581793[87] = 0;
   out_6662765533021581793[88] = 0;
   out_6662765533021581793[89] = 0;
   out_6662765533021581793[90] = 0;
   out_6662765533021581793[91] = 0;
   out_6662765533021581793[92] = 0;
   out_6662765533021581793[93] = 0;
   out_6662765533021581793[94] = 0;
   out_6662765533021581793[95] = 1;
   out_6662765533021581793[96] = 0;
   out_6662765533021581793[97] = 0;
   out_6662765533021581793[98] = 0;
   out_6662765533021581793[99] = 0;
   out_6662765533021581793[100] = 0;
   out_6662765533021581793[101] = 0;
   out_6662765533021581793[102] = 0;
   out_6662765533021581793[103] = 0;
   out_6662765533021581793[104] = dt;
   out_6662765533021581793[105] = 0;
   out_6662765533021581793[106] = 0;
   out_6662765533021581793[107] = 0;
   out_6662765533021581793[108] = 0;
   out_6662765533021581793[109] = 0;
   out_6662765533021581793[110] = 0;
   out_6662765533021581793[111] = 0;
   out_6662765533021581793[112] = 0;
   out_6662765533021581793[113] = 0;
   out_6662765533021581793[114] = 1;
   out_6662765533021581793[115] = 0;
   out_6662765533021581793[116] = 0;
   out_6662765533021581793[117] = 0;
   out_6662765533021581793[118] = 0;
   out_6662765533021581793[119] = 0;
   out_6662765533021581793[120] = 0;
   out_6662765533021581793[121] = 0;
   out_6662765533021581793[122] = 0;
   out_6662765533021581793[123] = 0;
   out_6662765533021581793[124] = 0;
   out_6662765533021581793[125] = 0;
   out_6662765533021581793[126] = 0;
   out_6662765533021581793[127] = 0;
   out_6662765533021581793[128] = 0;
   out_6662765533021581793[129] = 0;
   out_6662765533021581793[130] = 0;
   out_6662765533021581793[131] = 0;
   out_6662765533021581793[132] = 0;
   out_6662765533021581793[133] = 1;
   out_6662765533021581793[134] = 0;
   out_6662765533021581793[135] = 0;
   out_6662765533021581793[136] = 0;
   out_6662765533021581793[137] = 0;
   out_6662765533021581793[138] = 0;
   out_6662765533021581793[139] = 0;
   out_6662765533021581793[140] = 0;
   out_6662765533021581793[141] = 0;
   out_6662765533021581793[142] = 0;
   out_6662765533021581793[143] = 0;
   out_6662765533021581793[144] = 0;
   out_6662765533021581793[145] = 0;
   out_6662765533021581793[146] = 0;
   out_6662765533021581793[147] = 0;
   out_6662765533021581793[148] = 0;
   out_6662765533021581793[149] = 0;
   out_6662765533021581793[150] = 0;
   out_6662765533021581793[151] = 0;
   out_6662765533021581793[152] = 1;
   out_6662765533021581793[153] = 0;
   out_6662765533021581793[154] = 0;
   out_6662765533021581793[155] = 0;
   out_6662765533021581793[156] = 0;
   out_6662765533021581793[157] = 0;
   out_6662765533021581793[158] = 0;
   out_6662765533021581793[159] = 0;
   out_6662765533021581793[160] = 0;
   out_6662765533021581793[161] = 0;
   out_6662765533021581793[162] = 0;
   out_6662765533021581793[163] = 0;
   out_6662765533021581793[164] = 0;
   out_6662765533021581793[165] = 0;
   out_6662765533021581793[166] = 0;
   out_6662765533021581793[167] = 0;
   out_6662765533021581793[168] = 0;
   out_6662765533021581793[169] = 0;
   out_6662765533021581793[170] = 0;
   out_6662765533021581793[171] = 1;
   out_6662765533021581793[172] = 0;
   out_6662765533021581793[173] = 0;
   out_6662765533021581793[174] = 0;
   out_6662765533021581793[175] = 0;
   out_6662765533021581793[176] = 0;
   out_6662765533021581793[177] = 0;
   out_6662765533021581793[178] = 0;
   out_6662765533021581793[179] = 0;
   out_6662765533021581793[180] = 0;
   out_6662765533021581793[181] = 0;
   out_6662765533021581793[182] = 0;
   out_6662765533021581793[183] = 0;
   out_6662765533021581793[184] = 0;
   out_6662765533021581793[185] = 0;
   out_6662765533021581793[186] = 0;
   out_6662765533021581793[187] = 0;
   out_6662765533021581793[188] = 0;
   out_6662765533021581793[189] = 0;
   out_6662765533021581793[190] = 1;
   out_6662765533021581793[191] = 0;
   out_6662765533021581793[192] = 0;
   out_6662765533021581793[193] = 0;
   out_6662765533021581793[194] = 0;
   out_6662765533021581793[195] = 0;
   out_6662765533021581793[196] = 0;
   out_6662765533021581793[197] = 0;
   out_6662765533021581793[198] = 0;
   out_6662765533021581793[199] = 0;
   out_6662765533021581793[200] = 0;
   out_6662765533021581793[201] = 0;
   out_6662765533021581793[202] = 0;
   out_6662765533021581793[203] = 0;
   out_6662765533021581793[204] = 0;
   out_6662765533021581793[205] = 0;
   out_6662765533021581793[206] = 0;
   out_6662765533021581793[207] = 0;
   out_6662765533021581793[208] = 0;
   out_6662765533021581793[209] = 1;
   out_6662765533021581793[210] = 0;
   out_6662765533021581793[211] = 0;
   out_6662765533021581793[212] = 0;
   out_6662765533021581793[213] = 0;
   out_6662765533021581793[214] = 0;
   out_6662765533021581793[215] = 0;
   out_6662765533021581793[216] = 0;
   out_6662765533021581793[217] = 0;
   out_6662765533021581793[218] = 0;
   out_6662765533021581793[219] = 0;
   out_6662765533021581793[220] = 0;
   out_6662765533021581793[221] = 0;
   out_6662765533021581793[222] = 0;
   out_6662765533021581793[223] = 0;
   out_6662765533021581793[224] = 0;
   out_6662765533021581793[225] = 0;
   out_6662765533021581793[226] = 0;
   out_6662765533021581793[227] = 0;
   out_6662765533021581793[228] = 1;
   out_6662765533021581793[229] = 0;
   out_6662765533021581793[230] = 0;
   out_6662765533021581793[231] = 0;
   out_6662765533021581793[232] = 0;
   out_6662765533021581793[233] = 0;
   out_6662765533021581793[234] = 0;
   out_6662765533021581793[235] = 0;
   out_6662765533021581793[236] = 0;
   out_6662765533021581793[237] = 0;
   out_6662765533021581793[238] = 0;
   out_6662765533021581793[239] = 0;
   out_6662765533021581793[240] = 0;
   out_6662765533021581793[241] = 0;
   out_6662765533021581793[242] = 0;
   out_6662765533021581793[243] = 0;
   out_6662765533021581793[244] = 0;
   out_6662765533021581793[245] = 0;
   out_6662765533021581793[246] = 0;
   out_6662765533021581793[247] = 1;
   out_6662765533021581793[248] = 0;
   out_6662765533021581793[249] = 0;
   out_6662765533021581793[250] = 0;
   out_6662765533021581793[251] = 0;
   out_6662765533021581793[252] = 0;
   out_6662765533021581793[253] = 0;
   out_6662765533021581793[254] = 0;
   out_6662765533021581793[255] = 0;
   out_6662765533021581793[256] = 0;
   out_6662765533021581793[257] = 0;
   out_6662765533021581793[258] = 0;
   out_6662765533021581793[259] = 0;
   out_6662765533021581793[260] = 0;
   out_6662765533021581793[261] = 0;
   out_6662765533021581793[262] = 0;
   out_6662765533021581793[263] = 0;
   out_6662765533021581793[264] = 0;
   out_6662765533021581793[265] = 0;
   out_6662765533021581793[266] = 1;
   out_6662765533021581793[267] = 0;
   out_6662765533021581793[268] = 0;
   out_6662765533021581793[269] = 0;
   out_6662765533021581793[270] = 0;
   out_6662765533021581793[271] = 0;
   out_6662765533021581793[272] = 0;
   out_6662765533021581793[273] = 0;
   out_6662765533021581793[274] = 0;
   out_6662765533021581793[275] = 0;
   out_6662765533021581793[276] = 0;
   out_6662765533021581793[277] = 0;
   out_6662765533021581793[278] = 0;
   out_6662765533021581793[279] = 0;
   out_6662765533021581793[280] = 0;
   out_6662765533021581793[281] = 0;
   out_6662765533021581793[282] = 0;
   out_6662765533021581793[283] = 0;
   out_6662765533021581793[284] = 0;
   out_6662765533021581793[285] = 1;
   out_6662765533021581793[286] = 0;
   out_6662765533021581793[287] = 0;
   out_6662765533021581793[288] = 0;
   out_6662765533021581793[289] = 0;
   out_6662765533021581793[290] = 0;
   out_6662765533021581793[291] = 0;
   out_6662765533021581793[292] = 0;
   out_6662765533021581793[293] = 0;
   out_6662765533021581793[294] = 0;
   out_6662765533021581793[295] = 0;
   out_6662765533021581793[296] = 0;
   out_6662765533021581793[297] = 0;
   out_6662765533021581793[298] = 0;
   out_6662765533021581793[299] = 0;
   out_6662765533021581793[300] = 0;
   out_6662765533021581793[301] = 0;
   out_6662765533021581793[302] = 0;
   out_6662765533021581793[303] = 0;
   out_6662765533021581793[304] = 1;
   out_6662765533021581793[305] = 0;
   out_6662765533021581793[306] = 0;
   out_6662765533021581793[307] = 0;
   out_6662765533021581793[308] = 0;
   out_6662765533021581793[309] = 0;
   out_6662765533021581793[310] = 0;
   out_6662765533021581793[311] = 0;
   out_6662765533021581793[312] = 0;
   out_6662765533021581793[313] = 0;
   out_6662765533021581793[314] = 0;
   out_6662765533021581793[315] = 0;
   out_6662765533021581793[316] = 0;
   out_6662765533021581793[317] = 0;
   out_6662765533021581793[318] = 0;
   out_6662765533021581793[319] = 0;
   out_6662765533021581793[320] = 0;
   out_6662765533021581793[321] = 0;
   out_6662765533021581793[322] = 0;
   out_6662765533021581793[323] = 1;
}
void h_4(double *state, double *unused, double *out_2834632828994169778) {
   out_2834632828994169778[0] = state[6] + state[9];
   out_2834632828994169778[1] = state[7] + state[10];
   out_2834632828994169778[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4086798773631621081) {
   out_4086798773631621081[0] = 0;
   out_4086798773631621081[1] = 0;
   out_4086798773631621081[2] = 0;
   out_4086798773631621081[3] = 0;
   out_4086798773631621081[4] = 0;
   out_4086798773631621081[5] = 0;
   out_4086798773631621081[6] = 1;
   out_4086798773631621081[7] = 0;
   out_4086798773631621081[8] = 0;
   out_4086798773631621081[9] = 1;
   out_4086798773631621081[10] = 0;
   out_4086798773631621081[11] = 0;
   out_4086798773631621081[12] = 0;
   out_4086798773631621081[13] = 0;
   out_4086798773631621081[14] = 0;
   out_4086798773631621081[15] = 0;
   out_4086798773631621081[16] = 0;
   out_4086798773631621081[17] = 0;
   out_4086798773631621081[18] = 0;
   out_4086798773631621081[19] = 0;
   out_4086798773631621081[20] = 0;
   out_4086798773631621081[21] = 0;
   out_4086798773631621081[22] = 0;
   out_4086798773631621081[23] = 0;
   out_4086798773631621081[24] = 0;
   out_4086798773631621081[25] = 1;
   out_4086798773631621081[26] = 0;
   out_4086798773631621081[27] = 0;
   out_4086798773631621081[28] = 1;
   out_4086798773631621081[29] = 0;
   out_4086798773631621081[30] = 0;
   out_4086798773631621081[31] = 0;
   out_4086798773631621081[32] = 0;
   out_4086798773631621081[33] = 0;
   out_4086798773631621081[34] = 0;
   out_4086798773631621081[35] = 0;
   out_4086798773631621081[36] = 0;
   out_4086798773631621081[37] = 0;
   out_4086798773631621081[38] = 0;
   out_4086798773631621081[39] = 0;
   out_4086798773631621081[40] = 0;
   out_4086798773631621081[41] = 0;
   out_4086798773631621081[42] = 0;
   out_4086798773631621081[43] = 0;
   out_4086798773631621081[44] = 1;
   out_4086798773631621081[45] = 0;
   out_4086798773631621081[46] = 0;
   out_4086798773631621081[47] = 1;
   out_4086798773631621081[48] = 0;
   out_4086798773631621081[49] = 0;
   out_4086798773631621081[50] = 0;
   out_4086798773631621081[51] = 0;
   out_4086798773631621081[52] = 0;
   out_4086798773631621081[53] = 0;
}
void h_10(double *state, double *unused, double *out_1712914576874124643) {
   out_1712914576874124643[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_1712914576874124643[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_1712914576874124643[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_1959122121965491232) {
   out_1959122121965491232[0] = 0;
   out_1959122121965491232[1] = 9.8100000000000005*cos(state[1]);
   out_1959122121965491232[2] = 0;
   out_1959122121965491232[3] = 0;
   out_1959122121965491232[4] = -state[8];
   out_1959122121965491232[5] = state[7];
   out_1959122121965491232[6] = 0;
   out_1959122121965491232[7] = state[5];
   out_1959122121965491232[8] = -state[4];
   out_1959122121965491232[9] = 0;
   out_1959122121965491232[10] = 0;
   out_1959122121965491232[11] = 0;
   out_1959122121965491232[12] = 1;
   out_1959122121965491232[13] = 0;
   out_1959122121965491232[14] = 0;
   out_1959122121965491232[15] = 1;
   out_1959122121965491232[16] = 0;
   out_1959122121965491232[17] = 0;
   out_1959122121965491232[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_1959122121965491232[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_1959122121965491232[20] = 0;
   out_1959122121965491232[21] = state[8];
   out_1959122121965491232[22] = 0;
   out_1959122121965491232[23] = -state[6];
   out_1959122121965491232[24] = -state[5];
   out_1959122121965491232[25] = 0;
   out_1959122121965491232[26] = state[3];
   out_1959122121965491232[27] = 0;
   out_1959122121965491232[28] = 0;
   out_1959122121965491232[29] = 0;
   out_1959122121965491232[30] = 0;
   out_1959122121965491232[31] = 1;
   out_1959122121965491232[32] = 0;
   out_1959122121965491232[33] = 0;
   out_1959122121965491232[34] = 1;
   out_1959122121965491232[35] = 0;
   out_1959122121965491232[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_1959122121965491232[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_1959122121965491232[38] = 0;
   out_1959122121965491232[39] = -state[7];
   out_1959122121965491232[40] = state[6];
   out_1959122121965491232[41] = 0;
   out_1959122121965491232[42] = state[4];
   out_1959122121965491232[43] = -state[3];
   out_1959122121965491232[44] = 0;
   out_1959122121965491232[45] = 0;
   out_1959122121965491232[46] = 0;
   out_1959122121965491232[47] = 0;
   out_1959122121965491232[48] = 0;
   out_1959122121965491232[49] = 0;
   out_1959122121965491232[50] = 1;
   out_1959122121965491232[51] = 0;
   out_1959122121965491232[52] = 0;
   out_1959122121965491232[53] = 1;
}
void h_13(double *state, double *unused, double *out_3468100881563169718) {
   out_3468100881563169718[0] = state[3];
   out_3468100881563169718[1] = state[4];
   out_3468100881563169718[2] = state[5];
}
void H_13(double *state, double *unused, double *out_6749314091761229606) {
   out_6749314091761229606[0] = 0;
   out_6749314091761229606[1] = 0;
   out_6749314091761229606[2] = 0;
   out_6749314091761229606[3] = 1;
   out_6749314091761229606[4] = 0;
   out_6749314091761229606[5] = 0;
   out_6749314091761229606[6] = 0;
   out_6749314091761229606[7] = 0;
   out_6749314091761229606[8] = 0;
   out_6749314091761229606[9] = 0;
   out_6749314091761229606[10] = 0;
   out_6749314091761229606[11] = 0;
   out_6749314091761229606[12] = 0;
   out_6749314091761229606[13] = 0;
   out_6749314091761229606[14] = 0;
   out_6749314091761229606[15] = 0;
   out_6749314091761229606[16] = 0;
   out_6749314091761229606[17] = 0;
   out_6749314091761229606[18] = 0;
   out_6749314091761229606[19] = 0;
   out_6749314091761229606[20] = 0;
   out_6749314091761229606[21] = 0;
   out_6749314091761229606[22] = 1;
   out_6749314091761229606[23] = 0;
   out_6749314091761229606[24] = 0;
   out_6749314091761229606[25] = 0;
   out_6749314091761229606[26] = 0;
   out_6749314091761229606[27] = 0;
   out_6749314091761229606[28] = 0;
   out_6749314091761229606[29] = 0;
   out_6749314091761229606[30] = 0;
   out_6749314091761229606[31] = 0;
   out_6749314091761229606[32] = 0;
   out_6749314091761229606[33] = 0;
   out_6749314091761229606[34] = 0;
   out_6749314091761229606[35] = 0;
   out_6749314091761229606[36] = 0;
   out_6749314091761229606[37] = 0;
   out_6749314091761229606[38] = 0;
   out_6749314091761229606[39] = 0;
   out_6749314091761229606[40] = 0;
   out_6749314091761229606[41] = 1;
   out_6749314091761229606[42] = 0;
   out_6749314091761229606[43] = 0;
   out_6749314091761229606[44] = 0;
   out_6749314091761229606[45] = 0;
   out_6749314091761229606[46] = 0;
   out_6749314091761229606[47] = 0;
   out_6749314091761229606[48] = 0;
   out_6749314091761229606[49] = 0;
   out_6749314091761229606[50] = 0;
   out_6749314091761229606[51] = 0;
   out_6749314091761229606[52] = 0;
   out_6749314091761229606[53] = 0;
}
void h_14(double *state, double *unused, double *out_3148208993463278653) {
   out_3148208993463278653[0] = state[6];
   out_3148208993463278653[1] = state[7];
   out_3148208993463278653[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1004010341336248785) {
   out_1004010341336248785[0] = 0;
   out_1004010341336248785[1] = 0;
   out_1004010341336248785[2] = 0;
   out_1004010341336248785[3] = 0;
   out_1004010341336248785[4] = 0;
   out_1004010341336248785[5] = 0;
   out_1004010341336248785[6] = 1;
   out_1004010341336248785[7] = 0;
   out_1004010341336248785[8] = 0;
   out_1004010341336248785[9] = 0;
   out_1004010341336248785[10] = 0;
   out_1004010341336248785[11] = 0;
   out_1004010341336248785[12] = 0;
   out_1004010341336248785[13] = 0;
   out_1004010341336248785[14] = 0;
   out_1004010341336248785[15] = 0;
   out_1004010341336248785[16] = 0;
   out_1004010341336248785[17] = 0;
   out_1004010341336248785[18] = 0;
   out_1004010341336248785[19] = 0;
   out_1004010341336248785[20] = 0;
   out_1004010341336248785[21] = 0;
   out_1004010341336248785[22] = 0;
   out_1004010341336248785[23] = 0;
   out_1004010341336248785[24] = 0;
   out_1004010341336248785[25] = 1;
   out_1004010341336248785[26] = 0;
   out_1004010341336248785[27] = 0;
   out_1004010341336248785[28] = 0;
   out_1004010341336248785[29] = 0;
   out_1004010341336248785[30] = 0;
   out_1004010341336248785[31] = 0;
   out_1004010341336248785[32] = 0;
   out_1004010341336248785[33] = 0;
   out_1004010341336248785[34] = 0;
   out_1004010341336248785[35] = 0;
   out_1004010341336248785[36] = 0;
   out_1004010341336248785[37] = 0;
   out_1004010341336248785[38] = 0;
   out_1004010341336248785[39] = 0;
   out_1004010341336248785[40] = 0;
   out_1004010341336248785[41] = 0;
   out_1004010341336248785[42] = 0;
   out_1004010341336248785[43] = 0;
   out_1004010341336248785[44] = 1;
   out_1004010341336248785[45] = 0;
   out_1004010341336248785[46] = 0;
   out_1004010341336248785[47] = 0;
   out_1004010341336248785[48] = 0;
   out_1004010341336248785[49] = 0;
   out_1004010341336248785[50] = 0;
   out_1004010341336248785[51] = 0;
   out_1004010341336248785[52] = 0;
   out_1004010341336248785[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_2740203603399450496) {
  err_fun(nom_x, delta_x, out_2740203603399450496);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2944854366871732438) {
  inv_err_fun(nom_x, true_x, out_2944854366871732438);
}
void pose_H_mod_fun(double *state, double *out_1017226251035741065) {
  H_mod_fun(state, out_1017226251035741065);
}
void pose_f_fun(double *state, double dt, double *out_3838696256166161148) {
  f_fun(state,  dt, out_3838696256166161148);
}
void pose_F_fun(double *state, double dt, double *out_6662765533021581793) {
  F_fun(state,  dt, out_6662765533021581793);
}
void pose_h_4(double *state, double *unused, double *out_2834632828994169778) {
  h_4(state, unused, out_2834632828994169778);
}
void pose_H_4(double *state, double *unused, double *out_4086798773631621081) {
  H_4(state, unused, out_4086798773631621081);
}
void pose_h_10(double *state, double *unused, double *out_1712914576874124643) {
  h_10(state, unused, out_1712914576874124643);
}
void pose_H_10(double *state, double *unused, double *out_1959122121965491232) {
  H_10(state, unused, out_1959122121965491232);
}
void pose_h_13(double *state, double *unused, double *out_3468100881563169718) {
  h_13(state, unused, out_3468100881563169718);
}
void pose_H_13(double *state, double *unused, double *out_6749314091761229606) {
  H_13(state, unused, out_6749314091761229606);
}
void pose_h_14(double *state, double *unused, double *out_3148208993463278653) {
  h_14(state, unused, out_3148208993463278653);
}
void pose_H_14(double *state, double *unused, double *out_1004010341336248785) {
  H_14(state, unused, out_1004010341336248785);
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
