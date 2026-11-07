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
void err_fun(double *nom_x, double *delta_x, double *out_7049367403244546975) {
   out_7049367403244546975[0] = delta_x[0] + nom_x[0];
   out_7049367403244546975[1] = delta_x[1] + nom_x[1];
   out_7049367403244546975[2] = delta_x[2] + nom_x[2];
   out_7049367403244546975[3] = delta_x[3] + nom_x[3];
   out_7049367403244546975[4] = delta_x[4] + nom_x[4];
   out_7049367403244546975[5] = delta_x[5] + nom_x[5];
   out_7049367403244546975[6] = delta_x[6] + nom_x[6];
   out_7049367403244546975[7] = delta_x[7] + nom_x[7];
   out_7049367403244546975[8] = delta_x[8] + nom_x[8];
   out_7049367403244546975[9] = delta_x[9] + nom_x[9];
   out_7049367403244546975[10] = delta_x[10] + nom_x[10];
   out_7049367403244546975[11] = delta_x[11] + nom_x[11];
   out_7049367403244546975[12] = delta_x[12] + nom_x[12];
   out_7049367403244546975[13] = delta_x[13] + nom_x[13];
   out_7049367403244546975[14] = delta_x[14] + nom_x[14];
   out_7049367403244546975[15] = delta_x[15] + nom_x[15];
   out_7049367403244546975[16] = delta_x[16] + nom_x[16];
   out_7049367403244546975[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_941402359144984825) {
   out_941402359144984825[0] = -nom_x[0] + true_x[0];
   out_941402359144984825[1] = -nom_x[1] + true_x[1];
   out_941402359144984825[2] = -nom_x[2] + true_x[2];
   out_941402359144984825[3] = -nom_x[3] + true_x[3];
   out_941402359144984825[4] = -nom_x[4] + true_x[4];
   out_941402359144984825[5] = -nom_x[5] + true_x[5];
   out_941402359144984825[6] = -nom_x[6] + true_x[6];
   out_941402359144984825[7] = -nom_x[7] + true_x[7];
   out_941402359144984825[8] = -nom_x[8] + true_x[8];
   out_941402359144984825[9] = -nom_x[9] + true_x[9];
   out_941402359144984825[10] = -nom_x[10] + true_x[10];
   out_941402359144984825[11] = -nom_x[11] + true_x[11];
   out_941402359144984825[12] = -nom_x[12] + true_x[12];
   out_941402359144984825[13] = -nom_x[13] + true_x[13];
   out_941402359144984825[14] = -nom_x[14] + true_x[14];
   out_941402359144984825[15] = -nom_x[15] + true_x[15];
   out_941402359144984825[16] = -nom_x[16] + true_x[16];
   out_941402359144984825[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_4100613614008638725) {
   out_4100613614008638725[0] = 1.0;
   out_4100613614008638725[1] = 0.0;
   out_4100613614008638725[2] = 0.0;
   out_4100613614008638725[3] = 0.0;
   out_4100613614008638725[4] = 0.0;
   out_4100613614008638725[5] = 0.0;
   out_4100613614008638725[6] = 0.0;
   out_4100613614008638725[7] = 0.0;
   out_4100613614008638725[8] = 0.0;
   out_4100613614008638725[9] = 0.0;
   out_4100613614008638725[10] = 0.0;
   out_4100613614008638725[11] = 0.0;
   out_4100613614008638725[12] = 0.0;
   out_4100613614008638725[13] = 0.0;
   out_4100613614008638725[14] = 0.0;
   out_4100613614008638725[15] = 0.0;
   out_4100613614008638725[16] = 0.0;
   out_4100613614008638725[17] = 0.0;
   out_4100613614008638725[18] = 0.0;
   out_4100613614008638725[19] = 1.0;
   out_4100613614008638725[20] = 0.0;
   out_4100613614008638725[21] = 0.0;
   out_4100613614008638725[22] = 0.0;
   out_4100613614008638725[23] = 0.0;
   out_4100613614008638725[24] = 0.0;
   out_4100613614008638725[25] = 0.0;
   out_4100613614008638725[26] = 0.0;
   out_4100613614008638725[27] = 0.0;
   out_4100613614008638725[28] = 0.0;
   out_4100613614008638725[29] = 0.0;
   out_4100613614008638725[30] = 0.0;
   out_4100613614008638725[31] = 0.0;
   out_4100613614008638725[32] = 0.0;
   out_4100613614008638725[33] = 0.0;
   out_4100613614008638725[34] = 0.0;
   out_4100613614008638725[35] = 0.0;
   out_4100613614008638725[36] = 0.0;
   out_4100613614008638725[37] = 0.0;
   out_4100613614008638725[38] = 1.0;
   out_4100613614008638725[39] = 0.0;
   out_4100613614008638725[40] = 0.0;
   out_4100613614008638725[41] = 0.0;
   out_4100613614008638725[42] = 0.0;
   out_4100613614008638725[43] = 0.0;
   out_4100613614008638725[44] = 0.0;
   out_4100613614008638725[45] = 0.0;
   out_4100613614008638725[46] = 0.0;
   out_4100613614008638725[47] = 0.0;
   out_4100613614008638725[48] = 0.0;
   out_4100613614008638725[49] = 0.0;
   out_4100613614008638725[50] = 0.0;
   out_4100613614008638725[51] = 0.0;
   out_4100613614008638725[52] = 0.0;
   out_4100613614008638725[53] = 0.0;
   out_4100613614008638725[54] = 0.0;
   out_4100613614008638725[55] = 0.0;
   out_4100613614008638725[56] = 0.0;
   out_4100613614008638725[57] = 1.0;
   out_4100613614008638725[58] = 0.0;
   out_4100613614008638725[59] = 0.0;
   out_4100613614008638725[60] = 0.0;
   out_4100613614008638725[61] = 0.0;
   out_4100613614008638725[62] = 0.0;
   out_4100613614008638725[63] = 0.0;
   out_4100613614008638725[64] = 0.0;
   out_4100613614008638725[65] = 0.0;
   out_4100613614008638725[66] = 0.0;
   out_4100613614008638725[67] = 0.0;
   out_4100613614008638725[68] = 0.0;
   out_4100613614008638725[69] = 0.0;
   out_4100613614008638725[70] = 0.0;
   out_4100613614008638725[71] = 0.0;
   out_4100613614008638725[72] = 0.0;
   out_4100613614008638725[73] = 0.0;
   out_4100613614008638725[74] = 0.0;
   out_4100613614008638725[75] = 0.0;
   out_4100613614008638725[76] = 1.0;
   out_4100613614008638725[77] = 0.0;
   out_4100613614008638725[78] = 0.0;
   out_4100613614008638725[79] = 0.0;
   out_4100613614008638725[80] = 0.0;
   out_4100613614008638725[81] = 0.0;
   out_4100613614008638725[82] = 0.0;
   out_4100613614008638725[83] = 0.0;
   out_4100613614008638725[84] = 0.0;
   out_4100613614008638725[85] = 0.0;
   out_4100613614008638725[86] = 0.0;
   out_4100613614008638725[87] = 0.0;
   out_4100613614008638725[88] = 0.0;
   out_4100613614008638725[89] = 0.0;
   out_4100613614008638725[90] = 0.0;
   out_4100613614008638725[91] = 0.0;
   out_4100613614008638725[92] = 0.0;
   out_4100613614008638725[93] = 0.0;
   out_4100613614008638725[94] = 0.0;
   out_4100613614008638725[95] = 1.0;
   out_4100613614008638725[96] = 0.0;
   out_4100613614008638725[97] = 0.0;
   out_4100613614008638725[98] = 0.0;
   out_4100613614008638725[99] = 0.0;
   out_4100613614008638725[100] = 0.0;
   out_4100613614008638725[101] = 0.0;
   out_4100613614008638725[102] = 0.0;
   out_4100613614008638725[103] = 0.0;
   out_4100613614008638725[104] = 0.0;
   out_4100613614008638725[105] = 0.0;
   out_4100613614008638725[106] = 0.0;
   out_4100613614008638725[107] = 0.0;
   out_4100613614008638725[108] = 0.0;
   out_4100613614008638725[109] = 0.0;
   out_4100613614008638725[110] = 0.0;
   out_4100613614008638725[111] = 0.0;
   out_4100613614008638725[112] = 0.0;
   out_4100613614008638725[113] = 0.0;
   out_4100613614008638725[114] = 1.0;
   out_4100613614008638725[115] = 0.0;
   out_4100613614008638725[116] = 0.0;
   out_4100613614008638725[117] = 0.0;
   out_4100613614008638725[118] = 0.0;
   out_4100613614008638725[119] = 0.0;
   out_4100613614008638725[120] = 0.0;
   out_4100613614008638725[121] = 0.0;
   out_4100613614008638725[122] = 0.0;
   out_4100613614008638725[123] = 0.0;
   out_4100613614008638725[124] = 0.0;
   out_4100613614008638725[125] = 0.0;
   out_4100613614008638725[126] = 0.0;
   out_4100613614008638725[127] = 0.0;
   out_4100613614008638725[128] = 0.0;
   out_4100613614008638725[129] = 0.0;
   out_4100613614008638725[130] = 0.0;
   out_4100613614008638725[131] = 0.0;
   out_4100613614008638725[132] = 0.0;
   out_4100613614008638725[133] = 1.0;
   out_4100613614008638725[134] = 0.0;
   out_4100613614008638725[135] = 0.0;
   out_4100613614008638725[136] = 0.0;
   out_4100613614008638725[137] = 0.0;
   out_4100613614008638725[138] = 0.0;
   out_4100613614008638725[139] = 0.0;
   out_4100613614008638725[140] = 0.0;
   out_4100613614008638725[141] = 0.0;
   out_4100613614008638725[142] = 0.0;
   out_4100613614008638725[143] = 0.0;
   out_4100613614008638725[144] = 0.0;
   out_4100613614008638725[145] = 0.0;
   out_4100613614008638725[146] = 0.0;
   out_4100613614008638725[147] = 0.0;
   out_4100613614008638725[148] = 0.0;
   out_4100613614008638725[149] = 0.0;
   out_4100613614008638725[150] = 0.0;
   out_4100613614008638725[151] = 0.0;
   out_4100613614008638725[152] = 1.0;
   out_4100613614008638725[153] = 0.0;
   out_4100613614008638725[154] = 0.0;
   out_4100613614008638725[155] = 0.0;
   out_4100613614008638725[156] = 0.0;
   out_4100613614008638725[157] = 0.0;
   out_4100613614008638725[158] = 0.0;
   out_4100613614008638725[159] = 0.0;
   out_4100613614008638725[160] = 0.0;
   out_4100613614008638725[161] = 0.0;
   out_4100613614008638725[162] = 0.0;
   out_4100613614008638725[163] = 0.0;
   out_4100613614008638725[164] = 0.0;
   out_4100613614008638725[165] = 0.0;
   out_4100613614008638725[166] = 0.0;
   out_4100613614008638725[167] = 0.0;
   out_4100613614008638725[168] = 0.0;
   out_4100613614008638725[169] = 0.0;
   out_4100613614008638725[170] = 0.0;
   out_4100613614008638725[171] = 1.0;
   out_4100613614008638725[172] = 0.0;
   out_4100613614008638725[173] = 0.0;
   out_4100613614008638725[174] = 0.0;
   out_4100613614008638725[175] = 0.0;
   out_4100613614008638725[176] = 0.0;
   out_4100613614008638725[177] = 0.0;
   out_4100613614008638725[178] = 0.0;
   out_4100613614008638725[179] = 0.0;
   out_4100613614008638725[180] = 0.0;
   out_4100613614008638725[181] = 0.0;
   out_4100613614008638725[182] = 0.0;
   out_4100613614008638725[183] = 0.0;
   out_4100613614008638725[184] = 0.0;
   out_4100613614008638725[185] = 0.0;
   out_4100613614008638725[186] = 0.0;
   out_4100613614008638725[187] = 0.0;
   out_4100613614008638725[188] = 0.0;
   out_4100613614008638725[189] = 0.0;
   out_4100613614008638725[190] = 1.0;
   out_4100613614008638725[191] = 0.0;
   out_4100613614008638725[192] = 0.0;
   out_4100613614008638725[193] = 0.0;
   out_4100613614008638725[194] = 0.0;
   out_4100613614008638725[195] = 0.0;
   out_4100613614008638725[196] = 0.0;
   out_4100613614008638725[197] = 0.0;
   out_4100613614008638725[198] = 0.0;
   out_4100613614008638725[199] = 0.0;
   out_4100613614008638725[200] = 0.0;
   out_4100613614008638725[201] = 0.0;
   out_4100613614008638725[202] = 0.0;
   out_4100613614008638725[203] = 0.0;
   out_4100613614008638725[204] = 0.0;
   out_4100613614008638725[205] = 0.0;
   out_4100613614008638725[206] = 0.0;
   out_4100613614008638725[207] = 0.0;
   out_4100613614008638725[208] = 0.0;
   out_4100613614008638725[209] = 1.0;
   out_4100613614008638725[210] = 0.0;
   out_4100613614008638725[211] = 0.0;
   out_4100613614008638725[212] = 0.0;
   out_4100613614008638725[213] = 0.0;
   out_4100613614008638725[214] = 0.0;
   out_4100613614008638725[215] = 0.0;
   out_4100613614008638725[216] = 0.0;
   out_4100613614008638725[217] = 0.0;
   out_4100613614008638725[218] = 0.0;
   out_4100613614008638725[219] = 0.0;
   out_4100613614008638725[220] = 0.0;
   out_4100613614008638725[221] = 0.0;
   out_4100613614008638725[222] = 0.0;
   out_4100613614008638725[223] = 0.0;
   out_4100613614008638725[224] = 0.0;
   out_4100613614008638725[225] = 0.0;
   out_4100613614008638725[226] = 0.0;
   out_4100613614008638725[227] = 0.0;
   out_4100613614008638725[228] = 1.0;
   out_4100613614008638725[229] = 0.0;
   out_4100613614008638725[230] = 0.0;
   out_4100613614008638725[231] = 0.0;
   out_4100613614008638725[232] = 0.0;
   out_4100613614008638725[233] = 0.0;
   out_4100613614008638725[234] = 0.0;
   out_4100613614008638725[235] = 0.0;
   out_4100613614008638725[236] = 0.0;
   out_4100613614008638725[237] = 0.0;
   out_4100613614008638725[238] = 0.0;
   out_4100613614008638725[239] = 0.0;
   out_4100613614008638725[240] = 0.0;
   out_4100613614008638725[241] = 0.0;
   out_4100613614008638725[242] = 0.0;
   out_4100613614008638725[243] = 0.0;
   out_4100613614008638725[244] = 0.0;
   out_4100613614008638725[245] = 0.0;
   out_4100613614008638725[246] = 0.0;
   out_4100613614008638725[247] = 1.0;
   out_4100613614008638725[248] = 0.0;
   out_4100613614008638725[249] = 0.0;
   out_4100613614008638725[250] = 0.0;
   out_4100613614008638725[251] = 0.0;
   out_4100613614008638725[252] = 0.0;
   out_4100613614008638725[253] = 0.0;
   out_4100613614008638725[254] = 0.0;
   out_4100613614008638725[255] = 0.0;
   out_4100613614008638725[256] = 0.0;
   out_4100613614008638725[257] = 0.0;
   out_4100613614008638725[258] = 0.0;
   out_4100613614008638725[259] = 0.0;
   out_4100613614008638725[260] = 0.0;
   out_4100613614008638725[261] = 0.0;
   out_4100613614008638725[262] = 0.0;
   out_4100613614008638725[263] = 0.0;
   out_4100613614008638725[264] = 0.0;
   out_4100613614008638725[265] = 0.0;
   out_4100613614008638725[266] = 1.0;
   out_4100613614008638725[267] = 0.0;
   out_4100613614008638725[268] = 0.0;
   out_4100613614008638725[269] = 0.0;
   out_4100613614008638725[270] = 0.0;
   out_4100613614008638725[271] = 0.0;
   out_4100613614008638725[272] = 0.0;
   out_4100613614008638725[273] = 0.0;
   out_4100613614008638725[274] = 0.0;
   out_4100613614008638725[275] = 0.0;
   out_4100613614008638725[276] = 0.0;
   out_4100613614008638725[277] = 0.0;
   out_4100613614008638725[278] = 0.0;
   out_4100613614008638725[279] = 0.0;
   out_4100613614008638725[280] = 0.0;
   out_4100613614008638725[281] = 0.0;
   out_4100613614008638725[282] = 0.0;
   out_4100613614008638725[283] = 0.0;
   out_4100613614008638725[284] = 0.0;
   out_4100613614008638725[285] = 1.0;
   out_4100613614008638725[286] = 0.0;
   out_4100613614008638725[287] = 0.0;
   out_4100613614008638725[288] = 0.0;
   out_4100613614008638725[289] = 0.0;
   out_4100613614008638725[290] = 0.0;
   out_4100613614008638725[291] = 0.0;
   out_4100613614008638725[292] = 0.0;
   out_4100613614008638725[293] = 0.0;
   out_4100613614008638725[294] = 0.0;
   out_4100613614008638725[295] = 0.0;
   out_4100613614008638725[296] = 0.0;
   out_4100613614008638725[297] = 0.0;
   out_4100613614008638725[298] = 0.0;
   out_4100613614008638725[299] = 0.0;
   out_4100613614008638725[300] = 0.0;
   out_4100613614008638725[301] = 0.0;
   out_4100613614008638725[302] = 0.0;
   out_4100613614008638725[303] = 0.0;
   out_4100613614008638725[304] = 1.0;
   out_4100613614008638725[305] = 0.0;
   out_4100613614008638725[306] = 0.0;
   out_4100613614008638725[307] = 0.0;
   out_4100613614008638725[308] = 0.0;
   out_4100613614008638725[309] = 0.0;
   out_4100613614008638725[310] = 0.0;
   out_4100613614008638725[311] = 0.0;
   out_4100613614008638725[312] = 0.0;
   out_4100613614008638725[313] = 0.0;
   out_4100613614008638725[314] = 0.0;
   out_4100613614008638725[315] = 0.0;
   out_4100613614008638725[316] = 0.0;
   out_4100613614008638725[317] = 0.0;
   out_4100613614008638725[318] = 0.0;
   out_4100613614008638725[319] = 0.0;
   out_4100613614008638725[320] = 0.0;
   out_4100613614008638725[321] = 0.0;
   out_4100613614008638725[322] = 0.0;
   out_4100613614008638725[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8397176677496298785) {
   out_8397176677496298785[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8397176677496298785[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8397176677496298785[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8397176677496298785[3] = dt*state[12] + state[3];
   out_8397176677496298785[4] = dt*state[13] + state[4];
   out_8397176677496298785[5] = dt*state[14] + state[5];
   out_8397176677496298785[6] = state[6];
   out_8397176677496298785[7] = state[7];
   out_8397176677496298785[8] = state[8];
   out_8397176677496298785[9] = state[9];
   out_8397176677496298785[10] = state[10];
   out_8397176677496298785[11] = state[11];
   out_8397176677496298785[12] = state[12];
   out_8397176677496298785[13] = state[13];
   out_8397176677496298785[14] = state[14];
   out_8397176677496298785[15] = state[15];
   out_8397176677496298785[16] = state[16];
   out_8397176677496298785[17] = state[17];
}
void F_fun(double *state, double dt, double *out_3499620883405643949) {
   out_3499620883405643949[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3499620883405643949[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3499620883405643949[2] = 0;
   out_3499620883405643949[3] = 0;
   out_3499620883405643949[4] = 0;
   out_3499620883405643949[5] = 0;
   out_3499620883405643949[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3499620883405643949[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3499620883405643949[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_3499620883405643949[9] = 0;
   out_3499620883405643949[10] = 0;
   out_3499620883405643949[11] = 0;
   out_3499620883405643949[12] = 0;
   out_3499620883405643949[13] = 0;
   out_3499620883405643949[14] = 0;
   out_3499620883405643949[15] = 0;
   out_3499620883405643949[16] = 0;
   out_3499620883405643949[17] = 0;
   out_3499620883405643949[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3499620883405643949[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3499620883405643949[20] = 0;
   out_3499620883405643949[21] = 0;
   out_3499620883405643949[22] = 0;
   out_3499620883405643949[23] = 0;
   out_3499620883405643949[24] = 0;
   out_3499620883405643949[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3499620883405643949[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_3499620883405643949[27] = 0;
   out_3499620883405643949[28] = 0;
   out_3499620883405643949[29] = 0;
   out_3499620883405643949[30] = 0;
   out_3499620883405643949[31] = 0;
   out_3499620883405643949[32] = 0;
   out_3499620883405643949[33] = 0;
   out_3499620883405643949[34] = 0;
   out_3499620883405643949[35] = 0;
   out_3499620883405643949[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3499620883405643949[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3499620883405643949[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3499620883405643949[39] = 0;
   out_3499620883405643949[40] = 0;
   out_3499620883405643949[41] = 0;
   out_3499620883405643949[42] = 0;
   out_3499620883405643949[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3499620883405643949[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_3499620883405643949[45] = 0;
   out_3499620883405643949[46] = 0;
   out_3499620883405643949[47] = 0;
   out_3499620883405643949[48] = 0;
   out_3499620883405643949[49] = 0;
   out_3499620883405643949[50] = 0;
   out_3499620883405643949[51] = 0;
   out_3499620883405643949[52] = 0;
   out_3499620883405643949[53] = 0;
   out_3499620883405643949[54] = 0;
   out_3499620883405643949[55] = 0;
   out_3499620883405643949[56] = 0;
   out_3499620883405643949[57] = 1;
   out_3499620883405643949[58] = 0;
   out_3499620883405643949[59] = 0;
   out_3499620883405643949[60] = 0;
   out_3499620883405643949[61] = 0;
   out_3499620883405643949[62] = 0;
   out_3499620883405643949[63] = 0;
   out_3499620883405643949[64] = 0;
   out_3499620883405643949[65] = 0;
   out_3499620883405643949[66] = dt;
   out_3499620883405643949[67] = 0;
   out_3499620883405643949[68] = 0;
   out_3499620883405643949[69] = 0;
   out_3499620883405643949[70] = 0;
   out_3499620883405643949[71] = 0;
   out_3499620883405643949[72] = 0;
   out_3499620883405643949[73] = 0;
   out_3499620883405643949[74] = 0;
   out_3499620883405643949[75] = 0;
   out_3499620883405643949[76] = 1;
   out_3499620883405643949[77] = 0;
   out_3499620883405643949[78] = 0;
   out_3499620883405643949[79] = 0;
   out_3499620883405643949[80] = 0;
   out_3499620883405643949[81] = 0;
   out_3499620883405643949[82] = 0;
   out_3499620883405643949[83] = 0;
   out_3499620883405643949[84] = 0;
   out_3499620883405643949[85] = dt;
   out_3499620883405643949[86] = 0;
   out_3499620883405643949[87] = 0;
   out_3499620883405643949[88] = 0;
   out_3499620883405643949[89] = 0;
   out_3499620883405643949[90] = 0;
   out_3499620883405643949[91] = 0;
   out_3499620883405643949[92] = 0;
   out_3499620883405643949[93] = 0;
   out_3499620883405643949[94] = 0;
   out_3499620883405643949[95] = 1;
   out_3499620883405643949[96] = 0;
   out_3499620883405643949[97] = 0;
   out_3499620883405643949[98] = 0;
   out_3499620883405643949[99] = 0;
   out_3499620883405643949[100] = 0;
   out_3499620883405643949[101] = 0;
   out_3499620883405643949[102] = 0;
   out_3499620883405643949[103] = 0;
   out_3499620883405643949[104] = dt;
   out_3499620883405643949[105] = 0;
   out_3499620883405643949[106] = 0;
   out_3499620883405643949[107] = 0;
   out_3499620883405643949[108] = 0;
   out_3499620883405643949[109] = 0;
   out_3499620883405643949[110] = 0;
   out_3499620883405643949[111] = 0;
   out_3499620883405643949[112] = 0;
   out_3499620883405643949[113] = 0;
   out_3499620883405643949[114] = 1;
   out_3499620883405643949[115] = 0;
   out_3499620883405643949[116] = 0;
   out_3499620883405643949[117] = 0;
   out_3499620883405643949[118] = 0;
   out_3499620883405643949[119] = 0;
   out_3499620883405643949[120] = 0;
   out_3499620883405643949[121] = 0;
   out_3499620883405643949[122] = 0;
   out_3499620883405643949[123] = 0;
   out_3499620883405643949[124] = 0;
   out_3499620883405643949[125] = 0;
   out_3499620883405643949[126] = 0;
   out_3499620883405643949[127] = 0;
   out_3499620883405643949[128] = 0;
   out_3499620883405643949[129] = 0;
   out_3499620883405643949[130] = 0;
   out_3499620883405643949[131] = 0;
   out_3499620883405643949[132] = 0;
   out_3499620883405643949[133] = 1;
   out_3499620883405643949[134] = 0;
   out_3499620883405643949[135] = 0;
   out_3499620883405643949[136] = 0;
   out_3499620883405643949[137] = 0;
   out_3499620883405643949[138] = 0;
   out_3499620883405643949[139] = 0;
   out_3499620883405643949[140] = 0;
   out_3499620883405643949[141] = 0;
   out_3499620883405643949[142] = 0;
   out_3499620883405643949[143] = 0;
   out_3499620883405643949[144] = 0;
   out_3499620883405643949[145] = 0;
   out_3499620883405643949[146] = 0;
   out_3499620883405643949[147] = 0;
   out_3499620883405643949[148] = 0;
   out_3499620883405643949[149] = 0;
   out_3499620883405643949[150] = 0;
   out_3499620883405643949[151] = 0;
   out_3499620883405643949[152] = 1;
   out_3499620883405643949[153] = 0;
   out_3499620883405643949[154] = 0;
   out_3499620883405643949[155] = 0;
   out_3499620883405643949[156] = 0;
   out_3499620883405643949[157] = 0;
   out_3499620883405643949[158] = 0;
   out_3499620883405643949[159] = 0;
   out_3499620883405643949[160] = 0;
   out_3499620883405643949[161] = 0;
   out_3499620883405643949[162] = 0;
   out_3499620883405643949[163] = 0;
   out_3499620883405643949[164] = 0;
   out_3499620883405643949[165] = 0;
   out_3499620883405643949[166] = 0;
   out_3499620883405643949[167] = 0;
   out_3499620883405643949[168] = 0;
   out_3499620883405643949[169] = 0;
   out_3499620883405643949[170] = 0;
   out_3499620883405643949[171] = 1;
   out_3499620883405643949[172] = 0;
   out_3499620883405643949[173] = 0;
   out_3499620883405643949[174] = 0;
   out_3499620883405643949[175] = 0;
   out_3499620883405643949[176] = 0;
   out_3499620883405643949[177] = 0;
   out_3499620883405643949[178] = 0;
   out_3499620883405643949[179] = 0;
   out_3499620883405643949[180] = 0;
   out_3499620883405643949[181] = 0;
   out_3499620883405643949[182] = 0;
   out_3499620883405643949[183] = 0;
   out_3499620883405643949[184] = 0;
   out_3499620883405643949[185] = 0;
   out_3499620883405643949[186] = 0;
   out_3499620883405643949[187] = 0;
   out_3499620883405643949[188] = 0;
   out_3499620883405643949[189] = 0;
   out_3499620883405643949[190] = 1;
   out_3499620883405643949[191] = 0;
   out_3499620883405643949[192] = 0;
   out_3499620883405643949[193] = 0;
   out_3499620883405643949[194] = 0;
   out_3499620883405643949[195] = 0;
   out_3499620883405643949[196] = 0;
   out_3499620883405643949[197] = 0;
   out_3499620883405643949[198] = 0;
   out_3499620883405643949[199] = 0;
   out_3499620883405643949[200] = 0;
   out_3499620883405643949[201] = 0;
   out_3499620883405643949[202] = 0;
   out_3499620883405643949[203] = 0;
   out_3499620883405643949[204] = 0;
   out_3499620883405643949[205] = 0;
   out_3499620883405643949[206] = 0;
   out_3499620883405643949[207] = 0;
   out_3499620883405643949[208] = 0;
   out_3499620883405643949[209] = 1;
   out_3499620883405643949[210] = 0;
   out_3499620883405643949[211] = 0;
   out_3499620883405643949[212] = 0;
   out_3499620883405643949[213] = 0;
   out_3499620883405643949[214] = 0;
   out_3499620883405643949[215] = 0;
   out_3499620883405643949[216] = 0;
   out_3499620883405643949[217] = 0;
   out_3499620883405643949[218] = 0;
   out_3499620883405643949[219] = 0;
   out_3499620883405643949[220] = 0;
   out_3499620883405643949[221] = 0;
   out_3499620883405643949[222] = 0;
   out_3499620883405643949[223] = 0;
   out_3499620883405643949[224] = 0;
   out_3499620883405643949[225] = 0;
   out_3499620883405643949[226] = 0;
   out_3499620883405643949[227] = 0;
   out_3499620883405643949[228] = 1;
   out_3499620883405643949[229] = 0;
   out_3499620883405643949[230] = 0;
   out_3499620883405643949[231] = 0;
   out_3499620883405643949[232] = 0;
   out_3499620883405643949[233] = 0;
   out_3499620883405643949[234] = 0;
   out_3499620883405643949[235] = 0;
   out_3499620883405643949[236] = 0;
   out_3499620883405643949[237] = 0;
   out_3499620883405643949[238] = 0;
   out_3499620883405643949[239] = 0;
   out_3499620883405643949[240] = 0;
   out_3499620883405643949[241] = 0;
   out_3499620883405643949[242] = 0;
   out_3499620883405643949[243] = 0;
   out_3499620883405643949[244] = 0;
   out_3499620883405643949[245] = 0;
   out_3499620883405643949[246] = 0;
   out_3499620883405643949[247] = 1;
   out_3499620883405643949[248] = 0;
   out_3499620883405643949[249] = 0;
   out_3499620883405643949[250] = 0;
   out_3499620883405643949[251] = 0;
   out_3499620883405643949[252] = 0;
   out_3499620883405643949[253] = 0;
   out_3499620883405643949[254] = 0;
   out_3499620883405643949[255] = 0;
   out_3499620883405643949[256] = 0;
   out_3499620883405643949[257] = 0;
   out_3499620883405643949[258] = 0;
   out_3499620883405643949[259] = 0;
   out_3499620883405643949[260] = 0;
   out_3499620883405643949[261] = 0;
   out_3499620883405643949[262] = 0;
   out_3499620883405643949[263] = 0;
   out_3499620883405643949[264] = 0;
   out_3499620883405643949[265] = 0;
   out_3499620883405643949[266] = 1;
   out_3499620883405643949[267] = 0;
   out_3499620883405643949[268] = 0;
   out_3499620883405643949[269] = 0;
   out_3499620883405643949[270] = 0;
   out_3499620883405643949[271] = 0;
   out_3499620883405643949[272] = 0;
   out_3499620883405643949[273] = 0;
   out_3499620883405643949[274] = 0;
   out_3499620883405643949[275] = 0;
   out_3499620883405643949[276] = 0;
   out_3499620883405643949[277] = 0;
   out_3499620883405643949[278] = 0;
   out_3499620883405643949[279] = 0;
   out_3499620883405643949[280] = 0;
   out_3499620883405643949[281] = 0;
   out_3499620883405643949[282] = 0;
   out_3499620883405643949[283] = 0;
   out_3499620883405643949[284] = 0;
   out_3499620883405643949[285] = 1;
   out_3499620883405643949[286] = 0;
   out_3499620883405643949[287] = 0;
   out_3499620883405643949[288] = 0;
   out_3499620883405643949[289] = 0;
   out_3499620883405643949[290] = 0;
   out_3499620883405643949[291] = 0;
   out_3499620883405643949[292] = 0;
   out_3499620883405643949[293] = 0;
   out_3499620883405643949[294] = 0;
   out_3499620883405643949[295] = 0;
   out_3499620883405643949[296] = 0;
   out_3499620883405643949[297] = 0;
   out_3499620883405643949[298] = 0;
   out_3499620883405643949[299] = 0;
   out_3499620883405643949[300] = 0;
   out_3499620883405643949[301] = 0;
   out_3499620883405643949[302] = 0;
   out_3499620883405643949[303] = 0;
   out_3499620883405643949[304] = 1;
   out_3499620883405643949[305] = 0;
   out_3499620883405643949[306] = 0;
   out_3499620883405643949[307] = 0;
   out_3499620883405643949[308] = 0;
   out_3499620883405643949[309] = 0;
   out_3499620883405643949[310] = 0;
   out_3499620883405643949[311] = 0;
   out_3499620883405643949[312] = 0;
   out_3499620883405643949[313] = 0;
   out_3499620883405643949[314] = 0;
   out_3499620883405643949[315] = 0;
   out_3499620883405643949[316] = 0;
   out_3499620883405643949[317] = 0;
   out_3499620883405643949[318] = 0;
   out_3499620883405643949[319] = 0;
   out_3499620883405643949[320] = 0;
   out_3499620883405643949[321] = 0;
   out_3499620883405643949[322] = 0;
   out_3499620883405643949[323] = 1;
}
void h_4(double *state, double *unused, double *out_8172190577682765870) {
   out_8172190577682765870[0] = state[6] + state[9];
   out_8172190577682765870[1] = state[7] + state[10];
   out_8172190577682765870[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_9193612074661832156) {
   out_9193612074661832156[0] = 0;
   out_9193612074661832156[1] = 0;
   out_9193612074661832156[2] = 0;
   out_9193612074661832156[3] = 0;
   out_9193612074661832156[4] = 0;
   out_9193612074661832156[5] = 0;
   out_9193612074661832156[6] = 1;
   out_9193612074661832156[7] = 0;
   out_9193612074661832156[8] = 0;
   out_9193612074661832156[9] = 1;
   out_9193612074661832156[10] = 0;
   out_9193612074661832156[11] = 0;
   out_9193612074661832156[12] = 0;
   out_9193612074661832156[13] = 0;
   out_9193612074661832156[14] = 0;
   out_9193612074661832156[15] = 0;
   out_9193612074661832156[16] = 0;
   out_9193612074661832156[17] = 0;
   out_9193612074661832156[18] = 0;
   out_9193612074661832156[19] = 0;
   out_9193612074661832156[20] = 0;
   out_9193612074661832156[21] = 0;
   out_9193612074661832156[22] = 0;
   out_9193612074661832156[23] = 0;
   out_9193612074661832156[24] = 0;
   out_9193612074661832156[25] = 1;
   out_9193612074661832156[26] = 0;
   out_9193612074661832156[27] = 0;
   out_9193612074661832156[28] = 1;
   out_9193612074661832156[29] = 0;
   out_9193612074661832156[30] = 0;
   out_9193612074661832156[31] = 0;
   out_9193612074661832156[32] = 0;
   out_9193612074661832156[33] = 0;
   out_9193612074661832156[34] = 0;
   out_9193612074661832156[35] = 0;
   out_9193612074661832156[36] = 0;
   out_9193612074661832156[37] = 0;
   out_9193612074661832156[38] = 0;
   out_9193612074661832156[39] = 0;
   out_9193612074661832156[40] = 0;
   out_9193612074661832156[41] = 0;
   out_9193612074661832156[42] = 0;
   out_9193612074661832156[43] = 0;
   out_9193612074661832156[44] = 1;
   out_9193612074661832156[45] = 0;
   out_9193612074661832156[46] = 0;
   out_9193612074661832156[47] = 1;
   out_9193612074661832156[48] = 0;
   out_9193612074661832156[49] = 0;
   out_9193612074661832156[50] = 0;
   out_9193612074661832156[51] = 0;
   out_9193612074661832156[52] = 0;
   out_9193612074661832156[53] = 0;
}
void h_10(double *state, double *unused, double *out_8347089001260123640) {
   out_8347089001260123640[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_8347089001260123640[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_8347089001260123640[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_7557781302877688021) {
   out_7557781302877688021[0] = 0;
   out_7557781302877688021[1] = 9.8100000000000005*cos(state[1]);
   out_7557781302877688021[2] = 0;
   out_7557781302877688021[3] = 0;
   out_7557781302877688021[4] = -state[8];
   out_7557781302877688021[5] = state[7];
   out_7557781302877688021[6] = 0;
   out_7557781302877688021[7] = state[5];
   out_7557781302877688021[8] = -state[4];
   out_7557781302877688021[9] = 0;
   out_7557781302877688021[10] = 0;
   out_7557781302877688021[11] = 0;
   out_7557781302877688021[12] = 1;
   out_7557781302877688021[13] = 0;
   out_7557781302877688021[14] = 0;
   out_7557781302877688021[15] = 1;
   out_7557781302877688021[16] = 0;
   out_7557781302877688021[17] = 0;
   out_7557781302877688021[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_7557781302877688021[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_7557781302877688021[20] = 0;
   out_7557781302877688021[21] = state[8];
   out_7557781302877688021[22] = 0;
   out_7557781302877688021[23] = -state[6];
   out_7557781302877688021[24] = -state[5];
   out_7557781302877688021[25] = 0;
   out_7557781302877688021[26] = state[3];
   out_7557781302877688021[27] = 0;
   out_7557781302877688021[28] = 0;
   out_7557781302877688021[29] = 0;
   out_7557781302877688021[30] = 0;
   out_7557781302877688021[31] = 1;
   out_7557781302877688021[32] = 0;
   out_7557781302877688021[33] = 0;
   out_7557781302877688021[34] = 1;
   out_7557781302877688021[35] = 0;
   out_7557781302877688021[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_7557781302877688021[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_7557781302877688021[38] = 0;
   out_7557781302877688021[39] = -state[7];
   out_7557781302877688021[40] = state[6];
   out_7557781302877688021[41] = 0;
   out_7557781302877688021[42] = state[4];
   out_7557781302877688021[43] = -state[3];
   out_7557781302877688021[44] = 0;
   out_7557781302877688021[45] = 0;
   out_7557781302877688021[46] = 0;
   out_7557781302877688021[47] = 0;
   out_7557781302877688021[48] = 0;
   out_7557781302877688021[49] = 0;
   out_7557781302877688021[50] = 1;
   out_7557781302877688021[51] = 0;
   out_7557781302877688021[52] = 0;
   out_7557781302877688021[53] = 1;
}
void h_13(double *state, double *unused, double *out_8271019048962668661) {
   out_8271019048962668661[0] = state[3];
   out_8271019048962668661[1] = state[4];
   out_8271019048962668661[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5981338249329499355) {
   out_5981338249329499355[0] = 0;
   out_5981338249329499355[1] = 0;
   out_5981338249329499355[2] = 0;
   out_5981338249329499355[3] = 1;
   out_5981338249329499355[4] = 0;
   out_5981338249329499355[5] = 0;
   out_5981338249329499355[6] = 0;
   out_5981338249329499355[7] = 0;
   out_5981338249329499355[8] = 0;
   out_5981338249329499355[9] = 0;
   out_5981338249329499355[10] = 0;
   out_5981338249329499355[11] = 0;
   out_5981338249329499355[12] = 0;
   out_5981338249329499355[13] = 0;
   out_5981338249329499355[14] = 0;
   out_5981338249329499355[15] = 0;
   out_5981338249329499355[16] = 0;
   out_5981338249329499355[17] = 0;
   out_5981338249329499355[18] = 0;
   out_5981338249329499355[19] = 0;
   out_5981338249329499355[20] = 0;
   out_5981338249329499355[21] = 0;
   out_5981338249329499355[22] = 1;
   out_5981338249329499355[23] = 0;
   out_5981338249329499355[24] = 0;
   out_5981338249329499355[25] = 0;
   out_5981338249329499355[26] = 0;
   out_5981338249329499355[27] = 0;
   out_5981338249329499355[28] = 0;
   out_5981338249329499355[29] = 0;
   out_5981338249329499355[30] = 0;
   out_5981338249329499355[31] = 0;
   out_5981338249329499355[32] = 0;
   out_5981338249329499355[33] = 0;
   out_5981338249329499355[34] = 0;
   out_5981338249329499355[35] = 0;
   out_5981338249329499355[36] = 0;
   out_5981338249329499355[37] = 0;
   out_5981338249329499355[38] = 0;
   out_5981338249329499355[39] = 0;
   out_5981338249329499355[40] = 0;
   out_5981338249329499355[41] = 1;
   out_5981338249329499355[42] = 0;
   out_5981338249329499355[43] = 0;
   out_5981338249329499355[44] = 0;
   out_5981338249329499355[45] = 0;
   out_5981338249329499355[46] = 0;
   out_5981338249329499355[47] = 0;
   out_5981338249329499355[48] = 0;
   out_5981338249329499355[49] = 0;
   out_5981338249329499355[50] = 0;
   out_5981338249329499355[51] = 0;
   out_5981338249329499355[52] = 0;
   out_5981338249329499355[53] = 0;
}
void h_14(double *state, double *unused, double *out_6097499650426216806) {
   out_6097499650426216806[0] = state[6];
   out_6097499650426216806[1] = state[7];
   out_6097499650426216806[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5230371218322347627) {
   out_5230371218322347627[0] = 0;
   out_5230371218322347627[1] = 0;
   out_5230371218322347627[2] = 0;
   out_5230371218322347627[3] = 0;
   out_5230371218322347627[4] = 0;
   out_5230371218322347627[5] = 0;
   out_5230371218322347627[6] = 1;
   out_5230371218322347627[7] = 0;
   out_5230371218322347627[8] = 0;
   out_5230371218322347627[9] = 0;
   out_5230371218322347627[10] = 0;
   out_5230371218322347627[11] = 0;
   out_5230371218322347627[12] = 0;
   out_5230371218322347627[13] = 0;
   out_5230371218322347627[14] = 0;
   out_5230371218322347627[15] = 0;
   out_5230371218322347627[16] = 0;
   out_5230371218322347627[17] = 0;
   out_5230371218322347627[18] = 0;
   out_5230371218322347627[19] = 0;
   out_5230371218322347627[20] = 0;
   out_5230371218322347627[21] = 0;
   out_5230371218322347627[22] = 0;
   out_5230371218322347627[23] = 0;
   out_5230371218322347627[24] = 0;
   out_5230371218322347627[25] = 1;
   out_5230371218322347627[26] = 0;
   out_5230371218322347627[27] = 0;
   out_5230371218322347627[28] = 0;
   out_5230371218322347627[29] = 0;
   out_5230371218322347627[30] = 0;
   out_5230371218322347627[31] = 0;
   out_5230371218322347627[32] = 0;
   out_5230371218322347627[33] = 0;
   out_5230371218322347627[34] = 0;
   out_5230371218322347627[35] = 0;
   out_5230371218322347627[36] = 0;
   out_5230371218322347627[37] = 0;
   out_5230371218322347627[38] = 0;
   out_5230371218322347627[39] = 0;
   out_5230371218322347627[40] = 0;
   out_5230371218322347627[41] = 0;
   out_5230371218322347627[42] = 0;
   out_5230371218322347627[43] = 0;
   out_5230371218322347627[44] = 1;
   out_5230371218322347627[45] = 0;
   out_5230371218322347627[46] = 0;
   out_5230371218322347627[47] = 0;
   out_5230371218322347627[48] = 0;
   out_5230371218322347627[49] = 0;
   out_5230371218322347627[50] = 0;
   out_5230371218322347627[51] = 0;
   out_5230371218322347627[52] = 0;
   out_5230371218322347627[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_7049367403244546975) {
  err_fun(nom_x, delta_x, out_7049367403244546975);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_941402359144984825) {
  inv_err_fun(nom_x, true_x, out_941402359144984825);
}
void pose_H_mod_fun(double *state, double *out_4100613614008638725) {
  H_mod_fun(state, out_4100613614008638725);
}
void pose_f_fun(double *state, double dt, double *out_8397176677496298785) {
  f_fun(state,  dt, out_8397176677496298785);
}
void pose_F_fun(double *state, double dt, double *out_3499620883405643949) {
  F_fun(state,  dt, out_3499620883405643949);
}
void pose_h_4(double *state, double *unused, double *out_8172190577682765870) {
  h_4(state, unused, out_8172190577682765870);
}
void pose_H_4(double *state, double *unused, double *out_9193612074661832156) {
  H_4(state, unused, out_9193612074661832156);
}
void pose_h_10(double *state, double *unused, double *out_8347089001260123640) {
  h_10(state, unused, out_8347089001260123640);
}
void pose_H_10(double *state, double *unused, double *out_7557781302877688021) {
  H_10(state, unused, out_7557781302877688021);
}
void pose_h_13(double *state, double *unused, double *out_8271019048962668661) {
  h_13(state, unused, out_8271019048962668661);
}
void pose_H_13(double *state, double *unused, double *out_5981338249329499355) {
  H_13(state, unused, out_5981338249329499355);
}
void pose_h_14(double *state, double *unused, double *out_6097499650426216806) {
  h_14(state, unused, out_6097499650426216806);
}
void pose_H_14(double *state, double *unused, double *out_5230371218322347627) {
  H_14(state, unused, out_5230371218322347627);
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
