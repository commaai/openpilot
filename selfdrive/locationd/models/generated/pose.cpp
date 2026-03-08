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
void err_fun(double *nom_x, double *delta_x, double *out_8566306649041746244) {
   out_8566306649041746244[0] = delta_x[0] + nom_x[0];
   out_8566306649041746244[1] = delta_x[1] + nom_x[1];
   out_8566306649041746244[2] = delta_x[2] + nom_x[2];
   out_8566306649041746244[3] = delta_x[3] + nom_x[3];
   out_8566306649041746244[4] = delta_x[4] + nom_x[4];
   out_8566306649041746244[5] = delta_x[5] + nom_x[5];
   out_8566306649041746244[6] = delta_x[6] + nom_x[6];
   out_8566306649041746244[7] = delta_x[7] + nom_x[7];
   out_8566306649041746244[8] = delta_x[8] + nom_x[8];
   out_8566306649041746244[9] = delta_x[9] + nom_x[9];
   out_8566306649041746244[10] = delta_x[10] + nom_x[10];
   out_8566306649041746244[11] = delta_x[11] + nom_x[11];
   out_8566306649041746244[12] = delta_x[12] + nom_x[12];
   out_8566306649041746244[13] = delta_x[13] + nom_x[13];
   out_8566306649041746244[14] = delta_x[14] + nom_x[14];
   out_8566306649041746244[15] = delta_x[15] + nom_x[15];
   out_8566306649041746244[16] = delta_x[16] + nom_x[16];
   out_8566306649041746244[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1696404042837177676) {
   out_1696404042837177676[0] = -nom_x[0] + true_x[0];
   out_1696404042837177676[1] = -nom_x[1] + true_x[1];
   out_1696404042837177676[2] = -nom_x[2] + true_x[2];
   out_1696404042837177676[3] = -nom_x[3] + true_x[3];
   out_1696404042837177676[4] = -nom_x[4] + true_x[4];
   out_1696404042837177676[5] = -nom_x[5] + true_x[5];
   out_1696404042837177676[6] = -nom_x[6] + true_x[6];
   out_1696404042837177676[7] = -nom_x[7] + true_x[7];
   out_1696404042837177676[8] = -nom_x[8] + true_x[8];
   out_1696404042837177676[9] = -nom_x[9] + true_x[9];
   out_1696404042837177676[10] = -nom_x[10] + true_x[10];
   out_1696404042837177676[11] = -nom_x[11] + true_x[11];
   out_1696404042837177676[12] = -nom_x[12] + true_x[12];
   out_1696404042837177676[13] = -nom_x[13] + true_x[13];
   out_1696404042837177676[14] = -nom_x[14] + true_x[14];
   out_1696404042837177676[15] = -nom_x[15] + true_x[15];
   out_1696404042837177676[16] = -nom_x[16] + true_x[16];
   out_1696404042837177676[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_4823007598658938475) {
   out_4823007598658938475[0] = 1.0;
   out_4823007598658938475[1] = 0.0;
   out_4823007598658938475[2] = 0.0;
   out_4823007598658938475[3] = 0.0;
   out_4823007598658938475[4] = 0.0;
   out_4823007598658938475[5] = 0.0;
   out_4823007598658938475[6] = 0.0;
   out_4823007598658938475[7] = 0.0;
   out_4823007598658938475[8] = 0.0;
   out_4823007598658938475[9] = 0.0;
   out_4823007598658938475[10] = 0.0;
   out_4823007598658938475[11] = 0.0;
   out_4823007598658938475[12] = 0.0;
   out_4823007598658938475[13] = 0.0;
   out_4823007598658938475[14] = 0.0;
   out_4823007598658938475[15] = 0.0;
   out_4823007598658938475[16] = 0.0;
   out_4823007598658938475[17] = 0.0;
   out_4823007598658938475[18] = 0.0;
   out_4823007598658938475[19] = 1.0;
   out_4823007598658938475[20] = 0.0;
   out_4823007598658938475[21] = 0.0;
   out_4823007598658938475[22] = 0.0;
   out_4823007598658938475[23] = 0.0;
   out_4823007598658938475[24] = 0.0;
   out_4823007598658938475[25] = 0.0;
   out_4823007598658938475[26] = 0.0;
   out_4823007598658938475[27] = 0.0;
   out_4823007598658938475[28] = 0.0;
   out_4823007598658938475[29] = 0.0;
   out_4823007598658938475[30] = 0.0;
   out_4823007598658938475[31] = 0.0;
   out_4823007598658938475[32] = 0.0;
   out_4823007598658938475[33] = 0.0;
   out_4823007598658938475[34] = 0.0;
   out_4823007598658938475[35] = 0.0;
   out_4823007598658938475[36] = 0.0;
   out_4823007598658938475[37] = 0.0;
   out_4823007598658938475[38] = 1.0;
   out_4823007598658938475[39] = 0.0;
   out_4823007598658938475[40] = 0.0;
   out_4823007598658938475[41] = 0.0;
   out_4823007598658938475[42] = 0.0;
   out_4823007598658938475[43] = 0.0;
   out_4823007598658938475[44] = 0.0;
   out_4823007598658938475[45] = 0.0;
   out_4823007598658938475[46] = 0.0;
   out_4823007598658938475[47] = 0.0;
   out_4823007598658938475[48] = 0.0;
   out_4823007598658938475[49] = 0.0;
   out_4823007598658938475[50] = 0.0;
   out_4823007598658938475[51] = 0.0;
   out_4823007598658938475[52] = 0.0;
   out_4823007598658938475[53] = 0.0;
   out_4823007598658938475[54] = 0.0;
   out_4823007598658938475[55] = 0.0;
   out_4823007598658938475[56] = 0.0;
   out_4823007598658938475[57] = 1.0;
   out_4823007598658938475[58] = 0.0;
   out_4823007598658938475[59] = 0.0;
   out_4823007598658938475[60] = 0.0;
   out_4823007598658938475[61] = 0.0;
   out_4823007598658938475[62] = 0.0;
   out_4823007598658938475[63] = 0.0;
   out_4823007598658938475[64] = 0.0;
   out_4823007598658938475[65] = 0.0;
   out_4823007598658938475[66] = 0.0;
   out_4823007598658938475[67] = 0.0;
   out_4823007598658938475[68] = 0.0;
   out_4823007598658938475[69] = 0.0;
   out_4823007598658938475[70] = 0.0;
   out_4823007598658938475[71] = 0.0;
   out_4823007598658938475[72] = 0.0;
   out_4823007598658938475[73] = 0.0;
   out_4823007598658938475[74] = 0.0;
   out_4823007598658938475[75] = 0.0;
   out_4823007598658938475[76] = 1.0;
   out_4823007598658938475[77] = 0.0;
   out_4823007598658938475[78] = 0.0;
   out_4823007598658938475[79] = 0.0;
   out_4823007598658938475[80] = 0.0;
   out_4823007598658938475[81] = 0.0;
   out_4823007598658938475[82] = 0.0;
   out_4823007598658938475[83] = 0.0;
   out_4823007598658938475[84] = 0.0;
   out_4823007598658938475[85] = 0.0;
   out_4823007598658938475[86] = 0.0;
   out_4823007598658938475[87] = 0.0;
   out_4823007598658938475[88] = 0.0;
   out_4823007598658938475[89] = 0.0;
   out_4823007598658938475[90] = 0.0;
   out_4823007598658938475[91] = 0.0;
   out_4823007598658938475[92] = 0.0;
   out_4823007598658938475[93] = 0.0;
   out_4823007598658938475[94] = 0.0;
   out_4823007598658938475[95] = 1.0;
   out_4823007598658938475[96] = 0.0;
   out_4823007598658938475[97] = 0.0;
   out_4823007598658938475[98] = 0.0;
   out_4823007598658938475[99] = 0.0;
   out_4823007598658938475[100] = 0.0;
   out_4823007598658938475[101] = 0.0;
   out_4823007598658938475[102] = 0.0;
   out_4823007598658938475[103] = 0.0;
   out_4823007598658938475[104] = 0.0;
   out_4823007598658938475[105] = 0.0;
   out_4823007598658938475[106] = 0.0;
   out_4823007598658938475[107] = 0.0;
   out_4823007598658938475[108] = 0.0;
   out_4823007598658938475[109] = 0.0;
   out_4823007598658938475[110] = 0.0;
   out_4823007598658938475[111] = 0.0;
   out_4823007598658938475[112] = 0.0;
   out_4823007598658938475[113] = 0.0;
   out_4823007598658938475[114] = 1.0;
   out_4823007598658938475[115] = 0.0;
   out_4823007598658938475[116] = 0.0;
   out_4823007598658938475[117] = 0.0;
   out_4823007598658938475[118] = 0.0;
   out_4823007598658938475[119] = 0.0;
   out_4823007598658938475[120] = 0.0;
   out_4823007598658938475[121] = 0.0;
   out_4823007598658938475[122] = 0.0;
   out_4823007598658938475[123] = 0.0;
   out_4823007598658938475[124] = 0.0;
   out_4823007598658938475[125] = 0.0;
   out_4823007598658938475[126] = 0.0;
   out_4823007598658938475[127] = 0.0;
   out_4823007598658938475[128] = 0.0;
   out_4823007598658938475[129] = 0.0;
   out_4823007598658938475[130] = 0.0;
   out_4823007598658938475[131] = 0.0;
   out_4823007598658938475[132] = 0.0;
   out_4823007598658938475[133] = 1.0;
   out_4823007598658938475[134] = 0.0;
   out_4823007598658938475[135] = 0.0;
   out_4823007598658938475[136] = 0.0;
   out_4823007598658938475[137] = 0.0;
   out_4823007598658938475[138] = 0.0;
   out_4823007598658938475[139] = 0.0;
   out_4823007598658938475[140] = 0.0;
   out_4823007598658938475[141] = 0.0;
   out_4823007598658938475[142] = 0.0;
   out_4823007598658938475[143] = 0.0;
   out_4823007598658938475[144] = 0.0;
   out_4823007598658938475[145] = 0.0;
   out_4823007598658938475[146] = 0.0;
   out_4823007598658938475[147] = 0.0;
   out_4823007598658938475[148] = 0.0;
   out_4823007598658938475[149] = 0.0;
   out_4823007598658938475[150] = 0.0;
   out_4823007598658938475[151] = 0.0;
   out_4823007598658938475[152] = 1.0;
   out_4823007598658938475[153] = 0.0;
   out_4823007598658938475[154] = 0.0;
   out_4823007598658938475[155] = 0.0;
   out_4823007598658938475[156] = 0.0;
   out_4823007598658938475[157] = 0.0;
   out_4823007598658938475[158] = 0.0;
   out_4823007598658938475[159] = 0.0;
   out_4823007598658938475[160] = 0.0;
   out_4823007598658938475[161] = 0.0;
   out_4823007598658938475[162] = 0.0;
   out_4823007598658938475[163] = 0.0;
   out_4823007598658938475[164] = 0.0;
   out_4823007598658938475[165] = 0.0;
   out_4823007598658938475[166] = 0.0;
   out_4823007598658938475[167] = 0.0;
   out_4823007598658938475[168] = 0.0;
   out_4823007598658938475[169] = 0.0;
   out_4823007598658938475[170] = 0.0;
   out_4823007598658938475[171] = 1.0;
   out_4823007598658938475[172] = 0.0;
   out_4823007598658938475[173] = 0.0;
   out_4823007598658938475[174] = 0.0;
   out_4823007598658938475[175] = 0.0;
   out_4823007598658938475[176] = 0.0;
   out_4823007598658938475[177] = 0.0;
   out_4823007598658938475[178] = 0.0;
   out_4823007598658938475[179] = 0.0;
   out_4823007598658938475[180] = 0.0;
   out_4823007598658938475[181] = 0.0;
   out_4823007598658938475[182] = 0.0;
   out_4823007598658938475[183] = 0.0;
   out_4823007598658938475[184] = 0.0;
   out_4823007598658938475[185] = 0.0;
   out_4823007598658938475[186] = 0.0;
   out_4823007598658938475[187] = 0.0;
   out_4823007598658938475[188] = 0.0;
   out_4823007598658938475[189] = 0.0;
   out_4823007598658938475[190] = 1.0;
   out_4823007598658938475[191] = 0.0;
   out_4823007598658938475[192] = 0.0;
   out_4823007598658938475[193] = 0.0;
   out_4823007598658938475[194] = 0.0;
   out_4823007598658938475[195] = 0.0;
   out_4823007598658938475[196] = 0.0;
   out_4823007598658938475[197] = 0.0;
   out_4823007598658938475[198] = 0.0;
   out_4823007598658938475[199] = 0.0;
   out_4823007598658938475[200] = 0.0;
   out_4823007598658938475[201] = 0.0;
   out_4823007598658938475[202] = 0.0;
   out_4823007598658938475[203] = 0.0;
   out_4823007598658938475[204] = 0.0;
   out_4823007598658938475[205] = 0.0;
   out_4823007598658938475[206] = 0.0;
   out_4823007598658938475[207] = 0.0;
   out_4823007598658938475[208] = 0.0;
   out_4823007598658938475[209] = 1.0;
   out_4823007598658938475[210] = 0.0;
   out_4823007598658938475[211] = 0.0;
   out_4823007598658938475[212] = 0.0;
   out_4823007598658938475[213] = 0.0;
   out_4823007598658938475[214] = 0.0;
   out_4823007598658938475[215] = 0.0;
   out_4823007598658938475[216] = 0.0;
   out_4823007598658938475[217] = 0.0;
   out_4823007598658938475[218] = 0.0;
   out_4823007598658938475[219] = 0.0;
   out_4823007598658938475[220] = 0.0;
   out_4823007598658938475[221] = 0.0;
   out_4823007598658938475[222] = 0.0;
   out_4823007598658938475[223] = 0.0;
   out_4823007598658938475[224] = 0.0;
   out_4823007598658938475[225] = 0.0;
   out_4823007598658938475[226] = 0.0;
   out_4823007598658938475[227] = 0.0;
   out_4823007598658938475[228] = 1.0;
   out_4823007598658938475[229] = 0.0;
   out_4823007598658938475[230] = 0.0;
   out_4823007598658938475[231] = 0.0;
   out_4823007598658938475[232] = 0.0;
   out_4823007598658938475[233] = 0.0;
   out_4823007598658938475[234] = 0.0;
   out_4823007598658938475[235] = 0.0;
   out_4823007598658938475[236] = 0.0;
   out_4823007598658938475[237] = 0.0;
   out_4823007598658938475[238] = 0.0;
   out_4823007598658938475[239] = 0.0;
   out_4823007598658938475[240] = 0.0;
   out_4823007598658938475[241] = 0.0;
   out_4823007598658938475[242] = 0.0;
   out_4823007598658938475[243] = 0.0;
   out_4823007598658938475[244] = 0.0;
   out_4823007598658938475[245] = 0.0;
   out_4823007598658938475[246] = 0.0;
   out_4823007598658938475[247] = 1.0;
   out_4823007598658938475[248] = 0.0;
   out_4823007598658938475[249] = 0.0;
   out_4823007598658938475[250] = 0.0;
   out_4823007598658938475[251] = 0.0;
   out_4823007598658938475[252] = 0.0;
   out_4823007598658938475[253] = 0.0;
   out_4823007598658938475[254] = 0.0;
   out_4823007598658938475[255] = 0.0;
   out_4823007598658938475[256] = 0.0;
   out_4823007598658938475[257] = 0.0;
   out_4823007598658938475[258] = 0.0;
   out_4823007598658938475[259] = 0.0;
   out_4823007598658938475[260] = 0.0;
   out_4823007598658938475[261] = 0.0;
   out_4823007598658938475[262] = 0.0;
   out_4823007598658938475[263] = 0.0;
   out_4823007598658938475[264] = 0.0;
   out_4823007598658938475[265] = 0.0;
   out_4823007598658938475[266] = 1.0;
   out_4823007598658938475[267] = 0.0;
   out_4823007598658938475[268] = 0.0;
   out_4823007598658938475[269] = 0.0;
   out_4823007598658938475[270] = 0.0;
   out_4823007598658938475[271] = 0.0;
   out_4823007598658938475[272] = 0.0;
   out_4823007598658938475[273] = 0.0;
   out_4823007598658938475[274] = 0.0;
   out_4823007598658938475[275] = 0.0;
   out_4823007598658938475[276] = 0.0;
   out_4823007598658938475[277] = 0.0;
   out_4823007598658938475[278] = 0.0;
   out_4823007598658938475[279] = 0.0;
   out_4823007598658938475[280] = 0.0;
   out_4823007598658938475[281] = 0.0;
   out_4823007598658938475[282] = 0.0;
   out_4823007598658938475[283] = 0.0;
   out_4823007598658938475[284] = 0.0;
   out_4823007598658938475[285] = 1.0;
   out_4823007598658938475[286] = 0.0;
   out_4823007598658938475[287] = 0.0;
   out_4823007598658938475[288] = 0.0;
   out_4823007598658938475[289] = 0.0;
   out_4823007598658938475[290] = 0.0;
   out_4823007598658938475[291] = 0.0;
   out_4823007598658938475[292] = 0.0;
   out_4823007598658938475[293] = 0.0;
   out_4823007598658938475[294] = 0.0;
   out_4823007598658938475[295] = 0.0;
   out_4823007598658938475[296] = 0.0;
   out_4823007598658938475[297] = 0.0;
   out_4823007598658938475[298] = 0.0;
   out_4823007598658938475[299] = 0.0;
   out_4823007598658938475[300] = 0.0;
   out_4823007598658938475[301] = 0.0;
   out_4823007598658938475[302] = 0.0;
   out_4823007598658938475[303] = 0.0;
   out_4823007598658938475[304] = 1.0;
   out_4823007598658938475[305] = 0.0;
   out_4823007598658938475[306] = 0.0;
   out_4823007598658938475[307] = 0.0;
   out_4823007598658938475[308] = 0.0;
   out_4823007598658938475[309] = 0.0;
   out_4823007598658938475[310] = 0.0;
   out_4823007598658938475[311] = 0.0;
   out_4823007598658938475[312] = 0.0;
   out_4823007598658938475[313] = 0.0;
   out_4823007598658938475[314] = 0.0;
   out_4823007598658938475[315] = 0.0;
   out_4823007598658938475[316] = 0.0;
   out_4823007598658938475[317] = 0.0;
   out_4823007598658938475[318] = 0.0;
   out_4823007598658938475[319] = 0.0;
   out_4823007598658938475[320] = 0.0;
   out_4823007598658938475[321] = 0.0;
   out_4823007598658938475[322] = 0.0;
   out_4823007598658938475[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8206133842831466424) {
   out_8206133842831466424[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8206133842831466424[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8206133842831466424[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8206133842831466424[3] = dt*state[12] + state[3];
   out_8206133842831466424[4] = dt*state[13] + state[4];
   out_8206133842831466424[5] = dt*state[14] + state[5];
   out_8206133842831466424[6] = state[6];
   out_8206133842831466424[7] = state[7];
   out_8206133842831466424[8] = state[8];
   out_8206133842831466424[9] = state[9];
   out_8206133842831466424[10] = state[10];
   out_8206133842831466424[11] = state[11];
   out_8206133842831466424[12] = state[12];
   out_8206133842831466424[13] = state[13];
   out_8206133842831466424[14] = state[14];
   out_8206133842831466424[15] = state[15];
   out_8206133842831466424[16] = state[16];
   out_8206133842831466424[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2449703379901721914) {
   out_2449703379901721914[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2449703379901721914[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2449703379901721914[2] = 0;
   out_2449703379901721914[3] = 0;
   out_2449703379901721914[4] = 0;
   out_2449703379901721914[5] = 0;
   out_2449703379901721914[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2449703379901721914[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2449703379901721914[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2449703379901721914[9] = 0;
   out_2449703379901721914[10] = 0;
   out_2449703379901721914[11] = 0;
   out_2449703379901721914[12] = 0;
   out_2449703379901721914[13] = 0;
   out_2449703379901721914[14] = 0;
   out_2449703379901721914[15] = 0;
   out_2449703379901721914[16] = 0;
   out_2449703379901721914[17] = 0;
   out_2449703379901721914[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2449703379901721914[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2449703379901721914[20] = 0;
   out_2449703379901721914[21] = 0;
   out_2449703379901721914[22] = 0;
   out_2449703379901721914[23] = 0;
   out_2449703379901721914[24] = 0;
   out_2449703379901721914[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2449703379901721914[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2449703379901721914[27] = 0;
   out_2449703379901721914[28] = 0;
   out_2449703379901721914[29] = 0;
   out_2449703379901721914[30] = 0;
   out_2449703379901721914[31] = 0;
   out_2449703379901721914[32] = 0;
   out_2449703379901721914[33] = 0;
   out_2449703379901721914[34] = 0;
   out_2449703379901721914[35] = 0;
   out_2449703379901721914[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2449703379901721914[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2449703379901721914[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2449703379901721914[39] = 0;
   out_2449703379901721914[40] = 0;
   out_2449703379901721914[41] = 0;
   out_2449703379901721914[42] = 0;
   out_2449703379901721914[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2449703379901721914[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2449703379901721914[45] = 0;
   out_2449703379901721914[46] = 0;
   out_2449703379901721914[47] = 0;
   out_2449703379901721914[48] = 0;
   out_2449703379901721914[49] = 0;
   out_2449703379901721914[50] = 0;
   out_2449703379901721914[51] = 0;
   out_2449703379901721914[52] = 0;
   out_2449703379901721914[53] = 0;
   out_2449703379901721914[54] = 0;
   out_2449703379901721914[55] = 0;
   out_2449703379901721914[56] = 0;
   out_2449703379901721914[57] = 1;
   out_2449703379901721914[58] = 0;
   out_2449703379901721914[59] = 0;
   out_2449703379901721914[60] = 0;
   out_2449703379901721914[61] = 0;
   out_2449703379901721914[62] = 0;
   out_2449703379901721914[63] = 0;
   out_2449703379901721914[64] = 0;
   out_2449703379901721914[65] = 0;
   out_2449703379901721914[66] = dt;
   out_2449703379901721914[67] = 0;
   out_2449703379901721914[68] = 0;
   out_2449703379901721914[69] = 0;
   out_2449703379901721914[70] = 0;
   out_2449703379901721914[71] = 0;
   out_2449703379901721914[72] = 0;
   out_2449703379901721914[73] = 0;
   out_2449703379901721914[74] = 0;
   out_2449703379901721914[75] = 0;
   out_2449703379901721914[76] = 1;
   out_2449703379901721914[77] = 0;
   out_2449703379901721914[78] = 0;
   out_2449703379901721914[79] = 0;
   out_2449703379901721914[80] = 0;
   out_2449703379901721914[81] = 0;
   out_2449703379901721914[82] = 0;
   out_2449703379901721914[83] = 0;
   out_2449703379901721914[84] = 0;
   out_2449703379901721914[85] = dt;
   out_2449703379901721914[86] = 0;
   out_2449703379901721914[87] = 0;
   out_2449703379901721914[88] = 0;
   out_2449703379901721914[89] = 0;
   out_2449703379901721914[90] = 0;
   out_2449703379901721914[91] = 0;
   out_2449703379901721914[92] = 0;
   out_2449703379901721914[93] = 0;
   out_2449703379901721914[94] = 0;
   out_2449703379901721914[95] = 1;
   out_2449703379901721914[96] = 0;
   out_2449703379901721914[97] = 0;
   out_2449703379901721914[98] = 0;
   out_2449703379901721914[99] = 0;
   out_2449703379901721914[100] = 0;
   out_2449703379901721914[101] = 0;
   out_2449703379901721914[102] = 0;
   out_2449703379901721914[103] = 0;
   out_2449703379901721914[104] = dt;
   out_2449703379901721914[105] = 0;
   out_2449703379901721914[106] = 0;
   out_2449703379901721914[107] = 0;
   out_2449703379901721914[108] = 0;
   out_2449703379901721914[109] = 0;
   out_2449703379901721914[110] = 0;
   out_2449703379901721914[111] = 0;
   out_2449703379901721914[112] = 0;
   out_2449703379901721914[113] = 0;
   out_2449703379901721914[114] = 1;
   out_2449703379901721914[115] = 0;
   out_2449703379901721914[116] = 0;
   out_2449703379901721914[117] = 0;
   out_2449703379901721914[118] = 0;
   out_2449703379901721914[119] = 0;
   out_2449703379901721914[120] = 0;
   out_2449703379901721914[121] = 0;
   out_2449703379901721914[122] = 0;
   out_2449703379901721914[123] = 0;
   out_2449703379901721914[124] = 0;
   out_2449703379901721914[125] = 0;
   out_2449703379901721914[126] = 0;
   out_2449703379901721914[127] = 0;
   out_2449703379901721914[128] = 0;
   out_2449703379901721914[129] = 0;
   out_2449703379901721914[130] = 0;
   out_2449703379901721914[131] = 0;
   out_2449703379901721914[132] = 0;
   out_2449703379901721914[133] = 1;
   out_2449703379901721914[134] = 0;
   out_2449703379901721914[135] = 0;
   out_2449703379901721914[136] = 0;
   out_2449703379901721914[137] = 0;
   out_2449703379901721914[138] = 0;
   out_2449703379901721914[139] = 0;
   out_2449703379901721914[140] = 0;
   out_2449703379901721914[141] = 0;
   out_2449703379901721914[142] = 0;
   out_2449703379901721914[143] = 0;
   out_2449703379901721914[144] = 0;
   out_2449703379901721914[145] = 0;
   out_2449703379901721914[146] = 0;
   out_2449703379901721914[147] = 0;
   out_2449703379901721914[148] = 0;
   out_2449703379901721914[149] = 0;
   out_2449703379901721914[150] = 0;
   out_2449703379901721914[151] = 0;
   out_2449703379901721914[152] = 1;
   out_2449703379901721914[153] = 0;
   out_2449703379901721914[154] = 0;
   out_2449703379901721914[155] = 0;
   out_2449703379901721914[156] = 0;
   out_2449703379901721914[157] = 0;
   out_2449703379901721914[158] = 0;
   out_2449703379901721914[159] = 0;
   out_2449703379901721914[160] = 0;
   out_2449703379901721914[161] = 0;
   out_2449703379901721914[162] = 0;
   out_2449703379901721914[163] = 0;
   out_2449703379901721914[164] = 0;
   out_2449703379901721914[165] = 0;
   out_2449703379901721914[166] = 0;
   out_2449703379901721914[167] = 0;
   out_2449703379901721914[168] = 0;
   out_2449703379901721914[169] = 0;
   out_2449703379901721914[170] = 0;
   out_2449703379901721914[171] = 1;
   out_2449703379901721914[172] = 0;
   out_2449703379901721914[173] = 0;
   out_2449703379901721914[174] = 0;
   out_2449703379901721914[175] = 0;
   out_2449703379901721914[176] = 0;
   out_2449703379901721914[177] = 0;
   out_2449703379901721914[178] = 0;
   out_2449703379901721914[179] = 0;
   out_2449703379901721914[180] = 0;
   out_2449703379901721914[181] = 0;
   out_2449703379901721914[182] = 0;
   out_2449703379901721914[183] = 0;
   out_2449703379901721914[184] = 0;
   out_2449703379901721914[185] = 0;
   out_2449703379901721914[186] = 0;
   out_2449703379901721914[187] = 0;
   out_2449703379901721914[188] = 0;
   out_2449703379901721914[189] = 0;
   out_2449703379901721914[190] = 1;
   out_2449703379901721914[191] = 0;
   out_2449703379901721914[192] = 0;
   out_2449703379901721914[193] = 0;
   out_2449703379901721914[194] = 0;
   out_2449703379901721914[195] = 0;
   out_2449703379901721914[196] = 0;
   out_2449703379901721914[197] = 0;
   out_2449703379901721914[198] = 0;
   out_2449703379901721914[199] = 0;
   out_2449703379901721914[200] = 0;
   out_2449703379901721914[201] = 0;
   out_2449703379901721914[202] = 0;
   out_2449703379901721914[203] = 0;
   out_2449703379901721914[204] = 0;
   out_2449703379901721914[205] = 0;
   out_2449703379901721914[206] = 0;
   out_2449703379901721914[207] = 0;
   out_2449703379901721914[208] = 0;
   out_2449703379901721914[209] = 1;
   out_2449703379901721914[210] = 0;
   out_2449703379901721914[211] = 0;
   out_2449703379901721914[212] = 0;
   out_2449703379901721914[213] = 0;
   out_2449703379901721914[214] = 0;
   out_2449703379901721914[215] = 0;
   out_2449703379901721914[216] = 0;
   out_2449703379901721914[217] = 0;
   out_2449703379901721914[218] = 0;
   out_2449703379901721914[219] = 0;
   out_2449703379901721914[220] = 0;
   out_2449703379901721914[221] = 0;
   out_2449703379901721914[222] = 0;
   out_2449703379901721914[223] = 0;
   out_2449703379901721914[224] = 0;
   out_2449703379901721914[225] = 0;
   out_2449703379901721914[226] = 0;
   out_2449703379901721914[227] = 0;
   out_2449703379901721914[228] = 1;
   out_2449703379901721914[229] = 0;
   out_2449703379901721914[230] = 0;
   out_2449703379901721914[231] = 0;
   out_2449703379901721914[232] = 0;
   out_2449703379901721914[233] = 0;
   out_2449703379901721914[234] = 0;
   out_2449703379901721914[235] = 0;
   out_2449703379901721914[236] = 0;
   out_2449703379901721914[237] = 0;
   out_2449703379901721914[238] = 0;
   out_2449703379901721914[239] = 0;
   out_2449703379901721914[240] = 0;
   out_2449703379901721914[241] = 0;
   out_2449703379901721914[242] = 0;
   out_2449703379901721914[243] = 0;
   out_2449703379901721914[244] = 0;
   out_2449703379901721914[245] = 0;
   out_2449703379901721914[246] = 0;
   out_2449703379901721914[247] = 1;
   out_2449703379901721914[248] = 0;
   out_2449703379901721914[249] = 0;
   out_2449703379901721914[250] = 0;
   out_2449703379901721914[251] = 0;
   out_2449703379901721914[252] = 0;
   out_2449703379901721914[253] = 0;
   out_2449703379901721914[254] = 0;
   out_2449703379901721914[255] = 0;
   out_2449703379901721914[256] = 0;
   out_2449703379901721914[257] = 0;
   out_2449703379901721914[258] = 0;
   out_2449703379901721914[259] = 0;
   out_2449703379901721914[260] = 0;
   out_2449703379901721914[261] = 0;
   out_2449703379901721914[262] = 0;
   out_2449703379901721914[263] = 0;
   out_2449703379901721914[264] = 0;
   out_2449703379901721914[265] = 0;
   out_2449703379901721914[266] = 1;
   out_2449703379901721914[267] = 0;
   out_2449703379901721914[268] = 0;
   out_2449703379901721914[269] = 0;
   out_2449703379901721914[270] = 0;
   out_2449703379901721914[271] = 0;
   out_2449703379901721914[272] = 0;
   out_2449703379901721914[273] = 0;
   out_2449703379901721914[274] = 0;
   out_2449703379901721914[275] = 0;
   out_2449703379901721914[276] = 0;
   out_2449703379901721914[277] = 0;
   out_2449703379901721914[278] = 0;
   out_2449703379901721914[279] = 0;
   out_2449703379901721914[280] = 0;
   out_2449703379901721914[281] = 0;
   out_2449703379901721914[282] = 0;
   out_2449703379901721914[283] = 0;
   out_2449703379901721914[284] = 0;
   out_2449703379901721914[285] = 1;
   out_2449703379901721914[286] = 0;
   out_2449703379901721914[287] = 0;
   out_2449703379901721914[288] = 0;
   out_2449703379901721914[289] = 0;
   out_2449703379901721914[290] = 0;
   out_2449703379901721914[291] = 0;
   out_2449703379901721914[292] = 0;
   out_2449703379901721914[293] = 0;
   out_2449703379901721914[294] = 0;
   out_2449703379901721914[295] = 0;
   out_2449703379901721914[296] = 0;
   out_2449703379901721914[297] = 0;
   out_2449703379901721914[298] = 0;
   out_2449703379901721914[299] = 0;
   out_2449703379901721914[300] = 0;
   out_2449703379901721914[301] = 0;
   out_2449703379901721914[302] = 0;
   out_2449703379901721914[303] = 0;
   out_2449703379901721914[304] = 1;
   out_2449703379901721914[305] = 0;
   out_2449703379901721914[306] = 0;
   out_2449703379901721914[307] = 0;
   out_2449703379901721914[308] = 0;
   out_2449703379901721914[309] = 0;
   out_2449703379901721914[310] = 0;
   out_2449703379901721914[311] = 0;
   out_2449703379901721914[312] = 0;
   out_2449703379901721914[313] = 0;
   out_2449703379901721914[314] = 0;
   out_2449703379901721914[315] = 0;
   out_2449703379901721914[316] = 0;
   out_2449703379901721914[317] = 0;
   out_2449703379901721914[318] = 0;
   out_2449703379901721914[319] = 0;
   out_2449703379901721914[320] = 0;
   out_2449703379901721914[321] = 0;
   out_2449703379901721914[322] = 0;
   out_2449703379901721914[323] = 1;
}
void h_4(double *state, double *unused, double *out_1814135670413092001) {
   out_1814135670413092001[0] = state[6] + state[9];
   out_1814135670413092001[1] = state[7] + state[10];
   out_1814135670413092001[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_8594814808502846047) {
   out_8594814808502846047[0] = 0;
   out_8594814808502846047[1] = 0;
   out_8594814808502846047[2] = 0;
   out_8594814808502846047[3] = 0;
   out_8594814808502846047[4] = 0;
   out_8594814808502846047[5] = 0;
   out_8594814808502846047[6] = 1;
   out_8594814808502846047[7] = 0;
   out_8594814808502846047[8] = 0;
   out_8594814808502846047[9] = 1;
   out_8594814808502846047[10] = 0;
   out_8594814808502846047[11] = 0;
   out_8594814808502846047[12] = 0;
   out_8594814808502846047[13] = 0;
   out_8594814808502846047[14] = 0;
   out_8594814808502846047[15] = 0;
   out_8594814808502846047[16] = 0;
   out_8594814808502846047[17] = 0;
   out_8594814808502846047[18] = 0;
   out_8594814808502846047[19] = 0;
   out_8594814808502846047[20] = 0;
   out_8594814808502846047[21] = 0;
   out_8594814808502846047[22] = 0;
   out_8594814808502846047[23] = 0;
   out_8594814808502846047[24] = 0;
   out_8594814808502846047[25] = 1;
   out_8594814808502846047[26] = 0;
   out_8594814808502846047[27] = 0;
   out_8594814808502846047[28] = 1;
   out_8594814808502846047[29] = 0;
   out_8594814808502846047[30] = 0;
   out_8594814808502846047[31] = 0;
   out_8594814808502846047[32] = 0;
   out_8594814808502846047[33] = 0;
   out_8594814808502846047[34] = 0;
   out_8594814808502846047[35] = 0;
   out_8594814808502846047[36] = 0;
   out_8594814808502846047[37] = 0;
   out_8594814808502846047[38] = 0;
   out_8594814808502846047[39] = 0;
   out_8594814808502846047[40] = 0;
   out_8594814808502846047[41] = 0;
   out_8594814808502846047[42] = 0;
   out_8594814808502846047[43] = 0;
   out_8594814808502846047[44] = 1;
   out_8594814808502846047[45] = 0;
   out_8594814808502846047[46] = 0;
   out_8594814808502846047[47] = 1;
   out_8594814808502846047[48] = 0;
   out_8594814808502846047[49] = 0;
   out_8594814808502846047[50] = 0;
   out_8594814808502846047[51] = 0;
   out_8594814808502846047[52] = 0;
   out_8594814808502846047[53] = 0;
}
void h_10(double *state, double *unused, double *out_977191010376645903) {
   out_977191010376645903[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_977191010376645903[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_977191010376645903[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4157265640572721743) {
   out_4157265640572721743[0] = 0;
   out_4157265640572721743[1] = 9.8100000000000005*cos(state[1]);
   out_4157265640572721743[2] = 0;
   out_4157265640572721743[3] = 0;
   out_4157265640572721743[4] = -state[8];
   out_4157265640572721743[5] = state[7];
   out_4157265640572721743[6] = 0;
   out_4157265640572721743[7] = state[5];
   out_4157265640572721743[8] = -state[4];
   out_4157265640572721743[9] = 0;
   out_4157265640572721743[10] = 0;
   out_4157265640572721743[11] = 0;
   out_4157265640572721743[12] = 1;
   out_4157265640572721743[13] = 0;
   out_4157265640572721743[14] = 0;
   out_4157265640572721743[15] = 1;
   out_4157265640572721743[16] = 0;
   out_4157265640572721743[17] = 0;
   out_4157265640572721743[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4157265640572721743[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4157265640572721743[20] = 0;
   out_4157265640572721743[21] = state[8];
   out_4157265640572721743[22] = 0;
   out_4157265640572721743[23] = -state[6];
   out_4157265640572721743[24] = -state[5];
   out_4157265640572721743[25] = 0;
   out_4157265640572721743[26] = state[3];
   out_4157265640572721743[27] = 0;
   out_4157265640572721743[28] = 0;
   out_4157265640572721743[29] = 0;
   out_4157265640572721743[30] = 0;
   out_4157265640572721743[31] = 1;
   out_4157265640572721743[32] = 0;
   out_4157265640572721743[33] = 0;
   out_4157265640572721743[34] = 1;
   out_4157265640572721743[35] = 0;
   out_4157265640572721743[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4157265640572721743[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4157265640572721743[38] = 0;
   out_4157265640572721743[39] = -state[7];
   out_4157265640572721743[40] = state[6];
   out_4157265640572721743[41] = 0;
   out_4157265640572721743[42] = state[4];
   out_4157265640572721743[43] = -state[3];
   out_4157265640572721743[44] = 0;
   out_4157265640572721743[45] = 0;
   out_4157265640572721743[46] = 0;
   out_4157265640572721743[47] = 0;
   out_4157265640572721743[48] = 0;
   out_4157265640572721743[49] = 0;
   out_4157265640572721743[50] = 1;
   out_4157265640572721743[51] = 0;
   out_4157265640572721743[52] = 0;
   out_4157265640572721743[53] = 1;
}
void h_13(double *state, double *unused, double *out_51042124798717067) {
   out_51042124798717067[0] = state[3];
   out_51042124798717067[1] = state[4];
   out_51042124798717067[2] = state[5];
}
void H_13(double *state, double *unused, double *out_9159416728184690151) {
   out_9159416728184690151[0] = 0;
   out_9159416728184690151[1] = 0;
   out_9159416728184690151[2] = 0;
   out_9159416728184690151[3] = 1;
   out_9159416728184690151[4] = 0;
   out_9159416728184690151[5] = 0;
   out_9159416728184690151[6] = 0;
   out_9159416728184690151[7] = 0;
   out_9159416728184690151[8] = 0;
   out_9159416728184690151[9] = 0;
   out_9159416728184690151[10] = 0;
   out_9159416728184690151[11] = 0;
   out_9159416728184690151[12] = 0;
   out_9159416728184690151[13] = 0;
   out_9159416728184690151[14] = 0;
   out_9159416728184690151[15] = 0;
   out_9159416728184690151[16] = 0;
   out_9159416728184690151[17] = 0;
   out_9159416728184690151[18] = 0;
   out_9159416728184690151[19] = 0;
   out_9159416728184690151[20] = 0;
   out_9159416728184690151[21] = 0;
   out_9159416728184690151[22] = 1;
   out_9159416728184690151[23] = 0;
   out_9159416728184690151[24] = 0;
   out_9159416728184690151[25] = 0;
   out_9159416728184690151[26] = 0;
   out_9159416728184690151[27] = 0;
   out_9159416728184690151[28] = 0;
   out_9159416728184690151[29] = 0;
   out_9159416728184690151[30] = 0;
   out_9159416728184690151[31] = 0;
   out_9159416728184690151[32] = 0;
   out_9159416728184690151[33] = 0;
   out_9159416728184690151[34] = 0;
   out_9159416728184690151[35] = 0;
   out_9159416728184690151[36] = 0;
   out_9159416728184690151[37] = 0;
   out_9159416728184690151[38] = 0;
   out_9159416728184690151[39] = 0;
   out_9159416728184690151[40] = 0;
   out_9159416728184690151[41] = 1;
   out_9159416728184690151[42] = 0;
   out_9159416728184690151[43] = 0;
   out_9159416728184690151[44] = 0;
   out_9159416728184690151[45] = 0;
   out_9159416728184690151[46] = 0;
   out_9159416728184690151[47] = 0;
   out_9159416728184690151[48] = 0;
   out_9159416728184690151[49] = 0;
   out_9159416728184690151[50] = 0;
   out_9159416728184690151[51] = 0;
   out_9159416728184690151[52] = 0;
   out_9159416728184690151[53] = 0;
}
void h_14(double *state, double *unused, double *out_6024313293648676072) {
   out_6024313293648676072[0] = state[6];
   out_6024313293648676072[1] = state[7];
   out_6024313293648676072[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5512026376207473751) {
   out_5512026376207473751[0] = 0;
   out_5512026376207473751[1] = 0;
   out_5512026376207473751[2] = 0;
   out_5512026376207473751[3] = 0;
   out_5512026376207473751[4] = 0;
   out_5512026376207473751[5] = 0;
   out_5512026376207473751[6] = 1;
   out_5512026376207473751[7] = 0;
   out_5512026376207473751[8] = 0;
   out_5512026376207473751[9] = 0;
   out_5512026376207473751[10] = 0;
   out_5512026376207473751[11] = 0;
   out_5512026376207473751[12] = 0;
   out_5512026376207473751[13] = 0;
   out_5512026376207473751[14] = 0;
   out_5512026376207473751[15] = 0;
   out_5512026376207473751[16] = 0;
   out_5512026376207473751[17] = 0;
   out_5512026376207473751[18] = 0;
   out_5512026376207473751[19] = 0;
   out_5512026376207473751[20] = 0;
   out_5512026376207473751[21] = 0;
   out_5512026376207473751[22] = 0;
   out_5512026376207473751[23] = 0;
   out_5512026376207473751[24] = 0;
   out_5512026376207473751[25] = 1;
   out_5512026376207473751[26] = 0;
   out_5512026376207473751[27] = 0;
   out_5512026376207473751[28] = 0;
   out_5512026376207473751[29] = 0;
   out_5512026376207473751[30] = 0;
   out_5512026376207473751[31] = 0;
   out_5512026376207473751[32] = 0;
   out_5512026376207473751[33] = 0;
   out_5512026376207473751[34] = 0;
   out_5512026376207473751[35] = 0;
   out_5512026376207473751[36] = 0;
   out_5512026376207473751[37] = 0;
   out_5512026376207473751[38] = 0;
   out_5512026376207473751[39] = 0;
   out_5512026376207473751[40] = 0;
   out_5512026376207473751[41] = 0;
   out_5512026376207473751[42] = 0;
   out_5512026376207473751[43] = 0;
   out_5512026376207473751[44] = 1;
   out_5512026376207473751[45] = 0;
   out_5512026376207473751[46] = 0;
   out_5512026376207473751[47] = 0;
   out_5512026376207473751[48] = 0;
   out_5512026376207473751[49] = 0;
   out_5512026376207473751[50] = 0;
   out_5512026376207473751[51] = 0;
   out_5512026376207473751[52] = 0;
   out_5512026376207473751[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_8566306649041746244) {
  err_fun(nom_x, delta_x, out_8566306649041746244);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1696404042837177676) {
  inv_err_fun(nom_x, true_x, out_1696404042837177676);
}
void pose_H_mod_fun(double *state, double *out_4823007598658938475) {
  H_mod_fun(state, out_4823007598658938475);
}
void pose_f_fun(double *state, double dt, double *out_8206133842831466424) {
  f_fun(state,  dt, out_8206133842831466424);
}
void pose_F_fun(double *state, double dt, double *out_2449703379901721914) {
  F_fun(state,  dt, out_2449703379901721914);
}
void pose_h_4(double *state, double *unused, double *out_1814135670413092001) {
  h_4(state, unused, out_1814135670413092001);
}
void pose_H_4(double *state, double *unused, double *out_8594814808502846047) {
  H_4(state, unused, out_8594814808502846047);
}
void pose_h_10(double *state, double *unused, double *out_977191010376645903) {
  h_10(state, unused, out_977191010376645903);
}
void pose_H_10(double *state, double *unused, double *out_4157265640572721743) {
  H_10(state, unused, out_4157265640572721743);
}
void pose_h_13(double *state, double *unused, double *out_51042124798717067) {
  h_13(state, unused, out_51042124798717067);
}
void pose_H_13(double *state, double *unused, double *out_9159416728184690151) {
  H_13(state, unused, out_9159416728184690151);
}
void pose_h_14(double *state, double *unused, double *out_6024313293648676072) {
  h_14(state, unused, out_6024313293648676072);
}
void pose_H_14(double *state, double *unused, double *out_5512026376207473751) {
  H_14(state, unused, out_5512026376207473751);
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
