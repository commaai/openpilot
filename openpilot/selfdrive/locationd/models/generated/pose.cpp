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
void err_fun(double *nom_x, double *delta_x, double *out_348428071618360721) {
   out_348428071618360721[0] = delta_x[0] + nom_x[0];
   out_348428071618360721[1] = delta_x[1] + nom_x[1];
   out_348428071618360721[2] = delta_x[2] + nom_x[2];
   out_348428071618360721[3] = delta_x[3] + nom_x[3];
   out_348428071618360721[4] = delta_x[4] + nom_x[4];
   out_348428071618360721[5] = delta_x[5] + nom_x[5];
   out_348428071618360721[6] = delta_x[6] + nom_x[6];
   out_348428071618360721[7] = delta_x[7] + nom_x[7];
   out_348428071618360721[8] = delta_x[8] + nom_x[8];
   out_348428071618360721[9] = delta_x[9] + nom_x[9];
   out_348428071618360721[10] = delta_x[10] + nom_x[10];
   out_348428071618360721[11] = delta_x[11] + nom_x[11];
   out_348428071618360721[12] = delta_x[12] + nom_x[12];
   out_348428071618360721[13] = delta_x[13] + nom_x[13];
   out_348428071618360721[14] = delta_x[14] + nom_x[14];
   out_348428071618360721[15] = delta_x[15] + nom_x[15];
   out_348428071618360721[16] = delta_x[16] + nom_x[16];
   out_348428071618360721[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2713115070426494842) {
   out_2713115070426494842[0] = -nom_x[0] + true_x[0];
   out_2713115070426494842[1] = -nom_x[1] + true_x[1];
   out_2713115070426494842[2] = -nom_x[2] + true_x[2];
   out_2713115070426494842[3] = -nom_x[3] + true_x[3];
   out_2713115070426494842[4] = -nom_x[4] + true_x[4];
   out_2713115070426494842[5] = -nom_x[5] + true_x[5];
   out_2713115070426494842[6] = -nom_x[6] + true_x[6];
   out_2713115070426494842[7] = -nom_x[7] + true_x[7];
   out_2713115070426494842[8] = -nom_x[8] + true_x[8];
   out_2713115070426494842[9] = -nom_x[9] + true_x[9];
   out_2713115070426494842[10] = -nom_x[10] + true_x[10];
   out_2713115070426494842[11] = -nom_x[11] + true_x[11];
   out_2713115070426494842[12] = -nom_x[12] + true_x[12];
   out_2713115070426494842[13] = -nom_x[13] + true_x[13];
   out_2713115070426494842[14] = -nom_x[14] + true_x[14];
   out_2713115070426494842[15] = -nom_x[15] + true_x[15];
   out_2713115070426494842[16] = -nom_x[16] + true_x[16];
   out_2713115070426494842[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1740654701803477855) {
   out_1740654701803477855[0] = 1.0;
   out_1740654701803477855[1] = 0.0;
   out_1740654701803477855[2] = 0.0;
   out_1740654701803477855[3] = 0.0;
   out_1740654701803477855[4] = 0.0;
   out_1740654701803477855[5] = 0.0;
   out_1740654701803477855[6] = 0.0;
   out_1740654701803477855[7] = 0.0;
   out_1740654701803477855[8] = 0.0;
   out_1740654701803477855[9] = 0.0;
   out_1740654701803477855[10] = 0.0;
   out_1740654701803477855[11] = 0.0;
   out_1740654701803477855[12] = 0.0;
   out_1740654701803477855[13] = 0.0;
   out_1740654701803477855[14] = 0.0;
   out_1740654701803477855[15] = 0.0;
   out_1740654701803477855[16] = 0.0;
   out_1740654701803477855[17] = 0.0;
   out_1740654701803477855[18] = 0.0;
   out_1740654701803477855[19] = 1.0;
   out_1740654701803477855[20] = 0.0;
   out_1740654701803477855[21] = 0.0;
   out_1740654701803477855[22] = 0.0;
   out_1740654701803477855[23] = 0.0;
   out_1740654701803477855[24] = 0.0;
   out_1740654701803477855[25] = 0.0;
   out_1740654701803477855[26] = 0.0;
   out_1740654701803477855[27] = 0.0;
   out_1740654701803477855[28] = 0.0;
   out_1740654701803477855[29] = 0.0;
   out_1740654701803477855[30] = 0.0;
   out_1740654701803477855[31] = 0.0;
   out_1740654701803477855[32] = 0.0;
   out_1740654701803477855[33] = 0.0;
   out_1740654701803477855[34] = 0.0;
   out_1740654701803477855[35] = 0.0;
   out_1740654701803477855[36] = 0.0;
   out_1740654701803477855[37] = 0.0;
   out_1740654701803477855[38] = 1.0;
   out_1740654701803477855[39] = 0.0;
   out_1740654701803477855[40] = 0.0;
   out_1740654701803477855[41] = 0.0;
   out_1740654701803477855[42] = 0.0;
   out_1740654701803477855[43] = 0.0;
   out_1740654701803477855[44] = 0.0;
   out_1740654701803477855[45] = 0.0;
   out_1740654701803477855[46] = 0.0;
   out_1740654701803477855[47] = 0.0;
   out_1740654701803477855[48] = 0.0;
   out_1740654701803477855[49] = 0.0;
   out_1740654701803477855[50] = 0.0;
   out_1740654701803477855[51] = 0.0;
   out_1740654701803477855[52] = 0.0;
   out_1740654701803477855[53] = 0.0;
   out_1740654701803477855[54] = 0.0;
   out_1740654701803477855[55] = 0.0;
   out_1740654701803477855[56] = 0.0;
   out_1740654701803477855[57] = 1.0;
   out_1740654701803477855[58] = 0.0;
   out_1740654701803477855[59] = 0.0;
   out_1740654701803477855[60] = 0.0;
   out_1740654701803477855[61] = 0.0;
   out_1740654701803477855[62] = 0.0;
   out_1740654701803477855[63] = 0.0;
   out_1740654701803477855[64] = 0.0;
   out_1740654701803477855[65] = 0.0;
   out_1740654701803477855[66] = 0.0;
   out_1740654701803477855[67] = 0.0;
   out_1740654701803477855[68] = 0.0;
   out_1740654701803477855[69] = 0.0;
   out_1740654701803477855[70] = 0.0;
   out_1740654701803477855[71] = 0.0;
   out_1740654701803477855[72] = 0.0;
   out_1740654701803477855[73] = 0.0;
   out_1740654701803477855[74] = 0.0;
   out_1740654701803477855[75] = 0.0;
   out_1740654701803477855[76] = 1.0;
   out_1740654701803477855[77] = 0.0;
   out_1740654701803477855[78] = 0.0;
   out_1740654701803477855[79] = 0.0;
   out_1740654701803477855[80] = 0.0;
   out_1740654701803477855[81] = 0.0;
   out_1740654701803477855[82] = 0.0;
   out_1740654701803477855[83] = 0.0;
   out_1740654701803477855[84] = 0.0;
   out_1740654701803477855[85] = 0.0;
   out_1740654701803477855[86] = 0.0;
   out_1740654701803477855[87] = 0.0;
   out_1740654701803477855[88] = 0.0;
   out_1740654701803477855[89] = 0.0;
   out_1740654701803477855[90] = 0.0;
   out_1740654701803477855[91] = 0.0;
   out_1740654701803477855[92] = 0.0;
   out_1740654701803477855[93] = 0.0;
   out_1740654701803477855[94] = 0.0;
   out_1740654701803477855[95] = 1.0;
   out_1740654701803477855[96] = 0.0;
   out_1740654701803477855[97] = 0.0;
   out_1740654701803477855[98] = 0.0;
   out_1740654701803477855[99] = 0.0;
   out_1740654701803477855[100] = 0.0;
   out_1740654701803477855[101] = 0.0;
   out_1740654701803477855[102] = 0.0;
   out_1740654701803477855[103] = 0.0;
   out_1740654701803477855[104] = 0.0;
   out_1740654701803477855[105] = 0.0;
   out_1740654701803477855[106] = 0.0;
   out_1740654701803477855[107] = 0.0;
   out_1740654701803477855[108] = 0.0;
   out_1740654701803477855[109] = 0.0;
   out_1740654701803477855[110] = 0.0;
   out_1740654701803477855[111] = 0.0;
   out_1740654701803477855[112] = 0.0;
   out_1740654701803477855[113] = 0.0;
   out_1740654701803477855[114] = 1.0;
   out_1740654701803477855[115] = 0.0;
   out_1740654701803477855[116] = 0.0;
   out_1740654701803477855[117] = 0.0;
   out_1740654701803477855[118] = 0.0;
   out_1740654701803477855[119] = 0.0;
   out_1740654701803477855[120] = 0.0;
   out_1740654701803477855[121] = 0.0;
   out_1740654701803477855[122] = 0.0;
   out_1740654701803477855[123] = 0.0;
   out_1740654701803477855[124] = 0.0;
   out_1740654701803477855[125] = 0.0;
   out_1740654701803477855[126] = 0.0;
   out_1740654701803477855[127] = 0.0;
   out_1740654701803477855[128] = 0.0;
   out_1740654701803477855[129] = 0.0;
   out_1740654701803477855[130] = 0.0;
   out_1740654701803477855[131] = 0.0;
   out_1740654701803477855[132] = 0.0;
   out_1740654701803477855[133] = 1.0;
   out_1740654701803477855[134] = 0.0;
   out_1740654701803477855[135] = 0.0;
   out_1740654701803477855[136] = 0.0;
   out_1740654701803477855[137] = 0.0;
   out_1740654701803477855[138] = 0.0;
   out_1740654701803477855[139] = 0.0;
   out_1740654701803477855[140] = 0.0;
   out_1740654701803477855[141] = 0.0;
   out_1740654701803477855[142] = 0.0;
   out_1740654701803477855[143] = 0.0;
   out_1740654701803477855[144] = 0.0;
   out_1740654701803477855[145] = 0.0;
   out_1740654701803477855[146] = 0.0;
   out_1740654701803477855[147] = 0.0;
   out_1740654701803477855[148] = 0.0;
   out_1740654701803477855[149] = 0.0;
   out_1740654701803477855[150] = 0.0;
   out_1740654701803477855[151] = 0.0;
   out_1740654701803477855[152] = 1.0;
   out_1740654701803477855[153] = 0.0;
   out_1740654701803477855[154] = 0.0;
   out_1740654701803477855[155] = 0.0;
   out_1740654701803477855[156] = 0.0;
   out_1740654701803477855[157] = 0.0;
   out_1740654701803477855[158] = 0.0;
   out_1740654701803477855[159] = 0.0;
   out_1740654701803477855[160] = 0.0;
   out_1740654701803477855[161] = 0.0;
   out_1740654701803477855[162] = 0.0;
   out_1740654701803477855[163] = 0.0;
   out_1740654701803477855[164] = 0.0;
   out_1740654701803477855[165] = 0.0;
   out_1740654701803477855[166] = 0.0;
   out_1740654701803477855[167] = 0.0;
   out_1740654701803477855[168] = 0.0;
   out_1740654701803477855[169] = 0.0;
   out_1740654701803477855[170] = 0.0;
   out_1740654701803477855[171] = 1.0;
   out_1740654701803477855[172] = 0.0;
   out_1740654701803477855[173] = 0.0;
   out_1740654701803477855[174] = 0.0;
   out_1740654701803477855[175] = 0.0;
   out_1740654701803477855[176] = 0.0;
   out_1740654701803477855[177] = 0.0;
   out_1740654701803477855[178] = 0.0;
   out_1740654701803477855[179] = 0.0;
   out_1740654701803477855[180] = 0.0;
   out_1740654701803477855[181] = 0.0;
   out_1740654701803477855[182] = 0.0;
   out_1740654701803477855[183] = 0.0;
   out_1740654701803477855[184] = 0.0;
   out_1740654701803477855[185] = 0.0;
   out_1740654701803477855[186] = 0.0;
   out_1740654701803477855[187] = 0.0;
   out_1740654701803477855[188] = 0.0;
   out_1740654701803477855[189] = 0.0;
   out_1740654701803477855[190] = 1.0;
   out_1740654701803477855[191] = 0.0;
   out_1740654701803477855[192] = 0.0;
   out_1740654701803477855[193] = 0.0;
   out_1740654701803477855[194] = 0.0;
   out_1740654701803477855[195] = 0.0;
   out_1740654701803477855[196] = 0.0;
   out_1740654701803477855[197] = 0.0;
   out_1740654701803477855[198] = 0.0;
   out_1740654701803477855[199] = 0.0;
   out_1740654701803477855[200] = 0.0;
   out_1740654701803477855[201] = 0.0;
   out_1740654701803477855[202] = 0.0;
   out_1740654701803477855[203] = 0.0;
   out_1740654701803477855[204] = 0.0;
   out_1740654701803477855[205] = 0.0;
   out_1740654701803477855[206] = 0.0;
   out_1740654701803477855[207] = 0.0;
   out_1740654701803477855[208] = 0.0;
   out_1740654701803477855[209] = 1.0;
   out_1740654701803477855[210] = 0.0;
   out_1740654701803477855[211] = 0.0;
   out_1740654701803477855[212] = 0.0;
   out_1740654701803477855[213] = 0.0;
   out_1740654701803477855[214] = 0.0;
   out_1740654701803477855[215] = 0.0;
   out_1740654701803477855[216] = 0.0;
   out_1740654701803477855[217] = 0.0;
   out_1740654701803477855[218] = 0.0;
   out_1740654701803477855[219] = 0.0;
   out_1740654701803477855[220] = 0.0;
   out_1740654701803477855[221] = 0.0;
   out_1740654701803477855[222] = 0.0;
   out_1740654701803477855[223] = 0.0;
   out_1740654701803477855[224] = 0.0;
   out_1740654701803477855[225] = 0.0;
   out_1740654701803477855[226] = 0.0;
   out_1740654701803477855[227] = 0.0;
   out_1740654701803477855[228] = 1.0;
   out_1740654701803477855[229] = 0.0;
   out_1740654701803477855[230] = 0.0;
   out_1740654701803477855[231] = 0.0;
   out_1740654701803477855[232] = 0.0;
   out_1740654701803477855[233] = 0.0;
   out_1740654701803477855[234] = 0.0;
   out_1740654701803477855[235] = 0.0;
   out_1740654701803477855[236] = 0.0;
   out_1740654701803477855[237] = 0.0;
   out_1740654701803477855[238] = 0.0;
   out_1740654701803477855[239] = 0.0;
   out_1740654701803477855[240] = 0.0;
   out_1740654701803477855[241] = 0.0;
   out_1740654701803477855[242] = 0.0;
   out_1740654701803477855[243] = 0.0;
   out_1740654701803477855[244] = 0.0;
   out_1740654701803477855[245] = 0.0;
   out_1740654701803477855[246] = 0.0;
   out_1740654701803477855[247] = 1.0;
   out_1740654701803477855[248] = 0.0;
   out_1740654701803477855[249] = 0.0;
   out_1740654701803477855[250] = 0.0;
   out_1740654701803477855[251] = 0.0;
   out_1740654701803477855[252] = 0.0;
   out_1740654701803477855[253] = 0.0;
   out_1740654701803477855[254] = 0.0;
   out_1740654701803477855[255] = 0.0;
   out_1740654701803477855[256] = 0.0;
   out_1740654701803477855[257] = 0.0;
   out_1740654701803477855[258] = 0.0;
   out_1740654701803477855[259] = 0.0;
   out_1740654701803477855[260] = 0.0;
   out_1740654701803477855[261] = 0.0;
   out_1740654701803477855[262] = 0.0;
   out_1740654701803477855[263] = 0.0;
   out_1740654701803477855[264] = 0.0;
   out_1740654701803477855[265] = 0.0;
   out_1740654701803477855[266] = 1.0;
   out_1740654701803477855[267] = 0.0;
   out_1740654701803477855[268] = 0.0;
   out_1740654701803477855[269] = 0.0;
   out_1740654701803477855[270] = 0.0;
   out_1740654701803477855[271] = 0.0;
   out_1740654701803477855[272] = 0.0;
   out_1740654701803477855[273] = 0.0;
   out_1740654701803477855[274] = 0.0;
   out_1740654701803477855[275] = 0.0;
   out_1740654701803477855[276] = 0.0;
   out_1740654701803477855[277] = 0.0;
   out_1740654701803477855[278] = 0.0;
   out_1740654701803477855[279] = 0.0;
   out_1740654701803477855[280] = 0.0;
   out_1740654701803477855[281] = 0.0;
   out_1740654701803477855[282] = 0.0;
   out_1740654701803477855[283] = 0.0;
   out_1740654701803477855[284] = 0.0;
   out_1740654701803477855[285] = 1.0;
   out_1740654701803477855[286] = 0.0;
   out_1740654701803477855[287] = 0.0;
   out_1740654701803477855[288] = 0.0;
   out_1740654701803477855[289] = 0.0;
   out_1740654701803477855[290] = 0.0;
   out_1740654701803477855[291] = 0.0;
   out_1740654701803477855[292] = 0.0;
   out_1740654701803477855[293] = 0.0;
   out_1740654701803477855[294] = 0.0;
   out_1740654701803477855[295] = 0.0;
   out_1740654701803477855[296] = 0.0;
   out_1740654701803477855[297] = 0.0;
   out_1740654701803477855[298] = 0.0;
   out_1740654701803477855[299] = 0.0;
   out_1740654701803477855[300] = 0.0;
   out_1740654701803477855[301] = 0.0;
   out_1740654701803477855[302] = 0.0;
   out_1740654701803477855[303] = 0.0;
   out_1740654701803477855[304] = 1.0;
   out_1740654701803477855[305] = 0.0;
   out_1740654701803477855[306] = 0.0;
   out_1740654701803477855[307] = 0.0;
   out_1740654701803477855[308] = 0.0;
   out_1740654701803477855[309] = 0.0;
   out_1740654701803477855[310] = 0.0;
   out_1740654701803477855[311] = 0.0;
   out_1740654701803477855[312] = 0.0;
   out_1740654701803477855[313] = 0.0;
   out_1740654701803477855[314] = 0.0;
   out_1740654701803477855[315] = 0.0;
   out_1740654701803477855[316] = 0.0;
   out_1740654701803477855[317] = 0.0;
   out_1740654701803477855[318] = 0.0;
   out_1740654701803477855[319] = 0.0;
   out_1740654701803477855[320] = 0.0;
   out_1740654701803477855[321] = 0.0;
   out_1740654701803477855[322] = 0.0;
   out_1740654701803477855[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7474451683258418033) {
   out_7474451683258418033[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7474451683258418033[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7474451683258418033[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7474451683258418033[3] = dt*state[12] + state[3];
   out_7474451683258418033[4] = dt*state[13] + state[4];
   out_7474451683258418033[5] = dt*state[14] + state[5];
   out_7474451683258418033[6] = state[6];
   out_7474451683258418033[7] = state[7];
   out_7474451683258418033[8] = state[8];
   out_7474451683258418033[9] = state[9];
   out_7474451683258418033[10] = state[10];
   out_7474451683258418033[11] = state[11];
   out_7474451683258418033[12] = state[12];
   out_7474451683258418033[13] = state[13];
   out_7474451683258418033[14] = state[14];
   out_7474451683258418033[15] = state[15];
   out_7474451683258418033[16] = state[16];
   out_7474451683258418033[17] = state[17];
}
void F_fun(double *state, double dt, double *out_953892830646151789) {
   out_953892830646151789[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_953892830646151789[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_953892830646151789[2] = 0;
   out_953892830646151789[3] = 0;
   out_953892830646151789[4] = 0;
   out_953892830646151789[5] = 0;
   out_953892830646151789[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_953892830646151789[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_953892830646151789[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_953892830646151789[9] = 0;
   out_953892830646151789[10] = 0;
   out_953892830646151789[11] = 0;
   out_953892830646151789[12] = 0;
   out_953892830646151789[13] = 0;
   out_953892830646151789[14] = 0;
   out_953892830646151789[15] = 0;
   out_953892830646151789[16] = 0;
   out_953892830646151789[17] = 0;
   out_953892830646151789[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_953892830646151789[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_953892830646151789[20] = 0;
   out_953892830646151789[21] = 0;
   out_953892830646151789[22] = 0;
   out_953892830646151789[23] = 0;
   out_953892830646151789[24] = 0;
   out_953892830646151789[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_953892830646151789[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_953892830646151789[27] = 0;
   out_953892830646151789[28] = 0;
   out_953892830646151789[29] = 0;
   out_953892830646151789[30] = 0;
   out_953892830646151789[31] = 0;
   out_953892830646151789[32] = 0;
   out_953892830646151789[33] = 0;
   out_953892830646151789[34] = 0;
   out_953892830646151789[35] = 0;
   out_953892830646151789[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_953892830646151789[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_953892830646151789[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_953892830646151789[39] = 0;
   out_953892830646151789[40] = 0;
   out_953892830646151789[41] = 0;
   out_953892830646151789[42] = 0;
   out_953892830646151789[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_953892830646151789[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_953892830646151789[45] = 0;
   out_953892830646151789[46] = 0;
   out_953892830646151789[47] = 0;
   out_953892830646151789[48] = 0;
   out_953892830646151789[49] = 0;
   out_953892830646151789[50] = 0;
   out_953892830646151789[51] = 0;
   out_953892830646151789[52] = 0;
   out_953892830646151789[53] = 0;
   out_953892830646151789[54] = 0;
   out_953892830646151789[55] = 0;
   out_953892830646151789[56] = 0;
   out_953892830646151789[57] = 1;
   out_953892830646151789[58] = 0;
   out_953892830646151789[59] = 0;
   out_953892830646151789[60] = 0;
   out_953892830646151789[61] = 0;
   out_953892830646151789[62] = 0;
   out_953892830646151789[63] = 0;
   out_953892830646151789[64] = 0;
   out_953892830646151789[65] = 0;
   out_953892830646151789[66] = dt;
   out_953892830646151789[67] = 0;
   out_953892830646151789[68] = 0;
   out_953892830646151789[69] = 0;
   out_953892830646151789[70] = 0;
   out_953892830646151789[71] = 0;
   out_953892830646151789[72] = 0;
   out_953892830646151789[73] = 0;
   out_953892830646151789[74] = 0;
   out_953892830646151789[75] = 0;
   out_953892830646151789[76] = 1;
   out_953892830646151789[77] = 0;
   out_953892830646151789[78] = 0;
   out_953892830646151789[79] = 0;
   out_953892830646151789[80] = 0;
   out_953892830646151789[81] = 0;
   out_953892830646151789[82] = 0;
   out_953892830646151789[83] = 0;
   out_953892830646151789[84] = 0;
   out_953892830646151789[85] = dt;
   out_953892830646151789[86] = 0;
   out_953892830646151789[87] = 0;
   out_953892830646151789[88] = 0;
   out_953892830646151789[89] = 0;
   out_953892830646151789[90] = 0;
   out_953892830646151789[91] = 0;
   out_953892830646151789[92] = 0;
   out_953892830646151789[93] = 0;
   out_953892830646151789[94] = 0;
   out_953892830646151789[95] = 1;
   out_953892830646151789[96] = 0;
   out_953892830646151789[97] = 0;
   out_953892830646151789[98] = 0;
   out_953892830646151789[99] = 0;
   out_953892830646151789[100] = 0;
   out_953892830646151789[101] = 0;
   out_953892830646151789[102] = 0;
   out_953892830646151789[103] = 0;
   out_953892830646151789[104] = dt;
   out_953892830646151789[105] = 0;
   out_953892830646151789[106] = 0;
   out_953892830646151789[107] = 0;
   out_953892830646151789[108] = 0;
   out_953892830646151789[109] = 0;
   out_953892830646151789[110] = 0;
   out_953892830646151789[111] = 0;
   out_953892830646151789[112] = 0;
   out_953892830646151789[113] = 0;
   out_953892830646151789[114] = 1;
   out_953892830646151789[115] = 0;
   out_953892830646151789[116] = 0;
   out_953892830646151789[117] = 0;
   out_953892830646151789[118] = 0;
   out_953892830646151789[119] = 0;
   out_953892830646151789[120] = 0;
   out_953892830646151789[121] = 0;
   out_953892830646151789[122] = 0;
   out_953892830646151789[123] = 0;
   out_953892830646151789[124] = 0;
   out_953892830646151789[125] = 0;
   out_953892830646151789[126] = 0;
   out_953892830646151789[127] = 0;
   out_953892830646151789[128] = 0;
   out_953892830646151789[129] = 0;
   out_953892830646151789[130] = 0;
   out_953892830646151789[131] = 0;
   out_953892830646151789[132] = 0;
   out_953892830646151789[133] = 1;
   out_953892830646151789[134] = 0;
   out_953892830646151789[135] = 0;
   out_953892830646151789[136] = 0;
   out_953892830646151789[137] = 0;
   out_953892830646151789[138] = 0;
   out_953892830646151789[139] = 0;
   out_953892830646151789[140] = 0;
   out_953892830646151789[141] = 0;
   out_953892830646151789[142] = 0;
   out_953892830646151789[143] = 0;
   out_953892830646151789[144] = 0;
   out_953892830646151789[145] = 0;
   out_953892830646151789[146] = 0;
   out_953892830646151789[147] = 0;
   out_953892830646151789[148] = 0;
   out_953892830646151789[149] = 0;
   out_953892830646151789[150] = 0;
   out_953892830646151789[151] = 0;
   out_953892830646151789[152] = 1;
   out_953892830646151789[153] = 0;
   out_953892830646151789[154] = 0;
   out_953892830646151789[155] = 0;
   out_953892830646151789[156] = 0;
   out_953892830646151789[157] = 0;
   out_953892830646151789[158] = 0;
   out_953892830646151789[159] = 0;
   out_953892830646151789[160] = 0;
   out_953892830646151789[161] = 0;
   out_953892830646151789[162] = 0;
   out_953892830646151789[163] = 0;
   out_953892830646151789[164] = 0;
   out_953892830646151789[165] = 0;
   out_953892830646151789[166] = 0;
   out_953892830646151789[167] = 0;
   out_953892830646151789[168] = 0;
   out_953892830646151789[169] = 0;
   out_953892830646151789[170] = 0;
   out_953892830646151789[171] = 1;
   out_953892830646151789[172] = 0;
   out_953892830646151789[173] = 0;
   out_953892830646151789[174] = 0;
   out_953892830646151789[175] = 0;
   out_953892830646151789[176] = 0;
   out_953892830646151789[177] = 0;
   out_953892830646151789[178] = 0;
   out_953892830646151789[179] = 0;
   out_953892830646151789[180] = 0;
   out_953892830646151789[181] = 0;
   out_953892830646151789[182] = 0;
   out_953892830646151789[183] = 0;
   out_953892830646151789[184] = 0;
   out_953892830646151789[185] = 0;
   out_953892830646151789[186] = 0;
   out_953892830646151789[187] = 0;
   out_953892830646151789[188] = 0;
   out_953892830646151789[189] = 0;
   out_953892830646151789[190] = 1;
   out_953892830646151789[191] = 0;
   out_953892830646151789[192] = 0;
   out_953892830646151789[193] = 0;
   out_953892830646151789[194] = 0;
   out_953892830646151789[195] = 0;
   out_953892830646151789[196] = 0;
   out_953892830646151789[197] = 0;
   out_953892830646151789[198] = 0;
   out_953892830646151789[199] = 0;
   out_953892830646151789[200] = 0;
   out_953892830646151789[201] = 0;
   out_953892830646151789[202] = 0;
   out_953892830646151789[203] = 0;
   out_953892830646151789[204] = 0;
   out_953892830646151789[205] = 0;
   out_953892830646151789[206] = 0;
   out_953892830646151789[207] = 0;
   out_953892830646151789[208] = 0;
   out_953892830646151789[209] = 1;
   out_953892830646151789[210] = 0;
   out_953892830646151789[211] = 0;
   out_953892830646151789[212] = 0;
   out_953892830646151789[213] = 0;
   out_953892830646151789[214] = 0;
   out_953892830646151789[215] = 0;
   out_953892830646151789[216] = 0;
   out_953892830646151789[217] = 0;
   out_953892830646151789[218] = 0;
   out_953892830646151789[219] = 0;
   out_953892830646151789[220] = 0;
   out_953892830646151789[221] = 0;
   out_953892830646151789[222] = 0;
   out_953892830646151789[223] = 0;
   out_953892830646151789[224] = 0;
   out_953892830646151789[225] = 0;
   out_953892830646151789[226] = 0;
   out_953892830646151789[227] = 0;
   out_953892830646151789[228] = 1;
   out_953892830646151789[229] = 0;
   out_953892830646151789[230] = 0;
   out_953892830646151789[231] = 0;
   out_953892830646151789[232] = 0;
   out_953892830646151789[233] = 0;
   out_953892830646151789[234] = 0;
   out_953892830646151789[235] = 0;
   out_953892830646151789[236] = 0;
   out_953892830646151789[237] = 0;
   out_953892830646151789[238] = 0;
   out_953892830646151789[239] = 0;
   out_953892830646151789[240] = 0;
   out_953892830646151789[241] = 0;
   out_953892830646151789[242] = 0;
   out_953892830646151789[243] = 0;
   out_953892830646151789[244] = 0;
   out_953892830646151789[245] = 0;
   out_953892830646151789[246] = 0;
   out_953892830646151789[247] = 1;
   out_953892830646151789[248] = 0;
   out_953892830646151789[249] = 0;
   out_953892830646151789[250] = 0;
   out_953892830646151789[251] = 0;
   out_953892830646151789[252] = 0;
   out_953892830646151789[253] = 0;
   out_953892830646151789[254] = 0;
   out_953892830646151789[255] = 0;
   out_953892830646151789[256] = 0;
   out_953892830646151789[257] = 0;
   out_953892830646151789[258] = 0;
   out_953892830646151789[259] = 0;
   out_953892830646151789[260] = 0;
   out_953892830646151789[261] = 0;
   out_953892830646151789[262] = 0;
   out_953892830646151789[263] = 0;
   out_953892830646151789[264] = 0;
   out_953892830646151789[265] = 0;
   out_953892830646151789[266] = 1;
   out_953892830646151789[267] = 0;
   out_953892830646151789[268] = 0;
   out_953892830646151789[269] = 0;
   out_953892830646151789[270] = 0;
   out_953892830646151789[271] = 0;
   out_953892830646151789[272] = 0;
   out_953892830646151789[273] = 0;
   out_953892830646151789[274] = 0;
   out_953892830646151789[275] = 0;
   out_953892830646151789[276] = 0;
   out_953892830646151789[277] = 0;
   out_953892830646151789[278] = 0;
   out_953892830646151789[279] = 0;
   out_953892830646151789[280] = 0;
   out_953892830646151789[281] = 0;
   out_953892830646151789[282] = 0;
   out_953892830646151789[283] = 0;
   out_953892830646151789[284] = 0;
   out_953892830646151789[285] = 1;
   out_953892830646151789[286] = 0;
   out_953892830646151789[287] = 0;
   out_953892830646151789[288] = 0;
   out_953892830646151789[289] = 0;
   out_953892830646151789[290] = 0;
   out_953892830646151789[291] = 0;
   out_953892830646151789[292] = 0;
   out_953892830646151789[293] = 0;
   out_953892830646151789[294] = 0;
   out_953892830646151789[295] = 0;
   out_953892830646151789[296] = 0;
   out_953892830646151789[297] = 0;
   out_953892830646151789[298] = 0;
   out_953892830646151789[299] = 0;
   out_953892830646151789[300] = 0;
   out_953892830646151789[301] = 0;
   out_953892830646151789[302] = 0;
   out_953892830646151789[303] = 0;
   out_953892830646151789[304] = 1;
   out_953892830646151789[305] = 0;
   out_953892830646151789[306] = 0;
   out_953892830646151789[307] = 0;
   out_953892830646151789[308] = 0;
   out_953892830646151789[309] = 0;
   out_953892830646151789[310] = 0;
   out_953892830646151789[311] = 0;
   out_953892830646151789[312] = 0;
   out_953892830646151789[313] = 0;
   out_953892830646151789[314] = 0;
   out_953892830646151789[315] = 0;
   out_953892830646151789[316] = 0;
   out_953892830646151789[317] = 0;
   out_953892830646151789[318] = 0;
   out_953892830646151789[319] = 0;
   out_953892830646151789[320] = 0;
   out_953892830646151789[321] = 0;
   out_953892830646151789[322] = 0;
   out_953892830646151789[323] = 1;
}
void h_4(double *state, double *unused, double *out_44459550781696338) {
   out_44459550781696338[0] = state[6] + state[9];
   out_44459550781696338[1] = state[7] + state[10];
   out_44459550781696338[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_79044024102819006) {
   out_79044024102819006[0] = 0;
   out_79044024102819006[1] = 0;
   out_79044024102819006[2] = 0;
   out_79044024102819006[3] = 0;
   out_79044024102819006[4] = 0;
   out_79044024102819006[5] = 0;
   out_79044024102819006[6] = 1;
   out_79044024102819006[7] = 0;
   out_79044024102819006[8] = 0;
   out_79044024102819006[9] = 1;
   out_79044024102819006[10] = 0;
   out_79044024102819006[11] = 0;
   out_79044024102819006[12] = 0;
   out_79044024102819006[13] = 0;
   out_79044024102819006[14] = 0;
   out_79044024102819006[15] = 0;
   out_79044024102819006[16] = 0;
   out_79044024102819006[17] = 0;
   out_79044024102819006[18] = 0;
   out_79044024102819006[19] = 0;
   out_79044024102819006[20] = 0;
   out_79044024102819006[21] = 0;
   out_79044024102819006[22] = 0;
   out_79044024102819006[23] = 0;
   out_79044024102819006[24] = 0;
   out_79044024102819006[25] = 1;
   out_79044024102819006[26] = 0;
   out_79044024102819006[27] = 0;
   out_79044024102819006[28] = 1;
   out_79044024102819006[29] = 0;
   out_79044024102819006[30] = 0;
   out_79044024102819006[31] = 0;
   out_79044024102819006[32] = 0;
   out_79044024102819006[33] = 0;
   out_79044024102819006[34] = 0;
   out_79044024102819006[35] = 0;
   out_79044024102819006[36] = 0;
   out_79044024102819006[37] = 0;
   out_79044024102819006[38] = 0;
   out_79044024102819006[39] = 0;
   out_79044024102819006[40] = 0;
   out_79044024102819006[41] = 0;
   out_79044024102819006[42] = 0;
   out_79044024102819006[43] = 0;
   out_79044024102819006[44] = 1;
   out_79044024102819006[45] = 0;
   out_79044024102819006[46] = 0;
   out_79044024102819006[47] = 1;
   out_79044024102819006[48] = 0;
   out_79044024102819006[49] = 0;
   out_79044024102819006[50] = 0;
   out_79044024102819006[51] = 0;
   out_79044024102819006[52] = 0;
   out_79044024102819006[53] = 0;
}
void h_10(double *state, double *unused, double *out_81618196104347189) {
   out_81618196104347189[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_81618196104347189[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_81618196104347189[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4021134994901400003) {
   out_4021134994901400003[0] = 0;
   out_4021134994901400003[1] = 9.8100000000000005*cos(state[1]);
   out_4021134994901400003[2] = 0;
   out_4021134994901400003[3] = 0;
   out_4021134994901400003[4] = -state[8];
   out_4021134994901400003[5] = state[7];
   out_4021134994901400003[6] = 0;
   out_4021134994901400003[7] = state[5];
   out_4021134994901400003[8] = -state[4];
   out_4021134994901400003[9] = 0;
   out_4021134994901400003[10] = 0;
   out_4021134994901400003[11] = 0;
   out_4021134994901400003[12] = 1;
   out_4021134994901400003[13] = 0;
   out_4021134994901400003[14] = 0;
   out_4021134994901400003[15] = 1;
   out_4021134994901400003[16] = 0;
   out_4021134994901400003[17] = 0;
   out_4021134994901400003[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4021134994901400003[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4021134994901400003[20] = 0;
   out_4021134994901400003[21] = state[8];
   out_4021134994901400003[22] = 0;
   out_4021134994901400003[23] = -state[6];
   out_4021134994901400003[24] = -state[5];
   out_4021134994901400003[25] = 0;
   out_4021134994901400003[26] = state[3];
   out_4021134994901400003[27] = 0;
   out_4021134994901400003[28] = 0;
   out_4021134994901400003[29] = 0;
   out_4021134994901400003[30] = 0;
   out_4021134994901400003[31] = 1;
   out_4021134994901400003[32] = 0;
   out_4021134994901400003[33] = 0;
   out_4021134994901400003[34] = 1;
   out_4021134994901400003[35] = 0;
   out_4021134994901400003[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4021134994901400003[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4021134994901400003[38] = 0;
   out_4021134994901400003[39] = -state[7];
   out_4021134994901400003[40] = state[6];
   out_4021134994901400003[41] = 0;
   out_4021134994901400003[42] = state[4];
   out_4021134994901400003[43] = -state[3];
   out_4021134994901400003[44] = 0;
   out_4021134994901400003[45] = 0;
   out_4021134994901400003[46] = 0;
   out_4021134994901400003[47] = 0;
   out_4021134994901400003[48] = 0;
   out_4021134994901400003[49] = 0;
   out_4021134994901400003[50] = 1;
   out_4021134994901400003[51] = 0;
   out_4021134994901400003[52] = 0;
   out_4021134994901400003[53] = 1;
}
void h_13(double *state, double *unused, double *out_3818565266702701970) {
   out_3818565266702701970[0] = state[3];
   out_3818565266702701970[1] = state[4];
   out_3818565266702701970[2] = state[5];
}
void H_13(double *state, double *unused, double *out_3133229801229513795) {
   out_3133229801229513795[0] = 0;
   out_3133229801229513795[1] = 0;
   out_3133229801229513795[2] = 0;
   out_3133229801229513795[3] = 1;
   out_3133229801229513795[4] = 0;
   out_3133229801229513795[5] = 0;
   out_3133229801229513795[6] = 0;
   out_3133229801229513795[7] = 0;
   out_3133229801229513795[8] = 0;
   out_3133229801229513795[9] = 0;
   out_3133229801229513795[10] = 0;
   out_3133229801229513795[11] = 0;
   out_3133229801229513795[12] = 0;
   out_3133229801229513795[13] = 0;
   out_3133229801229513795[14] = 0;
   out_3133229801229513795[15] = 0;
   out_3133229801229513795[16] = 0;
   out_3133229801229513795[17] = 0;
   out_3133229801229513795[18] = 0;
   out_3133229801229513795[19] = 0;
   out_3133229801229513795[20] = 0;
   out_3133229801229513795[21] = 0;
   out_3133229801229513795[22] = 1;
   out_3133229801229513795[23] = 0;
   out_3133229801229513795[24] = 0;
   out_3133229801229513795[25] = 0;
   out_3133229801229513795[26] = 0;
   out_3133229801229513795[27] = 0;
   out_3133229801229513795[28] = 0;
   out_3133229801229513795[29] = 0;
   out_3133229801229513795[30] = 0;
   out_3133229801229513795[31] = 0;
   out_3133229801229513795[32] = 0;
   out_3133229801229513795[33] = 0;
   out_3133229801229513795[34] = 0;
   out_3133229801229513795[35] = 0;
   out_3133229801229513795[36] = 0;
   out_3133229801229513795[37] = 0;
   out_3133229801229513795[38] = 0;
   out_3133229801229513795[39] = 0;
   out_3133229801229513795[40] = 0;
   out_3133229801229513795[41] = 1;
   out_3133229801229513795[42] = 0;
   out_3133229801229513795[43] = 0;
   out_3133229801229513795[44] = 0;
   out_3133229801229513795[45] = 0;
   out_3133229801229513795[46] = 0;
   out_3133229801229513795[47] = 0;
   out_3133229801229513795[48] = 0;
   out_3133229801229513795[49] = 0;
   out_3133229801229513795[50] = 0;
   out_3133229801229513795[51] = 0;
   out_3133229801229513795[52] = 0;
   out_3133229801229513795[53] = 0;
}
void h_14(double *state, double *unused, double *out_1807207524562730417) {
   out_1807207524562730417[0] = state[6];
   out_1807207524562730417[1] = state[7];
   out_1807207524562730417[2] = state[8];
}
void H_14(double *state, double *unused, double *out_3161832456398191302) {
   out_3161832456398191302[0] = 0;
   out_3161832456398191302[1] = 0;
   out_3161832456398191302[2] = 0;
   out_3161832456398191302[3] = 0;
   out_3161832456398191302[4] = 0;
   out_3161832456398191302[5] = 0;
   out_3161832456398191302[6] = 1;
   out_3161832456398191302[7] = 0;
   out_3161832456398191302[8] = 0;
   out_3161832456398191302[9] = 0;
   out_3161832456398191302[10] = 0;
   out_3161832456398191302[11] = 0;
   out_3161832456398191302[12] = 0;
   out_3161832456398191302[13] = 0;
   out_3161832456398191302[14] = 0;
   out_3161832456398191302[15] = 0;
   out_3161832456398191302[16] = 0;
   out_3161832456398191302[17] = 0;
   out_3161832456398191302[18] = 0;
   out_3161832456398191302[19] = 0;
   out_3161832456398191302[20] = 0;
   out_3161832456398191302[21] = 0;
   out_3161832456398191302[22] = 0;
   out_3161832456398191302[23] = 0;
   out_3161832456398191302[24] = 0;
   out_3161832456398191302[25] = 1;
   out_3161832456398191302[26] = 0;
   out_3161832456398191302[27] = 0;
   out_3161832456398191302[28] = 0;
   out_3161832456398191302[29] = 0;
   out_3161832456398191302[30] = 0;
   out_3161832456398191302[31] = 0;
   out_3161832456398191302[32] = 0;
   out_3161832456398191302[33] = 0;
   out_3161832456398191302[34] = 0;
   out_3161832456398191302[35] = 0;
   out_3161832456398191302[36] = 0;
   out_3161832456398191302[37] = 0;
   out_3161832456398191302[38] = 0;
   out_3161832456398191302[39] = 0;
   out_3161832456398191302[40] = 0;
   out_3161832456398191302[41] = 0;
   out_3161832456398191302[42] = 0;
   out_3161832456398191302[43] = 0;
   out_3161832456398191302[44] = 1;
   out_3161832456398191302[45] = 0;
   out_3161832456398191302[46] = 0;
   out_3161832456398191302[47] = 0;
   out_3161832456398191302[48] = 0;
   out_3161832456398191302[49] = 0;
   out_3161832456398191302[50] = 0;
   out_3161832456398191302[51] = 0;
   out_3161832456398191302[52] = 0;
   out_3161832456398191302[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_348428071618360721) {
  err_fun(nom_x, delta_x, out_348428071618360721);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2713115070426494842) {
  inv_err_fun(nom_x, true_x, out_2713115070426494842);
}
void pose_H_mod_fun(double *state, double *out_1740654701803477855) {
  H_mod_fun(state, out_1740654701803477855);
}
void pose_f_fun(double *state, double dt, double *out_7474451683258418033) {
  f_fun(state,  dt, out_7474451683258418033);
}
void pose_F_fun(double *state, double dt, double *out_953892830646151789) {
  F_fun(state,  dt, out_953892830646151789);
}
void pose_h_4(double *state, double *unused, double *out_44459550781696338) {
  h_4(state, unused, out_44459550781696338);
}
void pose_H_4(double *state, double *unused, double *out_79044024102819006) {
  H_4(state, unused, out_79044024102819006);
}
void pose_h_10(double *state, double *unused, double *out_81618196104347189) {
  h_10(state, unused, out_81618196104347189);
}
void pose_H_10(double *state, double *unused, double *out_4021134994901400003) {
  H_10(state, unused, out_4021134994901400003);
}
void pose_h_13(double *state, double *unused, double *out_3818565266702701970) {
  h_13(state, unused, out_3818565266702701970);
}
void pose_H_13(double *state, double *unused, double *out_3133229801229513795) {
  H_13(state, unused, out_3133229801229513795);
}
void pose_h_14(double *state, double *unused, double *out_1807207524562730417) {
  h_14(state, unused, out_1807207524562730417);
}
void pose_H_14(double *state, double *unused, double *out_3161832456398191302) {
  H_14(state, unused, out_3161832456398191302);
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
