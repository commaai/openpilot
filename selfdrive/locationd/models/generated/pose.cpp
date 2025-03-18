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
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_3867314615092444191) {
   out_3867314615092444191[0] = delta_x[0] + nom_x[0];
   out_3867314615092444191[1] = delta_x[1] + nom_x[1];
   out_3867314615092444191[2] = delta_x[2] + nom_x[2];
   out_3867314615092444191[3] = delta_x[3] + nom_x[3];
   out_3867314615092444191[4] = delta_x[4] + nom_x[4];
   out_3867314615092444191[5] = delta_x[5] + nom_x[5];
   out_3867314615092444191[6] = delta_x[6] + nom_x[6];
   out_3867314615092444191[7] = delta_x[7] + nom_x[7];
   out_3867314615092444191[8] = delta_x[8] + nom_x[8];
   out_3867314615092444191[9] = delta_x[9] + nom_x[9];
   out_3867314615092444191[10] = delta_x[10] + nom_x[10];
   out_3867314615092444191[11] = delta_x[11] + nom_x[11];
   out_3867314615092444191[12] = delta_x[12] + nom_x[12];
   out_3867314615092444191[13] = delta_x[13] + nom_x[13];
   out_3867314615092444191[14] = delta_x[14] + nom_x[14];
   out_3867314615092444191[15] = delta_x[15] + nom_x[15];
   out_3867314615092444191[16] = delta_x[16] + nom_x[16];
   out_3867314615092444191[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_357340291324325113) {
   out_357340291324325113[0] = -nom_x[0] + true_x[0];
   out_357340291324325113[1] = -nom_x[1] + true_x[1];
   out_357340291324325113[2] = -nom_x[2] + true_x[2];
   out_357340291324325113[3] = -nom_x[3] + true_x[3];
   out_357340291324325113[4] = -nom_x[4] + true_x[4];
   out_357340291324325113[5] = -nom_x[5] + true_x[5];
   out_357340291324325113[6] = -nom_x[6] + true_x[6];
   out_357340291324325113[7] = -nom_x[7] + true_x[7];
   out_357340291324325113[8] = -nom_x[8] + true_x[8];
   out_357340291324325113[9] = -nom_x[9] + true_x[9];
   out_357340291324325113[10] = -nom_x[10] + true_x[10];
   out_357340291324325113[11] = -nom_x[11] + true_x[11];
   out_357340291324325113[12] = -nom_x[12] + true_x[12];
   out_357340291324325113[13] = -nom_x[13] + true_x[13];
   out_357340291324325113[14] = -nom_x[14] + true_x[14];
   out_357340291324325113[15] = -nom_x[15] + true_x[15];
   out_357340291324325113[16] = -nom_x[16] + true_x[16];
   out_357340291324325113[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_5335377228813010670) {
   out_5335377228813010670[0] = 1.0;
   out_5335377228813010670[1] = 0;
   out_5335377228813010670[2] = 0;
   out_5335377228813010670[3] = 0;
   out_5335377228813010670[4] = 0;
   out_5335377228813010670[5] = 0;
   out_5335377228813010670[6] = 0;
   out_5335377228813010670[7] = 0;
   out_5335377228813010670[8] = 0;
   out_5335377228813010670[9] = 0;
   out_5335377228813010670[10] = 0;
   out_5335377228813010670[11] = 0;
   out_5335377228813010670[12] = 0;
   out_5335377228813010670[13] = 0;
   out_5335377228813010670[14] = 0;
   out_5335377228813010670[15] = 0;
   out_5335377228813010670[16] = 0;
   out_5335377228813010670[17] = 0;
   out_5335377228813010670[18] = 0;
   out_5335377228813010670[19] = 1.0;
   out_5335377228813010670[20] = 0;
   out_5335377228813010670[21] = 0;
   out_5335377228813010670[22] = 0;
   out_5335377228813010670[23] = 0;
   out_5335377228813010670[24] = 0;
   out_5335377228813010670[25] = 0;
   out_5335377228813010670[26] = 0;
   out_5335377228813010670[27] = 0;
   out_5335377228813010670[28] = 0;
   out_5335377228813010670[29] = 0;
   out_5335377228813010670[30] = 0;
   out_5335377228813010670[31] = 0;
   out_5335377228813010670[32] = 0;
   out_5335377228813010670[33] = 0;
   out_5335377228813010670[34] = 0;
   out_5335377228813010670[35] = 0;
   out_5335377228813010670[36] = 0;
   out_5335377228813010670[37] = 0;
   out_5335377228813010670[38] = 1.0;
   out_5335377228813010670[39] = 0;
   out_5335377228813010670[40] = 0;
   out_5335377228813010670[41] = 0;
   out_5335377228813010670[42] = 0;
   out_5335377228813010670[43] = 0;
   out_5335377228813010670[44] = 0;
   out_5335377228813010670[45] = 0;
   out_5335377228813010670[46] = 0;
   out_5335377228813010670[47] = 0;
   out_5335377228813010670[48] = 0;
   out_5335377228813010670[49] = 0;
   out_5335377228813010670[50] = 0;
   out_5335377228813010670[51] = 0;
   out_5335377228813010670[52] = 0;
   out_5335377228813010670[53] = 0;
   out_5335377228813010670[54] = 0;
   out_5335377228813010670[55] = 0;
   out_5335377228813010670[56] = 0;
   out_5335377228813010670[57] = 1.0;
   out_5335377228813010670[58] = 0;
   out_5335377228813010670[59] = 0;
   out_5335377228813010670[60] = 0;
   out_5335377228813010670[61] = 0;
   out_5335377228813010670[62] = 0;
   out_5335377228813010670[63] = 0;
   out_5335377228813010670[64] = 0;
   out_5335377228813010670[65] = 0;
   out_5335377228813010670[66] = 0;
   out_5335377228813010670[67] = 0;
   out_5335377228813010670[68] = 0;
   out_5335377228813010670[69] = 0;
   out_5335377228813010670[70] = 0;
   out_5335377228813010670[71] = 0;
   out_5335377228813010670[72] = 0;
   out_5335377228813010670[73] = 0;
   out_5335377228813010670[74] = 0;
   out_5335377228813010670[75] = 0;
   out_5335377228813010670[76] = 1.0;
   out_5335377228813010670[77] = 0;
   out_5335377228813010670[78] = 0;
   out_5335377228813010670[79] = 0;
   out_5335377228813010670[80] = 0;
   out_5335377228813010670[81] = 0;
   out_5335377228813010670[82] = 0;
   out_5335377228813010670[83] = 0;
   out_5335377228813010670[84] = 0;
   out_5335377228813010670[85] = 0;
   out_5335377228813010670[86] = 0;
   out_5335377228813010670[87] = 0;
   out_5335377228813010670[88] = 0;
   out_5335377228813010670[89] = 0;
   out_5335377228813010670[90] = 0;
   out_5335377228813010670[91] = 0;
   out_5335377228813010670[92] = 0;
   out_5335377228813010670[93] = 0;
   out_5335377228813010670[94] = 0;
   out_5335377228813010670[95] = 1.0;
   out_5335377228813010670[96] = 0;
   out_5335377228813010670[97] = 0;
   out_5335377228813010670[98] = 0;
   out_5335377228813010670[99] = 0;
   out_5335377228813010670[100] = 0;
   out_5335377228813010670[101] = 0;
   out_5335377228813010670[102] = 0;
   out_5335377228813010670[103] = 0;
   out_5335377228813010670[104] = 0;
   out_5335377228813010670[105] = 0;
   out_5335377228813010670[106] = 0;
   out_5335377228813010670[107] = 0;
   out_5335377228813010670[108] = 0;
   out_5335377228813010670[109] = 0;
   out_5335377228813010670[110] = 0;
   out_5335377228813010670[111] = 0;
   out_5335377228813010670[112] = 0;
   out_5335377228813010670[113] = 0;
   out_5335377228813010670[114] = 1.0;
   out_5335377228813010670[115] = 0;
   out_5335377228813010670[116] = 0;
   out_5335377228813010670[117] = 0;
   out_5335377228813010670[118] = 0;
   out_5335377228813010670[119] = 0;
   out_5335377228813010670[120] = 0;
   out_5335377228813010670[121] = 0;
   out_5335377228813010670[122] = 0;
   out_5335377228813010670[123] = 0;
   out_5335377228813010670[124] = 0;
   out_5335377228813010670[125] = 0;
   out_5335377228813010670[126] = 0;
   out_5335377228813010670[127] = 0;
   out_5335377228813010670[128] = 0;
   out_5335377228813010670[129] = 0;
   out_5335377228813010670[130] = 0;
   out_5335377228813010670[131] = 0;
   out_5335377228813010670[132] = 0;
   out_5335377228813010670[133] = 1.0;
   out_5335377228813010670[134] = 0;
   out_5335377228813010670[135] = 0;
   out_5335377228813010670[136] = 0;
   out_5335377228813010670[137] = 0;
   out_5335377228813010670[138] = 0;
   out_5335377228813010670[139] = 0;
   out_5335377228813010670[140] = 0;
   out_5335377228813010670[141] = 0;
   out_5335377228813010670[142] = 0;
   out_5335377228813010670[143] = 0;
   out_5335377228813010670[144] = 0;
   out_5335377228813010670[145] = 0;
   out_5335377228813010670[146] = 0;
   out_5335377228813010670[147] = 0;
   out_5335377228813010670[148] = 0;
   out_5335377228813010670[149] = 0;
   out_5335377228813010670[150] = 0;
   out_5335377228813010670[151] = 0;
   out_5335377228813010670[152] = 1.0;
   out_5335377228813010670[153] = 0;
   out_5335377228813010670[154] = 0;
   out_5335377228813010670[155] = 0;
   out_5335377228813010670[156] = 0;
   out_5335377228813010670[157] = 0;
   out_5335377228813010670[158] = 0;
   out_5335377228813010670[159] = 0;
   out_5335377228813010670[160] = 0;
   out_5335377228813010670[161] = 0;
   out_5335377228813010670[162] = 0;
   out_5335377228813010670[163] = 0;
   out_5335377228813010670[164] = 0;
   out_5335377228813010670[165] = 0;
   out_5335377228813010670[166] = 0;
   out_5335377228813010670[167] = 0;
   out_5335377228813010670[168] = 0;
   out_5335377228813010670[169] = 0;
   out_5335377228813010670[170] = 0;
   out_5335377228813010670[171] = 1.0;
   out_5335377228813010670[172] = 0;
   out_5335377228813010670[173] = 0;
   out_5335377228813010670[174] = 0;
   out_5335377228813010670[175] = 0;
   out_5335377228813010670[176] = 0;
   out_5335377228813010670[177] = 0;
   out_5335377228813010670[178] = 0;
   out_5335377228813010670[179] = 0;
   out_5335377228813010670[180] = 0;
   out_5335377228813010670[181] = 0;
   out_5335377228813010670[182] = 0;
   out_5335377228813010670[183] = 0;
   out_5335377228813010670[184] = 0;
   out_5335377228813010670[185] = 0;
   out_5335377228813010670[186] = 0;
   out_5335377228813010670[187] = 0;
   out_5335377228813010670[188] = 0;
   out_5335377228813010670[189] = 0;
   out_5335377228813010670[190] = 1.0;
   out_5335377228813010670[191] = 0;
   out_5335377228813010670[192] = 0;
   out_5335377228813010670[193] = 0;
   out_5335377228813010670[194] = 0;
   out_5335377228813010670[195] = 0;
   out_5335377228813010670[196] = 0;
   out_5335377228813010670[197] = 0;
   out_5335377228813010670[198] = 0;
   out_5335377228813010670[199] = 0;
   out_5335377228813010670[200] = 0;
   out_5335377228813010670[201] = 0;
   out_5335377228813010670[202] = 0;
   out_5335377228813010670[203] = 0;
   out_5335377228813010670[204] = 0;
   out_5335377228813010670[205] = 0;
   out_5335377228813010670[206] = 0;
   out_5335377228813010670[207] = 0;
   out_5335377228813010670[208] = 0;
   out_5335377228813010670[209] = 1.0;
   out_5335377228813010670[210] = 0;
   out_5335377228813010670[211] = 0;
   out_5335377228813010670[212] = 0;
   out_5335377228813010670[213] = 0;
   out_5335377228813010670[214] = 0;
   out_5335377228813010670[215] = 0;
   out_5335377228813010670[216] = 0;
   out_5335377228813010670[217] = 0;
   out_5335377228813010670[218] = 0;
   out_5335377228813010670[219] = 0;
   out_5335377228813010670[220] = 0;
   out_5335377228813010670[221] = 0;
   out_5335377228813010670[222] = 0;
   out_5335377228813010670[223] = 0;
   out_5335377228813010670[224] = 0;
   out_5335377228813010670[225] = 0;
   out_5335377228813010670[226] = 0;
   out_5335377228813010670[227] = 0;
   out_5335377228813010670[228] = 1.0;
   out_5335377228813010670[229] = 0;
   out_5335377228813010670[230] = 0;
   out_5335377228813010670[231] = 0;
   out_5335377228813010670[232] = 0;
   out_5335377228813010670[233] = 0;
   out_5335377228813010670[234] = 0;
   out_5335377228813010670[235] = 0;
   out_5335377228813010670[236] = 0;
   out_5335377228813010670[237] = 0;
   out_5335377228813010670[238] = 0;
   out_5335377228813010670[239] = 0;
   out_5335377228813010670[240] = 0;
   out_5335377228813010670[241] = 0;
   out_5335377228813010670[242] = 0;
   out_5335377228813010670[243] = 0;
   out_5335377228813010670[244] = 0;
   out_5335377228813010670[245] = 0;
   out_5335377228813010670[246] = 0;
   out_5335377228813010670[247] = 1.0;
   out_5335377228813010670[248] = 0;
   out_5335377228813010670[249] = 0;
   out_5335377228813010670[250] = 0;
   out_5335377228813010670[251] = 0;
   out_5335377228813010670[252] = 0;
   out_5335377228813010670[253] = 0;
   out_5335377228813010670[254] = 0;
   out_5335377228813010670[255] = 0;
   out_5335377228813010670[256] = 0;
   out_5335377228813010670[257] = 0;
   out_5335377228813010670[258] = 0;
   out_5335377228813010670[259] = 0;
   out_5335377228813010670[260] = 0;
   out_5335377228813010670[261] = 0;
   out_5335377228813010670[262] = 0;
   out_5335377228813010670[263] = 0;
   out_5335377228813010670[264] = 0;
   out_5335377228813010670[265] = 0;
   out_5335377228813010670[266] = 1.0;
   out_5335377228813010670[267] = 0;
   out_5335377228813010670[268] = 0;
   out_5335377228813010670[269] = 0;
   out_5335377228813010670[270] = 0;
   out_5335377228813010670[271] = 0;
   out_5335377228813010670[272] = 0;
   out_5335377228813010670[273] = 0;
   out_5335377228813010670[274] = 0;
   out_5335377228813010670[275] = 0;
   out_5335377228813010670[276] = 0;
   out_5335377228813010670[277] = 0;
   out_5335377228813010670[278] = 0;
   out_5335377228813010670[279] = 0;
   out_5335377228813010670[280] = 0;
   out_5335377228813010670[281] = 0;
   out_5335377228813010670[282] = 0;
   out_5335377228813010670[283] = 0;
   out_5335377228813010670[284] = 0;
   out_5335377228813010670[285] = 1.0;
   out_5335377228813010670[286] = 0;
   out_5335377228813010670[287] = 0;
   out_5335377228813010670[288] = 0;
   out_5335377228813010670[289] = 0;
   out_5335377228813010670[290] = 0;
   out_5335377228813010670[291] = 0;
   out_5335377228813010670[292] = 0;
   out_5335377228813010670[293] = 0;
   out_5335377228813010670[294] = 0;
   out_5335377228813010670[295] = 0;
   out_5335377228813010670[296] = 0;
   out_5335377228813010670[297] = 0;
   out_5335377228813010670[298] = 0;
   out_5335377228813010670[299] = 0;
   out_5335377228813010670[300] = 0;
   out_5335377228813010670[301] = 0;
   out_5335377228813010670[302] = 0;
   out_5335377228813010670[303] = 0;
   out_5335377228813010670[304] = 1.0;
   out_5335377228813010670[305] = 0;
   out_5335377228813010670[306] = 0;
   out_5335377228813010670[307] = 0;
   out_5335377228813010670[308] = 0;
   out_5335377228813010670[309] = 0;
   out_5335377228813010670[310] = 0;
   out_5335377228813010670[311] = 0;
   out_5335377228813010670[312] = 0;
   out_5335377228813010670[313] = 0;
   out_5335377228813010670[314] = 0;
   out_5335377228813010670[315] = 0;
   out_5335377228813010670[316] = 0;
   out_5335377228813010670[317] = 0;
   out_5335377228813010670[318] = 0;
   out_5335377228813010670[319] = 0;
   out_5335377228813010670[320] = 0;
   out_5335377228813010670[321] = 0;
   out_5335377228813010670[322] = 0;
   out_5335377228813010670[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8449822447276776350) {
   out_8449822447276776350[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8449822447276776350[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8449822447276776350[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8449822447276776350[3] = dt*state[12] + state[3];
   out_8449822447276776350[4] = dt*state[13] + state[4];
   out_8449822447276776350[5] = dt*state[14] + state[5];
   out_8449822447276776350[6] = state[6];
   out_8449822447276776350[7] = state[7];
   out_8449822447276776350[8] = state[8];
   out_8449822447276776350[9] = state[9];
   out_8449822447276776350[10] = state[10];
   out_8449822447276776350[11] = state[11];
   out_8449822447276776350[12] = state[12];
   out_8449822447276776350[13] = state[13];
   out_8449822447276776350[14] = state[14];
   out_8449822447276776350[15] = state[15];
   out_8449822447276776350[16] = state[16];
   out_8449822447276776350[17] = state[17];
}
void F_fun(double *state, double dt, double *out_8918868180555199161) {
   out_8918868180555199161[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8918868180555199161[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8918868180555199161[2] = 0;
   out_8918868180555199161[3] = 0;
   out_8918868180555199161[4] = 0;
   out_8918868180555199161[5] = 0;
   out_8918868180555199161[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8918868180555199161[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8918868180555199161[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8918868180555199161[9] = 0;
   out_8918868180555199161[10] = 0;
   out_8918868180555199161[11] = 0;
   out_8918868180555199161[12] = 0;
   out_8918868180555199161[13] = 0;
   out_8918868180555199161[14] = 0;
   out_8918868180555199161[15] = 0;
   out_8918868180555199161[16] = 0;
   out_8918868180555199161[17] = 0;
   out_8918868180555199161[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8918868180555199161[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8918868180555199161[20] = 0;
   out_8918868180555199161[21] = 0;
   out_8918868180555199161[22] = 0;
   out_8918868180555199161[23] = 0;
   out_8918868180555199161[24] = 0;
   out_8918868180555199161[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8918868180555199161[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8918868180555199161[27] = 0;
   out_8918868180555199161[28] = 0;
   out_8918868180555199161[29] = 0;
   out_8918868180555199161[30] = 0;
   out_8918868180555199161[31] = 0;
   out_8918868180555199161[32] = 0;
   out_8918868180555199161[33] = 0;
   out_8918868180555199161[34] = 0;
   out_8918868180555199161[35] = 0;
   out_8918868180555199161[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8918868180555199161[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8918868180555199161[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8918868180555199161[39] = 0;
   out_8918868180555199161[40] = 0;
   out_8918868180555199161[41] = 0;
   out_8918868180555199161[42] = 0;
   out_8918868180555199161[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8918868180555199161[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8918868180555199161[45] = 0;
   out_8918868180555199161[46] = 0;
   out_8918868180555199161[47] = 0;
   out_8918868180555199161[48] = 0;
   out_8918868180555199161[49] = 0;
   out_8918868180555199161[50] = 0;
   out_8918868180555199161[51] = 0;
   out_8918868180555199161[52] = 0;
   out_8918868180555199161[53] = 0;
   out_8918868180555199161[54] = 0;
   out_8918868180555199161[55] = 0;
   out_8918868180555199161[56] = 0;
   out_8918868180555199161[57] = 1;
   out_8918868180555199161[58] = 0;
   out_8918868180555199161[59] = 0;
   out_8918868180555199161[60] = 0;
   out_8918868180555199161[61] = 0;
   out_8918868180555199161[62] = 0;
   out_8918868180555199161[63] = 0;
   out_8918868180555199161[64] = 0;
   out_8918868180555199161[65] = 0;
   out_8918868180555199161[66] = dt;
   out_8918868180555199161[67] = 0;
   out_8918868180555199161[68] = 0;
   out_8918868180555199161[69] = 0;
   out_8918868180555199161[70] = 0;
   out_8918868180555199161[71] = 0;
   out_8918868180555199161[72] = 0;
   out_8918868180555199161[73] = 0;
   out_8918868180555199161[74] = 0;
   out_8918868180555199161[75] = 0;
   out_8918868180555199161[76] = 1;
   out_8918868180555199161[77] = 0;
   out_8918868180555199161[78] = 0;
   out_8918868180555199161[79] = 0;
   out_8918868180555199161[80] = 0;
   out_8918868180555199161[81] = 0;
   out_8918868180555199161[82] = 0;
   out_8918868180555199161[83] = 0;
   out_8918868180555199161[84] = 0;
   out_8918868180555199161[85] = dt;
   out_8918868180555199161[86] = 0;
   out_8918868180555199161[87] = 0;
   out_8918868180555199161[88] = 0;
   out_8918868180555199161[89] = 0;
   out_8918868180555199161[90] = 0;
   out_8918868180555199161[91] = 0;
   out_8918868180555199161[92] = 0;
   out_8918868180555199161[93] = 0;
   out_8918868180555199161[94] = 0;
   out_8918868180555199161[95] = 1;
   out_8918868180555199161[96] = 0;
   out_8918868180555199161[97] = 0;
   out_8918868180555199161[98] = 0;
   out_8918868180555199161[99] = 0;
   out_8918868180555199161[100] = 0;
   out_8918868180555199161[101] = 0;
   out_8918868180555199161[102] = 0;
   out_8918868180555199161[103] = 0;
   out_8918868180555199161[104] = dt;
   out_8918868180555199161[105] = 0;
   out_8918868180555199161[106] = 0;
   out_8918868180555199161[107] = 0;
   out_8918868180555199161[108] = 0;
   out_8918868180555199161[109] = 0;
   out_8918868180555199161[110] = 0;
   out_8918868180555199161[111] = 0;
   out_8918868180555199161[112] = 0;
   out_8918868180555199161[113] = 0;
   out_8918868180555199161[114] = 1;
   out_8918868180555199161[115] = 0;
   out_8918868180555199161[116] = 0;
   out_8918868180555199161[117] = 0;
   out_8918868180555199161[118] = 0;
   out_8918868180555199161[119] = 0;
   out_8918868180555199161[120] = 0;
   out_8918868180555199161[121] = 0;
   out_8918868180555199161[122] = 0;
   out_8918868180555199161[123] = 0;
   out_8918868180555199161[124] = 0;
   out_8918868180555199161[125] = 0;
   out_8918868180555199161[126] = 0;
   out_8918868180555199161[127] = 0;
   out_8918868180555199161[128] = 0;
   out_8918868180555199161[129] = 0;
   out_8918868180555199161[130] = 0;
   out_8918868180555199161[131] = 0;
   out_8918868180555199161[132] = 0;
   out_8918868180555199161[133] = 1;
   out_8918868180555199161[134] = 0;
   out_8918868180555199161[135] = 0;
   out_8918868180555199161[136] = 0;
   out_8918868180555199161[137] = 0;
   out_8918868180555199161[138] = 0;
   out_8918868180555199161[139] = 0;
   out_8918868180555199161[140] = 0;
   out_8918868180555199161[141] = 0;
   out_8918868180555199161[142] = 0;
   out_8918868180555199161[143] = 0;
   out_8918868180555199161[144] = 0;
   out_8918868180555199161[145] = 0;
   out_8918868180555199161[146] = 0;
   out_8918868180555199161[147] = 0;
   out_8918868180555199161[148] = 0;
   out_8918868180555199161[149] = 0;
   out_8918868180555199161[150] = 0;
   out_8918868180555199161[151] = 0;
   out_8918868180555199161[152] = 1;
   out_8918868180555199161[153] = 0;
   out_8918868180555199161[154] = 0;
   out_8918868180555199161[155] = 0;
   out_8918868180555199161[156] = 0;
   out_8918868180555199161[157] = 0;
   out_8918868180555199161[158] = 0;
   out_8918868180555199161[159] = 0;
   out_8918868180555199161[160] = 0;
   out_8918868180555199161[161] = 0;
   out_8918868180555199161[162] = 0;
   out_8918868180555199161[163] = 0;
   out_8918868180555199161[164] = 0;
   out_8918868180555199161[165] = 0;
   out_8918868180555199161[166] = 0;
   out_8918868180555199161[167] = 0;
   out_8918868180555199161[168] = 0;
   out_8918868180555199161[169] = 0;
   out_8918868180555199161[170] = 0;
   out_8918868180555199161[171] = 1;
   out_8918868180555199161[172] = 0;
   out_8918868180555199161[173] = 0;
   out_8918868180555199161[174] = 0;
   out_8918868180555199161[175] = 0;
   out_8918868180555199161[176] = 0;
   out_8918868180555199161[177] = 0;
   out_8918868180555199161[178] = 0;
   out_8918868180555199161[179] = 0;
   out_8918868180555199161[180] = 0;
   out_8918868180555199161[181] = 0;
   out_8918868180555199161[182] = 0;
   out_8918868180555199161[183] = 0;
   out_8918868180555199161[184] = 0;
   out_8918868180555199161[185] = 0;
   out_8918868180555199161[186] = 0;
   out_8918868180555199161[187] = 0;
   out_8918868180555199161[188] = 0;
   out_8918868180555199161[189] = 0;
   out_8918868180555199161[190] = 1;
   out_8918868180555199161[191] = 0;
   out_8918868180555199161[192] = 0;
   out_8918868180555199161[193] = 0;
   out_8918868180555199161[194] = 0;
   out_8918868180555199161[195] = 0;
   out_8918868180555199161[196] = 0;
   out_8918868180555199161[197] = 0;
   out_8918868180555199161[198] = 0;
   out_8918868180555199161[199] = 0;
   out_8918868180555199161[200] = 0;
   out_8918868180555199161[201] = 0;
   out_8918868180555199161[202] = 0;
   out_8918868180555199161[203] = 0;
   out_8918868180555199161[204] = 0;
   out_8918868180555199161[205] = 0;
   out_8918868180555199161[206] = 0;
   out_8918868180555199161[207] = 0;
   out_8918868180555199161[208] = 0;
   out_8918868180555199161[209] = 1;
   out_8918868180555199161[210] = 0;
   out_8918868180555199161[211] = 0;
   out_8918868180555199161[212] = 0;
   out_8918868180555199161[213] = 0;
   out_8918868180555199161[214] = 0;
   out_8918868180555199161[215] = 0;
   out_8918868180555199161[216] = 0;
   out_8918868180555199161[217] = 0;
   out_8918868180555199161[218] = 0;
   out_8918868180555199161[219] = 0;
   out_8918868180555199161[220] = 0;
   out_8918868180555199161[221] = 0;
   out_8918868180555199161[222] = 0;
   out_8918868180555199161[223] = 0;
   out_8918868180555199161[224] = 0;
   out_8918868180555199161[225] = 0;
   out_8918868180555199161[226] = 0;
   out_8918868180555199161[227] = 0;
   out_8918868180555199161[228] = 1;
   out_8918868180555199161[229] = 0;
   out_8918868180555199161[230] = 0;
   out_8918868180555199161[231] = 0;
   out_8918868180555199161[232] = 0;
   out_8918868180555199161[233] = 0;
   out_8918868180555199161[234] = 0;
   out_8918868180555199161[235] = 0;
   out_8918868180555199161[236] = 0;
   out_8918868180555199161[237] = 0;
   out_8918868180555199161[238] = 0;
   out_8918868180555199161[239] = 0;
   out_8918868180555199161[240] = 0;
   out_8918868180555199161[241] = 0;
   out_8918868180555199161[242] = 0;
   out_8918868180555199161[243] = 0;
   out_8918868180555199161[244] = 0;
   out_8918868180555199161[245] = 0;
   out_8918868180555199161[246] = 0;
   out_8918868180555199161[247] = 1;
   out_8918868180555199161[248] = 0;
   out_8918868180555199161[249] = 0;
   out_8918868180555199161[250] = 0;
   out_8918868180555199161[251] = 0;
   out_8918868180555199161[252] = 0;
   out_8918868180555199161[253] = 0;
   out_8918868180555199161[254] = 0;
   out_8918868180555199161[255] = 0;
   out_8918868180555199161[256] = 0;
   out_8918868180555199161[257] = 0;
   out_8918868180555199161[258] = 0;
   out_8918868180555199161[259] = 0;
   out_8918868180555199161[260] = 0;
   out_8918868180555199161[261] = 0;
   out_8918868180555199161[262] = 0;
   out_8918868180555199161[263] = 0;
   out_8918868180555199161[264] = 0;
   out_8918868180555199161[265] = 0;
   out_8918868180555199161[266] = 1;
   out_8918868180555199161[267] = 0;
   out_8918868180555199161[268] = 0;
   out_8918868180555199161[269] = 0;
   out_8918868180555199161[270] = 0;
   out_8918868180555199161[271] = 0;
   out_8918868180555199161[272] = 0;
   out_8918868180555199161[273] = 0;
   out_8918868180555199161[274] = 0;
   out_8918868180555199161[275] = 0;
   out_8918868180555199161[276] = 0;
   out_8918868180555199161[277] = 0;
   out_8918868180555199161[278] = 0;
   out_8918868180555199161[279] = 0;
   out_8918868180555199161[280] = 0;
   out_8918868180555199161[281] = 0;
   out_8918868180555199161[282] = 0;
   out_8918868180555199161[283] = 0;
   out_8918868180555199161[284] = 0;
   out_8918868180555199161[285] = 1;
   out_8918868180555199161[286] = 0;
   out_8918868180555199161[287] = 0;
   out_8918868180555199161[288] = 0;
   out_8918868180555199161[289] = 0;
   out_8918868180555199161[290] = 0;
   out_8918868180555199161[291] = 0;
   out_8918868180555199161[292] = 0;
   out_8918868180555199161[293] = 0;
   out_8918868180555199161[294] = 0;
   out_8918868180555199161[295] = 0;
   out_8918868180555199161[296] = 0;
   out_8918868180555199161[297] = 0;
   out_8918868180555199161[298] = 0;
   out_8918868180555199161[299] = 0;
   out_8918868180555199161[300] = 0;
   out_8918868180555199161[301] = 0;
   out_8918868180555199161[302] = 0;
   out_8918868180555199161[303] = 0;
   out_8918868180555199161[304] = 1;
   out_8918868180555199161[305] = 0;
   out_8918868180555199161[306] = 0;
   out_8918868180555199161[307] = 0;
   out_8918868180555199161[308] = 0;
   out_8918868180555199161[309] = 0;
   out_8918868180555199161[310] = 0;
   out_8918868180555199161[311] = 0;
   out_8918868180555199161[312] = 0;
   out_8918868180555199161[313] = 0;
   out_8918868180555199161[314] = 0;
   out_8918868180555199161[315] = 0;
   out_8918868180555199161[316] = 0;
   out_8918868180555199161[317] = 0;
   out_8918868180555199161[318] = 0;
   out_8918868180555199161[319] = 0;
   out_8918868180555199161[320] = 0;
   out_8918868180555199161[321] = 0;
   out_8918868180555199161[322] = 0;
   out_8918868180555199161[323] = 1;
}
void h_4(double *state, double *unused, double *out_6617669203308538531) {
   out_6617669203308538531[0] = state[6] + state[9];
   out_6617669203308538531[1] = state[7] + state[10];
   out_6617669203308538531[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_1875678236267692381) {
   out_1875678236267692381[0] = 0;
   out_1875678236267692381[1] = 0;
   out_1875678236267692381[2] = 0;
   out_1875678236267692381[3] = 0;
   out_1875678236267692381[4] = 0;
   out_1875678236267692381[5] = 0;
   out_1875678236267692381[6] = 1;
   out_1875678236267692381[7] = 0;
   out_1875678236267692381[8] = 0;
   out_1875678236267692381[9] = 1;
   out_1875678236267692381[10] = 0;
   out_1875678236267692381[11] = 0;
   out_1875678236267692381[12] = 0;
   out_1875678236267692381[13] = 0;
   out_1875678236267692381[14] = 0;
   out_1875678236267692381[15] = 0;
   out_1875678236267692381[16] = 0;
   out_1875678236267692381[17] = 0;
   out_1875678236267692381[18] = 0;
   out_1875678236267692381[19] = 0;
   out_1875678236267692381[20] = 0;
   out_1875678236267692381[21] = 0;
   out_1875678236267692381[22] = 0;
   out_1875678236267692381[23] = 0;
   out_1875678236267692381[24] = 0;
   out_1875678236267692381[25] = 1;
   out_1875678236267692381[26] = 0;
   out_1875678236267692381[27] = 0;
   out_1875678236267692381[28] = 1;
   out_1875678236267692381[29] = 0;
   out_1875678236267692381[30] = 0;
   out_1875678236267692381[31] = 0;
   out_1875678236267692381[32] = 0;
   out_1875678236267692381[33] = 0;
   out_1875678236267692381[34] = 0;
   out_1875678236267692381[35] = 0;
   out_1875678236267692381[36] = 0;
   out_1875678236267692381[37] = 0;
   out_1875678236267692381[38] = 0;
   out_1875678236267692381[39] = 0;
   out_1875678236267692381[40] = 0;
   out_1875678236267692381[41] = 0;
   out_1875678236267692381[42] = 0;
   out_1875678236267692381[43] = 0;
   out_1875678236267692381[44] = 1;
   out_1875678236267692381[45] = 0;
   out_1875678236267692381[46] = 0;
   out_1875678236267692381[47] = 1;
   out_1875678236267692381[48] = 0;
   out_1875678236267692381[49] = 0;
   out_1875678236267692381[50] = 0;
   out_1875678236267692381[51] = 0;
   out_1875678236267692381[52] = 0;
   out_1875678236267692381[53] = 0;
}
void h_10(double *state, double *unused, double *out_8387436857003783836) {
   out_8387436857003783836[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_8387436857003783836[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_8387436857003783836[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_253999754152164366) {
   out_253999754152164366[0] = 0;
   out_253999754152164366[1] = 9.8100000000000005*cos(state[1]);
   out_253999754152164366[2] = 0;
   out_253999754152164366[3] = 0;
   out_253999754152164366[4] = -state[8];
   out_253999754152164366[5] = state[7];
   out_253999754152164366[6] = 0;
   out_253999754152164366[7] = state[5];
   out_253999754152164366[8] = -state[4];
   out_253999754152164366[9] = 0;
   out_253999754152164366[10] = 0;
   out_253999754152164366[11] = 0;
   out_253999754152164366[12] = 1;
   out_253999754152164366[13] = 0;
   out_253999754152164366[14] = 0;
   out_253999754152164366[15] = 1;
   out_253999754152164366[16] = 0;
   out_253999754152164366[17] = 0;
   out_253999754152164366[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_253999754152164366[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_253999754152164366[20] = 0;
   out_253999754152164366[21] = state[8];
   out_253999754152164366[22] = 0;
   out_253999754152164366[23] = -state[6];
   out_253999754152164366[24] = -state[5];
   out_253999754152164366[25] = 0;
   out_253999754152164366[26] = state[3];
   out_253999754152164366[27] = 0;
   out_253999754152164366[28] = 0;
   out_253999754152164366[29] = 0;
   out_253999754152164366[30] = 0;
   out_253999754152164366[31] = 1;
   out_253999754152164366[32] = 0;
   out_253999754152164366[33] = 0;
   out_253999754152164366[34] = 1;
   out_253999754152164366[35] = 0;
   out_253999754152164366[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_253999754152164366[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_253999754152164366[38] = 0;
   out_253999754152164366[39] = -state[7];
   out_253999754152164366[40] = state[6];
   out_253999754152164366[41] = 0;
   out_253999754152164366[42] = state[4];
   out_253999754152164366[43] = -state[3];
   out_253999754152164366[44] = 0;
   out_253999754152164366[45] = 0;
   out_253999754152164366[46] = 0;
   out_253999754152164366[47] = 0;
   out_253999754152164366[48] = 0;
   out_253999754152164366[49] = 0;
   out_253999754152164366[50] = 1;
   out_253999754152164366[51] = 0;
   out_253999754152164366[52] = 0;
   out_253999754152164366[53] = 1;
}
void h_13(double *state, double *unused, double *out_3782328115152295086) {
   out_3782328115152295086[0] = state[3];
   out_3782328115152295086[1] = state[4];
   out_3782328115152295086[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5709433699570216405) {
   out_5709433699570216405[0] = 0;
   out_5709433699570216405[1] = 0;
   out_5709433699570216405[2] = 0;
   out_5709433699570216405[3] = 1;
   out_5709433699570216405[4] = 0;
   out_5709433699570216405[5] = 0;
   out_5709433699570216405[6] = 0;
   out_5709433699570216405[7] = 0;
   out_5709433699570216405[8] = 0;
   out_5709433699570216405[9] = 0;
   out_5709433699570216405[10] = 0;
   out_5709433699570216405[11] = 0;
   out_5709433699570216405[12] = 0;
   out_5709433699570216405[13] = 0;
   out_5709433699570216405[14] = 0;
   out_5709433699570216405[15] = 0;
   out_5709433699570216405[16] = 0;
   out_5709433699570216405[17] = 0;
   out_5709433699570216405[18] = 0;
   out_5709433699570216405[19] = 0;
   out_5709433699570216405[20] = 0;
   out_5709433699570216405[21] = 0;
   out_5709433699570216405[22] = 1;
   out_5709433699570216405[23] = 0;
   out_5709433699570216405[24] = 0;
   out_5709433699570216405[25] = 0;
   out_5709433699570216405[26] = 0;
   out_5709433699570216405[27] = 0;
   out_5709433699570216405[28] = 0;
   out_5709433699570216405[29] = 0;
   out_5709433699570216405[30] = 0;
   out_5709433699570216405[31] = 0;
   out_5709433699570216405[32] = 0;
   out_5709433699570216405[33] = 0;
   out_5709433699570216405[34] = 0;
   out_5709433699570216405[35] = 0;
   out_5709433699570216405[36] = 0;
   out_5709433699570216405[37] = 0;
   out_5709433699570216405[38] = 0;
   out_5709433699570216405[39] = 0;
   out_5709433699570216405[40] = 0;
   out_5709433699570216405[41] = 1;
   out_5709433699570216405[42] = 0;
   out_5709433699570216405[43] = 0;
   out_5709433699570216405[44] = 0;
   out_5709433699570216405[45] = 0;
   out_5709433699570216405[46] = 0;
   out_5709433699570216405[47] = 0;
   out_5709433699570216405[48] = 0;
   out_5709433699570216405[49] = 0;
   out_5709433699570216405[50] = 0;
   out_5709433699570216405[51] = 0;
   out_5709433699570216405[52] = 0;
   out_5709433699570216405[53] = 0;
}
void h_14(double *state, double *unused, double *out_8292081498849349712) {
   out_8292081498849349712[0] = state[6];
   out_8292081498849349712[1] = state[7];
   out_8292081498849349712[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4958466668563064677) {
   out_4958466668563064677[0] = 0;
   out_4958466668563064677[1] = 0;
   out_4958466668563064677[2] = 0;
   out_4958466668563064677[3] = 0;
   out_4958466668563064677[4] = 0;
   out_4958466668563064677[5] = 0;
   out_4958466668563064677[6] = 1;
   out_4958466668563064677[7] = 0;
   out_4958466668563064677[8] = 0;
   out_4958466668563064677[9] = 0;
   out_4958466668563064677[10] = 0;
   out_4958466668563064677[11] = 0;
   out_4958466668563064677[12] = 0;
   out_4958466668563064677[13] = 0;
   out_4958466668563064677[14] = 0;
   out_4958466668563064677[15] = 0;
   out_4958466668563064677[16] = 0;
   out_4958466668563064677[17] = 0;
   out_4958466668563064677[18] = 0;
   out_4958466668563064677[19] = 0;
   out_4958466668563064677[20] = 0;
   out_4958466668563064677[21] = 0;
   out_4958466668563064677[22] = 0;
   out_4958466668563064677[23] = 0;
   out_4958466668563064677[24] = 0;
   out_4958466668563064677[25] = 1;
   out_4958466668563064677[26] = 0;
   out_4958466668563064677[27] = 0;
   out_4958466668563064677[28] = 0;
   out_4958466668563064677[29] = 0;
   out_4958466668563064677[30] = 0;
   out_4958466668563064677[31] = 0;
   out_4958466668563064677[32] = 0;
   out_4958466668563064677[33] = 0;
   out_4958466668563064677[34] = 0;
   out_4958466668563064677[35] = 0;
   out_4958466668563064677[36] = 0;
   out_4958466668563064677[37] = 0;
   out_4958466668563064677[38] = 0;
   out_4958466668563064677[39] = 0;
   out_4958466668563064677[40] = 0;
   out_4958466668563064677[41] = 0;
   out_4958466668563064677[42] = 0;
   out_4958466668563064677[43] = 0;
   out_4958466668563064677[44] = 1;
   out_4958466668563064677[45] = 0;
   out_4958466668563064677[46] = 0;
   out_4958466668563064677[47] = 0;
   out_4958466668563064677[48] = 0;
   out_4958466668563064677[49] = 0;
   out_4958466668563064677[50] = 0;
   out_4958466668563064677[51] = 0;
   out_4958466668563064677[52] = 0;
   out_4958466668563064677[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_3867314615092444191) {
  err_fun(nom_x, delta_x, out_3867314615092444191);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_357340291324325113) {
  inv_err_fun(nom_x, true_x, out_357340291324325113);
}
void pose_H_mod_fun(double *state, double *out_5335377228813010670) {
  H_mod_fun(state, out_5335377228813010670);
}
void pose_f_fun(double *state, double dt, double *out_8449822447276776350) {
  f_fun(state,  dt, out_8449822447276776350);
}
void pose_F_fun(double *state, double dt, double *out_8918868180555199161) {
  F_fun(state,  dt, out_8918868180555199161);
}
void pose_h_4(double *state, double *unused, double *out_6617669203308538531) {
  h_4(state, unused, out_6617669203308538531);
}
void pose_H_4(double *state, double *unused, double *out_1875678236267692381) {
  H_4(state, unused, out_1875678236267692381);
}
void pose_h_10(double *state, double *unused, double *out_8387436857003783836) {
  h_10(state, unused, out_8387436857003783836);
}
void pose_H_10(double *state, double *unused, double *out_253999754152164366) {
  H_10(state, unused, out_253999754152164366);
}
void pose_h_13(double *state, double *unused, double *out_3782328115152295086) {
  h_13(state, unused, out_3782328115152295086);
}
void pose_H_13(double *state, double *unused, double *out_5709433699570216405) {
  H_13(state, unused, out_5709433699570216405);
}
void pose_h_14(double *state, double *unused, double *out_8292081498849349712) {
  h_14(state, unused, out_8292081498849349712);
}
void pose_H_14(double *state, double *unused, double *out_4958466668563064677) {
  H_14(state, unused, out_4958466668563064677);
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
