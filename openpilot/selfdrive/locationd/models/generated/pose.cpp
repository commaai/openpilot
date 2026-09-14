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
void err_fun(double *nom_x, double *delta_x, double *out_498200963581026915) {
   out_498200963581026915[0] = delta_x[0] + nom_x[0];
   out_498200963581026915[1] = delta_x[1] + nom_x[1];
   out_498200963581026915[2] = delta_x[2] + nom_x[2];
   out_498200963581026915[3] = delta_x[3] + nom_x[3];
   out_498200963581026915[4] = delta_x[4] + nom_x[4];
   out_498200963581026915[5] = delta_x[5] + nom_x[5];
   out_498200963581026915[6] = delta_x[6] + nom_x[6];
   out_498200963581026915[7] = delta_x[7] + nom_x[7];
   out_498200963581026915[8] = delta_x[8] + nom_x[8];
   out_498200963581026915[9] = delta_x[9] + nom_x[9];
   out_498200963581026915[10] = delta_x[10] + nom_x[10];
   out_498200963581026915[11] = delta_x[11] + nom_x[11];
   out_498200963581026915[12] = delta_x[12] + nom_x[12];
   out_498200963581026915[13] = delta_x[13] + nom_x[13];
   out_498200963581026915[14] = delta_x[14] + nom_x[14];
   out_498200963581026915[15] = delta_x[15] + nom_x[15];
   out_498200963581026915[16] = delta_x[16] + nom_x[16];
   out_498200963581026915[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_191777889625927103) {
   out_191777889625927103[0] = -nom_x[0] + true_x[0];
   out_191777889625927103[1] = -nom_x[1] + true_x[1];
   out_191777889625927103[2] = -nom_x[2] + true_x[2];
   out_191777889625927103[3] = -nom_x[3] + true_x[3];
   out_191777889625927103[4] = -nom_x[4] + true_x[4];
   out_191777889625927103[5] = -nom_x[5] + true_x[5];
   out_191777889625927103[6] = -nom_x[6] + true_x[6];
   out_191777889625927103[7] = -nom_x[7] + true_x[7];
   out_191777889625927103[8] = -nom_x[8] + true_x[8];
   out_191777889625927103[9] = -nom_x[9] + true_x[9];
   out_191777889625927103[10] = -nom_x[10] + true_x[10];
   out_191777889625927103[11] = -nom_x[11] + true_x[11];
   out_191777889625927103[12] = -nom_x[12] + true_x[12];
   out_191777889625927103[13] = -nom_x[13] + true_x[13];
   out_191777889625927103[14] = -nom_x[14] + true_x[14];
   out_191777889625927103[15] = -nom_x[15] + true_x[15];
   out_191777889625927103[16] = -nom_x[16] + true_x[16];
   out_191777889625927103[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_8577665774918696216) {
   out_8577665774918696216[0] = 1.0;
   out_8577665774918696216[1] = 0.0;
   out_8577665774918696216[2] = 0.0;
   out_8577665774918696216[3] = 0.0;
   out_8577665774918696216[4] = 0.0;
   out_8577665774918696216[5] = 0.0;
   out_8577665774918696216[6] = 0.0;
   out_8577665774918696216[7] = 0.0;
   out_8577665774918696216[8] = 0.0;
   out_8577665774918696216[9] = 0.0;
   out_8577665774918696216[10] = 0.0;
   out_8577665774918696216[11] = 0.0;
   out_8577665774918696216[12] = 0.0;
   out_8577665774918696216[13] = 0.0;
   out_8577665774918696216[14] = 0.0;
   out_8577665774918696216[15] = 0.0;
   out_8577665774918696216[16] = 0.0;
   out_8577665774918696216[17] = 0.0;
   out_8577665774918696216[18] = 0.0;
   out_8577665774918696216[19] = 1.0;
   out_8577665774918696216[20] = 0.0;
   out_8577665774918696216[21] = 0.0;
   out_8577665774918696216[22] = 0.0;
   out_8577665774918696216[23] = 0.0;
   out_8577665774918696216[24] = 0.0;
   out_8577665774918696216[25] = 0.0;
   out_8577665774918696216[26] = 0.0;
   out_8577665774918696216[27] = 0.0;
   out_8577665774918696216[28] = 0.0;
   out_8577665774918696216[29] = 0.0;
   out_8577665774918696216[30] = 0.0;
   out_8577665774918696216[31] = 0.0;
   out_8577665774918696216[32] = 0.0;
   out_8577665774918696216[33] = 0.0;
   out_8577665774918696216[34] = 0.0;
   out_8577665774918696216[35] = 0.0;
   out_8577665774918696216[36] = 0.0;
   out_8577665774918696216[37] = 0.0;
   out_8577665774918696216[38] = 1.0;
   out_8577665774918696216[39] = 0.0;
   out_8577665774918696216[40] = 0.0;
   out_8577665774918696216[41] = 0.0;
   out_8577665774918696216[42] = 0.0;
   out_8577665774918696216[43] = 0.0;
   out_8577665774918696216[44] = 0.0;
   out_8577665774918696216[45] = 0.0;
   out_8577665774918696216[46] = 0.0;
   out_8577665774918696216[47] = 0.0;
   out_8577665774918696216[48] = 0.0;
   out_8577665774918696216[49] = 0.0;
   out_8577665774918696216[50] = 0.0;
   out_8577665774918696216[51] = 0.0;
   out_8577665774918696216[52] = 0.0;
   out_8577665774918696216[53] = 0.0;
   out_8577665774918696216[54] = 0.0;
   out_8577665774918696216[55] = 0.0;
   out_8577665774918696216[56] = 0.0;
   out_8577665774918696216[57] = 1.0;
   out_8577665774918696216[58] = 0.0;
   out_8577665774918696216[59] = 0.0;
   out_8577665774918696216[60] = 0.0;
   out_8577665774918696216[61] = 0.0;
   out_8577665774918696216[62] = 0.0;
   out_8577665774918696216[63] = 0.0;
   out_8577665774918696216[64] = 0.0;
   out_8577665774918696216[65] = 0.0;
   out_8577665774918696216[66] = 0.0;
   out_8577665774918696216[67] = 0.0;
   out_8577665774918696216[68] = 0.0;
   out_8577665774918696216[69] = 0.0;
   out_8577665774918696216[70] = 0.0;
   out_8577665774918696216[71] = 0.0;
   out_8577665774918696216[72] = 0.0;
   out_8577665774918696216[73] = 0.0;
   out_8577665774918696216[74] = 0.0;
   out_8577665774918696216[75] = 0.0;
   out_8577665774918696216[76] = 1.0;
   out_8577665774918696216[77] = 0.0;
   out_8577665774918696216[78] = 0.0;
   out_8577665774918696216[79] = 0.0;
   out_8577665774918696216[80] = 0.0;
   out_8577665774918696216[81] = 0.0;
   out_8577665774918696216[82] = 0.0;
   out_8577665774918696216[83] = 0.0;
   out_8577665774918696216[84] = 0.0;
   out_8577665774918696216[85] = 0.0;
   out_8577665774918696216[86] = 0.0;
   out_8577665774918696216[87] = 0.0;
   out_8577665774918696216[88] = 0.0;
   out_8577665774918696216[89] = 0.0;
   out_8577665774918696216[90] = 0.0;
   out_8577665774918696216[91] = 0.0;
   out_8577665774918696216[92] = 0.0;
   out_8577665774918696216[93] = 0.0;
   out_8577665774918696216[94] = 0.0;
   out_8577665774918696216[95] = 1.0;
   out_8577665774918696216[96] = 0.0;
   out_8577665774918696216[97] = 0.0;
   out_8577665774918696216[98] = 0.0;
   out_8577665774918696216[99] = 0.0;
   out_8577665774918696216[100] = 0.0;
   out_8577665774918696216[101] = 0.0;
   out_8577665774918696216[102] = 0.0;
   out_8577665774918696216[103] = 0.0;
   out_8577665774918696216[104] = 0.0;
   out_8577665774918696216[105] = 0.0;
   out_8577665774918696216[106] = 0.0;
   out_8577665774918696216[107] = 0.0;
   out_8577665774918696216[108] = 0.0;
   out_8577665774918696216[109] = 0.0;
   out_8577665774918696216[110] = 0.0;
   out_8577665774918696216[111] = 0.0;
   out_8577665774918696216[112] = 0.0;
   out_8577665774918696216[113] = 0.0;
   out_8577665774918696216[114] = 1.0;
   out_8577665774918696216[115] = 0.0;
   out_8577665774918696216[116] = 0.0;
   out_8577665774918696216[117] = 0.0;
   out_8577665774918696216[118] = 0.0;
   out_8577665774918696216[119] = 0.0;
   out_8577665774918696216[120] = 0.0;
   out_8577665774918696216[121] = 0.0;
   out_8577665774918696216[122] = 0.0;
   out_8577665774918696216[123] = 0.0;
   out_8577665774918696216[124] = 0.0;
   out_8577665774918696216[125] = 0.0;
   out_8577665774918696216[126] = 0.0;
   out_8577665774918696216[127] = 0.0;
   out_8577665774918696216[128] = 0.0;
   out_8577665774918696216[129] = 0.0;
   out_8577665774918696216[130] = 0.0;
   out_8577665774918696216[131] = 0.0;
   out_8577665774918696216[132] = 0.0;
   out_8577665774918696216[133] = 1.0;
   out_8577665774918696216[134] = 0.0;
   out_8577665774918696216[135] = 0.0;
   out_8577665774918696216[136] = 0.0;
   out_8577665774918696216[137] = 0.0;
   out_8577665774918696216[138] = 0.0;
   out_8577665774918696216[139] = 0.0;
   out_8577665774918696216[140] = 0.0;
   out_8577665774918696216[141] = 0.0;
   out_8577665774918696216[142] = 0.0;
   out_8577665774918696216[143] = 0.0;
   out_8577665774918696216[144] = 0.0;
   out_8577665774918696216[145] = 0.0;
   out_8577665774918696216[146] = 0.0;
   out_8577665774918696216[147] = 0.0;
   out_8577665774918696216[148] = 0.0;
   out_8577665774918696216[149] = 0.0;
   out_8577665774918696216[150] = 0.0;
   out_8577665774918696216[151] = 0.0;
   out_8577665774918696216[152] = 1.0;
   out_8577665774918696216[153] = 0.0;
   out_8577665774918696216[154] = 0.0;
   out_8577665774918696216[155] = 0.0;
   out_8577665774918696216[156] = 0.0;
   out_8577665774918696216[157] = 0.0;
   out_8577665774918696216[158] = 0.0;
   out_8577665774918696216[159] = 0.0;
   out_8577665774918696216[160] = 0.0;
   out_8577665774918696216[161] = 0.0;
   out_8577665774918696216[162] = 0.0;
   out_8577665774918696216[163] = 0.0;
   out_8577665774918696216[164] = 0.0;
   out_8577665774918696216[165] = 0.0;
   out_8577665774918696216[166] = 0.0;
   out_8577665774918696216[167] = 0.0;
   out_8577665774918696216[168] = 0.0;
   out_8577665774918696216[169] = 0.0;
   out_8577665774918696216[170] = 0.0;
   out_8577665774918696216[171] = 1.0;
   out_8577665774918696216[172] = 0.0;
   out_8577665774918696216[173] = 0.0;
   out_8577665774918696216[174] = 0.0;
   out_8577665774918696216[175] = 0.0;
   out_8577665774918696216[176] = 0.0;
   out_8577665774918696216[177] = 0.0;
   out_8577665774918696216[178] = 0.0;
   out_8577665774918696216[179] = 0.0;
   out_8577665774918696216[180] = 0.0;
   out_8577665774918696216[181] = 0.0;
   out_8577665774918696216[182] = 0.0;
   out_8577665774918696216[183] = 0.0;
   out_8577665774918696216[184] = 0.0;
   out_8577665774918696216[185] = 0.0;
   out_8577665774918696216[186] = 0.0;
   out_8577665774918696216[187] = 0.0;
   out_8577665774918696216[188] = 0.0;
   out_8577665774918696216[189] = 0.0;
   out_8577665774918696216[190] = 1.0;
   out_8577665774918696216[191] = 0.0;
   out_8577665774918696216[192] = 0.0;
   out_8577665774918696216[193] = 0.0;
   out_8577665774918696216[194] = 0.0;
   out_8577665774918696216[195] = 0.0;
   out_8577665774918696216[196] = 0.0;
   out_8577665774918696216[197] = 0.0;
   out_8577665774918696216[198] = 0.0;
   out_8577665774918696216[199] = 0.0;
   out_8577665774918696216[200] = 0.0;
   out_8577665774918696216[201] = 0.0;
   out_8577665774918696216[202] = 0.0;
   out_8577665774918696216[203] = 0.0;
   out_8577665774918696216[204] = 0.0;
   out_8577665774918696216[205] = 0.0;
   out_8577665774918696216[206] = 0.0;
   out_8577665774918696216[207] = 0.0;
   out_8577665774918696216[208] = 0.0;
   out_8577665774918696216[209] = 1.0;
   out_8577665774918696216[210] = 0.0;
   out_8577665774918696216[211] = 0.0;
   out_8577665774918696216[212] = 0.0;
   out_8577665774918696216[213] = 0.0;
   out_8577665774918696216[214] = 0.0;
   out_8577665774918696216[215] = 0.0;
   out_8577665774918696216[216] = 0.0;
   out_8577665774918696216[217] = 0.0;
   out_8577665774918696216[218] = 0.0;
   out_8577665774918696216[219] = 0.0;
   out_8577665774918696216[220] = 0.0;
   out_8577665774918696216[221] = 0.0;
   out_8577665774918696216[222] = 0.0;
   out_8577665774918696216[223] = 0.0;
   out_8577665774918696216[224] = 0.0;
   out_8577665774918696216[225] = 0.0;
   out_8577665774918696216[226] = 0.0;
   out_8577665774918696216[227] = 0.0;
   out_8577665774918696216[228] = 1.0;
   out_8577665774918696216[229] = 0.0;
   out_8577665774918696216[230] = 0.0;
   out_8577665774918696216[231] = 0.0;
   out_8577665774918696216[232] = 0.0;
   out_8577665774918696216[233] = 0.0;
   out_8577665774918696216[234] = 0.0;
   out_8577665774918696216[235] = 0.0;
   out_8577665774918696216[236] = 0.0;
   out_8577665774918696216[237] = 0.0;
   out_8577665774918696216[238] = 0.0;
   out_8577665774918696216[239] = 0.0;
   out_8577665774918696216[240] = 0.0;
   out_8577665774918696216[241] = 0.0;
   out_8577665774918696216[242] = 0.0;
   out_8577665774918696216[243] = 0.0;
   out_8577665774918696216[244] = 0.0;
   out_8577665774918696216[245] = 0.0;
   out_8577665774918696216[246] = 0.0;
   out_8577665774918696216[247] = 1.0;
   out_8577665774918696216[248] = 0.0;
   out_8577665774918696216[249] = 0.0;
   out_8577665774918696216[250] = 0.0;
   out_8577665774918696216[251] = 0.0;
   out_8577665774918696216[252] = 0.0;
   out_8577665774918696216[253] = 0.0;
   out_8577665774918696216[254] = 0.0;
   out_8577665774918696216[255] = 0.0;
   out_8577665774918696216[256] = 0.0;
   out_8577665774918696216[257] = 0.0;
   out_8577665774918696216[258] = 0.0;
   out_8577665774918696216[259] = 0.0;
   out_8577665774918696216[260] = 0.0;
   out_8577665774918696216[261] = 0.0;
   out_8577665774918696216[262] = 0.0;
   out_8577665774918696216[263] = 0.0;
   out_8577665774918696216[264] = 0.0;
   out_8577665774918696216[265] = 0.0;
   out_8577665774918696216[266] = 1.0;
   out_8577665774918696216[267] = 0.0;
   out_8577665774918696216[268] = 0.0;
   out_8577665774918696216[269] = 0.0;
   out_8577665774918696216[270] = 0.0;
   out_8577665774918696216[271] = 0.0;
   out_8577665774918696216[272] = 0.0;
   out_8577665774918696216[273] = 0.0;
   out_8577665774918696216[274] = 0.0;
   out_8577665774918696216[275] = 0.0;
   out_8577665774918696216[276] = 0.0;
   out_8577665774918696216[277] = 0.0;
   out_8577665774918696216[278] = 0.0;
   out_8577665774918696216[279] = 0.0;
   out_8577665774918696216[280] = 0.0;
   out_8577665774918696216[281] = 0.0;
   out_8577665774918696216[282] = 0.0;
   out_8577665774918696216[283] = 0.0;
   out_8577665774918696216[284] = 0.0;
   out_8577665774918696216[285] = 1.0;
   out_8577665774918696216[286] = 0.0;
   out_8577665774918696216[287] = 0.0;
   out_8577665774918696216[288] = 0.0;
   out_8577665774918696216[289] = 0.0;
   out_8577665774918696216[290] = 0.0;
   out_8577665774918696216[291] = 0.0;
   out_8577665774918696216[292] = 0.0;
   out_8577665774918696216[293] = 0.0;
   out_8577665774918696216[294] = 0.0;
   out_8577665774918696216[295] = 0.0;
   out_8577665774918696216[296] = 0.0;
   out_8577665774918696216[297] = 0.0;
   out_8577665774918696216[298] = 0.0;
   out_8577665774918696216[299] = 0.0;
   out_8577665774918696216[300] = 0.0;
   out_8577665774918696216[301] = 0.0;
   out_8577665774918696216[302] = 0.0;
   out_8577665774918696216[303] = 0.0;
   out_8577665774918696216[304] = 1.0;
   out_8577665774918696216[305] = 0.0;
   out_8577665774918696216[306] = 0.0;
   out_8577665774918696216[307] = 0.0;
   out_8577665774918696216[308] = 0.0;
   out_8577665774918696216[309] = 0.0;
   out_8577665774918696216[310] = 0.0;
   out_8577665774918696216[311] = 0.0;
   out_8577665774918696216[312] = 0.0;
   out_8577665774918696216[313] = 0.0;
   out_8577665774918696216[314] = 0.0;
   out_8577665774918696216[315] = 0.0;
   out_8577665774918696216[316] = 0.0;
   out_8577665774918696216[317] = 0.0;
   out_8577665774918696216[318] = 0.0;
   out_8577665774918696216[319] = 0.0;
   out_8577665774918696216[320] = 0.0;
   out_8577665774918696216[321] = 0.0;
   out_8577665774918696216[322] = 0.0;
   out_8577665774918696216[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5599332296746008014) {
   out_5599332296746008014[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5599332296746008014[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5599332296746008014[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5599332296746008014[3] = dt*state[12] + state[3];
   out_5599332296746008014[4] = dt*state[13] + state[4];
   out_5599332296746008014[5] = dt*state[14] + state[5];
   out_5599332296746008014[6] = state[6];
   out_5599332296746008014[7] = state[7];
   out_5599332296746008014[8] = state[8];
   out_5599332296746008014[9] = state[9];
   out_5599332296746008014[10] = state[10];
   out_5599332296746008014[11] = state[11];
   out_5599332296746008014[12] = state[12];
   out_5599332296746008014[13] = state[13];
   out_5599332296746008014[14] = state[14];
   out_5599332296746008014[15] = state[15];
   out_5599332296746008014[16] = state[16];
   out_5599332296746008014[17] = state[17];
}
void F_fun(double *state, double dt, double *out_1584783165911621436) {
   out_1584783165911621436[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1584783165911621436[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1584783165911621436[2] = 0;
   out_1584783165911621436[3] = 0;
   out_1584783165911621436[4] = 0;
   out_1584783165911621436[5] = 0;
   out_1584783165911621436[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1584783165911621436[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1584783165911621436[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_1584783165911621436[9] = 0;
   out_1584783165911621436[10] = 0;
   out_1584783165911621436[11] = 0;
   out_1584783165911621436[12] = 0;
   out_1584783165911621436[13] = 0;
   out_1584783165911621436[14] = 0;
   out_1584783165911621436[15] = 0;
   out_1584783165911621436[16] = 0;
   out_1584783165911621436[17] = 0;
   out_1584783165911621436[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1584783165911621436[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1584783165911621436[20] = 0;
   out_1584783165911621436[21] = 0;
   out_1584783165911621436[22] = 0;
   out_1584783165911621436[23] = 0;
   out_1584783165911621436[24] = 0;
   out_1584783165911621436[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1584783165911621436[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_1584783165911621436[27] = 0;
   out_1584783165911621436[28] = 0;
   out_1584783165911621436[29] = 0;
   out_1584783165911621436[30] = 0;
   out_1584783165911621436[31] = 0;
   out_1584783165911621436[32] = 0;
   out_1584783165911621436[33] = 0;
   out_1584783165911621436[34] = 0;
   out_1584783165911621436[35] = 0;
   out_1584783165911621436[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1584783165911621436[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1584783165911621436[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1584783165911621436[39] = 0;
   out_1584783165911621436[40] = 0;
   out_1584783165911621436[41] = 0;
   out_1584783165911621436[42] = 0;
   out_1584783165911621436[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1584783165911621436[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_1584783165911621436[45] = 0;
   out_1584783165911621436[46] = 0;
   out_1584783165911621436[47] = 0;
   out_1584783165911621436[48] = 0;
   out_1584783165911621436[49] = 0;
   out_1584783165911621436[50] = 0;
   out_1584783165911621436[51] = 0;
   out_1584783165911621436[52] = 0;
   out_1584783165911621436[53] = 0;
   out_1584783165911621436[54] = 0;
   out_1584783165911621436[55] = 0;
   out_1584783165911621436[56] = 0;
   out_1584783165911621436[57] = 1;
   out_1584783165911621436[58] = 0;
   out_1584783165911621436[59] = 0;
   out_1584783165911621436[60] = 0;
   out_1584783165911621436[61] = 0;
   out_1584783165911621436[62] = 0;
   out_1584783165911621436[63] = 0;
   out_1584783165911621436[64] = 0;
   out_1584783165911621436[65] = 0;
   out_1584783165911621436[66] = dt;
   out_1584783165911621436[67] = 0;
   out_1584783165911621436[68] = 0;
   out_1584783165911621436[69] = 0;
   out_1584783165911621436[70] = 0;
   out_1584783165911621436[71] = 0;
   out_1584783165911621436[72] = 0;
   out_1584783165911621436[73] = 0;
   out_1584783165911621436[74] = 0;
   out_1584783165911621436[75] = 0;
   out_1584783165911621436[76] = 1;
   out_1584783165911621436[77] = 0;
   out_1584783165911621436[78] = 0;
   out_1584783165911621436[79] = 0;
   out_1584783165911621436[80] = 0;
   out_1584783165911621436[81] = 0;
   out_1584783165911621436[82] = 0;
   out_1584783165911621436[83] = 0;
   out_1584783165911621436[84] = 0;
   out_1584783165911621436[85] = dt;
   out_1584783165911621436[86] = 0;
   out_1584783165911621436[87] = 0;
   out_1584783165911621436[88] = 0;
   out_1584783165911621436[89] = 0;
   out_1584783165911621436[90] = 0;
   out_1584783165911621436[91] = 0;
   out_1584783165911621436[92] = 0;
   out_1584783165911621436[93] = 0;
   out_1584783165911621436[94] = 0;
   out_1584783165911621436[95] = 1;
   out_1584783165911621436[96] = 0;
   out_1584783165911621436[97] = 0;
   out_1584783165911621436[98] = 0;
   out_1584783165911621436[99] = 0;
   out_1584783165911621436[100] = 0;
   out_1584783165911621436[101] = 0;
   out_1584783165911621436[102] = 0;
   out_1584783165911621436[103] = 0;
   out_1584783165911621436[104] = dt;
   out_1584783165911621436[105] = 0;
   out_1584783165911621436[106] = 0;
   out_1584783165911621436[107] = 0;
   out_1584783165911621436[108] = 0;
   out_1584783165911621436[109] = 0;
   out_1584783165911621436[110] = 0;
   out_1584783165911621436[111] = 0;
   out_1584783165911621436[112] = 0;
   out_1584783165911621436[113] = 0;
   out_1584783165911621436[114] = 1;
   out_1584783165911621436[115] = 0;
   out_1584783165911621436[116] = 0;
   out_1584783165911621436[117] = 0;
   out_1584783165911621436[118] = 0;
   out_1584783165911621436[119] = 0;
   out_1584783165911621436[120] = 0;
   out_1584783165911621436[121] = 0;
   out_1584783165911621436[122] = 0;
   out_1584783165911621436[123] = 0;
   out_1584783165911621436[124] = 0;
   out_1584783165911621436[125] = 0;
   out_1584783165911621436[126] = 0;
   out_1584783165911621436[127] = 0;
   out_1584783165911621436[128] = 0;
   out_1584783165911621436[129] = 0;
   out_1584783165911621436[130] = 0;
   out_1584783165911621436[131] = 0;
   out_1584783165911621436[132] = 0;
   out_1584783165911621436[133] = 1;
   out_1584783165911621436[134] = 0;
   out_1584783165911621436[135] = 0;
   out_1584783165911621436[136] = 0;
   out_1584783165911621436[137] = 0;
   out_1584783165911621436[138] = 0;
   out_1584783165911621436[139] = 0;
   out_1584783165911621436[140] = 0;
   out_1584783165911621436[141] = 0;
   out_1584783165911621436[142] = 0;
   out_1584783165911621436[143] = 0;
   out_1584783165911621436[144] = 0;
   out_1584783165911621436[145] = 0;
   out_1584783165911621436[146] = 0;
   out_1584783165911621436[147] = 0;
   out_1584783165911621436[148] = 0;
   out_1584783165911621436[149] = 0;
   out_1584783165911621436[150] = 0;
   out_1584783165911621436[151] = 0;
   out_1584783165911621436[152] = 1;
   out_1584783165911621436[153] = 0;
   out_1584783165911621436[154] = 0;
   out_1584783165911621436[155] = 0;
   out_1584783165911621436[156] = 0;
   out_1584783165911621436[157] = 0;
   out_1584783165911621436[158] = 0;
   out_1584783165911621436[159] = 0;
   out_1584783165911621436[160] = 0;
   out_1584783165911621436[161] = 0;
   out_1584783165911621436[162] = 0;
   out_1584783165911621436[163] = 0;
   out_1584783165911621436[164] = 0;
   out_1584783165911621436[165] = 0;
   out_1584783165911621436[166] = 0;
   out_1584783165911621436[167] = 0;
   out_1584783165911621436[168] = 0;
   out_1584783165911621436[169] = 0;
   out_1584783165911621436[170] = 0;
   out_1584783165911621436[171] = 1;
   out_1584783165911621436[172] = 0;
   out_1584783165911621436[173] = 0;
   out_1584783165911621436[174] = 0;
   out_1584783165911621436[175] = 0;
   out_1584783165911621436[176] = 0;
   out_1584783165911621436[177] = 0;
   out_1584783165911621436[178] = 0;
   out_1584783165911621436[179] = 0;
   out_1584783165911621436[180] = 0;
   out_1584783165911621436[181] = 0;
   out_1584783165911621436[182] = 0;
   out_1584783165911621436[183] = 0;
   out_1584783165911621436[184] = 0;
   out_1584783165911621436[185] = 0;
   out_1584783165911621436[186] = 0;
   out_1584783165911621436[187] = 0;
   out_1584783165911621436[188] = 0;
   out_1584783165911621436[189] = 0;
   out_1584783165911621436[190] = 1;
   out_1584783165911621436[191] = 0;
   out_1584783165911621436[192] = 0;
   out_1584783165911621436[193] = 0;
   out_1584783165911621436[194] = 0;
   out_1584783165911621436[195] = 0;
   out_1584783165911621436[196] = 0;
   out_1584783165911621436[197] = 0;
   out_1584783165911621436[198] = 0;
   out_1584783165911621436[199] = 0;
   out_1584783165911621436[200] = 0;
   out_1584783165911621436[201] = 0;
   out_1584783165911621436[202] = 0;
   out_1584783165911621436[203] = 0;
   out_1584783165911621436[204] = 0;
   out_1584783165911621436[205] = 0;
   out_1584783165911621436[206] = 0;
   out_1584783165911621436[207] = 0;
   out_1584783165911621436[208] = 0;
   out_1584783165911621436[209] = 1;
   out_1584783165911621436[210] = 0;
   out_1584783165911621436[211] = 0;
   out_1584783165911621436[212] = 0;
   out_1584783165911621436[213] = 0;
   out_1584783165911621436[214] = 0;
   out_1584783165911621436[215] = 0;
   out_1584783165911621436[216] = 0;
   out_1584783165911621436[217] = 0;
   out_1584783165911621436[218] = 0;
   out_1584783165911621436[219] = 0;
   out_1584783165911621436[220] = 0;
   out_1584783165911621436[221] = 0;
   out_1584783165911621436[222] = 0;
   out_1584783165911621436[223] = 0;
   out_1584783165911621436[224] = 0;
   out_1584783165911621436[225] = 0;
   out_1584783165911621436[226] = 0;
   out_1584783165911621436[227] = 0;
   out_1584783165911621436[228] = 1;
   out_1584783165911621436[229] = 0;
   out_1584783165911621436[230] = 0;
   out_1584783165911621436[231] = 0;
   out_1584783165911621436[232] = 0;
   out_1584783165911621436[233] = 0;
   out_1584783165911621436[234] = 0;
   out_1584783165911621436[235] = 0;
   out_1584783165911621436[236] = 0;
   out_1584783165911621436[237] = 0;
   out_1584783165911621436[238] = 0;
   out_1584783165911621436[239] = 0;
   out_1584783165911621436[240] = 0;
   out_1584783165911621436[241] = 0;
   out_1584783165911621436[242] = 0;
   out_1584783165911621436[243] = 0;
   out_1584783165911621436[244] = 0;
   out_1584783165911621436[245] = 0;
   out_1584783165911621436[246] = 0;
   out_1584783165911621436[247] = 1;
   out_1584783165911621436[248] = 0;
   out_1584783165911621436[249] = 0;
   out_1584783165911621436[250] = 0;
   out_1584783165911621436[251] = 0;
   out_1584783165911621436[252] = 0;
   out_1584783165911621436[253] = 0;
   out_1584783165911621436[254] = 0;
   out_1584783165911621436[255] = 0;
   out_1584783165911621436[256] = 0;
   out_1584783165911621436[257] = 0;
   out_1584783165911621436[258] = 0;
   out_1584783165911621436[259] = 0;
   out_1584783165911621436[260] = 0;
   out_1584783165911621436[261] = 0;
   out_1584783165911621436[262] = 0;
   out_1584783165911621436[263] = 0;
   out_1584783165911621436[264] = 0;
   out_1584783165911621436[265] = 0;
   out_1584783165911621436[266] = 1;
   out_1584783165911621436[267] = 0;
   out_1584783165911621436[268] = 0;
   out_1584783165911621436[269] = 0;
   out_1584783165911621436[270] = 0;
   out_1584783165911621436[271] = 0;
   out_1584783165911621436[272] = 0;
   out_1584783165911621436[273] = 0;
   out_1584783165911621436[274] = 0;
   out_1584783165911621436[275] = 0;
   out_1584783165911621436[276] = 0;
   out_1584783165911621436[277] = 0;
   out_1584783165911621436[278] = 0;
   out_1584783165911621436[279] = 0;
   out_1584783165911621436[280] = 0;
   out_1584783165911621436[281] = 0;
   out_1584783165911621436[282] = 0;
   out_1584783165911621436[283] = 0;
   out_1584783165911621436[284] = 0;
   out_1584783165911621436[285] = 1;
   out_1584783165911621436[286] = 0;
   out_1584783165911621436[287] = 0;
   out_1584783165911621436[288] = 0;
   out_1584783165911621436[289] = 0;
   out_1584783165911621436[290] = 0;
   out_1584783165911621436[291] = 0;
   out_1584783165911621436[292] = 0;
   out_1584783165911621436[293] = 0;
   out_1584783165911621436[294] = 0;
   out_1584783165911621436[295] = 0;
   out_1584783165911621436[296] = 0;
   out_1584783165911621436[297] = 0;
   out_1584783165911621436[298] = 0;
   out_1584783165911621436[299] = 0;
   out_1584783165911621436[300] = 0;
   out_1584783165911621436[301] = 0;
   out_1584783165911621436[302] = 0;
   out_1584783165911621436[303] = 0;
   out_1584783165911621436[304] = 1;
   out_1584783165911621436[305] = 0;
   out_1584783165911621436[306] = 0;
   out_1584783165911621436[307] = 0;
   out_1584783165911621436[308] = 0;
   out_1584783165911621436[309] = 0;
   out_1584783165911621436[310] = 0;
   out_1584783165911621436[311] = 0;
   out_1584783165911621436[312] = 0;
   out_1584783165911621436[313] = 0;
   out_1584783165911621436[314] = 0;
   out_1584783165911621436[315] = 0;
   out_1584783165911621436[316] = 0;
   out_1584783165911621436[317] = 0;
   out_1584783165911621436[318] = 0;
   out_1584783165911621436[319] = 0;
   out_1584783165911621436[320] = 0;
   out_1584783165911621436[321] = 0;
   out_1584783165911621436[322] = 0;
   out_1584783165911621436[323] = 1;
}
void h_4(double *state, double *unused, double *out_8170273359625996454) {
   out_8170273359625996454[0] = state[6] + state[9];
   out_8170273359625996454[1] = state[7] + state[10];
   out_8170273359625996454[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7481347933629845101) {
   out_7481347933629845101[0] = 0;
   out_7481347933629845101[1] = 0;
   out_7481347933629845101[2] = 0;
   out_7481347933629845101[3] = 0;
   out_7481347933629845101[4] = 0;
   out_7481347933629845101[5] = 0;
   out_7481347933629845101[6] = 1;
   out_7481347933629845101[7] = 0;
   out_7481347933629845101[8] = 0;
   out_7481347933629845101[9] = 1;
   out_7481347933629845101[10] = 0;
   out_7481347933629845101[11] = 0;
   out_7481347933629845101[12] = 0;
   out_7481347933629845101[13] = 0;
   out_7481347933629845101[14] = 0;
   out_7481347933629845101[15] = 0;
   out_7481347933629845101[16] = 0;
   out_7481347933629845101[17] = 0;
   out_7481347933629845101[18] = 0;
   out_7481347933629845101[19] = 0;
   out_7481347933629845101[20] = 0;
   out_7481347933629845101[21] = 0;
   out_7481347933629845101[22] = 0;
   out_7481347933629845101[23] = 0;
   out_7481347933629845101[24] = 0;
   out_7481347933629845101[25] = 1;
   out_7481347933629845101[26] = 0;
   out_7481347933629845101[27] = 0;
   out_7481347933629845101[28] = 1;
   out_7481347933629845101[29] = 0;
   out_7481347933629845101[30] = 0;
   out_7481347933629845101[31] = 0;
   out_7481347933629845101[32] = 0;
   out_7481347933629845101[33] = 0;
   out_7481347933629845101[34] = 0;
   out_7481347933629845101[35] = 0;
   out_7481347933629845101[36] = 0;
   out_7481347933629845101[37] = 0;
   out_7481347933629845101[38] = 0;
   out_7481347933629845101[39] = 0;
   out_7481347933629845101[40] = 0;
   out_7481347933629845101[41] = 0;
   out_7481347933629845101[42] = 0;
   out_7481347933629845101[43] = 0;
   out_7481347933629845101[44] = 1;
   out_7481347933629845101[45] = 0;
   out_7481347933629845101[46] = 0;
   out_7481347933629845101[47] = 1;
   out_7481347933629845101[48] = 0;
   out_7481347933629845101[49] = 0;
   out_7481347933629845101[50] = 0;
   out_7481347933629845101[51] = 0;
   out_7481347933629845101[52] = 0;
   out_7481347933629845101[53] = 0;
}
void h_10(double *state, double *unused, double *out_4791397978297849235) {
   out_4791397978297849235[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_4791397978297849235[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_4791397978297849235[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_7583350988764981753) {
   out_7583350988764981753[0] = 0;
   out_7583350988764981753[1] = 9.8100000000000005*cos(state[1]);
   out_7583350988764981753[2] = 0;
   out_7583350988764981753[3] = 0;
   out_7583350988764981753[4] = -state[8];
   out_7583350988764981753[5] = state[7];
   out_7583350988764981753[6] = 0;
   out_7583350988764981753[7] = state[5];
   out_7583350988764981753[8] = -state[4];
   out_7583350988764981753[9] = 0;
   out_7583350988764981753[10] = 0;
   out_7583350988764981753[11] = 0;
   out_7583350988764981753[12] = 1;
   out_7583350988764981753[13] = 0;
   out_7583350988764981753[14] = 0;
   out_7583350988764981753[15] = 1;
   out_7583350988764981753[16] = 0;
   out_7583350988764981753[17] = 0;
   out_7583350988764981753[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_7583350988764981753[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_7583350988764981753[20] = 0;
   out_7583350988764981753[21] = state[8];
   out_7583350988764981753[22] = 0;
   out_7583350988764981753[23] = -state[6];
   out_7583350988764981753[24] = -state[5];
   out_7583350988764981753[25] = 0;
   out_7583350988764981753[26] = state[3];
   out_7583350988764981753[27] = 0;
   out_7583350988764981753[28] = 0;
   out_7583350988764981753[29] = 0;
   out_7583350988764981753[30] = 0;
   out_7583350988764981753[31] = 1;
   out_7583350988764981753[32] = 0;
   out_7583350988764981753[33] = 0;
   out_7583350988764981753[34] = 1;
   out_7583350988764981753[35] = 0;
   out_7583350988764981753[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_7583350988764981753[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_7583350988764981753[38] = 0;
   out_7583350988764981753[39] = -state[7];
   out_7583350988764981753[40] = state[6];
   out_7583350988764981753[41] = 0;
   out_7583350988764981753[42] = state[4];
   out_7583350988764981753[43] = -state[3];
   out_7583350988764981753[44] = 0;
   out_7583350988764981753[45] = 0;
   out_7583350988764981753[46] = 0;
   out_7583350988764981753[47] = 0;
   out_7583350988764981753[48] = 0;
   out_7583350988764981753[49] = 0;
   out_7583350988764981753[50] = 1;
   out_7583350988764981753[51] = 0;
   out_7583350988764981753[52] = 0;
   out_7583350988764981753[53] = 1;
}
void h_13(double *state, double *unused, double *out_5529921368018353304) {
   out_5529921368018353304[0] = state[3];
   out_5529921368018353304[1] = state[4];
   out_5529921368018353304[2] = state[5];
}
void H_13(double *state, double *unused, double *out_3647592470327321077) {
   out_3647592470327321077[0] = 0;
   out_3647592470327321077[1] = 0;
   out_3647592470327321077[2] = 0;
   out_3647592470327321077[3] = 1;
   out_3647592470327321077[4] = 0;
   out_3647592470327321077[5] = 0;
   out_3647592470327321077[6] = 0;
   out_3647592470327321077[7] = 0;
   out_3647592470327321077[8] = 0;
   out_3647592470327321077[9] = 0;
   out_3647592470327321077[10] = 0;
   out_3647592470327321077[11] = 0;
   out_3647592470327321077[12] = 0;
   out_3647592470327321077[13] = 0;
   out_3647592470327321077[14] = 0;
   out_3647592470327321077[15] = 0;
   out_3647592470327321077[16] = 0;
   out_3647592470327321077[17] = 0;
   out_3647592470327321077[18] = 0;
   out_3647592470327321077[19] = 0;
   out_3647592470327321077[20] = 0;
   out_3647592470327321077[21] = 0;
   out_3647592470327321077[22] = 1;
   out_3647592470327321077[23] = 0;
   out_3647592470327321077[24] = 0;
   out_3647592470327321077[25] = 0;
   out_3647592470327321077[26] = 0;
   out_3647592470327321077[27] = 0;
   out_3647592470327321077[28] = 0;
   out_3647592470327321077[29] = 0;
   out_3647592470327321077[30] = 0;
   out_3647592470327321077[31] = 0;
   out_3647592470327321077[32] = 0;
   out_3647592470327321077[33] = 0;
   out_3647592470327321077[34] = 0;
   out_3647592470327321077[35] = 0;
   out_3647592470327321077[36] = 0;
   out_3647592470327321077[37] = 0;
   out_3647592470327321077[38] = 0;
   out_3647592470327321077[39] = 0;
   out_3647592470327321077[40] = 0;
   out_3647592470327321077[41] = 1;
   out_3647592470327321077[42] = 0;
   out_3647592470327321077[43] = 0;
   out_3647592470327321077[44] = 0;
   out_3647592470327321077[45] = 0;
   out_3647592470327321077[46] = 0;
   out_3647592470327321077[47] = 0;
   out_3647592470327321077[48] = 0;
   out_3647592470327321077[49] = 0;
   out_3647592470327321077[50] = 0;
   out_3647592470327321077[51] = 0;
   out_3647592470327321077[52] = 0;
   out_3647592470327321077[53] = 0;
}
void h_14(double *state, double *unused, double *out_4744126367926772826) {
   out_4744126367926772826[0] = state[6];
   out_4744126367926772826[1] = state[7];
   out_4744126367926772826[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4398559501334472805) {
   out_4398559501334472805[0] = 0;
   out_4398559501334472805[1] = 0;
   out_4398559501334472805[2] = 0;
   out_4398559501334472805[3] = 0;
   out_4398559501334472805[4] = 0;
   out_4398559501334472805[5] = 0;
   out_4398559501334472805[6] = 1;
   out_4398559501334472805[7] = 0;
   out_4398559501334472805[8] = 0;
   out_4398559501334472805[9] = 0;
   out_4398559501334472805[10] = 0;
   out_4398559501334472805[11] = 0;
   out_4398559501334472805[12] = 0;
   out_4398559501334472805[13] = 0;
   out_4398559501334472805[14] = 0;
   out_4398559501334472805[15] = 0;
   out_4398559501334472805[16] = 0;
   out_4398559501334472805[17] = 0;
   out_4398559501334472805[18] = 0;
   out_4398559501334472805[19] = 0;
   out_4398559501334472805[20] = 0;
   out_4398559501334472805[21] = 0;
   out_4398559501334472805[22] = 0;
   out_4398559501334472805[23] = 0;
   out_4398559501334472805[24] = 0;
   out_4398559501334472805[25] = 1;
   out_4398559501334472805[26] = 0;
   out_4398559501334472805[27] = 0;
   out_4398559501334472805[28] = 0;
   out_4398559501334472805[29] = 0;
   out_4398559501334472805[30] = 0;
   out_4398559501334472805[31] = 0;
   out_4398559501334472805[32] = 0;
   out_4398559501334472805[33] = 0;
   out_4398559501334472805[34] = 0;
   out_4398559501334472805[35] = 0;
   out_4398559501334472805[36] = 0;
   out_4398559501334472805[37] = 0;
   out_4398559501334472805[38] = 0;
   out_4398559501334472805[39] = 0;
   out_4398559501334472805[40] = 0;
   out_4398559501334472805[41] = 0;
   out_4398559501334472805[42] = 0;
   out_4398559501334472805[43] = 0;
   out_4398559501334472805[44] = 1;
   out_4398559501334472805[45] = 0;
   out_4398559501334472805[46] = 0;
   out_4398559501334472805[47] = 0;
   out_4398559501334472805[48] = 0;
   out_4398559501334472805[49] = 0;
   out_4398559501334472805[50] = 0;
   out_4398559501334472805[51] = 0;
   out_4398559501334472805[52] = 0;
   out_4398559501334472805[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_498200963581026915) {
  err_fun(nom_x, delta_x, out_498200963581026915);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_191777889625927103) {
  inv_err_fun(nom_x, true_x, out_191777889625927103);
}
void pose_H_mod_fun(double *state, double *out_8577665774918696216) {
  H_mod_fun(state, out_8577665774918696216);
}
void pose_f_fun(double *state, double dt, double *out_5599332296746008014) {
  f_fun(state,  dt, out_5599332296746008014);
}
void pose_F_fun(double *state, double dt, double *out_1584783165911621436) {
  F_fun(state,  dt, out_1584783165911621436);
}
void pose_h_4(double *state, double *unused, double *out_8170273359625996454) {
  h_4(state, unused, out_8170273359625996454);
}
void pose_H_4(double *state, double *unused, double *out_7481347933629845101) {
  H_4(state, unused, out_7481347933629845101);
}
void pose_h_10(double *state, double *unused, double *out_4791397978297849235) {
  h_10(state, unused, out_4791397978297849235);
}
void pose_H_10(double *state, double *unused, double *out_7583350988764981753) {
  H_10(state, unused, out_7583350988764981753);
}
void pose_h_13(double *state, double *unused, double *out_5529921368018353304) {
  h_13(state, unused, out_5529921368018353304);
}
void pose_H_13(double *state, double *unused, double *out_3647592470327321077) {
  H_13(state, unused, out_3647592470327321077);
}
void pose_h_14(double *state, double *unused, double *out_4744126367926772826) {
  h_14(state, unused, out_4744126367926772826);
}
void pose_H_14(double *state, double *unused, double *out_4398559501334472805) {
  H_14(state, unused, out_4398559501334472805);
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
