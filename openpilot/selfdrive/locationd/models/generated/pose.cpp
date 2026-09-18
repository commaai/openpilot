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
void err_fun(double *nom_x, double *delta_x, double *out_5087435637205728933) {
   out_5087435637205728933[0] = delta_x[0] + nom_x[0];
   out_5087435637205728933[1] = delta_x[1] + nom_x[1];
   out_5087435637205728933[2] = delta_x[2] + nom_x[2];
   out_5087435637205728933[3] = delta_x[3] + nom_x[3];
   out_5087435637205728933[4] = delta_x[4] + nom_x[4];
   out_5087435637205728933[5] = delta_x[5] + nom_x[5];
   out_5087435637205728933[6] = delta_x[6] + nom_x[6];
   out_5087435637205728933[7] = delta_x[7] + nom_x[7];
   out_5087435637205728933[8] = delta_x[8] + nom_x[8];
   out_5087435637205728933[9] = delta_x[9] + nom_x[9];
   out_5087435637205728933[10] = delta_x[10] + nom_x[10];
   out_5087435637205728933[11] = delta_x[11] + nom_x[11];
   out_5087435637205728933[12] = delta_x[12] + nom_x[12];
   out_5087435637205728933[13] = delta_x[13] + nom_x[13];
   out_5087435637205728933[14] = delta_x[14] + nom_x[14];
   out_5087435637205728933[15] = delta_x[15] + nom_x[15];
   out_5087435637205728933[16] = delta_x[16] + nom_x[16];
   out_5087435637205728933[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2851508015392009149) {
   out_2851508015392009149[0] = -nom_x[0] + true_x[0];
   out_2851508015392009149[1] = -nom_x[1] + true_x[1];
   out_2851508015392009149[2] = -nom_x[2] + true_x[2];
   out_2851508015392009149[3] = -nom_x[3] + true_x[3];
   out_2851508015392009149[4] = -nom_x[4] + true_x[4];
   out_2851508015392009149[5] = -nom_x[5] + true_x[5];
   out_2851508015392009149[6] = -nom_x[6] + true_x[6];
   out_2851508015392009149[7] = -nom_x[7] + true_x[7];
   out_2851508015392009149[8] = -nom_x[8] + true_x[8];
   out_2851508015392009149[9] = -nom_x[9] + true_x[9];
   out_2851508015392009149[10] = -nom_x[10] + true_x[10];
   out_2851508015392009149[11] = -nom_x[11] + true_x[11];
   out_2851508015392009149[12] = -nom_x[12] + true_x[12];
   out_2851508015392009149[13] = -nom_x[13] + true_x[13];
   out_2851508015392009149[14] = -nom_x[14] + true_x[14];
   out_2851508015392009149[15] = -nom_x[15] + true_x[15];
   out_2851508015392009149[16] = -nom_x[16] + true_x[16];
   out_2851508015392009149[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3140738153404446410) {
   out_3140738153404446410[0] = 1.0;
   out_3140738153404446410[1] = 0.0;
   out_3140738153404446410[2] = 0.0;
   out_3140738153404446410[3] = 0.0;
   out_3140738153404446410[4] = 0.0;
   out_3140738153404446410[5] = 0.0;
   out_3140738153404446410[6] = 0.0;
   out_3140738153404446410[7] = 0.0;
   out_3140738153404446410[8] = 0.0;
   out_3140738153404446410[9] = 0.0;
   out_3140738153404446410[10] = 0.0;
   out_3140738153404446410[11] = 0.0;
   out_3140738153404446410[12] = 0.0;
   out_3140738153404446410[13] = 0.0;
   out_3140738153404446410[14] = 0.0;
   out_3140738153404446410[15] = 0.0;
   out_3140738153404446410[16] = 0.0;
   out_3140738153404446410[17] = 0.0;
   out_3140738153404446410[18] = 0.0;
   out_3140738153404446410[19] = 1.0;
   out_3140738153404446410[20] = 0.0;
   out_3140738153404446410[21] = 0.0;
   out_3140738153404446410[22] = 0.0;
   out_3140738153404446410[23] = 0.0;
   out_3140738153404446410[24] = 0.0;
   out_3140738153404446410[25] = 0.0;
   out_3140738153404446410[26] = 0.0;
   out_3140738153404446410[27] = 0.0;
   out_3140738153404446410[28] = 0.0;
   out_3140738153404446410[29] = 0.0;
   out_3140738153404446410[30] = 0.0;
   out_3140738153404446410[31] = 0.0;
   out_3140738153404446410[32] = 0.0;
   out_3140738153404446410[33] = 0.0;
   out_3140738153404446410[34] = 0.0;
   out_3140738153404446410[35] = 0.0;
   out_3140738153404446410[36] = 0.0;
   out_3140738153404446410[37] = 0.0;
   out_3140738153404446410[38] = 1.0;
   out_3140738153404446410[39] = 0.0;
   out_3140738153404446410[40] = 0.0;
   out_3140738153404446410[41] = 0.0;
   out_3140738153404446410[42] = 0.0;
   out_3140738153404446410[43] = 0.0;
   out_3140738153404446410[44] = 0.0;
   out_3140738153404446410[45] = 0.0;
   out_3140738153404446410[46] = 0.0;
   out_3140738153404446410[47] = 0.0;
   out_3140738153404446410[48] = 0.0;
   out_3140738153404446410[49] = 0.0;
   out_3140738153404446410[50] = 0.0;
   out_3140738153404446410[51] = 0.0;
   out_3140738153404446410[52] = 0.0;
   out_3140738153404446410[53] = 0.0;
   out_3140738153404446410[54] = 0.0;
   out_3140738153404446410[55] = 0.0;
   out_3140738153404446410[56] = 0.0;
   out_3140738153404446410[57] = 1.0;
   out_3140738153404446410[58] = 0.0;
   out_3140738153404446410[59] = 0.0;
   out_3140738153404446410[60] = 0.0;
   out_3140738153404446410[61] = 0.0;
   out_3140738153404446410[62] = 0.0;
   out_3140738153404446410[63] = 0.0;
   out_3140738153404446410[64] = 0.0;
   out_3140738153404446410[65] = 0.0;
   out_3140738153404446410[66] = 0.0;
   out_3140738153404446410[67] = 0.0;
   out_3140738153404446410[68] = 0.0;
   out_3140738153404446410[69] = 0.0;
   out_3140738153404446410[70] = 0.0;
   out_3140738153404446410[71] = 0.0;
   out_3140738153404446410[72] = 0.0;
   out_3140738153404446410[73] = 0.0;
   out_3140738153404446410[74] = 0.0;
   out_3140738153404446410[75] = 0.0;
   out_3140738153404446410[76] = 1.0;
   out_3140738153404446410[77] = 0.0;
   out_3140738153404446410[78] = 0.0;
   out_3140738153404446410[79] = 0.0;
   out_3140738153404446410[80] = 0.0;
   out_3140738153404446410[81] = 0.0;
   out_3140738153404446410[82] = 0.0;
   out_3140738153404446410[83] = 0.0;
   out_3140738153404446410[84] = 0.0;
   out_3140738153404446410[85] = 0.0;
   out_3140738153404446410[86] = 0.0;
   out_3140738153404446410[87] = 0.0;
   out_3140738153404446410[88] = 0.0;
   out_3140738153404446410[89] = 0.0;
   out_3140738153404446410[90] = 0.0;
   out_3140738153404446410[91] = 0.0;
   out_3140738153404446410[92] = 0.0;
   out_3140738153404446410[93] = 0.0;
   out_3140738153404446410[94] = 0.0;
   out_3140738153404446410[95] = 1.0;
   out_3140738153404446410[96] = 0.0;
   out_3140738153404446410[97] = 0.0;
   out_3140738153404446410[98] = 0.0;
   out_3140738153404446410[99] = 0.0;
   out_3140738153404446410[100] = 0.0;
   out_3140738153404446410[101] = 0.0;
   out_3140738153404446410[102] = 0.0;
   out_3140738153404446410[103] = 0.0;
   out_3140738153404446410[104] = 0.0;
   out_3140738153404446410[105] = 0.0;
   out_3140738153404446410[106] = 0.0;
   out_3140738153404446410[107] = 0.0;
   out_3140738153404446410[108] = 0.0;
   out_3140738153404446410[109] = 0.0;
   out_3140738153404446410[110] = 0.0;
   out_3140738153404446410[111] = 0.0;
   out_3140738153404446410[112] = 0.0;
   out_3140738153404446410[113] = 0.0;
   out_3140738153404446410[114] = 1.0;
   out_3140738153404446410[115] = 0.0;
   out_3140738153404446410[116] = 0.0;
   out_3140738153404446410[117] = 0.0;
   out_3140738153404446410[118] = 0.0;
   out_3140738153404446410[119] = 0.0;
   out_3140738153404446410[120] = 0.0;
   out_3140738153404446410[121] = 0.0;
   out_3140738153404446410[122] = 0.0;
   out_3140738153404446410[123] = 0.0;
   out_3140738153404446410[124] = 0.0;
   out_3140738153404446410[125] = 0.0;
   out_3140738153404446410[126] = 0.0;
   out_3140738153404446410[127] = 0.0;
   out_3140738153404446410[128] = 0.0;
   out_3140738153404446410[129] = 0.0;
   out_3140738153404446410[130] = 0.0;
   out_3140738153404446410[131] = 0.0;
   out_3140738153404446410[132] = 0.0;
   out_3140738153404446410[133] = 1.0;
   out_3140738153404446410[134] = 0.0;
   out_3140738153404446410[135] = 0.0;
   out_3140738153404446410[136] = 0.0;
   out_3140738153404446410[137] = 0.0;
   out_3140738153404446410[138] = 0.0;
   out_3140738153404446410[139] = 0.0;
   out_3140738153404446410[140] = 0.0;
   out_3140738153404446410[141] = 0.0;
   out_3140738153404446410[142] = 0.0;
   out_3140738153404446410[143] = 0.0;
   out_3140738153404446410[144] = 0.0;
   out_3140738153404446410[145] = 0.0;
   out_3140738153404446410[146] = 0.0;
   out_3140738153404446410[147] = 0.0;
   out_3140738153404446410[148] = 0.0;
   out_3140738153404446410[149] = 0.0;
   out_3140738153404446410[150] = 0.0;
   out_3140738153404446410[151] = 0.0;
   out_3140738153404446410[152] = 1.0;
   out_3140738153404446410[153] = 0.0;
   out_3140738153404446410[154] = 0.0;
   out_3140738153404446410[155] = 0.0;
   out_3140738153404446410[156] = 0.0;
   out_3140738153404446410[157] = 0.0;
   out_3140738153404446410[158] = 0.0;
   out_3140738153404446410[159] = 0.0;
   out_3140738153404446410[160] = 0.0;
   out_3140738153404446410[161] = 0.0;
   out_3140738153404446410[162] = 0.0;
   out_3140738153404446410[163] = 0.0;
   out_3140738153404446410[164] = 0.0;
   out_3140738153404446410[165] = 0.0;
   out_3140738153404446410[166] = 0.0;
   out_3140738153404446410[167] = 0.0;
   out_3140738153404446410[168] = 0.0;
   out_3140738153404446410[169] = 0.0;
   out_3140738153404446410[170] = 0.0;
   out_3140738153404446410[171] = 1.0;
   out_3140738153404446410[172] = 0.0;
   out_3140738153404446410[173] = 0.0;
   out_3140738153404446410[174] = 0.0;
   out_3140738153404446410[175] = 0.0;
   out_3140738153404446410[176] = 0.0;
   out_3140738153404446410[177] = 0.0;
   out_3140738153404446410[178] = 0.0;
   out_3140738153404446410[179] = 0.0;
   out_3140738153404446410[180] = 0.0;
   out_3140738153404446410[181] = 0.0;
   out_3140738153404446410[182] = 0.0;
   out_3140738153404446410[183] = 0.0;
   out_3140738153404446410[184] = 0.0;
   out_3140738153404446410[185] = 0.0;
   out_3140738153404446410[186] = 0.0;
   out_3140738153404446410[187] = 0.0;
   out_3140738153404446410[188] = 0.0;
   out_3140738153404446410[189] = 0.0;
   out_3140738153404446410[190] = 1.0;
   out_3140738153404446410[191] = 0.0;
   out_3140738153404446410[192] = 0.0;
   out_3140738153404446410[193] = 0.0;
   out_3140738153404446410[194] = 0.0;
   out_3140738153404446410[195] = 0.0;
   out_3140738153404446410[196] = 0.0;
   out_3140738153404446410[197] = 0.0;
   out_3140738153404446410[198] = 0.0;
   out_3140738153404446410[199] = 0.0;
   out_3140738153404446410[200] = 0.0;
   out_3140738153404446410[201] = 0.0;
   out_3140738153404446410[202] = 0.0;
   out_3140738153404446410[203] = 0.0;
   out_3140738153404446410[204] = 0.0;
   out_3140738153404446410[205] = 0.0;
   out_3140738153404446410[206] = 0.0;
   out_3140738153404446410[207] = 0.0;
   out_3140738153404446410[208] = 0.0;
   out_3140738153404446410[209] = 1.0;
   out_3140738153404446410[210] = 0.0;
   out_3140738153404446410[211] = 0.0;
   out_3140738153404446410[212] = 0.0;
   out_3140738153404446410[213] = 0.0;
   out_3140738153404446410[214] = 0.0;
   out_3140738153404446410[215] = 0.0;
   out_3140738153404446410[216] = 0.0;
   out_3140738153404446410[217] = 0.0;
   out_3140738153404446410[218] = 0.0;
   out_3140738153404446410[219] = 0.0;
   out_3140738153404446410[220] = 0.0;
   out_3140738153404446410[221] = 0.0;
   out_3140738153404446410[222] = 0.0;
   out_3140738153404446410[223] = 0.0;
   out_3140738153404446410[224] = 0.0;
   out_3140738153404446410[225] = 0.0;
   out_3140738153404446410[226] = 0.0;
   out_3140738153404446410[227] = 0.0;
   out_3140738153404446410[228] = 1.0;
   out_3140738153404446410[229] = 0.0;
   out_3140738153404446410[230] = 0.0;
   out_3140738153404446410[231] = 0.0;
   out_3140738153404446410[232] = 0.0;
   out_3140738153404446410[233] = 0.0;
   out_3140738153404446410[234] = 0.0;
   out_3140738153404446410[235] = 0.0;
   out_3140738153404446410[236] = 0.0;
   out_3140738153404446410[237] = 0.0;
   out_3140738153404446410[238] = 0.0;
   out_3140738153404446410[239] = 0.0;
   out_3140738153404446410[240] = 0.0;
   out_3140738153404446410[241] = 0.0;
   out_3140738153404446410[242] = 0.0;
   out_3140738153404446410[243] = 0.0;
   out_3140738153404446410[244] = 0.0;
   out_3140738153404446410[245] = 0.0;
   out_3140738153404446410[246] = 0.0;
   out_3140738153404446410[247] = 1.0;
   out_3140738153404446410[248] = 0.0;
   out_3140738153404446410[249] = 0.0;
   out_3140738153404446410[250] = 0.0;
   out_3140738153404446410[251] = 0.0;
   out_3140738153404446410[252] = 0.0;
   out_3140738153404446410[253] = 0.0;
   out_3140738153404446410[254] = 0.0;
   out_3140738153404446410[255] = 0.0;
   out_3140738153404446410[256] = 0.0;
   out_3140738153404446410[257] = 0.0;
   out_3140738153404446410[258] = 0.0;
   out_3140738153404446410[259] = 0.0;
   out_3140738153404446410[260] = 0.0;
   out_3140738153404446410[261] = 0.0;
   out_3140738153404446410[262] = 0.0;
   out_3140738153404446410[263] = 0.0;
   out_3140738153404446410[264] = 0.0;
   out_3140738153404446410[265] = 0.0;
   out_3140738153404446410[266] = 1.0;
   out_3140738153404446410[267] = 0.0;
   out_3140738153404446410[268] = 0.0;
   out_3140738153404446410[269] = 0.0;
   out_3140738153404446410[270] = 0.0;
   out_3140738153404446410[271] = 0.0;
   out_3140738153404446410[272] = 0.0;
   out_3140738153404446410[273] = 0.0;
   out_3140738153404446410[274] = 0.0;
   out_3140738153404446410[275] = 0.0;
   out_3140738153404446410[276] = 0.0;
   out_3140738153404446410[277] = 0.0;
   out_3140738153404446410[278] = 0.0;
   out_3140738153404446410[279] = 0.0;
   out_3140738153404446410[280] = 0.0;
   out_3140738153404446410[281] = 0.0;
   out_3140738153404446410[282] = 0.0;
   out_3140738153404446410[283] = 0.0;
   out_3140738153404446410[284] = 0.0;
   out_3140738153404446410[285] = 1.0;
   out_3140738153404446410[286] = 0.0;
   out_3140738153404446410[287] = 0.0;
   out_3140738153404446410[288] = 0.0;
   out_3140738153404446410[289] = 0.0;
   out_3140738153404446410[290] = 0.0;
   out_3140738153404446410[291] = 0.0;
   out_3140738153404446410[292] = 0.0;
   out_3140738153404446410[293] = 0.0;
   out_3140738153404446410[294] = 0.0;
   out_3140738153404446410[295] = 0.0;
   out_3140738153404446410[296] = 0.0;
   out_3140738153404446410[297] = 0.0;
   out_3140738153404446410[298] = 0.0;
   out_3140738153404446410[299] = 0.0;
   out_3140738153404446410[300] = 0.0;
   out_3140738153404446410[301] = 0.0;
   out_3140738153404446410[302] = 0.0;
   out_3140738153404446410[303] = 0.0;
   out_3140738153404446410[304] = 1.0;
   out_3140738153404446410[305] = 0.0;
   out_3140738153404446410[306] = 0.0;
   out_3140738153404446410[307] = 0.0;
   out_3140738153404446410[308] = 0.0;
   out_3140738153404446410[309] = 0.0;
   out_3140738153404446410[310] = 0.0;
   out_3140738153404446410[311] = 0.0;
   out_3140738153404446410[312] = 0.0;
   out_3140738153404446410[313] = 0.0;
   out_3140738153404446410[314] = 0.0;
   out_3140738153404446410[315] = 0.0;
   out_3140738153404446410[316] = 0.0;
   out_3140738153404446410[317] = 0.0;
   out_3140738153404446410[318] = 0.0;
   out_3140738153404446410[319] = 0.0;
   out_3140738153404446410[320] = 0.0;
   out_3140738153404446410[321] = 0.0;
   out_3140738153404446410[322] = 0.0;
   out_3140738153404446410[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8261207861188225982) {
   out_8261207861188225982[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8261207861188225982[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8261207861188225982[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8261207861188225982[3] = dt*state[12] + state[3];
   out_8261207861188225982[4] = dt*state[13] + state[4];
   out_8261207861188225982[5] = dt*state[14] + state[5];
   out_8261207861188225982[6] = state[6];
   out_8261207861188225982[7] = state[7];
   out_8261207861188225982[8] = state[8];
   out_8261207861188225982[9] = state[9];
   out_8261207861188225982[10] = state[10];
   out_8261207861188225982[11] = state[11];
   out_8261207861188225982[12] = state[12];
   out_8261207861188225982[13] = state[13];
   out_8261207861188225982[14] = state[14];
   out_8261207861188225982[15] = state[15];
   out_8261207861188225982[16] = state[16];
   out_8261207861188225982[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2195864972581525307) {
   out_2195864972581525307[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2195864972581525307[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2195864972581525307[2] = 0;
   out_2195864972581525307[3] = 0;
   out_2195864972581525307[4] = 0;
   out_2195864972581525307[5] = 0;
   out_2195864972581525307[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2195864972581525307[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2195864972581525307[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2195864972581525307[9] = 0;
   out_2195864972581525307[10] = 0;
   out_2195864972581525307[11] = 0;
   out_2195864972581525307[12] = 0;
   out_2195864972581525307[13] = 0;
   out_2195864972581525307[14] = 0;
   out_2195864972581525307[15] = 0;
   out_2195864972581525307[16] = 0;
   out_2195864972581525307[17] = 0;
   out_2195864972581525307[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2195864972581525307[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2195864972581525307[20] = 0;
   out_2195864972581525307[21] = 0;
   out_2195864972581525307[22] = 0;
   out_2195864972581525307[23] = 0;
   out_2195864972581525307[24] = 0;
   out_2195864972581525307[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2195864972581525307[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2195864972581525307[27] = 0;
   out_2195864972581525307[28] = 0;
   out_2195864972581525307[29] = 0;
   out_2195864972581525307[30] = 0;
   out_2195864972581525307[31] = 0;
   out_2195864972581525307[32] = 0;
   out_2195864972581525307[33] = 0;
   out_2195864972581525307[34] = 0;
   out_2195864972581525307[35] = 0;
   out_2195864972581525307[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2195864972581525307[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2195864972581525307[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2195864972581525307[39] = 0;
   out_2195864972581525307[40] = 0;
   out_2195864972581525307[41] = 0;
   out_2195864972581525307[42] = 0;
   out_2195864972581525307[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2195864972581525307[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2195864972581525307[45] = 0;
   out_2195864972581525307[46] = 0;
   out_2195864972581525307[47] = 0;
   out_2195864972581525307[48] = 0;
   out_2195864972581525307[49] = 0;
   out_2195864972581525307[50] = 0;
   out_2195864972581525307[51] = 0;
   out_2195864972581525307[52] = 0;
   out_2195864972581525307[53] = 0;
   out_2195864972581525307[54] = 0;
   out_2195864972581525307[55] = 0;
   out_2195864972581525307[56] = 0;
   out_2195864972581525307[57] = 1;
   out_2195864972581525307[58] = 0;
   out_2195864972581525307[59] = 0;
   out_2195864972581525307[60] = 0;
   out_2195864972581525307[61] = 0;
   out_2195864972581525307[62] = 0;
   out_2195864972581525307[63] = 0;
   out_2195864972581525307[64] = 0;
   out_2195864972581525307[65] = 0;
   out_2195864972581525307[66] = dt;
   out_2195864972581525307[67] = 0;
   out_2195864972581525307[68] = 0;
   out_2195864972581525307[69] = 0;
   out_2195864972581525307[70] = 0;
   out_2195864972581525307[71] = 0;
   out_2195864972581525307[72] = 0;
   out_2195864972581525307[73] = 0;
   out_2195864972581525307[74] = 0;
   out_2195864972581525307[75] = 0;
   out_2195864972581525307[76] = 1;
   out_2195864972581525307[77] = 0;
   out_2195864972581525307[78] = 0;
   out_2195864972581525307[79] = 0;
   out_2195864972581525307[80] = 0;
   out_2195864972581525307[81] = 0;
   out_2195864972581525307[82] = 0;
   out_2195864972581525307[83] = 0;
   out_2195864972581525307[84] = 0;
   out_2195864972581525307[85] = dt;
   out_2195864972581525307[86] = 0;
   out_2195864972581525307[87] = 0;
   out_2195864972581525307[88] = 0;
   out_2195864972581525307[89] = 0;
   out_2195864972581525307[90] = 0;
   out_2195864972581525307[91] = 0;
   out_2195864972581525307[92] = 0;
   out_2195864972581525307[93] = 0;
   out_2195864972581525307[94] = 0;
   out_2195864972581525307[95] = 1;
   out_2195864972581525307[96] = 0;
   out_2195864972581525307[97] = 0;
   out_2195864972581525307[98] = 0;
   out_2195864972581525307[99] = 0;
   out_2195864972581525307[100] = 0;
   out_2195864972581525307[101] = 0;
   out_2195864972581525307[102] = 0;
   out_2195864972581525307[103] = 0;
   out_2195864972581525307[104] = dt;
   out_2195864972581525307[105] = 0;
   out_2195864972581525307[106] = 0;
   out_2195864972581525307[107] = 0;
   out_2195864972581525307[108] = 0;
   out_2195864972581525307[109] = 0;
   out_2195864972581525307[110] = 0;
   out_2195864972581525307[111] = 0;
   out_2195864972581525307[112] = 0;
   out_2195864972581525307[113] = 0;
   out_2195864972581525307[114] = 1;
   out_2195864972581525307[115] = 0;
   out_2195864972581525307[116] = 0;
   out_2195864972581525307[117] = 0;
   out_2195864972581525307[118] = 0;
   out_2195864972581525307[119] = 0;
   out_2195864972581525307[120] = 0;
   out_2195864972581525307[121] = 0;
   out_2195864972581525307[122] = 0;
   out_2195864972581525307[123] = 0;
   out_2195864972581525307[124] = 0;
   out_2195864972581525307[125] = 0;
   out_2195864972581525307[126] = 0;
   out_2195864972581525307[127] = 0;
   out_2195864972581525307[128] = 0;
   out_2195864972581525307[129] = 0;
   out_2195864972581525307[130] = 0;
   out_2195864972581525307[131] = 0;
   out_2195864972581525307[132] = 0;
   out_2195864972581525307[133] = 1;
   out_2195864972581525307[134] = 0;
   out_2195864972581525307[135] = 0;
   out_2195864972581525307[136] = 0;
   out_2195864972581525307[137] = 0;
   out_2195864972581525307[138] = 0;
   out_2195864972581525307[139] = 0;
   out_2195864972581525307[140] = 0;
   out_2195864972581525307[141] = 0;
   out_2195864972581525307[142] = 0;
   out_2195864972581525307[143] = 0;
   out_2195864972581525307[144] = 0;
   out_2195864972581525307[145] = 0;
   out_2195864972581525307[146] = 0;
   out_2195864972581525307[147] = 0;
   out_2195864972581525307[148] = 0;
   out_2195864972581525307[149] = 0;
   out_2195864972581525307[150] = 0;
   out_2195864972581525307[151] = 0;
   out_2195864972581525307[152] = 1;
   out_2195864972581525307[153] = 0;
   out_2195864972581525307[154] = 0;
   out_2195864972581525307[155] = 0;
   out_2195864972581525307[156] = 0;
   out_2195864972581525307[157] = 0;
   out_2195864972581525307[158] = 0;
   out_2195864972581525307[159] = 0;
   out_2195864972581525307[160] = 0;
   out_2195864972581525307[161] = 0;
   out_2195864972581525307[162] = 0;
   out_2195864972581525307[163] = 0;
   out_2195864972581525307[164] = 0;
   out_2195864972581525307[165] = 0;
   out_2195864972581525307[166] = 0;
   out_2195864972581525307[167] = 0;
   out_2195864972581525307[168] = 0;
   out_2195864972581525307[169] = 0;
   out_2195864972581525307[170] = 0;
   out_2195864972581525307[171] = 1;
   out_2195864972581525307[172] = 0;
   out_2195864972581525307[173] = 0;
   out_2195864972581525307[174] = 0;
   out_2195864972581525307[175] = 0;
   out_2195864972581525307[176] = 0;
   out_2195864972581525307[177] = 0;
   out_2195864972581525307[178] = 0;
   out_2195864972581525307[179] = 0;
   out_2195864972581525307[180] = 0;
   out_2195864972581525307[181] = 0;
   out_2195864972581525307[182] = 0;
   out_2195864972581525307[183] = 0;
   out_2195864972581525307[184] = 0;
   out_2195864972581525307[185] = 0;
   out_2195864972581525307[186] = 0;
   out_2195864972581525307[187] = 0;
   out_2195864972581525307[188] = 0;
   out_2195864972581525307[189] = 0;
   out_2195864972581525307[190] = 1;
   out_2195864972581525307[191] = 0;
   out_2195864972581525307[192] = 0;
   out_2195864972581525307[193] = 0;
   out_2195864972581525307[194] = 0;
   out_2195864972581525307[195] = 0;
   out_2195864972581525307[196] = 0;
   out_2195864972581525307[197] = 0;
   out_2195864972581525307[198] = 0;
   out_2195864972581525307[199] = 0;
   out_2195864972581525307[200] = 0;
   out_2195864972581525307[201] = 0;
   out_2195864972581525307[202] = 0;
   out_2195864972581525307[203] = 0;
   out_2195864972581525307[204] = 0;
   out_2195864972581525307[205] = 0;
   out_2195864972581525307[206] = 0;
   out_2195864972581525307[207] = 0;
   out_2195864972581525307[208] = 0;
   out_2195864972581525307[209] = 1;
   out_2195864972581525307[210] = 0;
   out_2195864972581525307[211] = 0;
   out_2195864972581525307[212] = 0;
   out_2195864972581525307[213] = 0;
   out_2195864972581525307[214] = 0;
   out_2195864972581525307[215] = 0;
   out_2195864972581525307[216] = 0;
   out_2195864972581525307[217] = 0;
   out_2195864972581525307[218] = 0;
   out_2195864972581525307[219] = 0;
   out_2195864972581525307[220] = 0;
   out_2195864972581525307[221] = 0;
   out_2195864972581525307[222] = 0;
   out_2195864972581525307[223] = 0;
   out_2195864972581525307[224] = 0;
   out_2195864972581525307[225] = 0;
   out_2195864972581525307[226] = 0;
   out_2195864972581525307[227] = 0;
   out_2195864972581525307[228] = 1;
   out_2195864972581525307[229] = 0;
   out_2195864972581525307[230] = 0;
   out_2195864972581525307[231] = 0;
   out_2195864972581525307[232] = 0;
   out_2195864972581525307[233] = 0;
   out_2195864972581525307[234] = 0;
   out_2195864972581525307[235] = 0;
   out_2195864972581525307[236] = 0;
   out_2195864972581525307[237] = 0;
   out_2195864972581525307[238] = 0;
   out_2195864972581525307[239] = 0;
   out_2195864972581525307[240] = 0;
   out_2195864972581525307[241] = 0;
   out_2195864972581525307[242] = 0;
   out_2195864972581525307[243] = 0;
   out_2195864972581525307[244] = 0;
   out_2195864972581525307[245] = 0;
   out_2195864972581525307[246] = 0;
   out_2195864972581525307[247] = 1;
   out_2195864972581525307[248] = 0;
   out_2195864972581525307[249] = 0;
   out_2195864972581525307[250] = 0;
   out_2195864972581525307[251] = 0;
   out_2195864972581525307[252] = 0;
   out_2195864972581525307[253] = 0;
   out_2195864972581525307[254] = 0;
   out_2195864972581525307[255] = 0;
   out_2195864972581525307[256] = 0;
   out_2195864972581525307[257] = 0;
   out_2195864972581525307[258] = 0;
   out_2195864972581525307[259] = 0;
   out_2195864972581525307[260] = 0;
   out_2195864972581525307[261] = 0;
   out_2195864972581525307[262] = 0;
   out_2195864972581525307[263] = 0;
   out_2195864972581525307[264] = 0;
   out_2195864972581525307[265] = 0;
   out_2195864972581525307[266] = 1;
   out_2195864972581525307[267] = 0;
   out_2195864972581525307[268] = 0;
   out_2195864972581525307[269] = 0;
   out_2195864972581525307[270] = 0;
   out_2195864972581525307[271] = 0;
   out_2195864972581525307[272] = 0;
   out_2195864972581525307[273] = 0;
   out_2195864972581525307[274] = 0;
   out_2195864972581525307[275] = 0;
   out_2195864972581525307[276] = 0;
   out_2195864972581525307[277] = 0;
   out_2195864972581525307[278] = 0;
   out_2195864972581525307[279] = 0;
   out_2195864972581525307[280] = 0;
   out_2195864972581525307[281] = 0;
   out_2195864972581525307[282] = 0;
   out_2195864972581525307[283] = 0;
   out_2195864972581525307[284] = 0;
   out_2195864972581525307[285] = 1;
   out_2195864972581525307[286] = 0;
   out_2195864972581525307[287] = 0;
   out_2195864972581525307[288] = 0;
   out_2195864972581525307[289] = 0;
   out_2195864972581525307[290] = 0;
   out_2195864972581525307[291] = 0;
   out_2195864972581525307[292] = 0;
   out_2195864972581525307[293] = 0;
   out_2195864972581525307[294] = 0;
   out_2195864972581525307[295] = 0;
   out_2195864972581525307[296] = 0;
   out_2195864972581525307[297] = 0;
   out_2195864972581525307[298] = 0;
   out_2195864972581525307[299] = 0;
   out_2195864972581525307[300] = 0;
   out_2195864972581525307[301] = 0;
   out_2195864972581525307[302] = 0;
   out_2195864972581525307[303] = 0;
   out_2195864972581525307[304] = 1;
   out_2195864972581525307[305] = 0;
   out_2195864972581525307[306] = 0;
   out_2195864972581525307[307] = 0;
   out_2195864972581525307[308] = 0;
   out_2195864972581525307[309] = 0;
   out_2195864972581525307[310] = 0;
   out_2195864972581525307[311] = 0;
   out_2195864972581525307[312] = 0;
   out_2195864972581525307[313] = 0;
   out_2195864972581525307[314] = 0;
   out_2195864972581525307[315] = 0;
   out_2195864972581525307[316] = 0;
   out_2195864972581525307[317] = 0;
   out_2195864972581525307[318] = 0;
   out_2195864972581525307[319] = 0;
   out_2195864972581525307[320] = 0;
   out_2195864972581525307[321] = 0;
   out_2195864972581525307[322] = 0;
   out_2195864972581525307[323] = 1;
}
void h_4(double *state, double *unused, double *out_7615763258081257669) {
   out_7615763258081257669[0] = state[6] + state[9];
   out_7615763258081257669[1] = state[7] + state[10];
   out_7615763258081257669[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4338685635287344514) {
   out_4338685635287344514[0] = 0;
   out_4338685635287344514[1] = 0;
   out_4338685635287344514[2] = 0;
   out_4338685635287344514[3] = 0;
   out_4338685635287344514[4] = 0;
   out_4338685635287344514[5] = 0;
   out_4338685635287344514[6] = 1;
   out_4338685635287344514[7] = 0;
   out_4338685635287344514[8] = 0;
   out_4338685635287344514[9] = 1;
   out_4338685635287344514[10] = 0;
   out_4338685635287344514[11] = 0;
   out_4338685635287344514[12] = 0;
   out_4338685635287344514[13] = 0;
   out_4338685635287344514[14] = 0;
   out_4338685635287344514[15] = 0;
   out_4338685635287344514[16] = 0;
   out_4338685635287344514[17] = 0;
   out_4338685635287344514[18] = 0;
   out_4338685635287344514[19] = 0;
   out_4338685635287344514[20] = 0;
   out_4338685635287344514[21] = 0;
   out_4338685635287344514[22] = 0;
   out_4338685635287344514[23] = 0;
   out_4338685635287344514[24] = 0;
   out_4338685635287344514[25] = 1;
   out_4338685635287344514[26] = 0;
   out_4338685635287344514[27] = 0;
   out_4338685635287344514[28] = 1;
   out_4338685635287344514[29] = 0;
   out_4338685635287344514[30] = 0;
   out_4338685635287344514[31] = 0;
   out_4338685635287344514[32] = 0;
   out_4338685635287344514[33] = 0;
   out_4338685635287344514[34] = 0;
   out_4338685635287344514[35] = 0;
   out_4338685635287344514[36] = 0;
   out_4338685635287344514[37] = 0;
   out_4338685635287344514[38] = 0;
   out_4338685635287344514[39] = 0;
   out_4338685635287344514[40] = 0;
   out_4338685635287344514[41] = 0;
   out_4338685635287344514[42] = 0;
   out_4338685635287344514[43] = 0;
   out_4338685635287344514[44] = 1;
   out_4338685635287344514[45] = 0;
   out_4338685635287344514[46] = 0;
   out_4338685635287344514[47] = 1;
   out_4338685635287344514[48] = 0;
   out_4338685635287344514[49] = 0;
   out_4338685635287344514[50] = 0;
   out_4338685635287344514[51] = 0;
   out_4338685635287344514[52] = 0;
   out_4338685635287344514[53] = 0;
}
void h_10(double *state, double *unused, double *out_4081004568469840555) {
   out_4081004568469840555[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_4081004568469840555[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_4081004568469840555[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2118589289993173556) {
   out_2118589289993173556[0] = 0;
   out_2118589289993173556[1] = 9.8100000000000005*cos(state[1]);
   out_2118589289993173556[2] = 0;
   out_2118589289993173556[3] = 0;
   out_2118589289993173556[4] = -state[8];
   out_2118589289993173556[5] = state[7];
   out_2118589289993173556[6] = 0;
   out_2118589289993173556[7] = state[5];
   out_2118589289993173556[8] = -state[4];
   out_2118589289993173556[9] = 0;
   out_2118589289993173556[10] = 0;
   out_2118589289993173556[11] = 0;
   out_2118589289993173556[12] = 1;
   out_2118589289993173556[13] = 0;
   out_2118589289993173556[14] = 0;
   out_2118589289993173556[15] = 1;
   out_2118589289993173556[16] = 0;
   out_2118589289993173556[17] = 0;
   out_2118589289993173556[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2118589289993173556[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2118589289993173556[20] = 0;
   out_2118589289993173556[21] = state[8];
   out_2118589289993173556[22] = 0;
   out_2118589289993173556[23] = -state[6];
   out_2118589289993173556[24] = -state[5];
   out_2118589289993173556[25] = 0;
   out_2118589289993173556[26] = state[3];
   out_2118589289993173556[27] = 0;
   out_2118589289993173556[28] = 0;
   out_2118589289993173556[29] = 0;
   out_2118589289993173556[30] = 0;
   out_2118589289993173556[31] = 1;
   out_2118589289993173556[32] = 0;
   out_2118589289993173556[33] = 0;
   out_2118589289993173556[34] = 1;
   out_2118589289993173556[35] = 0;
   out_2118589289993173556[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2118589289993173556[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2118589289993173556[38] = 0;
   out_2118589289993173556[39] = -state[7];
   out_2118589289993173556[40] = state[6];
   out_2118589289993173556[41] = 0;
   out_2118589289993173556[42] = state[4];
   out_2118589289993173556[43] = -state[3];
   out_2118589289993173556[44] = 0;
   out_2118589289993173556[45] = 0;
   out_2118589289993173556[46] = 0;
   out_2118589289993173556[47] = 0;
   out_2118589289993173556[48] = 0;
   out_2118589289993173556[49] = 0;
   out_2118589289993173556[50] = 1;
   out_2118589289993173556[51] = 0;
   out_2118589289993173556[52] = 0;
   out_2118589289993173556[53] = 1;
}
void h_13(double *state, double *unused, double *out_7322787283816672919) {
   out_7322787283816672919[0] = state[3];
   out_7322787283816672919[1] = state[4];
   out_7322787283816672919[2] = state[5];
}
void H_13(double *state, double *unused, double *out_504930171984820490) {
   out_504930171984820490[0] = 0;
   out_504930171984820490[1] = 0;
   out_504930171984820490[2] = 0;
   out_504930171984820490[3] = 1;
   out_504930171984820490[4] = 0;
   out_504930171984820490[5] = 0;
   out_504930171984820490[6] = 0;
   out_504930171984820490[7] = 0;
   out_504930171984820490[8] = 0;
   out_504930171984820490[9] = 0;
   out_504930171984820490[10] = 0;
   out_504930171984820490[11] = 0;
   out_504930171984820490[12] = 0;
   out_504930171984820490[13] = 0;
   out_504930171984820490[14] = 0;
   out_504930171984820490[15] = 0;
   out_504930171984820490[16] = 0;
   out_504930171984820490[17] = 0;
   out_504930171984820490[18] = 0;
   out_504930171984820490[19] = 0;
   out_504930171984820490[20] = 0;
   out_504930171984820490[21] = 0;
   out_504930171984820490[22] = 1;
   out_504930171984820490[23] = 0;
   out_504930171984820490[24] = 0;
   out_504930171984820490[25] = 0;
   out_504930171984820490[26] = 0;
   out_504930171984820490[27] = 0;
   out_504930171984820490[28] = 0;
   out_504930171984820490[29] = 0;
   out_504930171984820490[30] = 0;
   out_504930171984820490[31] = 0;
   out_504930171984820490[32] = 0;
   out_504930171984820490[33] = 0;
   out_504930171984820490[34] = 0;
   out_504930171984820490[35] = 0;
   out_504930171984820490[36] = 0;
   out_504930171984820490[37] = 0;
   out_504930171984820490[38] = 0;
   out_504930171984820490[39] = 0;
   out_504930171984820490[40] = 0;
   out_504930171984820490[41] = 1;
   out_504930171984820490[42] = 0;
   out_504930171984820490[43] = 0;
   out_504930171984820490[44] = 0;
   out_504930171984820490[45] = 0;
   out_504930171984820490[46] = 0;
   out_504930171984820490[47] = 0;
   out_504930171984820490[48] = 0;
   out_504930171984820490[49] = 0;
   out_504930171984820490[50] = 0;
   out_504930171984820490[51] = 0;
   out_504930171984820490[52] = 0;
   out_504930171984820490[53] = 0;
}
void h_14(double *state, double *unused, double *out_2194728518549899811) {
   out_2194728518549899811[0] = state[6];
   out_2194728518549899811[1] = state[7];
   out_2194728518549899811[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1255897202991972218) {
   out_1255897202991972218[0] = 0;
   out_1255897202991972218[1] = 0;
   out_1255897202991972218[2] = 0;
   out_1255897202991972218[3] = 0;
   out_1255897202991972218[4] = 0;
   out_1255897202991972218[5] = 0;
   out_1255897202991972218[6] = 1;
   out_1255897202991972218[7] = 0;
   out_1255897202991972218[8] = 0;
   out_1255897202991972218[9] = 0;
   out_1255897202991972218[10] = 0;
   out_1255897202991972218[11] = 0;
   out_1255897202991972218[12] = 0;
   out_1255897202991972218[13] = 0;
   out_1255897202991972218[14] = 0;
   out_1255897202991972218[15] = 0;
   out_1255897202991972218[16] = 0;
   out_1255897202991972218[17] = 0;
   out_1255897202991972218[18] = 0;
   out_1255897202991972218[19] = 0;
   out_1255897202991972218[20] = 0;
   out_1255897202991972218[21] = 0;
   out_1255897202991972218[22] = 0;
   out_1255897202991972218[23] = 0;
   out_1255897202991972218[24] = 0;
   out_1255897202991972218[25] = 1;
   out_1255897202991972218[26] = 0;
   out_1255897202991972218[27] = 0;
   out_1255897202991972218[28] = 0;
   out_1255897202991972218[29] = 0;
   out_1255897202991972218[30] = 0;
   out_1255897202991972218[31] = 0;
   out_1255897202991972218[32] = 0;
   out_1255897202991972218[33] = 0;
   out_1255897202991972218[34] = 0;
   out_1255897202991972218[35] = 0;
   out_1255897202991972218[36] = 0;
   out_1255897202991972218[37] = 0;
   out_1255897202991972218[38] = 0;
   out_1255897202991972218[39] = 0;
   out_1255897202991972218[40] = 0;
   out_1255897202991972218[41] = 0;
   out_1255897202991972218[42] = 0;
   out_1255897202991972218[43] = 0;
   out_1255897202991972218[44] = 1;
   out_1255897202991972218[45] = 0;
   out_1255897202991972218[46] = 0;
   out_1255897202991972218[47] = 0;
   out_1255897202991972218[48] = 0;
   out_1255897202991972218[49] = 0;
   out_1255897202991972218[50] = 0;
   out_1255897202991972218[51] = 0;
   out_1255897202991972218[52] = 0;
   out_1255897202991972218[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_5087435637205728933) {
  err_fun(nom_x, delta_x, out_5087435637205728933);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2851508015392009149) {
  inv_err_fun(nom_x, true_x, out_2851508015392009149);
}
void pose_H_mod_fun(double *state, double *out_3140738153404446410) {
  H_mod_fun(state, out_3140738153404446410);
}
void pose_f_fun(double *state, double dt, double *out_8261207861188225982) {
  f_fun(state,  dt, out_8261207861188225982);
}
void pose_F_fun(double *state, double dt, double *out_2195864972581525307) {
  F_fun(state,  dt, out_2195864972581525307);
}
void pose_h_4(double *state, double *unused, double *out_7615763258081257669) {
  h_4(state, unused, out_7615763258081257669);
}
void pose_H_4(double *state, double *unused, double *out_4338685635287344514) {
  H_4(state, unused, out_4338685635287344514);
}
void pose_h_10(double *state, double *unused, double *out_4081004568469840555) {
  h_10(state, unused, out_4081004568469840555);
}
void pose_H_10(double *state, double *unused, double *out_2118589289993173556) {
  H_10(state, unused, out_2118589289993173556);
}
void pose_h_13(double *state, double *unused, double *out_7322787283816672919) {
  h_13(state, unused, out_7322787283816672919);
}
void pose_H_13(double *state, double *unused, double *out_504930171984820490) {
  H_13(state, unused, out_504930171984820490);
}
void pose_h_14(double *state, double *unused, double *out_2194728518549899811) {
  h_14(state, unused, out_2194728518549899811);
}
void pose_H_14(double *state, double *unused, double *out_1255897202991972218) {
  H_14(state, unused, out_1255897202991972218);
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
