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
void err_fun(double *nom_x, double *delta_x, double *out_8310779210450879699) {
   out_8310779210450879699[0] = delta_x[0] + nom_x[0];
   out_8310779210450879699[1] = delta_x[1] + nom_x[1];
   out_8310779210450879699[2] = delta_x[2] + nom_x[2];
   out_8310779210450879699[3] = delta_x[3] + nom_x[3];
   out_8310779210450879699[4] = delta_x[4] + nom_x[4];
   out_8310779210450879699[5] = delta_x[5] + nom_x[5];
   out_8310779210450879699[6] = delta_x[6] + nom_x[6];
   out_8310779210450879699[7] = delta_x[7] + nom_x[7];
   out_8310779210450879699[8] = delta_x[8] + nom_x[8];
   out_8310779210450879699[9] = delta_x[9] + nom_x[9];
   out_8310779210450879699[10] = delta_x[10] + nom_x[10];
   out_8310779210450879699[11] = delta_x[11] + nom_x[11];
   out_8310779210450879699[12] = delta_x[12] + nom_x[12];
   out_8310779210450879699[13] = delta_x[13] + nom_x[13];
   out_8310779210450879699[14] = delta_x[14] + nom_x[14];
   out_8310779210450879699[15] = delta_x[15] + nom_x[15];
   out_8310779210450879699[16] = delta_x[16] + nom_x[16];
   out_8310779210450879699[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7443261741520201626) {
   out_7443261741520201626[0] = -nom_x[0] + true_x[0];
   out_7443261741520201626[1] = -nom_x[1] + true_x[1];
   out_7443261741520201626[2] = -nom_x[2] + true_x[2];
   out_7443261741520201626[3] = -nom_x[3] + true_x[3];
   out_7443261741520201626[4] = -nom_x[4] + true_x[4];
   out_7443261741520201626[5] = -nom_x[5] + true_x[5];
   out_7443261741520201626[6] = -nom_x[6] + true_x[6];
   out_7443261741520201626[7] = -nom_x[7] + true_x[7];
   out_7443261741520201626[8] = -nom_x[8] + true_x[8];
   out_7443261741520201626[9] = -nom_x[9] + true_x[9];
   out_7443261741520201626[10] = -nom_x[10] + true_x[10];
   out_7443261741520201626[11] = -nom_x[11] + true_x[11];
   out_7443261741520201626[12] = -nom_x[12] + true_x[12];
   out_7443261741520201626[13] = -nom_x[13] + true_x[13];
   out_7443261741520201626[14] = -nom_x[14] + true_x[14];
   out_7443261741520201626[15] = -nom_x[15] + true_x[15];
   out_7443261741520201626[16] = -nom_x[16] + true_x[16];
   out_7443261741520201626[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3280844175493419238) {
   out_3280844175493419238[0] = 1.0;
   out_3280844175493419238[1] = 0.0;
   out_3280844175493419238[2] = 0.0;
   out_3280844175493419238[3] = 0.0;
   out_3280844175493419238[4] = 0.0;
   out_3280844175493419238[5] = 0.0;
   out_3280844175493419238[6] = 0.0;
   out_3280844175493419238[7] = 0.0;
   out_3280844175493419238[8] = 0.0;
   out_3280844175493419238[9] = 0.0;
   out_3280844175493419238[10] = 0.0;
   out_3280844175493419238[11] = 0.0;
   out_3280844175493419238[12] = 0.0;
   out_3280844175493419238[13] = 0.0;
   out_3280844175493419238[14] = 0.0;
   out_3280844175493419238[15] = 0.0;
   out_3280844175493419238[16] = 0.0;
   out_3280844175493419238[17] = 0.0;
   out_3280844175493419238[18] = 0.0;
   out_3280844175493419238[19] = 1.0;
   out_3280844175493419238[20] = 0.0;
   out_3280844175493419238[21] = 0.0;
   out_3280844175493419238[22] = 0.0;
   out_3280844175493419238[23] = 0.0;
   out_3280844175493419238[24] = 0.0;
   out_3280844175493419238[25] = 0.0;
   out_3280844175493419238[26] = 0.0;
   out_3280844175493419238[27] = 0.0;
   out_3280844175493419238[28] = 0.0;
   out_3280844175493419238[29] = 0.0;
   out_3280844175493419238[30] = 0.0;
   out_3280844175493419238[31] = 0.0;
   out_3280844175493419238[32] = 0.0;
   out_3280844175493419238[33] = 0.0;
   out_3280844175493419238[34] = 0.0;
   out_3280844175493419238[35] = 0.0;
   out_3280844175493419238[36] = 0.0;
   out_3280844175493419238[37] = 0.0;
   out_3280844175493419238[38] = 1.0;
   out_3280844175493419238[39] = 0.0;
   out_3280844175493419238[40] = 0.0;
   out_3280844175493419238[41] = 0.0;
   out_3280844175493419238[42] = 0.0;
   out_3280844175493419238[43] = 0.0;
   out_3280844175493419238[44] = 0.0;
   out_3280844175493419238[45] = 0.0;
   out_3280844175493419238[46] = 0.0;
   out_3280844175493419238[47] = 0.0;
   out_3280844175493419238[48] = 0.0;
   out_3280844175493419238[49] = 0.0;
   out_3280844175493419238[50] = 0.0;
   out_3280844175493419238[51] = 0.0;
   out_3280844175493419238[52] = 0.0;
   out_3280844175493419238[53] = 0.0;
   out_3280844175493419238[54] = 0.0;
   out_3280844175493419238[55] = 0.0;
   out_3280844175493419238[56] = 0.0;
   out_3280844175493419238[57] = 1.0;
   out_3280844175493419238[58] = 0.0;
   out_3280844175493419238[59] = 0.0;
   out_3280844175493419238[60] = 0.0;
   out_3280844175493419238[61] = 0.0;
   out_3280844175493419238[62] = 0.0;
   out_3280844175493419238[63] = 0.0;
   out_3280844175493419238[64] = 0.0;
   out_3280844175493419238[65] = 0.0;
   out_3280844175493419238[66] = 0.0;
   out_3280844175493419238[67] = 0.0;
   out_3280844175493419238[68] = 0.0;
   out_3280844175493419238[69] = 0.0;
   out_3280844175493419238[70] = 0.0;
   out_3280844175493419238[71] = 0.0;
   out_3280844175493419238[72] = 0.0;
   out_3280844175493419238[73] = 0.0;
   out_3280844175493419238[74] = 0.0;
   out_3280844175493419238[75] = 0.0;
   out_3280844175493419238[76] = 1.0;
   out_3280844175493419238[77] = 0.0;
   out_3280844175493419238[78] = 0.0;
   out_3280844175493419238[79] = 0.0;
   out_3280844175493419238[80] = 0.0;
   out_3280844175493419238[81] = 0.0;
   out_3280844175493419238[82] = 0.0;
   out_3280844175493419238[83] = 0.0;
   out_3280844175493419238[84] = 0.0;
   out_3280844175493419238[85] = 0.0;
   out_3280844175493419238[86] = 0.0;
   out_3280844175493419238[87] = 0.0;
   out_3280844175493419238[88] = 0.0;
   out_3280844175493419238[89] = 0.0;
   out_3280844175493419238[90] = 0.0;
   out_3280844175493419238[91] = 0.0;
   out_3280844175493419238[92] = 0.0;
   out_3280844175493419238[93] = 0.0;
   out_3280844175493419238[94] = 0.0;
   out_3280844175493419238[95] = 1.0;
   out_3280844175493419238[96] = 0.0;
   out_3280844175493419238[97] = 0.0;
   out_3280844175493419238[98] = 0.0;
   out_3280844175493419238[99] = 0.0;
   out_3280844175493419238[100] = 0.0;
   out_3280844175493419238[101] = 0.0;
   out_3280844175493419238[102] = 0.0;
   out_3280844175493419238[103] = 0.0;
   out_3280844175493419238[104] = 0.0;
   out_3280844175493419238[105] = 0.0;
   out_3280844175493419238[106] = 0.0;
   out_3280844175493419238[107] = 0.0;
   out_3280844175493419238[108] = 0.0;
   out_3280844175493419238[109] = 0.0;
   out_3280844175493419238[110] = 0.0;
   out_3280844175493419238[111] = 0.0;
   out_3280844175493419238[112] = 0.0;
   out_3280844175493419238[113] = 0.0;
   out_3280844175493419238[114] = 1.0;
   out_3280844175493419238[115] = 0.0;
   out_3280844175493419238[116] = 0.0;
   out_3280844175493419238[117] = 0.0;
   out_3280844175493419238[118] = 0.0;
   out_3280844175493419238[119] = 0.0;
   out_3280844175493419238[120] = 0.0;
   out_3280844175493419238[121] = 0.0;
   out_3280844175493419238[122] = 0.0;
   out_3280844175493419238[123] = 0.0;
   out_3280844175493419238[124] = 0.0;
   out_3280844175493419238[125] = 0.0;
   out_3280844175493419238[126] = 0.0;
   out_3280844175493419238[127] = 0.0;
   out_3280844175493419238[128] = 0.0;
   out_3280844175493419238[129] = 0.0;
   out_3280844175493419238[130] = 0.0;
   out_3280844175493419238[131] = 0.0;
   out_3280844175493419238[132] = 0.0;
   out_3280844175493419238[133] = 1.0;
   out_3280844175493419238[134] = 0.0;
   out_3280844175493419238[135] = 0.0;
   out_3280844175493419238[136] = 0.0;
   out_3280844175493419238[137] = 0.0;
   out_3280844175493419238[138] = 0.0;
   out_3280844175493419238[139] = 0.0;
   out_3280844175493419238[140] = 0.0;
   out_3280844175493419238[141] = 0.0;
   out_3280844175493419238[142] = 0.0;
   out_3280844175493419238[143] = 0.0;
   out_3280844175493419238[144] = 0.0;
   out_3280844175493419238[145] = 0.0;
   out_3280844175493419238[146] = 0.0;
   out_3280844175493419238[147] = 0.0;
   out_3280844175493419238[148] = 0.0;
   out_3280844175493419238[149] = 0.0;
   out_3280844175493419238[150] = 0.0;
   out_3280844175493419238[151] = 0.0;
   out_3280844175493419238[152] = 1.0;
   out_3280844175493419238[153] = 0.0;
   out_3280844175493419238[154] = 0.0;
   out_3280844175493419238[155] = 0.0;
   out_3280844175493419238[156] = 0.0;
   out_3280844175493419238[157] = 0.0;
   out_3280844175493419238[158] = 0.0;
   out_3280844175493419238[159] = 0.0;
   out_3280844175493419238[160] = 0.0;
   out_3280844175493419238[161] = 0.0;
   out_3280844175493419238[162] = 0.0;
   out_3280844175493419238[163] = 0.0;
   out_3280844175493419238[164] = 0.0;
   out_3280844175493419238[165] = 0.0;
   out_3280844175493419238[166] = 0.0;
   out_3280844175493419238[167] = 0.0;
   out_3280844175493419238[168] = 0.0;
   out_3280844175493419238[169] = 0.0;
   out_3280844175493419238[170] = 0.0;
   out_3280844175493419238[171] = 1.0;
   out_3280844175493419238[172] = 0.0;
   out_3280844175493419238[173] = 0.0;
   out_3280844175493419238[174] = 0.0;
   out_3280844175493419238[175] = 0.0;
   out_3280844175493419238[176] = 0.0;
   out_3280844175493419238[177] = 0.0;
   out_3280844175493419238[178] = 0.0;
   out_3280844175493419238[179] = 0.0;
   out_3280844175493419238[180] = 0.0;
   out_3280844175493419238[181] = 0.0;
   out_3280844175493419238[182] = 0.0;
   out_3280844175493419238[183] = 0.0;
   out_3280844175493419238[184] = 0.0;
   out_3280844175493419238[185] = 0.0;
   out_3280844175493419238[186] = 0.0;
   out_3280844175493419238[187] = 0.0;
   out_3280844175493419238[188] = 0.0;
   out_3280844175493419238[189] = 0.0;
   out_3280844175493419238[190] = 1.0;
   out_3280844175493419238[191] = 0.0;
   out_3280844175493419238[192] = 0.0;
   out_3280844175493419238[193] = 0.0;
   out_3280844175493419238[194] = 0.0;
   out_3280844175493419238[195] = 0.0;
   out_3280844175493419238[196] = 0.0;
   out_3280844175493419238[197] = 0.0;
   out_3280844175493419238[198] = 0.0;
   out_3280844175493419238[199] = 0.0;
   out_3280844175493419238[200] = 0.0;
   out_3280844175493419238[201] = 0.0;
   out_3280844175493419238[202] = 0.0;
   out_3280844175493419238[203] = 0.0;
   out_3280844175493419238[204] = 0.0;
   out_3280844175493419238[205] = 0.0;
   out_3280844175493419238[206] = 0.0;
   out_3280844175493419238[207] = 0.0;
   out_3280844175493419238[208] = 0.0;
   out_3280844175493419238[209] = 1.0;
   out_3280844175493419238[210] = 0.0;
   out_3280844175493419238[211] = 0.0;
   out_3280844175493419238[212] = 0.0;
   out_3280844175493419238[213] = 0.0;
   out_3280844175493419238[214] = 0.0;
   out_3280844175493419238[215] = 0.0;
   out_3280844175493419238[216] = 0.0;
   out_3280844175493419238[217] = 0.0;
   out_3280844175493419238[218] = 0.0;
   out_3280844175493419238[219] = 0.0;
   out_3280844175493419238[220] = 0.0;
   out_3280844175493419238[221] = 0.0;
   out_3280844175493419238[222] = 0.0;
   out_3280844175493419238[223] = 0.0;
   out_3280844175493419238[224] = 0.0;
   out_3280844175493419238[225] = 0.0;
   out_3280844175493419238[226] = 0.0;
   out_3280844175493419238[227] = 0.0;
   out_3280844175493419238[228] = 1.0;
   out_3280844175493419238[229] = 0.0;
   out_3280844175493419238[230] = 0.0;
   out_3280844175493419238[231] = 0.0;
   out_3280844175493419238[232] = 0.0;
   out_3280844175493419238[233] = 0.0;
   out_3280844175493419238[234] = 0.0;
   out_3280844175493419238[235] = 0.0;
   out_3280844175493419238[236] = 0.0;
   out_3280844175493419238[237] = 0.0;
   out_3280844175493419238[238] = 0.0;
   out_3280844175493419238[239] = 0.0;
   out_3280844175493419238[240] = 0.0;
   out_3280844175493419238[241] = 0.0;
   out_3280844175493419238[242] = 0.0;
   out_3280844175493419238[243] = 0.0;
   out_3280844175493419238[244] = 0.0;
   out_3280844175493419238[245] = 0.0;
   out_3280844175493419238[246] = 0.0;
   out_3280844175493419238[247] = 1.0;
   out_3280844175493419238[248] = 0.0;
   out_3280844175493419238[249] = 0.0;
   out_3280844175493419238[250] = 0.0;
   out_3280844175493419238[251] = 0.0;
   out_3280844175493419238[252] = 0.0;
   out_3280844175493419238[253] = 0.0;
   out_3280844175493419238[254] = 0.0;
   out_3280844175493419238[255] = 0.0;
   out_3280844175493419238[256] = 0.0;
   out_3280844175493419238[257] = 0.0;
   out_3280844175493419238[258] = 0.0;
   out_3280844175493419238[259] = 0.0;
   out_3280844175493419238[260] = 0.0;
   out_3280844175493419238[261] = 0.0;
   out_3280844175493419238[262] = 0.0;
   out_3280844175493419238[263] = 0.0;
   out_3280844175493419238[264] = 0.0;
   out_3280844175493419238[265] = 0.0;
   out_3280844175493419238[266] = 1.0;
   out_3280844175493419238[267] = 0.0;
   out_3280844175493419238[268] = 0.0;
   out_3280844175493419238[269] = 0.0;
   out_3280844175493419238[270] = 0.0;
   out_3280844175493419238[271] = 0.0;
   out_3280844175493419238[272] = 0.0;
   out_3280844175493419238[273] = 0.0;
   out_3280844175493419238[274] = 0.0;
   out_3280844175493419238[275] = 0.0;
   out_3280844175493419238[276] = 0.0;
   out_3280844175493419238[277] = 0.0;
   out_3280844175493419238[278] = 0.0;
   out_3280844175493419238[279] = 0.0;
   out_3280844175493419238[280] = 0.0;
   out_3280844175493419238[281] = 0.0;
   out_3280844175493419238[282] = 0.0;
   out_3280844175493419238[283] = 0.0;
   out_3280844175493419238[284] = 0.0;
   out_3280844175493419238[285] = 1.0;
   out_3280844175493419238[286] = 0.0;
   out_3280844175493419238[287] = 0.0;
   out_3280844175493419238[288] = 0.0;
   out_3280844175493419238[289] = 0.0;
   out_3280844175493419238[290] = 0.0;
   out_3280844175493419238[291] = 0.0;
   out_3280844175493419238[292] = 0.0;
   out_3280844175493419238[293] = 0.0;
   out_3280844175493419238[294] = 0.0;
   out_3280844175493419238[295] = 0.0;
   out_3280844175493419238[296] = 0.0;
   out_3280844175493419238[297] = 0.0;
   out_3280844175493419238[298] = 0.0;
   out_3280844175493419238[299] = 0.0;
   out_3280844175493419238[300] = 0.0;
   out_3280844175493419238[301] = 0.0;
   out_3280844175493419238[302] = 0.0;
   out_3280844175493419238[303] = 0.0;
   out_3280844175493419238[304] = 1.0;
   out_3280844175493419238[305] = 0.0;
   out_3280844175493419238[306] = 0.0;
   out_3280844175493419238[307] = 0.0;
   out_3280844175493419238[308] = 0.0;
   out_3280844175493419238[309] = 0.0;
   out_3280844175493419238[310] = 0.0;
   out_3280844175493419238[311] = 0.0;
   out_3280844175493419238[312] = 0.0;
   out_3280844175493419238[313] = 0.0;
   out_3280844175493419238[314] = 0.0;
   out_3280844175493419238[315] = 0.0;
   out_3280844175493419238[316] = 0.0;
   out_3280844175493419238[317] = 0.0;
   out_3280844175493419238[318] = 0.0;
   out_3280844175493419238[319] = 0.0;
   out_3280844175493419238[320] = 0.0;
   out_3280844175493419238[321] = 0.0;
   out_3280844175493419238[322] = 0.0;
   out_3280844175493419238[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_2175321056186301973) {
   out_2175321056186301973[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_2175321056186301973[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_2175321056186301973[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_2175321056186301973[3] = dt*state[12] + state[3];
   out_2175321056186301973[4] = dt*state[13] + state[4];
   out_2175321056186301973[5] = dt*state[14] + state[5];
   out_2175321056186301973[6] = state[6];
   out_2175321056186301973[7] = state[7];
   out_2175321056186301973[8] = state[8];
   out_2175321056186301973[9] = state[9];
   out_2175321056186301973[10] = state[10];
   out_2175321056186301973[11] = state[11];
   out_2175321056186301973[12] = state[12];
   out_2175321056186301973[13] = state[13];
   out_2175321056186301973[14] = state[14];
   out_2175321056186301973[15] = state[15];
   out_2175321056186301973[16] = state[16];
   out_2175321056186301973[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6067447866951354337) {
   out_6067447866951354337[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6067447866951354337[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6067447866951354337[2] = 0;
   out_6067447866951354337[3] = 0;
   out_6067447866951354337[4] = 0;
   out_6067447866951354337[5] = 0;
   out_6067447866951354337[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6067447866951354337[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6067447866951354337[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6067447866951354337[9] = 0;
   out_6067447866951354337[10] = 0;
   out_6067447866951354337[11] = 0;
   out_6067447866951354337[12] = 0;
   out_6067447866951354337[13] = 0;
   out_6067447866951354337[14] = 0;
   out_6067447866951354337[15] = 0;
   out_6067447866951354337[16] = 0;
   out_6067447866951354337[17] = 0;
   out_6067447866951354337[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6067447866951354337[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6067447866951354337[20] = 0;
   out_6067447866951354337[21] = 0;
   out_6067447866951354337[22] = 0;
   out_6067447866951354337[23] = 0;
   out_6067447866951354337[24] = 0;
   out_6067447866951354337[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6067447866951354337[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6067447866951354337[27] = 0;
   out_6067447866951354337[28] = 0;
   out_6067447866951354337[29] = 0;
   out_6067447866951354337[30] = 0;
   out_6067447866951354337[31] = 0;
   out_6067447866951354337[32] = 0;
   out_6067447866951354337[33] = 0;
   out_6067447866951354337[34] = 0;
   out_6067447866951354337[35] = 0;
   out_6067447866951354337[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6067447866951354337[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6067447866951354337[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6067447866951354337[39] = 0;
   out_6067447866951354337[40] = 0;
   out_6067447866951354337[41] = 0;
   out_6067447866951354337[42] = 0;
   out_6067447866951354337[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6067447866951354337[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6067447866951354337[45] = 0;
   out_6067447866951354337[46] = 0;
   out_6067447866951354337[47] = 0;
   out_6067447866951354337[48] = 0;
   out_6067447866951354337[49] = 0;
   out_6067447866951354337[50] = 0;
   out_6067447866951354337[51] = 0;
   out_6067447866951354337[52] = 0;
   out_6067447866951354337[53] = 0;
   out_6067447866951354337[54] = 0;
   out_6067447866951354337[55] = 0;
   out_6067447866951354337[56] = 0;
   out_6067447866951354337[57] = 1;
   out_6067447866951354337[58] = 0;
   out_6067447866951354337[59] = 0;
   out_6067447866951354337[60] = 0;
   out_6067447866951354337[61] = 0;
   out_6067447866951354337[62] = 0;
   out_6067447866951354337[63] = 0;
   out_6067447866951354337[64] = 0;
   out_6067447866951354337[65] = 0;
   out_6067447866951354337[66] = dt;
   out_6067447866951354337[67] = 0;
   out_6067447866951354337[68] = 0;
   out_6067447866951354337[69] = 0;
   out_6067447866951354337[70] = 0;
   out_6067447866951354337[71] = 0;
   out_6067447866951354337[72] = 0;
   out_6067447866951354337[73] = 0;
   out_6067447866951354337[74] = 0;
   out_6067447866951354337[75] = 0;
   out_6067447866951354337[76] = 1;
   out_6067447866951354337[77] = 0;
   out_6067447866951354337[78] = 0;
   out_6067447866951354337[79] = 0;
   out_6067447866951354337[80] = 0;
   out_6067447866951354337[81] = 0;
   out_6067447866951354337[82] = 0;
   out_6067447866951354337[83] = 0;
   out_6067447866951354337[84] = 0;
   out_6067447866951354337[85] = dt;
   out_6067447866951354337[86] = 0;
   out_6067447866951354337[87] = 0;
   out_6067447866951354337[88] = 0;
   out_6067447866951354337[89] = 0;
   out_6067447866951354337[90] = 0;
   out_6067447866951354337[91] = 0;
   out_6067447866951354337[92] = 0;
   out_6067447866951354337[93] = 0;
   out_6067447866951354337[94] = 0;
   out_6067447866951354337[95] = 1;
   out_6067447866951354337[96] = 0;
   out_6067447866951354337[97] = 0;
   out_6067447866951354337[98] = 0;
   out_6067447866951354337[99] = 0;
   out_6067447866951354337[100] = 0;
   out_6067447866951354337[101] = 0;
   out_6067447866951354337[102] = 0;
   out_6067447866951354337[103] = 0;
   out_6067447866951354337[104] = dt;
   out_6067447866951354337[105] = 0;
   out_6067447866951354337[106] = 0;
   out_6067447866951354337[107] = 0;
   out_6067447866951354337[108] = 0;
   out_6067447866951354337[109] = 0;
   out_6067447866951354337[110] = 0;
   out_6067447866951354337[111] = 0;
   out_6067447866951354337[112] = 0;
   out_6067447866951354337[113] = 0;
   out_6067447866951354337[114] = 1;
   out_6067447866951354337[115] = 0;
   out_6067447866951354337[116] = 0;
   out_6067447866951354337[117] = 0;
   out_6067447866951354337[118] = 0;
   out_6067447866951354337[119] = 0;
   out_6067447866951354337[120] = 0;
   out_6067447866951354337[121] = 0;
   out_6067447866951354337[122] = 0;
   out_6067447866951354337[123] = 0;
   out_6067447866951354337[124] = 0;
   out_6067447866951354337[125] = 0;
   out_6067447866951354337[126] = 0;
   out_6067447866951354337[127] = 0;
   out_6067447866951354337[128] = 0;
   out_6067447866951354337[129] = 0;
   out_6067447866951354337[130] = 0;
   out_6067447866951354337[131] = 0;
   out_6067447866951354337[132] = 0;
   out_6067447866951354337[133] = 1;
   out_6067447866951354337[134] = 0;
   out_6067447866951354337[135] = 0;
   out_6067447866951354337[136] = 0;
   out_6067447866951354337[137] = 0;
   out_6067447866951354337[138] = 0;
   out_6067447866951354337[139] = 0;
   out_6067447866951354337[140] = 0;
   out_6067447866951354337[141] = 0;
   out_6067447866951354337[142] = 0;
   out_6067447866951354337[143] = 0;
   out_6067447866951354337[144] = 0;
   out_6067447866951354337[145] = 0;
   out_6067447866951354337[146] = 0;
   out_6067447866951354337[147] = 0;
   out_6067447866951354337[148] = 0;
   out_6067447866951354337[149] = 0;
   out_6067447866951354337[150] = 0;
   out_6067447866951354337[151] = 0;
   out_6067447866951354337[152] = 1;
   out_6067447866951354337[153] = 0;
   out_6067447866951354337[154] = 0;
   out_6067447866951354337[155] = 0;
   out_6067447866951354337[156] = 0;
   out_6067447866951354337[157] = 0;
   out_6067447866951354337[158] = 0;
   out_6067447866951354337[159] = 0;
   out_6067447866951354337[160] = 0;
   out_6067447866951354337[161] = 0;
   out_6067447866951354337[162] = 0;
   out_6067447866951354337[163] = 0;
   out_6067447866951354337[164] = 0;
   out_6067447866951354337[165] = 0;
   out_6067447866951354337[166] = 0;
   out_6067447866951354337[167] = 0;
   out_6067447866951354337[168] = 0;
   out_6067447866951354337[169] = 0;
   out_6067447866951354337[170] = 0;
   out_6067447866951354337[171] = 1;
   out_6067447866951354337[172] = 0;
   out_6067447866951354337[173] = 0;
   out_6067447866951354337[174] = 0;
   out_6067447866951354337[175] = 0;
   out_6067447866951354337[176] = 0;
   out_6067447866951354337[177] = 0;
   out_6067447866951354337[178] = 0;
   out_6067447866951354337[179] = 0;
   out_6067447866951354337[180] = 0;
   out_6067447866951354337[181] = 0;
   out_6067447866951354337[182] = 0;
   out_6067447866951354337[183] = 0;
   out_6067447866951354337[184] = 0;
   out_6067447866951354337[185] = 0;
   out_6067447866951354337[186] = 0;
   out_6067447866951354337[187] = 0;
   out_6067447866951354337[188] = 0;
   out_6067447866951354337[189] = 0;
   out_6067447866951354337[190] = 1;
   out_6067447866951354337[191] = 0;
   out_6067447866951354337[192] = 0;
   out_6067447866951354337[193] = 0;
   out_6067447866951354337[194] = 0;
   out_6067447866951354337[195] = 0;
   out_6067447866951354337[196] = 0;
   out_6067447866951354337[197] = 0;
   out_6067447866951354337[198] = 0;
   out_6067447866951354337[199] = 0;
   out_6067447866951354337[200] = 0;
   out_6067447866951354337[201] = 0;
   out_6067447866951354337[202] = 0;
   out_6067447866951354337[203] = 0;
   out_6067447866951354337[204] = 0;
   out_6067447866951354337[205] = 0;
   out_6067447866951354337[206] = 0;
   out_6067447866951354337[207] = 0;
   out_6067447866951354337[208] = 0;
   out_6067447866951354337[209] = 1;
   out_6067447866951354337[210] = 0;
   out_6067447866951354337[211] = 0;
   out_6067447866951354337[212] = 0;
   out_6067447866951354337[213] = 0;
   out_6067447866951354337[214] = 0;
   out_6067447866951354337[215] = 0;
   out_6067447866951354337[216] = 0;
   out_6067447866951354337[217] = 0;
   out_6067447866951354337[218] = 0;
   out_6067447866951354337[219] = 0;
   out_6067447866951354337[220] = 0;
   out_6067447866951354337[221] = 0;
   out_6067447866951354337[222] = 0;
   out_6067447866951354337[223] = 0;
   out_6067447866951354337[224] = 0;
   out_6067447866951354337[225] = 0;
   out_6067447866951354337[226] = 0;
   out_6067447866951354337[227] = 0;
   out_6067447866951354337[228] = 1;
   out_6067447866951354337[229] = 0;
   out_6067447866951354337[230] = 0;
   out_6067447866951354337[231] = 0;
   out_6067447866951354337[232] = 0;
   out_6067447866951354337[233] = 0;
   out_6067447866951354337[234] = 0;
   out_6067447866951354337[235] = 0;
   out_6067447866951354337[236] = 0;
   out_6067447866951354337[237] = 0;
   out_6067447866951354337[238] = 0;
   out_6067447866951354337[239] = 0;
   out_6067447866951354337[240] = 0;
   out_6067447866951354337[241] = 0;
   out_6067447866951354337[242] = 0;
   out_6067447866951354337[243] = 0;
   out_6067447866951354337[244] = 0;
   out_6067447866951354337[245] = 0;
   out_6067447866951354337[246] = 0;
   out_6067447866951354337[247] = 1;
   out_6067447866951354337[248] = 0;
   out_6067447866951354337[249] = 0;
   out_6067447866951354337[250] = 0;
   out_6067447866951354337[251] = 0;
   out_6067447866951354337[252] = 0;
   out_6067447866951354337[253] = 0;
   out_6067447866951354337[254] = 0;
   out_6067447866951354337[255] = 0;
   out_6067447866951354337[256] = 0;
   out_6067447866951354337[257] = 0;
   out_6067447866951354337[258] = 0;
   out_6067447866951354337[259] = 0;
   out_6067447866951354337[260] = 0;
   out_6067447866951354337[261] = 0;
   out_6067447866951354337[262] = 0;
   out_6067447866951354337[263] = 0;
   out_6067447866951354337[264] = 0;
   out_6067447866951354337[265] = 0;
   out_6067447866951354337[266] = 1;
   out_6067447866951354337[267] = 0;
   out_6067447866951354337[268] = 0;
   out_6067447866951354337[269] = 0;
   out_6067447866951354337[270] = 0;
   out_6067447866951354337[271] = 0;
   out_6067447866951354337[272] = 0;
   out_6067447866951354337[273] = 0;
   out_6067447866951354337[274] = 0;
   out_6067447866951354337[275] = 0;
   out_6067447866951354337[276] = 0;
   out_6067447866951354337[277] = 0;
   out_6067447866951354337[278] = 0;
   out_6067447866951354337[279] = 0;
   out_6067447866951354337[280] = 0;
   out_6067447866951354337[281] = 0;
   out_6067447866951354337[282] = 0;
   out_6067447866951354337[283] = 0;
   out_6067447866951354337[284] = 0;
   out_6067447866951354337[285] = 1;
   out_6067447866951354337[286] = 0;
   out_6067447866951354337[287] = 0;
   out_6067447866951354337[288] = 0;
   out_6067447866951354337[289] = 0;
   out_6067447866951354337[290] = 0;
   out_6067447866951354337[291] = 0;
   out_6067447866951354337[292] = 0;
   out_6067447866951354337[293] = 0;
   out_6067447866951354337[294] = 0;
   out_6067447866951354337[295] = 0;
   out_6067447866951354337[296] = 0;
   out_6067447866951354337[297] = 0;
   out_6067447866951354337[298] = 0;
   out_6067447866951354337[299] = 0;
   out_6067447866951354337[300] = 0;
   out_6067447866951354337[301] = 0;
   out_6067447866951354337[302] = 0;
   out_6067447866951354337[303] = 0;
   out_6067447866951354337[304] = 1;
   out_6067447866951354337[305] = 0;
   out_6067447866951354337[306] = 0;
   out_6067447866951354337[307] = 0;
   out_6067447866951354337[308] = 0;
   out_6067447866951354337[309] = 0;
   out_6067447866951354337[310] = 0;
   out_6067447866951354337[311] = 0;
   out_6067447866951354337[312] = 0;
   out_6067447866951354337[313] = 0;
   out_6067447866951354337[314] = 0;
   out_6067447866951354337[315] = 0;
   out_6067447866951354337[316] = 0;
   out_6067447866951354337[317] = 0;
   out_6067447866951354337[318] = 0;
   out_6067447866951354337[319] = 0;
   out_6067447866951354337[320] = 0;
   out_6067447866951354337[321] = 0;
   out_6067447866951354337[322] = 0;
   out_6067447866951354337[323] = 1;
}
void h_4(double *state, double *unused, double *out_5162177481250063533) {
   out_5162177481250063533[0] = state[6] + state[9];
   out_5162177481250063533[1] = state[7] + state[10];
   out_5162177481250063533[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4478791657376317342) {
   out_4478791657376317342[0] = 0;
   out_4478791657376317342[1] = 0;
   out_4478791657376317342[2] = 0;
   out_4478791657376317342[3] = 0;
   out_4478791657376317342[4] = 0;
   out_4478791657376317342[5] = 0;
   out_4478791657376317342[6] = 1;
   out_4478791657376317342[7] = 0;
   out_4478791657376317342[8] = 0;
   out_4478791657376317342[9] = 1;
   out_4478791657376317342[10] = 0;
   out_4478791657376317342[11] = 0;
   out_4478791657376317342[12] = 0;
   out_4478791657376317342[13] = 0;
   out_4478791657376317342[14] = 0;
   out_4478791657376317342[15] = 0;
   out_4478791657376317342[16] = 0;
   out_4478791657376317342[17] = 0;
   out_4478791657376317342[18] = 0;
   out_4478791657376317342[19] = 0;
   out_4478791657376317342[20] = 0;
   out_4478791657376317342[21] = 0;
   out_4478791657376317342[22] = 0;
   out_4478791657376317342[23] = 0;
   out_4478791657376317342[24] = 0;
   out_4478791657376317342[25] = 1;
   out_4478791657376317342[26] = 0;
   out_4478791657376317342[27] = 0;
   out_4478791657376317342[28] = 1;
   out_4478791657376317342[29] = 0;
   out_4478791657376317342[30] = 0;
   out_4478791657376317342[31] = 0;
   out_4478791657376317342[32] = 0;
   out_4478791657376317342[33] = 0;
   out_4478791657376317342[34] = 0;
   out_4478791657376317342[35] = 0;
   out_4478791657376317342[36] = 0;
   out_4478791657376317342[37] = 0;
   out_4478791657376317342[38] = 0;
   out_4478791657376317342[39] = 0;
   out_4478791657376317342[40] = 0;
   out_4478791657376317342[41] = 0;
   out_4478791657376317342[42] = 0;
   out_4478791657376317342[43] = 0;
   out_4478791657376317342[44] = 1;
   out_4478791657376317342[45] = 0;
   out_4478791657376317342[46] = 0;
   out_4478791657376317342[47] = 1;
   out_4478791657376317342[48] = 0;
   out_4478791657376317342[49] = 0;
   out_4478791657376317342[50] = 0;
   out_4478791657376317342[51] = 0;
   out_4478791657376317342[52] = 0;
   out_4478791657376317342[53] = 0;
}
void h_10(double *state, double *unused, double *out_2981368779043647817) {
   out_2981368779043647817[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2981368779043647817[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2981368779043647817[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_919074047538936453) {
   out_919074047538936453[0] = 0;
   out_919074047538936453[1] = 9.8100000000000005*cos(state[1]);
   out_919074047538936453[2] = 0;
   out_919074047538936453[3] = 0;
   out_919074047538936453[4] = -state[8];
   out_919074047538936453[5] = state[7];
   out_919074047538936453[6] = 0;
   out_919074047538936453[7] = state[5];
   out_919074047538936453[8] = -state[4];
   out_919074047538936453[9] = 0;
   out_919074047538936453[10] = 0;
   out_919074047538936453[11] = 0;
   out_919074047538936453[12] = 1;
   out_919074047538936453[13] = 0;
   out_919074047538936453[14] = 0;
   out_919074047538936453[15] = 1;
   out_919074047538936453[16] = 0;
   out_919074047538936453[17] = 0;
   out_919074047538936453[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_919074047538936453[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_919074047538936453[20] = 0;
   out_919074047538936453[21] = state[8];
   out_919074047538936453[22] = 0;
   out_919074047538936453[23] = -state[6];
   out_919074047538936453[24] = -state[5];
   out_919074047538936453[25] = 0;
   out_919074047538936453[26] = state[3];
   out_919074047538936453[27] = 0;
   out_919074047538936453[28] = 0;
   out_919074047538936453[29] = 0;
   out_919074047538936453[30] = 0;
   out_919074047538936453[31] = 1;
   out_919074047538936453[32] = 0;
   out_919074047538936453[33] = 0;
   out_919074047538936453[34] = 1;
   out_919074047538936453[35] = 0;
   out_919074047538936453[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_919074047538936453[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_919074047538936453[38] = 0;
   out_919074047538936453[39] = -state[7];
   out_919074047538936453[40] = state[6];
   out_919074047538936453[41] = 0;
   out_919074047538936453[42] = state[4];
   out_919074047538936453[43] = -state[3];
   out_919074047538936453[44] = 0;
   out_919074047538936453[45] = 0;
   out_919074047538936453[46] = 0;
   out_919074047538936453[47] = 0;
   out_919074047538936453[48] = 0;
   out_919074047538936453[49] = 0;
   out_919074047538936453[50] = 1;
   out_919074047538936453[51] = 0;
   out_919074047538936453[52] = 0;
   out_919074047538936453[53] = 1;
}
void h_13(double *state, double *unused, double *out_3635540254248668032) {
   out_3635540254248668032[0] = state[3];
   out_3635540254248668032[1] = state[4];
   out_3635540254248668032[2] = state[5];
}
void H_13(double *state, double *unused, double *out_645036194073793318) {
   out_645036194073793318[0] = 0;
   out_645036194073793318[1] = 0;
   out_645036194073793318[2] = 0;
   out_645036194073793318[3] = 1;
   out_645036194073793318[4] = 0;
   out_645036194073793318[5] = 0;
   out_645036194073793318[6] = 0;
   out_645036194073793318[7] = 0;
   out_645036194073793318[8] = 0;
   out_645036194073793318[9] = 0;
   out_645036194073793318[10] = 0;
   out_645036194073793318[11] = 0;
   out_645036194073793318[12] = 0;
   out_645036194073793318[13] = 0;
   out_645036194073793318[14] = 0;
   out_645036194073793318[15] = 0;
   out_645036194073793318[16] = 0;
   out_645036194073793318[17] = 0;
   out_645036194073793318[18] = 0;
   out_645036194073793318[19] = 0;
   out_645036194073793318[20] = 0;
   out_645036194073793318[21] = 0;
   out_645036194073793318[22] = 1;
   out_645036194073793318[23] = 0;
   out_645036194073793318[24] = 0;
   out_645036194073793318[25] = 0;
   out_645036194073793318[26] = 0;
   out_645036194073793318[27] = 0;
   out_645036194073793318[28] = 0;
   out_645036194073793318[29] = 0;
   out_645036194073793318[30] = 0;
   out_645036194073793318[31] = 0;
   out_645036194073793318[32] = 0;
   out_645036194073793318[33] = 0;
   out_645036194073793318[34] = 0;
   out_645036194073793318[35] = 0;
   out_645036194073793318[36] = 0;
   out_645036194073793318[37] = 0;
   out_645036194073793318[38] = 0;
   out_645036194073793318[39] = 0;
   out_645036194073793318[40] = 0;
   out_645036194073793318[41] = 1;
   out_645036194073793318[42] = 0;
   out_645036194073793318[43] = 0;
   out_645036194073793318[44] = 0;
   out_645036194073793318[45] = 0;
   out_645036194073793318[46] = 0;
   out_645036194073793318[47] = 0;
   out_645036194073793318[48] = 0;
   out_645036194073793318[49] = 0;
   out_645036194073793318[50] = 0;
   out_645036194073793318[51] = 0;
   out_645036194073793318[52] = 0;
   out_645036194073793318[53] = 0;
}
void h_14(double *state, double *unused, double *out_4192481788271112889) {
   out_4192481788271112889[0] = state[6];
   out_4192481788271112889[1] = state[7];
   out_4192481788271112889[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1396003225080945046) {
   out_1396003225080945046[0] = 0;
   out_1396003225080945046[1] = 0;
   out_1396003225080945046[2] = 0;
   out_1396003225080945046[3] = 0;
   out_1396003225080945046[4] = 0;
   out_1396003225080945046[5] = 0;
   out_1396003225080945046[6] = 1;
   out_1396003225080945046[7] = 0;
   out_1396003225080945046[8] = 0;
   out_1396003225080945046[9] = 0;
   out_1396003225080945046[10] = 0;
   out_1396003225080945046[11] = 0;
   out_1396003225080945046[12] = 0;
   out_1396003225080945046[13] = 0;
   out_1396003225080945046[14] = 0;
   out_1396003225080945046[15] = 0;
   out_1396003225080945046[16] = 0;
   out_1396003225080945046[17] = 0;
   out_1396003225080945046[18] = 0;
   out_1396003225080945046[19] = 0;
   out_1396003225080945046[20] = 0;
   out_1396003225080945046[21] = 0;
   out_1396003225080945046[22] = 0;
   out_1396003225080945046[23] = 0;
   out_1396003225080945046[24] = 0;
   out_1396003225080945046[25] = 1;
   out_1396003225080945046[26] = 0;
   out_1396003225080945046[27] = 0;
   out_1396003225080945046[28] = 0;
   out_1396003225080945046[29] = 0;
   out_1396003225080945046[30] = 0;
   out_1396003225080945046[31] = 0;
   out_1396003225080945046[32] = 0;
   out_1396003225080945046[33] = 0;
   out_1396003225080945046[34] = 0;
   out_1396003225080945046[35] = 0;
   out_1396003225080945046[36] = 0;
   out_1396003225080945046[37] = 0;
   out_1396003225080945046[38] = 0;
   out_1396003225080945046[39] = 0;
   out_1396003225080945046[40] = 0;
   out_1396003225080945046[41] = 0;
   out_1396003225080945046[42] = 0;
   out_1396003225080945046[43] = 0;
   out_1396003225080945046[44] = 1;
   out_1396003225080945046[45] = 0;
   out_1396003225080945046[46] = 0;
   out_1396003225080945046[47] = 0;
   out_1396003225080945046[48] = 0;
   out_1396003225080945046[49] = 0;
   out_1396003225080945046[50] = 0;
   out_1396003225080945046[51] = 0;
   out_1396003225080945046[52] = 0;
   out_1396003225080945046[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_8310779210450879699) {
  err_fun(nom_x, delta_x, out_8310779210450879699);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_7443261741520201626) {
  inv_err_fun(nom_x, true_x, out_7443261741520201626);
}
void pose_H_mod_fun(double *state, double *out_3280844175493419238) {
  H_mod_fun(state, out_3280844175493419238);
}
void pose_f_fun(double *state, double dt, double *out_2175321056186301973) {
  f_fun(state,  dt, out_2175321056186301973);
}
void pose_F_fun(double *state, double dt, double *out_6067447866951354337) {
  F_fun(state,  dt, out_6067447866951354337);
}
void pose_h_4(double *state, double *unused, double *out_5162177481250063533) {
  h_4(state, unused, out_5162177481250063533);
}
void pose_H_4(double *state, double *unused, double *out_4478791657376317342) {
  H_4(state, unused, out_4478791657376317342);
}
void pose_h_10(double *state, double *unused, double *out_2981368779043647817) {
  h_10(state, unused, out_2981368779043647817);
}
void pose_H_10(double *state, double *unused, double *out_919074047538936453) {
  H_10(state, unused, out_919074047538936453);
}
void pose_h_13(double *state, double *unused, double *out_3635540254248668032) {
  h_13(state, unused, out_3635540254248668032);
}
void pose_H_13(double *state, double *unused, double *out_645036194073793318) {
  H_13(state, unused, out_645036194073793318);
}
void pose_h_14(double *state, double *unused, double *out_4192481788271112889) {
  h_14(state, unused, out_4192481788271112889);
}
void pose_H_14(double *state, double *unused, double *out_1396003225080945046) {
  H_14(state, unused, out_1396003225080945046);
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
