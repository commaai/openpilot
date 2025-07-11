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
 *                      Code generated with SymPy 1.13.2                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7464579881794296048) {
   out_7464579881794296048[0] = delta_x[0] + nom_x[0];
   out_7464579881794296048[1] = delta_x[1] + nom_x[1];
   out_7464579881794296048[2] = delta_x[2] + nom_x[2];
   out_7464579881794296048[3] = delta_x[3] + nom_x[3];
   out_7464579881794296048[4] = delta_x[4] + nom_x[4];
   out_7464579881794296048[5] = delta_x[5] + nom_x[5];
   out_7464579881794296048[6] = delta_x[6] + nom_x[6];
   out_7464579881794296048[7] = delta_x[7] + nom_x[7];
   out_7464579881794296048[8] = delta_x[8] + nom_x[8];
   out_7464579881794296048[9] = delta_x[9] + nom_x[9];
   out_7464579881794296048[10] = delta_x[10] + nom_x[10];
   out_7464579881794296048[11] = delta_x[11] + nom_x[11];
   out_7464579881794296048[12] = delta_x[12] + nom_x[12];
   out_7464579881794296048[13] = delta_x[13] + nom_x[13];
   out_7464579881794296048[14] = delta_x[14] + nom_x[14];
   out_7464579881794296048[15] = delta_x[15] + nom_x[15];
   out_7464579881794296048[16] = delta_x[16] + nom_x[16];
   out_7464579881794296048[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2725303949871892042) {
   out_2725303949871892042[0] = -nom_x[0] + true_x[0];
   out_2725303949871892042[1] = -nom_x[1] + true_x[1];
   out_2725303949871892042[2] = -nom_x[2] + true_x[2];
   out_2725303949871892042[3] = -nom_x[3] + true_x[3];
   out_2725303949871892042[4] = -nom_x[4] + true_x[4];
   out_2725303949871892042[5] = -nom_x[5] + true_x[5];
   out_2725303949871892042[6] = -nom_x[6] + true_x[6];
   out_2725303949871892042[7] = -nom_x[7] + true_x[7];
   out_2725303949871892042[8] = -nom_x[8] + true_x[8];
   out_2725303949871892042[9] = -nom_x[9] + true_x[9];
   out_2725303949871892042[10] = -nom_x[10] + true_x[10];
   out_2725303949871892042[11] = -nom_x[11] + true_x[11];
   out_2725303949871892042[12] = -nom_x[12] + true_x[12];
   out_2725303949871892042[13] = -nom_x[13] + true_x[13];
   out_2725303949871892042[14] = -nom_x[14] + true_x[14];
   out_2725303949871892042[15] = -nom_x[15] + true_x[15];
   out_2725303949871892042[16] = -nom_x[16] + true_x[16];
   out_2725303949871892042[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_2728954879743535149) {
   out_2728954879743535149[0] = 1.0;
   out_2728954879743535149[1] = 0.0;
   out_2728954879743535149[2] = 0.0;
   out_2728954879743535149[3] = 0.0;
   out_2728954879743535149[4] = 0.0;
   out_2728954879743535149[5] = 0.0;
   out_2728954879743535149[6] = 0.0;
   out_2728954879743535149[7] = 0.0;
   out_2728954879743535149[8] = 0.0;
   out_2728954879743535149[9] = 0.0;
   out_2728954879743535149[10] = 0.0;
   out_2728954879743535149[11] = 0.0;
   out_2728954879743535149[12] = 0.0;
   out_2728954879743535149[13] = 0.0;
   out_2728954879743535149[14] = 0.0;
   out_2728954879743535149[15] = 0.0;
   out_2728954879743535149[16] = 0.0;
   out_2728954879743535149[17] = 0.0;
   out_2728954879743535149[18] = 0.0;
   out_2728954879743535149[19] = 1.0;
   out_2728954879743535149[20] = 0.0;
   out_2728954879743535149[21] = 0.0;
   out_2728954879743535149[22] = 0.0;
   out_2728954879743535149[23] = 0.0;
   out_2728954879743535149[24] = 0.0;
   out_2728954879743535149[25] = 0.0;
   out_2728954879743535149[26] = 0.0;
   out_2728954879743535149[27] = 0.0;
   out_2728954879743535149[28] = 0.0;
   out_2728954879743535149[29] = 0.0;
   out_2728954879743535149[30] = 0.0;
   out_2728954879743535149[31] = 0.0;
   out_2728954879743535149[32] = 0.0;
   out_2728954879743535149[33] = 0.0;
   out_2728954879743535149[34] = 0.0;
   out_2728954879743535149[35] = 0.0;
   out_2728954879743535149[36] = 0.0;
   out_2728954879743535149[37] = 0.0;
   out_2728954879743535149[38] = 1.0;
   out_2728954879743535149[39] = 0.0;
   out_2728954879743535149[40] = 0.0;
   out_2728954879743535149[41] = 0.0;
   out_2728954879743535149[42] = 0.0;
   out_2728954879743535149[43] = 0.0;
   out_2728954879743535149[44] = 0.0;
   out_2728954879743535149[45] = 0.0;
   out_2728954879743535149[46] = 0.0;
   out_2728954879743535149[47] = 0.0;
   out_2728954879743535149[48] = 0.0;
   out_2728954879743535149[49] = 0.0;
   out_2728954879743535149[50] = 0.0;
   out_2728954879743535149[51] = 0.0;
   out_2728954879743535149[52] = 0.0;
   out_2728954879743535149[53] = 0.0;
   out_2728954879743535149[54] = 0.0;
   out_2728954879743535149[55] = 0.0;
   out_2728954879743535149[56] = 0.0;
   out_2728954879743535149[57] = 1.0;
   out_2728954879743535149[58] = 0.0;
   out_2728954879743535149[59] = 0.0;
   out_2728954879743535149[60] = 0.0;
   out_2728954879743535149[61] = 0.0;
   out_2728954879743535149[62] = 0.0;
   out_2728954879743535149[63] = 0.0;
   out_2728954879743535149[64] = 0.0;
   out_2728954879743535149[65] = 0.0;
   out_2728954879743535149[66] = 0.0;
   out_2728954879743535149[67] = 0.0;
   out_2728954879743535149[68] = 0.0;
   out_2728954879743535149[69] = 0.0;
   out_2728954879743535149[70] = 0.0;
   out_2728954879743535149[71] = 0.0;
   out_2728954879743535149[72] = 0.0;
   out_2728954879743535149[73] = 0.0;
   out_2728954879743535149[74] = 0.0;
   out_2728954879743535149[75] = 0.0;
   out_2728954879743535149[76] = 1.0;
   out_2728954879743535149[77] = 0.0;
   out_2728954879743535149[78] = 0.0;
   out_2728954879743535149[79] = 0.0;
   out_2728954879743535149[80] = 0.0;
   out_2728954879743535149[81] = 0.0;
   out_2728954879743535149[82] = 0.0;
   out_2728954879743535149[83] = 0.0;
   out_2728954879743535149[84] = 0.0;
   out_2728954879743535149[85] = 0.0;
   out_2728954879743535149[86] = 0.0;
   out_2728954879743535149[87] = 0.0;
   out_2728954879743535149[88] = 0.0;
   out_2728954879743535149[89] = 0.0;
   out_2728954879743535149[90] = 0.0;
   out_2728954879743535149[91] = 0.0;
   out_2728954879743535149[92] = 0.0;
   out_2728954879743535149[93] = 0.0;
   out_2728954879743535149[94] = 0.0;
   out_2728954879743535149[95] = 1.0;
   out_2728954879743535149[96] = 0.0;
   out_2728954879743535149[97] = 0.0;
   out_2728954879743535149[98] = 0.0;
   out_2728954879743535149[99] = 0.0;
   out_2728954879743535149[100] = 0.0;
   out_2728954879743535149[101] = 0.0;
   out_2728954879743535149[102] = 0.0;
   out_2728954879743535149[103] = 0.0;
   out_2728954879743535149[104] = 0.0;
   out_2728954879743535149[105] = 0.0;
   out_2728954879743535149[106] = 0.0;
   out_2728954879743535149[107] = 0.0;
   out_2728954879743535149[108] = 0.0;
   out_2728954879743535149[109] = 0.0;
   out_2728954879743535149[110] = 0.0;
   out_2728954879743535149[111] = 0.0;
   out_2728954879743535149[112] = 0.0;
   out_2728954879743535149[113] = 0.0;
   out_2728954879743535149[114] = 1.0;
   out_2728954879743535149[115] = 0.0;
   out_2728954879743535149[116] = 0.0;
   out_2728954879743535149[117] = 0.0;
   out_2728954879743535149[118] = 0.0;
   out_2728954879743535149[119] = 0.0;
   out_2728954879743535149[120] = 0.0;
   out_2728954879743535149[121] = 0.0;
   out_2728954879743535149[122] = 0.0;
   out_2728954879743535149[123] = 0.0;
   out_2728954879743535149[124] = 0.0;
   out_2728954879743535149[125] = 0.0;
   out_2728954879743535149[126] = 0.0;
   out_2728954879743535149[127] = 0.0;
   out_2728954879743535149[128] = 0.0;
   out_2728954879743535149[129] = 0.0;
   out_2728954879743535149[130] = 0.0;
   out_2728954879743535149[131] = 0.0;
   out_2728954879743535149[132] = 0.0;
   out_2728954879743535149[133] = 1.0;
   out_2728954879743535149[134] = 0.0;
   out_2728954879743535149[135] = 0.0;
   out_2728954879743535149[136] = 0.0;
   out_2728954879743535149[137] = 0.0;
   out_2728954879743535149[138] = 0.0;
   out_2728954879743535149[139] = 0.0;
   out_2728954879743535149[140] = 0.0;
   out_2728954879743535149[141] = 0.0;
   out_2728954879743535149[142] = 0.0;
   out_2728954879743535149[143] = 0.0;
   out_2728954879743535149[144] = 0.0;
   out_2728954879743535149[145] = 0.0;
   out_2728954879743535149[146] = 0.0;
   out_2728954879743535149[147] = 0.0;
   out_2728954879743535149[148] = 0.0;
   out_2728954879743535149[149] = 0.0;
   out_2728954879743535149[150] = 0.0;
   out_2728954879743535149[151] = 0.0;
   out_2728954879743535149[152] = 1.0;
   out_2728954879743535149[153] = 0.0;
   out_2728954879743535149[154] = 0.0;
   out_2728954879743535149[155] = 0.0;
   out_2728954879743535149[156] = 0.0;
   out_2728954879743535149[157] = 0.0;
   out_2728954879743535149[158] = 0.0;
   out_2728954879743535149[159] = 0.0;
   out_2728954879743535149[160] = 0.0;
   out_2728954879743535149[161] = 0.0;
   out_2728954879743535149[162] = 0.0;
   out_2728954879743535149[163] = 0.0;
   out_2728954879743535149[164] = 0.0;
   out_2728954879743535149[165] = 0.0;
   out_2728954879743535149[166] = 0.0;
   out_2728954879743535149[167] = 0.0;
   out_2728954879743535149[168] = 0.0;
   out_2728954879743535149[169] = 0.0;
   out_2728954879743535149[170] = 0.0;
   out_2728954879743535149[171] = 1.0;
   out_2728954879743535149[172] = 0.0;
   out_2728954879743535149[173] = 0.0;
   out_2728954879743535149[174] = 0.0;
   out_2728954879743535149[175] = 0.0;
   out_2728954879743535149[176] = 0.0;
   out_2728954879743535149[177] = 0.0;
   out_2728954879743535149[178] = 0.0;
   out_2728954879743535149[179] = 0.0;
   out_2728954879743535149[180] = 0.0;
   out_2728954879743535149[181] = 0.0;
   out_2728954879743535149[182] = 0.0;
   out_2728954879743535149[183] = 0.0;
   out_2728954879743535149[184] = 0.0;
   out_2728954879743535149[185] = 0.0;
   out_2728954879743535149[186] = 0.0;
   out_2728954879743535149[187] = 0.0;
   out_2728954879743535149[188] = 0.0;
   out_2728954879743535149[189] = 0.0;
   out_2728954879743535149[190] = 1.0;
   out_2728954879743535149[191] = 0.0;
   out_2728954879743535149[192] = 0.0;
   out_2728954879743535149[193] = 0.0;
   out_2728954879743535149[194] = 0.0;
   out_2728954879743535149[195] = 0.0;
   out_2728954879743535149[196] = 0.0;
   out_2728954879743535149[197] = 0.0;
   out_2728954879743535149[198] = 0.0;
   out_2728954879743535149[199] = 0.0;
   out_2728954879743535149[200] = 0.0;
   out_2728954879743535149[201] = 0.0;
   out_2728954879743535149[202] = 0.0;
   out_2728954879743535149[203] = 0.0;
   out_2728954879743535149[204] = 0.0;
   out_2728954879743535149[205] = 0.0;
   out_2728954879743535149[206] = 0.0;
   out_2728954879743535149[207] = 0.0;
   out_2728954879743535149[208] = 0.0;
   out_2728954879743535149[209] = 1.0;
   out_2728954879743535149[210] = 0.0;
   out_2728954879743535149[211] = 0.0;
   out_2728954879743535149[212] = 0.0;
   out_2728954879743535149[213] = 0.0;
   out_2728954879743535149[214] = 0.0;
   out_2728954879743535149[215] = 0.0;
   out_2728954879743535149[216] = 0.0;
   out_2728954879743535149[217] = 0.0;
   out_2728954879743535149[218] = 0.0;
   out_2728954879743535149[219] = 0.0;
   out_2728954879743535149[220] = 0.0;
   out_2728954879743535149[221] = 0.0;
   out_2728954879743535149[222] = 0.0;
   out_2728954879743535149[223] = 0.0;
   out_2728954879743535149[224] = 0.0;
   out_2728954879743535149[225] = 0.0;
   out_2728954879743535149[226] = 0.0;
   out_2728954879743535149[227] = 0.0;
   out_2728954879743535149[228] = 1.0;
   out_2728954879743535149[229] = 0.0;
   out_2728954879743535149[230] = 0.0;
   out_2728954879743535149[231] = 0.0;
   out_2728954879743535149[232] = 0.0;
   out_2728954879743535149[233] = 0.0;
   out_2728954879743535149[234] = 0.0;
   out_2728954879743535149[235] = 0.0;
   out_2728954879743535149[236] = 0.0;
   out_2728954879743535149[237] = 0.0;
   out_2728954879743535149[238] = 0.0;
   out_2728954879743535149[239] = 0.0;
   out_2728954879743535149[240] = 0.0;
   out_2728954879743535149[241] = 0.0;
   out_2728954879743535149[242] = 0.0;
   out_2728954879743535149[243] = 0.0;
   out_2728954879743535149[244] = 0.0;
   out_2728954879743535149[245] = 0.0;
   out_2728954879743535149[246] = 0.0;
   out_2728954879743535149[247] = 1.0;
   out_2728954879743535149[248] = 0.0;
   out_2728954879743535149[249] = 0.0;
   out_2728954879743535149[250] = 0.0;
   out_2728954879743535149[251] = 0.0;
   out_2728954879743535149[252] = 0.0;
   out_2728954879743535149[253] = 0.0;
   out_2728954879743535149[254] = 0.0;
   out_2728954879743535149[255] = 0.0;
   out_2728954879743535149[256] = 0.0;
   out_2728954879743535149[257] = 0.0;
   out_2728954879743535149[258] = 0.0;
   out_2728954879743535149[259] = 0.0;
   out_2728954879743535149[260] = 0.0;
   out_2728954879743535149[261] = 0.0;
   out_2728954879743535149[262] = 0.0;
   out_2728954879743535149[263] = 0.0;
   out_2728954879743535149[264] = 0.0;
   out_2728954879743535149[265] = 0.0;
   out_2728954879743535149[266] = 1.0;
   out_2728954879743535149[267] = 0.0;
   out_2728954879743535149[268] = 0.0;
   out_2728954879743535149[269] = 0.0;
   out_2728954879743535149[270] = 0.0;
   out_2728954879743535149[271] = 0.0;
   out_2728954879743535149[272] = 0.0;
   out_2728954879743535149[273] = 0.0;
   out_2728954879743535149[274] = 0.0;
   out_2728954879743535149[275] = 0.0;
   out_2728954879743535149[276] = 0.0;
   out_2728954879743535149[277] = 0.0;
   out_2728954879743535149[278] = 0.0;
   out_2728954879743535149[279] = 0.0;
   out_2728954879743535149[280] = 0.0;
   out_2728954879743535149[281] = 0.0;
   out_2728954879743535149[282] = 0.0;
   out_2728954879743535149[283] = 0.0;
   out_2728954879743535149[284] = 0.0;
   out_2728954879743535149[285] = 1.0;
   out_2728954879743535149[286] = 0.0;
   out_2728954879743535149[287] = 0.0;
   out_2728954879743535149[288] = 0.0;
   out_2728954879743535149[289] = 0.0;
   out_2728954879743535149[290] = 0.0;
   out_2728954879743535149[291] = 0.0;
   out_2728954879743535149[292] = 0.0;
   out_2728954879743535149[293] = 0.0;
   out_2728954879743535149[294] = 0.0;
   out_2728954879743535149[295] = 0.0;
   out_2728954879743535149[296] = 0.0;
   out_2728954879743535149[297] = 0.0;
   out_2728954879743535149[298] = 0.0;
   out_2728954879743535149[299] = 0.0;
   out_2728954879743535149[300] = 0.0;
   out_2728954879743535149[301] = 0.0;
   out_2728954879743535149[302] = 0.0;
   out_2728954879743535149[303] = 0.0;
   out_2728954879743535149[304] = 1.0;
   out_2728954879743535149[305] = 0.0;
   out_2728954879743535149[306] = 0.0;
   out_2728954879743535149[307] = 0.0;
   out_2728954879743535149[308] = 0.0;
   out_2728954879743535149[309] = 0.0;
   out_2728954879743535149[310] = 0.0;
   out_2728954879743535149[311] = 0.0;
   out_2728954879743535149[312] = 0.0;
   out_2728954879743535149[313] = 0.0;
   out_2728954879743535149[314] = 0.0;
   out_2728954879743535149[315] = 0.0;
   out_2728954879743535149[316] = 0.0;
   out_2728954879743535149[317] = 0.0;
   out_2728954879743535149[318] = 0.0;
   out_2728954879743535149[319] = 0.0;
   out_2728954879743535149[320] = 0.0;
   out_2728954879743535149[321] = 0.0;
   out_2728954879743535149[322] = 0.0;
   out_2728954879743535149[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_4998926688300100454) {
   out_4998926688300100454[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_4998926688300100454[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_4998926688300100454[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_4998926688300100454[3] = dt*state[12] + state[3];
   out_4998926688300100454[4] = dt*state[13] + state[4];
   out_4998926688300100454[5] = dt*state[14] + state[5];
   out_4998926688300100454[6] = state[6];
   out_4998926688300100454[7] = state[7];
   out_4998926688300100454[8] = state[8];
   out_4998926688300100454[9] = state[9];
   out_4998926688300100454[10] = state[10];
   out_4998926688300100454[11] = state[11];
   out_4998926688300100454[12] = state[12];
   out_4998926688300100454[13] = state[13];
   out_4998926688300100454[14] = state[14];
   out_4998926688300100454[15] = state[15];
   out_4998926688300100454[16] = state[16];
   out_4998926688300100454[17] = state[17];
}
void F_fun(double *state, double dt, double *out_8226313807756837064) {
   out_8226313807756837064[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8226313807756837064[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8226313807756837064[2] = 0;
   out_8226313807756837064[3] = 0;
   out_8226313807756837064[4] = 0;
   out_8226313807756837064[5] = 0;
   out_8226313807756837064[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8226313807756837064[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8226313807756837064[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8226313807756837064[9] = 0;
   out_8226313807756837064[10] = 0;
   out_8226313807756837064[11] = 0;
   out_8226313807756837064[12] = 0;
   out_8226313807756837064[13] = 0;
   out_8226313807756837064[14] = 0;
   out_8226313807756837064[15] = 0;
   out_8226313807756837064[16] = 0;
   out_8226313807756837064[17] = 0;
   out_8226313807756837064[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8226313807756837064[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8226313807756837064[20] = 0;
   out_8226313807756837064[21] = 0;
   out_8226313807756837064[22] = 0;
   out_8226313807756837064[23] = 0;
   out_8226313807756837064[24] = 0;
   out_8226313807756837064[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8226313807756837064[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8226313807756837064[27] = 0;
   out_8226313807756837064[28] = 0;
   out_8226313807756837064[29] = 0;
   out_8226313807756837064[30] = 0;
   out_8226313807756837064[31] = 0;
   out_8226313807756837064[32] = 0;
   out_8226313807756837064[33] = 0;
   out_8226313807756837064[34] = 0;
   out_8226313807756837064[35] = 0;
   out_8226313807756837064[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8226313807756837064[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8226313807756837064[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8226313807756837064[39] = 0;
   out_8226313807756837064[40] = 0;
   out_8226313807756837064[41] = 0;
   out_8226313807756837064[42] = 0;
   out_8226313807756837064[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8226313807756837064[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8226313807756837064[45] = 0;
   out_8226313807756837064[46] = 0;
   out_8226313807756837064[47] = 0;
   out_8226313807756837064[48] = 0;
   out_8226313807756837064[49] = 0;
   out_8226313807756837064[50] = 0;
   out_8226313807756837064[51] = 0;
   out_8226313807756837064[52] = 0;
   out_8226313807756837064[53] = 0;
   out_8226313807756837064[54] = 0;
   out_8226313807756837064[55] = 0;
   out_8226313807756837064[56] = 0;
   out_8226313807756837064[57] = 1;
   out_8226313807756837064[58] = 0;
   out_8226313807756837064[59] = 0;
   out_8226313807756837064[60] = 0;
   out_8226313807756837064[61] = 0;
   out_8226313807756837064[62] = 0;
   out_8226313807756837064[63] = 0;
   out_8226313807756837064[64] = 0;
   out_8226313807756837064[65] = 0;
   out_8226313807756837064[66] = dt;
   out_8226313807756837064[67] = 0;
   out_8226313807756837064[68] = 0;
   out_8226313807756837064[69] = 0;
   out_8226313807756837064[70] = 0;
   out_8226313807756837064[71] = 0;
   out_8226313807756837064[72] = 0;
   out_8226313807756837064[73] = 0;
   out_8226313807756837064[74] = 0;
   out_8226313807756837064[75] = 0;
   out_8226313807756837064[76] = 1;
   out_8226313807756837064[77] = 0;
   out_8226313807756837064[78] = 0;
   out_8226313807756837064[79] = 0;
   out_8226313807756837064[80] = 0;
   out_8226313807756837064[81] = 0;
   out_8226313807756837064[82] = 0;
   out_8226313807756837064[83] = 0;
   out_8226313807756837064[84] = 0;
   out_8226313807756837064[85] = dt;
   out_8226313807756837064[86] = 0;
   out_8226313807756837064[87] = 0;
   out_8226313807756837064[88] = 0;
   out_8226313807756837064[89] = 0;
   out_8226313807756837064[90] = 0;
   out_8226313807756837064[91] = 0;
   out_8226313807756837064[92] = 0;
   out_8226313807756837064[93] = 0;
   out_8226313807756837064[94] = 0;
   out_8226313807756837064[95] = 1;
   out_8226313807756837064[96] = 0;
   out_8226313807756837064[97] = 0;
   out_8226313807756837064[98] = 0;
   out_8226313807756837064[99] = 0;
   out_8226313807756837064[100] = 0;
   out_8226313807756837064[101] = 0;
   out_8226313807756837064[102] = 0;
   out_8226313807756837064[103] = 0;
   out_8226313807756837064[104] = dt;
   out_8226313807756837064[105] = 0;
   out_8226313807756837064[106] = 0;
   out_8226313807756837064[107] = 0;
   out_8226313807756837064[108] = 0;
   out_8226313807756837064[109] = 0;
   out_8226313807756837064[110] = 0;
   out_8226313807756837064[111] = 0;
   out_8226313807756837064[112] = 0;
   out_8226313807756837064[113] = 0;
   out_8226313807756837064[114] = 1;
   out_8226313807756837064[115] = 0;
   out_8226313807756837064[116] = 0;
   out_8226313807756837064[117] = 0;
   out_8226313807756837064[118] = 0;
   out_8226313807756837064[119] = 0;
   out_8226313807756837064[120] = 0;
   out_8226313807756837064[121] = 0;
   out_8226313807756837064[122] = 0;
   out_8226313807756837064[123] = 0;
   out_8226313807756837064[124] = 0;
   out_8226313807756837064[125] = 0;
   out_8226313807756837064[126] = 0;
   out_8226313807756837064[127] = 0;
   out_8226313807756837064[128] = 0;
   out_8226313807756837064[129] = 0;
   out_8226313807756837064[130] = 0;
   out_8226313807756837064[131] = 0;
   out_8226313807756837064[132] = 0;
   out_8226313807756837064[133] = 1;
   out_8226313807756837064[134] = 0;
   out_8226313807756837064[135] = 0;
   out_8226313807756837064[136] = 0;
   out_8226313807756837064[137] = 0;
   out_8226313807756837064[138] = 0;
   out_8226313807756837064[139] = 0;
   out_8226313807756837064[140] = 0;
   out_8226313807756837064[141] = 0;
   out_8226313807756837064[142] = 0;
   out_8226313807756837064[143] = 0;
   out_8226313807756837064[144] = 0;
   out_8226313807756837064[145] = 0;
   out_8226313807756837064[146] = 0;
   out_8226313807756837064[147] = 0;
   out_8226313807756837064[148] = 0;
   out_8226313807756837064[149] = 0;
   out_8226313807756837064[150] = 0;
   out_8226313807756837064[151] = 0;
   out_8226313807756837064[152] = 1;
   out_8226313807756837064[153] = 0;
   out_8226313807756837064[154] = 0;
   out_8226313807756837064[155] = 0;
   out_8226313807756837064[156] = 0;
   out_8226313807756837064[157] = 0;
   out_8226313807756837064[158] = 0;
   out_8226313807756837064[159] = 0;
   out_8226313807756837064[160] = 0;
   out_8226313807756837064[161] = 0;
   out_8226313807756837064[162] = 0;
   out_8226313807756837064[163] = 0;
   out_8226313807756837064[164] = 0;
   out_8226313807756837064[165] = 0;
   out_8226313807756837064[166] = 0;
   out_8226313807756837064[167] = 0;
   out_8226313807756837064[168] = 0;
   out_8226313807756837064[169] = 0;
   out_8226313807756837064[170] = 0;
   out_8226313807756837064[171] = 1;
   out_8226313807756837064[172] = 0;
   out_8226313807756837064[173] = 0;
   out_8226313807756837064[174] = 0;
   out_8226313807756837064[175] = 0;
   out_8226313807756837064[176] = 0;
   out_8226313807756837064[177] = 0;
   out_8226313807756837064[178] = 0;
   out_8226313807756837064[179] = 0;
   out_8226313807756837064[180] = 0;
   out_8226313807756837064[181] = 0;
   out_8226313807756837064[182] = 0;
   out_8226313807756837064[183] = 0;
   out_8226313807756837064[184] = 0;
   out_8226313807756837064[185] = 0;
   out_8226313807756837064[186] = 0;
   out_8226313807756837064[187] = 0;
   out_8226313807756837064[188] = 0;
   out_8226313807756837064[189] = 0;
   out_8226313807756837064[190] = 1;
   out_8226313807756837064[191] = 0;
   out_8226313807756837064[192] = 0;
   out_8226313807756837064[193] = 0;
   out_8226313807756837064[194] = 0;
   out_8226313807756837064[195] = 0;
   out_8226313807756837064[196] = 0;
   out_8226313807756837064[197] = 0;
   out_8226313807756837064[198] = 0;
   out_8226313807756837064[199] = 0;
   out_8226313807756837064[200] = 0;
   out_8226313807756837064[201] = 0;
   out_8226313807756837064[202] = 0;
   out_8226313807756837064[203] = 0;
   out_8226313807756837064[204] = 0;
   out_8226313807756837064[205] = 0;
   out_8226313807756837064[206] = 0;
   out_8226313807756837064[207] = 0;
   out_8226313807756837064[208] = 0;
   out_8226313807756837064[209] = 1;
   out_8226313807756837064[210] = 0;
   out_8226313807756837064[211] = 0;
   out_8226313807756837064[212] = 0;
   out_8226313807756837064[213] = 0;
   out_8226313807756837064[214] = 0;
   out_8226313807756837064[215] = 0;
   out_8226313807756837064[216] = 0;
   out_8226313807756837064[217] = 0;
   out_8226313807756837064[218] = 0;
   out_8226313807756837064[219] = 0;
   out_8226313807756837064[220] = 0;
   out_8226313807756837064[221] = 0;
   out_8226313807756837064[222] = 0;
   out_8226313807756837064[223] = 0;
   out_8226313807756837064[224] = 0;
   out_8226313807756837064[225] = 0;
   out_8226313807756837064[226] = 0;
   out_8226313807756837064[227] = 0;
   out_8226313807756837064[228] = 1;
   out_8226313807756837064[229] = 0;
   out_8226313807756837064[230] = 0;
   out_8226313807756837064[231] = 0;
   out_8226313807756837064[232] = 0;
   out_8226313807756837064[233] = 0;
   out_8226313807756837064[234] = 0;
   out_8226313807756837064[235] = 0;
   out_8226313807756837064[236] = 0;
   out_8226313807756837064[237] = 0;
   out_8226313807756837064[238] = 0;
   out_8226313807756837064[239] = 0;
   out_8226313807756837064[240] = 0;
   out_8226313807756837064[241] = 0;
   out_8226313807756837064[242] = 0;
   out_8226313807756837064[243] = 0;
   out_8226313807756837064[244] = 0;
   out_8226313807756837064[245] = 0;
   out_8226313807756837064[246] = 0;
   out_8226313807756837064[247] = 1;
   out_8226313807756837064[248] = 0;
   out_8226313807756837064[249] = 0;
   out_8226313807756837064[250] = 0;
   out_8226313807756837064[251] = 0;
   out_8226313807756837064[252] = 0;
   out_8226313807756837064[253] = 0;
   out_8226313807756837064[254] = 0;
   out_8226313807756837064[255] = 0;
   out_8226313807756837064[256] = 0;
   out_8226313807756837064[257] = 0;
   out_8226313807756837064[258] = 0;
   out_8226313807756837064[259] = 0;
   out_8226313807756837064[260] = 0;
   out_8226313807756837064[261] = 0;
   out_8226313807756837064[262] = 0;
   out_8226313807756837064[263] = 0;
   out_8226313807756837064[264] = 0;
   out_8226313807756837064[265] = 0;
   out_8226313807756837064[266] = 1;
   out_8226313807756837064[267] = 0;
   out_8226313807756837064[268] = 0;
   out_8226313807756837064[269] = 0;
   out_8226313807756837064[270] = 0;
   out_8226313807756837064[271] = 0;
   out_8226313807756837064[272] = 0;
   out_8226313807756837064[273] = 0;
   out_8226313807756837064[274] = 0;
   out_8226313807756837064[275] = 0;
   out_8226313807756837064[276] = 0;
   out_8226313807756837064[277] = 0;
   out_8226313807756837064[278] = 0;
   out_8226313807756837064[279] = 0;
   out_8226313807756837064[280] = 0;
   out_8226313807756837064[281] = 0;
   out_8226313807756837064[282] = 0;
   out_8226313807756837064[283] = 0;
   out_8226313807756837064[284] = 0;
   out_8226313807756837064[285] = 1;
   out_8226313807756837064[286] = 0;
   out_8226313807756837064[287] = 0;
   out_8226313807756837064[288] = 0;
   out_8226313807756837064[289] = 0;
   out_8226313807756837064[290] = 0;
   out_8226313807756837064[291] = 0;
   out_8226313807756837064[292] = 0;
   out_8226313807756837064[293] = 0;
   out_8226313807756837064[294] = 0;
   out_8226313807756837064[295] = 0;
   out_8226313807756837064[296] = 0;
   out_8226313807756837064[297] = 0;
   out_8226313807756837064[298] = 0;
   out_8226313807756837064[299] = 0;
   out_8226313807756837064[300] = 0;
   out_8226313807756837064[301] = 0;
   out_8226313807756837064[302] = 0;
   out_8226313807756837064[303] = 0;
   out_8226313807756837064[304] = 1;
   out_8226313807756837064[305] = 0;
   out_8226313807756837064[306] = 0;
   out_8226313807756837064[307] = 0;
   out_8226313807756837064[308] = 0;
   out_8226313807756837064[309] = 0;
   out_8226313807756837064[310] = 0;
   out_8226313807756837064[311] = 0;
   out_8226313807756837064[312] = 0;
   out_8226313807756837064[313] = 0;
   out_8226313807756837064[314] = 0;
   out_8226313807756837064[315] = 0;
   out_8226313807756837064[316] = 0;
   out_8226313807756837064[317] = 0;
   out_8226313807756837064[318] = 0;
   out_8226313807756837064[319] = 0;
   out_8226313807756837064[320] = 0;
   out_8226313807756837064[321] = 0;
   out_8226313807756837064[322] = 0;
   out_8226313807756837064[323] = 1;
}
void h_4(double *state, double *unused, double *out_3272625519318058906) {
   out_3272625519318058906[0] = state[6] + state[9];
   out_3272625519318058906[1] = state[7] + state[10];
   out_3272625519318058906[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_8577036686495493870) {
   out_8577036686495493870[0] = 0;
   out_8577036686495493870[1] = 0;
   out_8577036686495493870[2] = 0;
   out_8577036686495493870[3] = 0;
   out_8577036686495493870[4] = 0;
   out_8577036686495493870[5] = 0;
   out_8577036686495493870[6] = 1;
   out_8577036686495493870[7] = 0;
   out_8577036686495493870[8] = 0;
   out_8577036686495493870[9] = 1;
   out_8577036686495493870[10] = 0;
   out_8577036686495493870[11] = 0;
   out_8577036686495493870[12] = 0;
   out_8577036686495493870[13] = 0;
   out_8577036686495493870[14] = 0;
   out_8577036686495493870[15] = 0;
   out_8577036686495493870[16] = 0;
   out_8577036686495493870[17] = 0;
   out_8577036686495493870[18] = 0;
   out_8577036686495493870[19] = 0;
   out_8577036686495493870[20] = 0;
   out_8577036686495493870[21] = 0;
   out_8577036686495493870[22] = 0;
   out_8577036686495493870[23] = 0;
   out_8577036686495493870[24] = 0;
   out_8577036686495493870[25] = 1;
   out_8577036686495493870[26] = 0;
   out_8577036686495493870[27] = 0;
   out_8577036686495493870[28] = 1;
   out_8577036686495493870[29] = 0;
   out_8577036686495493870[30] = 0;
   out_8577036686495493870[31] = 0;
   out_8577036686495493870[32] = 0;
   out_8577036686495493870[33] = 0;
   out_8577036686495493870[34] = 0;
   out_8577036686495493870[35] = 0;
   out_8577036686495493870[36] = 0;
   out_8577036686495493870[37] = 0;
   out_8577036686495493870[38] = 0;
   out_8577036686495493870[39] = 0;
   out_8577036686495493870[40] = 0;
   out_8577036686495493870[41] = 0;
   out_8577036686495493870[42] = 0;
   out_8577036686495493870[43] = 0;
   out_8577036686495493870[44] = 1;
   out_8577036686495493870[45] = 0;
   out_8577036686495493870[46] = 0;
   out_8577036686495493870[47] = 1;
   out_8577036686495493870[48] = 0;
   out_8577036686495493870[49] = 0;
   out_8577036686495493870[50] = 0;
   out_8577036686495493870[51] = 0;
   out_8577036686495493870[52] = 0;
   out_8577036686495493870[53] = 0;
}
void h_10(double *state, double *unused, double *out_7478176691014094533) {
   out_7478176691014094533[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_7478176691014094533[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_7478176691014094533[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_3894289231047876903) {
   out_3894289231047876903[0] = 0;
   out_3894289231047876903[1] = 9.8100000000000005*cos(state[1]);
   out_3894289231047876903[2] = 0;
   out_3894289231047876903[3] = 0;
   out_3894289231047876903[4] = -state[8];
   out_3894289231047876903[5] = state[7];
   out_3894289231047876903[6] = 0;
   out_3894289231047876903[7] = state[5];
   out_3894289231047876903[8] = -state[4];
   out_3894289231047876903[9] = 0;
   out_3894289231047876903[10] = 0;
   out_3894289231047876903[11] = 0;
   out_3894289231047876903[12] = 1;
   out_3894289231047876903[13] = 0;
   out_3894289231047876903[14] = 0;
   out_3894289231047876903[15] = 1;
   out_3894289231047876903[16] = 0;
   out_3894289231047876903[17] = 0;
   out_3894289231047876903[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_3894289231047876903[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_3894289231047876903[20] = 0;
   out_3894289231047876903[21] = state[8];
   out_3894289231047876903[22] = 0;
   out_3894289231047876903[23] = -state[6];
   out_3894289231047876903[24] = -state[5];
   out_3894289231047876903[25] = 0;
   out_3894289231047876903[26] = state[3];
   out_3894289231047876903[27] = 0;
   out_3894289231047876903[28] = 0;
   out_3894289231047876903[29] = 0;
   out_3894289231047876903[30] = 0;
   out_3894289231047876903[31] = 1;
   out_3894289231047876903[32] = 0;
   out_3894289231047876903[33] = 0;
   out_3894289231047876903[34] = 1;
   out_3894289231047876903[35] = 0;
   out_3894289231047876903[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_3894289231047876903[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_3894289231047876903[38] = 0;
   out_3894289231047876903[39] = -state[7];
   out_3894289231047876903[40] = state[6];
   out_3894289231047876903[41] = 0;
   out_3894289231047876903[42] = state[4];
   out_3894289231047876903[43] = -state[3];
   out_3894289231047876903[44] = 0;
   out_3894289231047876903[45] = 0;
   out_3894289231047876903[46] = 0;
   out_3894289231047876903[47] = 0;
   out_3894289231047876903[48] = 0;
   out_3894289231047876903[49] = 0;
   out_3894289231047876903[50] = 1;
   out_3894289231047876903[51] = 0;
   out_3894289231047876903[52] = 0;
   out_3894289231047876903[53] = 1;
}
void h_13(double *state, double *unused, double *out_257622233170053408) {
   out_257622233170053408[0] = state[3];
   out_257622233170053408[1] = state[4];
   out_257622233170053408[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5364762861163161069) {
   out_5364762861163161069[0] = 0;
   out_5364762861163161069[1] = 0;
   out_5364762861163161069[2] = 0;
   out_5364762861163161069[3] = 1;
   out_5364762861163161069[4] = 0;
   out_5364762861163161069[5] = 0;
   out_5364762861163161069[6] = 0;
   out_5364762861163161069[7] = 0;
   out_5364762861163161069[8] = 0;
   out_5364762861163161069[9] = 0;
   out_5364762861163161069[10] = 0;
   out_5364762861163161069[11] = 0;
   out_5364762861163161069[12] = 0;
   out_5364762861163161069[13] = 0;
   out_5364762861163161069[14] = 0;
   out_5364762861163161069[15] = 0;
   out_5364762861163161069[16] = 0;
   out_5364762861163161069[17] = 0;
   out_5364762861163161069[18] = 0;
   out_5364762861163161069[19] = 0;
   out_5364762861163161069[20] = 0;
   out_5364762861163161069[21] = 0;
   out_5364762861163161069[22] = 1;
   out_5364762861163161069[23] = 0;
   out_5364762861163161069[24] = 0;
   out_5364762861163161069[25] = 0;
   out_5364762861163161069[26] = 0;
   out_5364762861163161069[27] = 0;
   out_5364762861163161069[28] = 0;
   out_5364762861163161069[29] = 0;
   out_5364762861163161069[30] = 0;
   out_5364762861163161069[31] = 0;
   out_5364762861163161069[32] = 0;
   out_5364762861163161069[33] = 0;
   out_5364762861163161069[34] = 0;
   out_5364762861163161069[35] = 0;
   out_5364762861163161069[36] = 0;
   out_5364762861163161069[37] = 0;
   out_5364762861163161069[38] = 0;
   out_5364762861163161069[39] = 0;
   out_5364762861163161069[40] = 0;
   out_5364762861163161069[41] = 1;
   out_5364762861163161069[42] = 0;
   out_5364762861163161069[43] = 0;
   out_5364762861163161069[44] = 0;
   out_5364762861163161069[45] = 0;
   out_5364762861163161069[46] = 0;
   out_5364762861163161069[47] = 0;
   out_5364762861163161069[48] = 0;
   out_5364762861163161069[49] = 0;
   out_5364762861163161069[50] = 0;
   out_5364762861163161069[51] = 0;
   out_5364762861163161069[52] = 0;
   out_5364762861163161069[53] = 0;
}
void h_14(double *state, double *unused, double *out_2543715519988584578) {
   out_2543715519988584578[0] = state[6];
   out_2543715519988584578[1] = state[7];
   out_2543715519988584578[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4613795830156009341) {
   out_4613795830156009341[0] = 0;
   out_4613795830156009341[1] = 0;
   out_4613795830156009341[2] = 0;
   out_4613795830156009341[3] = 0;
   out_4613795830156009341[4] = 0;
   out_4613795830156009341[5] = 0;
   out_4613795830156009341[6] = 1;
   out_4613795830156009341[7] = 0;
   out_4613795830156009341[8] = 0;
   out_4613795830156009341[9] = 0;
   out_4613795830156009341[10] = 0;
   out_4613795830156009341[11] = 0;
   out_4613795830156009341[12] = 0;
   out_4613795830156009341[13] = 0;
   out_4613795830156009341[14] = 0;
   out_4613795830156009341[15] = 0;
   out_4613795830156009341[16] = 0;
   out_4613795830156009341[17] = 0;
   out_4613795830156009341[18] = 0;
   out_4613795830156009341[19] = 0;
   out_4613795830156009341[20] = 0;
   out_4613795830156009341[21] = 0;
   out_4613795830156009341[22] = 0;
   out_4613795830156009341[23] = 0;
   out_4613795830156009341[24] = 0;
   out_4613795830156009341[25] = 1;
   out_4613795830156009341[26] = 0;
   out_4613795830156009341[27] = 0;
   out_4613795830156009341[28] = 0;
   out_4613795830156009341[29] = 0;
   out_4613795830156009341[30] = 0;
   out_4613795830156009341[31] = 0;
   out_4613795830156009341[32] = 0;
   out_4613795830156009341[33] = 0;
   out_4613795830156009341[34] = 0;
   out_4613795830156009341[35] = 0;
   out_4613795830156009341[36] = 0;
   out_4613795830156009341[37] = 0;
   out_4613795830156009341[38] = 0;
   out_4613795830156009341[39] = 0;
   out_4613795830156009341[40] = 0;
   out_4613795830156009341[41] = 0;
   out_4613795830156009341[42] = 0;
   out_4613795830156009341[43] = 0;
   out_4613795830156009341[44] = 1;
   out_4613795830156009341[45] = 0;
   out_4613795830156009341[46] = 0;
   out_4613795830156009341[47] = 0;
   out_4613795830156009341[48] = 0;
   out_4613795830156009341[49] = 0;
   out_4613795830156009341[50] = 0;
   out_4613795830156009341[51] = 0;
   out_4613795830156009341[52] = 0;
   out_4613795830156009341[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_7464579881794296048) {
  err_fun(nom_x, delta_x, out_7464579881794296048);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2725303949871892042) {
  inv_err_fun(nom_x, true_x, out_2725303949871892042);
}
void pose_H_mod_fun(double *state, double *out_2728954879743535149) {
  H_mod_fun(state, out_2728954879743535149);
}
void pose_f_fun(double *state, double dt, double *out_4998926688300100454) {
  f_fun(state,  dt, out_4998926688300100454);
}
void pose_F_fun(double *state, double dt, double *out_8226313807756837064) {
  F_fun(state,  dt, out_8226313807756837064);
}
void pose_h_4(double *state, double *unused, double *out_3272625519318058906) {
  h_4(state, unused, out_3272625519318058906);
}
void pose_H_4(double *state, double *unused, double *out_8577036686495493870) {
  H_4(state, unused, out_8577036686495493870);
}
void pose_h_10(double *state, double *unused, double *out_7478176691014094533) {
  h_10(state, unused, out_7478176691014094533);
}
void pose_H_10(double *state, double *unused, double *out_3894289231047876903) {
  H_10(state, unused, out_3894289231047876903);
}
void pose_h_13(double *state, double *unused, double *out_257622233170053408) {
  h_13(state, unused, out_257622233170053408);
}
void pose_H_13(double *state, double *unused, double *out_5364762861163161069) {
  H_13(state, unused, out_5364762861163161069);
}
void pose_h_14(double *state, double *unused, double *out_2543715519988584578) {
  h_14(state, unused, out_2543715519988584578);
}
void pose_H_14(double *state, double *unused, double *out_4613795830156009341) {
  H_14(state, unused, out_4613795830156009341);
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
