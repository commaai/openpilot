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
void err_fun(double *nom_x, double *delta_x, double *out_6544138591762530502) {
   out_6544138591762530502[0] = delta_x[0] + nom_x[0];
   out_6544138591762530502[1] = delta_x[1] + nom_x[1];
   out_6544138591762530502[2] = delta_x[2] + nom_x[2];
   out_6544138591762530502[3] = delta_x[3] + nom_x[3];
   out_6544138591762530502[4] = delta_x[4] + nom_x[4];
   out_6544138591762530502[5] = delta_x[5] + nom_x[5];
   out_6544138591762530502[6] = delta_x[6] + nom_x[6];
   out_6544138591762530502[7] = delta_x[7] + nom_x[7];
   out_6544138591762530502[8] = delta_x[8] + nom_x[8];
   out_6544138591762530502[9] = delta_x[9] + nom_x[9];
   out_6544138591762530502[10] = delta_x[10] + nom_x[10];
   out_6544138591762530502[11] = delta_x[11] + nom_x[11];
   out_6544138591762530502[12] = delta_x[12] + nom_x[12];
   out_6544138591762530502[13] = delta_x[13] + nom_x[13];
   out_6544138591762530502[14] = delta_x[14] + nom_x[14];
   out_6544138591762530502[15] = delta_x[15] + nom_x[15];
   out_6544138591762530502[16] = delta_x[16] + nom_x[16];
   out_6544138591762530502[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5883871402494837793) {
   out_5883871402494837793[0] = -nom_x[0] + true_x[0];
   out_5883871402494837793[1] = -nom_x[1] + true_x[1];
   out_5883871402494837793[2] = -nom_x[2] + true_x[2];
   out_5883871402494837793[3] = -nom_x[3] + true_x[3];
   out_5883871402494837793[4] = -nom_x[4] + true_x[4];
   out_5883871402494837793[5] = -nom_x[5] + true_x[5];
   out_5883871402494837793[6] = -nom_x[6] + true_x[6];
   out_5883871402494837793[7] = -nom_x[7] + true_x[7];
   out_5883871402494837793[8] = -nom_x[8] + true_x[8];
   out_5883871402494837793[9] = -nom_x[9] + true_x[9];
   out_5883871402494837793[10] = -nom_x[10] + true_x[10];
   out_5883871402494837793[11] = -nom_x[11] + true_x[11];
   out_5883871402494837793[12] = -nom_x[12] + true_x[12];
   out_5883871402494837793[13] = -nom_x[13] + true_x[13];
   out_5883871402494837793[14] = -nom_x[14] + true_x[14];
   out_5883871402494837793[15] = -nom_x[15] + true_x[15];
   out_5883871402494837793[16] = -nom_x[16] + true_x[16];
   out_5883871402494837793[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_7964293451172914980) {
   out_7964293451172914980[0] = 1.0;
   out_7964293451172914980[1] = 0.0;
   out_7964293451172914980[2] = 0.0;
   out_7964293451172914980[3] = 0.0;
   out_7964293451172914980[4] = 0.0;
   out_7964293451172914980[5] = 0.0;
   out_7964293451172914980[6] = 0.0;
   out_7964293451172914980[7] = 0.0;
   out_7964293451172914980[8] = 0.0;
   out_7964293451172914980[9] = 0.0;
   out_7964293451172914980[10] = 0.0;
   out_7964293451172914980[11] = 0.0;
   out_7964293451172914980[12] = 0.0;
   out_7964293451172914980[13] = 0.0;
   out_7964293451172914980[14] = 0.0;
   out_7964293451172914980[15] = 0.0;
   out_7964293451172914980[16] = 0.0;
   out_7964293451172914980[17] = 0.0;
   out_7964293451172914980[18] = 0.0;
   out_7964293451172914980[19] = 1.0;
   out_7964293451172914980[20] = 0.0;
   out_7964293451172914980[21] = 0.0;
   out_7964293451172914980[22] = 0.0;
   out_7964293451172914980[23] = 0.0;
   out_7964293451172914980[24] = 0.0;
   out_7964293451172914980[25] = 0.0;
   out_7964293451172914980[26] = 0.0;
   out_7964293451172914980[27] = 0.0;
   out_7964293451172914980[28] = 0.0;
   out_7964293451172914980[29] = 0.0;
   out_7964293451172914980[30] = 0.0;
   out_7964293451172914980[31] = 0.0;
   out_7964293451172914980[32] = 0.0;
   out_7964293451172914980[33] = 0.0;
   out_7964293451172914980[34] = 0.0;
   out_7964293451172914980[35] = 0.0;
   out_7964293451172914980[36] = 0.0;
   out_7964293451172914980[37] = 0.0;
   out_7964293451172914980[38] = 1.0;
   out_7964293451172914980[39] = 0.0;
   out_7964293451172914980[40] = 0.0;
   out_7964293451172914980[41] = 0.0;
   out_7964293451172914980[42] = 0.0;
   out_7964293451172914980[43] = 0.0;
   out_7964293451172914980[44] = 0.0;
   out_7964293451172914980[45] = 0.0;
   out_7964293451172914980[46] = 0.0;
   out_7964293451172914980[47] = 0.0;
   out_7964293451172914980[48] = 0.0;
   out_7964293451172914980[49] = 0.0;
   out_7964293451172914980[50] = 0.0;
   out_7964293451172914980[51] = 0.0;
   out_7964293451172914980[52] = 0.0;
   out_7964293451172914980[53] = 0.0;
   out_7964293451172914980[54] = 0.0;
   out_7964293451172914980[55] = 0.0;
   out_7964293451172914980[56] = 0.0;
   out_7964293451172914980[57] = 1.0;
   out_7964293451172914980[58] = 0.0;
   out_7964293451172914980[59] = 0.0;
   out_7964293451172914980[60] = 0.0;
   out_7964293451172914980[61] = 0.0;
   out_7964293451172914980[62] = 0.0;
   out_7964293451172914980[63] = 0.0;
   out_7964293451172914980[64] = 0.0;
   out_7964293451172914980[65] = 0.0;
   out_7964293451172914980[66] = 0.0;
   out_7964293451172914980[67] = 0.0;
   out_7964293451172914980[68] = 0.0;
   out_7964293451172914980[69] = 0.0;
   out_7964293451172914980[70] = 0.0;
   out_7964293451172914980[71] = 0.0;
   out_7964293451172914980[72] = 0.0;
   out_7964293451172914980[73] = 0.0;
   out_7964293451172914980[74] = 0.0;
   out_7964293451172914980[75] = 0.0;
   out_7964293451172914980[76] = 1.0;
   out_7964293451172914980[77] = 0.0;
   out_7964293451172914980[78] = 0.0;
   out_7964293451172914980[79] = 0.0;
   out_7964293451172914980[80] = 0.0;
   out_7964293451172914980[81] = 0.0;
   out_7964293451172914980[82] = 0.0;
   out_7964293451172914980[83] = 0.0;
   out_7964293451172914980[84] = 0.0;
   out_7964293451172914980[85] = 0.0;
   out_7964293451172914980[86] = 0.0;
   out_7964293451172914980[87] = 0.0;
   out_7964293451172914980[88] = 0.0;
   out_7964293451172914980[89] = 0.0;
   out_7964293451172914980[90] = 0.0;
   out_7964293451172914980[91] = 0.0;
   out_7964293451172914980[92] = 0.0;
   out_7964293451172914980[93] = 0.0;
   out_7964293451172914980[94] = 0.0;
   out_7964293451172914980[95] = 1.0;
   out_7964293451172914980[96] = 0.0;
   out_7964293451172914980[97] = 0.0;
   out_7964293451172914980[98] = 0.0;
   out_7964293451172914980[99] = 0.0;
   out_7964293451172914980[100] = 0.0;
   out_7964293451172914980[101] = 0.0;
   out_7964293451172914980[102] = 0.0;
   out_7964293451172914980[103] = 0.0;
   out_7964293451172914980[104] = 0.0;
   out_7964293451172914980[105] = 0.0;
   out_7964293451172914980[106] = 0.0;
   out_7964293451172914980[107] = 0.0;
   out_7964293451172914980[108] = 0.0;
   out_7964293451172914980[109] = 0.0;
   out_7964293451172914980[110] = 0.0;
   out_7964293451172914980[111] = 0.0;
   out_7964293451172914980[112] = 0.0;
   out_7964293451172914980[113] = 0.0;
   out_7964293451172914980[114] = 1.0;
   out_7964293451172914980[115] = 0.0;
   out_7964293451172914980[116] = 0.0;
   out_7964293451172914980[117] = 0.0;
   out_7964293451172914980[118] = 0.0;
   out_7964293451172914980[119] = 0.0;
   out_7964293451172914980[120] = 0.0;
   out_7964293451172914980[121] = 0.0;
   out_7964293451172914980[122] = 0.0;
   out_7964293451172914980[123] = 0.0;
   out_7964293451172914980[124] = 0.0;
   out_7964293451172914980[125] = 0.0;
   out_7964293451172914980[126] = 0.0;
   out_7964293451172914980[127] = 0.0;
   out_7964293451172914980[128] = 0.0;
   out_7964293451172914980[129] = 0.0;
   out_7964293451172914980[130] = 0.0;
   out_7964293451172914980[131] = 0.0;
   out_7964293451172914980[132] = 0.0;
   out_7964293451172914980[133] = 1.0;
   out_7964293451172914980[134] = 0.0;
   out_7964293451172914980[135] = 0.0;
   out_7964293451172914980[136] = 0.0;
   out_7964293451172914980[137] = 0.0;
   out_7964293451172914980[138] = 0.0;
   out_7964293451172914980[139] = 0.0;
   out_7964293451172914980[140] = 0.0;
   out_7964293451172914980[141] = 0.0;
   out_7964293451172914980[142] = 0.0;
   out_7964293451172914980[143] = 0.0;
   out_7964293451172914980[144] = 0.0;
   out_7964293451172914980[145] = 0.0;
   out_7964293451172914980[146] = 0.0;
   out_7964293451172914980[147] = 0.0;
   out_7964293451172914980[148] = 0.0;
   out_7964293451172914980[149] = 0.0;
   out_7964293451172914980[150] = 0.0;
   out_7964293451172914980[151] = 0.0;
   out_7964293451172914980[152] = 1.0;
   out_7964293451172914980[153] = 0.0;
   out_7964293451172914980[154] = 0.0;
   out_7964293451172914980[155] = 0.0;
   out_7964293451172914980[156] = 0.0;
   out_7964293451172914980[157] = 0.0;
   out_7964293451172914980[158] = 0.0;
   out_7964293451172914980[159] = 0.0;
   out_7964293451172914980[160] = 0.0;
   out_7964293451172914980[161] = 0.0;
   out_7964293451172914980[162] = 0.0;
   out_7964293451172914980[163] = 0.0;
   out_7964293451172914980[164] = 0.0;
   out_7964293451172914980[165] = 0.0;
   out_7964293451172914980[166] = 0.0;
   out_7964293451172914980[167] = 0.0;
   out_7964293451172914980[168] = 0.0;
   out_7964293451172914980[169] = 0.0;
   out_7964293451172914980[170] = 0.0;
   out_7964293451172914980[171] = 1.0;
   out_7964293451172914980[172] = 0.0;
   out_7964293451172914980[173] = 0.0;
   out_7964293451172914980[174] = 0.0;
   out_7964293451172914980[175] = 0.0;
   out_7964293451172914980[176] = 0.0;
   out_7964293451172914980[177] = 0.0;
   out_7964293451172914980[178] = 0.0;
   out_7964293451172914980[179] = 0.0;
   out_7964293451172914980[180] = 0.0;
   out_7964293451172914980[181] = 0.0;
   out_7964293451172914980[182] = 0.0;
   out_7964293451172914980[183] = 0.0;
   out_7964293451172914980[184] = 0.0;
   out_7964293451172914980[185] = 0.0;
   out_7964293451172914980[186] = 0.0;
   out_7964293451172914980[187] = 0.0;
   out_7964293451172914980[188] = 0.0;
   out_7964293451172914980[189] = 0.0;
   out_7964293451172914980[190] = 1.0;
   out_7964293451172914980[191] = 0.0;
   out_7964293451172914980[192] = 0.0;
   out_7964293451172914980[193] = 0.0;
   out_7964293451172914980[194] = 0.0;
   out_7964293451172914980[195] = 0.0;
   out_7964293451172914980[196] = 0.0;
   out_7964293451172914980[197] = 0.0;
   out_7964293451172914980[198] = 0.0;
   out_7964293451172914980[199] = 0.0;
   out_7964293451172914980[200] = 0.0;
   out_7964293451172914980[201] = 0.0;
   out_7964293451172914980[202] = 0.0;
   out_7964293451172914980[203] = 0.0;
   out_7964293451172914980[204] = 0.0;
   out_7964293451172914980[205] = 0.0;
   out_7964293451172914980[206] = 0.0;
   out_7964293451172914980[207] = 0.0;
   out_7964293451172914980[208] = 0.0;
   out_7964293451172914980[209] = 1.0;
   out_7964293451172914980[210] = 0.0;
   out_7964293451172914980[211] = 0.0;
   out_7964293451172914980[212] = 0.0;
   out_7964293451172914980[213] = 0.0;
   out_7964293451172914980[214] = 0.0;
   out_7964293451172914980[215] = 0.0;
   out_7964293451172914980[216] = 0.0;
   out_7964293451172914980[217] = 0.0;
   out_7964293451172914980[218] = 0.0;
   out_7964293451172914980[219] = 0.0;
   out_7964293451172914980[220] = 0.0;
   out_7964293451172914980[221] = 0.0;
   out_7964293451172914980[222] = 0.0;
   out_7964293451172914980[223] = 0.0;
   out_7964293451172914980[224] = 0.0;
   out_7964293451172914980[225] = 0.0;
   out_7964293451172914980[226] = 0.0;
   out_7964293451172914980[227] = 0.0;
   out_7964293451172914980[228] = 1.0;
   out_7964293451172914980[229] = 0.0;
   out_7964293451172914980[230] = 0.0;
   out_7964293451172914980[231] = 0.0;
   out_7964293451172914980[232] = 0.0;
   out_7964293451172914980[233] = 0.0;
   out_7964293451172914980[234] = 0.0;
   out_7964293451172914980[235] = 0.0;
   out_7964293451172914980[236] = 0.0;
   out_7964293451172914980[237] = 0.0;
   out_7964293451172914980[238] = 0.0;
   out_7964293451172914980[239] = 0.0;
   out_7964293451172914980[240] = 0.0;
   out_7964293451172914980[241] = 0.0;
   out_7964293451172914980[242] = 0.0;
   out_7964293451172914980[243] = 0.0;
   out_7964293451172914980[244] = 0.0;
   out_7964293451172914980[245] = 0.0;
   out_7964293451172914980[246] = 0.0;
   out_7964293451172914980[247] = 1.0;
   out_7964293451172914980[248] = 0.0;
   out_7964293451172914980[249] = 0.0;
   out_7964293451172914980[250] = 0.0;
   out_7964293451172914980[251] = 0.0;
   out_7964293451172914980[252] = 0.0;
   out_7964293451172914980[253] = 0.0;
   out_7964293451172914980[254] = 0.0;
   out_7964293451172914980[255] = 0.0;
   out_7964293451172914980[256] = 0.0;
   out_7964293451172914980[257] = 0.0;
   out_7964293451172914980[258] = 0.0;
   out_7964293451172914980[259] = 0.0;
   out_7964293451172914980[260] = 0.0;
   out_7964293451172914980[261] = 0.0;
   out_7964293451172914980[262] = 0.0;
   out_7964293451172914980[263] = 0.0;
   out_7964293451172914980[264] = 0.0;
   out_7964293451172914980[265] = 0.0;
   out_7964293451172914980[266] = 1.0;
   out_7964293451172914980[267] = 0.0;
   out_7964293451172914980[268] = 0.0;
   out_7964293451172914980[269] = 0.0;
   out_7964293451172914980[270] = 0.0;
   out_7964293451172914980[271] = 0.0;
   out_7964293451172914980[272] = 0.0;
   out_7964293451172914980[273] = 0.0;
   out_7964293451172914980[274] = 0.0;
   out_7964293451172914980[275] = 0.0;
   out_7964293451172914980[276] = 0.0;
   out_7964293451172914980[277] = 0.0;
   out_7964293451172914980[278] = 0.0;
   out_7964293451172914980[279] = 0.0;
   out_7964293451172914980[280] = 0.0;
   out_7964293451172914980[281] = 0.0;
   out_7964293451172914980[282] = 0.0;
   out_7964293451172914980[283] = 0.0;
   out_7964293451172914980[284] = 0.0;
   out_7964293451172914980[285] = 1.0;
   out_7964293451172914980[286] = 0.0;
   out_7964293451172914980[287] = 0.0;
   out_7964293451172914980[288] = 0.0;
   out_7964293451172914980[289] = 0.0;
   out_7964293451172914980[290] = 0.0;
   out_7964293451172914980[291] = 0.0;
   out_7964293451172914980[292] = 0.0;
   out_7964293451172914980[293] = 0.0;
   out_7964293451172914980[294] = 0.0;
   out_7964293451172914980[295] = 0.0;
   out_7964293451172914980[296] = 0.0;
   out_7964293451172914980[297] = 0.0;
   out_7964293451172914980[298] = 0.0;
   out_7964293451172914980[299] = 0.0;
   out_7964293451172914980[300] = 0.0;
   out_7964293451172914980[301] = 0.0;
   out_7964293451172914980[302] = 0.0;
   out_7964293451172914980[303] = 0.0;
   out_7964293451172914980[304] = 1.0;
   out_7964293451172914980[305] = 0.0;
   out_7964293451172914980[306] = 0.0;
   out_7964293451172914980[307] = 0.0;
   out_7964293451172914980[308] = 0.0;
   out_7964293451172914980[309] = 0.0;
   out_7964293451172914980[310] = 0.0;
   out_7964293451172914980[311] = 0.0;
   out_7964293451172914980[312] = 0.0;
   out_7964293451172914980[313] = 0.0;
   out_7964293451172914980[314] = 0.0;
   out_7964293451172914980[315] = 0.0;
   out_7964293451172914980[316] = 0.0;
   out_7964293451172914980[317] = 0.0;
   out_7964293451172914980[318] = 0.0;
   out_7964293451172914980[319] = 0.0;
   out_7964293451172914980[320] = 0.0;
   out_7964293451172914980[321] = 0.0;
   out_7964293451172914980[322] = 0.0;
   out_7964293451172914980[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7677670493198828015) {
   out_7677670493198828015[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7677670493198828015[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7677670493198828015[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7677670493198828015[3] = dt*state[12] + state[3];
   out_7677670493198828015[4] = dt*state[13] + state[4];
   out_7677670493198828015[5] = dt*state[14] + state[5];
   out_7677670493198828015[6] = state[6];
   out_7677670493198828015[7] = state[7];
   out_7677670493198828015[8] = state[8];
   out_7677670493198828015[9] = state[9];
   out_7677670493198828015[10] = state[10];
   out_7677670493198828015[11] = state[11];
   out_7677670493198828015[12] = state[12];
   out_7677670493198828015[13] = state[13];
   out_7677670493198828015[14] = state[14];
   out_7677670493198828015[15] = state[15];
   out_7677670493198828015[16] = state[16];
   out_7677670493198828015[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2599667736124097937) {
   out_2599667736124097937[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2599667736124097937[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2599667736124097937[2] = 0;
   out_2599667736124097937[3] = 0;
   out_2599667736124097937[4] = 0;
   out_2599667736124097937[5] = 0;
   out_2599667736124097937[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2599667736124097937[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2599667736124097937[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2599667736124097937[9] = 0;
   out_2599667736124097937[10] = 0;
   out_2599667736124097937[11] = 0;
   out_2599667736124097937[12] = 0;
   out_2599667736124097937[13] = 0;
   out_2599667736124097937[14] = 0;
   out_2599667736124097937[15] = 0;
   out_2599667736124097937[16] = 0;
   out_2599667736124097937[17] = 0;
   out_2599667736124097937[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2599667736124097937[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2599667736124097937[20] = 0;
   out_2599667736124097937[21] = 0;
   out_2599667736124097937[22] = 0;
   out_2599667736124097937[23] = 0;
   out_2599667736124097937[24] = 0;
   out_2599667736124097937[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2599667736124097937[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2599667736124097937[27] = 0;
   out_2599667736124097937[28] = 0;
   out_2599667736124097937[29] = 0;
   out_2599667736124097937[30] = 0;
   out_2599667736124097937[31] = 0;
   out_2599667736124097937[32] = 0;
   out_2599667736124097937[33] = 0;
   out_2599667736124097937[34] = 0;
   out_2599667736124097937[35] = 0;
   out_2599667736124097937[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2599667736124097937[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2599667736124097937[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2599667736124097937[39] = 0;
   out_2599667736124097937[40] = 0;
   out_2599667736124097937[41] = 0;
   out_2599667736124097937[42] = 0;
   out_2599667736124097937[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2599667736124097937[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2599667736124097937[45] = 0;
   out_2599667736124097937[46] = 0;
   out_2599667736124097937[47] = 0;
   out_2599667736124097937[48] = 0;
   out_2599667736124097937[49] = 0;
   out_2599667736124097937[50] = 0;
   out_2599667736124097937[51] = 0;
   out_2599667736124097937[52] = 0;
   out_2599667736124097937[53] = 0;
   out_2599667736124097937[54] = 0;
   out_2599667736124097937[55] = 0;
   out_2599667736124097937[56] = 0;
   out_2599667736124097937[57] = 1;
   out_2599667736124097937[58] = 0;
   out_2599667736124097937[59] = 0;
   out_2599667736124097937[60] = 0;
   out_2599667736124097937[61] = 0;
   out_2599667736124097937[62] = 0;
   out_2599667736124097937[63] = 0;
   out_2599667736124097937[64] = 0;
   out_2599667736124097937[65] = 0;
   out_2599667736124097937[66] = dt;
   out_2599667736124097937[67] = 0;
   out_2599667736124097937[68] = 0;
   out_2599667736124097937[69] = 0;
   out_2599667736124097937[70] = 0;
   out_2599667736124097937[71] = 0;
   out_2599667736124097937[72] = 0;
   out_2599667736124097937[73] = 0;
   out_2599667736124097937[74] = 0;
   out_2599667736124097937[75] = 0;
   out_2599667736124097937[76] = 1;
   out_2599667736124097937[77] = 0;
   out_2599667736124097937[78] = 0;
   out_2599667736124097937[79] = 0;
   out_2599667736124097937[80] = 0;
   out_2599667736124097937[81] = 0;
   out_2599667736124097937[82] = 0;
   out_2599667736124097937[83] = 0;
   out_2599667736124097937[84] = 0;
   out_2599667736124097937[85] = dt;
   out_2599667736124097937[86] = 0;
   out_2599667736124097937[87] = 0;
   out_2599667736124097937[88] = 0;
   out_2599667736124097937[89] = 0;
   out_2599667736124097937[90] = 0;
   out_2599667736124097937[91] = 0;
   out_2599667736124097937[92] = 0;
   out_2599667736124097937[93] = 0;
   out_2599667736124097937[94] = 0;
   out_2599667736124097937[95] = 1;
   out_2599667736124097937[96] = 0;
   out_2599667736124097937[97] = 0;
   out_2599667736124097937[98] = 0;
   out_2599667736124097937[99] = 0;
   out_2599667736124097937[100] = 0;
   out_2599667736124097937[101] = 0;
   out_2599667736124097937[102] = 0;
   out_2599667736124097937[103] = 0;
   out_2599667736124097937[104] = dt;
   out_2599667736124097937[105] = 0;
   out_2599667736124097937[106] = 0;
   out_2599667736124097937[107] = 0;
   out_2599667736124097937[108] = 0;
   out_2599667736124097937[109] = 0;
   out_2599667736124097937[110] = 0;
   out_2599667736124097937[111] = 0;
   out_2599667736124097937[112] = 0;
   out_2599667736124097937[113] = 0;
   out_2599667736124097937[114] = 1;
   out_2599667736124097937[115] = 0;
   out_2599667736124097937[116] = 0;
   out_2599667736124097937[117] = 0;
   out_2599667736124097937[118] = 0;
   out_2599667736124097937[119] = 0;
   out_2599667736124097937[120] = 0;
   out_2599667736124097937[121] = 0;
   out_2599667736124097937[122] = 0;
   out_2599667736124097937[123] = 0;
   out_2599667736124097937[124] = 0;
   out_2599667736124097937[125] = 0;
   out_2599667736124097937[126] = 0;
   out_2599667736124097937[127] = 0;
   out_2599667736124097937[128] = 0;
   out_2599667736124097937[129] = 0;
   out_2599667736124097937[130] = 0;
   out_2599667736124097937[131] = 0;
   out_2599667736124097937[132] = 0;
   out_2599667736124097937[133] = 1;
   out_2599667736124097937[134] = 0;
   out_2599667736124097937[135] = 0;
   out_2599667736124097937[136] = 0;
   out_2599667736124097937[137] = 0;
   out_2599667736124097937[138] = 0;
   out_2599667736124097937[139] = 0;
   out_2599667736124097937[140] = 0;
   out_2599667736124097937[141] = 0;
   out_2599667736124097937[142] = 0;
   out_2599667736124097937[143] = 0;
   out_2599667736124097937[144] = 0;
   out_2599667736124097937[145] = 0;
   out_2599667736124097937[146] = 0;
   out_2599667736124097937[147] = 0;
   out_2599667736124097937[148] = 0;
   out_2599667736124097937[149] = 0;
   out_2599667736124097937[150] = 0;
   out_2599667736124097937[151] = 0;
   out_2599667736124097937[152] = 1;
   out_2599667736124097937[153] = 0;
   out_2599667736124097937[154] = 0;
   out_2599667736124097937[155] = 0;
   out_2599667736124097937[156] = 0;
   out_2599667736124097937[157] = 0;
   out_2599667736124097937[158] = 0;
   out_2599667736124097937[159] = 0;
   out_2599667736124097937[160] = 0;
   out_2599667736124097937[161] = 0;
   out_2599667736124097937[162] = 0;
   out_2599667736124097937[163] = 0;
   out_2599667736124097937[164] = 0;
   out_2599667736124097937[165] = 0;
   out_2599667736124097937[166] = 0;
   out_2599667736124097937[167] = 0;
   out_2599667736124097937[168] = 0;
   out_2599667736124097937[169] = 0;
   out_2599667736124097937[170] = 0;
   out_2599667736124097937[171] = 1;
   out_2599667736124097937[172] = 0;
   out_2599667736124097937[173] = 0;
   out_2599667736124097937[174] = 0;
   out_2599667736124097937[175] = 0;
   out_2599667736124097937[176] = 0;
   out_2599667736124097937[177] = 0;
   out_2599667736124097937[178] = 0;
   out_2599667736124097937[179] = 0;
   out_2599667736124097937[180] = 0;
   out_2599667736124097937[181] = 0;
   out_2599667736124097937[182] = 0;
   out_2599667736124097937[183] = 0;
   out_2599667736124097937[184] = 0;
   out_2599667736124097937[185] = 0;
   out_2599667736124097937[186] = 0;
   out_2599667736124097937[187] = 0;
   out_2599667736124097937[188] = 0;
   out_2599667736124097937[189] = 0;
   out_2599667736124097937[190] = 1;
   out_2599667736124097937[191] = 0;
   out_2599667736124097937[192] = 0;
   out_2599667736124097937[193] = 0;
   out_2599667736124097937[194] = 0;
   out_2599667736124097937[195] = 0;
   out_2599667736124097937[196] = 0;
   out_2599667736124097937[197] = 0;
   out_2599667736124097937[198] = 0;
   out_2599667736124097937[199] = 0;
   out_2599667736124097937[200] = 0;
   out_2599667736124097937[201] = 0;
   out_2599667736124097937[202] = 0;
   out_2599667736124097937[203] = 0;
   out_2599667736124097937[204] = 0;
   out_2599667736124097937[205] = 0;
   out_2599667736124097937[206] = 0;
   out_2599667736124097937[207] = 0;
   out_2599667736124097937[208] = 0;
   out_2599667736124097937[209] = 1;
   out_2599667736124097937[210] = 0;
   out_2599667736124097937[211] = 0;
   out_2599667736124097937[212] = 0;
   out_2599667736124097937[213] = 0;
   out_2599667736124097937[214] = 0;
   out_2599667736124097937[215] = 0;
   out_2599667736124097937[216] = 0;
   out_2599667736124097937[217] = 0;
   out_2599667736124097937[218] = 0;
   out_2599667736124097937[219] = 0;
   out_2599667736124097937[220] = 0;
   out_2599667736124097937[221] = 0;
   out_2599667736124097937[222] = 0;
   out_2599667736124097937[223] = 0;
   out_2599667736124097937[224] = 0;
   out_2599667736124097937[225] = 0;
   out_2599667736124097937[226] = 0;
   out_2599667736124097937[227] = 0;
   out_2599667736124097937[228] = 1;
   out_2599667736124097937[229] = 0;
   out_2599667736124097937[230] = 0;
   out_2599667736124097937[231] = 0;
   out_2599667736124097937[232] = 0;
   out_2599667736124097937[233] = 0;
   out_2599667736124097937[234] = 0;
   out_2599667736124097937[235] = 0;
   out_2599667736124097937[236] = 0;
   out_2599667736124097937[237] = 0;
   out_2599667736124097937[238] = 0;
   out_2599667736124097937[239] = 0;
   out_2599667736124097937[240] = 0;
   out_2599667736124097937[241] = 0;
   out_2599667736124097937[242] = 0;
   out_2599667736124097937[243] = 0;
   out_2599667736124097937[244] = 0;
   out_2599667736124097937[245] = 0;
   out_2599667736124097937[246] = 0;
   out_2599667736124097937[247] = 1;
   out_2599667736124097937[248] = 0;
   out_2599667736124097937[249] = 0;
   out_2599667736124097937[250] = 0;
   out_2599667736124097937[251] = 0;
   out_2599667736124097937[252] = 0;
   out_2599667736124097937[253] = 0;
   out_2599667736124097937[254] = 0;
   out_2599667736124097937[255] = 0;
   out_2599667736124097937[256] = 0;
   out_2599667736124097937[257] = 0;
   out_2599667736124097937[258] = 0;
   out_2599667736124097937[259] = 0;
   out_2599667736124097937[260] = 0;
   out_2599667736124097937[261] = 0;
   out_2599667736124097937[262] = 0;
   out_2599667736124097937[263] = 0;
   out_2599667736124097937[264] = 0;
   out_2599667736124097937[265] = 0;
   out_2599667736124097937[266] = 1;
   out_2599667736124097937[267] = 0;
   out_2599667736124097937[268] = 0;
   out_2599667736124097937[269] = 0;
   out_2599667736124097937[270] = 0;
   out_2599667736124097937[271] = 0;
   out_2599667736124097937[272] = 0;
   out_2599667736124097937[273] = 0;
   out_2599667736124097937[274] = 0;
   out_2599667736124097937[275] = 0;
   out_2599667736124097937[276] = 0;
   out_2599667736124097937[277] = 0;
   out_2599667736124097937[278] = 0;
   out_2599667736124097937[279] = 0;
   out_2599667736124097937[280] = 0;
   out_2599667736124097937[281] = 0;
   out_2599667736124097937[282] = 0;
   out_2599667736124097937[283] = 0;
   out_2599667736124097937[284] = 0;
   out_2599667736124097937[285] = 1;
   out_2599667736124097937[286] = 0;
   out_2599667736124097937[287] = 0;
   out_2599667736124097937[288] = 0;
   out_2599667736124097937[289] = 0;
   out_2599667736124097937[290] = 0;
   out_2599667736124097937[291] = 0;
   out_2599667736124097937[292] = 0;
   out_2599667736124097937[293] = 0;
   out_2599667736124097937[294] = 0;
   out_2599667736124097937[295] = 0;
   out_2599667736124097937[296] = 0;
   out_2599667736124097937[297] = 0;
   out_2599667736124097937[298] = 0;
   out_2599667736124097937[299] = 0;
   out_2599667736124097937[300] = 0;
   out_2599667736124097937[301] = 0;
   out_2599667736124097937[302] = 0;
   out_2599667736124097937[303] = 0;
   out_2599667736124097937[304] = 1;
   out_2599667736124097937[305] = 0;
   out_2599667736124097937[306] = 0;
   out_2599667736124097937[307] = 0;
   out_2599667736124097937[308] = 0;
   out_2599667736124097937[309] = 0;
   out_2599667736124097937[310] = 0;
   out_2599667736124097937[311] = 0;
   out_2599667736124097937[312] = 0;
   out_2599667736124097937[313] = 0;
   out_2599667736124097937[314] = 0;
   out_2599667736124097937[315] = 0;
   out_2599667736124097937[316] = 0;
   out_2599667736124097937[317] = 0;
   out_2599667736124097937[318] = 0;
   out_2599667736124097937[319] = 0;
   out_2599667736124097937[320] = 0;
   out_2599667736124097937[321] = 0;
   out_2599667736124097937[322] = 0;
   out_2599667736124097937[323] = 1;
}
void h_4(double *state, double *unused, double *out_6486377135734027290) {
   out_6486377135734027290[0] = state[6] + state[9];
   out_6486377135734027290[1] = state[7] + state[10];
   out_6486377135734027290[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4956672416897748544) {
   out_4956672416897748544[0] = 0;
   out_4956672416897748544[1] = 0;
   out_4956672416897748544[2] = 0;
   out_4956672416897748544[3] = 0;
   out_4956672416897748544[4] = 0;
   out_4956672416897748544[5] = 0;
   out_4956672416897748544[6] = 1;
   out_4956672416897748544[7] = 0;
   out_4956672416897748544[8] = 0;
   out_4956672416897748544[9] = 1;
   out_4956672416897748544[10] = 0;
   out_4956672416897748544[11] = 0;
   out_4956672416897748544[12] = 0;
   out_4956672416897748544[13] = 0;
   out_4956672416897748544[14] = 0;
   out_4956672416897748544[15] = 0;
   out_4956672416897748544[16] = 0;
   out_4956672416897748544[17] = 0;
   out_4956672416897748544[18] = 0;
   out_4956672416897748544[19] = 0;
   out_4956672416897748544[20] = 0;
   out_4956672416897748544[21] = 0;
   out_4956672416897748544[22] = 0;
   out_4956672416897748544[23] = 0;
   out_4956672416897748544[24] = 0;
   out_4956672416897748544[25] = 1;
   out_4956672416897748544[26] = 0;
   out_4956672416897748544[27] = 0;
   out_4956672416897748544[28] = 1;
   out_4956672416897748544[29] = 0;
   out_4956672416897748544[30] = 0;
   out_4956672416897748544[31] = 0;
   out_4956672416897748544[32] = 0;
   out_4956672416897748544[33] = 0;
   out_4956672416897748544[34] = 0;
   out_4956672416897748544[35] = 0;
   out_4956672416897748544[36] = 0;
   out_4956672416897748544[37] = 0;
   out_4956672416897748544[38] = 0;
   out_4956672416897748544[39] = 0;
   out_4956672416897748544[40] = 0;
   out_4956672416897748544[41] = 0;
   out_4956672416897748544[42] = 0;
   out_4956672416897748544[43] = 0;
   out_4956672416897748544[44] = 1;
   out_4956672416897748544[45] = 0;
   out_4956672416897748544[46] = 0;
   out_4956672416897748544[47] = 1;
   out_4956672416897748544[48] = 0;
   out_4956672416897748544[49] = 0;
   out_4956672416897748544[50] = 0;
   out_4956672416897748544[51] = 0;
   out_4956672416897748544[52] = 0;
   out_4956672416897748544[53] = 0;
}
void h_10(double *state, double *unused, double *out_2228325365749495789) {
   out_2228325365749495789[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2228325365749495789[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2228325365749495789[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_6859217750491902718) {
   out_6859217750491902718[0] = 0;
   out_6859217750491902718[1] = 9.8100000000000005*cos(state[1]);
   out_6859217750491902718[2] = 0;
   out_6859217750491902718[3] = 0;
   out_6859217750491902718[4] = -state[8];
   out_6859217750491902718[5] = state[7];
   out_6859217750491902718[6] = 0;
   out_6859217750491902718[7] = state[5];
   out_6859217750491902718[8] = -state[4];
   out_6859217750491902718[9] = 0;
   out_6859217750491902718[10] = 0;
   out_6859217750491902718[11] = 0;
   out_6859217750491902718[12] = 1;
   out_6859217750491902718[13] = 0;
   out_6859217750491902718[14] = 0;
   out_6859217750491902718[15] = 1;
   out_6859217750491902718[16] = 0;
   out_6859217750491902718[17] = 0;
   out_6859217750491902718[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_6859217750491902718[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_6859217750491902718[20] = 0;
   out_6859217750491902718[21] = state[8];
   out_6859217750491902718[22] = 0;
   out_6859217750491902718[23] = -state[6];
   out_6859217750491902718[24] = -state[5];
   out_6859217750491902718[25] = 0;
   out_6859217750491902718[26] = state[3];
   out_6859217750491902718[27] = 0;
   out_6859217750491902718[28] = 0;
   out_6859217750491902718[29] = 0;
   out_6859217750491902718[30] = 0;
   out_6859217750491902718[31] = 1;
   out_6859217750491902718[32] = 0;
   out_6859217750491902718[33] = 0;
   out_6859217750491902718[34] = 1;
   out_6859217750491902718[35] = 0;
   out_6859217750491902718[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_6859217750491902718[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_6859217750491902718[38] = 0;
   out_6859217750491902718[39] = -state[7];
   out_6859217750491902718[40] = state[6];
   out_6859217750491902718[41] = 0;
   out_6859217750491902718[42] = state[4];
   out_6859217750491902718[43] = -state[3];
   out_6859217750491902718[44] = 0;
   out_6859217750491902718[45] = 0;
   out_6859217750491902718[46] = 0;
   out_6859217750491902718[47] = 0;
   out_6859217750491902718[48] = 0;
   out_6859217750491902718[49] = 0;
   out_6859217750491902718[50] = 1;
   out_6859217750491902718[51] = 0;
   out_6859217750491902718[52] = 0;
   out_6859217750491902718[53] = 1;
}
void h_13(double *state, double *unused, double *out_4348549909198884608) {
   out_4348549909198884608[0] = state[3];
   out_4348549909198884608[1] = state[4];
   out_4348549909198884608[2] = state[5];
}
void H_13(double *state, double *unused, double *out_1744398591565415743) {
   out_1744398591565415743[0] = 0;
   out_1744398591565415743[1] = 0;
   out_1744398591565415743[2] = 0;
   out_1744398591565415743[3] = 1;
   out_1744398591565415743[4] = 0;
   out_1744398591565415743[5] = 0;
   out_1744398591565415743[6] = 0;
   out_1744398591565415743[7] = 0;
   out_1744398591565415743[8] = 0;
   out_1744398591565415743[9] = 0;
   out_1744398591565415743[10] = 0;
   out_1744398591565415743[11] = 0;
   out_1744398591565415743[12] = 0;
   out_1744398591565415743[13] = 0;
   out_1744398591565415743[14] = 0;
   out_1744398591565415743[15] = 0;
   out_1744398591565415743[16] = 0;
   out_1744398591565415743[17] = 0;
   out_1744398591565415743[18] = 0;
   out_1744398591565415743[19] = 0;
   out_1744398591565415743[20] = 0;
   out_1744398591565415743[21] = 0;
   out_1744398591565415743[22] = 1;
   out_1744398591565415743[23] = 0;
   out_1744398591565415743[24] = 0;
   out_1744398591565415743[25] = 0;
   out_1744398591565415743[26] = 0;
   out_1744398591565415743[27] = 0;
   out_1744398591565415743[28] = 0;
   out_1744398591565415743[29] = 0;
   out_1744398591565415743[30] = 0;
   out_1744398591565415743[31] = 0;
   out_1744398591565415743[32] = 0;
   out_1744398591565415743[33] = 0;
   out_1744398591565415743[34] = 0;
   out_1744398591565415743[35] = 0;
   out_1744398591565415743[36] = 0;
   out_1744398591565415743[37] = 0;
   out_1744398591565415743[38] = 0;
   out_1744398591565415743[39] = 0;
   out_1744398591565415743[40] = 0;
   out_1744398591565415743[41] = 1;
   out_1744398591565415743[42] = 0;
   out_1744398591565415743[43] = 0;
   out_1744398591565415743[44] = 0;
   out_1744398591565415743[45] = 0;
   out_1744398591565415743[46] = 0;
   out_1744398591565415743[47] = 0;
   out_1744398591565415743[48] = 0;
   out_1744398591565415743[49] = 0;
   out_1744398591565415743[50] = 0;
   out_1744398591565415743[51] = 0;
   out_1744398591565415743[52] = 0;
   out_1744398591565415743[53] = 0;
}
void h_14(double *state, double *unused, double *out_5034700456740961966) {
   out_5034700456740961966[0] = state[6];
   out_5034700456740961966[1] = state[7];
   out_5034700456740961966[2] = state[8];
}
void H_14(double *state, double *unused, double *out_8039460849193120840) {
   out_8039460849193120840[0] = 0;
   out_8039460849193120840[1] = 0;
   out_8039460849193120840[2] = 0;
   out_8039460849193120840[3] = 0;
   out_8039460849193120840[4] = 0;
   out_8039460849193120840[5] = 0;
   out_8039460849193120840[6] = 1;
   out_8039460849193120840[7] = 0;
   out_8039460849193120840[8] = 0;
   out_8039460849193120840[9] = 0;
   out_8039460849193120840[10] = 0;
   out_8039460849193120840[11] = 0;
   out_8039460849193120840[12] = 0;
   out_8039460849193120840[13] = 0;
   out_8039460849193120840[14] = 0;
   out_8039460849193120840[15] = 0;
   out_8039460849193120840[16] = 0;
   out_8039460849193120840[17] = 0;
   out_8039460849193120840[18] = 0;
   out_8039460849193120840[19] = 0;
   out_8039460849193120840[20] = 0;
   out_8039460849193120840[21] = 0;
   out_8039460849193120840[22] = 0;
   out_8039460849193120840[23] = 0;
   out_8039460849193120840[24] = 0;
   out_8039460849193120840[25] = 1;
   out_8039460849193120840[26] = 0;
   out_8039460849193120840[27] = 0;
   out_8039460849193120840[28] = 0;
   out_8039460849193120840[29] = 0;
   out_8039460849193120840[30] = 0;
   out_8039460849193120840[31] = 0;
   out_8039460849193120840[32] = 0;
   out_8039460849193120840[33] = 0;
   out_8039460849193120840[34] = 0;
   out_8039460849193120840[35] = 0;
   out_8039460849193120840[36] = 0;
   out_8039460849193120840[37] = 0;
   out_8039460849193120840[38] = 0;
   out_8039460849193120840[39] = 0;
   out_8039460849193120840[40] = 0;
   out_8039460849193120840[41] = 0;
   out_8039460849193120840[42] = 0;
   out_8039460849193120840[43] = 0;
   out_8039460849193120840[44] = 1;
   out_8039460849193120840[45] = 0;
   out_8039460849193120840[46] = 0;
   out_8039460849193120840[47] = 0;
   out_8039460849193120840[48] = 0;
   out_8039460849193120840[49] = 0;
   out_8039460849193120840[50] = 0;
   out_8039460849193120840[51] = 0;
   out_8039460849193120840[52] = 0;
   out_8039460849193120840[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_6544138591762530502) {
  err_fun(nom_x, delta_x, out_6544138591762530502);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5883871402494837793) {
  inv_err_fun(nom_x, true_x, out_5883871402494837793);
}
void pose_H_mod_fun(double *state, double *out_7964293451172914980) {
  H_mod_fun(state, out_7964293451172914980);
}
void pose_f_fun(double *state, double dt, double *out_7677670493198828015) {
  f_fun(state,  dt, out_7677670493198828015);
}
void pose_F_fun(double *state, double dt, double *out_2599667736124097937) {
  F_fun(state,  dt, out_2599667736124097937);
}
void pose_h_4(double *state, double *unused, double *out_6486377135734027290) {
  h_4(state, unused, out_6486377135734027290);
}
void pose_H_4(double *state, double *unused, double *out_4956672416897748544) {
  H_4(state, unused, out_4956672416897748544);
}
void pose_h_10(double *state, double *unused, double *out_2228325365749495789) {
  h_10(state, unused, out_2228325365749495789);
}
void pose_H_10(double *state, double *unused, double *out_6859217750491902718) {
  H_10(state, unused, out_6859217750491902718);
}
void pose_h_13(double *state, double *unused, double *out_4348549909198884608) {
  h_13(state, unused, out_4348549909198884608);
}
void pose_H_13(double *state, double *unused, double *out_1744398591565415743) {
  H_13(state, unused, out_1744398591565415743);
}
void pose_h_14(double *state, double *unused, double *out_5034700456740961966) {
  h_14(state, unused, out_5034700456740961966);
}
void pose_H_14(double *state, double *unused, double *out_8039460849193120840) {
  H_14(state, unused, out_8039460849193120840);
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
