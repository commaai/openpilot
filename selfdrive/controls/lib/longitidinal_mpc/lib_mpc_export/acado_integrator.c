/*
 *    This file was auto-generated using the ACADO Toolkit.
 *    
 *    While ACADO Toolkit is free software released under the terms of
 *    the GNU Lesser General Public License (LGPL), the generated code
 *    as such remains the property of the user who used ACADO Toolkit
 *    to generate this code. In particular, user dependent data of the code
 *    do not inherit the GNU LGPL license. On the other hand, parts of the
 *    generated code that are a direct copy of source code from the
 *    ACADO Toolkit or the software tools it is based on, remain, as derived
 *    work, automatically covered by the LGPL license.
 *    
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *    
 */


#include "acado_common.h"


void acado_rhs_forw(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 48;
const real_t* od = in + 49;
/* Vector of auxiliary variables; number of elements: 10. */
real_t* a = acadoWorkspace.rhs_aux;

/* Compute intermediate quantities: */
a[0] = (exp((((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[5])/(real_t)(2.0000000000000000e+00))));
a[1] = ((real_t)(1.0000000000000000e+00)/(real_t)(2.0000000000000000e+00));
a[2] = (exp((((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[5])/(real_t)(2.0000000000000000e+00))));
a[3] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[36])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[36]))*a[1])*a[2]);
a[4] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[37])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[37]))*a[1])*a[2]);
a[5] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[38])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[38]))*a[1])*a[2]);
a[6] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[39])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[39]))*a[1])*a[2]);
a[7] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[40])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[40]))*a[1])*a[2]);
a[8] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[41])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[41]))*a[1])*a[2]);
a[9] = (((((((real_t)(0.0000000000000000e+00)-od[0])*xd[47])*xd[5])+((((real_t)(0.0000000000000000e+00)-od[0])*xd[5])*xd[47]))*a[1])*a[2]);

/* Compute outputs: */
out[0] = xd[1];
out[1] = xd[2];
out[2] = u[0];
out[3] = xd[4];
out[4] = (od[1]*a[0]);
out[5] = (real_t)(1.0000000000000000e+00);
out[6] = xd[12];
out[7] = xd[13];
out[8] = xd[14];
out[9] = xd[15];
out[10] = xd[16];
out[11] = xd[17];
out[12] = xd[18];
out[13] = xd[19];
out[14] = xd[20];
out[15] = xd[21];
out[16] = xd[22];
out[17] = xd[23];
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (real_t)(0.0000000000000000e+00);
out[21] = (real_t)(0.0000000000000000e+00);
out[22] = (real_t)(0.0000000000000000e+00);
out[23] = (real_t)(0.0000000000000000e+00);
out[24] = xd[30];
out[25] = xd[31];
out[26] = xd[32];
out[27] = xd[33];
out[28] = xd[34];
out[29] = xd[35];
out[30] = (od[1]*a[3]);
out[31] = (od[1]*a[4]);
out[32] = (od[1]*a[5]);
out[33] = (od[1]*a[6]);
out[34] = (od[1]*a[7]);
out[35] = (od[1]*a[8]);
out[36] = (real_t)(0.0000000000000000e+00);
out[37] = (real_t)(0.0000000000000000e+00);
out[38] = (real_t)(0.0000000000000000e+00);
out[39] = (real_t)(0.0000000000000000e+00);
out[40] = (real_t)(0.0000000000000000e+00);
out[41] = (real_t)(0.0000000000000000e+00);
out[42] = xd[43];
out[43] = xd[44];
out[44] = (real_t)(1.0000000000000000e+00);
out[45] = xd[46];
out[46] = (od[1]*a[9]);
out[47] = (real_t)(0.0000000000000000e+00);
}

/* Fixed step size:0.2 */
int acado_integrate( real_t* const rk_eta, int resetIntegrator, int rk_index )
{
int error;

int run1;
int numSteps[20] = {1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
int numInts = numSteps[rk_index];
acadoWorkspace.rk_ttt = 0.0000000000000000e+00;
rk_eta[6] = 1.0000000000000000e+00;
rk_eta[7] = 0.0000000000000000e+00;
rk_eta[8] = 0.0000000000000000e+00;
rk_eta[9] = 0.0000000000000000e+00;
rk_eta[10] = 0.0000000000000000e+00;
rk_eta[11] = 0.0000000000000000e+00;
rk_eta[12] = 0.0000000000000000e+00;
rk_eta[13] = 1.0000000000000000e+00;
rk_eta[14] = 0.0000000000000000e+00;
rk_eta[15] = 0.0000000000000000e+00;
rk_eta[16] = 0.0000000000000000e+00;
rk_eta[17] = 0.0000000000000000e+00;
rk_eta[18] = 0.0000000000000000e+00;
rk_eta[19] = 0.0000000000000000e+00;
rk_eta[20] = 1.0000000000000000e+00;
rk_eta[21] = 0.0000000000000000e+00;
rk_eta[22] = 0.0000000000000000e+00;
rk_eta[23] = 0.0000000000000000e+00;
rk_eta[24] = 0.0000000000000000e+00;
rk_eta[25] = 0.0000000000000000e+00;
rk_eta[26] = 0.0000000000000000e+00;
rk_eta[27] = 1.0000000000000000e+00;
rk_eta[28] = 0.0000000000000000e+00;
rk_eta[29] = 0.0000000000000000e+00;
rk_eta[30] = 0.0000000000000000e+00;
rk_eta[31] = 0.0000000000000000e+00;
rk_eta[32] = 0.0000000000000000e+00;
rk_eta[33] = 0.0000000000000000e+00;
rk_eta[34] = 1.0000000000000000e+00;
rk_eta[35] = 0.0000000000000000e+00;
rk_eta[36] = 0.0000000000000000e+00;
rk_eta[37] = 0.0000000000000000e+00;
rk_eta[38] = 0.0000000000000000e+00;
rk_eta[39] = 0.0000000000000000e+00;
rk_eta[40] = 0.0000000000000000e+00;
rk_eta[41] = 1.0000000000000000e+00;
rk_eta[42] = 0.0000000000000000e+00;
rk_eta[43] = 0.0000000000000000e+00;
rk_eta[44] = 0.0000000000000000e+00;
rk_eta[45] = 0.0000000000000000e+00;
rk_eta[46] = 0.0000000000000000e+00;
rk_eta[47] = 0.0000000000000000e+00;
acadoWorkspace.rk_xxx[48] = rk_eta[48];
acadoWorkspace.rk_xxx[49] = rk_eta[49];
acadoWorkspace.rk_xxx[50] = rk_eta[50];

for (run1 = 0; run1 < 1; ++run1)
{
for(run1 = 0; run1 < numInts; run1++ ) {
acadoWorkspace.rk_xxx[0] = + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + rk_eta[23];
acadoWorkspace.rk_xxx[24] = + rk_eta[24];
acadoWorkspace.rk_xxx[25] = + rk_eta[25];
acadoWorkspace.rk_xxx[26] = + rk_eta[26];
acadoWorkspace.rk_xxx[27] = + rk_eta[27];
acadoWorkspace.rk_xxx[28] = + rk_eta[28];
acadoWorkspace.rk_xxx[29] = + rk_eta[29];
acadoWorkspace.rk_xxx[30] = + rk_eta[30];
acadoWorkspace.rk_xxx[31] = + rk_eta[31];
acadoWorkspace.rk_xxx[32] = + rk_eta[32];
acadoWorkspace.rk_xxx[33] = + rk_eta[33];
acadoWorkspace.rk_xxx[34] = + rk_eta[34];
acadoWorkspace.rk_xxx[35] = + rk_eta[35];
acadoWorkspace.rk_xxx[36] = + rk_eta[36];
acadoWorkspace.rk_xxx[37] = + rk_eta[37];
acadoWorkspace.rk_xxx[38] = + rk_eta[38];
acadoWorkspace.rk_xxx[39] = + rk_eta[39];
acadoWorkspace.rk_xxx[40] = + rk_eta[40];
acadoWorkspace.rk_xxx[41] = + rk_eta[41];
acadoWorkspace.rk_xxx[42] = + rk_eta[42];
acadoWorkspace.rk_xxx[43] = + rk_eta[43];
acadoWorkspace.rk_xxx[44] = + rk_eta[44];
acadoWorkspace.rk_xxx[45] = + rk_eta[45];
acadoWorkspace.rk_xxx[46] = + rk_eta[46];
acadoWorkspace.rk_xxx[47] = + rk_eta[47];
acado_rhs_forw( acadoWorkspace.rk_xxx, acadoWorkspace.rk_kkk );
acadoWorkspace.rk_xxx[0] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[0] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[1] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[2] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[3] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[4] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[5] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[6] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[7] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[8] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[9] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[10] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[11] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[12] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[13] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[14] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[15] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[16] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[17] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[18] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[19] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[20] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[21] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[22] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[23] + rk_eta[23];
acadoWorkspace.rk_xxx[24] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[24] + rk_eta[24];
acadoWorkspace.rk_xxx[25] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[25] + rk_eta[25];
acadoWorkspace.rk_xxx[26] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[26] + rk_eta[26];
acadoWorkspace.rk_xxx[27] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[27] + rk_eta[27];
acadoWorkspace.rk_xxx[28] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[28] + rk_eta[28];
acadoWorkspace.rk_xxx[29] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[29] + rk_eta[29];
acadoWorkspace.rk_xxx[30] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[30] + rk_eta[30];
acadoWorkspace.rk_xxx[31] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[31] + rk_eta[31];
acadoWorkspace.rk_xxx[32] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[32] + rk_eta[32];
acadoWorkspace.rk_xxx[33] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[33] + rk_eta[33];
acadoWorkspace.rk_xxx[34] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[34] + rk_eta[34];
acadoWorkspace.rk_xxx[35] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[35] + rk_eta[35];
acadoWorkspace.rk_xxx[36] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[36] + rk_eta[36];
acadoWorkspace.rk_xxx[37] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[37] + rk_eta[37];
acadoWorkspace.rk_xxx[38] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[38] + rk_eta[38];
acadoWorkspace.rk_xxx[39] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[39] + rk_eta[39];
acadoWorkspace.rk_xxx[40] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[40] + rk_eta[40];
acadoWorkspace.rk_xxx[41] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[41] + rk_eta[41];
acadoWorkspace.rk_xxx[42] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[42] + rk_eta[42];
acadoWorkspace.rk_xxx[43] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[43] + rk_eta[43];
acadoWorkspace.rk_xxx[44] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[44] + rk_eta[44];
acadoWorkspace.rk_xxx[45] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[45] + rk_eta[45];
acadoWorkspace.rk_xxx[46] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[46] + rk_eta[46];
acadoWorkspace.rk_xxx[47] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[47] + rk_eta[47];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 48 ]) );
acadoWorkspace.rk_xxx[0] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[48] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[49] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[50] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[51] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[52] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[53] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[54] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[55] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[56] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[57] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[58] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[59] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[60] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[61] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[62] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[63] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[64] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[65] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[66] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[67] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[68] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[69] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[70] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[71] + rk_eta[23];
acadoWorkspace.rk_xxx[24] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[72] + rk_eta[24];
acadoWorkspace.rk_xxx[25] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[73] + rk_eta[25];
acadoWorkspace.rk_xxx[26] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[74] + rk_eta[26];
acadoWorkspace.rk_xxx[27] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[75] + rk_eta[27];
acadoWorkspace.rk_xxx[28] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[76] + rk_eta[28];
acadoWorkspace.rk_xxx[29] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[77] + rk_eta[29];
acadoWorkspace.rk_xxx[30] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[78] + rk_eta[30];
acadoWorkspace.rk_xxx[31] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[79] + rk_eta[31];
acadoWorkspace.rk_xxx[32] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[80] + rk_eta[32];
acadoWorkspace.rk_xxx[33] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[81] + rk_eta[33];
acadoWorkspace.rk_xxx[34] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[82] + rk_eta[34];
acadoWorkspace.rk_xxx[35] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[83] + rk_eta[35];
acadoWorkspace.rk_xxx[36] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[84] + rk_eta[36];
acadoWorkspace.rk_xxx[37] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[85] + rk_eta[37];
acadoWorkspace.rk_xxx[38] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[86] + rk_eta[38];
acadoWorkspace.rk_xxx[39] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[87] + rk_eta[39];
acadoWorkspace.rk_xxx[40] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[88] + rk_eta[40];
acadoWorkspace.rk_xxx[41] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[89] + rk_eta[41];
acadoWorkspace.rk_xxx[42] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[90] + rk_eta[42];
acadoWorkspace.rk_xxx[43] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[91] + rk_eta[43];
acadoWorkspace.rk_xxx[44] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[92] + rk_eta[44];
acadoWorkspace.rk_xxx[45] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[93] + rk_eta[45];
acadoWorkspace.rk_xxx[46] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[94] + rk_eta[46];
acadoWorkspace.rk_xxx[47] = + (real_t)9.9999999999999964e-02*acadoWorkspace.rk_kkk[95] + rk_eta[47];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 96 ]) );
acadoWorkspace.rk_xxx[0] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[96] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[97] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[98] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[99] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[100] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[101] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[102] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[103] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[104] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[105] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[106] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[107] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[108] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[109] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[110] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[111] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[112] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[113] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[114] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[115] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[116] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[117] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[118] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[119] + rk_eta[23];
acadoWorkspace.rk_xxx[24] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[120] + rk_eta[24];
acadoWorkspace.rk_xxx[25] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[121] + rk_eta[25];
acadoWorkspace.rk_xxx[26] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[122] + rk_eta[26];
acadoWorkspace.rk_xxx[27] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[123] + rk_eta[27];
acadoWorkspace.rk_xxx[28] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[124] + rk_eta[28];
acadoWorkspace.rk_xxx[29] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[125] + rk_eta[29];
acadoWorkspace.rk_xxx[30] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[126] + rk_eta[30];
acadoWorkspace.rk_xxx[31] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[127] + rk_eta[31];
acadoWorkspace.rk_xxx[32] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[128] + rk_eta[32];
acadoWorkspace.rk_xxx[33] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[129] + rk_eta[33];
acadoWorkspace.rk_xxx[34] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[130] + rk_eta[34];
acadoWorkspace.rk_xxx[35] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[131] + rk_eta[35];
acadoWorkspace.rk_xxx[36] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[132] + rk_eta[36];
acadoWorkspace.rk_xxx[37] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[133] + rk_eta[37];
acadoWorkspace.rk_xxx[38] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[134] + rk_eta[38];
acadoWorkspace.rk_xxx[39] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[135] + rk_eta[39];
acadoWorkspace.rk_xxx[40] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[136] + rk_eta[40];
acadoWorkspace.rk_xxx[41] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[137] + rk_eta[41];
acadoWorkspace.rk_xxx[42] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[138] + rk_eta[42];
acadoWorkspace.rk_xxx[43] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[139] + rk_eta[43];
acadoWorkspace.rk_xxx[44] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[140] + rk_eta[44];
acadoWorkspace.rk_xxx[45] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[141] + rk_eta[45];
acadoWorkspace.rk_xxx[46] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[142] + rk_eta[46];
acadoWorkspace.rk_xxx[47] = + (real_t)1.9999999999999993e-01*acadoWorkspace.rk_kkk[143] + rk_eta[47];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 144 ]) );
rk_eta[0] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[0] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[48] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[96] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[144];
rk_eta[1] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[1] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[49] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[97] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[145];
rk_eta[2] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[2] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[50] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[98] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[146];
rk_eta[3] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[3] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[51] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[99] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[147];
rk_eta[4] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[4] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[52] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[100] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[148];
rk_eta[5] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[5] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[53] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[101] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[149];
rk_eta[6] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[6] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[54] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[102] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[150];
rk_eta[7] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[7] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[55] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[103] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[151];
rk_eta[8] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[8] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[56] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[104] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[152];
rk_eta[9] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[9] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[57] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[105] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[153];
rk_eta[10] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[10] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[58] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[106] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[154];
rk_eta[11] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[11] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[59] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[107] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[155];
rk_eta[12] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[12] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[60] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[108] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[156];
rk_eta[13] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[13] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[61] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[109] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[157];
rk_eta[14] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[14] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[62] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[110] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[158];
rk_eta[15] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[15] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[63] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[111] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[159];
rk_eta[16] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[16] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[64] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[112] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[160];
rk_eta[17] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[17] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[65] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[113] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[161];
rk_eta[18] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[18] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[66] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[114] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[162];
rk_eta[19] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[19] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[67] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[115] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[163];
rk_eta[20] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[20] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[68] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[116] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[164];
rk_eta[21] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[21] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[69] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[117] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[165];
rk_eta[22] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[22] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[70] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[118] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[166];
rk_eta[23] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[23] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[71] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[119] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[167];
rk_eta[24] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[24] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[72] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[120] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[168];
rk_eta[25] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[25] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[73] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[121] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[169];
rk_eta[26] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[26] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[74] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[122] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[170];
rk_eta[27] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[27] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[75] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[123] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[171];
rk_eta[28] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[28] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[76] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[124] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[172];
rk_eta[29] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[29] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[77] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[125] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[173];
rk_eta[30] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[30] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[78] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[126] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[174];
rk_eta[31] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[31] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[79] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[127] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[175];
rk_eta[32] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[32] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[80] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[128] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[176];
rk_eta[33] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[33] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[81] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[129] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[177];
rk_eta[34] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[34] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[82] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[130] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[178];
rk_eta[35] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[35] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[83] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[131] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[179];
rk_eta[36] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[36] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[84] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[132] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[180];
rk_eta[37] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[37] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[85] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[133] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[181];
rk_eta[38] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[38] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[86] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[134] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[182];
rk_eta[39] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[39] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[87] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[135] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[183];
rk_eta[40] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[40] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[88] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[136] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[184];
rk_eta[41] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[41] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[89] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[137] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[185];
rk_eta[42] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[42] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[90] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[138] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[186];
rk_eta[43] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[43] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[91] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[139] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[187];
rk_eta[44] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[44] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[92] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[140] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[188];
rk_eta[45] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[45] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[93] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[141] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[189];
rk_eta[46] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[46] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[94] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[142] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[190];
rk_eta[47] += + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[47] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[95] + (real_t)6.6666666666666638e-02*acadoWorkspace.rk_kkk[143] + (real_t)3.3333333333333319e-02*acadoWorkspace.rk_kkk[191];
acadoWorkspace.rk_ttt += 1.0000000000000000e+00;
}
}
error = 0;
return error;
}

