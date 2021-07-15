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
const real_t* u = in + 24;
const real_t* od = in + 25;
/* Vector of auxiliary variables; number of elements: 14. */
real_t* a = acadoWorkspace.rhs_aux;

/* Compute intermediate quantities: */
a[0] = (cos(xd[2]));
a[1] = (sin(xd[2]));
a[2] = ((real_t)(-1.0000000000000000e+00)*(sin(xd[2])));
a[3] = (xd[12]*a[2]);
a[4] = (xd[13]*a[2]);
a[5] = (xd[14]*a[2]);
a[6] = (xd[15]*a[2]);
a[7] = (cos(xd[2]));
a[8] = (xd[12]*a[7]);
a[9] = (xd[13]*a[7]);
a[10] = (xd[14]*a[7]);
a[11] = (xd[15]*a[7]);
a[12] = (xd[22]*a[2]);
a[13] = (xd[22]*a[7]);

/* Compute outputs: */
out[0] = (od[0]*a[0]);
out[1] = (od[0]*a[1]);
out[2] = (od[0]*xd[3]);
out[3] = u[0];
out[4] = (od[0]*a[3]);
out[5] = (od[0]*a[4]);
out[6] = (od[0]*a[5]);
out[7] = (od[0]*a[6]);
out[8] = (od[0]*a[8]);
out[9] = (od[0]*a[9]);
out[10] = (od[0]*a[10]);
out[11] = (od[0]*a[11]);
out[12] = (od[0]*xd[16]);
out[13] = (od[0]*xd[17]);
out[14] = (od[0]*xd[18]);
out[15] = (od[0]*xd[19]);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = (real_t)(0.0000000000000000e+00);
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (od[0]*a[12]);
out[21] = (od[0]*a[13]);
out[22] = (od[0]*xd[23]);
out[23] = (real_t)(1.0000000000000000e+00);
}

/* Fixed step size:0.0025 */
int acado_integrate( real_t* const rk_eta, int resetIntegrator, int rk_index )
{
int error;

int run1;
int numSteps[16] = {4, 12, 20, 27, 35, 43, 51, 59, 66, 74, 82, 90, 98, 105, 113, 121};
int numInts = numSteps[rk_index];
acadoWorkspace.rk_ttt = 0.0000000000000000e+00;
rk_eta[4] = 1.0000000000000000e+00;
rk_eta[5] = 0.0000000000000000e+00;
rk_eta[6] = 0.0000000000000000e+00;
rk_eta[7] = 0.0000000000000000e+00;
rk_eta[8] = 0.0000000000000000e+00;
rk_eta[9] = 1.0000000000000000e+00;
rk_eta[10] = 0.0000000000000000e+00;
rk_eta[11] = 0.0000000000000000e+00;
rk_eta[12] = 0.0000000000000000e+00;
rk_eta[13] = 0.0000000000000000e+00;
rk_eta[14] = 1.0000000000000000e+00;
rk_eta[15] = 0.0000000000000000e+00;
rk_eta[16] = 0.0000000000000000e+00;
rk_eta[17] = 0.0000000000000000e+00;
rk_eta[18] = 0.0000000000000000e+00;
rk_eta[19] = 1.0000000000000000e+00;
rk_eta[20] = 0.0000000000000000e+00;
rk_eta[21] = 0.0000000000000000e+00;
rk_eta[22] = 0.0000000000000000e+00;
rk_eta[23] = 0.0000000000000000e+00;
acadoWorkspace.rk_xxx[24] = rk_eta[24];
acadoWorkspace.rk_xxx[25] = rk_eta[25];
acadoWorkspace.rk_xxx[26] = rk_eta[26];

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
acado_rhs_forw( acadoWorkspace.rk_xxx, acadoWorkspace.rk_kkk );
acadoWorkspace.rk_xxx[0] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[0] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[1] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[2] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[3] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[4] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[5] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[6] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[7] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[8] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[9] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[10] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[11] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[12] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[13] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[14] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[15] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[16] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[17] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[18] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[19] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[20] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[21] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[22] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[23] + rk_eta[23];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 24 ]) );
acadoWorkspace.rk_xxx[0] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[24] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[25] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[26] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[27] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[28] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[29] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[30] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[31] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[32] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[33] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[34] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[35] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[36] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[37] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[38] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[39] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[40] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[41] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[42] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[43] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[44] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[45] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[46] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)1.2500000000000000e-03*acadoWorkspace.rk_kkk[47] + rk_eta[23];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 48 ]) );
acadoWorkspace.rk_xxx[0] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[48] + rk_eta[0];
acadoWorkspace.rk_xxx[1] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[49] + rk_eta[1];
acadoWorkspace.rk_xxx[2] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[50] + rk_eta[2];
acadoWorkspace.rk_xxx[3] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[51] + rk_eta[3];
acadoWorkspace.rk_xxx[4] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[52] + rk_eta[4];
acadoWorkspace.rk_xxx[5] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[53] + rk_eta[5];
acadoWorkspace.rk_xxx[6] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[54] + rk_eta[6];
acadoWorkspace.rk_xxx[7] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[55] + rk_eta[7];
acadoWorkspace.rk_xxx[8] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[56] + rk_eta[8];
acadoWorkspace.rk_xxx[9] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[57] + rk_eta[9];
acadoWorkspace.rk_xxx[10] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[58] + rk_eta[10];
acadoWorkspace.rk_xxx[11] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[59] + rk_eta[11];
acadoWorkspace.rk_xxx[12] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[60] + rk_eta[12];
acadoWorkspace.rk_xxx[13] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[61] + rk_eta[13];
acadoWorkspace.rk_xxx[14] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[62] + rk_eta[14];
acadoWorkspace.rk_xxx[15] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[63] + rk_eta[15];
acadoWorkspace.rk_xxx[16] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[64] + rk_eta[16];
acadoWorkspace.rk_xxx[17] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[65] + rk_eta[17];
acadoWorkspace.rk_xxx[18] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[66] + rk_eta[18];
acadoWorkspace.rk_xxx[19] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[67] + rk_eta[19];
acadoWorkspace.rk_xxx[20] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[68] + rk_eta[20];
acadoWorkspace.rk_xxx[21] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[69] + rk_eta[21];
acadoWorkspace.rk_xxx[22] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[70] + rk_eta[22];
acadoWorkspace.rk_xxx[23] = + (real_t)2.5000000000000001e-03*acadoWorkspace.rk_kkk[71] + rk_eta[23];
acado_rhs_forw( acadoWorkspace.rk_xxx, &(acadoWorkspace.rk_kkk[ 72 ]) );
rk_eta[0] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[0] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[24] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[48] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[72];
rk_eta[1] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[1] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[25] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[49] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[73];
rk_eta[2] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[2] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[26] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[50] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[74];
rk_eta[3] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[3] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[27] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[51] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[75];
rk_eta[4] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[4] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[28] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[52] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[76];
rk_eta[5] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[5] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[29] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[53] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[77];
rk_eta[6] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[6] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[30] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[54] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[78];
rk_eta[7] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[7] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[31] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[55] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[79];
rk_eta[8] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[8] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[32] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[56] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[80];
rk_eta[9] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[9] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[33] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[57] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[81];
rk_eta[10] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[10] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[34] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[58] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[82];
rk_eta[11] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[11] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[35] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[59] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[83];
rk_eta[12] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[12] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[36] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[60] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[84];
rk_eta[13] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[13] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[37] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[61] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[85];
rk_eta[14] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[14] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[38] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[62] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[86];
rk_eta[15] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[15] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[39] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[63] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[87];
rk_eta[16] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[16] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[40] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[64] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[88];
rk_eta[17] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[17] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[41] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[65] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[89];
rk_eta[18] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[18] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[42] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[66] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[90];
rk_eta[19] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[19] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[43] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[67] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[91];
rk_eta[20] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[20] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[44] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[68] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[92];
rk_eta[21] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[21] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[45] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[69] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[93];
rk_eta[22] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[22] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[46] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[70] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[94];
rk_eta[23] += + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[23] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[47] + (real_t)8.3333333333333328e-04*acadoWorkspace.rk_kkk[71] + (real_t)4.1666666666666664e-04*acadoWorkspace.rk_kkk[95];
acadoWorkspace.rk_ttt += 1.0000000000000000e+00;
}
}
error = 0;
return error;
}

