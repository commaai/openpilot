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




/******************************************************************************/
/*                                                                            */
/* ACADO code generation                                                      */
/*                                                                            */
/******************************************************************************/


int acado_modelSimulation(  )
{
int ret;

int lRun1;
ret = 0;
for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 5];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 5 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 5 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 5 + 3];
acadoWorkspace.state[4] = acadoVariables.x[lRun1 * 5 + 4];

acadoWorkspace.state[45] = acadoVariables.u[lRun1 * 3];
acadoWorkspace.state[46] = acadoVariables.u[lRun1 * 3 + 1];
acadoWorkspace.state[47] = acadoVariables.u[lRun1 * 3 + 2];
acadoWorkspace.state[48] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.state[49] = acadoVariables.od[lRun1 * 2 + 1];

ret = acado_integrate(acadoWorkspace.state, 1, lRun1);

acadoWorkspace.d[lRun1 * 5] = acadoWorkspace.state[0] - acadoVariables.x[lRun1 * 5 + 5];
acadoWorkspace.d[lRun1 * 5 + 1] = acadoWorkspace.state[1] - acadoVariables.x[lRun1 * 5 + 6];
acadoWorkspace.d[lRun1 * 5 + 2] = acadoWorkspace.state[2] - acadoVariables.x[lRun1 * 5 + 7];
acadoWorkspace.d[lRun1 * 5 + 3] = acadoWorkspace.state[3] - acadoVariables.x[lRun1 * 5 + 8];
acadoWorkspace.d[lRun1 * 5 + 4] = acadoWorkspace.state[4] - acadoVariables.x[lRun1 * 5 + 9];

acadoWorkspace.evGx[lRun1 * 25] = acadoWorkspace.state[5];
acadoWorkspace.evGx[lRun1 * 25 + 1] = acadoWorkspace.state[6];
acadoWorkspace.evGx[lRun1 * 25 + 2] = acadoWorkspace.state[7];
acadoWorkspace.evGx[lRun1 * 25 + 3] = acadoWorkspace.state[8];
acadoWorkspace.evGx[lRun1 * 25 + 4] = acadoWorkspace.state[9];
acadoWorkspace.evGx[lRun1 * 25 + 5] = acadoWorkspace.state[10];
acadoWorkspace.evGx[lRun1 * 25 + 6] = acadoWorkspace.state[11];
acadoWorkspace.evGx[lRun1 * 25 + 7] = acadoWorkspace.state[12];
acadoWorkspace.evGx[lRun1 * 25 + 8] = acadoWorkspace.state[13];
acadoWorkspace.evGx[lRun1 * 25 + 9] = acadoWorkspace.state[14];
acadoWorkspace.evGx[lRun1 * 25 + 10] = acadoWorkspace.state[15];
acadoWorkspace.evGx[lRun1 * 25 + 11] = acadoWorkspace.state[16];
acadoWorkspace.evGx[lRun1 * 25 + 12] = acadoWorkspace.state[17];
acadoWorkspace.evGx[lRun1 * 25 + 13] = acadoWorkspace.state[18];
acadoWorkspace.evGx[lRun1 * 25 + 14] = acadoWorkspace.state[19];
acadoWorkspace.evGx[lRun1 * 25 + 15] = acadoWorkspace.state[20];
acadoWorkspace.evGx[lRun1 * 25 + 16] = acadoWorkspace.state[21];
acadoWorkspace.evGx[lRun1 * 25 + 17] = acadoWorkspace.state[22];
acadoWorkspace.evGx[lRun1 * 25 + 18] = acadoWorkspace.state[23];
acadoWorkspace.evGx[lRun1 * 25 + 19] = acadoWorkspace.state[24];
acadoWorkspace.evGx[lRun1 * 25 + 20] = acadoWorkspace.state[25];
acadoWorkspace.evGx[lRun1 * 25 + 21] = acadoWorkspace.state[26];
acadoWorkspace.evGx[lRun1 * 25 + 22] = acadoWorkspace.state[27];
acadoWorkspace.evGx[lRun1 * 25 + 23] = acadoWorkspace.state[28];
acadoWorkspace.evGx[lRun1 * 25 + 24] = acadoWorkspace.state[29];

acadoWorkspace.evGu[lRun1 * 15] = acadoWorkspace.state[30];
acadoWorkspace.evGu[lRun1 * 15 + 1] = acadoWorkspace.state[31];
acadoWorkspace.evGu[lRun1 * 15 + 2] = acadoWorkspace.state[32];
acadoWorkspace.evGu[lRun1 * 15 + 3] = acadoWorkspace.state[33];
acadoWorkspace.evGu[lRun1 * 15 + 4] = acadoWorkspace.state[34];
acadoWorkspace.evGu[lRun1 * 15 + 5] = acadoWorkspace.state[35];
acadoWorkspace.evGu[lRun1 * 15 + 6] = acadoWorkspace.state[36];
acadoWorkspace.evGu[lRun1 * 15 + 7] = acadoWorkspace.state[37];
acadoWorkspace.evGu[lRun1 * 15 + 8] = acadoWorkspace.state[38];
acadoWorkspace.evGu[lRun1 * 15 + 9] = acadoWorkspace.state[39];
acadoWorkspace.evGu[lRun1 * 15 + 10] = acadoWorkspace.state[40];
acadoWorkspace.evGu[lRun1 * 15 + 11] = acadoWorkspace.state[41];
acadoWorkspace.evGu[lRun1 * 15 + 12] = acadoWorkspace.state[42];
acadoWorkspace.evGu[lRun1 * 15 + 13] = acadoWorkspace.state[43];
acadoWorkspace.evGu[lRun1 * 15 + 14] = acadoWorkspace.state[44];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 5;
const real_t* od = in + 8;
/* Vector of auxiliary variables; number of elements: 6. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = ((real_t)(1.0000000000000000e+00)/(xd[1]+(real_t)(1.0000000000000000e+00)));
a[1] = (a[0]*a[0]);
a[2] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[3] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[4] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[5] = ((real_t)(1.0000000000000000e+00)/(real_t)(2.0000000000000000e+00));

/* Compute outputs: */
out[0] = (u[0]/(xd[1]+(real_t)(1.0000000000000000e+00)));
out[1] = u[1];
out[2] = u[2];
out[3] = u[2];
out[4] = (((((od[0]-xd[0])-(real_t)(4.0000000000000000e+00))+((od[1]*(od[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(od[1]/(real_t)(7.0000000000000000e+00)))*(od[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((xd[1]*(xd[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)))*(xd[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((real_t)(1.8000000000000000e+00)*xd[1]));
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = ((real_t)(0.0000000000000000e+00)-(u[0]*a[1]));
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = (real_t)(0.0000000000000000e+00);
out[14] = (real_t)(0.0000000000000000e+00);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = (real_t)(0.0000000000000000e+00);
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (real_t)(0.0000000000000000e+00);
out[21] = (real_t)(0.0000000000000000e+00);
out[22] = (real_t)(0.0000000000000000e+00);
out[23] = (real_t)(0.0000000000000000e+00);
out[24] = (real_t)(0.0000000000000000e+00);
out[25] = ((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00));
out[26] = (((real_t)(0.0000000000000000e+00)-(((xd[1]/(real_t)(7.0000000000000000e+00))+(xd[1]*a[2]))-(((((real_t)(7.0000000000000000e+00)*a[3])*(xd[1]/(real_t)(7.0000000000000000e+00)))+(((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)))*a[4]))*a[5])))-(real_t)(1.8000000000000000e+00));
out[27] = (real_t)(0.0000000000000000e+00);
out[28] = (real_t)(0.0000000000000000e+00);
out[29] = (real_t)(0.0000000000000000e+00);
out[30] = a[0];
out[31] = (real_t)(0.0000000000000000e+00);
out[32] = (real_t)(0.0000000000000000e+00);
out[33] = (real_t)(0.0000000000000000e+00);
out[34] = (real_t)(1.0000000000000000e+00);
out[35] = (real_t)(0.0000000000000000e+00);
out[36] = (real_t)(0.0000000000000000e+00);
out[37] = (real_t)(0.0000000000000000e+00);
out[38] = (real_t)(1.0000000000000000e+00);
out[39] = (real_t)(0.0000000000000000e+00);
out[40] = (real_t)(0.0000000000000000e+00);
out[41] = (real_t)(1.0000000000000000e+00);
out[42] = (real_t)(0.0000000000000000e+00);
out[43] = (real_t)(0.0000000000000000e+00);
out[44] = (real_t)(0.0000000000000000e+00);
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 5;
/* Vector of auxiliary variables; number of elements: 4. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[1] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[2] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[3] = ((real_t)(1.0000000000000000e+00)/(real_t)(2.0000000000000000e+00));

/* Compute outputs: */
out[0] = (((((od[0]-xd[0])-(real_t)(4.0000000000000000e+00))+((od[1]*(od[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(od[1]/(real_t)(7.0000000000000000e+00)))*(od[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((xd[1]*(xd[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)))*(xd[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((real_t)(1.8000000000000000e+00)*xd[1]));
out[1] = ((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00));
out[2] = (((real_t)(0.0000000000000000e+00)-(((xd[1]/(real_t)(7.0000000000000000e+00))+(xd[1]*a[0]))-(((((real_t)(7.0000000000000000e+00)*a[1])*(xd[1]/(real_t)(7.0000000000000000e+00)))+(((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)))*a[2]))*a[3])))-(real_t)(1.8000000000000000e+00));
out[3] = (real_t)(0.0000000000000000e+00);
out[4] = (real_t)(0.0000000000000000e+00);
out[5] = (real_t)(0.0000000000000000e+00);
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0]*tmpObjS[0] + tmpFx[5]*tmpObjS[5] + tmpFx[10]*tmpObjS[10] + tmpFx[15]*tmpObjS[15] + tmpFx[20]*tmpObjS[20];
tmpQ2[1] = + tmpFx[0]*tmpObjS[1] + tmpFx[5]*tmpObjS[6] + tmpFx[10]*tmpObjS[11] + tmpFx[15]*tmpObjS[16] + tmpFx[20]*tmpObjS[21];
tmpQ2[2] = + tmpFx[0]*tmpObjS[2] + tmpFx[5]*tmpObjS[7] + tmpFx[10]*tmpObjS[12] + tmpFx[15]*tmpObjS[17] + tmpFx[20]*tmpObjS[22];
tmpQ2[3] = + tmpFx[0]*tmpObjS[3] + tmpFx[5]*tmpObjS[8] + tmpFx[10]*tmpObjS[13] + tmpFx[15]*tmpObjS[18] + tmpFx[20]*tmpObjS[23];
tmpQ2[4] = + tmpFx[0]*tmpObjS[4] + tmpFx[5]*tmpObjS[9] + tmpFx[10]*tmpObjS[14] + tmpFx[15]*tmpObjS[19] + tmpFx[20]*tmpObjS[24];
tmpQ2[5] = + tmpFx[1]*tmpObjS[0] + tmpFx[6]*tmpObjS[5] + tmpFx[11]*tmpObjS[10] + tmpFx[16]*tmpObjS[15] + tmpFx[21]*tmpObjS[20];
tmpQ2[6] = + tmpFx[1]*tmpObjS[1] + tmpFx[6]*tmpObjS[6] + tmpFx[11]*tmpObjS[11] + tmpFx[16]*tmpObjS[16] + tmpFx[21]*tmpObjS[21];
tmpQ2[7] = + tmpFx[1]*tmpObjS[2] + tmpFx[6]*tmpObjS[7] + tmpFx[11]*tmpObjS[12] + tmpFx[16]*tmpObjS[17] + tmpFx[21]*tmpObjS[22];
tmpQ2[8] = + tmpFx[1]*tmpObjS[3] + tmpFx[6]*tmpObjS[8] + tmpFx[11]*tmpObjS[13] + tmpFx[16]*tmpObjS[18] + tmpFx[21]*tmpObjS[23];
tmpQ2[9] = + tmpFx[1]*tmpObjS[4] + tmpFx[6]*tmpObjS[9] + tmpFx[11]*tmpObjS[14] + tmpFx[16]*tmpObjS[19] + tmpFx[21]*tmpObjS[24];
tmpQ2[10] = + tmpFx[2]*tmpObjS[0] + tmpFx[7]*tmpObjS[5] + tmpFx[12]*tmpObjS[10] + tmpFx[17]*tmpObjS[15] + tmpFx[22]*tmpObjS[20];
tmpQ2[11] = + tmpFx[2]*tmpObjS[1] + tmpFx[7]*tmpObjS[6] + tmpFx[12]*tmpObjS[11] + tmpFx[17]*tmpObjS[16] + tmpFx[22]*tmpObjS[21];
tmpQ2[12] = + tmpFx[2]*tmpObjS[2] + tmpFx[7]*tmpObjS[7] + tmpFx[12]*tmpObjS[12] + tmpFx[17]*tmpObjS[17] + tmpFx[22]*tmpObjS[22];
tmpQ2[13] = + tmpFx[2]*tmpObjS[3] + tmpFx[7]*tmpObjS[8] + tmpFx[12]*tmpObjS[13] + tmpFx[17]*tmpObjS[18] + tmpFx[22]*tmpObjS[23];
tmpQ2[14] = + tmpFx[2]*tmpObjS[4] + tmpFx[7]*tmpObjS[9] + tmpFx[12]*tmpObjS[14] + tmpFx[17]*tmpObjS[19] + tmpFx[22]*tmpObjS[24];
tmpQ2[15] = + tmpFx[3]*tmpObjS[0] + tmpFx[8]*tmpObjS[5] + tmpFx[13]*tmpObjS[10] + tmpFx[18]*tmpObjS[15] + tmpFx[23]*tmpObjS[20];
tmpQ2[16] = + tmpFx[3]*tmpObjS[1] + tmpFx[8]*tmpObjS[6] + tmpFx[13]*tmpObjS[11] + tmpFx[18]*tmpObjS[16] + tmpFx[23]*tmpObjS[21];
tmpQ2[17] = + tmpFx[3]*tmpObjS[2] + tmpFx[8]*tmpObjS[7] + tmpFx[13]*tmpObjS[12] + tmpFx[18]*tmpObjS[17] + tmpFx[23]*tmpObjS[22];
tmpQ2[18] = + tmpFx[3]*tmpObjS[3] + tmpFx[8]*tmpObjS[8] + tmpFx[13]*tmpObjS[13] + tmpFx[18]*tmpObjS[18] + tmpFx[23]*tmpObjS[23];
tmpQ2[19] = + tmpFx[3]*tmpObjS[4] + tmpFx[8]*tmpObjS[9] + tmpFx[13]*tmpObjS[14] + tmpFx[18]*tmpObjS[19] + tmpFx[23]*tmpObjS[24];
tmpQ2[20] = + tmpFx[4]*tmpObjS[0] + tmpFx[9]*tmpObjS[5] + tmpFx[14]*tmpObjS[10] + tmpFx[19]*tmpObjS[15] + tmpFx[24]*tmpObjS[20];
tmpQ2[21] = + tmpFx[4]*tmpObjS[1] + tmpFx[9]*tmpObjS[6] + tmpFx[14]*tmpObjS[11] + tmpFx[19]*tmpObjS[16] + tmpFx[24]*tmpObjS[21];
tmpQ2[22] = + tmpFx[4]*tmpObjS[2] + tmpFx[9]*tmpObjS[7] + tmpFx[14]*tmpObjS[12] + tmpFx[19]*tmpObjS[17] + tmpFx[24]*tmpObjS[22];
tmpQ2[23] = + tmpFx[4]*tmpObjS[3] + tmpFx[9]*tmpObjS[8] + tmpFx[14]*tmpObjS[13] + tmpFx[19]*tmpObjS[18] + tmpFx[24]*tmpObjS[23];
tmpQ2[24] = + tmpFx[4]*tmpObjS[4] + tmpFx[9]*tmpObjS[9] + tmpFx[14]*tmpObjS[14] + tmpFx[19]*tmpObjS[19] + tmpFx[24]*tmpObjS[24];
tmpQ1[0] = + tmpQ2[0]*tmpFx[0] + tmpQ2[1]*tmpFx[5] + tmpQ2[2]*tmpFx[10] + tmpQ2[3]*tmpFx[15] + tmpQ2[4]*tmpFx[20];
tmpQ1[1] = + tmpQ2[0]*tmpFx[1] + tmpQ2[1]*tmpFx[6] + tmpQ2[2]*tmpFx[11] + tmpQ2[3]*tmpFx[16] + tmpQ2[4]*tmpFx[21];
tmpQ1[2] = + tmpQ2[0]*tmpFx[2] + tmpQ2[1]*tmpFx[7] + tmpQ2[2]*tmpFx[12] + tmpQ2[3]*tmpFx[17] + tmpQ2[4]*tmpFx[22];
tmpQ1[3] = + tmpQ2[0]*tmpFx[3] + tmpQ2[1]*tmpFx[8] + tmpQ2[2]*tmpFx[13] + tmpQ2[3]*tmpFx[18] + tmpQ2[4]*tmpFx[23];
tmpQ1[4] = + tmpQ2[0]*tmpFx[4] + tmpQ2[1]*tmpFx[9] + tmpQ2[2]*tmpFx[14] + tmpQ2[3]*tmpFx[19] + tmpQ2[4]*tmpFx[24];
tmpQ1[5] = + tmpQ2[5]*tmpFx[0] + tmpQ2[6]*tmpFx[5] + tmpQ2[7]*tmpFx[10] + tmpQ2[8]*tmpFx[15] + tmpQ2[9]*tmpFx[20];
tmpQ1[6] = + tmpQ2[5]*tmpFx[1] + tmpQ2[6]*tmpFx[6] + tmpQ2[7]*tmpFx[11] + tmpQ2[8]*tmpFx[16] + tmpQ2[9]*tmpFx[21];
tmpQ1[7] = + tmpQ2[5]*tmpFx[2] + tmpQ2[6]*tmpFx[7] + tmpQ2[7]*tmpFx[12] + tmpQ2[8]*tmpFx[17] + tmpQ2[9]*tmpFx[22];
tmpQ1[8] = + tmpQ2[5]*tmpFx[3] + tmpQ2[6]*tmpFx[8] + tmpQ2[7]*tmpFx[13] + tmpQ2[8]*tmpFx[18] + tmpQ2[9]*tmpFx[23];
tmpQ1[9] = + tmpQ2[5]*tmpFx[4] + tmpQ2[6]*tmpFx[9] + tmpQ2[7]*tmpFx[14] + tmpQ2[8]*tmpFx[19] + tmpQ2[9]*tmpFx[24];
tmpQ1[10] = + tmpQ2[10]*tmpFx[0] + tmpQ2[11]*tmpFx[5] + tmpQ2[12]*tmpFx[10] + tmpQ2[13]*tmpFx[15] + tmpQ2[14]*tmpFx[20];
tmpQ1[11] = + tmpQ2[10]*tmpFx[1] + tmpQ2[11]*tmpFx[6] + tmpQ2[12]*tmpFx[11] + tmpQ2[13]*tmpFx[16] + tmpQ2[14]*tmpFx[21];
tmpQ1[12] = + tmpQ2[10]*tmpFx[2] + tmpQ2[11]*tmpFx[7] + tmpQ2[12]*tmpFx[12] + tmpQ2[13]*tmpFx[17] + tmpQ2[14]*tmpFx[22];
tmpQ1[13] = + tmpQ2[10]*tmpFx[3] + tmpQ2[11]*tmpFx[8] + tmpQ2[12]*tmpFx[13] + tmpQ2[13]*tmpFx[18] + tmpQ2[14]*tmpFx[23];
tmpQ1[14] = + tmpQ2[10]*tmpFx[4] + tmpQ2[11]*tmpFx[9] + tmpQ2[12]*tmpFx[14] + tmpQ2[13]*tmpFx[19] + tmpQ2[14]*tmpFx[24];
tmpQ1[15] = + tmpQ2[15]*tmpFx[0] + tmpQ2[16]*tmpFx[5] + tmpQ2[17]*tmpFx[10] + tmpQ2[18]*tmpFx[15] + tmpQ2[19]*tmpFx[20];
tmpQ1[16] = + tmpQ2[15]*tmpFx[1] + tmpQ2[16]*tmpFx[6] + tmpQ2[17]*tmpFx[11] + tmpQ2[18]*tmpFx[16] + tmpQ2[19]*tmpFx[21];
tmpQ1[17] = + tmpQ2[15]*tmpFx[2] + tmpQ2[16]*tmpFx[7] + tmpQ2[17]*tmpFx[12] + tmpQ2[18]*tmpFx[17] + tmpQ2[19]*tmpFx[22];
tmpQ1[18] = + tmpQ2[15]*tmpFx[3] + tmpQ2[16]*tmpFx[8] + tmpQ2[17]*tmpFx[13] + tmpQ2[18]*tmpFx[18] + tmpQ2[19]*tmpFx[23];
tmpQ1[19] = + tmpQ2[15]*tmpFx[4] + tmpQ2[16]*tmpFx[9] + tmpQ2[17]*tmpFx[14] + tmpQ2[18]*tmpFx[19] + tmpQ2[19]*tmpFx[24];
tmpQ1[20] = + tmpQ2[20]*tmpFx[0] + tmpQ2[21]*tmpFx[5] + tmpQ2[22]*tmpFx[10] + tmpQ2[23]*tmpFx[15] + tmpQ2[24]*tmpFx[20];
tmpQ1[21] = + tmpQ2[20]*tmpFx[1] + tmpQ2[21]*tmpFx[6] + tmpQ2[22]*tmpFx[11] + tmpQ2[23]*tmpFx[16] + tmpQ2[24]*tmpFx[21];
tmpQ1[22] = + tmpQ2[20]*tmpFx[2] + tmpQ2[21]*tmpFx[7] + tmpQ2[22]*tmpFx[12] + tmpQ2[23]*tmpFx[17] + tmpQ2[24]*tmpFx[22];
tmpQ1[23] = + tmpQ2[20]*tmpFx[3] + tmpQ2[21]*tmpFx[8] + tmpQ2[22]*tmpFx[13] + tmpQ2[23]*tmpFx[18] + tmpQ2[24]*tmpFx[23];
tmpQ1[24] = + tmpQ2[20]*tmpFx[4] + tmpQ2[21]*tmpFx[9] + tmpQ2[22]*tmpFx[14] + tmpQ2[23]*tmpFx[19] + tmpQ2[24]*tmpFx[24];
}

void acado_setObjR1R2( real_t* const tmpFu, real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = + tmpFu[0]*tmpObjS[0] + tmpFu[3]*tmpObjS[5] + tmpFu[6]*tmpObjS[10] + tmpFu[9]*tmpObjS[15] + tmpFu[12]*tmpObjS[20];
tmpR2[1] = + tmpFu[0]*tmpObjS[1] + tmpFu[3]*tmpObjS[6] + tmpFu[6]*tmpObjS[11] + tmpFu[9]*tmpObjS[16] + tmpFu[12]*tmpObjS[21];
tmpR2[2] = + tmpFu[0]*tmpObjS[2] + tmpFu[3]*tmpObjS[7] + tmpFu[6]*tmpObjS[12] + tmpFu[9]*tmpObjS[17] + tmpFu[12]*tmpObjS[22];
tmpR2[3] = + tmpFu[0]*tmpObjS[3] + tmpFu[3]*tmpObjS[8] + tmpFu[6]*tmpObjS[13] + tmpFu[9]*tmpObjS[18] + tmpFu[12]*tmpObjS[23];
tmpR2[4] = + tmpFu[0]*tmpObjS[4] + tmpFu[3]*tmpObjS[9] + tmpFu[6]*tmpObjS[14] + tmpFu[9]*tmpObjS[19] + tmpFu[12]*tmpObjS[24];
tmpR2[5] = + tmpFu[1]*tmpObjS[0] + tmpFu[4]*tmpObjS[5] + tmpFu[7]*tmpObjS[10] + tmpFu[10]*tmpObjS[15] + tmpFu[13]*tmpObjS[20];
tmpR2[6] = + tmpFu[1]*tmpObjS[1] + tmpFu[4]*tmpObjS[6] + tmpFu[7]*tmpObjS[11] + tmpFu[10]*tmpObjS[16] + tmpFu[13]*tmpObjS[21];
tmpR2[7] = + tmpFu[1]*tmpObjS[2] + tmpFu[4]*tmpObjS[7] + tmpFu[7]*tmpObjS[12] + tmpFu[10]*tmpObjS[17] + tmpFu[13]*tmpObjS[22];
tmpR2[8] = + tmpFu[1]*tmpObjS[3] + tmpFu[4]*tmpObjS[8] + tmpFu[7]*tmpObjS[13] + tmpFu[10]*tmpObjS[18] + tmpFu[13]*tmpObjS[23];
tmpR2[9] = + tmpFu[1]*tmpObjS[4] + tmpFu[4]*tmpObjS[9] + tmpFu[7]*tmpObjS[14] + tmpFu[10]*tmpObjS[19] + tmpFu[13]*tmpObjS[24];
tmpR2[10] = + tmpFu[2]*tmpObjS[0] + tmpFu[5]*tmpObjS[5] + tmpFu[8]*tmpObjS[10] + tmpFu[11]*tmpObjS[15] + tmpFu[14]*tmpObjS[20];
tmpR2[11] = + tmpFu[2]*tmpObjS[1] + tmpFu[5]*tmpObjS[6] + tmpFu[8]*tmpObjS[11] + tmpFu[11]*tmpObjS[16] + tmpFu[14]*tmpObjS[21];
tmpR2[12] = + tmpFu[2]*tmpObjS[2] + tmpFu[5]*tmpObjS[7] + tmpFu[8]*tmpObjS[12] + tmpFu[11]*tmpObjS[17] + tmpFu[14]*tmpObjS[22];
tmpR2[13] = + tmpFu[2]*tmpObjS[3] + tmpFu[5]*tmpObjS[8] + tmpFu[8]*tmpObjS[13] + tmpFu[11]*tmpObjS[18] + tmpFu[14]*tmpObjS[23];
tmpR2[14] = + tmpFu[2]*tmpObjS[4] + tmpFu[5]*tmpObjS[9] + tmpFu[8]*tmpObjS[14] + tmpFu[11]*tmpObjS[19] + tmpFu[14]*tmpObjS[24];
tmpR1[0] = + tmpR2[0]*tmpFu[0] + tmpR2[1]*tmpFu[3] + tmpR2[2]*tmpFu[6] + tmpR2[3]*tmpFu[9] + tmpR2[4]*tmpFu[12];
tmpR1[1] = + tmpR2[0]*tmpFu[1] + tmpR2[1]*tmpFu[4] + tmpR2[2]*tmpFu[7] + tmpR2[3]*tmpFu[10] + tmpR2[4]*tmpFu[13];
tmpR1[2] = + tmpR2[0]*tmpFu[2] + tmpR2[1]*tmpFu[5] + tmpR2[2]*tmpFu[8] + tmpR2[3]*tmpFu[11] + tmpR2[4]*tmpFu[14];
tmpR1[3] = + tmpR2[5]*tmpFu[0] + tmpR2[6]*tmpFu[3] + tmpR2[7]*tmpFu[6] + tmpR2[8]*tmpFu[9] + tmpR2[9]*tmpFu[12];
tmpR1[4] = + tmpR2[5]*tmpFu[1] + tmpR2[6]*tmpFu[4] + tmpR2[7]*tmpFu[7] + tmpR2[8]*tmpFu[10] + tmpR2[9]*tmpFu[13];
tmpR1[5] = + tmpR2[5]*tmpFu[2] + tmpR2[6]*tmpFu[5] + tmpR2[7]*tmpFu[8] + tmpR2[8]*tmpFu[11] + tmpR2[9]*tmpFu[14];
tmpR1[6] = + tmpR2[10]*tmpFu[0] + tmpR2[11]*tmpFu[3] + tmpR2[12]*tmpFu[6] + tmpR2[13]*tmpFu[9] + tmpR2[14]*tmpFu[12];
tmpR1[7] = + tmpR2[10]*tmpFu[1] + tmpR2[11]*tmpFu[4] + tmpR2[12]*tmpFu[7] + tmpR2[13]*tmpFu[10] + tmpR2[14]*tmpFu[13];
tmpR1[8] = + tmpR2[10]*tmpFu[2] + tmpR2[11]*tmpFu[5] + tmpR2[12]*tmpFu[8] + tmpR2[13]*tmpFu[11] + tmpR2[14]*tmpFu[14];
}

void acado_setObjQN1QN2( real_t* const tmpFx, real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = + tmpFx[0]*tmpObjSEndTerm[0];
tmpQN2[1] = + tmpFx[1]*tmpObjSEndTerm[0];
tmpQN2[2] = + tmpFx[2]*tmpObjSEndTerm[0];
tmpQN2[3] = + tmpFx[3]*tmpObjSEndTerm[0];
tmpQN2[4] = + tmpFx[4]*tmpObjSEndTerm[0];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2];
tmpQN1[3] = + tmpQN2[0]*tmpFx[3];
tmpQN1[4] = + tmpQN2[0]*tmpFx[4];
tmpQN1[5] = + tmpQN2[1]*tmpFx[0];
tmpQN1[6] = + tmpQN2[1]*tmpFx[1];
tmpQN1[7] = + tmpQN2[1]*tmpFx[2];
tmpQN1[8] = + tmpQN2[1]*tmpFx[3];
tmpQN1[9] = + tmpQN2[1]*tmpFx[4];
tmpQN1[10] = + tmpQN2[2]*tmpFx[0];
tmpQN1[11] = + tmpQN2[2]*tmpFx[1];
tmpQN1[12] = + tmpQN2[2]*tmpFx[2];
tmpQN1[13] = + tmpQN2[2]*tmpFx[3];
tmpQN1[14] = + tmpQN2[2]*tmpFx[4];
tmpQN1[15] = + tmpQN2[3]*tmpFx[0];
tmpQN1[16] = + tmpQN2[3]*tmpFx[1];
tmpQN1[17] = + tmpQN2[3]*tmpFx[2];
tmpQN1[18] = + tmpQN2[3]*tmpFx[3];
tmpQN1[19] = + tmpQN2[3]*tmpFx[4];
tmpQN1[20] = + tmpQN2[4]*tmpFx[0];
tmpQN1[21] = + tmpQN2[4]*tmpFx[1];
tmpQN1[22] = + tmpQN2[4]*tmpFx[2];
tmpQN1[23] = + tmpQN2[4]*tmpFx[3];
tmpQN1[24] = + tmpQN2[4]*tmpFx[4];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 20; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 5];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 5 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 5 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 5 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.x[runObj * 5 + 4];
acadoWorkspace.objValueIn[5] = acadoVariables.u[runObj * 3];
acadoWorkspace.objValueIn[6] = acadoVariables.u[runObj * 3 + 1];
acadoWorkspace.objValueIn[7] = acadoVariables.u[runObj * 3 + 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[runObj * 2];
acadoWorkspace.objValueIn[9] = acadoVariables.od[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 5] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 5 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 5 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 5 + 3] = acadoWorkspace.objValueOut[3];
acadoWorkspace.Dy[runObj * 5 + 4] = acadoWorkspace.objValueOut[4];

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 5 ]), &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.Q1[ runObj * 25 ]), &(acadoWorkspace.Q2[ runObj * 25 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 30 ]), &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.R1[ runObj * 9 ]), &(acadoWorkspace.R2[ runObj * 15 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[100];
acadoWorkspace.objValueIn[1] = acadoVariables.x[101];
acadoWorkspace.objValueIn[2] = acadoVariables.x[102];
acadoWorkspace.objValueIn[3] = acadoVariables.x[103];
acadoWorkspace.objValueIn[4] = acadoVariables.x[104];
acadoWorkspace.objValueIn[5] = acadoVariables.od[40];
acadoWorkspace.objValueIn[6] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 1 ]), acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_multGxd( real_t* const dOld, real_t* const Gx1, real_t* const dNew )
{
dNew[0] += + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3] + Gx1[4]*dOld[4];
dNew[1] += + Gx1[5]*dOld[0] + Gx1[6]*dOld[1] + Gx1[7]*dOld[2] + Gx1[8]*dOld[3] + Gx1[9]*dOld[4];
dNew[2] += + Gx1[10]*dOld[0] + Gx1[11]*dOld[1] + Gx1[12]*dOld[2] + Gx1[13]*dOld[3] + Gx1[14]*dOld[4];
dNew[3] += + Gx1[15]*dOld[0] + Gx1[16]*dOld[1] + Gx1[17]*dOld[2] + Gx1[18]*dOld[3] + Gx1[19]*dOld[4];
dNew[4] += + Gx1[20]*dOld[0] + Gx1[21]*dOld[1] + Gx1[22]*dOld[2] + Gx1[23]*dOld[3] + Gx1[24]*dOld[4];
}

void acado_moveGxT( real_t* const Gx1, real_t* const Gx2 )
{
Gx2[0] = Gx1[0];
Gx2[1] = Gx1[1];
Gx2[2] = Gx1[2];
Gx2[3] = Gx1[3];
Gx2[4] = Gx1[4];
Gx2[5] = Gx1[5];
Gx2[6] = Gx1[6];
Gx2[7] = Gx1[7];
Gx2[8] = Gx1[8];
Gx2[9] = Gx1[9];
Gx2[10] = Gx1[10];
Gx2[11] = Gx1[11];
Gx2[12] = Gx1[12];
Gx2[13] = Gx1[13];
Gx2[14] = Gx1[14];
Gx2[15] = Gx1[15];
Gx2[16] = Gx1[16];
Gx2[17] = Gx1[17];
Gx2[18] = Gx1[18];
Gx2[19] = Gx1[19];
Gx2[20] = Gx1[20];
Gx2[21] = Gx1[21];
Gx2[22] = Gx1[22];
Gx2[23] = Gx1[23];
Gx2[24] = Gx1[24];
}

void acado_multGxGx( real_t* const Gx1, real_t* const Gx2, real_t* const Gx3 )
{
Gx3[0] = + Gx1[0]*Gx2[0] + Gx1[1]*Gx2[5] + Gx1[2]*Gx2[10] + Gx1[3]*Gx2[15] + Gx1[4]*Gx2[20];
Gx3[1] = + Gx1[0]*Gx2[1] + Gx1[1]*Gx2[6] + Gx1[2]*Gx2[11] + Gx1[3]*Gx2[16] + Gx1[4]*Gx2[21];
Gx3[2] = + Gx1[0]*Gx2[2] + Gx1[1]*Gx2[7] + Gx1[2]*Gx2[12] + Gx1[3]*Gx2[17] + Gx1[4]*Gx2[22];
Gx3[3] = + Gx1[0]*Gx2[3] + Gx1[1]*Gx2[8] + Gx1[2]*Gx2[13] + Gx1[3]*Gx2[18] + Gx1[4]*Gx2[23];
Gx3[4] = + Gx1[0]*Gx2[4] + Gx1[1]*Gx2[9] + Gx1[2]*Gx2[14] + Gx1[3]*Gx2[19] + Gx1[4]*Gx2[24];
Gx3[5] = + Gx1[5]*Gx2[0] + Gx1[6]*Gx2[5] + Gx1[7]*Gx2[10] + Gx1[8]*Gx2[15] + Gx1[9]*Gx2[20];
Gx3[6] = + Gx1[5]*Gx2[1] + Gx1[6]*Gx2[6] + Gx1[7]*Gx2[11] + Gx1[8]*Gx2[16] + Gx1[9]*Gx2[21];
Gx3[7] = + Gx1[5]*Gx2[2] + Gx1[6]*Gx2[7] + Gx1[7]*Gx2[12] + Gx1[8]*Gx2[17] + Gx1[9]*Gx2[22];
Gx3[8] = + Gx1[5]*Gx2[3] + Gx1[6]*Gx2[8] + Gx1[7]*Gx2[13] + Gx1[8]*Gx2[18] + Gx1[9]*Gx2[23];
Gx3[9] = + Gx1[5]*Gx2[4] + Gx1[6]*Gx2[9] + Gx1[7]*Gx2[14] + Gx1[8]*Gx2[19] + Gx1[9]*Gx2[24];
Gx3[10] = + Gx1[10]*Gx2[0] + Gx1[11]*Gx2[5] + Gx1[12]*Gx2[10] + Gx1[13]*Gx2[15] + Gx1[14]*Gx2[20];
Gx3[11] = + Gx1[10]*Gx2[1] + Gx1[11]*Gx2[6] + Gx1[12]*Gx2[11] + Gx1[13]*Gx2[16] + Gx1[14]*Gx2[21];
Gx3[12] = + Gx1[10]*Gx2[2] + Gx1[11]*Gx2[7] + Gx1[12]*Gx2[12] + Gx1[13]*Gx2[17] + Gx1[14]*Gx2[22];
Gx3[13] = + Gx1[10]*Gx2[3] + Gx1[11]*Gx2[8] + Gx1[12]*Gx2[13] + Gx1[13]*Gx2[18] + Gx1[14]*Gx2[23];
Gx3[14] = + Gx1[10]*Gx2[4] + Gx1[11]*Gx2[9] + Gx1[12]*Gx2[14] + Gx1[13]*Gx2[19] + Gx1[14]*Gx2[24];
Gx3[15] = + Gx1[15]*Gx2[0] + Gx1[16]*Gx2[5] + Gx1[17]*Gx2[10] + Gx1[18]*Gx2[15] + Gx1[19]*Gx2[20];
Gx3[16] = + Gx1[15]*Gx2[1] + Gx1[16]*Gx2[6] + Gx1[17]*Gx2[11] + Gx1[18]*Gx2[16] + Gx1[19]*Gx2[21];
Gx3[17] = + Gx1[15]*Gx2[2] + Gx1[16]*Gx2[7] + Gx1[17]*Gx2[12] + Gx1[18]*Gx2[17] + Gx1[19]*Gx2[22];
Gx3[18] = + Gx1[15]*Gx2[3] + Gx1[16]*Gx2[8] + Gx1[17]*Gx2[13] + Gx1[18]*Gx2[18] + Gx1[19]*Gx2[23];
Gx3[19] = + Gx1[15]*Gx2[4] + Gx1[16]*Gx2[9] + Gx1[17]*Gx2[14] + Gx1[18]*Gx2[19] + Gx1[19]*Gx2[24];
Gx3[20] = + Gx1[20]*Gx2[0] + Gx1[21]*Gx2[5] + Gx1[22]*Gx2[10] + Gx1[23]*Gx2[15] + Gx1[24]*Gx2[20];
Gx3[21] = + Gx1[20]*Gx2[1] + Gx1[21]*Gx2[6] + Gx1[22]*Gx2[11] + Gx1[23]*Gx2[16] + Gx1[24]*Gx2[21];
Gx3[22] = + Gx1[20]*Gx2[2] + Gx1[21]*Gx2[7] + Gx1[22]*Gx2[12] + Gx1[23]*Gx2[17] + Gx1[24]*Gx2[22];
Gx3[23] = + Gx1[20]*Gx2[3] + Gx1[21]*Gx2[8] + Gx1[22]*Gx2[13] + Gx1[23]*Gx2[18] + Gx1[24]*Gx2[23];
Gx3[24] = + Gx1[20]*Gx2[4] + Gx1[21]*Gx2[9] + Gx1[22]*Gx2[14] + Gx1[23]*Gx2[19] + Gx1[24]*Gx2[24];
}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[3] + Gx1[2]*Gu1[6] + Gx1[3]*Gu1[9] + Gx1[4]*Gu1[12];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[1]*Gu1[4] + Gx1[2]*Gu1[7] + Gx1[3]*Gu1[10] + Gx1[4]*Gu1[13];
Gu2[2] = + Gx1[0]*Gu1[2] + Gx1[1]*Gu1[5] + Gx1[2]*Gu1[8] + Gx1[3]*Gu1[11] + Gx1[4]*Gu1[14];
Gu2[3] = + Gx1[5]*Gu1[0] + Gx1[6]*Gu1[3] + Gx1[7]*Gu1[6] + Gx1[8]*Gu1[9] + Gx1[9]*Gu1[12];
Gu2[4] = + Gx1[5]*Gu1[1] + Gx1[6]*Gu1[4] + Gx1[7]*Gu1[7] + Gx1[8]*Gu1[10] + Gx1[9]*Gu1[13];
Gu2[5] = + Gx1[5]*Gu1[2] + Gx1[6]*Gu1[5] + Gx1[7]*Gu1[8] + Gx1[8]*Gu1[11] + Gx1[9]*Gu1[14];
Gu2[6] = + Gx1[10]*Gu1[0] + Gx1[11]*Gu1[3] + Gx1[12]*Gu1[6] + Gx1[13]*Gu1[9] + Gx1[14]*Gu1[12];
Gu2[7] = + Gx1[10]*Gu1[1] + Gx1[11]*Gu1[4] + Gx1[12]*Gu1[7] + Gx1[13]*Gu1[10] + Gx1[14]*Gu1[13];
Gu2[8] = + Gx1[10]*Gu1[2] + Gx1[11]*Gu1[5] + Gx1[12]*Gu1[8] + Gx1[13]*Gu1[11] + Gx1[14]*Gu1[14];
Gu2[9] = + Gx1[15]*Gu1[0] + Gx1[16]*Gu1[3] + Gx1[17]*Gu1[6] + Gx1[18]*Gu1[9] + Gx1[19]*Gu1[12];
Gu2[10] = + Gx1[15]*Gu1[1] + Gx1[16]*Gu1[4] + Gx1[17]*Gu1[7] + Gx1[18]*Gu1[10] + Gx1[19]*Gu1[13];
Gu2[11] = + Gx1[15]*Gu1[2] + Gx1[16]*Gu1[5] + Gx1[17]*Gu1[8] + Gx1[18]*Gu1[11] + Gx1[19]*Gu1[14];
Gu2[12] = + Gx1[20]*Gu1[0] + Gx1[21]*Gu1[3] + Gx1[22]*Gu1[6] + Gx1[23]*Gu1[9] + Gx1[24]*Gu1[12];
Gu2[13] = + Gx1[20]*Gu1[1] + Gx1[21]*Gu1[4] + Gx1[22]*Gu1[7] + Gx1[23]*Gu1[10] + Gx1[24]*Gu1[13];
Gu2[14] = + Gx1[20]*Gu1[2] + Gx1[21]*Gu1[5] + Gx1[22]*Gu1[8] + Gx1[23]*Gu1[11] + Gx1[24]*Gu1[14];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
Gu2[3] = Gu1[3];
Gu2[4] = Gu1[4];
Gu2[5] = Gu1[5];
Gu2[6] = Gu1[6];
Gu2[7] = Gu1[7];
Gu2[8] = Gu1[8];
Gu2[9] = Gu1[9];
Gu2[10] = Gu1[10];
Gu2[11] = Gu1[11];
Gu2[12] = Gu1[12];
Gu2[13] = Gu1[13];
Gu2[14] = Gu1[14];
}

void acado_setBlockH11( int iRow, int iCol, real_t* const Gu1, real_t* const Gu2 )
{
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 5)] += + Gu1[0]*Gu2[0] + Gu1[3]*Gu2[3] + Gu1[6]*Gu2[6] + Gu1[9]*Gu2[9] + Gu1[12]*Gu2[12];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 6)] += + Gu1[0]*Gu2[1] + Gu1[3]*Gu2[4] + Gu1[6]*Gu2[7] + Gu1[9]*Gu2[10] + Gu1[12]*Gu2[13];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 7)] += + Gu1[0]*Gu2[2] + Gu1[3]*Gu2[5] + Gu1[6]*Gu2[8] + Gu1[9]*Gu2[11] + Gu1[12]*Gu2[14];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 5)] += + Gu1[1]*Gu2[0] + Gu1[4]*Gu2[3] + Gu1[7]*Gu2[6] + Gu1[10]*Gu2[9] + Gu1[13]*Gu2[12];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 6)] += + Gu1[1]*Gu2[1] + Gu1[4]*Gu2[4] + Gu1[7]*Gu2[7] + Gu1[10]*Gu2[10] + Gu1[13]*Gu2[13];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 7)] += + Gu1[1]*Gu2[2] + Gu1[4]*Gu2[5] + Gu1[7]*Gu2[8] + Gu1[10]*Gu2[11] + Gu1[13]*Gu2[14];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 5)] += + Gu1[2]*Gu2[0] + Gu1[5]*Gu2[3] + Gu1[8]*Gu2[6] + Gu1[11]*Gu2[9] + Gu1[14]*Gu2[12];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 6)] += + Gu1[2]*Gu2[1] + Gu1[5]*Gu2[4] + Gu1[8]*Gu2[7] + Gu1[11]*Gu2[10] + Gu1[14]*Gu2[13];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 7)] += + Gu1[2]*Gu2[2] + Gu1[5]*Gu2[5] + Gu1[8]*Gu2[8] + Gu1[11]*Gu2[11] + Gu1[14]*Gu2[14];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 5)] = R11[0];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 6)] = R11[1];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 7)] = R11[2];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 5)] = R11[3];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 6)] = R11[4];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 7)] = R11[5];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 5)] = R11[6];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 6)] = R11[7];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 7)] = R11[8];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 195 + 325) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 195 + 390) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 195 + 325) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 195 + 455) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 195 + 325) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 195 + 390) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 195 + 390) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 195 + 455) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 195 + 325) + (iRow * 3 + 7)];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 195 + 390) + (iRow * 3 + 7)];
acadoWorkspace.H[(iRow * 195 + 455) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 195 + 455) + (iRow * 3 + 7)];
}

void acado_multQ1d( real_t* const Gx1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3] + Gx1[4]*dOld[4];
dNew[1] = + Gx1[5]*dOld[0] + Gx1[6]*dOld[1] + Gx1[7]*dOld[2] + Gx1[8]*dOld[3] + Gx1[9]*dOld[4];
dNew[2] = + Gx1[10]*dOld[0] + Gx1[11]*dOld[1] + Gx1[12]*dOld[2] + Gx1[13]*dOld[3] + Gx1[14]*dOld[4];
dNew[3] = + Gx1[15]*dOld[0] + Gx1[16]*dOld[1] + Gx1[17]*dOld[2] + Gx1[18]*dOld[3] + Gx1[19]*dOld[4];
dNew[4] = + Gx1[20]*dOld[0] + Gx1[21]*dOld[1] + Gx1[22]*dOld[2] + Gx1[23]*dOld[3] + Gx1[24]*dOld[4];
}

void acado_multQN1d( real_t* const QN1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + acadoWorkspace.QN1[0]*dOld[0] + acadoWorkspace.QN1[1]*dOld[1] + acadoWorkspace.QN1[2]*dOld[2] + acadoWorkspace.QN1[3]*dOld[3] + acadoWorkspace.QN1[4]*dOld[4];
dNew[1] = + acadoWorkspace.QN1[5]*dOld[0] + acadoWorkspace.QN1[6]*dOld[1] + acadoWorkspace.QN1[7]*dOld[2] + acadoWorkspace.QN1[8]*dOld[3] + acadoWorkspace.QN1[9]*dOld[4];
dNew[2] = + acadoWorkspace.QN1[10]*dOld[0] + acadoWorkspace.QN1[11]*dOld[1] + acadoWorkspace.QN1[12]*dOld[2] + acadoWorkspace.QN1[13]*dOld[3] + acadoWorkspace.QN1[14]*dOld[4];
dNew[3] = + acadoWorkspace.QN1[15]*dOld[0] + acadoWorkspace.QN1[16]*dOld[1] + acadoWorkspace.QN1[17]*dOld[2] + acadoWorkspace.QN1[18]*dOld[3] + acadoWorkspace.QN1[19]*dOld[4];
dNew[4] = + acadoWorkspace.QN1[20]*dOld[0] + acadoWorkspace.QN1[21]*dOld[1] + acadoWorkspace.QN1[22]*dOld[2] + acadoWorkspace.QN1[23]*dOld[3] + acadoWorkspace.QN1[24]*dOld[4];
}

void acado_multRDy( real_t* const R2, real_t* const Dy1, real_t* const RDy1 )
{
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2] + R2[3]*Dy1[3] + R2[4]*Dy1[4];
RDy1[1] = + R2[5]*Dy1[0] + R2[6]*Dy1[1] + R2[7]*Dy1[2] + R2[8]*Dy1[3] + R2[9]*Dy1[4];
RDy1[2] = + R2[10]*Dy1[0] + R2[11]*Dy1[1] + R2[12]*Dy1[2] + R2[13]*Dy1[3] + R2[14]*Dy1[4];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3] + Q2[4]*Dy1[4];
QDy1[1] = + Q2[5]*Dy1[0] + Q2[6]*Dy1[1] + Q2[7]*Dy1[2] + Q2[8]*Dy1[3] + Q2[9]*Dy1[4];
QDy1[2] = + Q2[10]*Dy1[0] + Q2[11]*Dy1[1] + Q2[12]*Dy1[2] + Q2[13]*Dy1[3] + Q2[14]*Dy1[4];
QDy1[3] = + Q2[15]*Dy1[0] + Q2[16]*Dy1[1] + Q2[17]*Dy1[2] + Q2[18]*Dy1[3] + Q2[19]*Dy1[4];
QDy1[4] = + Q2[20]*Dy1[0] + Q2[21]*Dy1[1] + Q2[22]*Dy1[2] + Q2[23]*Dy1[3] + Q2[24]*Dy1[4];
}

void acado_multEQDy( real_t* const E1, real_t* const QDy1, real_t* const U1 )
{
U1[0] += + E1[0]*QDy1[0] + E1[3]*QDy1[1] + E1[6]*QDy1[2] + E1[9]*QDy1[3] + E1[12]*QDy1[4];
U1[1] += + E1[1]*QDy1[0] + E1[4]*QDy1[1] + E1[7]*QDy1[2] + E1[10]*QDy1[3] + E1[13]*QDy1[4];
U1[2] += + E1[2]*QDy1[0] + E1[5]*QDy1[1] + E1[8]*QDy1[2] + E1[11]*QDy1[3] + E1[14]*QDy1[4];
}

void acado_multQETGx( real_t* const E1, real_t* const Gx1, real_t* const H101 )
{
H101[0] += + E1[0]*Gx1[0] + E1[3]*Gx1[5] + E1[6]*Gx1[10] + E1[9]*Gx1[15] + E1[12]*Gx1[20];
H101[1] += + E1[0]*Gx1[1] + E1[3]*Gx1[6] + E1[6]*Gx1[11] + E1[9]*Gx1[16] + E1[12]*Gx1[21];
H101[2] += + E1[0]*Gx1[2] + E1[3]*Gx1[7] + E1[6]*Gx1[12] + E1[9]*Gx1[17] + E1[12]*Gx1[22];
H101[3] += + E1[0]*Gx1[3] + E1[3]*Gx1[8] + E1[6]*Gx1[13] + E1[9]*Gx1[18] + E1[12]*Gx1[23];
H101[4] += + E1[0]*Gx1[4] + E1[3]*Gx1[9] + E1[6]*Gx1[14] + E1[9]*Gx1[19] + E1[12]*Gx1[24];
H101[5] += + E1[1]*Gx1[0] + E1[4]*Gx1[5] + E1[7]*Gx1[10] + E1[10]*Gx1[15] + E1[13]*Gx1[20];
H101[6] += + E1[1]*Gx1[1] + E1[4]*Gx1[6] + E1[7]*Gx1[11] + E1[10]*Gx1[16] + E1[13]*Gx1[21];
H101[7] += + E1[1]*Gx1[2] + E1[4]*Gx1[7] + E1[7]*Gx1[12] + E1[10]*Gx1[17] + E1[13]*Gx1[22];
H101[8] += + E1[1]*Gx1[3] + E1[4]*Gx1[8] + E1[7]*Gx1[13] + E1[10]*Gx1[18] + E1[13]*Gx1[23];
H101[9] += + E1[1]*Gx1[4] + E1[4]*Gx1[9] + E1[7]*Gx1[14] + E1[10]*Gx1[19] + E1[13]*Gx1[24];
H101[10] += + E1[2]*Gx1[0] + E1[5]*Gx1[5] + E1[8]*Gx1[10] + E1[11]*Gx1[15] + E1[14]*Gx1[20];
H101[11] += + E1[2]*Gx1[1] + E1[5]*Gx1[6] + E1[8]*Gx1[11] + E1[11]*Gx1[16] + E1[14]*Gx1[21];
H101[12] += + E1[2]*Gx1[2] + E1[5]*Gx1[7] + E1[8]*Gx1[12] + E1[11]*Gx1[17] + E1[14]*Gx1[22];
H101[13] += + E1[2]*Gx1[3] + E1[5]*Gx1[8] + E1[8]*Gx1[13] + E1[11]*Gx1[18] + E1[14]*Gx1[23];
H101[14] += + E1[2]*Gx1[4] + E1[5]*Gx1[9] + E1[8]*Gx1[14] + E1[11]*Gx1[19] + E1[14]*Gx1[24];
}

void acado_zeroBlockH10( real_t* const H101 )
{
{ int lCopy; for (lCopy = 0; lCopy < 15; lCopy++) H101[ lCopy ] = 0; }
}

void acado_multEDu( real_t* const E1, real_t* const U1, real_t* const dNew )
{
dNew[0] += + E1[0]*U1[0] + E1[1]*U1[1] + E1[2]*U1[2];
dNew[1] += + E1[3]*U1[0] + E1[4]*U1[1] + E1[5]*U1[2];
dNew[2] += + E1[6]*U1[0] + E1[7]*U1[1] + E1[8]*U1[2];
dNew[3] += + E1[9]*U1[0] + E1[10]*U1[1] + E1[11]*U1[2];
dNew[4] += + E1[12]*U1[0] + E1[13]*U1[1] + E1[14]*U1[2];
}

void acado_zeroBlockH00(  )
{
acadoWorkspace.H[0] = 0.0000000000000000e+00;
acadoWorkspace.H[1] = 0.0000000000000000e+00;
acadoWorkspace.H[2] = 0.0000000000000000e+00;
acadoWorkspace.H[3] = 0.0000000000000000e+00;
acadoWorkspace.H[4] = 0.0000000000000000e+00;
acadoWorkspace.H[65] = 0.0000000000000000e+00;
acadoWorkspace.H[66] = 0.0000000000000000e+00;
acadoWorkspace.H[67] = 0.0000000000000000e+00;
acadoWorkspace.H[68] = 0.0000000000000000e+00;
acadoWorkspace.H[69] = 0.0000000000000000e+00;
acadoWorkspace.H[130] = 0.0000000000000000e+00;
acadoWorkspace.H[131] = 0.0000000000000000e+00;
acadoWorkspace.H[132] = 0.0000000000000000e+00;
acadoWorkspace.H[133] = 0.0000000000000000e+00;
acadoWorkspace.H[134] = 0.0000000000000000e+00;
acadoWorkspace.H[195] = 0.0000000000000000e+00;
acadoWorkspace.H[196] = 0.0000000000000000e+00;
acadoWorkspace.H[197] = 0.0000000000000000e+00;
acadoWorkspace.H[198] = 0.0000000000000000e+00;
acadoWorkspace.H[199] = 0.0000000000000000e+00;
acadoWorkspace.H[260] = 0.0000000000000000e+00;
acadoWorkspace.H[261] = 0.0000000000000000e+00;
acadoWorkspace.H[262] = 0.0000000000000000e+00;
acadoWorkspace.H[263] = 0.0000000000000000e+00;
acadoWorkspace.H[264] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[5]*Gx2[5] + Gx1[10]*Gx2[10] + Gx1[15]*Gx2[15] + Gx1[20]*Gx2[20];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[5]*Gx2[6] + Gx1[10]*Gx2[11] + Gx1[15]*Gx2[16] + Gx1[20]*Gx2[21];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[5]*Gx2[7] + Gx1[10]*Gx2[12] + Gx1[15]*Gx2[17] + Gx1[20]*Gx2[22];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[5]*Gx2[8] + Gx1[10]*Gx2[13] + Gx1[15]*Gx2[18] + Gx1[20]*Gx2[23];
acadoWorkspace.H[4] += + Gx1[0]*Gx2[4] + Gx1[5]*Gx2[9] + Gx1[10]*Gx2[14] + Gx1[15]*Gx2[19] + Gx1[20]*Gx2[24];
acadoWorkspace.H[65] += + Gx1[1]*Gx2[0] + Gx1[6]*Gx2[5] + Gx1[11]*Gx2[10] + Gx1[16]*Gx2[15] + Gx1[21]*Gx2[20];
acadoWorkspace.H[66] += + Gx1[1]*Gx2[1] + Gx1[6]*Gx2[6] + Gx1[11]*Gx2[11] + Gx1[16]*Gx2[16] + Gx1[21]*Gx2[21];
acadoWorkspace.H[67] += + Gx1[1]*Gx2[2] + Gx1[6]*Gx2[7] + Gx1[11]*Gx2[12] + Gx1[16]*Gx2[17] + Gx1[21]*Gx2[22];
acadoWorkspace.H[68] += + Gx1[1]*Gx2[3] + Gx1[6]*Gx2[8] + Gx1[11]*Gx2[13] + Gx1[16]*Gx2[18] + Gx1[21]*Gx2[23];
acadoWorkspace.H[69] += + Gx1[1]*Gx2[4] + Gx1[6]*Gx2[9] + Gx1[11]*Gx2[14] + Gx1[16]*Gx2[19] + Gx1[21]*Gx2[24];
acadoWorkspace.H[130] += + Gx1[2]*Gx2[0] + Gx1[7]*Gx2[5] + Gx1[12]*Gx2[10] + Gx1[17]*Gx2[15] + Gx1[22]*Gx2[20];
acadoWorkspace.H[131] += + Gx1[2]*Gx2[1] + Gx1[7]*Gx2[6] + Gx1[12]*Gx2[11] + Gx1[17]*Gx2[16] + Gx1[22]*Gx2[21];
acadoWorkspace.H[132] += + Gx1[2]*Gx2[2] + Gx1[7]*Gx2[7] + Gx1[12]*Gx2[12] + Gx1[17]*Gx2[17] + Gx1[22]*Gx2[22];
acadoWorkspace.H[133] += + Gx1[2]*Gx2[3] + Gx1[7]*Gx2[8] + Gx1[12]*Gx2[13] + Gx1[17]*Gx2[18] + Gx1[22]*Gx2[23];
acadoWorkspace.H[134] += + Gx1[2]*Gx2[4] + Gx1[7]*Gx2[9] + Gx1[12]*Gx2[14] + Gx1[17]*Gx2[19] + Gx1[22]*Gx2[24];
acadoWorkspace.H[195] += + Gx1[3]*Gx2[0] + Gx1[8]*Gx2[5] + Gx1[13]*Gx2[10] + Gx1[18]*Gx2[15] + Gx1[23]*Gx2[20];
acadoWorkspace.H[196] += + Gx1[3]*Gx2[1] + Gx1[8]*Gx2[6] + Gx1[13]*Gx2[11] + Gx1[18]*Gx2[16] + Gx1[23]*Gx2[21];
acadoWorkspace.H[197] += + Gx1[3]*Gx2[2] + Gx1[8]*Gx2[7] + Gx1[13]*Gx2[12] + Gx1[18]*Gx2[17] + Gx1[23]*Gx2[22];
acadoWorkspace.H[198] += + Gx1[3]*Gx2[3] + Gx1[8]*Gx2[8] + Gx1[13]*Gx2[13] + Gx1[18]*Gx2[18] + Gx1[23]*Gx2[23];
acadoWorkspace.H[199] += + Gx1[3]*Gx2[4] + Gx1[8]*Gx2[9] + Gx1[13]*Gx2[14] + Gx1[18]*Gx2[19] + Gx1[23]*Gx2[24];
acadoWorkspace.H[260] += + Gx1[4]*Gx2[0] + Gx1[9]*Gx2[5] + Gx1[14]*Gx2[10] + Gx1[19]*Gx2[15] + Gx1[24]*Gx2[20];
acadoWorkspace.H[261] += + Gx1[4]*Gx2[1] + Gx1[9]*Gx2[6] + Gx1[14]*Gx2[11] + Gx1[19]*Gx2[16] + Gx1[24]*Gx2[21];
acadoWorkspace.H[262] += + Gx1[4]*Gx2[2] + Gx1[9]*Gx2[7] + Gx1[14]*Gx2[12] + Gx1[19]*Gx2[17] + Gx1[24]*Gx2[22];
acadoWorkspace.H[263] += + Gx1[4]*Gx2[3] + Gx1[9]*Gx2[8] + Gx1[14]*Gx2[13] + Gx1[19]*Gx2[18] + Gx1[24]*Gx2[23];
acadoWorkspace.H[264] += + Gx1[4]*Gx2[4] + Gx1[9]*Gx2[9] + Gx1[14]*Gx2[14] + Gx1[19]*Gx2[19] + Gx1[24]*Gx2[24];
}

void acado_multHxC( real_t* const Hx, real_t* const Gx, real_t* const A01 )
{
A01[0] = + Hx[0]*Gx[0] + Hx[1]*Gx[5] + Hx[2]*Gx[10] + Hx[3]*Gx[15] + Hx[4]*Gx[20];
A01[1] = + Hx[0]*Gx[1] + Hx[1]*Gx[6] + Hx[2]*Gx[11] + Hx[3]*Gx[16] + Hx[4]*Gx[21];
A01[2] = + Hx[0]*Gx[2] + Hx[1]*Gx[7] + Hx[2]*Gx[12] + Hx[3]*Gx[17] + Hx[4]*Gx[22];
A01[3] = + Hx[0]*Gx[3] + Hx[1]*Gx[8] + Hx[2]*Gx[13] + Hx[3]*Gx[18] + Hx[4]*Gx[23];
A01[4] = + Hx[0]*Gx[4] + Hx[1]*Gx[9] + Hx[2]*Gx[14] + Hx[3]*Gx[19] + Hx[4]*Gx[24];
A01[65] = + Hx[5]*Gx[0] + Hx[6]*Gx[5] + Hx[7]*Gx[10] + Hx[8]*Gx[15] + Hx[9]*Gx[20];
A01[66] = + Hx[5]*Gx[1] + Hx[6]*Gx[6] + Hx[7]*Gx[11] + Hx[8]*Gx[16] + Hx[9]*Gx[21];
A01[67] = + Hx[5]*Gx[2] + Hx[6]*Gx[7] + Hx[7]*Gx[12] + Hx[8]*Gx[17] + Hx[9]*Gx[22];
A01[68] = + Hx[5]*Gx[3] + Hx[6]*Gx[8] + Hx[7]*Gx[13] + Hx[8]*Gx[18] + Hx[9]*Gx[23];
A01[69] = + Hx[5]*Gx[4] + Hx[6]*Gx[9] + Hx[7]*Gx[14] + Hx[8]*Gx[19] + Hx[9]*Gx[24];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 130 + 1300) + (col * 3 + 5)] = + Hx[0]*E[0] + Hx[1]*E[3] + Hx[2]*E[6] + Hx[3]*E[9] + Hx[4]*E[12];
acadoWorkspace.A[(row * 130 + 1300) + (col * 3 + 6)] = + Hx[0]*E[1] + Hx[1]*E[4] + Hx[2]*E[7] + Hx[3]*E[10] + Hx[4]*E[13];
acadoWorkspace.A[(row * 130 + 1300) + (col * 3 + 7)] = + Hx[0]*E[2] + Hx[1]*E[5] + Hx[2]*E[8] + Hx[3]*E[11] + Hx[4]*E[14];
acadoWorkspace.A[(row * 130 + 1365) + (col * 3 + 5)] = + Hx[5]*E[0] + Hx[6]*E[3] + Hx[7]*E[6] + Hx[8]*E[9] + Hx[9]*E[12];
acadoWorkspace.A[(row * 130 + 1365) + (col * 3 + 6)] = + Hx[5]*E[1] + Hx[6]*E[4] + Hx[7]*E[7] + Hx[8]*E[10] + Hx[9]*E[13];
acadoWorkspace.A[(row * 130 + 1365) + (col * 3 + 7)] = + Hx[5]*E[2] + Hx[6]*E[5] + Hx[7]*E[8] + Hx[8]*E[11] + Hx[9]*E[14];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2] + Hx[3]*tmpd[3] + Hx[4]*tmpd[4];
acadoWorkspace.evHxd[1] = + Hx[5]*tmpd[0] + Hx[6]*tmpd[1] + Hx[7]*tmpd[2] + Hx[8]*tmpd[3] + Hx[9]*tmpd[4];
lbA[0] -= acadoWorkspace.evHxd[0];
lbA[1] -= acadoWorkspace.evHxd[1];
ubA[0] -= acadoWorkspace.evHxd[0];
ubA[1] -= acadoWorkspace.evHxd[1];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 5;
const real_t* od = in + 8;
/* Vector of auxiliary variables; number of elements: 37. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (real_t)(0.0000000000000000e+00);
a[1] = (real_t)(0.0000000000000000e+00);
a[2] = (real_t)(1.0000000000000000e+00);
a[3] = (real_t)(0.0000000000000000e+00);
a[4] = (real_t)(0.0000000000000000e+00);
a[5] = (real_t)(-1.0000000000000000e+00);
a[6] = (xd[1]/(real_t)(7.0000000000000000e+00));
a[7] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[8] = (a[7]*xd[1]);
a[9] = (a[6]+a[8]);
a[10] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[11] = (real_t)(7.0000000000000000e+00);
a[12] = (a[10]*a[11]);
a[13] = (xd[1]/(real_t)(7.0000000000000000e+00));
a[14] = (a[12]*a[13]);
a[15] = ((real_t)(1.0000000000000000e+00)/(real_t)(7.0000000000000000e+00));
a[16] = ((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)));
a[17] = (a[15]*a[16]);
a[18] = ((real_t)(1.0000000000000000e+00)/(real_t)(2.0000000000000000e+00));
a[19] = ((a[14]+a[17])*a[18]);
a[20] = (real_t)(-1.0000000000000000e+00);
a[21] = (a[19]*a[20]);
a[22] = (real_t)(-1.0000000000000000e+00);
a[23] = ((a[9]+a[21])*a[22]);
a[24] = (real_t)(1.8000000000000000e+00);
a[25] = (real_t)(-1.0000000000000000e+00);
a[26] = (a[24]*a[25]);
a[27] = (a[23]+a[26]);
a[28] = (real_t)(0.0000000000000000e+00);
a[29] = (real_t)(0.0000000000000000e+00);
a[30] = (real_t)(0.0000000000000000e+00);
a[31] = (real_t)(0.0000000000000000e+00);
a[32] = (real_t)(0.0000000000000000e+00);
a[33] = (real_t)(1.0000000000000000e+00);
a[34] = (real_t)(0.0000000000000000e+00);
a[35] = (real_t)(1.0000000000000000e+00);
a[36] = (real_t)(0.0000000000000000e+00);

/* Compute outputs: */
out[0] = (xd[2]+u[2]);
out[1] = ((((((od[0]-xd[0])-(real_t)(4.0000000000000000e+00))+((od[1]*(od[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(od[1]/(real_t)(7.0000000000000000e+00)))*(od[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((xd[1]*(xd[1]/(real_t)(7.0000000000000000e+00)))-((((real_t)(7.0000000000000000e+00)*(xd[1]/(real_t)(7.0000000000000000e+00)))*(xd[1]/(real_t)(7.0000000000000000e+00)))/(real_t)(2.0000000000000000e+00))))-((real_t)(1.8000000000000000e+00)*xd[1]))+u[1]);
out[2] = a[0];
out[3] = a[1];
out[4] = a[2];
out[5] = a[3];
out[6] = a[4];
out[7] = a[5];
out[8] = a[27];
out[9] = a[28];
out[10] = a[29];
out[11] = a[30];
out[12] = a[31];
out[13] = a[32];
out[14] = a[33];
out[15] = a[34];
out[16] = a[35];
out[17] = a[36];
}

void acado_macCTSlx( real_t* const C0, real_t* const g0 )
{
g0[0] += 0.0;
;
g0[1] += 0.0;
;
g0[2] += 0.0;
;
g0[3] += 0.0;
;
g0[4] += 0.0;
;
}

void acado_macETSlu( real_t* const E0, real_t* const g1 )
{
g1[0] += 0.0;
;
g1[1] += 0.0;
;
g1[2] += 0.0;
;
}

void acado_condensePrep(  )
{
int lRun1;
int lRun2;
int lRun3;
int lRun4;
int lRun5;
/** Row vector of size: 20 */
static const int xBoundIndices[ 20 ] = 
{ 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_moveGxT( &(acadoWorkspace.evGx[ 25 ]), acadoWorkspace.T );
acado_multGxd( acadoWorkspace.d, &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.d[ 5 ]) );
acado_multGxGx( acadoWorkspace.T, acadoWorkspace.evGx, &(acadoWorkspace.evGx[ 25 ]) );

acado_multGxGu( acadoWorkspace.T, acadoWorkspace.E, &(acadoWorkspace.E[ 15 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 15 ]), &(acadoWorkspace.E[ 30 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 50 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 5 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.d[ 10 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.evGx[ 50 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.E[ 45 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.E[ 60 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 30 ]), &(acadoWorkspace.E[ 75 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 75 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 10 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.d[ 15 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.evGx[ 75 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.E[ 90 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.E[ 105 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.E[ 120 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 45 ]), &(acadoWorkspace.E[ 135 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 100 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.d[ 20 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.evGx[ 100 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.E[ 150 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.E[ 165 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.E[ 180 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.E[ 195 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 60 ]), &(acadoWorkspace.E[ 210 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 125 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.d[ 25 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.evGx[ 125 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.E[ 225 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.E[ 240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.E[ 255 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.E[ 270 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.E[ 285 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 75 ]), &(acadoWorkspace.E[ 300 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 150 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 25 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.d[ 30 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.evGx[ 150 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.E[ 315 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.E[ 330 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.E[ 345 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.E[ 360 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.E[ 375 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.E[ 390 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 90 ]), &(acadoWorkspace.E[ 405 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 175 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.d[ 35 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.evGx[ 175 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.E[ 420 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.E[ 435 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.E[ 450 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.E[ 465 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.E[ 480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.E[ 495 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.E[ 510 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 105 ]), &(acadoWorkspace.E[ 525 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 200 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 35 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.d[ 40 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.evGx[ 200 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.E[ 540 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.E[ 555 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.E[ 570 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.E[ 585 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.E[ 600 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.E[ 615 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.E[ 630 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.E[ 645 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 120 ]), &(acadoWorkspace.E[ 660 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 225 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.d[ 45 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.evGx[ 225 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.E[ 675 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.E[ 690 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.E[ 705 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.E[ 720 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.E[ 735 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.E[ 750 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.E[ 765 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.E[ 780 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.E[ 795 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 135 ]), &(acadoWorkspace.E[ 810 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 250 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.d[ 50 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.evGx[ 250 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.E[ 825 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.E[ 840 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.E[ 855 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.E[ 870 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.E[ 885 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.E[ 900 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.E[ 915 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.E[ 930 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.E[ 945 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.E[ 960 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 150 ]), &(acadoWorkspace.E[ 975 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 275 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 50 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.d[ 55 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.evGx[ 275 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.E[ 990 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.E[ 1005 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.E[ 1020 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.E[ 1035 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.E[ 1050 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.E[ 1065 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.E[ 1080 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.E[ 1095 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.E[ 1110 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.E[ 1125 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 975 ]), &(acadoWorkspace.E[ 1140 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 165 ]), &(acadoWorkspace.E[ 1155 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 300 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 55 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.d[ 60 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.evGx[ 300 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.E[ 1170 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.E[ 1185 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.E[ 1200 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.E[ 1215 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.E[ 1230 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.E[ 1245 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.E[ 1260 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.E[ 1275 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.E[ 1290 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.E[ 1305 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.E[ 1320 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1155 ]), &(acadoWorkspace.E[ 1335 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 180 ]), &(acadoWorkspace.E[ 1350 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 325 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.d[ 65 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.evGx[ 325 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.E[ 1365 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.E[ 1380 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.E[ 1395 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.E[ 1410 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.E[ 1425 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.E[ 1440 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.E[ 1455 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.E[ 1470 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.E[ 1485 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.E[ 1500 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.E[ 1515 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.E[ 1530 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1350 ]), &(acadoWorkspace.E[ 1545 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 195 ]), &(acadoWorkspace.E[ 1560 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 350 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 65 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.d[ 70 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.evGx[ 350 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.E[ 1575 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.E[ 1590 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.E[ 1605 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.E[ 1620 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.E[ 1635 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.E[ 1650 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.E[ 1665 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.E[ 1680 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.E[ 1695 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.E[ 1710 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.E[ 1725 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.E[ 1740 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.E[ 1755 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.E[ 1770 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 210 ]), &(acadoWorkspace.E[ 1785 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 375 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 70 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.d[ 75 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.evGx[ 375 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.E[ 1800 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.E[ 1815 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.E[ 1830 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.E[ 1845 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.E[ 1860 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.E[ 1875 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.E[ 1890 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.E[ 1905 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.E[ 1920 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.E[ 1935 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.E[ 1950 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.E[ 1965 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.E[ 1980 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.E[ 1995 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1785 ]), &(acadoWorkspace.E[ 2010 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 225 ]), &(acadoWorkspace.E[ 2025 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 400 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.d[ 80 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.evGx[ 400 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.E[ 2040 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.E[ 2055 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.E[ 2070 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.E[ 2085 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.E[ 2100 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.E[ 2115 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.E[ 2130 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.E[ 2145 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.E[ 2160 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.E[ 2175 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.E[ 2190 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.E[ 2205 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.E[ 2220 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.E[ 2235 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.E[ 2250 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2025 ]), &(acadoWorkspace.E[ 2265 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 240 ]), &(acadoWorkspace.E[ 2280 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 425 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.d[ 85 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.evGx[ 425 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.E[ 2295 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.E[ 2310 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.E[ 2325 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.E[ 2340 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.E[ 2355 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.E[ 2370 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.E[ 2385 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.E[ 2400 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.E[ 2415 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.E[ 2430 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.E[ 2445 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.E[ 2460 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.E[ 2475 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.E[ 2490 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.E[ 2505 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.E[ 2520 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2280 ]), &(acadoWorkspace.E[ 2535 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 255 ]), &(acadoWorkspace.E[ 2550 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 450 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 85 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.d[ 90 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.evGx[ 450 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.E[ 2565 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.E[ 2580 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.E[ 2595 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.E[ 2610 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.E[ 2625 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.E[ 2640 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.E[ 2655 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.E[ 2670 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.E[ 2685 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.E[ 2700 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.E[ 2715 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.E[ 2730 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.E[ 2745 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.E[ 2760 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.E[ 2775 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.E[ 2790 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.E[ 2805 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2550 ]), &(acadoWorkspace.E[ 2820 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 270 ]), &(acadoWorkspace.E[ 2835 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 475 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.d[ 95 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.evGx[ 475 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.E[ 2850 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.E[ 2865 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.E[ 2880 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.E[ 2895 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.E[ 2910 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.E[ 2925 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.E[ 2940 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.E[ 2955 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.E[ 2970 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.E[ 2985 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.E[ 3000 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.E[ 3015 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.E[ 3030 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.E[ 3045 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.E[ 3060 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.E[ 3075 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.E[ 3090 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.E[ 3105 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 2835 ]), &(acadoWorkspace.E[ 3120 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 285 ]), &(acadoWorkspace.E[ 3135 ]) );

acado_multGxGx( &(acadoWorkspace.Q1[ 25 ]), acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multGxGx( &(acadoWorkspace.Q1[ 50 ]), &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.QGx[ 25 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 75 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.QGx[ 50 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.QGx[ 75 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.QGx[ 100 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.QGx[ 125 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.QGx[ 150 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.QGx[ 175 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.QGx[ 200 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.QGx[ 225 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.QGx[ 250 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.QGx[ 275 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.QGx[ 300 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.QGx[ 325 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.QGx[ 350 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.QGx[ 375 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.QGx[ 400 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.QGx[ 425 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.QGx[ 450 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.QGx[ 475 ]) );

acado_multGxGu( &(acadoWorkspace.Q1[ 25 ]), acadoWorkspace.E, acadoWorkspace.QE );
acado_multGxGu( &(acadoWorkspace.Q1[ 50 ]), &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 50 ]), &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 75 ]), &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 45 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 75 ]), &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 75 ]), &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 165 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 315 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 675 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 705 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 825 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 855 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.E[ 975 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1005 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1035 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.E[ 1155 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1170 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1185 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1215 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.E[ 1350 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1365 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1380 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1395 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1410 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1575 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1590 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1605 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1620 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.E[ 1785 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1800 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1815 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1830 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1845 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.E[ 2025 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2040 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2055 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2070 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2085 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.E[ 2280 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2295 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2310 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2325 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2340 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.E[ 2550 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2565 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2580 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2595 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2610 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.E[ 2835 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2850 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2865 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2880 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2895 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2910 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2925 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 2940 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 2955 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 2970 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 2985 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3000 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3015 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3030 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3045 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3060 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3075 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QE[ 3090 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.QE[ 3105 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3120 ]), &(acadoWorkspace.QE[ 3120 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 3135 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_zeroBlockH00(  );
acado_multCTQC( acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multCTQC( &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.QGx[ 25 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.QGx[ 50 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.QGx[ 75 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.QGx[ 100 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.QGx[ 125 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.QGx[ 150 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.QGx[ 175 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.QGx[ 200 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.QGx[ 225 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.QGx[ 250 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.QGx[ 275 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.QGx[ 300 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.QGx[ 325 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.QGx[ 350 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.QGx[ 375 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.QGx[ 400 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.QGx[ 425 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.QGx[ 450 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.QGx[ 475 ]) );

acado_zeroBlockH10( acadoWorkspace.H10 );
acado_multQETGx( acadoWorkspace.QE, acadoWorkspace.evGx, acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 15 ]), &(acadoWorkspace.evGx[ 25 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 45 ]), &(acadoWorkspace.evGx[ 50 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.evGx[ 75 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.evGx[ 100 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 225 ]), &(acadoWorkspace.evGx[ 125 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 315 ]), &(acadoWorkspace.evGx[ 150 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.evGx[ 175 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.evGx[ 200 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 675 ]), &(acadoWorkspace.evGx[ 225 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 825 ]), &(acadoWorkspace.evGx[ 250 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 990 ]), &(acadoWorkspace.evGx[ 275 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1170 ]), &(acadoWorkspace.evGx[ 300 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1365 ]), &(acadoWorkspace.evGx[ 325 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1575 ]), &(acadoWorkspace.evGx[ 350 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1800 ]), &(acadoWorkspace.evGx[ 375 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 2040 ]), &(acadoWorkspace.evGx[ 400 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 2295 ]), &(acadoWorkspace.evGx[ 425 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 2565 ]), &(acadoWorkspace.evGx[ 450 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 2850 ]), &(acadoWorkspace.evGx[ 475 ]), acadoWorkspace.H10 );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 105 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 165 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 435 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 555 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 690 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1005 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1185 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1380 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1590 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1815 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2055 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2310 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2580 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2865 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 75 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 255 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 345 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 705 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 855 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1020 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1395 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1605 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1830 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2070 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2325 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2595 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2880 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 135 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 195 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 465 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 585 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 870 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1035 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1215 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1410 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1620 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1845 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2085 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2340 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2610 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2895 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 285 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 375 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 735 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 885 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1050 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1230 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1425 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1635 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1860 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2100 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2355 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2625 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2910 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 495 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 615 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 750 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 900 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1065 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1245 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1440 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1650 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1875 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2115 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2370 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2640 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2925 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 75 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 405 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 630 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 765 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 915 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1260 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1455 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1665 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1890 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2130 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2385 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2655 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2940 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 525 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 645 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 780 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 930 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1095 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1275 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1470 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1680 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1905 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2145 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2400 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2670 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2955 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 105 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 660 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 795 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 945 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1110 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1290 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1485 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1695 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1920 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2160 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2415 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2685 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2970 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 810 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1125 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1305 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1500 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1710 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1935 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2175 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2430 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2700 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2985 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 135 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 975 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1140 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1320 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1515 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1725 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1950 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2190 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2445 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2715 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3000 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 150 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1155 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1335 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1530 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1740 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1965 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2205 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2460 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2730 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3015 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 165 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1350 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1545 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1755 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1980 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2220 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2475 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2745 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3030 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 180 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1560 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1770 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1995 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2235 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2490 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2760 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3045 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 195 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1785 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2010 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2250 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2505 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2775 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3060 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 210 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 225 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2025 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.H10[ 225 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2265 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 225 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2520 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 225 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2790 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 225 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3075 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 225 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 240 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2280 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.H10[ 240 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2535 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 240 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2805 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 240 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3090 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 240 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 255 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2550 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.H10[ 255 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2820 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 255 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3105 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 255 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 270 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 2835 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.H10[ 270 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3120 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 270 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 285 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 3135 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.H10[ 285 ]) );

for (lRun2 = 0;lRun2 < 5; ++lRun2)
for (lRun3 = 0;lRun3 < 60; ++lRun3)
acadoWorkspace.H[(lRun2 * 65) + (lRun3 + 5)] = acadoWorkspace.H10[(lRun3 * 5) + (lRun2)];

acado_setBlockH11_R1( 0, 0, acadoWorkspace.R1 );
acado_setBlockH11( 0, 0, acadoWorkspace.E, acadoWorkspace.QE );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 45 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 315 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 675 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 825 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1170 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1365 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1575 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1800 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2040 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2295 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2565 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2850 ]) );

acado_zeroBlockH11( 0, 1 );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 165 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1005 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1185 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1380 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1590 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1815 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2055 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2310 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2580 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2865 ]) );

acado_zeroBlockH11( 0, 2 );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 705 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 855 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1395 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1605 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1830 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2070 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2325 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2595 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2880 ]) );

acado_zeroBlockH11( 0, 3 );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1035 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1215 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1410 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1620 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1845 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2085 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2340 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2610 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2895 ]) );

acado_zeroBlockH11( 0, 4 );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2910 ]) );

acado_zeroBlockH11( 0, 5 );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 0, 6 );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 0, 7 );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 0, 8 );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 0, 9 );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 0, 10 );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 0, 11 );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 0, 12 );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 0, 13 );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 0, 14 );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 0, 15 );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 0, 16 );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 0, 17 );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 0, 18 );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 0, 19 );
acado_setBlockH11( 0, 19, &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 1, 1, &(acadoWorkspace.R1[ 9 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 165 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1005 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1185 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1380 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1590 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1815 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2055 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2310 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2580 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2865 ]) );

acado_zeroBlockH11( 1, 2 );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 705 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 855 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1395 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1605 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1830 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2070 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2325 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2595 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2880 ]) );

acado_zeroBlockH11( 1, 3 );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1035 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1215 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1410 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1620 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1845 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2085 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2340 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2610 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2895 ]) );

acado_zeroBlockH11( 1, 4 );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2910 ]) );

acado_zeroBlockH11( 1, 5 );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 1, 6 );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 1, 7 );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 1, 8 );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 1, 9 );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 1, 10 );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 1, 11 );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 1, 12 );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 1, 13 );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 1, 14 );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 1, 15 );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 1, 16 );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 1, 17 );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 1, 18 );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 1, 19 );
acado_setBlockH11( 1, 19, &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 2, 2, &(acadoWorkspace.R1[ 18 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 705 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 855 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1395 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1605 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1830 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2070 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2325 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2595 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2880 ]) );

acado_zeroBlockH11( 2, 3 );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1035 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1215 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1410 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1620 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1845 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2085 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2340 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2610 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2895 ]) );

acado_zeroBlockH11( 2, 4 );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2910 ]) );

acado_zeroBlockH11( 2, 5 );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 2, 6 );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 2, 7 );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 2, 8 );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 2, 9 );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 2, 10 );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 2, 11 );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 2, 12 );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 2, 13 );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 2, 14 );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 2, 15 );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 2, 16 );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 2, 17 );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 2, 18 );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 2, 19 );
acado_setBlockH11( 2, 19, &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 3, 3, &(acadoWorkspace.R1[ 27 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1035 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1215 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1410 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1620 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1845 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2085 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2340 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2610 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2895 ]) );

acado_zeroBlockH11( 3, 4 );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2910 ]) );

acado_zeroBlockH11( 3, 5 );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 3, 6 );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 3, 7 );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 3, 8 );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 3, 9 );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 3, 10 );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 3, 11 );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 3, 12 );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 3, 13 );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 3, 14 );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 3, 15 );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 3, 16 );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 3, 17 );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 3, 18 );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 3, 19 );
acado_setBlockH11( 3, 19, &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 4, 4, &(acadoWorkspace.R1[ 36 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 735 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 885 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1425 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1635 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1860 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2100 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2355 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2625 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2910 ]) );

acado_zeroBlockH11( 4, 5 );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 4, 6 );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 4, 7 );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 4, 8 );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 4, 9 );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 4, 10 );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 4, 11 );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 4, 12 );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 4, 13 );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 4, 14 );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 4, 15 );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 4, 16 );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 4, 17 );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 4, 18 );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 4, 19 );
acado_setBlockH11( 4, 19, &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 5, 5, &(acadoWorkspace.R1[ 45 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1065 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1245 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1650 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1875 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2115 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2370 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2640 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2925 ]) );

acado_zeroBlockH11( 5, 6 );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 5, 7 );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 5, 8 );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 5, 9 );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 5, 10 );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 5, 11 );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 5, 12 );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 5, 13 );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 5, 14 );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 5, 15 );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 5, 16 );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 5, 17 );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 5, 18 );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 5, 19 );
acado_setBlockH11( 5, 19, &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 6, 6, &(acadoWorkspace.R1[ 54 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QE[ 765 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 915 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1260 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1455 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1665 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1890 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2130 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2385 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2655 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 2940 ]) );

acado_zeroBlockH11( 6, 7 );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 6, 8 );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 6, 9 );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 6, 10 );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 6, 11 );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 6, 12 );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 6, 13 );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 6, 14 );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 6, 15 );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 6, 16 );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 6, 17 );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 6, 18 );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 6, 19 );
acado_setBlockH11( 6, 19, &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 7, 7, &(acadoWorkspace.R1[ 63 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.QE[ 645 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1095 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1275 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1470 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1680 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1905 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2145 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2400 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2670 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 2955 ]) );

acado_zeroBlockH11( 7, 8 );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 7, 9 );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 7, 10 );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 7, 11 );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 7, 12 );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 7, 13 );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 7, 14 );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 7, 15 );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 7, 16 );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 7, 17 );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 7, 18 );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 7, 19 );
acado_setBlockH11( 7, 19, &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 8, 8, &(acadoWorkspace.R1[ 72 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.QE[ 795 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.QE[ 945 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1290 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1485 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1695 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1920 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2160 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2415 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2685 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 2970 ]) );

acado_zeroBlockH11( 8, 9 );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 8, 10 );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 8, 11 );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 8, 12 );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 8, 13 );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 8, 14 );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 8, 15 );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 8, 16 );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 8, 17 );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 8, 18 );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 8, 19 );
acado_setBlockH11( 8, 19, &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 9, 9, &(acadoWorkspace.R1[ 81 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.QE[ 1125 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QE[ 1305 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1500 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1710 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1935 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2175 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2430 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2700 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 2985 ]) );

acado_zeroBlockH11( 9, 10 );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 9, 11 );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 9, 12 );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 9, 13 );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 9, 14 );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 9, 15 );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 9, 16 );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 9, 17 );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 9, 18 );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 9, 19 );
acado_setBlockH11( 9, 19, &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 10, 10, &(acadoWorkspace.R1[ 90 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 975 ]), &(acadoWorkspace.QE[ 975 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QE[ 1515 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1725 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 1950 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2190 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2445 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2715 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3000 ]) );

acado_zeroBlockH11( 10, 11 );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 10, 12 );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 10, 13 );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 10, 14 );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 10, 15 );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 10, 16 );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 10, 17 );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 10, 18 );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 10, 19 );
acado_setBlockH11( 10, 19, &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 11, 11, &(acadoWorkspace.R1[ 99 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1155 ]), &(acadoWorkspace.QE[ 1155 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.QE[ 1335 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.QE[ 1530 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QE[ 1740 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 1965 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2205 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2460 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2730 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3015 ]) );

acado_zeroBlockH11( 11, 12 );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 11, 13 );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 11, 14 );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 11, 15 );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 11, 16 );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 11, 17 );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 11, 18 );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 11, 19 );
acado_setBlockH11( 11, 19, &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 12, 12, &(acadoWorkspace.R1[ 108 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1350 ]), &(acadoWorkspace.QE[ 1350 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.QE[ 1545 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.QE[ 1755 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QE[ 1980 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2220 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2475 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2745 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3030 ]) );

acado_zeroBlockH11( 12, 13 );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 12, 14 );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 12, 15 );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 12, 16 );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 12, 17 );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 12, 18 );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 12, 19 );
acado_setBlockH11( 12, 19, &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 13, 13, &(acadoWorkspace.R1[ 117 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.QE[ 1770 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.QE[ 1995 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QE[ 2235 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2490 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2760 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3045 ]) );

acado_zeroBlockH11( 13, 14 );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 13, 15 );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 13, 16 );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 13, 17 );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 13, 18 );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 13, 19 );
acado_setBlockH11( 13, 19, &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 14, 14, &(acadoWorkspace.R1[ 126 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1785 ]), &(acadoWorkspace.QE[ 1785 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.QE[ 2010 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.QE[ 2250 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QE[ 2505 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2775 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3060 ]) );

acado_zeroBlockH11( 14, 15 );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 14, 16 );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 14, 17 );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 14, 18 );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 14, 19 );
acado_setBlockH11( 14, 19, &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 15, 15, &(acadoWorkspace.R1[ 135 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 2025 ]), &(acadoWorkspace.QE[ 2025 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.QE[ 2265 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.QE[ 2520 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QE[ 2790 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3075 ]) );

acado_zeroBlockH11( 15, 16 );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 15, 17 );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 15, 18 );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 15, 19 );
acado_setBlockH11( 15, 19, &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 16, 16, &(acadoWorkspace.R1[ 144 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 2280 ]), &(acadoWorkspace.QE[ 2280 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.QE[ 2535 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.QE[ 2805 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QE[ 3090 ]) );

acado_zeroBlockH11( 16, 17 );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 16, 18 );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 16, 19 );
acado_setBlockH11( 16, 19, &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 17, 17, &(acadoWorkspace.R1[ 153 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 2550 ]), &(acadoWorkspace.QE[ 2550 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.QE[ 2820 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.QE[ 3105 ]) );

acado_zeroBlockH11( 17, 18 );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 17, 19 );
acado_setBlockH11( 17, 19, &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 18, 18, &(acadoWorkspace.R1[ 162 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 2835 ]), &(acadoWorkspace.QE[ 2835 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 3120 ]), &(acadoWorkspace.QE[ 3120 ]) );

acado_zeroBlockH11( 18, 19 );
acado_setBlockH11( 18, 19, &(acadoWorkspace.E[ 3120 ]), &(acadoWorkspace.QE[ 3135 ]) );

acado_setBlockH11_R1( 19, 19, &(acadoWorkspace.R1[ 171 ]) );
acado_setBlockH11( 19, 19, &(acadoWorkspace.E[ 3135 ]), &(acadoWorkspace.QE[ 3135 ]) );


acado_copyHTH( 1, 0 );
acado_copyHTH( 2, 0 );
acado_copyHTH( 2, 1 );
acado_copyHTH( 3, 0 );
acado_copyHTH( 3, 1 );
acado_copyHTH( 3, 2 );
acado_copyHTH( 4, 0 );
acado_copyHTH( 4, 1 );
acado_copyHTH( 4, 2 );
acado_copyHTH( 4, 3 );
acado_copyHTH( 5, 0 );
acado_copyHTH( 5, 1 );
acado_copyHTH( 5, 2 );
acado_copyHTH( 5, 3 );
acado_copyHTH( 5, 4 );
acado_copyHTH( 6, 0 );
acado_copyHTH( 6, 1 );
acado_copyHTH( 6, 2 );
acado_copyHTH( 6, 3 );
acado_copyHTH( 6, 4 );
acado_copyHTH( 6, 5 );
acado_copyHTH( 7, 0 );
acado_copyHTH( 7, 1 );
acado_copyHTH( 7, 2 );
acado_copyHTH( 7, 3 );
acado_copyHTH( 7, 4 );
acado_copyHTH( 7, 5 );
acado_copyHTH( 7, 6 );
acado_copyHTH( 8, 0 );
acado_copyHTH( 8, 1 );
acado_copyHTH( 8, 2 );
acado_copyHTH( 8, 3 );
acado_copyHTH( 8, 4 );
acado_copyHTH( 8, 5 );
acado_copyHTH( 8, 6 );
acado_copyHTH( 8, 7 );
acado_copyHTH( 9, 0 );
acado_copyHTH( 9, 1 );
acado_copyHTH( 9, 2 );
acado_copyHTH( 9, 3 );
acado_copyHTH( 9, 4 );
acado_copyHTH( 9, 5 );
acado_copyHTH( 9, 6 );
acado_copyHTH( 9, 7 );
acado_copyHTH( 9, 8 );
acado_copyHTH( 10, 0 );
acado_copyHTH( 10, 1 );
acado_copyHTH( 10, 2 );
acado_copyHTH( 10, 3 );
acado_copyHTH( 10, 4 );
acado_copyHTH( 10, 5 );
acado_copyHTH( 10, 6 );
acado_copyHTH( 10, 7 );
acado_copyHTH( 10, 8 );
acado_copyHTH( 10, 9 );
acado_copyHTH( 11, 0 );
acado_copyHTH( 11, 1 );
acado_copyHTH( 11, 2 );
acado_copyHTH( 11, 3 );
acado_copyHTH( 11, 4 );
acado_copyHTH( 11, 5 );
acado_copyHTH( 11, 6 );
acado_copyHTH( 11, 7 );
acado_copyHTH( 11, 8 );
acado_copyHTH( 11, 9 );
acado_copyHTH( 11, 10 );
acado_copyHTH( 12, 0 );
acado_copyHTH( 12, 1 );
acado_copyHTH( 12, 2 );
acado_copyHTH( 12, 3 );
acado_copyHTH( 12, 4 );
acado_copyHTH( 12, 5 );
acado_copyHTH( 12, 6 );
acado_copyHTH( 12, 7 );
acado_copyHTH( 12, 8 );
acado_copyHTH( 12, 9 );
acado_copyHTH( 12, 10 );
acado_copyHTH( 12, 11 );
acado_copyHTH( 13, 0 );
acado_copyHTH( 13, 1 );
acado_copyHTH( 13, 2 );
acado_copyHTH( 13, 3 );
acado_copyHTH( 13, 4 );
acado_copyHTH( 13, 5 );
acado_copyHTH( 13, 6 );
acado_copyHTH( 13, 7 );
acado_copyHTH( 13, 8 );
acado_copyHTH( 13, 9 );
acado_copyHTH( 13, 10 );
acado_copyHTH( 13, 11 );
acado_copyHTH( 13, 12 );
acado_copyHTH( 14, 0 );
acado_copyHTH( 14, 1 );
acado_copyHTH( 14, 2 );
acado_copyHTH( 14, 3 );
acado_copyHTH( 14, 4 );
acado_copyHTH( 14, 5 );
acado_copyHTH( 14, 6 );
acado_copyHTH( 14, 7 );
acado_copyHTH( 14, 8 );
acado_copyHTH( 14, 9 );
acado_copyHTH( 14, 10 );
acado_copyHTH( 14, 11 );
acado_copyHTH( 14, 12 );
acado_copyHTH( 14, 13 );
acado_copyHTH( 15, 0 );
acado_copyHTH( 15, 1 );
acado_copyHTH( 15, 2 );
acado_copyHTH( 15, 3 );
acado_copyHTH( 15, 4 );
acado_copyHTH( 15, 5 );
acado_copyHTH( 15, 6 );
acado_copyHTH( 15, 7 );
acado_copyHTH( 15, 8 );
acado_copyHTH( 15, 9 );
acado_copyHTH( 15, 10 );
acado_copyHTH( 15, 11 );
acado_copyHTH( 15, 12 );
acado_copyHTH( 15, 13 );
acado_copyHTH( 15, 14 );
acado_copyHTH( 16, 0 );
acado_copyHTH( 16, 1 );
acado_copyHTH( 16, 2 );
acado_copyHTH( 16, 3 );
acado_copyHTH( 16, 4 );
acado_copyHTH( 16, 5 );
acado_copyHTH( 16, 6 );
acado_copyHTH( 16, 7 );
acado_copyHTH( 16, 8 );
acado_copyHTH( 16, 9 );
acado_copyHTH( 16, 10 );
acado_copyHTH( 16, 11 );
acado_copyHTH( 16, 12 );
acado_copyHTH( 16, 13 );
acado_copyHTH( 16, 14 );
acado_copyHTH( 16, 15 );
acado_copyHTH( 17, 0 );
acado_copyHTH( 17, 1 );
acado_copyHTH( 17, 2 );
acado_copyHTH( 17, 3 );
acado_copyHTH( 17, 4 );
acado_copyHTH( 17, 5 );
acado_copyHTH( 17, 6 );
acado_copyHTH( 17, 7 );
acado_copyHTH( 17, 8 );
acado_copyHTH( 17, 9 );
acado_copyHTH( 17, 10 );
acado_copyHTH( 17, 11 );
acado_copyHTH( 17, 12 );
acado_copyHTH( 17, 13 );
acado_copyHTH( 17, 14 );
acado_copyHTH( 17, 15 );
acado_copyHTH( 17, 16 );
acado_copyHTH( 18, 0 );
acado_copyHTH( 18, 1 );
acado_copyHTH( 18, 2 );
acado_copyHTH( 18, 3 );
acado_copyHTH( 18, 4 );
acado_copyHTH( 18, 5 );
acado_copyHTH( 18, 6 );
acado_copyHTH( 18, 7 );
acado_copyHTH( 18, 8 );
acado_copyHTH( 18, 9 );
acado_copyHTH( 18, 10 );
acado_copyHTH( 18, 11 );
acado_copyHTH( 18, 12 );
acado_copyHTH( 18, 13 );
acado_copyHTH( 18, 14 );
acado_copyHTH( 18, 15 );
acado_copyHTH( 18, 16 );
acado_copyHTH( 18, 17 );
acado_copyHTH( 19, 0 );
acado_copyHTH( 19, 1 );
acado_copyHTH( 19, 2 );
acado_copyHTH( 19, 3 );
acado_copyHTH( 19, 4 );
acado_copyHTH( 19, 5 );
acado_copyHTH( 19, 6 );
acado_copyHTH( 19, 7 );
acado_copyHTH( 19, 8 );
acado_copyHTH( 19, 9 );
acado_copyHTH( 19, 10 );
acado_copyHTH( 19, 11 );
acado_copyHTH( 19, 12 );
acado_copyHTH( 19, 13 );
acado_copyHTH( 19, 14 );
acado_copyHTH( 19, 15 );
acado_copyHTH( 19, 16 );
acado_copyHTH( 19, 17 );
acado_copyHTH( 19, 18 );

for (lRun2 = 0;lRun2 < 60; ++lRun2)
for (lRun3 = 0;lRun3 < 5; ++lRun3)
acadoWorkspace.H[(lRun2 * 65 + 325) + (lRun3)] = acadoWorkspace.H10[(lRun2 * 5) + (lRun3)];

acado_multQ1d( &(acadoWorkspace.Q1[ 25 ]), acadoWorkspace.d, acadoWorkspace.Qd );
acado_multQ1d( &(acadoWorkspace.Q1[ 50 ]), &(acadoWorkspace.d[ 5 ]), &(acadoWorkspace.Qd[ 5 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 75 ]), &(acadoWorkspace.d[ 10 ]), &(acadoWorkspace.Qd[ 10 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 100 ]), &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.Qd[ 15 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 125 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.Qd[ 20 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 150 ]), &(acadoWorkspace.d[ 25 ]), &(acadoWorkspace.Qd[ 25 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 175 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.Qd[ 30 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 200 ]), &(acadoWorkspace.d[ 35 ]), &(acadoWorkspace.Qd[ 35 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.Qd[ 40 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 250 ]), &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.Qd[ 45 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 275 ]), &(acadoWorkspace.d[ 50 ]), &(acadoWorkspace.Qd[ 50 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 300 ]), &(acadoWorkspace.d[ 55 ]), &(acadoWorkspace.Qd[ 55 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 325 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.Qd[ 60 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 350 ]), &(acadoWorkspace.d[ 65 ]), &(acadoWorkspace.Qd[ 65 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 375 ]), &(acadoWorkspace.d[ 70 ]), &(acadoWorkspace.Qd[ 70 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.Qd[ 75 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 425 ]), &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.Qd[ 80 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 450 ]), &(acadoWorkspace.d[ 85 ]), &(acadoWorkspace.Qd[ 85 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 475 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.Qd[ 90 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 95 ]), &(acadoWorkspace.Qd[ 95 ]) );

acado_macCTSlx( acadoWorkspace.evGx, acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 25 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 50 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 75 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 100 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 125 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 150 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 175 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 200 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 225 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 250 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 275 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 300 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 325 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 350 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 375 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 400 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 425 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 450 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 475 ]), acadoWorkspace.g );
acado_macETSlu( acadoWorkspace.QE, &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 15 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 45 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 225 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 315 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 675 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 825 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 990 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1170 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1365 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1575 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1800 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2040 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2295 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2565 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2850 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 105 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 165 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 435 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 555 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 690 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1005 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1185 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1380 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1590 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1815 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2055 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2310 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2580 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2865 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 75 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 255 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 345 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 705 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 855 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1020 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1395 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1605 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1830 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2070 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2325 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2595 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2880 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 135 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 195 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 465 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 585 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 870 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1035 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1215 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1410 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1620 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1845 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2085 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2340 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2610 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2895 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 285 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 375 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 735 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 885 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1050 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1230 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1425 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1635 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1860 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2100 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2355 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2625 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2910 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 495 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 615 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 750 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 900 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1065 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1245 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1440 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1650 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1875 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2115 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2370 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2640 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2925 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 405 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 630 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 765 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 915 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1260 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1455 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1665 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1890 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2130 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2385 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2655 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2940 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 525 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 645 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 780 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 930 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1095 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1275 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1470 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1680 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1905 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2145 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2400 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2670 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2955 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 660 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 795 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 945 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1110 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1290 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1485 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1695 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1920 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2160 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2415 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2685 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2970 ]), &(acadoWorkspace.g[ 29 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 810 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1125 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1305 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1500 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1710 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1935 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2175 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2430 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2700 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2985 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 975 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1140 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1320 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1515 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1725 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1950 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2190 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2445 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2715 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3000 ]), &(acadoWorkspace.g[ 35 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1155 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1335 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1530 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1740 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1965 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2205 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2460 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2730 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3015 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1350 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1545 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1755 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1980 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2220 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2475 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2745 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3030 ]), &(acadoWorkspace.g[ 41 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1560 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1770 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1995 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2235 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2490 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2760 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3045 ]), &(acadoWorkspace.g[ 44 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1785 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2010 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2250 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2505 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2775 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3060 ]), &(acadoWorkspace.g[ 47 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2025 ]), &(acadoWorkspace.g[ 50 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2265 ]), &(acadoWorkspace.g[ 50 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2520 ]), &(acadoWorkspace.g[ 50 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2790 ]), &(acadoWorkspace.g[ 50 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3075 ]), &(acadoWorkspace.g[ 50 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2280 ]), &(acadoWorkspace.g[ 53 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2535 ]), &(acadoWorkspace.g[ 53 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2805 ]), &(acadoWorkspace.g[ 53 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3090 ]), &(acadoWorkspace.g[ 53 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2550 ]), &(acadoWorkspace.g[ 56 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2820 ]), &(acadoWorkspace.g[ 56 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3105 ]), &(acadoWorkspace.g[ 56 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 2835 ]), &(acadoWorkspace.g[ 59 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3120 ]), &(acadoWorkspace.g[ 59 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3135 ]), &(acadoWorkspace.g[ 62 ]) );
acadoWorkspace.lb[5] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.lb[6] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.lb[10] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.lb[11] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.lb[12] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.lb[13] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.lb[14] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.lb[15] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.lb[16] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.lb[17] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.lb[18] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.lb[19] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.lb[20] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.lb[21] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.lb[22] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.lb[23] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.lb[24] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.lb[25] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.lb[26] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.lb[27] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.lb[28] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.lb[29] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.lb[30] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.lb[31] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.lb[32] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.lb[33] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.lb[34] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.lb[35] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.lb[36] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[31];
acadoWorkspace.lb[37] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[32];
acadoWorkspace.lb[38] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[33];
acadoWorkspace.lb[39] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[34];
acadoWorkspace.lb[40] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[35];
acadoWorkspace.lb[41] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[36];
acadoWorkspace.lb[42] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[37];
acadoWorkspace.lb[43] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[38];
acadoWorkspace.lb[44] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[39];
acadoWorkspace.lb[45] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[40];
acadoWorkspace.lb[46] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[41];
acadoWorkspace.lb[47] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[42];
acadoWorkspace.lb[48] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[43];
acadoWorkspace.lb[49] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[44];
acadoWorkspace.lb[50] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[45];
acadoWorkspace.lb[51] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[46];
acadoWorkspace.lb[52] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[47];
acadoWorkspace.lb[53] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[48];
acadoWorkspace.lb[54] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[49];
acadoWorkspace.lb[55] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[50];
acadoWorkspace.lb[56] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[51];
acadoWorkspace.lb[57] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[52];
acadoWorkspace.lb[58] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[53];
acadoWorkspace.lb[59] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[54];
acadoWorkspace.lb[60] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[55];
acadoWorkspace.lb[61] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[56];
acadoWorkspace.lb[62] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[57];
acadoWorkspace.lb[63] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[58];
acadoWorkspace.lb[64] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[59];
acadoWorkspace.ub[5] = (real_t)1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.ub[6] = (real_t)1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.ub[7] = (real_t)1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.ub[8] = (real_t)1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.ub[9] = (real_t)1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.ub[10] = (real_t)1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.ub[11] = (real_t)1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.ub[12] = (real_t)1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.ub[13] = (real_t)1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.ub[14] = (real_t)1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.ub[15] = (real_t)1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.ub[16] = (real_t)1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.ub[17] = (real_t)1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.ub[18] = (real_t)1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.ub[19] = (real_t)1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.ub[20] = (real_t)1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.ub[21] = (real_t)1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.ub[22] = (real_t)1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.ub[23] = (real_t)1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.ub[24] = (real_t)1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.ub[25] = (real_t)1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.ub[26] = (real_t)1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.ub[27] = (real_t)1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.ub[28] = (real_t)1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.ub[29] = (real_t)1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.ub[30] = (real_t)1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.ub[31] = (real_t)1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.ub[32] = (real_t)1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.ub[33] = (real_t)1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.ub[34] = (real_t)1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.ub[35] = (real_t)1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.ub[36] = (real_t)1.0000000000000000e+12 - acadoVariables.u[31];
acadoWorkspace.ub[37] = (real_t)1.0000000000000000e+12 - acadoVariables.u[32];
acadoWorkspace.ub[38] = (real_t)1.0000000000000000e+12 - acadoVariables.u[33];
acadoWorkspace.ub[39] = (real_t)1.0000000000000000e+12 - acadoVariables.u[34];
acadoWorkspace.ub[40] = (real_t)1.0000000000000000e+12 - acadoVariables.u[35];
acadoWorkspace.ub[41] = (real_t)1.0000000000000000e+12 - acadoVariables.u[36];
acadoWorkspace.ub[42] = (real_t)1.0000000000000000e+12 - acadoVariables.u[37];
acadoWorkspace.ub[43] = (real_t)1.0000000000000000e+12 - acadoVariables.u[38];
acadoWorkspace.ub[44] = (real_t)1.0000000000000000e+12 - acadoVariables.u[39];
acadoWorkspace.ub[45] = (real_t)1.0000000000000000e+12 - acadoVariables.u[40];
acadoWorkspace.ub[46] = (real_t)1.0000000000000000e+12 - acadoVariables.u[41];
acadoWorkspace.ub[47] = (real_t)1.0000000000000000e+12 - acadoVariables.u[42];
acadoWorkspace.ub[48] = (real_t)1.0000000000000000e+12 - acadoVariables.u[43];
acadoWorkspace.ub[49] = (real_t)1.0000000000000000e+12 - acadoVariables.u[44];
acadoWorkspace.ub[50] = (real_t)1.0000000000000000e+12 - acadoVariables.u[45];
acadoWorkspace.ub[51] = (real_t)1.0000000000000000e+12 - acadoVariables.u[46];
acadoWorkspace.ub[52] = (real_t)1.0000000000000000e+12 - acadoVariables.u[47];
acadoWorkspace.ub[53] = (real_t)1.0000000000000000e+12 - acadoVariables.u[48];
acadoWorkspace.ub[54] = (real_t)1.0000000000000000e+12 - acadoVariables.u[49];
acadoWorkspace.ub[55] = (real_t)1.0000000000000000e+12 - acadoVariables.u[50];
acadoWorkspace.ub[56] = (real_t)1.0000000000000000e+12 - acadoVariables.u[51];
acadoWorkspace.ub[57] = (real_t)1.0000000000000000e+12 - acadoVariables.u[52];
acadoWorkspace.ub[58] = (real_t)1.0000000000000000e+12 - acadoVariables.u[53];
acadoWorkspace.ub[59] = (real_t)1.0000000000000000e+12 - acadoVariables.u[54];
acadoWorkspace.ub[60] = (real_t)1.0000000000000000e+12 - acadoVariables.u[55];
acadoWorkspace.ub[61] = (real_t)1.0000000000000000e+12 - acadoVariables.u[56];
acadoWorkspace.ub[62] = (real_t)1.0000000000000000e+12 - acadoVariables.u[57];
acadoWorkspace.ub[63] = (real_t)1.0000000000000000e+12 - acadoVariables.u[58];
acadoWorkspace.ub[64] = (real_t)1.0000000000000000e+12 - acadoVariables.u[59];

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 5;
lRun4 = ((lRun3) / (5)) + (1);
acadoWorkspace.A[lRun1 * 65] = acadoWorkspace.evGx[lRun3 * 5];
acadoWorkspace.A[lRun1 * 65 + 1] = acadoWorkspace.evGx[lRun3 * 5 + 1];
acadoWorkspace.A[lRun1 * 65 + 2] = acadoWorkspace.evGx[lRun3 * 5 + 2];
acadoWorkspace.A[lRun1 * 65 + 3] = acadoWorkspace.evGx[lRun3 * 5 + 3];
acadoWorkspace.A[lRun1 * 65 + 4] = acadoWorkspace.evGx[lRun3 * 5 + 4];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (5)) + ((lRun3) % (5));
acadoWorkspace.A[(lRun1 * 65) + (lRun2 * 3 + 5)] = acadoWorkspace.E[lRun5 * 3];
acadoWorkspace.A[(lRun1 * 65) + (lRun2 * 3 + 6)] = acadoWorkspace.E[lRun5 * 3 + 1];
acadoWorkspace.A[(lRun1 * 65) + (lRun2 * 3 + 7)] = acadoWorkspace.E[lRun5 * 3 + 2];
}
}

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
acadoWorkspace.conValueIn[0] = acadoVariables.x[lRun1 * 5];
acadoWorkspace.conValueIn[1] = acadoVariables.x[lRun1 * 5 + 1];
acadoWorkspace.conValueIn[2] = acadoVariables.x[lRun1 * 5 + 2];
acadoWorkspace.conValueIn[3] = acadoVariables.x[lRun1 * 5 + 3];
acadoWorkspace.conValueIn[4] = acadoVariables.x[lRun1 * 5 + 4];
acadoWorkspace.conValueIn[5] = acadoVariables.u[lRun1 * 3];
acadoWorkspace.conValueIn[6] = acadoVariables.u[lRun1 * 3 + 1];
acadoWorkspace.conValueIn[7] = acadoVariables.u[lRun1 * 3 + 2];
acadoWorkspace.conValueIn[8] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.conValueIn[9] = acadoVariables.od[lRun1 * 2 + 1];
acado_evaluatePathConstraints( acadoWorkspace.conValueIn, acadoWorkspace.conValueOut );
acadoWorkspace.evH[lRun1 * 2] = acadoWorkspace.conValueOut[0];
acadoWorkspace.evH[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[1];

acadoWorkspace.evHx[lRun1 * 10] = acadoWorkspace.conValueOut[2];
acadoWorkspace.evHx[lRun1 * 10 + 1] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 10 + 2] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHx[lRun1 * 10 + 3] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHx[lRun1 * 10 + 4] = acadoWorkspace.conValueOut[6];
acadoWorkspace.evHx[lRun1 * 10 + 5] = acadoWorkspace.conValueOut[7];
acadoWorkspace.evHx[lRun1 * 10 + 6] = acadoWorkspace.conValueOut[8];
acadoWorkspace.evHx[lRun1 * 10 + 7] = acadoWorkspace.conValueOut[9];
acadoWorkspace.evHx[lRun1 * 10 + 8] = acadoWorkspace.conValueOut[10];
acadoWorkspace.evHx[lRun1 * 10 + 9] = acadoWorkspace.conValueOut[11];
acadoWorkspace.evHu[lRun1 * 6] = acadoWorkspace.conValueOut[12];
acadoWorkspace.evHu[lRun1 * 6 + 1] = acadoWorkspace.conValueOut[13];
acadoWorkspace.evHu[lRun1 * 6 + 2] = acadoWorkspace.conValueOut[14];
acadoWorkspace.evHu[lRun1 * 6 + 3] = acadoWorkspace.conValueOut[15];
acadoWorkspace.evHu[lRun1 * 6 + 4] = acadoWorkspace.conValueOut[16];
acadoWorkspace.evHu[lRun1 * 6 + 5] = acadoWorkspace.conValueOut[17];
}

acadoWorkspace.A[1300] = acadoWorkspace.evHx[0];
acadoWorkspace.A[1301] = acadoWorkspace.evHx[1];
acadoWorkspace.A[1302] = acadoWorkspace.evHx[2];
acadoWorkspace.A[1303] = acadoWorkspace.evHx[3];
acadoWorkspace.A[1304] = acadoWorkspace.evHx[4];
acadoWorkspace.A[1365] = acadoWorkspace.evHx[5];
acadoWorkspace.A[1366] = acadoWorkspace.evHx[6];
acadoWorkspace.A[1367] = acadoWorkspace.evHx[7];
acadoWorkspace.A[1368] = acadoWorkspace.evHx[8];
acadoWorkspace.A[1369] = acadoWorkspace.evHx[9];

acado_multHxC( &(acadoWorkspace.evHx[ 10 ]), acadoWorkspace.evGx, &(acadoWorkspace.A[ 1430 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.A[ 1560 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.A[ 1690 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.A[ 1820 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.A[ 1950 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.A[ 2080 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.A[ 2210 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.A[ 2340 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.A[ 2470 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.A[ 2600 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.A[ 2730 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.A[ 2860 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.A[ 2990 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.A[ 3120 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.A[ 3250 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.A[ 3380 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.A[ 3510 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.A[ 3640 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.A[ 3770 ]) );

acado_multHxE( &(acadoWorkspace.evHx[ 10 ]), acadoWorkspace.E, 1, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 15 ]), 2, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 30 ]), 2, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.E[ 45 ]), 3, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.E[ 60 ]), 3, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.E[ 75 ]), 3, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 90 ]), 4, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 105 ]), 4, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 120 ]), 4, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 135 ]), 4, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.E[ 150 ]), 5, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.E[ 165 ]), 5, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.E[ 180 ]), 5, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.E[ 195 ]), 5, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.E[ 210 ]), 5, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 225 ]), 6, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 240 ]), 6, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 255 ]), 6, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 270 ]), 6, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 285 ]), 6, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 300 ]), 6, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 315 ]), 7, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 330 ]), 7, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 345 ]), 7, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 360 ]), 7, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 375 ]), 7, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 390 ]), 7, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.E[ 405 ]), 7, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 420 ]), 8, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 435 ]), 8, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 450 ]), 8, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 465 ]), 8, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 480 ]), 8, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 495 ]), 8, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 510 ]), 8, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.E[ 525 ]), 8, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 540 ]), 9, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 555 ]), 9, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 570 ]), 9, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 585 ]), 9, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 600 ]), 9, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 615 ]), 9, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 630 ]), 9, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 645 ]), 9, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.E[ 660 ]), 9, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 675 ]), 10, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 690 ]), 10, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 705 ]), 10, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 720 ]), 10, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 735 ]), 10, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 750 ]), 10, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 765 ]), 10, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 780 ]), 10, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 795 ]), 10, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.E[ 810 ]), 10, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 825 ]), 11, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 840 ]), 11, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 855 ]), 11, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 870 ]), 11, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 885 ]), 11, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 900 ]), 11, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 915 ]), 11, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 930 ]), 11, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 945 ]), 11, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 960 ]), 11, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.E[ 975 ]), 11, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 990 ]), 12, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1005 ]), 12, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1020 ]), 12, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1035 ]), 12, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1050 ]), 12, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1065 ]), 12, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1080 ]), 12, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1095 ]), 12, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1110 ]), 12, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1125 ]), 12, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1140 ]), 12, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.E[ 1155 ]), 12, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1170 ]), 13, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1185 ]), 13, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1200 ]), 13, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1215 ]), 13, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1230 ]), 13, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1245 ]), 13, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1260 ]), 13, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1275 ]), 13, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1290 ]), 13, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1305 ]), 13, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1320 ]), 13, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1335 ]), 13, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.E[ 1350 ]), 13, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1365 ]), 14, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1380 ]), 14, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1395 ]), 14, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1410 ]), 14, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1425 ]), 14, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1440 ]), 14, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1455 ]), 14, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1470 ]), 14, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1485 ]), 14, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1500 ]), 14, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1515 ]), 14, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1530 ]), 14, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1545 ]), 14, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.E[ 1560 ]), 14, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1575 ]), 15, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1590 ]), 15, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1605 ]), 15, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1620 ]), 15, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1635 ]), 15, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1650 ]), 15, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1665 ]), 15, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1680 ]), 15, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1695 ]), 15, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1710 ]), 15, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1725 ]), 15, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1740 ]), 15, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1755 ]), 15, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1770 ]), 15, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.E[ 1785 ]), 15, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1800 ]), 16, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1815 ]), 16, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1830 ]), 16, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1845 ]), 16, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1860 ]), 16, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1875 ]), 16, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1890 ]), 16, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1905 ]), 16, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1920 ]), 16, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1935 ]), 16, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1950 ]), 16, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1965 ]), 16, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1980 ]), 16, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 1995 ]), 16, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 2010 ]), 16, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.E[ 2025 ]), 16, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2040 ]), 17, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2055 ]), 17, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2070 ]), 17, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2085 ]), 17, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2100 ]), 17, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2115 ]), 17, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2130 ]), 17, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2145 ]), 17, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2160 ]), 17, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2175 ]), 17, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2190 ]), 17, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2205 ]), 17, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2220 ]), 17, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2235 ]), 17, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2250 ]), 17, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2265 ]), 17, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.E[ 2280 ]), 17, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2295 ]), 18, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2310 ]), 18, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2325 ]), 18, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2340 ]), 18, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2355 ]), 18, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2370 ]), 18, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2385 ]), 18, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2400 ]), 18, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2415 ]), 18, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2430 ]), 18, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2445 ]), 18, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2460 ]), 18, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2475 ]), 18, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2490 ]), 18, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2505 ]), 18, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2520 ]), 18, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2535 ]), 18, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.E[ 2550 ]), 18, 17 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2565 ]), 19, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2580 ]), 19, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2595 ]), 19, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2610 ]), 19, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2625 ]), 19, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2640 ]), 19, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2655 ]), 19, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2670 ]), 19, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2685 ]), 19, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2700 ]), 19, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2715 ]), 19, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2730 ]), 19, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2745 ]), 19, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2760 ]), 19, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2775 ]), 19, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2790 ]), 19, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2805 ]), 19, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2820 ]), 19, 17 );
acado_multHxE( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.E[ 2835 ]), 19, 18 );

acadoWorkspace.A[1305] = acadoWorkspace.evHu[0];
acadoWorkspace.A[1306] = acadoWorkspace.evHu[1];
acadoWorkspace.A[1307] = acadoWorkspace.evHu[2];
acadoWorkspace.A[1370] = acadoWorkspace.evHu[3];
acadoWorkspace.A[1371] = acadoWorkspace.evHu[4];
acadoWorkspace.A[1372] = acadoWorkspace.evHu[5];
acadoWorkspace.A[1438] = acadoWorkspace.evHu[6];
acadoWorkspace.A[1439] = acadoWorkspace.evHu[7];
acadoWorkspace.A[1440] = acadoWorkspace.evHu[8];
acadoWorkspace.A[1503] = acadoWorkspace.evHu[9];
acadoWorkspace.A[1504] = acadoWorkspace.evHu[10];
acadoWorkspace.A[1505] = acadoWorkspace.evHu[11];
acadoWorkspace.A[1571] = acadoWorkspace.evHu[12];
acadoWorkspace.A[1572] = acadoWorkspace.evHu[13];
acadoWorkspace.A[1573] = acadoWorkspace.evHu[14];
acadoWorkspace.A[1636] = acadoWorkspace.evHu[15];
acadoWorkspace.A[1637] = acadoWorkspace.evHu[16];
acadoWorkspace.A[1638] = acadoWorkspace.evHu[17];
acadoWorkspace.A[1704] = acadoWorkspace.evHu[18];
acadoWorkspace.A[1705] = acadoWorkspace.evHu[19];
acadoWorkspace.A[1706] = acadoWorkspace.evHu[20];
acadoWorkspace.A[1769] = acadoWorkspace.evHu[21];
acadoWorkspace.A[1770] = acadoWorkspace.evHu[22];
acadoWorkspace.A[1771] = acadoWorkspace.evHu[23];
acadoWorkspace.A[1837] = acadoWorkspace.evHu[24];
acadoWorkspace.A[1838] = acadoWorkspace.evHu[25];
acadoWorkspace.A[1839] = acadoWorkspace.evHu[26];
acadoWorkspace.A[1902] = acadoWorkspace.evHu[27];
acadoWorkspace.A[1903] = acadoWorkspace.evHu[28];
acadoWorkspace.A[1904] = acadoWorkspace.evHu[29];
acadoWorkspace.A[1970] = acadoWorkspace.evHu[30];
acadoWorkspace.A[1971] = acadoWorkspace.evHu[31];
acadoWorkspace.A[1972] = acadoWorkspace.evHu[32];
acadoWorkspace.A[2035] = acadoWorkspace.evHu[33];
acadoWorkspace.A[2036] = acadoWorkspace.evHu[34];
acadoWorkspace.A[2037] = acadoWorkspace.evHu[35];
acadoWorkspace.A[2103] = acadoWorkspace.evHu[36];
acadoWorkspace.A[2104] = acadoWorkspace.evHu[37];
acadoWorkspace.A[2105] = acadoWorkspace.evHu[38];
acadoWorkspace.A[2168] = acadoWorkspace.evHu[39];
acadoWorkspace.A[2169] = acadoWorkspace.evHu[40];
acadoWorkspace.A[2170] = acadoWorkspace.evHu[41];
acadoWorkspace.A[2236] = acadoWorkspace.evHu[42];
acadoWorkspace.A[2237] = acadoWorkspace.evHu[43];
acadoWorkspace.A[2238] = acadoWorkspace.evHu[44];
acadoWorkspace.A[2301] = acadoWorkspace.evHu[45];
acadoWorkspace.A[2302] = acadoWorkspace.evHu[46];
acadoWorkspace.A[2303] = acadoWorkspace.evHu[47];
acadoWorkspace.A[2369] = acadoWorkspace.evHu[48];
acadoWorkspace.A[2370] = acadoWorkspace.evHu[49];
acadoWorkspace.A[2371] = acadoWorkspace.evHu[50];
acadoWorkspace.A[2434] = acadoWorkspace.evHu[51];
acadoWorkspace.A[2435] = acadoWorkspace.evHu[52];
acadoWorkspace.A[2436] = acadoWorkspace.evHu[53];
acadoWorkspace.A[2502] = acadoWorkspace.evHu[54];
acadoWorkspace.A[2503] = acadoWorkspace.evHu[55];
acadoWorkspace.A[2504] = acadoWorkspace.evHu[56];
acadoWorkspace.A[2567] = acadoWorkspace.evHu[57];
acadoWorkspace.A[2568] = acadoWorkspace.evHu[58];
acadoWorkspace.A[2569] = acadoWorkspace.evHu[59];
acadoWorkspace.A[2635] = acadoWorkspace.evHu[60];
acadoWorkspace.A[2636] = acadoWorkspace.evHu[61];
acadoWorkspace.A[2637] = acadoWorkspace.evHu[62];
acadoWorkspace.A[2700] = acadoWorkspace.evHu[63];
acadoWorkspace.A[2701] = acadoWorkspace.evHu[64];
acadoWorkspace.A[2702] = acadoWorkspace.evHu[65];
acadoWorkspace.A[2768] = acadoWorkspace.evHu[66];
acadoWorkspace.A[2769] = acadoWorkspace.evHu[67];
acadoWorkspace.A[2770] = acadoWorkspace.evHu[68];
acadoWorkspace.A[2833] = acadoWorkspace.evHu[69];
acadoWorkspace.A[2834] = acadoWorkspace.evHu[70];
acadoWorkspace.A[2835] = acadoWorkspace.evHu[71];
acadoWorkspace.A[2901] = acadoWorkspace.evHu[72];
acadoWorkspace.A[2902] = acadoWorkspace.evHu[73];
acadoWorkspace.A[2903] = acadoWorkspace.evHu[74];
acadoWorkspace.A[2966] = acadoWorkspace.evHu[75];
acadoWorkspace.A[2967] = acadoWorkspace.evHu[76];
acadoWorkspace.A[2968] = acadoWorkspace.evHu[77];
acadoWorkspace.A[3034] = acadoWorkspace.evHu[78];
acadoWorkspace.A[3035] = acadoWorkspace.evHu[79];
acadoWorkspace.A[3036] = acadoWorkspace.evHu[80];
acadoWorkspace.A[3099] = acadoWorkspace.evHu[81];
acadoWorkspace.A[3100] = acadoWorkspace.evHu[82];
acadoWorkspace.A[3101] = acadoWorkspace.evHu[83];
acadoWorkspace.A[3167] = acadoWorkspace.evHu[84];
acadoWorkspace.A[3168] = acadoWorkspace.evHu[85];
acadoWorkspace.A[3169] = acadoWorkspace.evHu[86];
acadoWorkspace.A[3232] = acadoWorkspace.evHu[87];
acadoWorkspace.A[3233] = acadoWorkspace.evHu[88];
acadoWorkspace.A[3234] = acadoWorkspace.evHu[89];
acadoWorkspace.A[3300] = acadoWorkspace.evHu[90];
acadoWorkspace.A[3301] = acadoWorkspace.evHu[91];
acadoWorkspace.A[3302] = acadoWorkspace.evHu[92];
acadoWorkspace.A[3365] = acadoWorkspace.evHu[93];
acadoWorkspace.A[3366] = acadoWorkspace.evHu[94];
acadoWorkspace.A[3367] = acadoWorkspace.evHu[95];
acadoWorkspace.A[3433] = acadoWorkspace.evHu[96];
acadoWorkspace.A[3434] = acadoWorkspace.evHu[97];
acadoWorkspace.A[3435] = acadoWorkspace.evHu[98];
acadoWorkspace.A[3498] = acadoWorkspace.evHu[99];
acadoWorkspace.A[3499] = acadoWorkspace.evHu[100];
acadoWorkspace.A[3500] = acadoWorkspace.evHu[101];
acadoWorkspace.A[3566] = acadoWorkspace.evHu[102];
acadoWorkspace.A[3567] = acadoWorkspace.evHu[103];
acadoWorkspace.A[3568] = acadoWorkspace.evHu[104];
acadoWorkspace.A[3631] = acadoWorkspace.evHu[105];
acadoWorkspace.A[3632] = acadoWorkspace.evHu[106];
acadoWorkspace.A[3633] = acadoWorkspace.evHu[107];
acadoWorkspace.A[3699] = acadoWorkspace.evHu[108];
acadoWorkspace.A[3700] = acadoWorkspace.evHu[109];
acadoWorkspace.A[3701] = acadoWorkspace.evHu[110];
acadoWorkspace.A[3764] = acadoWorkspace.evHu[111];
acadoWorkspace.A[3765] = acadoWorkspace.evHu[112];
acadoWorkspace.A[3766] = acadoWorkspace.evHu[113];
acadoWorkspace.A[3832] = acadoWorkspace.evHu[114];
acadoWorkspace.A[3833] = acadoWorkspace.evHu[115];
acadoWorkspace.A[3834] = acadoWorkspace.evHu[116];
acadoWorkspace.A[3897] = acadoWorkspace.evHu[117];
acadoWorkspace.A[3898] = acadoWorkspace.evHu[118];
acadoWorkspace.A[3899] = acadoWorkspace.evHu[119];
acadoWorkspace.lbA[20] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[0];
acadoWorkspace.lbA[21] = - acadoWorkspace.evH[1];
acadoWorkspace.lbA[22] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[2];
acadoWorkspace.lbA[23] = - acadoWorkspace.evH[3];
acadoWorkspace.lbA[24] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[4];
acadoWorkspace.lbA[25] = - acadoWorkspace.evH[5];
acadoWorkspace.lbA[26] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[6];
acadoWorkspace.lbA[27] = - acadoWorkspace.evH[7];
acadoWorkspace.lbA[28] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[8];
acadoWorkspace.lbA[29] = - acadoWorkspace.evH[9];
acadoWorkspace.lbA[30] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[10];
acadoWorkspace.lbA[31] = - acadoWorkspace.evH[11];
acadoWorkspace.lbA[32] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[12];
acadoWorkspace.lbA[33] = - acadoWorkspace.evH[13];
acadoWorkspace.lbA[34] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[14];
acadoWorkspace.lbA[35] = - acadoWorkspace.evH[15];
acadoWorkspace.lbA[36] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[16];
acadoWorkspace.lbA[37] = - acadoWorkspace.evH[17];
acadoWorkspace.lbA[38] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[18];
acadoWorkspace.lbA[39] = - acadoWorkspace.evH[19];
acadoWorkspace.lbA[40] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[20];
acadoWorkspace.lbA[41] = - acadoWorkspace.evH[21];
acadoWorkspace.lbA[42] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[22];
acadoWorkspace.lbA[43] = - acadoWorkspace.evH[23];
acadoWorkspace.lbA[44] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[24];
acadoWorkspace.lbA[45] = - acadoWorkspace.evH[25];
acadoWorkspace.lbA[46] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[26];
acadoWorkspace.lbA[47] = - acadoWorkspace.evH[27];
acadoWorkspace.lbA[48] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[28];
acadoWorkspace.lbA[49] = - acadoWorkspace.evH[29];
acadoWorkspace.lbA[50] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[30];
acadoWorkspace.lbA[51] = - acadoWorkspace.evH[31];
acadoWorkspace.lbA[52] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[32];
acadoWorkspace.lbA[53] = - acadoWorkspace.evH[33];
acadoWorkspace.lbA[54] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[34];
acadoWorkspace.lbA[55] = - acadoWorkspace.evH[35];
acadoWorkspace.lbA[56] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[36];
acadoWorkspace.lbA[57] = - acadoWorkspace.evH[37];
acadoWorkspace.lbA[58] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[38];
acadoWorkspace.lbA[59] = - acadoWorkspace.evH[39];

acadoWorkspace.ubA[20] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[0];
acadoWorkspace.ubA[21] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[1];
acadoWorkspace.ubA[22] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[2];
acadoWorkspace.ubA[23] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[3];
acadoWorkspace.ubA[24] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[4];
acadoWorkspace.ubA[25] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[5];
acadoWorkspace.ubA[26] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[6];
acadoWorkspace.ubA[27] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[7];
acadoWorkspace.ubA[28] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[8];
acadoWorkspace.ubA[29] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[9];
acadoWorkspace.ubA[30] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[10];
acadoWorkspace.ubA[31] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[11];
acadoWorkspace.ubA[32] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[12];
acadoWorkspace.ubA[33] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[13];
acadoWorkspace.ubA[34] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[14];
acadoWorkspace.ubA[35] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[15];
acadoWorkspace.ubA[36] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[16];
acadoWorkspace.ubA[37] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[17];
acadoWorkspace.ubA[38] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[18];
acadoWorkspace.ubA[39] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[19];
acadoWorkspace.ubA[40] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[20];
acadoWorkspace.ubA[41] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[21];
acadoWorkspace.ubA[42] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[22];
acadoWorkspace.ubA[43] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[23];
acadoWorkspace.ubA[44] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[24];
acadoWorkspace.ubA[45] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[25];
acadoWorkspace.ubA[46] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[26];
acadoWorkspace.ubA[47] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[27];
acadoWorkspace.ubA[48] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[28];
acadoWorkspace.ubA[49] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[29];
acadoWorkspace.ubA[50] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[30];
acadoWorkspace.ubA[51] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[31];
acadoWorkspace.ubA[52] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[32];
acadoWorkspace.ubA[53] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[33];
acadoWorkspace.ubA[54] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[34];
acadoWorkspace.ubA[55] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[35];
acadoWorkspace.ubA[56] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[36];
acadoWorkspace.ubA[57] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[37];
acadoWorkspace.ubA[58] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[38];
acadoWorkspace.ubA[59] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[39];

acado_macHxd( &(acadoWorkspace.evHx[ 10 ]), acadoWorkspace.d, &(acadoWorkspace.lbA[ 22 ]), &(acadoWorkspace.ubA[ 22 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.d[ 5 ]), &(acadoWorkspace.lbA[ 24 ]), &(acadoWorkspace.ubA[ 24 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.d[ 10 ]), &(acadoWorkspace.lbA[ 26 ]), &(acadoWorkspace.ubA[ 26 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.lbA[ 28 ]), &(acadoWorkspace.ubA[ 28 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 50 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.lbA[ 30 ]), &(acadoWorkspace.ubA[ 30 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.d[ 25 ]), &(acadoWorkspace.lbA[ 32 ]), &(acadoWorkspace.ubA[ 32 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 70 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.lbA[ 34 ]), &(acadoWorkspace.ubA[ 34 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.d[ 35 ]), &(acadoWorkspace.lbA[ 36 ]), &(acadoWorkspace.ubA[ 36 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.lbA[ 38 ]), &(acadoWorkspace.ubA[ 38 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 100 ]), &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.lbA[ 40 ]), &(acadoWorkspace.ubA[ 40 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 110 ]), &(acadoWorkspace.d[ 50 ]), &(acadoWorkspace.lbA[ 42 ]), &(acadoWorkspace.ubA[ 42 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.d[ 55 ]), &(acadoWorkspace.lbA[ 44 ]), &(acadoWorkspace.ubA[ 44 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 130 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.lbA[ 46 ]), &(acadoWorkspace.ubA[ 46 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 140 ]), &(acadoWorkspace.d[ 65 ]), &(acadoWorkspace.lbA[ 48 ]), &(acadoWorkspace.ubA[ 48 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.d[ 70 ]), &(acadoWorkspace.lbA[ 50 ]), &(acadoWorkspace.ubA[ 50 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.lbA[ 52 ]), &(acadoWorkspace.ubA[ 52 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 170 ]), &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.lbA[ 54 ]), &(acadoWorkspace.ubA[ 54 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.d[ 85 ]), &(acadoWorkspace.lbA[ 56 ]), &(acadoWorkspace.ubA[ 56 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 190 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.lbA[ 58 ]), &(acadoWorkspace.ubA[ 58 ]) );

}

void acado_condenseFdb(  )
{
real_t tmp;

acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];
acadoWorkspace.Dx0[4] = acadoVariables.x0[4] - acadoVariables.x[4];

acadoWorkspace.Dy[0] -= acadoVariables.y[0];
acadoWorkspace.Dy[1] -= acadoVariables.y[1];
acadoWorkspace.Dy[2] -= acadoVariables.y[2];
acadoWorkspace.Dy[3] -= acadoVariables.y[3];
acadoWorkspace.Dy[4] -= acadoVariables.y[4];
acadoWorkspace.Dy[5] -= acadoVariables.y[5];
acadoWorkspace.Dy[6] -= acadoVariables.y[6];
acadoWorkspace.Dy[7] -= acadoVariables.y[7];
acadoWorkspace.Dy[8] -= acadoVariables.y[8];
acadoWorkspace.Dy[9] -= acadoVariables.y[9];
acadoWorkspace.Dy[10] -= acadoVariables.y[10];
acadoWorkspace.Dy[11] -= acadoVariables.y[11];
acadoWorkspace.Dy[12] -= acadoVariables.y[12];
acadoWorkspace.Dy[13] -= acadoVariables.y[13];
acadoWorkspace.Dy[14] -= acadoVariables.y[14];
acadoWorkspace.Dy[15] -= acadoVariables.y[15];
acadoWorkspace.Dy[16] -= acadoVariables.y[16];
acadoWorkspace.Dy[17] -= acadoVariables.y[17];
acadoWorkspace.Dy[18] -= acadoVariables.y[18];
acadoWorkspace.Dy[19] -= acadoVariables.y[19];
acadoWorkspace.Dy[20] -= acadoVariables.y[20];
acadoWorkspace.Dy[21] -= acadoVariables.y[21];
acadoWorkspace.Dy[22] -= acadoVariables.y[22];
acadoWorkspace.Dy[23] -= acadoVariables.y[23];
acadoWorkspace.Dy[24] -= acadoVariables.y[24];
acadoWorkspace.Dy[25] -= acadoVariables.y[25];
acadoWorkspace.Dy[26] -= acadoVariables.y[26];
acadoWorkspace.Dy[27] -= acadoVariables.y[27];
acadoWorkspace.Dy[28] -= acadoVariables.y[28];
acadoWorkspace.Dy[29] -= acadoVariables.y[29];
acadoWorkspace.Dy[30] -= acadoVariables.y[30];
acadoWorkspace.Dy[31] -= acadoVariables.y[31];
acadoWorkspace.Dy[32] -= acadoVariables.y[32];
acadoWorkspace.Dy[33] -= acadoVariables.y[33];
acadoWorkspace.Dy[34] -= acadoVariables.y[34];
acadoWorkspace.Dy[35] -= acadoVariables.y[35];
acadoWorkspace.Dy[36] -= acadoVariables.y[36];
acadoWorkspace.Dy[37] -= acadoVariables.y[37];
acadoWorkspace.Dy[38] -= acadoVariables.y[38];
acadoWorkspace.Dy[39] -= acadoVariables.y[39];
acadoWorkspace.Dy[40] -= acadoVariables.y[40];
acadoWorkspace.Dy[41] -= acadoVariables.y[41];
acadoWorkspace.Dy[42] -= acadoVariables.y[42];
acadoWorkspace.Dy[43] -= acadoVariables.y[43];
acadoWorkspace.Dy[44] -= acadoVariables.y[44];
acadoWorkspace.Dy[45] -= acadoVariables.y[45];
acadoWorkspace.Dy[46] -= acadoVariables.y[46];
acadoWorkspace.Dy[47] -= acadoVariables.y[47];
acadoWorkspace.Dy[48] -= acadoVariables.y[48];
acadoWorkspace.Dy[49] -= acadoVariables.y[49];
acadoWorkspace.Dy[50] -= acadoVariables.y[50];
acadoWorkspace.Dy[51] -= acadoVariables.y[51];
acadoWorkspace.Dy[52] -= acadoVariables.y[52];
acadoWorkspace.Dy[53] -= acadoVariables.y[53];
acadoWorkspace.Dy[54] -= acadoVariables.y[54];
acadoWorkspace.Dy[55] -= acadoVariables.y[55];
acadoWorkspace.Dy[56] -= acadoVariables.y[56];
acadoWorkspace.Dy[57] -= acadoVariables.y[57];
acadoWorkspace.Dy[58] -= acadoVariables.y[58];
acadoWorkspace.Dy[59] -= acadoVariables.y[59];
acadoWorkspace.Dy[60] -= acadoVariables.y[60];
acadoWorkspace.Dy[61] -= acadoVariables.y[61];
acadoWorkspace.Dy[62] -= acadoVariables.y[62];
acadoWorkspace.Dy[63] -= acadoVariables.y[63];
acadoWorkspace.Dy[64] -= acadoVariables.y[64];
acadoWorkspace.Dy[65] -= acadoVariables.y[65];
acadoWorkspace.Dy[66] -= acadoVariables.y[66];
acadoWorkspace.Dy[67] -= acadoVariables.y[67];
acadoWorkspace.Dy[68] -= acadoVariables.y[68];
acadoWorkspace.Dy[69] -= acadoVariables.y[69];
acadoWorkspace.Dy[70] -= acadoVariables.y[70];
acadoWorkspace.Dy[71] -= acadoVariables.y[71];
acadoWorkspace.Dy[72] -= acadoVariables.y[72];
acadoWorkspace.Dy[73] -= acadoVariables.y[73];
acadoWorkspace.Dy[74] -= acadoVariables.y[74];
acadoWorkspace.Dy[75] -= acadoVariables.y[75];
acadoWorkspace.Dy[76] -= acadoVariables.y[76];
acadoWorkspace.Dy[77] -= acadoVariables.y[77];
acadoWorkspace.Dy[78] -= acadoVariables.y[78];
acadoWorkspace.Dy[79] -= acadoVariables.y[79];
acadoWorkspace.Dy[80] -= acadoVariables.y[80];
acadoWorkspace.Dy[81] -= acadoVariables.y[81];
acadoWorkspace.Dy[82] -= acadoVariables.y[82];
acadoWorkspace.Dy[83] -= acadoVariables.y[83];
acadoWorkspace.Dy[84] -= acadoVariables.y[84];
acadoWorkspace.Dy[85] -= acadoVariables.y[85];
acadoWorkspace.Dy[86] -= acadoVariables.y[86];
acadoWorkspace.Dy[87] -= acadoVariables.y[87];
acadoWorkspace.Dy[88] -= acadoVariables.y[88];
acadoWorkspace.Dy[89] -= acadoVariables.y[89];
acadoWorkspace.Dy[90] -= acadoVariables.y[90];
acadoWorkspace.Dy[91] -= acadoVariables.y[91];
acadoWorkspace.Dy[92] -= acadoVariables.y[92];
acadoWorkspace.Dy[93] -= acadoVariables.y[93];
acadoWorkspace.Dy[94] -= acadoVariables.y[94];
acadoWorkspace.Dy[95] -= acadoVariables.y[95];
acadoWorkspace.Dy[96] -= acadoVariables.y[96];
acadoWorkspace.Dy[97] -= acadoVariables.y[97];
acadoWorkspace.Dy[98] -= acadoVariables.y[98];
acadoWorkspace.Dy[99] -= acadoVariables.y[99];
acadoWorkspace.DyN[0] -= acadoVariables.yN[0];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 5 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 15 ]), &(acadoWorkspace.Dy[ 5 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 30 ]), &(acadoWorkspace.Dy[ 10 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 45 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 60 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 75 ]), &(acadoWorkspace.Dy[ 25 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 90 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 105 ]), &(acadoWorkspace.Dy[ 35 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 120 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 135 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 150 ]), &(acadoWorkspace.Dy[ 50 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 165 ]), &(acadoWorkspace.Dy[ 55 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 180 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 195 ]), &(acadoWorkspace.Dy[ 65 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 210 ]), &(acadoWorkspace.Dy[ 70 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 225 ]), &(acadoWorkspace.Dy[ 75 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 240 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 255 ]), &(acadoWorkspace.Dy[ 85 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 270 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.g[ 59 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 285 ]), &(acadoWorkspace.Dy[ 95 ]), &(acadoWorkspace.g[ 62 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 25 ]), &(acadoWorkspace.Dy[ 5 ]), &(acadoWorkspace.QDy[ 5 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 50 ]), &(acadoWorkspace.Dy[ 10 ]), &(acadoWorkspace.QDy[ 10 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 75 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.QDy[ 15 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 100 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.QDy[ 20 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 125 ]), &(acadoWorkspace.Dy[ 25 ]), &(acadoWorkspace.QDy[ 25 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 150 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.QDy[ 30 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 175 ]), &(acadoWorkspace.Dy[ 35 ]), &(acadoWorkspace.QDy[ 35 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 200 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.QDy[ 40 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 225 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.QDy[ 45 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 250 ]), &(acadoWorkspace.Dy[ 50 ]), &(acadoWorkspace.QDy[ 50 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 275 ]), &(acadoWorkspace.Dy[ 55 ]), &(acadoWorkspace.QDy[ 55 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 300 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 325 ]), &(acadoWorkspace.Dy[ 65 ]), &(acadoWorkspace.QDy[ 65 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 350 ]), &(acadoWorkspace.Dy[ 70 ]), &(acadoWorkspace.QDy[ 70 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 375 ]), &(acadoWorkspace.Dy[ 75 ]), &(acadoWorkspace.QDy[ 75 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 400 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.QDy[ 80 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 425 ]), &(acadoWorkspace.Dy[ 85 ]), &(acadoWorkspace.QDy[ 85 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 450 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.QDy[ 90 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 475 ]), &(acadoWorkspace.Dy[ 95 ]), &(acadoWorkspace.QDy[ 95 ]) );

acadoWorkspace.QDy[100] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0];
acadoWorkspace.QDy[101] = + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[0];
acadoWorkspace.QDy[102] = + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[0];
acadoWorkspace.QDy[103] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0];
acadoWorkspace.QDy[104] = + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[0];

acadoWorkspace.QDy[5] += acadoWorkspace.Qd[0];
acadoWorkspace.QDy[6] += acadoWorkspace.Qd[1];
acadoWorkspace.QDy[7] += acadoWorkspace.Qd[2];
acadoWorkspace.QDy[8] += acadoWorkspace.Qd[3];
acadoWorkspace.QDy[9] += acadoWorkspace.Qd[4];
acadoWorkspace.QDy[10] += acadoWorkspace.Qd[5];
acadoWorkspace.QDy[11] += acadoWorkspace.Qd[6];
acadoWorkspace.QDy[12] += acadoWorkspace.Qd[7];
acadoWorkspace.QDy[13] += acadoWorkspace.Qd[8];
acadoWorkspace.QDy[14] += acadoWorkspace.Qd[9];
acadoWorkspace.QDy[15] += acadoWorkspace.Qd[10];
acadoWorkspace.QDy[16] += acadoWorkspace.Qd[11];
acadoWorkspace.QDy[17] += acadoWorkspace.Qd[12];
acadoWorkspace.QDy[18] += acadoWorkspace.Qd[13];
acadoWorkspace.QDy[19] += acadoWorkspace.Qd[14];
acadoWorkspace.QDy[20] += acadoWorkspace.Qd[15];
acadoWorkspace.QDy[21] += acadoWorkspace.Qd[16];
acadoWorkspace.QDy[22] += acadoWorkspace.Qd[17];
acadoWorkspace.QDy[23] += acadoWorkspace.Qd[18];
acadoWorkspace.QDy[24] += acadoWorkspace.Qd[19];
acadoWorkspace.QDy[25] += acadoWorkspace.Qd[20];
acadoWorkspace.QDy[26] += acadoWorkspace.Qd[21];
acadoWorkspace.QDy[27] += acadoWorkspace.Qd[22];
acadoWorkspace.QDy[28] += acadoWorkspace.Qd[23];
acadoWorkspace.QDy[29] += acadoWorkspace.Qd[24];
acadoWorkspace.QDy[30] += acadoWorkspace.Qd[25];
acadoWorkspace.QDy[31] += acadoWorkspace.Qd[26];
acadoWorkspace.QDy[32] += acadoWorkspace.Qd[27];
acadoWorkspace.QDy[33] += acadoWorkspace.Qd[28];
acadoWorkspace.QDy[34] += acadoWorkspace.Qd[29];
acadoWorkspace.QDy[35] += acadoWorkspace.Qd[30];
acadoWorkspace.QDy[36] += acadoWorkspace.Qd[31];
acadoWorkspace.QDy[37] += acadoWorkspace.Qd[32];
acadoWorkspace.QDy[38] += acadoWorkspace.Qd[33];
acadoWorkspace.QDy[39] += acadoWorkspace.Qd[34];
acadoWorkspace.QDy[40] += acadoWorkspace.Qd[35];
acadoWorkspace.QDy[41] += acadoWorkspace.Qd[36];
acadoWorkspace.QDy[42] += acadoWorkspace.Qd[37];
acadoWorkspace.QDy[43] += acadoWorkspace.Qd[38];
acadoWorkspace.QDy[44] += acadoWorkspace.Qd[39];
acadoWorkspace.QDy[45] += acadoWorkspace.Qd[40];
acadoWorkspace.QDy[46] += acadoWorkspace.Qd[41];
acadoWorkspace.QDy[47] += acadoWorkspace.Qd[42];
acadoWorkspace.QDy[48] += acadoWorkspace.Qd[43];
acadoWorkspace.QDy[49] += acadoWorkspace.Qd[44];
acadoWorkspace.QDy[50] += acadoWorkspace.Qd[45];
acadoWorkspace.QDy[51] += acadoWorkspace.Qd[46];
acadoWorkspace.QDy[52] += acadoWorkspace.Qd[47];
acadoWorkspace.QDy[53] += acadoWorkspace.Qd[48];
acadoWorkspace.QDy[54] += acadoWorkspace.Qd[49];
acadoWorkspace.QDy[55] += acadoWorkspace.Qd[50];
acadoWorkspace.QDy[56] += acadoWorkspace.Qd[51];
acadoWorkspace.QDy[57] += acadoWorkspace.Qd[52];
acadoWorkspace.QDy[58] += acadoWorkspace.Qd[53];
acadoWorkspace.QDy[59] += acadoWorkspace.Qd[54];
acadoWorkspace.QDy[60] += acadoWorkspace.Qd[55];
acadoWorkspace.QDy[61] += acadoWorkspace.Qd[56];
acadoWorkspace.QDy[62] += acadoWorkspace.Qd[57];
acadoWorkspace.QDy[63] += acadoWorkspace.Qd[58];
acadoWorkspace.QDy[64] += acadoWorkspace.Qd[59];
acadoWorkspace.QDy[65] += acadoWorkspace.Qd[60];
acadoWorkspace.QDy[66] += acadoWorkspace.Qd[61];
acadoWorkspace.QDy[67] += acadoWorkspace.Qd[62];
acadoWorkspace.QDy[68] += acadoWorkspace.Qd[63];
acadoWorkspace.QDy[69] += acadoWorkspace.Qd[64];
acadoWorkspace.QDy[70] += acadoWorkspace.Qd[65];
acadoWorkspace.QDy[71] += acadoWorkspace.Qd[66];
acadoWorkspace.QDy[72] += acadoWorkspace.Qd[67];
acadoWorkspace.QDy[73] += acadoWorkspace.Qd[68];
acadoWorkspace.QDy[74] += acadoWorkspace.Qd[69];
acadoWorkspace.QDy[75] += acadoWorkspace.Qd[70];
acadoWorkspace.QDy[76] += acadoWorkspace.Qd[71];
acadoWorkspace.QDy[77] += acadoWorkspace.Qd[72];
acadoWorkspace.QDy[78] += acadoWorkspace.Qd[73];
acadoWorkspace.QDy[79] += acadoWorkspace.Qd[74];
acadoWorkspace.QDy[80] += acadoWorkspace.Qd[75];
acadoWorkspace.QDy[81] += acadoWorkspace.Qd[76];
acadoWorkspace.QDy[82] += acadoWorkspace.Qd[77];
acadoWorkspace.QDy[83] += acadoWorkspace.Qd[78];
acadoWorkspace.QDy[84] += acadoWorkspace.Qd[79];
acadoWorkspace.QDy[85] += acadoWorkspace.Qd[80];
acadoWorkspace.QDy[86] += acadoWorkspace.Qd[81];
acadoWorkspace.QDy[87] += acadoWorkspace.Qd[82];
acadoWorkspace.QDy[88] += acadoWorkspace.Qd[83];
acadoWorkspace.QDy[89] += acadoWorkspace.Qd[84];
acadoWorkspace.QDy[90] += acadoWorkspace.Qd[85];
acadoWorkspace.QDy[91] += acadoWorkspace.Qd[86];
acadoWorkspace.QDy[92] += acadoWorkspace.Qd[87];
acadoWorkspace.QDy[93] += acadoWorkspace.Qd[88];
acadoWorkspace.QDy[94] += acadoWorkspace.Qd[89];
acadoWorkspace.QDy[95] += acadoWorkspace.Qd[90];
acadoWorkspace.QDy[96] += acadoWorkspace.Qd[91];
acadoWorkspace.QDy[97] += acadoWorkspace.Qd[92];
acadoWorkspace.QDy[98] += acadoWorkspace.Qd[93];
acadoWorkspace.QDy[99] += acadoWorkspace.Qd[94];
acadoWorkspace.QDy[100] += acadoWorkspace.Qd[95];
acadoWorkspace.QDy[101] += acadoWorkspace.Qd[96];
acadoWorkspace.QDy[102] += acadoWorkspace.Qd[97];
acadoWorkspace.QDy[103] += acadoWorkspace.Qd[98];
acadoWorkspace.QDy[104] += acadoWorkspace.Qd[99];

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[320]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[325]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[330]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[335]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[340]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[345]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[350]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[355]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[360]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[365]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[370]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[375]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[380]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[385]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[390]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[395]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[400]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[405]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[410]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[415]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[420]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[425]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[430]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[435]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[440]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[445]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[450]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[455]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[460]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[465]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[470]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[475]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[480]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[485]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[490]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[495]*acadoWorkspace.QDy[104];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[321]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[326]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[331]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[336]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[341]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[346]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[351]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[356]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[361]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[366]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[371]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[376]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[381]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[386]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[391]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[396]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[401]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[406]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[411]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[416]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[421]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[426]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[431]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[436]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[441]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[446]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[451]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[456]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[461]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[466]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[471]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[476]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[481]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[486]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[491]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[496]*acadoWorkspace.QDy[104];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[322]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[327]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[332]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[337]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[342]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[347]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[352]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[357]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[362]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[367]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[372]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[377]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[382]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[387]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[392]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[397]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[402]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[407]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[412]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[417]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[422]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[427]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[432]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[437]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[442]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[447]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[452]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[457]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[462]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[467]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[472]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[477]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[482]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[487]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[492]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[497]*acadoWorkspace.QDy[104];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[323]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[328]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[333]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[338]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[343]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[348]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[353]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[358]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[363]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[368]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[373]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[378]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[383]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[388]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[393]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[398]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[403]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[408]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[413]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[418]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[423]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[428]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[433]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[438]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[443]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[448]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[453]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[458]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[463]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[468]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[473]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[478]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[483]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[488]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[493]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[498]*acadoWorkspace.QDy[104];
acadoWorkspace.g[4] = + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[324]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[329]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[334]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[339]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[344]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[349]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[354]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[359]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[364]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[369]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[374]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[379]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[384]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[389]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[394]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[399]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[404]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[409]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[414]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[419]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[424]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[429]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[434]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[439]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[444]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[449]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[454]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[459]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[464]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[469]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[474]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[479]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[484]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[489]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[494]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[499]*acadoWorkspace.QDy[104];


acado_multEQDy( acadoWorkspace.E, &(acadoWorkspace.QDy[ 5 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QDy[ 10 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QDy[ 25 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QDy[ 10 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QDy[ 25 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QDy[ 25 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QDy[ 25 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QDy[ 25 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QDy[ 35 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QDy[ 50 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 975 ]), &(acadoWorkspace.QDy[ 55 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1155 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1350 ]), &(acadoWorkspace.QDy[ 65 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QDy[ 70 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1785 ]), &(acadoWorkspace.QDy[ 75 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2025 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2280 ]), &(acadoWorkspace.QDy[ 85 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2550 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 2835 ]), &(acadoWorkspace.QDy[ 95 ]), &(acadoWorkspace.g[ 59 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3120 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 59 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3135 ]), &(acadoWorkspace.QDy[ 100 ]), &(acadoWorkspace.g[ 62 ]) );

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.lb[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.lb[4] = acadoWorkspace.Dx0[4];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.ub[4] = acadoWorkspace.Dx0[4];
tmp = acadoVariables.x[6] + acadoWorkspace.d[1];
acadoWorkspace.lbA[0] = - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[11] + acadoWorkspace.d[6];
acadoWorkspace.lbA[1] = - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[16] + acadoWorkspace.d[11];
acadoWorkspace.lbA[2] = - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[21] + acadoWorkspace.d[16];
acadoWorkspace.lbA[3] = - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[26] + acadoWorkspace.d[21];
acadoWorkspace.lbA[4] = - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[26];
acadoWorkspace.lbA[5] = - tmp;
acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[36] + acadoWorkspace.d[31];
acadoWorkspace.lbA[6] = - tmp;
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[41] + acadoWorkspace.d[36];
acadoWorkspace.lbA[7] = - tmp;
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[46] + acadoWorkspace.d[41];
acadoWorkspace.lbA[8] = - tmp;
acadoWorkspace.ubA[8] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[51] + acadoWorkspace.d[46];
acadoWorkspace.lbA[9] = - tmp;
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[56] + acadoWorkspace.d[51];
acadoWorkspace.lbA[10] = - tmp;
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[61] + acadoWorkspace.d[56];
acadoWorkspace.lbA[11] = - tmp;
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[66] + acadoWorkspace.d[61];
acadoWorkspace.lbA[12] = - tmp;
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[71] + acadoWorkspace.d[66];
acadoWorkspace.lbA[13] = - tmp;
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[76] + acadoWorkspace.d[71];
acadoWorkspace.lbA[14] = - tmp;
acadoWorkspace.ubA[14] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[81] + acadoWorkspace.d[76];
acadoWorkspace.lbA[15] = - tmp;
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[86] + acadoWorkspace.d[81];
acadoWorkspace.lbA[16] = - tmp;
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[91] + acadoWorkspace.d[86];
acadoWorkspace.lbA[17] = - tmp;
acadoWorkspace.ubA[17] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[96] + acadoWorkspace.d[91];
acadoWorkspace.lbA[18] = - tmp;
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[101] + acadoWorkspace.d[96];
acadoWorkspace.lbA[19] = - tmp;
acadoWorkspace.ubA[19] = (real_t)1.0000000000000000e+12 - tmp;

}

void acado_expand(  )
{
acadoVariables.x[0] += acadoWorkspace.x[0];
acadoVariables.x[1] += acadoWorkspace.x[1];
acadoVariables.x[2] += acadoWorkspace.x[2];
acadoVariables.x[3] += acadoWorkspace.x[3];
acadoVariables.x[4] += acadoWorkspace.x[4];

acadoVariables.u[0] += acadoWorkspace.x[5];
acadoVariables.u[1] += acadoWorkspace.x[6];
acadoVariables.u[2] += acadoWorkspace.x[7];
acadoVariables.u[3] += acadoWorkspace.x[8];
acadoVariables.u[4] += acadoWorkspace.x[9];
acadoVariables.u[5] += acadoWorkspace.x[10];
acadoVariables.u[6] += acadoWorkspace.x[11];
acadoVariables.u[7] += acadoWorkspace.x[12];
acadoVariables.u[8] += acadoWorkspace.x[13];
acadoVariables.u[9] += acadoWorkspace.x[14];
acadoVariables.u[10] += acadoWorkspace.x[15];
acadoVariables.u[11] += acadoWorkspace.x[16];
acadoVariables.u[12] += acadoWorkspace.x[17];
acadoVariables.u[13] += acadoWorkspace.x[18];
acadoVariables.u[14] += acadoWorkspace.x[19];
acadoVariables.u[15] += acadoWorkspace.x[20];
acadoVariables.u[16] += acadoWorkspace.x[21];
acadoVariables.u[17] += acadoWorkspace.x[22];
acadoVariables.u[18] += acadoWorkspace.x[23];
acadoVariables.u[19] += acadoWorkspace.x[24];
acadoVariables.u[20] += acadoWorkspace.x[25];
acadoVariables.u[21] += acadoWorkspace.x[26];
acadoVariables.u[22] += acadoWorkspace.x[27];
acadoVariables.u[23] += acadoWorkspace.x[28];
acadoVariables.u[24] += acadoWorkspace.x[29];
acadoVariables.u[25] += acadoWorkspace.x[30];
acadoVariables.u[26] += acadoWorkspace.x[31];
acadoVariables.u[27] += acadoWorkspace.x[32];
acadoVariables.u[28] += acadoWorkspace.x[33];
acadoVariables.u[29] += acadoWorkspace.x[34];
acadoVariables.u[30] += acadoWorkspace.x[35];
acadoVariables.u[31] += acadoWorkspace.x[36];
acadoVariables.u[32] += acadoWorkspace.x[37];
acadoVariables.u[33] += acadoWorkspace.x[38];
acadoVariables.u[34] += acadoWorkspace.x[39];
acadoVariables.u[35] += acadoWorkspace.x[40];
acadoVariables.u[36] += acadoWorkspace.x[41];
acadoVariables.u[37] += acadoWorkspace.x[42];
acadoVariables.u[38] += acadoWorkspace.x[43];
acadoVariables.u[39] += acadoWorkspace.x[44];
acadoVariables.u[40] += acadoWorkspace.x[45];
acadoVariables.u[41] += acadoWorkspace.x[46];
acadoVariables.u[42] += acadoWorkspace.x[47];
acadoVariables.u[43] += acadoWorkspace.x[48];
acadoVariables.u[44] += acadoWorkspace.x[49];
acadoVariables.u[45] += acadoWorkspace.x[50];
acadoVariables.u[46] += acadoWorkspace.x[51];
acadoVariables.u[47] += acadoWorkspace.x[52];
acadoVariables.u[48] += acadoWorkspace.x[53];
acadoVariables.u[49] += acadoWorkspace.x[54];
acadoVariables.u[50] += acadoWorkspace.x[55];
acadoVariables.u[51] += acadoWorkspace.x[56];
acadoVariables.u[52] += acadoWorkspace.x[57];
acadoVariables.u[53] += acadoWorkspace.x[58];
acadoVariables.u[54] += acadoWorkspace.x[59];
acadoVariables.u[55] += acadoWorkspace.x[60];
acadoVariables.u[56] += acadoWorkspace.x[61];
acadoVariables.u[57] += acadoWorkspace.x[62];
acadoVariables.u[58] += acadoWorkspace.x[63];
acadoVariables.u[59] += acadoWorkspace.x[64];

acadoVariables.x[5] += + acadoWorkspace.evGx[0]*acadoWorkspace.x[0] + acadoWorkspace.evGx[1]*acadoWorkspace.x[1] + acadoWorkspace.evGx[2]*acadoWorkspace.x[2] + acadoWorkspace.evGx[3]*acadoWorkspace.x[3] + acadoWorkspace.evGx[4]*acadoWorkspace.x[4] + acadoWorkspace.d[0];
acadoVariables.x[6] += + acadoWorkspace.evGx[5]*acadoWorkspace.x[0] + acadoWorkspace.evGx[6]*acadoWorkspace.x[1] + acadoWorkspace.evGx[7]*acadoWorkspace.x[2] + acadoWorkspace.evGx[8]*acadoWorkspace.x[3] + acadoWorkspace.evGx[9]*acadoWorkspace.x[4] + acadoWorkspace.d[1];
acadoVariables.x[7] += + acadoWorkspace.evGx[10]*acadoWorkspace.x[0] + acadoWorkspace.evGx[11]*acadoWorkspace.x[1] + acadoWorkspace.evGx[12]*acadoWorkspace.x[2] + acadoWorkspace.evGx[13]*acadoWorkspace.x[3] + acadoWorkspace.evGx[14]*acadoWorkspace.x[4] + acadoWorkspace.d[2];
acadoVariables.x[8] += + acadoWorkspace.evGx[15]*acadoWorkspace.x[0] + acadoWorkspace.evGx[16]*acadoWorkspace.x[1] + acadoWorkspace.evGx[17]*acadoWorkspace.x[2] + acadoWorkspace.evGx[18]*acadoWorkspace.x[3] + acadoWorkspace.evGx[19]*acadoWorkspace.x[4] + acadoWorkspace.d[3];
acadoVariables.x[9] += + acadoWorkspace.evGx[20]*acadoWorkspace.x[0] + acadoWorkspace.evGx[21]*acadoWorkspace.x[1] + acadoWorkspace.evGx[22]*acadoWorkspace.x[2] + acadoWorkspace.evGx[23]*acadoWorkspace.x[3] + acadoWorkspace.evGx[24]*acadoWorkspace.x[4] + acadoWorkspace.d[4];
acadoVariables.x[10] += + acadoWorkspace.evGx[25]*acadoWorkspace.x[0] + acadoWorkspace.evGx[26]*acadoWorkspace.x[1] + acadoWorkspace.evGx[27]*acadoWorkspace.x[2] + acadoWorkspace.evGx[28]*acadoWorkspace.x[3] + acadoWorkspace.evGx[29]*acadoWorkspace.x[4] + acadoWorkspace.d[5];
acadoVariables.x[11] += + acadoWorkspace.evGx[30]*acadoWorkspace.x[0] + acadoWorkspace.evGx[31]*acadoWorkspace.x[1] + acadoWorkspace.evGx[32]*acadoWorkspace.x[2] + acadoWorkspace.evGx[33]*acadoWorkspace.x[3] + acadoWorkspace.evGx[34]*acadoWorkspace.x[4] + acadoWorkspace.d[6];
acadoVariables.x[12] += + acadoWorkspace.evGx[35]*acadoWorkspace.x[0] + acadoWorkspace.evGx[36]*acadoWorkspace.x[1] + acadoWorkspace.evGx[37]*acadoWorkspace.x[2] + acadoWorkspace.evGx[38]*acadoWorkspace.x[3] + acadoWorkspace.evGx[39]*acadoWorkspace.x[4] + acadoWorkspace.d[7];
acadoVariables.x[13] += + acadoWorkspace.evGx[40]*acadoWorkspace.x[0] + acadoWorkspace.evGx[41]*acadoWorkspace.x[1] + acadoWorkspace.evGx[42]*acadoWorkspace.x[2] + acadoWorkspace.evGx[43]*acadoWorkspace.x[3] + acadoWorkspace.evGx[44]*acadoWorkspace.x[4] + acadoWorkspace.d[8];
acadoVariables.x[14] += + acadoWorkspace.evGx[45]*acadoWorkspace.x[0] + acadoWorkspace.evGx[46]*acadoWorkspace.x[1] + acadoWorkspace.evGx[47]*acadoWorkspace.x[2] + acadoWorkspace.evGx[48]*acadoWorkspace.x[3] + acadoWorkspace.evGx[49]*acadoWorkspace.x[4] + acadoWorkspace.d[9];
acadoVariables.x[15] += + acadoWorkspace.evGx[50]*acadoWorkspace.x[0] + acadoWorkspace.evGx[51]*acadoWorkspace.x[1] + acadoWorkspace.evGx[52]*acadoWorkspace.x[2] + acadoWorkspace.evGx[53]*acadoWorkspace.x[3] + acadoWorkspace.evGx[54]*acadoWorkspace.x[4] + acadoWorkspace.d[10];
acadoVariables.x[16] += + acadoWorkspace.evGx[55]*acadoWorkspace.x[0] + acadoWorkspace.evGx[56]*acadoWorkspace.x[1] + acadoWorkspace.evGx[57]*acadoWorkspace.x[2] + acadoWorkspace.evGx[58]*acadoWorkspace.x[3] + acadoWorkspace.evGx[59]*acadoWorkspace.x[4] + acadoWorkspace.d[11];
acadoVariables.x[17] += + acadoWorkspace.evGx[60]*acadoWorkspace.x[0] + acadoWorkspace.evGx[61]*acadoWorkspace.x[1] + acadoWorkspace.evGx[62]*acadoWorkspace.x[2] + acadoWorkspace.evGx[63]*acadoWorkspace.x[3] + acadoWorkspace.evGx[64]*acadoWorkspace.x[4] + acadoWorkspace.d[12];
acadoVariables.x[18] += + acadoWorkspace.evGx[65]*acadoWorkspace.x[0] + acadoWorkspace.evGx[66]*acadoWorkspace.x[1] + acadoWorkspace.evGx[67]*acadoWorkspace.x[2] + acadoWorkspace.evGx[68]*acadoWorkspace.x[3] + acadoWorkspace.evGx[69]*acadoWorkspace.x[4] + acadoWorkspace.d[13];
acadoVariables.x[19] += + acadoWorkspace.evGx[70]*acadoWorkspace.x[0] + acadoWorkspace.evGx[71]*acadoWorkspace.x[1] + acadoWorkspace.evGx[72]*acadoWorkspace.x[2] + acadoWorkspace.evGx[73]*acadoWorkspace.x[3] + acadoWorkspace.evGx[74]*acadoWorkspace.x[4] + acadoWorkspace.d[14];
acadoVariables.x[20] += + acadoWorkspace.evGx[75]*acadoWorkspace.x[0] + acadoWorkspace.evGx[76]*acadoWorkspace.x[1] + acadoWorkspace.evGx[77]*acadoWorkspace.x[2] + acadoWorkspace.evGx[78]*acadoWorkspace.x[3] + acadoWorkspace.evGx[79]*acadoWorkspace.x[4] + acadoWorkspace.d[15];
acadoVariables.x[21] += + acadoWorkspace.evGx[80]*acadoWorkspace.x[0] + acadoWorkspace.evGx[81]*acadoWorkspace.x[1] + acadoWorkspace.evGx[82]*acadoWorkspace.x[2] + acadoWorkspace.evGx[83]*acadoWorkspace.x[3] + acadoWorkspace.evGx[84]*acadoWorkspace.x[4] + acadoWorkspace.d[16];
acadoVariables.x[22] += + acadoWorkspace.evGx[85]*acadoWorkspace.x[0] + acadoWorkspace.evGx[86]*acadoWorkspace.x[1] + acadoWorkspace.evGx[87]*acadoWorkspace.x[2] + acadoWorkspace.evGx[88]*acadoWorkspace.x[3] + acadoWorkspace.evGx[89]*acadoWorkspace.x[4] + acadoWorkspace.d[17];
acadoVariables.x[23] += + acadoWorkspace.evGx[90]*acadoWorkspace.x[0] + acadoWorkspace.evGx[91]*acadoWorkspace.x[1] + acadoWorkspace.evGx[92]*acadoWorkspace.x[2] + acadoWorkspace.evGx[93]*acadoWorkspace.x[3] + acadoWorkspace.evGx[94]*acadoWorkspace.x[4] + acadoWorkspace.d[18];
acadoVariables.x[24] += + acadoWorkspace.evGx[95]*acadoWorkspace.x[0] + acadoWorkspace.evGx[96]*acadoWorkspace.x[1] + acadoWorkspace.evGx[97]*acadoWorkspace.x[2] + acadoWorkspace.evGx[98]*acadoWorkspace.x[3] + acadoWorkspace.evGx[99]*acadoWorkspace.x[4] + acadoWorkspace.d[19];
acadoVariables.x[25] += + acadoWorkspace.evGx[100]*acadoWorkspace.x[0] + acadoWorkspace.evGx[101]*acadoWorkspace.x[1] + acadoWorkspace.evGx[102]*acadoWorkspace.x[2] + acadoWorkspace.evGx[103]*acadoWorkspace.x[3] + acadoWorkspace.evGx[104]*acadoWorkspace.x[4] + acadoWorkspace.d[20];
acadoVariables.x[26] += + acadoWorkspace.evGx[105]*acadoWorkspace.x[0] + acadoWorkspace.evGx[106]*acadoWorkspace.x[1] + acadoWorkspace.evGx[107]*acadoWorkspace.x[2] + acadoWorkspace.evGx[108]*acadoWorkspace.x[3] + acadoWorkspace.evGx[109]*acadoWorkspace.x[4] + acadoWorkspace.d[21];
acadoVariables.x[27] += + acadoWorkspace.evGx[110]*acadoWorkspace.x[0] + acadoWorkspace.evGx[111]*acadoWorkspace.x[1] + acadoWorkspace.evGx[112]*acadoWorkspace.x[2] + acadoWorkspace.evGx[113]*acadoWorkspace.x[3] + acadoWorkspace.evGx[114]*acadoWorkspace.x[4] + acadoWorkspace.d[22];
acadoVariables.x[28] += + acadoWorkspace.evGx[115]*acadoWorkspace.x[0] + acadoWorkspace.evGx[116]*acadoWorkspace.x[1] + acadoWorkspace.evGx[117]*acadoWorkspace.x[2] + acadoWorkspace.evGx[118]*acadoWorkspace.x[3] + acadoWorkspace.evGx[119]*acadoWorkspace.x[4] + acadoWorkspace.d[23];
acadoVariables.x[29] += + acadoWorkspace.evGx[120]*acadoWorkspace.x[0] + acadoWorkspace.evGx[121]*acadoWorkspace.x[1] + acadoWorkspace.evGx[122]*acadoWorkspace.x[2] + acadoWorkspace.evGx[123]*acadoWorkspace.x[3] + acadoWorkspace.evGx[124]*acadoWorkspace.x[4] + acadoWorkspace.d[24];
acadoVariables.x[30] += + acadoWorkspace.evGx[125]*acadoWorkspace.x[0] + acadoWorkspace.evGx[126]*acadoWorkspace.x[1] + acadoWorkspace.evGx[127]*acadoWorkspace.x[2] + acadoWorkspace.evGx[128]*acadoWorkspace.x[3] + acadoWorkspace.evGx[129]*acadoWorkspace.x[4] + acadoWorkspace.d[25];
acadoVariables.x[31] += + acadoWorkspace.evGx[130]*acadoWorkspace.x[0] + acadoWorkspace.evGx[131]*acadoWorkspace.x[1] + acadoWorkspace.evGx[132]*acadoWorkspace.x[2] + acadoWorkspace.evGx[133]*acadoWorkspace.x[3] + acadoWorkspace.evGx[134]*acadoWorkspace.x[4] + acadoWorkspace.d[26];
acadoVariables.x[32] += + acadoWorkspace.evGx[135]*acadoWorkspace.x[0] + acadoWorkspace.evGx[136]*acadoWorkspace.x[1] + acadoWorkspace.evGx[137]*acadoWorkspace.x[2] + acadoWorkspace.evGx[138]*acadoWorkspace.x[3] + acadoWorkspace.evGx[139]*acadoWorkspace.x[4] + acadoWorkspace.d[27];
acadoVariables.x[33] += + acadoWorkspace.evGx[140]*acadoWorkspace.x[0] + acadoWorkspace.evGx[141]*acadoWorkspace.x[1] + acadoWorkspace.evGx[142]*acadoWorkspace.x[2] + acadoWorkspace.evGx[143]*acadoWorkspace.x[3] + acadoWorkspace.evGx[144]*acadoWorkspace.x[4] + acadoWorkspace.d[28];
acadoVariables.x[34] += + acadoWorkspace.evGx[145]*acadoWorkspace.x[0] + acadoWorkspace.evGx[146]*acadoWorkspace.x[1] + acadoWorkspace.evGx[147]*acadoWorkspace.x[2] + acadoWorkspace.evGx[148]*acadoWorkspace.x[3] + acadoWorkspace.evGx[149]*acadoWorkspace.x[4] + acadoWorkspace.d[29];
acadoVariables.x[35] += + acadoWorkspace.evGx[150]*acadoWorkspace.x[0] + acadoWorkspace.evGx[151]*acadoWorkspace.x[1] + acadoWorkspace.evGx[152]*acadoWorkspace.x[2] + acadoWorkspace.evGx[153]*acadoWorkspace.x[3] + acadoWorkspace.evGx[154]*acadoWorkspace.x[4] + acadoWorkspace.d[30];
acadoVariables.x[36] += + acadoWorkspace.evGx[155]*acadoWorkspace.x[0] + acadoWorkspace.evGx[156]*acadoWorkspace.x[1] + acadoWorkspace.evGx[157]*acadoWorkspace.x[2] + acadoWorkspace.evGx[158]*acadoWorkspace.x[3] + acadoWorkspace.evGx[159]*acadoWorkspace.x[4] + acadoWorkspace.d[31];
acadoVariables.x[37] += + acadoWorkspace.evGx[160]*acadoWorkspace.x[0] + acadoWorkspace.evGx[161]*acadoWorkspace.x[1] + acadoWorkspace.evGx[162]*acadoWorkspace.x[2] + acadoWorkspace.evGx[163]*acadoWorkspace.x[3] + acadoWorkspace.evGx[164]*acadoWorkspace.x[4] + acadoWorkspace.d[32];
acadoVariables.x[38] += + acadoWorkspace.evGx[165]*acadoWorkspace.x[0] + acadoWorkspace.evGx[166]*acadoWorkspace.x[1] + acadoWorkspace.evGx[167]*acadoWorkspace.x[2] + acadoWorkspace.evGx[168]*acadoWorkspace.x[3] + acadoWorkspace.evGx[169]*acadoWorkspace.x[4] + acadoWorkspace.d[33];
acadoVariables.x[39] += + acadoWorkspace.evGx[170]*acadoWorkspace.x[0] + acadoWorkspace.evGx[171]*acadoWorkspace.x[1] + acadoWorkspace.evGx[172]*acadoWorkspace.x[2] + acadoWorkspace.evGx[173]*acadoWorkspace.x[3] + acadoWorkspace.evGx[174]*acadoWorkspace.x[4] + acadoWorkspace.d[34];
acadoVariables.x[40] += + acadoWorkspace.evGx[175]*acadoWorkspace.x[0] + acadoWorkspace.evGx[176]*acadoWorkspace.x[1] + acadoWorkspace.evGx[177]*acadoWorkspace.x[2] + acadoWorkspace.evGx[178]*acadoWorkspace.x[3] + acadoWorkspace.evGx[179]*acadoWorkspace.x[4] + acadoWorkspace.d[35];
acadoVariables.x[41] += + acadoWorkspace.evGx[180]*acadoWorkspace.x[0] + acadoWorkspace.evGx[181]*acadoWorkspace.x[1] + acadoWorkspace.evGx[182]*acadoWorkspace.x[2] + acadoWorkspace.evGx[183]*acadoWorkspace.x[3] + acadoWorkspace.evGx[184]*acadoWorkspace.x[4] + acadoWorkspace.d[36];
acadoVariables.x[42] += + acadoWorkspace.evGx[185]*acadoWorkspace.x[0] + acadoWorkspace.evGx[186]*acadoWorkspace.x[1] + acadoWorkspace.evGx[187]*acadoWorkspace.x[2] + acadoWorkspace.evGx[188]*acadoWorkspace.x[3] + acadoWorkspace.evGx[189]*acadoWorkspace.x[4] + acadoWorkspace.d[37];
acadoVariables.x[43] += + acadoWorkspace.evGx[190]*acadoWorkspace.x[0] + acadoWorkspace.evGx[191]*acadoWorkspace.x[1] + acadoWorkspace.evGx[192]*acadoWorkspace.x[2] + acadoWorkspace.evGx[193]*acadoWorkspace.x[3] + acadoWorkspace.evGx[194]*acadoWorkspace.x[4] + acadoWorkspace.d[38];
acadoVariables.x[44] += + acadoWorkspace.evGx[195]*acadoWorkspace.x[0] + acadoWorkspace.evGx[196]*acadoWorkspace.x[1] + acadoWorkspace.evGx[197]*acadoWorkspace.x[2] + acadoWorkspace.evGx[198]*acadoWorkspace.x[3] + acadoWorkspace.evGx[199]*acadoWorkspace.x[4] + acadoWorkspace.d[39];
acadoVariables.x[45] += + acadoWorkspace.evGx[200]*acadoWorkspace.x[0] + acadoWorkspace.evGx[201]*acadoWorkspace.x[1] + acadoWorkspace.evGx[202]*acadoWorkspace.x[2] + acadoWorkspace.evGx[203]*acadoWorkspace.x[3] + acadoWorkspace.evGx[204]*acadoWorkspace.x[4] + acadoWorkspace.d[40];
acadoVariables.x[46] += + acadoWorkspace.evGx[205]*acadoWorkspace.x[0] + acadoWorkspace.evGx[206]*acadoWorkspace.x[1] + acadoWorkspace.evGx[207]*acadoWorkspace.x[2] + acadoWorkspace.evGx[208]*acadoWorkspace.x[3] + acadoWorkspace.evGx[209]*acadoWorkspace.x[4] + acadoWorkspace.d[41];
acadoVariables.x[47] += + acadoWorkspace.evGx[210]*acadoWorkspace.x[0] + acadoWorkspace.evGx[211]*acadoWorkspace.x[1] + acadoWorkspace.evGx[212]*acadoWorkspace.x[2] + acadoWorkspace.evGx[213]*acadoWorkspace.x[3] + acadoWorkspace.evGx[214]*acadoWorkspace.x[4] + acadoWorkspace.d[42];
acadoVariables.x[48] += + acadoWorkspace.evGx[215]*acadoWorkspace.x[0] + acadoWorkspace.evGx[216]*acadoWorkspace.x[1] + acadoWorkspace.evGx[217]*acadoWorkspace.x[2] + acadoWorkspace.evGx[218]*acadoWorkspace.x[3] + acadoWorkspace.evGx[219]*acadoWorkspace.x[4] + acadoWorkspace.d[43];
acadoVariables.x[49] += + acadoWorkspace.evGx[220]*acadoWorkspace.x[0] + acadoWorkspace.evGx[221]*acadoWorkspace.x[1] + acadoWorkspace.evGx[222]*acadoWorkspace.x[2] + acadoWorkspace.evGx[223]*acadoWorkspace.x[3] + acadoWorkspace.evGx[224]*acadoWorkspace.x[4] + acadoWorkspace.d[44];
acadoVariables.x[50] += + acadoWorkspace.evGx[225]*acadoWorkspace.x[0] + acadoWorkspace.evGx[226]*acadoWorkspace.x[1] + acadoWorkspace.evGx[227]*acadoWorkspace.x[2] + acadoWorkspace.evGx[228]*acadoWorkspace.x[3] + acadoWorkspace.evGx[229]*acadoWorkspace.x[4] + acadoWorkspace.d[45];
acadoVariables.x[51] += + acadoWorkspace.evGx[230]*acadoWorkspace.x[0] + acadoWorkspace.evGx[231]*acadoWorkspace.x[1] + acadoWorkspace.evGx[232]*acadoWorkspace.x[2] + acadoWorkspace.evGx[233]*acadoWorkspace.x[3] + acadoWorkspace.evGx[234]*acadoWorkspace.x[4] + acadoWorkspace.d[46];
acadoVariables.x[52] += + acadoWorkspace.evGx[235]*acadoWorkspace.x[0] + acadoWorkspace.evGx[236]*acadoWorkspace.x[1] + acadoWorkspace.evGx[237]*acadoWorkspace.x[2] + acadoWorkspace.evGx[238]*acadoWorkspace.x[3] + acadoWorkspace.evGx[239]*acadoWorkspace.x[4] + acadoWorkspace.d[47];
acadoVariables.x[53] += + acadoWorkspace.evGx[240]*acadoWorkspace.x[0] + acadoWorkspace.evGx[241]*acadoWorkspace.x[1] + acadoWorkspace.evGx[242]*acadoWorkspace.x[2] + acadoWorkspace.evGx[243]*acadoWorkspace.x[3] + acadoWorkspace.evGx[244]*acadoWorkspace.x[4] + acadoWorkspace.d[48];
acadoVariables.x[54] += + acadoWorkspace.evGx[245]*acadoWorkspace.x[0] + acadoWorkspace.evGx[246]*acadoWorkspace.x[1] + acadoWorkspace.evGx[247]*acadoWorkspace.x[2] + acadoWorkspace.evGx[248]*acadoWorkspace.x[3] + acadoWorkspace.evGx[249]*acadoWorkspace.x[4] + acadoWorkspace.d[49];
acadoVariables.x[55] += + acadoWorkspace.evGx[250]*acadoWorkspace.x[0] + acadoWorkspace.evGx[251]*acadoWorkspace.x[1] + acadoWorkspace.evGx[252]*acadoWorkspace.x[2] + acadoWorkspace.evGx[253]*acadoWorkspace.x[3] + acadoWorkspace.evGx[254]*acadoWorkspace.x[4] + acadoWorkspace.d[50];
acadoVariables.x[56] += + acadoWorkspace.evGx[255]*acadoWorkspace.x[0] + acadoWorkspace.evGx[256]*acadoWorkspace.x[1] + acadoWorkspace.evGx[257]*acadoWorkspace.x[2] + acadoWorkspace.evGx[258]*acadoWorkspace.x[3] + acadoWorkspace.evGx[259]*acadoWorkspace.x[4] + acadoWorkspace.d[51];
acadoVariables.x[57] += + acadoWorkspace.evGx[260]*acadoWorkspace.x[0] + acadoWorkspace.evGx[261]*acadoWorkspace.x[1] + acadoWorkspace.evGx[262]*acadoWorkspace.x[2] + acadoWorkspace.evGx[263]*acadoWorkspace.x[3] + acadoWorkspace.evGx[264]*acadoWorkspace.x[4] + acadoWorkspace.d[52];
acadoVariables.x[58] += + acadoWorkspace.evGx[265]*acadoWorkspace.x[0] + acadoWorkspace.evGx[266]*acadoWorkspace.x[1] + acadoWorkspace.evGx[267]*acadoWorkspace.x[2] + acadoWorkspace.evGx[268]*acadoWorkspace.x[3] + acadoWorkspace.evGx[269]*acadoWorkspace.x[4] + acadoWorkspace.d[53];
acadoVariables.x[59] += + acadoWorkspace.evGx[270]*acadoWorkspace.x[0] + acadoWorkspace.evGx[271]*acadoWorkspace.x[1] + acadoWorkspace.evGx[272]*acadoWorkspace.x[2] + acadoWorkspace.evGx[273]*acadoWorkspace.x[3] + acadoWorkspace.evGx[274]*acadoWorkspace.x[4] + acadoWorkspace.d[54];
acadoVariables.x[60] += + acadoWorkspace.evGx[275]*acadoWorkspace.x[0] + acadoWorkspace.evGx[276]*acadoWorkspace.x[1] + acadoWorkspace.evGx[277]*acadoWorkspace.x[2] + acadoWorkspace.evGx[278]*acadoWorkspace.x[3] + acadoWorkspace.evGx[279]*acadoWorkspace.x[4] + acadoWorkspace.d[55];
acadoVariables.x[61] += + acadoWorkspace.evGx[280]*acadoWorkspace.x[0] + acadoWorkspace.evGx[281]*acadoWorkspace.x[1] + acadoWorkspace.evGx[282]*acadoWorkspace.x[2] + acadoWorkspace.evGx[283]*acadoWorkspace.x[3] + acadoWorkspace.evGx[284]*acadoWorkspace.x[4] + acadoWorkspace.d[56];
acadoVariables.x[62] += + acadoWorkspace.evGx[285]*acadoWorkspace.x[0] + acadoWorkspace.evGx[286]*acadoWorkspace.x[1] + acadoWorkspace.evGx[287]*acadoWorkspace.x[2] + acadoWorkspace.evGx[288]*acadoWorkspace.x[3] + acadoWorkspace.evGx[289]*acadoWorkspace.x[4] + acadoWorkspace.d[57];
acadoVariables.x[63] += + acadoWorkspace.evGx[290]*acadoWorkspace.x[0] + acadoWorkspace.evGx[291]*acadoWorkspace.x[1] + acadoWorkspace.evGx[292]*acadoWorkspace.x[2] + acadoWorkspace.evGx[293]*acadoWorkspace.x[3] + acadoWorkspace.evGx[294]*acadoWorkspace.x[4] + acadoWorkspace.d[58];
acadoVariables.x[64] += + acadoWorkspace.evGx[295]*acadoWorkspace.x[0] + acadoWorkspace.evGx[296]*acadoWorkspace.x[1] + acadoWorkspace.evGx[297]*acadoWorkspace.x[2] + acadoWorkspace.evGx[298]*acadoWorkspace.x[3] + acadoWorkspace.evGx[299]*acadoWorkspace.x[4] + acadoWorkspace.d[59];
acadoVariables.x[65] += + acadoWorkspace.evGx[300]*acadoWorkspace.x[0] + acadoWorkspace.evGx[301]*acadoWorkspace.x[1] + acadoWorkspace.evGx[302]*acadoWorkspace.x[2] + acadoWorkspace.evGx[303]*acadoWorkspace.x[3] + acadoWorkspace.evGx[304]*acadoWorkspace.x[4] + acadoWorkspace.d[60];
acadoVariables.x[66] += + acadoWorkspace.evGx[305]*acadoWorkspace.x[0] + acadoWorkspace.evGx[306]*acadoWorkspace.x[1] + acadoWorkspace.evGx[307]*acadoWorkspace.x[2] + acadoWorkspace.evGx[308]*acadoWorkspace.x[3] + acadoWorkspace.evGx[309]*acadoWorkspace.x[4] + acadoWorkspace.d[61];
acadoVariables.x[67] += + acadoWorkspace.evGx[310]*acadoWorkspace.x[0] + acadoWorkspace.evGx[311]*acadoWorkspace.x[1] + acadoWorkspace.evGx[312]*acadoWorkspace.x[2] + acadoWorkspace.evGx[313]*acadoWorkspace.x[3] + acadoWorkspace.evGx[314]*acadoWorkspace.x[4] + acadoWorkspace.d[62];
acadoVariables.x[68] += + acadoWorkspace.evGx[315]*acadoWorkspace.x[0] + acadoWorkspace.evGx[316]*acadoWorkspace.x[1] + acadoWorkspace.evGx[317]*acadoWorkspace.x[2] + acadoWorkspace.evGx[318]*acadoWorkspace.x[3] + acadoWorkspace.evGx[319]*acadoWorkspace.x[4] + acadoWorkspace.d[63];
acadoVariables.x[69] += + acadoWorkspace.evGx[320]*acadoWorkspace.x[0] + acadoWorkspace.evGx[321]*acadoWorkspace.x[1] + acadoWorkspace.evGx[322]*acadoWorkspace.x[2] + acadoWorkspace.evGx[323]*acadoWorkspace.x[3] + acadoWorkspace.evGx[324]*acadoWorkspace.x[4] + acadoWorkspace.d[64];
acadoVariables.x[70] += + acadoWorkspace.evGx[325]*acadoWorkspace.x[0] + acadoWorkspace.evGx[326]*acadoWorkspace.x[1] + acadoWorkspace.evGx[327]*acadoWorkspace.x[2] + acadoWorkspace.evGx[328]*acadoWorkspace.x[3] + acadoWorkspace.evGx[329]*acadoWorkspace.x[4] + acadoWorkspace.d[65];
acadoVariables.x[71] += + acadoWorkspace.evGx[330]*acadoWorkspace.x[0] + acadoWorkspace.evGx[331]*acadoWorkspace.x[1] + acadoWorkspace.evGx[332]*acadoWorkspace.x[2] + acadoWorkspace.evGx[333]*acadoWorkspace.x[3] + acadoWorkspace.evGx[334]*acadoWorkspace.x[4] + acadoWorkspace.d[66];
acadoVariables.x[72] += + acadoWorkspace.evGx[335]*acadoWorkspace.x[0] + acadoWorkspace.evGx[336]*acadoWorkspace.x[1] + acadoWorkspace.evGx[337]*acadoWorkspace.x[2] + acadoWorkspace.evGx[338]*acadoWorkspace.x[3] + acadoWorkspace.evGx[339]*acadoWorkspace.x[4] + acadoWorkspace.d[67];
acadoVariables.x[73] += + acadoWorkspace.evGx[340]*acadoWorkspace.x[0] + acadoWorkspace.evGx[341]*acadoWorkspace.x[1] + acadoWorkspace.evGx[342]*acadoWorkspace.x[2] + acadoWorkspace.evGx[343]*acadoWorkspace.x[3] + acadoWorkspace.evGx[344]*acadoWorkspace.x[4] + acadoWorkspace.d[68];
acadoVariables.x[74] += + acadoWorkspace.evGx[345]*acadoWorkspace.x[0] + acadoWorkspace.evGx[346]*acadoWorkspace.x[1] + acadoWorkspace.evGx[347]*acadoWorkspace.x[2] + acadoWorkspace.evGx[348]*acadoWorkspace.x[3] + acadoWorkspace.evGx[349]*acadoWorkspace.x[4] + acadoWorkspace.d[69];
acadoVariables.x[75] += + acadoWorkspace.evGx[350]*acadoWorkspace.x[0] + acadoWorkspace.evGx[351]*acadoWorkspace.x[1] + acadoWorkspace.evGx[352]*acadoWorkspace.x[2] + acadoWorkspace.evGx[353]*acadoWorkspace.x[3] + acadoWorkspace.evGx[354]*acadoWorkspace.x[4] + acadoWorkspace.d[70];
acadoVariables.x[76] += + acadoWorkspace.evGx[355]*acadoWorkspace.x[0] + acadoWorkspace.evGx[356]*acadoWorkspace.x[1] + acadoWorkspace.evGx[357]*acadoWorkspace.x[2] + acadoWorkspace.evGx[358]*acadoWorkspace.x[3] + acadoWorkspace.evGx[359]*acadoWorkspace.x[4] + acadoWorkspace.d[71];
acadoVariables.x[77] += + acadoWorkspace.evGx[360]*acadoWorkspace.x[0] + acadoWorkspace.evGx[361]*acadoWorkspace.x[1] + acadoWorkspace.evGx[362]*acadoWorkspace.x[2] + acadoWorkspace.evGx[363]*acadoWorkspace.x[3] + acadoWorkspace.evGx[364]*acadoWorkspace.x[4] + acadoWorkspace.d[72];
acadoVariables.x[78] += + acadoWorkspace.evGx[365]*acadoWorkspace.x[0] + acadoWorkspace.evGx[366]*acadoWorkspace.x[1] + acadoWorkspace.evGx[367]*acadoWorkspace.x[2] + acadoWorkspace.evGx[368]*acadoWorkspace.x[3] + acadoWorkspace.evGx[369]*acadoWorkspace.x[4] + acadoWorkspace.d[73];
acadoVariables.x[79] += + acadoWorkspace.evGx[370]*acadoWorkspace.x[0] + acadoWorkspace.evGx[371]*acadoWorkspace.x[1] + acadoWorkspace.evGx[372]*acadoWorkspace.x[2] + acadoWorkspace.evGx[373]*acadoWorkspace.x[3] + acadoWorkspace.evGx[374]*acadoWorkspace.x[4] + acadoWorkspace.d[74];
acadoVariables.x[80] += + acadoWorkspace.evGx[375]*acadoWorkspace.x[0] + acadoWorkspace.evGx[376]*acadoWorkspace.x[1] + acadoWorkspace.evGx[377]*acadoWorkspace.x[2] + acadoWorkspace.evGx[378]*acadoWorkspace.x[3] + acadoWorkspace.evGx[379]*acadoWorkspace.x[4] + acadoWorkspace.d[75];
acadoVariables.x[81] += + acadoWorkspace.evGx[380]*acadoWorkspace.x[0] + acadoWorkspace.evGx[381]*acadoWorkspace.x[1] + acadoWorkspace.evGx[382]*acadoWorkspace.x[2] + acadoWorkspace.evGx[383]*acadoWorkspace.x[3] + acadoWorkspace.evGx[384]*acadoWorkspace.x[4] + acadoWorkspace.d[76];
acadoVariables.x[82] += + acadoWorkspace.evGx[385]*acadoWorkspace.x[0] + acadoWorkspace.evGx[386]*acadoWorkspace.x[1] + acadoWorkspace.evGx[387]*acadoWorkspace.x[2] + acadoWorkspace.evGx[388]*acadoWorkspace.x[3] + acadoWorkspace.evGx[389]*acadoWorkspace.x[4] + acadoWorkspace.d[77];
acadoVariables.x[83] += + acadoWorkspace.evGx[390]*acadoWorkspace.x[0] + acadoWorkspace.evGx[391]*acadoWorkspace.x[1] + acadoWorkspace.evGx[392]*acadoWorkspace.x[2] + acadoWorkspace.evGx[393]*acadoWorkspace.x[3] + acadoWorkspace.evGx[394]*acadoWorkspace.x[4] + acadoWorkspace.d[78];
acadoVariables.x[84] += + acadoWorkspace.evGx[395]*acadoWorkspace.x[0] + acadoWorkspace.evGx[396]*acadoWorkspace.x[1] + acadoWorkspace.evGx[397]*acadoWorkspace.x[2] + acadoWorkspace.evGx[398]*acadoWorkspace.x[3] + acadoWorkspace.evGx[399]*acadoWorkspace.x[4] + acadoWorkspace.d[79];
acadoVariables.x[85] += + acadoWorkspace.evGx[400]*acadoWorkspace.x[0] + acadoWorkspace.evGx[401]*acadoWorkspace.x[1] + acadoWorkspace.evGx[402]*acadoWorkspace.x[2] + acadoWorkspace.evGx[403]*acadoWorkspace.x[3] + acadoWorkspace.evGx[404]*acadoWorkspace.x[4] + acadoWorkspace.d[80];
acadoVariables.x[86] += + acadoWorkspace.evGx[405]*acadoWorkspace.x[0] + acadoWorkspace.evGx[406]*acadoWorkspace.x[1] + acadoWorkspace.evGx[407]*acadoWorkspace.x[2] + acadoWorkspace.evGx[408]*acadoWorkspace.x[3] + acadoWorkspace.evGx[409]*acadoWorkspace.x[4] + acadoWorkspace.d[81];
acadoVariables.x[87] += + acadoWorkspace.evGx[410]*acadoWorkspace.x[0] + acadoWorkspace.evGx[411]*acadoWorkspace.x[1] + acadoWorkspace.evGx[412]*acadoWorkspace.x[2] + acadoWorkspace.evGx[413]*acadoWorkspace.x[3] + acadoWorkspace.evGx[414]*acadoWorkspace.x[4] + acadoWorkspace.d[82];
acadoVariables.x[88] += + acadoWorkspace.evGx[415]*acadoWorkspace.x[0] + acadoWorkspace.evGx[416]*acadoWorkspace.x[1] + acadoWorkspace.evGx[417]*acadoWorkspace.x[2] + acadoWorkspace.evGx[418]*acadoWorkspace.x[3] + acadoWorkspace.evGx[419]*acadoWorkspace.x[4] + acadoWorkspace.d[83];
acadoVariables.x[89] += + acadoWorkspace.evGx[420]*acadoWorkspace.x[0] + acadoWorkspace.evGx[421]*acadoWorkspace.x[1] + acadoWorkspace.evGx[422]*acadoWorkspace.x[2] + acadoWorkspace.evGx[423]*acadoWorkspace.x[3] + acadoWorkspace.evGx[424]*acadoWorkspace.x[4] + acadoWorkspace.d[84];
acadoVariables.x[90] += + acadoWorkspace.evGx[425]*acadoWorkspace.x[0] + acadoWorkspace.evGx[426]*acadoWorkspace.x[1] + acadoWorkspace.evGx[427]*acadoWorkspace.x[2] + acadoWorkspace.evGx[428]*acadoWorkspace.x[3] + acadoWorkspace.evGx[429]*acadoWorkspace.x[4] + acadoWorkspace.d[85];
acadoVariables.x[91] += + acadoWorkspace.evGx[430]*acadoWorkspace.x[0] + acadoWorkspace.evGx[431]*acadoWorkspace.x[1] + acadoWorkspace.evGx[432]*acadoWorkspace.x[2] + acadoWorkspace.evGx[433]*acadoWorkspace.x[3] + acadoWorkspace.evGx[434]*acadoWorkspace.x[4] + acadoWorkspace.d[86];
acadoVariables.x[92] += + acadoWorkspace.evGx[435]*acadoWorkspace.x[0] + acadoWorkspace.evGx[436]*acadoWorkspace.x[1] + acadoWorkspace.evGx[437]*acadoWorkspace.x[2] + acadoWorkspace.evGx[438]*acadoWorkspace.x[3] + acadoWorkspace.evGx[439]*acadoWorkspace.x[4] + acadoWorkspace.d[87];
acadoVariables.x[93] += + acadoWorkspace.evGx[440]*acadoWorkspace.x[0] + acadoWorkspace.evGx[441]*acadoWorkspace.x[1] + acadoWorkspace.evGx[442]*acadoWorkspace.x[2] + acadoWorkspace.evGx[443]*acadoWorkspace.x[3] + acadoWorkspace.evGx[444]*acadoWorkspace.x[4] + acadoWorkspace.d[88];
acadoVariables.x[94] += + acadoWorkspace.evGx[445]*acadoWorkspace.x[0] + acadoWorkspace.evGx[446]*acadoWorkspace.x[1] + acadoWorkspace.evGx[447]*acadoWorkspace.x[2] + acadoWorkspace.evGx[448]*acadoWorkspace.x[3] + acadoWorkspace.evGx[449]*acadoWorkspace.x[4] + acadoWorkspace.d[89];
acadoVariables.x[95] += + acadoWorkspace.evGx[450]*acadoWorkspace.x[0] + acadoWorkspace.evGx[451]*acadoWorkspace.x[1] + acadoWorkspace.evGx[452]*acadoWorkspace.x[2] + acadoWorkspace.evGx[453]*acadoWorkspace.x[3] + acadoWorkspace.evGx[454]*acadoWorkspace.x[4] + acadoWorkspace.d[90];
acadoVariables.x[96] += + acadoWorkspace.evGx[455]*acadoWorkspace.x[0] + acadoWorkspace.evGx[456]*acadoWorkspace.x[1] + acadoWorkspace.evGx[457]*acadoWorkspace.x[2] + acadoWorkspace.evGx[458]*acadoWorkspace.x[3] + acadoWorkspace.evGx[459]*acadoWorkspace.x[4] + acadoWorkspace.d[91];
acadoVariables.x[97] += + acadoWorkspace.evGx[460]*acadoWorkspace.x[0] + acadoWorkspace.evGx[461]*acadoWorkspace.x[1] + acadoWorkspace.evGx[462]*acadoWorkspace.x[2] + acadoWorkspace.evGx[463]*acadoWorkspace.x[3] + acadoWorkspace.evGx[464]*acadoWorkspace.x[4] + acadoWorkspace.d[92];
acadoVariables.x[98] += + acadoWorkspace.evGx[465]*acadoWorkspace.x[0] + acadoWorkspace.evGx[466]*acadoWorkspace.x[1] + acadoWorkspace.evGx[467]*acadoWorkspace.x[2] + acadoWorkspace.evGx[468]*acadoWorkspace.x[3] + acadoWorkspace.evGx[469]*acadoWorkspace.x[4] + acadoWorkspace.d[93];
acadoVariables.x[99] += + acadoWorkspace.evGx[470]*acadoWorkspace.x[0] + acadoWorkspace.evGx[471]*acadoWorkspace.x[1] + acadoWorkspace.evGx[472]*acadoWorkspace.x[2] + acadoWorkspace.evGx[473]*acadoWorkspace.x[3] + acadoWorkspace.evGx[474]*acadoWorkspace.x[4] + acadoWorkspace.d[94];
acadoVariables.x[100] += + acadoWorkspace.evGx[475]*acadoWorkspace.x[0] + acadoWorkspace.evGx[476]*acadoWorkspace.x[1] + acadoWorkspace.evGx[477]*acadoWorkspace.x[2] + acadoWorkspace.evGx[478]*acadoWorkspace.x[3] + acadoWorkspace.evGx[479]*acadoWorkspace.x[4] + acadoWorkspace.d[95];
acadoVariables.x[101] += + acadoWorkspace.evGx[480]*acadoWorkspace.x[0] + acadoWorkspace.evGx[481]*acadoWorkspace.x[1] + acadoWorkspace.evGx[482]*acadoWorkspace.x[2] + acadoWorkspace.evGx[483]*acadoWorkspace.x[3] + acadoWorkspace.evGx[484]*acadoWorkspace.x[4] + acadoWorkspace.d[96];
acadoVariables.x[102] += + acadoWorkspace.evGx[485]*acadoWorkspace.x[0] + acadoWorkspace.evGx[486]*acadoWorkspace.x[1] + acadoWorkspace.evGx[487]*acadoWorkspace.x[2] + acadoWorkspace.evGx[488]*acadoWorkspace.x[3] + acadoWorkspace.evGx[489]*acadoWorkspace.x[4] + acadoWorkspace.d[97];
acadoVariables.x[103] += + acadoWorkspace.evGx[490]*acadoWorkspace.x[0] + acadoWorkspace.evGx[491]*acadoWorkspace.x[1] + acadoWorkspace.evGx[492]*acadoWorkspace.x[2] + acadoWorkspace.evGx[493]*acadoWorkspace.x[3] + acadoWorkspace.evGx[494]*acadoWorkspace.x[4] + acadoWorkspace.d[98];
acadoVariables.x[104] += + acadoWorkspace.evGx[495]*acadoWorkspace.x[0] + acadoWorkspace.evGx[496]*acadoWorkspace.x[1] + acadoWorkspace.evGx[497]*acadoWorkspace.x[2] + acadoWorkspace.evGx[498]*acadoWorkspace.x[3] + acadoWorkspace.evGx[499]*acadoWorkspace.x[4] + acadoWorkspace.d[99];

acado_multEDu( acadoWorkspace.E, &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 5 ]) );
acado_multEDu( &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 10 ]) );
acado_multEDu( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 10 ]) );
acado_multEDu( &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 25 ]) );
acado_multEDu( &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 25 ]) );
acado_multEDu( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 25 ]) );
acado_multEDu( &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 25 ]) );
acado_multEDu( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 25 ]) );
acado_multEDu( &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 35 ]) );
acado_multEDu( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 645 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 675 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 705 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 735 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 765 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 795 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 50 ]) );
acado_multEDu( &(acadoWorkspace.E[ 825 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 855 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 885 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 915 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 945 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 975 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 55 ]) );
acado_multEDu( &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1005 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1035 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1065 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1095 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1125 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1155 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1185 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1215 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1245 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1260 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1275 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1290 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1305 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1335 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1350 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 65 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1365 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1380 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1395 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1410 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1425 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1455 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1470 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1485 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1500 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1515 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1530 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1545 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 70 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1575 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1590 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1605 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1620 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1635 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1650 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1665 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1680 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1695 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1710 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1725 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1740 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1755 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1770 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1785 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 75 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1800 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1815 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1830 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1845 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1860 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1875 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1890 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1905 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1920 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1935 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1950 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1965 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1980 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1995 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2010 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2025 ]), &(acadoWorkspace.x[ 50 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2040 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2055 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2070 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2085 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2100 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2115 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2130 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2145 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2160 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2175 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2190 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2205 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2220 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2235 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2250 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2265 ]), &(acadoWorkspace.x[ 50 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2280 ]), &(acadoWorkspace.x[ 53 ]), &(acadoVariables.x[ 85 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2295 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2310 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2325 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2340 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2355 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2370 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2385 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2400 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2415 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2430 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2445 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2460 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2475 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2490 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2505 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2520 ]), &(acadoWorkspace.x[ 50 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2535 ]), &(acadoWorkspace.x[ 53 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2550 ]), &(acadoWorkspace.x[ 56 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2565 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2580 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2595 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2610 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2625 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2640 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2655 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2670 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2685 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2700 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2715 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2730 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2745 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2760 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2775 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2790 ]), &(acadoWorkspace.x[ 50 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2805 ]), &(acadoWorkspace.x[ 53 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2820 ]), &(acadoWorkspace.x[ 56 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2835 ]), &(acadoWorkspace.x[ 59 ]), &(acadoVariables.x[ 95 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2850 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2865 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2880 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2895 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2910 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2925 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2940 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2955 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2970 ]), &(acadoWorkspace.x[ 29 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 2985 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3000 ]), &(acadoWorkspace.x[ 35 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3015 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3030 ]), &(acadoWorkspace.x[ 41 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3045 ]), &(acadoWorkspace.x[ 44 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3060 ]), &(acadoWorkspace.x[ 47 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3075 ]), &(acadoWorkspace.x[ 50 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3090 ]), &(acadoWorkspace.x[ 53 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3105 ]), &(acadoWorkspace.x[ 56 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3120 ]), &(acadoWorkspace.x[ 59 ]), &(acadoVariables.x[ 100 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3135 ]), &(acadoWorkspace.x[ 62 ]), &(acadoVariables.x[ 100 ]) );
}

int acado_preparationStep(  )
{
int ret;

ret = acado_modelSimulation();
acado_evaluateObjective(  );
acado_condensePrep(  );
return ret;
}

int acado_feedbackStep(  )
{
int tmp;

acado_condenseFdb(  );

tmp = acado_solve( );

acado_expand(  );
return tmp;
}

int acado_initializeSolver(  )
{
int ret;

/* This is a function which must be called once before any other function call! */


ret = 0;

memset(&acadoWorkspace, 0, sizeof( acadoWorkspace ));
return ret;
}

void acado_initializeNodesByForwardSimulation(  )
{
int index;
for (index = 0; index < 20; ++index)
{
acadoWorkspace.state[0] = acadoVariables.x[index * 5];
acadoWorkspace.state[1] = acadoVariables.x[index * 5 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 5 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 5 + 3];
acadoWorkspace.state[4] = acadoVariables.x[index * 5 + 4];
acadoWorkspace.state[45] = acadoVariables.u[index * 3];
acadoWorkspace.state[46] = acadoVariables.u[index * 3 + 1];
acadoWorkspace.state[47] = acadoVariables.u[index * 3 + 2];
acadoWorkspace.state[48] = acadoVariables.od[index * 2];
acadoWorkspace.state[49] = acadoVariables.od[index * 2 + 1];

acado_integrate(acadoWorkspace.state, index == 0, index);

acadoVariables.x[index * 5 + 5] = acadoWorkspace.state[0];
acadoVariables.x[index * 5 + 6] = acadoWorkspace.state[1];
acadoVariables.x[index * 5 + 7] = acadoWorkspace.state[2];
acadoVariables.x[index * 5 + 8] = acadoWorkspace.state[3];
acadoVariables.x[index * 5 + 9] = acadoWorkspace.state[4];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 20; ++index)
{
acadoVariables.x[index * 5] = acadoVariables.x[index * 5 + 5];
acadoVariables.x[index * 5 + 1] = acadoVariables.x[index * 5 + 6];
acadoVariables.x[index * 5 + 2] = acadoVariables.x[index * 5 + 7];
acadoVariables.x[index * 5 + 3] = acadoVariables.x[index * 5 + 8];
acadoVariables.x[index * 5 + 4] = acadoVariables.x[index * 5 + 9];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[100] = xEnd[0];
acadoVariables.x[101] = xEnd[1];
acadoVariables.x[102] = xEnd[2];
acadoVariables.x[103] = xEnd[3];
acadoVariables.x[104] = xEnd[4];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[100];
acadoWorkspace.state[1] = acadoVariables.x[101];
acadoWorkspace.state[2] = acadoVariables.x[102];
acadoWorkspace.state[3] = acadoVariables.x[103];
acadoWorkspace.state[4] = acadoVariables.x[104];
if (uEnd != 0)
{
acadoWorkspace.state[45] = uEnd[0];
acadoWorkspace.state[46] = uEnd[1];
acadoWorkspace.state[47] = uEnd[2];
}
else
{
acadoWorkspace.state[45] = acadoVariables.u[57];
acadoWorkspace.state[46] = acadoVariables.u[58];
acadoWorkspace.state[47] = acadoVariables.u[59];
}
acadoWorkspace.state[48] = acadoVariables.od[40];
acadoWorkspace.state[49] = acadoVariables.od[41];

acado_integrate(acadoWorkspace.state, 1, 19);

acadoVariables.x[100] = acadoWorkspace.state[0];
acadoVariables.x[101] = acadoWorkspace.state[1];
acadoVariables.x[102] = acadoWorkspace.state[2];
acadoVariables.x[103] = acadoWorkspace.state[3];
acadoVariables.x[104] = acadoWorkspace.state[4];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 19; ++index)
{
acadoVariables.u[index * 3] = acadoVariables.u[index * 3 + 3];
acadoVariables.u[index * 3 + 1] = acadoVariables.u[index * 3 + 4];
acadoVariables.u[index * 3 + 2] = acadoVariables.u[index * 3 + 5];
}

if (uEnd != 0)
{
acadoVariables.u[57] = uEnd[0];
acadoVariables.u[58] = uEnd[1];
acadoVariables.u[59] = uEnd[2];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43] + acadoWorkspace.g[44]*acadoWorkspace.x[44] + acadoWorkspace.g[45]*acadoWorkspace.x[45] + acadoWorkspace.g[46]*acadoWorkspace.x[46] + acadoWorkspace.g[47]*acadoWorkspace.x[47] + acadoWorkspace.g[48]*acadoWorkspace.x[48] + acadoWorkspace.g[49]*acadoWorkspace.x[49] + acadoWorkspace.g[50]*acadoWorkspace.x[50] + acadoWorkspace.g[51]*acadoWorkspace.x[51] + acadoWorkspace.g[52]*acadoWorkspace.x[52] + acadoWorkspace.g[53]*acadoWorkspace.x[53] + acadoWorkspace.g[54]*acadoWorkspace.x[54] + acadoWorkspace.g[55]*acadoWorkspace.x[55] + acadoWorkspace.g[56]*acadoWorkspace.x[56] + acadoWorkspace.g[57]*acadoWorkspace.x[57] + acadoWorkspace.g[58]*acadoWorkspace.x[58] + acadoWorkspace.g[59]*acadoWorkspace.x[59] + acadoWorkspace.g[60]*acadoWorkspace.x[60] + acadoWorkspace.g[61]*acadoWorkspace.x[61] + acadoWorkspace.g[62]*acadoWorkspace.x[62] + acadoWorkspace.g[63]*acadoWorkspace.x[63] + acadoWorkspace.g[64]*acadoWorkspace.x[64];
kkt = fabs( kkt );
for (index = 0; index < 65; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 60; ++index)
{
prd = acadoWorkspace.y[index + 65];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lbA[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ubA[index] * prd);
}
return kkt;
}

real_t acado_getObjective(  )
{
real_t objVal;

int lRun1;
/** Row vector of size: 5 */
real_t tmpDy[ 5 ];

/** Column vector of size: 1 */
real_t tmpDyN[ 1 ];

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 5];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 5 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 5 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 5 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.x[lRun1 * 5 + 4];
acadoWorkspace.objValueIn[5] = acadoVariables.u[lRun1 * 3];
acadoWorkspace.objValueIn[6] = acadoVariables.u[lRun1 * 3 + 1];
acadoWorkspace.objValueIn[7] = acadoVariables.u[lRun1 * 3 + 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.objValueIn[9] = acadoVariables.od[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 5] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 5];
acadoWorkspace.Dy[lRun1 * 5 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 5 + 1];
acadoWorkspace.Dy[lRun1 * 5 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 5 + 2];
acadoWorkspace.Dy[lRun1 * 5 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 5 + 3];
acadoWorkspace.Dy[lRun1 * 5 + 4] = acadoWorkspace.objValueOut[4] - acadoVariables.y[lRun1 * 5 + 4];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[100];
acadoWorkspace.objValueIn[1] = acadoVariables.x[101];
acadoWorkspace.objValueIn[2] = acadoVariables.x[102];
acadoWorkspace.objValueIn[3] = acadoVariables.x[103];
acadoWorkspace.objValueIn[4] = acadoVariables.x[104];
acadoWorkspace.objValueIn[5] = acadoVariables.od[40];
acadoWorkspace.objValueIn[6] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 5] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 10] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 15] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 20];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 1] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 6] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 11] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 16] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 21];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 2] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 7] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 12] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 17] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 22];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 3] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 8] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 13] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 18] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 23];
tmpDy[4] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 4] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 9] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 14] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 19] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 24];
objVal += + acadoWorkspace.Dy[lRun1 * 5]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 5 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 5 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 5 + 3]*tmpDy[3] + acadoWorkspace.Dy[lRun1 * 5 + 4]*tmpDy[4];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0];

objVal *= 0.5;
return objVal;
}

