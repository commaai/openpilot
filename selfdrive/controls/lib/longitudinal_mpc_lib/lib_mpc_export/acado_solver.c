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
for (lRun1 = 0; lRun1 < 32; ++lRun1)
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

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = u[0];
out[4] = u[1];
out[5] = u[2];
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
}

void acado_setObjQ1Q2( real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = +tmpObjS[0];
tmpQ2[1] = +tmpObjS[1];
tmpQ2[2] = +tmpObjS[2];
tmpQ2[3] = +tmpObjS[3];
tmpQ2[4] = +tmpObjS[4];
tmpQ2[5] = +tmpObjS[5];
tmpQ2[6] = +tmpObjS[6];
tmpQ2[7] = +tmpObjS[7];
tmpQ2[8] = +tmpObjS[8];
tmpQ2[9] = +tmpObjS[9];
tmpQ2[10] = +tmpObjS[10];
tmpQ2[11] = +tmpObjS[11];
tmpQ2[12] = +tmpObjS[12];
tmpQ2[13] = +tmpObjS[13];
tmpQ2[14] = +tmpObjS[14];
tmpQ2[15] = +tmpObjS[15];
tmpQ2[16] = +tmpObjS[16];
tmpQ2[17] = +tmpObjS[17];
tmpQ2[18] = 0.0;
;
tmpQ2[19] = 0.0;
;
tmpQ2[20] = 0.0;
;
tmpQ2[21] = 0.0;
;
tmpQ2[22] = 0.0;
;
tmpQ2[23] = 0.0;
;
tmpQ2[24] = 0.0;
;
tmpQ2[25] = 0.0;
;
tmpQ2[26] = 0.0;
;
tmpQ2[27] = 0.0;
;
tmpQ2[28] = 0.0;
;
tmpQ2[29] = 0.0;
;
tmpQ1[0] = + tmpQ2[0];
tmpQ1[1] = + tmpQ2[1];
tmpQ1[2] = + tmpQ2[2];
tmpQ1[3] = 0.0;
;
tmpQ1[4] = 0.0;
;
tmpQ1[5] = + tmpQ2[6];
tmpQ1[6] = + tmpQ2[7];
tmpQ1[7] = + tmpQ2[8];
tmpQ1[8] = 0.0;
;
tmpQ1[9] = 0.0;
;
tmpQ1[10] = + tmpQ2[12];
tmpQ1[11] = + tmpQ2[13];
tmpQ1[12] = + tmpQ2[14];
tmpQ1[13] = 0.0;
;
tmpQ1[14] = 0.0;
;
tmpQ1[15] = + tmpQ2[18];
tmpQ1[16] = + tmpQ2[19];
tmpQ1[17] = + tmpQ2[20];
tmpQ1[18] = 0.0;
;
tmpQ1[19] = 0.0;
;
tmpQ1[20] = + tmpQ2[24];
tmpQ1[21] = + tmpQ2[25];
tmpQ1[22] = + tmpQ2[26];
tmpQ1[23] = 0.0;
;
tmpQ1[24] = 0.0;
;
}

void acado_setObjR1R2( real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = +tmpObjS[18];
tmpR2[1] = +tmpObjS[19];
tmpR2[2] = +tmpObjS[20];
tmpR2[3] = +tmpObjS[21];
tmpR2[4] = +tmpObjS[22];
tmpR2[5] = +tmpObjS[23];
tmpR2[6] = +tmpObjS[24];
tmpR2[7] = +tmpObjS[25];
tmpR2[8] = +tmpObjS[26];
tmpR2[9] = +tmpObjS[27];
tmpR2[10] = +tmpObjS[28];
tmpR2[11] = +tmpObjS[29];
tmpR2[12] = +tmpObjS[30];
tmpR2[13] = +tmpObjS[31];
tmpR2[14] = +tmpObjS[32];
tmpR2[15] = +tmpObjS[33];
tmpR2[16] = +tmpObjS[34];
tmpR2[17] = +tmpObjS[35];
tmpR1[0] = + tmpR2[3];
tmpR1[1] = + tmpR2[4];
tmpR1[2] = + tmpR2[5];
tmpR1[3] = + tmpR2[9];
tmpR1[4] = + tmpR2[10];
tmpR1[5] = + tmpR2[11];
tmpR1[6] = + tmpR2[15];
tmpR1[7] = + tmpR2[16];
tmpR1[8] = + tmpR2[17];
}

void acado_setObjQN1QN2( real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = +tmpObjSEndTerm[0];
tmpQN2[1] = +tmpObjSEndTerm[1];
tmpQN2[2] = +tmpObjSEndTerm[2];
tmpQN2[3] = +tmpObjSEndTerm[3];
tmpQN2[4] = +tmpObjSEndTerm[4];
tmpQN2[5] = +tmpObjSEndTerm[5];
tmpQN2[6] = +tmpObjSEndTerm[6];
tmpQN2[7] = +tmpObjSEndTerm[7];
tmpQN2[8] = +tmpObjSEndTerm[8];
tmpQN2[9] = 0.0;
;
tmpQN2[10] = 0.0;
;
tmpQN2[11] = 0.0;
;
tmpQN2[12] = 0.0;
;
tmpQN2[13] = 0.0;
;
tmpQN2[14] = 0.0;
;
tmpQN1[0] = + tmpQN2[0];
tmpQN1[1] = + tmpQN2[1];
tmpQN1[2] = + tmpQN2[2];
tmpQN1[3] = 0.0;
;
tmpQN1[4] = 0.0;
;
tmpQN1[5] = + tmpQN2[3];
tmpQN1[6] = + tmpQN2[4];
tmpQN1[7] = + tmpQN2[5];
tmpQN1[8] = 0.0;
;
tmpQN1[9] = 0.0;
;
tmpQN1[10] = + tmpQN2[6];
tmpQN1[11] = + tmpQN2[7];
tmpQN1[12] = + tmpQN2[8];
tmpQN1[13] = 0.0;
;
tmpQN1[14] = 0.0;
;
tmpQN1[15] = + tmpQN2[9];
tmpQN1[16] = + tmpQN2[10];
tmpQN1[17] = + tmpQN2[11];
tmpQN1[18] = 0.0;
;
tmpQN1[19] = 0.0;
;
tmpQN1[20] = + tmpQN2[12];
tmpQN1[21] = + tmpQN2[13];
tmpQN1[22] = + tmpQN2[14];
tmpQN1[23] = 0.0;
;
tmpQN1[24] = 0.0;
;
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 32; ++runObj)
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
acadoWorkspace.Dy[runObj * 6] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 6 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 6 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 6 + 3] = acadoWorkspace.objValueOut[3];
acadoWorkspace.Dy[runObj * 6 + 4] = acadoWorkspace.objValueOut[4];
acadoWorkspace.Dy[runObj * 6 + 5] = acadoWorkspace.objValueOut[5];

acado_setObjQ1Q2( &(acadoVariables.W[ runObj * 36 ]), &(acadoWorkspace.Q1[ runObj * 25 ]), &(acadoWorkspace.Q2[ runObj * 30 ]) );

acado_setObjR1R2( &(acadoVariables.W[ runObj * 36 ]), &(acadoWorkspace.R1[ runObj * 9 ]), &(acadoWorkspace.R2[ runObj * 18 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[160];
acadoWorkspace.objValueIn[1] = acadoVariables.x[161];
acadoWorkspace.objValueIn[2] = acadoVariables.x[162];
acadoWorkspace.objValueIn[3] = acadoVariables.x[163];
acadoWorkspace.objValueIn[4] = acadoVariables.x[164];
acadoWorkspace.objValueIn[5] = acadoVariables.od[64];
acadoWorkspace.objValueIn[6] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

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
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 5)] += + Gu1[0]*Gu2[0] + Gu1[3]*Gu2[3] + Gu1[6]*Gu2[6] + Gu1[9]*Gu2[9] + Gu1[12]*Gu2[12];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 6)] += + Gu1[0]*Gu2[1] + Gu1[3]*Gu2[4] + Gu1[6]*Gu2[7] + Gu1[9]*Gu2[10] + Gu1[12]*Gu2[13];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 7)] += + Gu1[0]*Gu2[2] + Gu1[3]*Gu2[5] + Gu1[6]*Gu2[8] + Gu1[9]*Gu2[11] + Gu1[12]*Gu2[14];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 5)] += + Gu1[1]*Gu2[0] + Gu1[4]*Gu2[3] + Gu1[7]*Gu2[6] + Gu1[10]*Gu2[9] + Gu1[13]*Gu2[12];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 6)] += + Gu1[1]*Gu2[1] + Gu1[4]*Gu2[4] + Gu1[7]*Gu2[7] + Gu1[10]*Gu2[10] + Gu1[13]*Gu2[13];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 7)] += + Gu1[1]*Gu2[2] + Gu1[4]*Gu2[5] + Gu1[7]*Gu2[8] + Gu1[10]*Gu2[11] + Gu1[13]*Gu2[14];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 5)] += + Gu1[2]*Gu2[0] + Gu1[5]*Gu2[3] + Gu1[8]*Gu2[6] + Gu1[11]*Gu2[9] + Gu1[14]*Gu2[12];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 6)] += + Gu1[2]*Gu2[1] + Gu1[5]*Gu2[4] + Gu1[8]*Gu2[7] + Gu1[11]*Gu2[10] + Gu1[14]*Gu2[13];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 7)] += + Gu1[2]*Gu2[2] + Gu1[5]*Gu2[5] + Gu1[8]*Gu2[8] + Gu1[11]*Gu2[11] + Gu1[14]*Gu2[14];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 5)] = R11[0];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 6)] = R11[1];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 7)] = R11[2];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 5)] = R11[3];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 6)] = R11[4];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 7)] = R11[5];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 5)] = R11[6];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 6)] = R11[7];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 7)] = R11[8];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 6)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 7)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 303 + 505) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 303 + 606) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 303 + 505) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 303 + 707) + (iRow * 3 + 5)];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 303 + 505) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 303 + 606) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 303 + 606) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 303 + 707) + (iRow * 3 + 6)];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 5)] = acadoWorkspace.H[(iCol * 303 + 505) + (iRow * 3 + 7)];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 6)] = acadoWorkspace.H[(iCol * 303 + 606) + (iRow * 3 + 7)];
acadoWorkspace.H[(iRow * 303 + 707) + (iCol * 3 + 7)] = acadoWorkspace.H[(iCol * 303 + 707) + (iRow * 3 + 7)];
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
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2] + R2[3]*Dy1[3] + R2[4]*Dy1[4] + R2[5]*Dy1[5];
RDy1[1] = + R2[6]*Dy1[0] + R2[7]*Dy1[1] + R2[8]*Dy1[2] + R2[9]*Dy1[3] + R2[10]*Dy1[4] + R2[11]*Dy1[5];
RDy1[2] = + R2[12]*Dy1[0] + R2[13]*Dy1[1] + R2[14]*Dy1[2] + R2[15]*Dy1[3] + R2[16]*Dy1[4] + R2[17]*Dy1[5];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3] + Q2[4]*Dy1[4] + Q2[5]*Dy1[5];
QDy1[1] = + Q2[6]*Dy1[0] + Q2[7]*Dy1[1] + Q2[8]*Dy1[2] + Q2[9]*Dy1[3] + Q2[10]*Dy1[4] + Q2[11]*Dy1[5];
QDy1[2] = + Q2[12]*Dy1[0] + Q2[13]*Dy1[1] + Q2[14]*Dy1[2] + Q2[15]*Dy1[3] + Q2[16]*Dy1[4] + Q2[17]*Dy1[5];
QDy1[3] = + Q2[18]*Dy1[0] + Q2[19]*Dy1[1] + Q2[20]*Dy1[2] + Q2[21]*Dy1[3] + Q2[22]*Dy1[4] + Q2[23]*Dy1[5];
QDy1[4] = + Q2[24]*Dy1[0] + Q2[25]*Dy1[1] + Q2[26]*Dy1[2] + Q2[27]*Dy1[3] + Q2[28]*Dy1[4] + Q2[29]*Dy1[5];
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
acadoWorkspace.H[101] = 0.0000000000000000e+00;
acadoWorkspace.H[102] = 0.0000000000000000e+00;
acadoWorkspace.H[103] = 0.0000000000000000e+00;
acadoWorkspace.H[104] = 0.0000000000000000e+00;
acadoWorkspace.H[105] = 0.0000000000000000e+00;
acadoWorkspace.H[202] = 0.0000000000000000e+00;
acadoWorkspace.H[203] = 0.0000000000000000e+00;
acadoWorkspace.H[204] = 0.0000000000000000e+00;
acadoWorkspace.H[205] = 0.0000000000000000e+00;
acadoWorkspace.H[206] = 0.0000000000000000e+00;
acadoWorkspace.H[303] = 0.0000000000000000e+00;
acadoWorkspace.H[304] = 0.0000000000000000e+00;
acadoWorkspace.H[305] = 0.0000000000000000e+00;
acadoWorkspace.H[306] = 0.0000000000000000e+00;
acadoWorkspace.H[307] = 0.0000000000000000e+00;
acadoWorkspace.H[404] = 0.0000000000000000e+00;
acadoWorkspace.H[405] = 0.0000000000000000e+00;
acadoWorkspace.H[406] = 0.0000000000000000e+00;
acadoWorkspace.H[407] = 0.0000000000000000e+00;
acadoWorkspace.H[408] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[5]*Gx2[5] + Gx1[10]*Gx2[10] + Gx1[15]*Gx2[15] + Gx1[20]*Gx2[20];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[5]*Gx2[6] + Gx1[10]*Gx2[11] + Gx1[15]*Gx2[16] + Gx1[20]*Gx2[21];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[5]*Gx2[7] + Gx1[10]*Gx2[12] + Gx1[15]*Gx2[17] + Gx1[20]*Gx2[22];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[5]*Gx2[8] + Gx1[10]*Gx2[13] + Gx1[15]*Gx2[18] + Gx1[20]*Gx2[23];
acadoWorkspace.H[4] += + Gx1[0]*Gx2[4] + Gx1[5]*Gx2[9] + Gx1[10]*Gx2[14] + Gx1[15]*Gx2[19] + Gx1[20]*Gx2[24];
acadoWorkspace.H[101] += + Gx1[1]*Gx2[0] + Gx1[6]*Gx2[5] + Gx1[11]*Gx2[10] + Gx1[16]*Gx2[15] + Gx1[21]*Gx2[20];
acadoWorkspace.H[102] += + Gx1[1]*Gx2[1] + Gx1[6]*Gx2[6] + Gx1[11]*Gx2[11] + Gx1[16]*Gx2[16] + Gx1[21]*Gx2[21];
acadoWorkspace.H[103] += + Gx1[1]*Gx2[2] + Gx1[6]*Gx2[7] + Gx1[11]*Gx2[12] + Gx1[16]*Gx2[17] + Gx1[21]*Gx2[22];
acadoWorkspace.H[104] += + Gx1[1]*Gx2[3] + Gx1[6]*Gx2[8] + Gx1[11]*Gx2[13] + Gx1[16]*Gx2[18] + Gx1[21]*Gx2[23];
acadoWorkspace.H[105] += + Gx1[1]*Gx2[4] + Gx1[6]*Gx2[9] + Gx1[11]*Gx2[14] + Gx1[16]*Gx2[19] + Gx1[21]*Gx2[24];
acadoWorkspace.H[202] += + Gx1[2]*Gx2[0] + Gx1[7]*Gx2[5] + Gx1[12]*Gx2[10] + Gx1[17]*Gx2[15] + Gx1[22]*Gx2[20];
acadoWorkspace.H[203] += + Gx1[2]*Gx2[1] + Gx1[7]*Gx2[6] + Gx1[12]*Gx2[11] + Gx1[17]*Gx2[16] + Gx1[22]*Gx2[21];
acadoWorkspace.H[204] += + Gx1[2]*Gx2[2] + Gx1[7]*Gx2[7] + Gx1[12]*Gx2[12] + Gx1[17]*Gx2[17] + Gx1[22]*Gx2[22];
acadoWorkspace.H[205] += + Gx1[2]*Gx2[3] + Gx1[7]*Gx2[8] + Gx1[12]*Gx2[13] + Gx1[17]*Gx2[18] + Gx1[22]*Gx2[23];
acadoWorkspace.H[206] += + Gx1[2]*Gx2[4] + Gx1[7]*Gx2[9] + Gx1[12]*Gx2[14] + Gx1[17]*Gx2[19] + Gx1[22]*Gx2[24];
acadoWorkspace.H[303] += + Gx1[3]*Gx2[0] + Gx1[8]*Gx2[5] + Gx1[13]*Gx2[10] + Gx1[18]*Gx2[15] + Gx1[23]*Gx2[20];
acadoWorkspace.H[304] += + Gx1[3]*Gx2[1] + Gx1[8]*Gx2[6] + Gx1[13]*Gx2[11] + Gx1[18]*Gx2[16] + Gx1[23]*Gx2[21];
acadoWorkspace.H[305] += + Gx1[3]*Gx2[2] + Gx1[8]*Gx2[7] + Gx1[13]*Gx2[12] + Gx1[18]*Gx2[17] + Gx1[23]*Gx2[22];
acadoWorkspace.H[306] += + Gx1[3]*Gx2[3] + Gx1[8]*Gx2[8] + Gx1[13]*Gx2[13] + Gx1[18]*Gx2[18] + Gx1[23]*Gx2[23];
acadoWorkspace.H[307] += + Gx1[3]*Gx2[4] + Gx1[8]*Gx2[9] + Gx1[13]*Gx2[14] + Gx1[18]*Gx2[19] + Gx1[23]*Gx2[24];
acadoWorkspace.H[404] += + Gx1[4]*Gx2[0] + Gx1[9]*Gx2[5] + Gx1[14]*Gx2[10] + Gx1[19]*Gx2[15] + Gx1[24]*Gx2[20];
acadoWorkspace.H[405] += + Gx1[4]*Gx2[1] + Gx1[9]*Gx2[6] + Gx1[14]*Gx2[11] + Gx1[19]*Gx2[16] + Gx1[24]*Gx2[21];
acadoWorkspace.H[406] += + Gx1[4]*Gx2[2] + Gx1[9]*Gx2[7] + Gx1[14]*Gx2[12] + Gx1[19]*Gx2[17] + Gx1[24]*Gx2[22];
acadoWorkspace.H[407] += + Gx1[4]*Gx2[3] + Gx1[9]*Gx2[8] + Gx1[14]*Gx2[13] + Gx1[19]*Gx2[18] + Gx1[24]*Gx2[23];
acadoWorkspace.H[408] += + Gx1[4]*Gx2[4] + Gx1[9]*Gx2[9] + Gx1[14]*Gx2[14] + Gx1[19]*Gx2[19] + Gx1[24]*Gx2[24];
}

void acado_multHxC( real_t* const Hx, real_t* const Gx, real_t* const A01 )
{
A01[0] = + Hx[0]*Gx[0] + Hx[1]*Gx[5] + Hx[2]*Gx[10] + Hx[3]*Gx[15] + Hx[4]*Gx[20];
A01[1] = + Hx[0]*Gx[1] + Hx[1]*Gx[6] + Hx[2]*Gx[11] + Hx[3]*Gx[16] + Hx[4]*Gx[21];
A01[2] = + Hx[0]*Gx[2] + Hx[1]*Gx[7] + Hx[2]*Gx[12] + Hx[3]*Gx[17] + Hx[4]*Gx[22];
A01[3] = + Hx[0]*Gx[3] + Hx[1]*Gx[8] + Hx[2]*Gx[13] + Hx[3]*Gx[18] + Hx[4]*Gx[23];
A01[4] = + Hx[0]*Gx[4] + Hx[1]*Gx[9] + Hx[2]*Gx[14] + Hx[3]*Gx[19] + Hx[4]*Gx[24];
A01[101] = + Hx[5]*Gx[0] + Hx[6]*Gx[5] + Hx[7]*Gx[10] + Hx[8]*Gx[15] + Hx[9]*Gx[20];
A01[102] = + Hx[5]*Gx[1] + Hx[6]*Gx[6] + Hx[7]*Gx[11] + Hx[8]*Gx[16] + Hx[9]*Gx[21];
A01[103] = + Hx[5]*Gx[2] + Hx[6]*Gx[7] + Hx[7]*Gx[12] + Hx[8]*Gx[17] + Hx[9]*Gx[22];
A01[104] = + Hx[5]*Gx[3] + Hx[6]*Gx[8] + Hx[7]*Gx[13] + Hx[8]*Gx[18] + Hx[9]*Gx[23];
A01[105] = + Hx[5]*Gx[4] + Hx[6]*Gx[9] + Hx[7]*Gx[14] + Hx[8]*Gx[19] + Hx[9]*Gx[24];
A01[202] = + Hx[10]*Gx[0] + Hx[11]*Gx[5] + Hx[12]*Gx[10] + Hx[13]*Gx[15] + Hx[14]*Gx[20];
A01[203] = + Hx[10]*Gx[1] + Hx[11]*Gx[6] + Hx[12]*Gx[11] + Hx[13]*Gx[16] + Hx[14]*Gx[21];
A01[204] = + Hx[10]*Gx[2] + Hx[11]*Gx[7] + Hx[12]*Gx[12] + Hx[13]*Gx[17] + Hx[14]*Gx[22];
A01[205] = + Hx[10]*Gx[3] + Hx[11]*Gx[8] + Hx[12]*Gx[13] + Hx[13]*Gx[18] + Hx[14]*Gx[23];
A01[206] = + Hx[10]*Gx[4] + Hx[11]*Gx[9] + Hx[12]*Gx[14] + Hx[13]*Gx[19] + Hx[14]*Gx[24];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 303) + (col * 3 + 5)] = + Hx[0]*E[0] + Hx[1]*E[3] + Hx[2]*E[6] + Hx[3]*E[9] + Hx[4]*E[12];
acadoWorkspace.A[(row * 303) + (col * 3 + 6)] = + Hx[0]*E[1] + Hx[1]*E[4] + Hx[2]*E[7] + Hx[3]*E[10] + Hx[4]*E[13];
acadoWorkspace.A[(row * 303) + (col * 3 + 7)] = + Hx[0]*E[2] + Hx[1]*E[5] + Hx[2]*E[8] + Hx[3]*E[11] + Hx[4]*E[14];
acadoWorkspace.A[(row * 303 + 101) + (col * 3 + 5)] = + Hx[5]*E[0] + Hx[6]*E[3] + Hx[7]*E[6] + Hx[8]*E[9] + Hx[9]*E[12];
acadoWorkspace.A[(row * 303 + 101) + (col * 3 + 6)] = + Hx[5]*E[1] + Hx[6]*E[4] + Hx[7]*E[7] + Hx[8]*E[10] + Hx[9]*E[13];
acadoWorkspace.A[(row * 303 + 101) + (col * 3 + 7)] = + Hx[5]*E[2] + Hx[6]*E[5] + Hx[7]*E[8] + Hx[8]*E[11] + Hx[9]*E[14];
acadoWorkspace.A[(row * 303 + 202) + (col * 3 + 5)] = + Hx[10]*E[0] + Hx[11]*E[3] + Hx[12]*E[6] + Hx[13]*E[9] + Hx[14]*E[12];
acadoWorkspace.A[(row * 303 + 202) + (col * 3 + 6)] = + Hx[10]*E[1] + Hx[11]*E[4] + Hx[12]*E[7] + Hx[13]*E[10] + Hx[14]*E[13];
acadoWorkspace.A[(row * 303 + 202) + (col * 3 + 7)] = + Hx[10]*E[2] + Hx[11]*E[5] + Hx[12]*E[8] + Hx[13]*E[11] + Hx[14]*E[14];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2] + Hx[3]*tmpd[3] + Hx[4]*tmpd[4];
acadoWorkspace.evHxd[1] = + Hx[5]*tmpd[0] + Hx[6]*tmpd[1] + Hx[7]*tmpd[2] + Hx[8]*tmpd[3] + Hx[9]*tmpd[4];
acadoWorkspace.evHxd[2] = + Hx[10]*tmpd[0] + Hx[11]*tmpd[1] + Hx[12]*tmpd[2] + Hx[13]*tmpd[3] + Hx[14]*tmpd[4];
lbA[0] -= acadoWorkspace.evHxd[0];
lbA[1] -= acadoWorkspace.evHxd[1];
lbA[2] -= acadoWorkspace.evHxd[2];
ubA[0] -= acadoWorkspace.evHxd[0];
ubA[1] -= acadoWorkspace.evHxd[1];
ubA[2] -= acadoWorkspace.evHxd[2];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 5;
const real_t* od = in + 8;
/* Vector of auxiliary variables; number of elements: 24. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (real_t)(0.0000000000000000e+00);
a[1] = (real_t)(1.0000000000000000e+00);
a[2] = (real_t)(0.0000000000000000e+00);
a[3] = (real_t)(0.0000000000000000e+00);
a[4] = (real_t)(0.0000000000000000e+00);
a[5] = (real_t)(0.0000000000000000e+00);
a[6] = (real_t)(0.0000000000000000e+00);
a[7] = (real_t)(1.0000000000000000e+00);
a[8] = (real_t)(0.0000000000000000e+00);
a[9] = (real_t)(0.0000000000000000e+00);
a[10] = (real_t)(0.0000000000000000e+00);
a[11] = (real_t)(0.0000000000000000e+00);
a[12] = (real_t)(1.0000000000000000e+00);
a[13] = (real_t)(0.0000000000000000e+00);
a[14] = (real_t)(0.0000000000000000e+00);
a[15] = (real_t)(0.0000000000000000e+00);
a[16] = (real_t)(1.0000000000000000e+00);
a[17] = (real_t)(0.0000000000000000e+00);
a[18] = (real_t)(0.0000000000000000e+00);
a[19] = (real_t)(0.0000000000000000e+00);
a[20] = (real_t)(1.0000000000000000e+00);
a[21] = (real_t)(0.0000000000000000e+00);
a[22] = (real_t)(0.0000000000000000e+00);
a[23] = (real_t)(1.0000000000000000e+00);

/* Compute outputs: */
out[0] = (xd[1]+u[1]);
out[1] = ((xd[2]-od[0])+u[2]);
out[2] = ((xd[2]-od[1])+u[2]);
out[3] = a[0];
out[4] = a[1];
out[5] = a[2];
out[6] = a[3];
out[7] = a[4];
out[8] = a[5];
out[9] = a[6];
out[10] = a[7];
out[11] = a[8];
out[12] = a[9];
out[13] = a[10];
out[14] = a[11];
out[15] = a[12];
out[16] = a[13];
out[17] = a[14];
out[18] = a[15];
out[19] = a[16];
out[20] = a[17];
out[21] = a[18];
out[22] = a[19];
out[23] = a[20];
out[24] = a[21];
out[25] = a[22];
out[26] = a[23];
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
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
for (lRun1 = 1; lRun1 < 32; ++lRun1)
{
acado_moveGxT( &(acadoWorkspace.evGx[ lRun1 * 25 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ lRun1 * 5-5 ]), &(acadoWorkspace.evGx[ lRun1 * 25 ]), &(acadoWorkspace.d[ lRun1 * 5 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ lRun1 * 25-25 ]), &(acadoWorkspace.evGx[ lRun1 * 25 ]) );
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
lRun4 = (((lRun1) * (lRun1-1)) / (2)) + (lRun2);
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ lRun4 * 15 ]), &(acadoWorkspace.E[ lRun3 * 15 ]) );
}
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_moveGuE( &(acadoWorkspace.evGu[ lRun1 * 15 ]), &(acadoWorkspace.E[ lRun3 * 15 ]) );
}

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
acado_multGxGx( &(acadoWorkspace.Q1[ 500 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.QGx[ 475 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 525 ]), &(acadoWorkspace.evGx[ 500 ]), &(acadoWorkspace.QGx[ 500 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 550 ]), &(acadoWorkspace.evGx[ 525 ]), &(acadoWorkspace.QGx[ 525 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 575 ]), &(acadoWorkspace.evGx[ 550 ]), &(acadoWorkspace.QGx[ 550 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 600 ]), &(acadoWorkspace.evGx[ 575 ]), &(acadoWorkspace.QGx[ 575 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 625 ]), &(acadoWorkspace.evGx[ 600 ]), &(acadoWorkspace.QGx[ 600 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 650 ]), &(acadoWorkspace.evGx[ 625 ]), &(acadoWorkspace.QGx[ 625 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 675 ]), &(acadoWorkspace.evGx[ 650 ]), &(acadoWorkspace.QGx[ 650 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 700 ]), &(acadoWorkspace.evGx[ 675 ]), &(acadoWorkspace.QGx[ 675 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 725 ]), &(acadoWorkspace.evGx[ 700 ]), &(acadoWorkspace.QGx[ 700 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 750 ]), &(acadoWorkspace.evGx[ 725 ]), &(acadoWorkspace.QGx[ 725 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 775 ]), &(acadoWorkspace.evGx[ 750 ]), &(acadoWorkspace.QGx[ 750 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 775 ]), &(acadoWorkspace.QGx[ 775 ]) );

for (lRun1 = 0; lRun1 < 31; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( &(acadoWorkspace.Q1[ lRun1 * 25 + 25 ]), &(acadoWorkspace.E[ lRun3 * 15 ]), &(acadoWorkspace.QE[ lRun3 * 15 ]) );
}
}

for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ lRun3 * 15 ]), &(acadoWorkspace.QE[ lRun3 * 15 ]) );
}

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
acado_multCTQC( &(acadoWorkspace.evGx[ 500 ]), &(acadoWorkspace.QGx[ 500 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 525 ]), &(acadoWorkspace.QGx[ 525 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 550 ]), &(acadoWorkspace.QGx[ 550 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 575 ]), &(acadoWorkspace.QGx[ 575 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 600 ]), &(acadoWorkspace.QGx[ 600 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 625 ]), &(acadoWorkspace.QGx[ 625 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 650 ]), &(acadoWorkspace.QGx[ 650 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 675 ]), &(acadoWorkspace.QGx[ 675 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 700 ]), &(acadoWorkspace.QGx[ 700 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 725 ]), &(acadoWorkspace.QGx[ 725 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 750 ]), &(acadoWorkspace.QGx[ 750 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 775 ]), &(acadoWorkspace.QGx[ 775 ]) );

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_zeroBlockH10( &(acadoWorkspace.H10[ lRun1 * 15 ]) );
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multQETGx( &(acadoWorkspace.QE[ lRun3 * 15 ]), &(acadoWorkspace.evGx[ lRun2 * 25 ]), &(acadoWorkspace.H10[ lRun1 * 15 ]) );
}
}

for (lRun2 = 0;lRun2 < 5; ++lRun2)
for (lRun3 = 0;lRun3 < 96; ++lRun3)
acadoWorkspace.H[(lRun2 * 101) + (lRun3 + 5)] = acadoWorkspace.H10[(lRun3 * 5) + (lRun2)];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_setBlockH11_R1( lRun1, lRun1, &(acadoWorkspace.R1[ lRun1 * 9 ]) );
lRun2 = lRun1;
for (lRun3 = lRun1; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 15 ]), &(acadoWorkspace.QE[ lRun5 * 15 ]) );
}
for (lRun2 = lRun1 + 1; lRun2 < 32; ++lRun2)
{
acado_zeroBlockH11( lRun1, lRun2 );
for (lRun3 = lRun2; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 15 ]), &(acadoWorkspace.QE[ lRun5 * 15 ]) );
}
}
}

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
acado_copyHTH( lRun1, lRun2 );
}
}

for (lRun2 = 0;lRun2 < 96; ++lRun2)
for (lRun3 = 0;lRun3 < 5; ++lRun3)
acadoWorkspace.H[(lRun2 * 101 + 505) + (lRun3)] = acadoWorkspace.H10[(lRun2 * 5) + (lRun3)];

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
acado_multQ1d( &(acadoWorkspace.Q1[ 500 ]), &(acadoWorkspace.d[ 95 ]), &(acadoWorkspace.Qd[ 95 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 525 ]), &(acadoWorkspace.d[ 100 ]), &(acadoWorkspace.Qd[ 100 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 550 ]), &(acadoWorkspace.d[ 105 ]), &(acadoWorkspace.Qd[ 105 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 575 ]), &(acadoWorkspace.d[ 110 ]), &(acadoWorkspace.Qd[ 110 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 600 ]), &(acadoWorkspace.d[ 115 ]), &(acadoWorkspace.Qd[ 115 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 625 ]), &(acadoWorkspace.d[ 120 ]), &(acadoWorkspace.Qd[ 120 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 650 ]), &(acadoWorkspace.d[ 125 ]), &(acadoWorkspace.Qd[ 125 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 675 ]), &(acadoWorkspace.d[ 130 ]), &(acadoWorkspace.Qd[ 130 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 700 ]), &(acadoWorkspace.d[ 135 ]), &(acadoWorkspace.Qd[ 135 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 725 ]), &(acadoWorkspace.d[ 140 ]), &(acadoWorkspace.Qd[ 140 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 750 ]), &(acadoWorkspace.d[ 145 ]), &(acadoWorkspace.Qd[ 145 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 775 ]), &(acadoWorkspace.d[ 150 ]), &(acadoWorkspace.Qd[ 150 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 155 ]), &(acadoWorkspace.Qd[ 155 ]) );

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
acado_macCTSlx( &(acadoWorkspace.evGx[ 500 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 525 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 550 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 575 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 600 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 625 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 650 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 675 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 700 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 725 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 750 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 775 ]), acadoWorkspace.g );
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_macETSlu( &(acadoWorkspace.QE[ lRun3 * 15 ]), &(acadoWorkspace.g[ lRun1 * 3 + 5 ]) );
}
}
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
acadoWorkspace.lb[65] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[60];
acadoWorkspace.lb[66] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[61];
acadoWorkspace.lb[67] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[62];
acadoWorkspace.lb[68] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[63];
acadoWorkspace.lb[69] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[64];
acadoWorkspace.lb[70] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[65];
acadoWorkspace.lb[71] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[66];
acadoWorkspace.lb[72] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[67];
acadoWorkspace.lb[73] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[68];
acadoWorkspace.lb[74] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[69];
acadoWorkspace.lb[75] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[70];
acadoWorkspace.lb[76] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[71];
acadoWorkspace.lb[77] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[72];
acadoWorkspace.lb[78] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[73];
acadoWorkspace.lb[79] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[74];
acadoWorkspace.lb[80] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[75];
acadoWorkspace.lb[81] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[76];
acadoWorkspace.lb[82] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[77];
acadoWorkspace.lb[83] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[78];
acadoWorkspace.lb[84] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[79];
acadoWorkspace.lb[85] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[80];
acadoWorkspace.lb[86] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[81];
acadoWorkspace.lb[87] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[82];
acadoWorkspace.lb[88] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[83];
acadoWorkspace.lb[89] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[84];
acadoWorkspace.lb[90] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[85];
acadoWorkspace.lb[91] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[86];
acadoWorkspace.lb[92] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[87];
acadoWorkspace.lb[93] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[88];
acadoWorkspace.lb[94] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[89];
acadoWorkspace.lb[95] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[90];
acadoWorkspace.lb[96] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[91];
acadoWorkspace.lb[97] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[92];
acadoWorkspace.lb[98] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[93];
acadoWorkspace.lb[99] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[94];
acadoWorkspace.lb[100] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[95];
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
acadoWorkspace.ub[65] = (real_t)1.0000000000000000e+12 - acadoVariables.u[60];
acadoWorkspace.ub[66] = (real_t)1.0000000000000000e+12 - acadoVariables.u[61];
acadoWorkspace.ub[67] = (real_t)1.0000000000000000e+12 - acadoVariables.u[62];
acadoWorkspace.ub[68] = (real_t)1.0000000000000000e+12 - acadoVariables.u[63];
acadoWorkspace.ub[69] = (real_t)1.0000000000000000e+12 - acadoVariables.u[64];
acadoWorkspace.ub[70] = (real_t)1.0000000000000000e+12 - acadoVariables.u[65];
acadoWorkspace.ub[71] = (real_t)1.0000000000000000e+12 - acadoVariables.u[66];
acadoWorkspace.ub[72] = (real_t)1.0000000000000000e+12 - acadoVariables.u[67];
acadoWorkspace.ub[73] = (real_t)1.0000000000000000e+12 - acadoVariables.u[68];
acadoWorkspace.ub[74] = (real_t)1.0000000000000000e+12 - acadoVariables.u[69];
acadoWorkspace.ub[75] = (real_t)1.0000000000000000e+12 - acadoVariables.u[70];
acadoWorkspace.ub[76] = (real_t)1.0000000000000000e+12 - acadoVariables.u[71];
acadoWorkspace.ub[77] = (real_t)1.0000000000000000e+12 - acadoVariables.u[72];
acadoWorkspace.ub[78] = (real_t)1.0000000000000000e+12 - acadoVariables.u[73];
acadoWorkspace.ub[79] = (real_t)1.0000000000000000e+12 - acadoVariables.u[74];
acadoWorkspace.ub[80] = (real_t)1.0000000000000000e+12 - acadoVariables.u[75];
acadoWorkspace.ub[81] = (real_t)1.0000000000000000e+12 - acadoVariables.u[76];
acadoWorkspace.ub[82] = (real_t)1.0000000000000000e+12 - acadoVariables.u[77];
acadoWorkspace.ub[83] = (real_t)1.0000000000000000e+12 - acadoVariables.u[78];
acadoWorkspace.ub[84] = (real_t)1.0000000000000000e+12 - acadoVariables.u[79];
acadoWorkspace.ub[85] = (real_t)1.0000000000000000e+12 - acadoVariables.u[80];
acadoWorkspace.ub[86] = (real_t)1.0000000000000000e+12 - acadoVariables.u[81];
acadoWorkspace.ub[87] = (real_t)1.0000000000000000e+12 - acadoVariables.u[82];
acadoWorkspace.ub[88] = (real_t)1.0000000000000000e+12 - acadoVariables.u[83];
acadoWorkspace.ub[89] = (real_t)1.0000000000000000e+12 - acadoVariables.u[84];
acadoWorkspace.ub[90] = (real_t)1.0000000000000000e+12 - acadoVariables.u[85];
acadoWorkspace.ub[91] = (real_t)1.0000000000000000e+12 - acadoVariables.u[86];
acadoWorkspace.ub[92] = (real_t)1.0000000000000000e+12 - acadoVariables.u[87];
acadoWorkspace.ub[93] = (real_t)1.0000000000000000e+12 - acadoVariables.u[88];
acadoWorkspace.ub[94] = (real_t)1.0000000000000000e+12 - acadoVariables.u[89];
acadoWorkspace.ub[95] = (real_t)1.0000000000000000e+12 - acadoVariables.u[90];
acadoWorkspace.ub[96] = (real_t)1.0000000000000000e+12 - acadoVariables.u[91];
acadoWorkspace.ub[97] = (real_t)1.0000000000000000e+12 - acadoVariables.u[92];
acadoWorkspace.ub[98] = (real_t)1.0000000000000000e+12 - acadoVariables.u[93];
acadoWorkspace.ub[99] = (real_t)1.0000000000000000e+12 - acadoVariables.u[94];
acadoWorkspace.ub[100] = (real_t)1.0000000000000000e+12 - acadoVariables.u[95];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
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
acadoWorkspace.evH[lRun1 * 3] = acadoWorkspace.conValueOut[0];
acadoWorkspace.evH[lRun1 * 3 + 1] = acadoWorkspace.conValueOut[1];
acadoWorkspace.evH[lRun1 * 3 + 2] = acadoWorkspace.conValueOut[2];

acadoWorkspace.evHx[lRun1 * 15] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 15 + 1] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHx[lRun1 * 15 + 2] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHx[lRun1 * 15 + 3] = acadoWorkspace.conValueOut[6];
acadoWorkspace.evHx[lRun1 * 15 + 4] = acadoWorkspace.conValueOut[7];
acadoWorkspace.evHx[lRun1 * 15 + 5] = acadoWorkspace.conValueOut[8];
acadoWorkspace.evHx[lRun1 * 15 + 6] = acadoWorkspace.conValueOut[9];
acadoWorkspace.evHx[lRun1 * 15 + 7] = acadoWorkspace.conValueOut[10];
acadoWorkspace.evHx[lRun1 * 15 + 8] = acadoWorkspace.conValueOut[11];
acadoWorkspace.evHx[lRun1 * 15 + 9] = acadoWorkspace.conValueOut[12];
acadoWorkspace.evHx[lRun1 * 15 + 10] = acadoWorkspace.conValueOut[13];
acadoWorkspace.evHx[lRun1 * 15 + 11] = acadoWorkspace.conValueOut[14];
acadoWorkspace.evHx[lRun1 * 15 + 12] = acadoWorkspace.conValueOut[15];
acadoWorkspace.evHx[lRun1 * 15 + 13] = acadoWorkspace.conValueOut[16];
acadoWorkspace.evHx[lRun1 * 15 + 14] = acadoWorkspace.conValueOut[17];
acadoWorkspace.evHu[lRun1 * 9] = acadoWorkspace.conValueOut[18];
acadoWorkspace.evHu[lRun1 * 9 + 1] = acadoWorkspace.conValueOut[19];
acadoWorkspace.evHu[lRun1 * 9 + 2] = acadoWorkspace.conValueOut[20];
acadoWorkspace.evHu[lRun1 * 9 + 3] = acadoWorkspace.conValueOut[21];
acadoWorkspace.evHu[lRun1 * 9 + 4] = acadoWorkspace.conValueOut[22];
acadoWorkspace.evHu[lRun1 * 9 + 5] = acadoWorkspace.conValueOut[23];
acadoWorkspace.evHu[lRun1 * 9 + 6] = acadoWorkspace.conValueOut[24];
acadoWorkspace.evHu[lRun1 * 9 + 7] = acadoWorkspace.conValueOut[25];
acadoWorkspace.evHu[lRun1 * 9 + 8] = acadoWorkspace.conValueOut[26];
}

acadoWorkspace.A[0] = acadoWorkspace.evHx[0];
acadoWorkspace.A[1] = acadoWorkspace.evHx[1];
acadoWorkspace.A[2] = acadoWorkspace.evHx[2];
acadoWorkspace.A[3] = acadoWorkspace.evHx[3];
acadoWorkspace.A[4] = acadoWorkspace.evHx[4];
acadoWorkspace.A[101] = acadoWorkspace.evHx[5];
acadoWorkspace.A[102] = acadoWorkspace.evHx[6];
acadoWorkspace.A[103] = acadoWorkspace.evHx[7];
acadoWorkspace.A[104] = acadoWorkspace.evHx[8];
acadoWorkspace.A[105] = acadoWorkspace.evHx[9];
acadoWorkspace.A[202] = acadoWorkspace.evHx[10];
acadoWorkspace.A[203] = acadoWorkspace.evHx[11];
acadoWorkspace.A[204] = acadoWorkspace.evHx[12];
acadoWorkspace.A[205] = acadoWorkspace.evHx[13];
acadoWorkspace.A[206] = acadoWorkspace.evHx[14];

acado_multHxC( &(acadoWorkspace.evHx[ 15 ]), acadoWorkspace.evGx, &(acadoWorkspace.A[ 303 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.evGx[ 25 ]), &(acadoWorkspace.A[ 606 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 45 ]), &(acadoWorkspace.evGx[ 50 ]), &(acadoWorkspace.A[ 909 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.evGx[ 75 ]), &(acadoWorkspace.A[ 1212 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 75 ]), &(acadoWorkspace.evGx[ 100 ]), &(acadoWorkspace.A[ 1515 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.evGx[ 125 ]), &(acadoWorkspace.A[ 1818 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 105 ]), &(acadoWorkspace.evGx[ 150 ]), &(acadoWorkspace.A[ 2121 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.evGx[ 175 ]), &(acadoWorkspace.A[ 2424 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 135 ]), &(acadoWorkspace.evGx[ 200 ]), &(acadoWorkspace.A[ 2727 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.A[ 3030 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 165 ]), &(acadoWorkspace.evGx[ 250 ]), &(acadoWorkspace.A[ 3333 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.evGx[ 275 ]), &(acadoWorkspace.A[ 3636 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 195 ]), &(acadoWorkspace.evGx[ 300 ]), &(acadoWorkspace.A[ 3939 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 210 ]), &(acadoWorkspace.evGx[ 325 ]), &(acadoWorkspace.A[ 4242 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 225 ]), &(acadoWorkspace.evGx[ 350 ]), &(acadoWorkspace.A[ 4545 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 240 ]), &(acadoWorkspace.evGx[ 375 ]), &(acadoWorkspace.A[ 4848 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 255 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.A[ 5151 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 270 ]), &(acadoWorkspace.evGx[ 425 ]), &(acadoWorkspace.A[ 5454 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 285 ]), &(acadoWorkspace.evGx[ 450 ]), &(acadoWorkspace.A[ 5757 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 300 ]), &(acadoWorkspace.evGx[ 475 ]), &(acadoWorkspace.A[ 6060 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 315 ]), &(acadoWorkspace.evGx[ 500 ]), &(acadoWorkspace.A[ 6363 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 330 ]), &(acadoWorkspace.evGx[ 525 ]), &(acadoWorkspace.A[ 6666 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 345 ]), &(acadoWorkspace.evGx[ 550 ]), &(acadoWorkspace.A[ 6969 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 360 ]), &(acadoWorkspace.evGx[ 575 ]), &(acadoWorkspace.A[ 7272 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 375 ]), &(acadoWorkspace.evGx[ 600 ]), &(acadoWorkspace.A[ 7575 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 390 ]), &(acadoWorkspace.evGx[ 625 ]), &(acadoWorkspace.A[ 7878 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 405 ]), &(acadoWorkspace.evGx[ 650 ]), &(acadoWorkspace.A[ 8181 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 420 ]), &(acadoWorkspace.evGx[ 675 ]), &(acadoWorkspace.A[ 8484 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 435 ]), &(acadoWorkspace.evGx[ 700 ]), &(acadoWorkspace.A[ 8787 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 450 ]), &(acadoWorkspace.evGx[ 725 ]), &(acadoWorkspace.A[ 9090 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 465 ]), &(acadoWorkspace.evGx[ 750 ]), &(acadoWorkspace.A[ 9393 ]) );

for (lRun2 = 0; lRun2 < 31; ++lRun2)
{
for (lRun3 = 0; lRun3 < lRun2 + 1; ++lRun3)
{
lRun4 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun3);
lRun5 = lRun2 + 1;
acado_multHxE( &(acadoWorkspace.evHx[ lRun2 * 15 + 15 ]), &(acadoWorkspace.E[ lRun4 * 15 ]), lRun5, lRun3 );
}
}

acadoWorkspace.A[5] = acadoWorkspace.evHu[0];
acadoWorkspace.A[6] = acadoWorkspace.evHu[1];
acadoWorkspace.A[7] = acadoWorkspace.evHu[2];
acadoWorkspace.A[106] = acadoWorkspace.evHu[3];
acadoWorkspace.A[107] = acadoWorkspace.evHu[4];
acadoWorkspace.A[108] = acadoWorkspace.evHu[5];
acadoWorkspace.A[207] = acadoWorkspace.evHu[6];
acadoWorkspace.A[208] = acadoWorkspace.evHu[7];
acadoWorkspace.A[209] = acadoWorkspace.evHu[8];
acadoWorkspace.A[311] = acadoWorkspace.evHu[9];
acadoWorkspace.A[312] = acadoWorkspace.evHu[10];
acadoWorkspace.A[313] = acadoWorkspace.evHu[11];
acadoWorkspace.A[412] = acadoWorkspace.evHu[12];
acadoWorkspace.A[413] = acadoWorkspace.evHu[13];
acadoWorkspace.A[414] = acadoWorkspace.evHu[14];
acadoWorkspace.A[513] = acadoWorkspace.evHu[15];
acadoWorkspace.A[514] = acadoWorkspace.evHu[16];
acadoWorkspace.A[515] = acadoWorkspace.evHu[17];
acadoWorkspace.A[617] = acadoWorkspace.evHu[18];
acadoWorkspace.A[618] = acadoWorkspace.evHu[19];
acadoWorkspace.A[619] = acadoWorkspace.evHu[20];
acadoWorkspace.A[718] = acadoWorkspace.evHu[21];
acadoWorkspace.A[719] = acadoWorkspace.evHu[22];
acadoWorkspace.A[720] = acadoWorkspace.evHu[23];
acadoWorkspace.A[819] = acadoWorkspace.evHu[24];
acadoWorkspace.A[820] = acadoWorkspace.evHu[25];
acadoWorkspace.A[821] = acadoWorkspace.evHu[26];
acadoWorkspace.A[923] = acadoWorkspace.evHu[27];
acadoWorkspace.A[924] = acadoWorkspace.evHu[28];
acadoWorkspace.A[925] = acadoWorkspace.evHu[29];
acadoWorkspace.A[1024] = acadoWorkspace.evHu[30];
acadoWorkspace.A[1025] = acadoWorkspace.evHu[31];
acadoWorkspace.A[1026] = acadoWorkspace.evHu[32];
acadoWorkspace.A[1125] = acadoWorkspace.evHu[33];
acadoWorkspace.A[1126] = acadoWorkspace.evHu[34];
acadoWorkspace.A[1127] = acadoWorkspace.evHu[35];
acadoWorkspace.A[1229] = acadoWorkspace.evHu[36];
acadoWorkspace.A[1230] = acadoWorkspace.evHu[37];
acadoWorkspace.A[1231] = acadoWorkspace.evHu[38];
acadoWorkspace.A[1330] = acadoWorkspace.evHu[39];
acadoWorkspace.A[1331] = acadoWorkspace.evHu[40];
acadoWorkspace.A[1332] = acadoWorkspace.evHu[41];
acadoWorkspace.A[1431] = acadoWorkspace.evHu[42];
acadoWorkspace.A[1432] = acadoWorkspace.evHu[43];
acadoWorkspace.A[1433] = acadoWorkspace.evHu[44];
acadoWorkspace.A[1535] = acadoWorkspace.evHu[45];
acadoWorkspace.A[1536] = acadoWorkspace.evHu[46];
acadoWorkspace.A[1537] = acadoWorkspace.evHu[47];
acadoWorkspace.A[1636] = acadoWorkspace.evHu[48];
acadoWorkspace.A[1637] = acadoWorkspace.evHu[49];
acadoWorkspace.A[1638] = acadoWorkspace.evHu[50];
acadoWorkspace.A[1737] = acadoWorkspace.evHu[51];
acadoWorkspace.A[1738] = acadoWorkspace.evHu[52];
acadoWorkspace.A[1739] = acadoWorkspace.evHu[53];
acadoWorkspace.A[1841] = acadoWorkspace.evHu[54];
acadoWorkspace.A[1842] = acadoWorkspace.evHu[55];
acadoWorkspace.A[1843] = acadoWorkspace.evHu[56];
acadoWorkspace.A[1942] = acadoWorkspace.evHu[57];
acadoWorkspace.A[1943] = acadoWorkspace.evHu[58];
acadoWorkspace.A[1944] = acadoWorkspace.evHu[59];
acadoWorkspace.A[2043] = acadoWorkspace.evHu[60];
acadoWorkspace.A[2044] = acadoWorkspace.evHu[61];
acadoWorkspace.A[2045] = acadoWorkspace.evHu[62];
acadoWorkspace.A[2147] = acadoWorkspace.evHu[63];
acadoWorkspace.A[2148] = acadoWorkspace.evHu[64];
acadoWorkspace.A[2149] = acadoWorkspace.evHu[65];
acadoWorkspace.A[2248] = acadoWorkspace.evHu[66];
acadoWorkspace.A[2249] = acadoWorkspace.evHu[67];
acadoWorkspace.A[2250] = acadoWorkspace.evHu[68];
acadoWorkspace.A[2349] = acadoWorkspace.evHu[69];
acadoWorkspace.A[2350] = acadoWorkspace.evHu[70];
acadoWorkspace.A[2351] = acadoWorkspace.evHu[71];
acadoWorkspace.A[2453] = acadoWorkspace.evHu[72];
acadoWorkspace.A[2454] = acadoWorkspace.evHu[73];
acadoWorkspace.A[2455] = acadoWorkspace.evHu[74];
acadoWorkspace.A[2554] = acadoWorkspace.evHu[75];
acadoWorkspace.A[2555] = acadoWorkspace.evHu[76];
acadoWorkspace.A[2556] = acadoWorkspace.evHu[77];
acadoWorkspace.A[2655] = acadoWorkspace.evHu[78];
acadoWorkspace.A[2656] = acadoWorkspace.evHu[79];
acadoWorkspace.A[2657] = acadoWorkspace.evHu[80];
acadoWorkspace.A[2759] = acadoWorkspace.evHu[81];
acadoWorkspace.A[2760] = acadoWorkspace.evHu[82];
acadoWorkspace.A[2761] = acadoWorkspace.evHu[83];
acadoWorkspace.A[2860] = acadoWorkspace.evHu[84];
acadoWorkspace.A[2861] = acadoWorkspace.evHu[85];
acadoWorkspace.A[2862] = acadoWorkspace.evHu[86];
acadoWorkspace.A[2961] = acadoWorkspace.evHu[87];
acadoWorkspace.A[2962] = acadoWorkspace.evHu[88];
acadoWorkspace.A[2963] = acadoWorkspace.evHu[89];
acadoWorkspace.A[3065] = acadoWorkspace.evHu[90];
acadoWorkspace.A[3066] = acadoWorkspace.evHu[91];
acadoWorkspace.A[3067] = acadoWorkspace.evHu[92];
acadoWorkspace.A[3166] = acadoWorkspace.evHu[93];
acadoWorkspace.A[3167] = acadoWorkspace.evHu[94];
acadoWorkspace.A[3168] = acadoWorkspace.evHu[95];
acadoWorkspace.A[3267] = acadoWorkspace.evHu[96];
acadoWorkspace.A[3268] = acadoWorkspace.evHu[97];
acadoWorkspace.A[3269] = acadoWorkspace.evHu[98];
acadoWorkspace.A[3371] = acadoWorkspace.evHu[99];
acadoWorkspace.A[3372] = acadoWorkspace.evHu[100];
acadoWorkspace.A[3373] = acadoWorkspace.evHu[101];
acadoWorkspace.A[3472] = acadoWorkspace.evHu[102];
acadoWorkspace.A[3473] = acadoWorkspace.evHu[103];
acadoWorkspace.A[3474] = acadoWorkspace.evHu[104];
acadoWorkspace.A[3573] = acadoWorkspace.evHu[105];
acadoWorkspace.A[3574] = acadoWorkspace.evHu[106];
acadoWorkspace.A[3575] = acadoWorkspace.evHu[107];
acadoWorkspace.A[3677] = acadoWorkspace.evHu[108];
acadoWorkspace.A[3678] = acadoWorkspace.evHu[109];
acadoWorkspace.A[3679] = acadoWorkspace.evHu[110];
acadoWorkspace.A[3778] = acadoWorkspace.evHu[111];
acadoWorkspace.A[3779] = acadoWorkspace.evHu[112];
acadoWorkspace.A[3780] = acadoWorkspace.evHu[113];
acadoWorkspace.A[3879] = acadoWorkspace.evHu[114];
acadoWorkspace.A[3880] = acadoWorkspace.evHu[115];
acadoWorkspace.A[3881] = acadoWorkspace.evHu[116];
acadoWorkspace.A[3983] = acadoWorkspace.evHu[117];
acadoWorkspace.A[3984] = acadoWorkspace.evHu[118];
acadoWorkspace.A[3985] = acadoWorkspace.evHu[119];
acadoWorkspace.A[4084] = acadoWorkspace.evHu[120];
acadoWorkspace.A[4085] = acadoWorkspace.evHu[121];
acadoWorkspace.A[4086] = acadoWorkspace.evHu[122];
acadoWorkspace.A[4185] = acadoWorkspace.evHu[123];
acadoWorkspace.A[4186] = acadoWorkspace.evHu[124];
acadoWorkspace.A[4187] = acadoWorkspace.evHu[125];
acadoWorkspace.A[4289] = acadoWorkspace.evHu[126];
acadoWorkspace.A[4290] = acadoWorkspace.evHu[127];
acadoWorkspace.A[4291] = acadoWorkspace.evHu[128];
acadoWorkspace.A[4390] = acadoWorkspace.evHu[129];
acadoWorkspace.A[4391] = acadoWorkspace.evHu[130];
acadoWorkspace.A[4392] = acadoWorkspace.evHu[131];
acadoWorkspace.A[4491] = acadoWorkspace.evHu[132];
acadoWorkspace.A[4492] = acadoWorkspace.evHu[133];
acadoWorkspace.A[4493] = acadoWorkspace.evHu[134];
acadoWorkspace.A[4595] = acadoWorkspace.evHu[135];
acadoWorkspace.A[4596] = acadoWorkspace.evHu[136];
acadoWorkspace.A[4597] = acadoWorkspace.evHu[137];
acadoWorkspace.A[4696] = acadoWorkspace.evHu[138];
acadoWorkspace.A[4697] = acadoWorkspace.evHu[139];
acadoWorkspace.A[4698] = acadoWorkspace.evHu[140];
acadoWorkspace.A[4797] = acadoWorkspace.evHu[141];
acadoWorkspace.A[4798] = acadoWorkspace.evHu[142];
acadoWorkspace.A[4799] = acadoWorkspace.evHu[143];
acadoWorkspace.A[4901] = acadoWorkspace.evHu[144];
acadoWorkspace.A[4902] = acadoWorkspace.evHu[145];
acadoWorkspace.A[4903] = acadoWorkspace.evHu[146];
acadoWorkspace.A[5002] = acadoWorkspace.evHu[147];
acadoWorkspace.A[5003] = acadoWorkspace.evHu[148];
acadoWorkspace.A[5004] = acadoWorkspace.evHu[149];
acadoWorkspace.A[5103] = acadoWorkspace.evHu[150];
acadoWorkspace.A[5104] = acadoWorkspace.evHu[151];
acadoWorkspace.A[5105] = acadoWorkspace.evHu[152];
acadoWorkspace.A[5207] = acadoWorkspace.evHu[153];
acadoWorkspace.A[5208] = acadoWorkspace.evHu[154];
acadoWorkspace.A[5209] = acadoWorkspace.evHu[155];
acadoWorkspace.A[5308] = acadoWorkspace.evHu[156];
acadoWorkspace.A[5309] = acadoWorkspace.evHu[157];
acadoWorkspace.A[5310] = acadoWorkspace.evHu[158];
acadoWorkspace.A[5409] = acadoWorkspace.evHu[159];
acadoWorkspace.A[5410] = acadoWorkspace.evHu[160];
acadoWorkspace.A[5411] = acadoWorkspace.evHu[161];
acadoWorkspace.A[5513] = acadoWorkspace.evHu[162];
acadoWorkspace.A[5514] = acadoWorkspace.evHu[163];
acadoWorkspace.A[5515] = acadoWorkspace.evHu[164];
acadoWorkspace.A[5614] = acadoWorkspace.evHu[165];
acadoWorkspace.A[5615] = acadoWorkspace.evHu[166];
acadoWorkspace.A[5616] = acadoWorkspace.evHu[167];
acadoWorkspace.A[5715] = acadoWorkspace.evHu[168];
acadoWorkspace.A[5716] = acadoWorkspace.evHu[169];
acadoWorkspace.A[5717] = acadoWorkspace.evHu[170];
acadoWorkspace.A[5819] = acadoWorkspace.evHu[171];
acadoWorkspace.A[5820] = acadoWorkspace.evHu[172];
acadoWorkspace.A[5821] = acadoWorkspace.evHu[173];
acadoWorkspace.A[5920] = acadoWorkspace.evHu[174];
acadoWorkspace.A[5921] = acadoWorkspace.evHu[175];
acadoWorkspace.A[5922] = acadoWorkspace.evHu[176];
acadoWorkspace.A[6021] = acadoWorkspace.evHu[177];
acadoWorkspace.A[6022] = acadoWorkspace.evHu[178];
acadoWorkspace.A[6023] = acadoWorkspace.evHu[179];
acadoWorkspace.A[6125] = acadoWorkspace.evHu[180];
acadoWorkspace.A[6126] = acadoWorkspace.evHu[181];
acadoWorkspace.A[6127] = acadoWorkspace.evHu[182];
acadoWorkspace.A[6226] = acadoWorkspace.evHu[183];
acadoWorkspace.A[6227] = acadoWorkspace.evHu[184];
acadoWorkspace.A[6228] = acadoWorkspace.evHu[185];
acadoWorkspace.A[6327] = acadoWorkspace.evHu[186];
acadoWorkspace.A[6328] = acadoWorkspace.evHu[187];
acadoWorkspace.A[6329] = acadoWorkspace.evHu[188];
acadoWorkspace.A[6431] = acadoWorkspace.evHu[189];
acadoWorkspace.A[6432] = acadoWorkspace.evHu[190];
acadoWorkspace.A[6433] = acadoWorkspace.evHu[191];
acadoWorkspace.A[6532] = acadoWorkspace.evHu[192];
acadoWorkspace.A[6533] = acadoWorkspace.evHu[193];
acadoWorkspace.A[6534] = acadoWorkspace.evHu[194];
acadoWorkspace.A[6633] = acadoWorkspace.evHu[195];
acadoWorkspace.A[6634] = acadoWorkspace.evHu[196];
acadoWorkspace.A[6635] = acadoWorkspace.evHu[197];
acadoWorkspace.A[6737] = acadoWorkspace.evHu[198];
acadoWorkspace.A[6738] = acadoWorkspace.evHu[199];
acadoWorkspace.A[6739] = acadoWorkspace.evHu[200];
acadoWorkspace.A[6838] = acadoWorkspace.evHu[201];
acadoWorkspace.A[6839] = acadoWorkspace.evHu[202];
acadoWorkspace.A[6840] = acadoWorkspace.evHu[203];
acadoWorkspace.A[6939] = acadoWorkspace.evHu[204];
acadoWorkspace.A[6940] = acadoWorkspace.evHu[205];
acadoWorkspace.A[6941] = acadoWorkspace.evHu[206];
acadoWorkspace.A[7043] = acadoWorkspace.evHu[207];
acadoWorkspace.A[7044] = acadoWorkspace.evHu[208];
acadoWorkspace.A[7045] = acadoWorkspace.evHu[209];
acadoWorkspace.A[7144] = acadoWorkspace.evHu[210];
acadoWorkspace.A[7145] = acadoWorkspace.evHu[211];
acadoWorkspace.A[7146] = acadoWorkspace.evHu[212];
acadoWorkspace.A[7245] = acadoWorkspace.evHu[213];
acadoWorkspace.A[7246] = acadoWorkspace.evHu[214];
acadoWorkspace.A[7247] = acadoWorkspace.evHu[215];
acadoWorkspace.A[7349] = acadoWorkspace.evHu[216];
acadoWorkspace.A[7350] = acadoWorkspace.evHu[217];
acadoWorkspace.A[7351] = acadoWorkspace.evHu[218];
acadoWorkspace.A[7450] = acadoWorkspace.evHu[219];
acadoWorkspace.A[7451] = acadoWorkspace.evHu[220];
acadoWorkspace.A[7452] = acadoWorkspace.evHu[221];
acadoWorkspace.A[7551] = acadoWorkspace.evHu[222];
acadoWorkspace.A[7552] = acadoWorkspace.evHu[223];
acadoWorkspace.A[7553] = acadoWorkspace.evHu[224];
acadoWorkspace.A[7655] = acadoWorkspace.evHu[225];
acadoWorkspace.A[7656] = acadoWorkspace.evHu[226];
acadoWorkspace.A[7657] = acadoWorkspace.evHu[227];
acadoWorkspace.A[7756] = acadoWorkspace.evHu[228];
acadoWorkspace.A[7757] = acadoWorkspace.evHu[229];
acadoWorkspace.A[7758] = acadoWorkspace.evHu[230];
acadoWorkspace.A[7857] = acadoWorkspace.evHu[231];
acadoWorkspace.A[7858] = acadoWorkspace.evHu[232];
acadoWorkspace.A[7859] = acadoWorkspace.evHu[233];
acadoWorkspace.A[7961] = acadoWorkspace.evHu[234];
acadoWorkspace.A[7962] = acadoWorkspace.evHu[235];
acadoWorkspace.A[7963] = acadoWorkspace.evHu[236];
acadoWorkspace.A[8062] = acadoWorkspace.evHu[237];
acadoWorkspace.A[8063] = acadoWorkspace.evHu[238];
acadoWorkspace.A[8064] = acadoWorkspace.evHu[239];
acadoWorkspace.A[8163] = acadoWorkspace.evHu[240];
acadoWorkspace.A[8164] = acadoWorkspace.evHu[241];
acadoWorkspace.A[8165] = acadoWorkspace.evHu[242];
acadoWorkspace.A[8267] = acadoWorkspace.evHu[243];
acadoWorkspace.A[8268] = acadoWorkspace.evHu[244];
acadoWorkspace.A[8269] = acadoWorkspace.evHu[245];
acadoWorkspace.A[8368] = acadoWorkspace.evHu[246];
acadoWorkspace.A[8369] = acadoWorkspace.evHu[247];
acadoWorkspace.A[8370] = acadoWorkspace.evHu[248];
acadoWorkspace.A[8469] = acadoWorkspace.evHu[249];
acadoWorkspace.A[8470] = acadoWorkspace.evHu[250];
acadoWorkspace.A[8471] = acadoWorkspace.evHu[251];
acadoWorkspace.A[8573] = acadoWorkspace.evHu[252];
acadoWorkspace.A[8574] = acadoWorkspace.evHu[253];
acadoWorkspace.A[8575] = acadoWorkspace.evHu[254];
acadoWorkspace.A[8674] = acadoWorkspace.evHu[255];
acadoWorkspace.A[8675] = acadoWorkspace.evHu[256];
acadoWorkspace.A[8676] = acadoWorkspace.evHu[257];
acadoWorkspace.A[8775] = acadoWorkspace.evHu[258];
acadoWorkspace.A[8776] = acadoWorkspace.evHu[259];
acadoWorkspace.A[8777] = acadoWorkspace.evHu[260];
acadoWorkspace.A[8879] = acadoWorkspace.evHu[261];
acadoWorkspace.A[8880] = acadoWorkspace.evHu[262];
acadoWorkspace.A[8881] = acadoWorkspace.evHu[263];
acadoWorkspace.A[8980] = acadoWorkspace.evHu[264];
acadoWorkspace.A[8981] = acadoWorkspace.evHu[265];
acadoWorkspace.A[8982] = acadoWorkspace.evHu[266];
acadoWorkspace.A[9081] = acadoWorkspace.evHu[267];
acadoWorkspace.A[9082] = acadoWorkspace.evHu[268];
acadoWorkspace.A[9083] = acadoWorkspace.evHu[269];
acadoWorkspace.A[9185] = acadoWorkspace.evHu[270];
acadoWorkspace.A[9186] = acadoWorkspace.evHu[271];
acadoWorkspace.A[9187] = acadoWorkspace.evHu[272];
acadoWorkspace.A[9286] = acadoWorkspace.evHu[273];
acadoWorkspace.A[9287] = acadoWorkspace.evHu[274];
acadoWorkspace.A[9288] = acadoWorkspace.evHu[275];
acadoWorkspace.A[9387] = acadoWorkspace.evHu[276];
acadoWorkspace.A[9388] = acadoWorkspace.evHu[277];
acadoWorkspace.A[9389] = acadoWorkspace.evHu[278];
acadoWorkspace.A[9491] = acadoWorkspace.evHu[279];
acadoWorkspace.A[9492] = acadoWorkspace.evHu[280];
acadoWorkspace.A[9493] = acadoWorkspace.evHu[281];
acadoWorkspace.A[9592] = acadoWorkspace.evHu[282];
acadoWorkspace.A[9593] = acadoWorkspace.evHu[283];
acadoWorkspace.A[9594] = acadoWorkspace.evHu[284];
acadoWorkspace.A[9693] = acadoWorkspace.evHu[285];
acadoWorkspace.A[9694] = acadoWorkspace.evHu[286];
acadoWorkspace.A[9695] = acadoWorkspace.evHu[287];
acadoWorkspace.lbA[0] = - acadoWorkspace.evH[0];
acadoWorkspace.lbA[1] = - acadoWorkspace.evH[1];
acadoWorkspace.lbA[2] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[2];
acadoWorkspace.lbA[3] = - acadoWorkspace.evH[3];
acadoWorkspace.lbA[4] = - acadoWorkspace.evH[4];
acadoWorkspace.lbA[5] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[5];
acadoWorkspace.lbA[6] = - acadoWorkspace.evH[6];
acadoWorkspace.lbA[7] = - acadoWorkspace.evH[7];
acadoWorkspace.lbA[8] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[8];
acadoWorkspace.lbA[9] = - acadoWorkspace.evH[9];
acadoWorkspace.lbA[10] = - acadoWorkspace.evH[10];
acadoWorkspace.lbA[11] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[11];
acadoWorkspace.lbA[12] = - acadoWorkspace.evH[12];
acadoWorkspace.lbA[13] = - acadoWorkspace.evH[13];
acadoWorkspace.lbA[14] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[14];
acadoWorkspace.lbA[15] = - acadoWorkspace.evH[15];
acadoWorkspace.lbA[16] = - acadoWorkspace.evH[16];
acadoWorkspace.lbA[17] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[17];
acadoWorkspace.lbA[18] = - acadoWorkspace.evH[18];
acadoWorkspace.lbA[19] = - acadoWorkspace.evH[19];
acadoWorkspace.lbA[20] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[20];
acadoWorkspace.lbA[21] = - acadoWorkspace.evH[21];
acadoWorkspace.lbA[22] = - acadoWorkspace.evH[22];
acadoWorkspace.lbA[23] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[23];
acadoWorkspace.lbA[24] = - acadoWorkspace.evH[24];
acadoWorkspace.lbA[25] = - acadoWorkspace.evH[25];
acadoWorkspace.lbA[26] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[26];
acadoWorkspace.lbA[27] = - acadoWorkspace.evH[27];
acadoWorkspace.lbA[28] = - acadoWorkspace.evH[28];
acadoWorkspace.lbA[29] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[29];
acadoWorkspace.lbA[30] = - acadoWorkspace.evH[30];
acadoWorkspace.lbA[31] = - acadoWorkspace.evH[31];
acadoWorkspace.lbA[32] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[32];
acadoWorkspace.lbA[33] = - acadoWorkspace.evH[33];
acadoWorkspace.lbA[34] = - acadoWorkspace.evH[34];
acadoWorkspace.lbA[35] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[35];
acadoWorkspace.lbA[36] = - acadoWorkspace.evH[36];
acadoWorkspace.lbA[37] = - acadoWorkspace.evH[37];
acadoWorkspace.lbA[38] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[38];
acadoWorkspace.lbA[39] = - acadoWorkspace.evH[39];
acadoWorkspace.lbA[40] = - acadoWorkspace.evH[40];
acadoWorkspace.lbA[41] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[41];
acadoWorkspace.lbA[42] = - acadoWorkspace.evH[42];
acadoWorkspace.lbA[43] = - acadoWorkspace.evH[43];
acadoWorkspace.lbA[44] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[44];
acadoWorkspace.lbA[45] = - acadoWorkspace.evH[45];
acadoWorkspace.lbA[46] = - acadoWorkspace.evH[46];
acadoWorkspace.lbA[47] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[47];
acadoWorkspace.lbA[48] = - acadoWorkspace.evH[48];
acadoWorkspace.lbA[49] = - acadoWorkspace.evH[49];
acadoWorkspace.lbA[50] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[50];
acadoWorkspace.lbA[51] = - acadoWorkspace.evH[51];
acadoWorkspace.lbA[52] = - acadoWorkspace.evH[52];
acadoWorkspace.lbA[53] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[53];
acadoWorkspace.lbA[54] = - acadoWorkspace.evH[54];
acadoWorkspace.lbA[55] = - acadoWorkspace.evH[55];
acadoWorkspace.lbA[56] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[56];
acadoWorkspace.lbA[57] = - acadoWorkspace.evH[57];
acadoWorkspace.lbA[58] = - acadoWorkspace.evH[58];
acadoWorkspace.lbA[59] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[59];
acadoWorkspace.lbA[60] = - acadoWorkspace.evH[60];
acadoWorkspace.lbA[61] = - acadoWorkspace.evH[61];
acadoWorkspace.lbA[62] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[62];
acadoWorkspace.lbA[63] = - acadoWorkspace.evH[63];
acadoWorkspace.lbA[64] = - acadoWorkspace.evH[64];
acadoWorkspace.lbA[65] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[65];
acadoWorkspace.lbA[66] = - acadoWorkspace.evH[66];
acadoWorkspace.lbA[67] = - acadoWorkspace.evH[67];
acadoWorkspace.lbA[68] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[68];
acadoWorkspace.lbA[69] = - acadoWorkspace.evH[69];
acadoWorkspace.lbA[70] = - acadoWorkspace.evH[70];
acadoWorkspace.lbA[71] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[71];
acadoWorkspace.lbA[72] = - acadoWorkspace.evH[72];
acadoWorkspace.lbA[73] = - acadoWorkspace.evH[73];
acadoWorkspace.lbA[74] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[74];
acadoWorkspace.lbA[75] = - acadoWorkspace.evH[75];
acadoWorkspace.lbA[76] = - acadoWorkspace.evH[76];
acadoWorkspace.lbA[77] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[77];
acadoWorkspace.lbA[78] = - acadoWorkspace.evH[78];
acadoWorkspace.lbA[79] = - acadoWorkspace.evH[79];
acadoWorkspace.lbA[80] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[80];
acadoWorkspace.lbA[81] = - acadoWorkspace.evH[81];
acadoWorkspace.lbA[82] = - acadoWorkspace.evH[82];
acadoWorkspace.lbA[83] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[83];
acadoWorkspace.lbA[84] = - acadoWorkspace.evH[84];
acadoWorkspace.lbA[85] = - acadoWorkspace.evH[85];
acadoWorkspace.lbA[86] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[86];
acadoWorkspace.lbA[87] = - acadoWorkspace.evH[87];
acadoWorkspace.lbA[88] = - acadoWorkspace.evH[88];
acadoWorkspace.lbA[89] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[89];
acadoWorkspace.lbA[90] = - acadoWorkspace.evH[90];
acadoWorkspace.lbA[91] = - acadoWorkspace.evH[91];
acadoWorkspace.lbA[92] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[92];
acadoWorkspace.lbA[93] = - acadoWorkspace.evH[93];
acadoWorkspace.lbA[94] = - acadoWorkspace.evH[94];
acadoWorkspace.lbA[95] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[95];

acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[0];
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[1];
acadoWorkspace.ubA[2] = - acadoWorkspace.evH[2];
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[3];
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[4];
acadoWorkspace.ubA[5] = - acadoWorkspace.evH[5];
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[6];
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[7];
acadoWorkspace.ubA[8] = - acadoWorkspace.evH[8];
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[9];
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[10];
acadoWorkspace.ubA[11] = - acadoWorkspace.evH[11];
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[12];
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[13];
acadoWorkspace.ubA[14] = - acadoWorkspace.evH[14];
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[15];
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[16];
acadoWorkspace.ubA[17] = - acadoWorkspace.evH[17];
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[18];
acadoWorkspace.ubA[19] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[19];
acadoWorkspace.ubA[20] = - acadoWorkspace.evH[20];
acadoWorkspace.ubA[21] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[21];
acadoWorkspace.ubA[22] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[22];
acadoWorkspace.ubA[23] = - acadoWorkspace.evH[23];
acadoWorkspace.ubA[24] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[24];
acadoWorkspace.ubA[25] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[25];
acadoWorkspace.ubA[26] = - acadoWorkspace.evH[26];
acadoWorkspace.ubA[27] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[27];
acadoWorkspace.ubA[28] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[28];
acadoWorkspace.ubA[29] = - acadoWorkspace.evH[29];
acadoWorkspace.ubA[30] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[30];
acadoWorkspace.ubA[31] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[31];
acadoWorkspace.ubA[32] = - acadoWorkspace.evH[32];
acadoWorkspace.ubA[33] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[33];
acadoWorkspace.ubA[34] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[34];
acadoWorkspace.ubA[35] = - acadoWorkspace.evH[35];
acadoWorkspace.ubA[36] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[36];
acadoWorkspace.ubA[37] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[37];
acadoWorkspace.ubA[38] = - acadoWorkspace.evH[38];
acadoWorkspace.ubA[39] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[39];
acadoWorkspace.ubA[40] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[40];
acadoWorkspace.ubA[41] = - acadoWorkspace.evH[41];
acadoWorkspace.ubA[42] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[42];
acadoWorkspace.ubA[43] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[43];
acadoWorkspace.ubA[44] = - acadoWorkspace.evH[44];
acadoWorkspace.ubA[45] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[45];
acadoWorkspace.ubA[46] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[46];
acadoWorkspace.ubA[47] = - acadoWorkspace.evH[47];
acadoWorkspace.ubA[48] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[48];
acadoWorkspace.ubA[49] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[49];
acadoWorkspace.ubA[50] = - acadoWorkspace.evH[50];
acadoWorkspace.ubA[51] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[51];
acadoWorkspace.ubA[52] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[52];
acadoWorkspace.ubA[53] = - acadoWorkspace.evH[53];
acadoWorkspace.ubA[54] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[54];
acadoWorkspace.ubA[55] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[55];
acadoWorkspace.ubA[56] = - acadoWorkspace.evH[56];
acadoWorkspace.ubA[57] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[57];
acadoWorkspace.ubA[58] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[58];
acadoWorkspace.ubA[59] = - acadoWorkspace.evH[59];
acadoWorkspace.ubA[60] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[60];
acadoWorkspace.ubA[61] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[61];
acadoWorkspace.ubA[62] = - acadoWorkspace.evH[62];
acadoWorkspace.ubA[63] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[63];
acadoWorkspace.ubA[64] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[64];
acadoWorkspace.ubA[65] = - acadoWorkspace.evH[65];
acadoWorkspace.ubA[66] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[66];
acadoWorkspace.ubA[67] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[67];
acadoWorkspace.ubA[68] = - acadoWorkspace.evH[68];
acadoWorkspace.ubA[69] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[69];
acadoWorkspace.ubA[70] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[70];
acadoWorkspace.ubA[71] = - acadoWorkspace.evH[71];
acadoWorkspace.ubA[72] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[72];
acadoWorkspace.ubA[73] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[73];
acadoWorkspace.ubA[74] = - acadoWorkspace.evH[74];
acadoWorkspace.ubA[75] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[75];
acadoWorkspace.ubA[76] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[76];
acadoWorkspace.ubA[77] = - acadoWorkspace.evH[77];
acadoWorkspace.ubA[78] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[78];
acadoWorkspace.ubA[79] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[79];
acadoWorkspace.ubA[80] = - acadoWorkspace.evH[80];
acadoWorkspace.ubA[81] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[81];
acadoWorkspace.ubA[82] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[82];
acadoWorkspace.ubA[83] = - acadoWorkspace.evH[83];
acadoWorkspace.ubA[84] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[84];
acadoWorkspace.ubA[85] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[85];
acadoWorkspace.ubA[86] = - acadoWorkspace.evH[86];
acadoWorkspace.ubA[87] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[87];
acadoWorkspace.ubA[88] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[88];
acadoWorkspace.ubA[89] = - acadoWorkspace.evH[89];
acadoWorkspace.ubA[90] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[90];
acadoWorkspace.ubA[91] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[91];
acadoWorkspace.ubA[92] = - acadoWorkspace.evH[92];
acadoWorkspace.ubA[93] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[93];
acadoWorkspace.ubA[94] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[94];
acadoWorkspace.ubA[95] = - acadoWorkspace.evH[95];

acado_macHxd( &(acadoWorkspace.evHx[ 15 ]), acadoWorkspace.d, &(acadoWorkspace.lbA[ 3 ]), &(acadoWorkspace.ubA[ 3 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.d[ 5 ]), &(acadoWorkspace.lbA[ 6 ]), &(acadoWorkspace.ubA[ 6 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 45 ]), &(acadoWorkspace.d[ 10 ]), &(acadoWorkspace.lbA[ 9 ]), &(acadoWorkspace.ubA[ 9 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.lbA[ 12 ]), &(acadoWorkspace.ubA[ 12 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 75 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.lbA[ 15 ]), &(acadoWorkspace.ubA[ 15 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.d[ 25 ]), &(acadoWorkspace.lbA[ 18 ]), &(acadoWorkspace.ubA[ 18 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 105 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.lbA[ 21 ]), &(acadoWorkspace.ubA[ 21 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.d[ 35 ]), &(acadoWorkspace.lbA[ 24 ]), &(acadoWorkspace.ubA[ 24 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 135 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.lbA[ 27 ]), &(acadoWorkspace.ubA[ 27 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.lbA[ 30 ]), &(acadoWorkspace.ubA[ 30 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 165 ]), &(acadoWorkspace.d[ 50 ]), &(acadoWorkspace.lbA[ 33 ]), &(acadoWorkspace.ubA[ 33 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.d[ 55 ]), &(acadoWorkspace.lbA[ 36 ]), &(acadoWorkspace.ubA[ 36 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 195 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.lbA[ 39 ]), &(acadoWorkspace.ubA[ 39 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 210 ]), &(acadoWorkspace.d[ 65 ]), &(acadoWorkspace.lbA[ 42 ]), &(acadoWorkspace.ubA[ 42 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 225 ]), &(acadoWorkspace.d[ 70 ]), &(acadoWorkspace.lbA[ 45 ]), &(acadoWorkspace.ubA[ 45 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 240 ]), &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.lbA[ 48 ]), &(acadoWorkspace.ubA[ 48 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 255 ]), &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.lbA[ 51 ]), &(acadoWorkspace.ubA[ 51 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 270 ]), &(acadoWorkspace.d[ 85 ]), &(acadoWorkspace.lbA[ 54 ]), &(acadoWorkspace.ubA[ 54 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 285 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.lbA[ 57 ]), &(acadoWorkspace.ubA[ 57 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 300 ]), &(acadoWorkspace.d[ 95 ]), &(acadoWorkspace.lbA[ 60 ]), &(acadoWorkspace.ubA[ 60 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 315 ]), &(acadoWorkspace.d[ 100 ]), &(acadoWorkspace.lbA[ 63 ]), &(acadoWorkspace.ubA[ 63 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 330 ]), &(acadoWorkspace.d[ 105 ]), &(acadoWorkspace.lbA[ 66 ]), &(acadoWorkspace.ubA[ 66 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 345 ]), &(acadoWorkspace.d[ 110 ]), &(acadoWorkspace.lbA[ 69 ]), &(acadoWorkspace.ubA[ 69 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 360 ]), &(acadoWorkspace.d[ 115 ]), &(acadoWorkspace.lbA[ 72 ]), &(acadoWorkspace.ubA[ 72 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 375 ]), &(acadoWorkspace.d[ 120 ]), &(acadoWorkspace.lbA[ 75 ]), &(acadoWorkspace.ubA[ 75 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 390 ]), &(acadoWorkspace.d[ 125 ]), &(acadoWorkspace.lbA[ 78 ]), &(acadoWorkspace.ubA[ 78 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 405 ]), &(acadoWorkspace.d[ 130 ]), &(acadoWorkspace.lbA[ 81 ]), &(acadoWorkspace.ubA[ 81 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 420 ]), &(acadoWorkspace.d[ 135 ]), &(acadoWorkspace.lbA[ 84 ]), &(acadoWorkspace.ubA[ 84 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 435 ]), &(acadoWorkspace.d[ 140 ]), &(acadoWorkspace.lbA[ 87 ]), &(acadoWorkspace.ubA[ 87 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 450 ]), &(acadoWorkspace.d[ 145 ]), &(acadoWorkspace.lbA[ 90 ]), &(acadoWorkspace.ubA[ 90 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 465 ]), &(acadoWorkspace.d[ 150 ]), &(acadoWorkspace.lbA[ 93 ]), &(acadoWorkspace.ubA[ 93 ]) );

}

void acado_condenseFdb(  )
{
int lRun1;
int lRun2;
int lRun3;
acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];
acadoWorkspace.Dx0[4] = acadoVariables.x0[4] - acadoVariables.x[4];

for (lRun2 = 0; lRun2 < 192; ++lRun2)
acadoWorkspace.Dy[lRun2] -= acadoVariables.y[lRun2];

acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 5 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 18 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 36 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 54 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 72 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 90 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 108 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 126 ]), &(acadoWorkspace.Dy[ 42 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 144 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 162 ]), &(acadoWorkspace.Dy[ 54 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 180 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 198 ]), &(acadoWorkspace.Dy[ 66 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 216 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 234 ]), &(acadoWorkspace.Dy[ 78 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 252 ]), &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 270 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 288 ]), &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.g[ 53 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 306 ]), &(acadoWorkspace.Dy[ 102 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 324 ]), &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.g[ 59 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 342 ]), &(acadoWorkspace.Dy[ 114 ]), &(acadoWorkspace.g[ 62 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 360 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.g[ 65 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 378 ]), &(acadoWorkspace.Dy[ 126 ]), &(acadoWorkspace.g[ 68 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 396 ]), &(acadoWorkspace.Dy[ 132 ]), &(acadoWorkspace.g[ 71 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 414 ]), &(acadoWorkspace.Dy[ 138 ]), &(acadoWorkspace.g[ 74 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 432 ]), &(acadoWorkspace.Dy[ 144 ]), &(acadoWorkspace.g[ 77 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 450 ]), &(acadoWorkspace.Dy[ 150 ]), &(acadoWorkspace.g[ 80 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 468 ]), &(acadoWorkspace.Dy[ 156 ]), &(acadoWorkspace.g[ 83 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 486 ]), &(acadoWorkspace.Dy[ 162 ]), &(acadoWorkspace.g[ 86 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 504 ]), &(acadoWorkspace.Dy[ 168 ]), &(acadoWorkspace.g[ 89 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 522 ]), &(acadoWorkspace.Dy[ 174 ]), &(acadoWorkspace.g[ 92 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 540 ]), &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.g[ 95 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 558 ]), &(acadoWorkspace.Dy[ 186 ]), &(acadoWorkspace.g[ 98 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 30 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.QDy[ 5 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 60 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 10 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 90 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.QDy[ 15 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 120 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 20 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 150 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.QDy[ 25 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 180 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.QDy[ 30 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 210 ]), &(acadoWorkspace.Dy[ 42 ]), &(acadoWorkspace.QDy[ 35 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 240 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.QDy[ 40 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 270 ]), &(acadoWorkspace.Dy[ 54 ]), &(acadoWorkspace.QDy[ 45 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 300 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 50 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 330 ]), &(acadoWorkspace.Dy[ 66 ]), &(acadoWorkspace.QDy[ 55 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 360 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 390 ]), &(acadoWorkspace.Dy[ 78 ]), &(acadoWorkspace.QDy[ 65 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 420 ]), &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.QDy[ 70 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 450 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.QDy[ 75 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 480 ]), &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.QDy[ 80 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 510 ]), &(acadoWorkspace.Dy[ 102 ]), &(acadoWorkspace.QDy[ 85 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 540 ]), &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.QDy[ 90 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 570 ]), &(acadoWorkspace.Dy[ 114 ]), &(acadoWorkspace.QDy[ 95 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 600 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.QDy[ 100 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 630 ]), &(acadoWorkspace.Dy[ 126 ]), &(acadoWorkspace.QDy[ 105 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 660 ]), &(acadoWorkspace.Dy[ 132 ]), &(acadoWorkspace.QDy[ 110 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 690 ]), &(acadoWorkspace.Dy[ 138 ]), &(acadoWorkspace.QDy[ 115 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 720 ]), &(acadoWorkspace.Dy[ 144 ]), &(acadoWorkspace.QDy[ 120 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 750 ]), &(acadoWorkspace.Dy[ 150 ]), &(acadoWorkspace.QDy[ 125 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 780 ]), &(acadoWorkspace.Dy[ 156 ]), &(acadoWorkspace.QDy[ 130 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 810 ]), &(acadoWorkspace.Dy[ 162 ]), &(acadoWorkspace.QDy[ 135 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 840 ]), &(acadoWorkspace.Dy[ 168 ]), &(acadoWorkspace.QDy[ 140 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 870 ]), &(acadoWorkspace.Dy[ 174 ]), &(acadoWorkspace.QDy[ 145 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 900 ]), &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.QDy[ 150 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 930 ]), &(acadoWorkspace.Dy[ 186 ]), &(acadoWorkspace.QDy[ 155 ]) );

acadoWorkspace.QDy[160] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[161] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[162] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[163] = + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[164] = + acadoWorkspace.QN2[12]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[13]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[14]*acadoWorkspace.DyN[2];

for (lRun2 = 0; lRun2 < 160; ++lRun2)
acadoWorkspace.QDy[lRun2 + 5] += acadoWorkspace.Qd[lRun2];


acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[320]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[325]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[330]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[335]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[340]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[345]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[350]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[355]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[360]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[365]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[370]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[375]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[380]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[385]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[390]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[395]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[400]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[405]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[410]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[415]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[420]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[425]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[430]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[435]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[440]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[445]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[450]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[455]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[460]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[465]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[470]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[475]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[480]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[485]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[490]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[495]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[500]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[505]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[510]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[515]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[520]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[525]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[530]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[535]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[540]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[545]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[550]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[555]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[560]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[565]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[570]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[575]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[580]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[585]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[590]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[595]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[600]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[605]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[610]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[615]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[620]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[625]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[630]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[635]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[640]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[645]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[650]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[655]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[660]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[665]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[670]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[675]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[680]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[685]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[690]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[695]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[700]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[705]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[710]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[715]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[720]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[725]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[730]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[735]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[740]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[745]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[750]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[755]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[760]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[765]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[770]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[775]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[780]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[785]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[790]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[795]*acadoWorkspace.QDy[164];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[321]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[326]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[331]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[336]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[341]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[346]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[351]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[356]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[361]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[366]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[371]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[376]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[381]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[386]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[391]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[396]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[401]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[406]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[411]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[416]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[421]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[426]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[431]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[436]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[441]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[446]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[451]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[456]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[461]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[466]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[471]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[476]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[481]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[486]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[491]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[496]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[501]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[506]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[511]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[516]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[521]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[526]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[531]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[536]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[541]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[546]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[551]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[556]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[561]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[566]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[571]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[576]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[581]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[586]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[591]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[596]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[601]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[606]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[611]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[616]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[621]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[626]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[631]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[636]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[641]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[646]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[651]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[656]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[661]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[666]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[671]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[676]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[681]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[686]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[691]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[696]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[701]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[706]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[711]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[716]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[721]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[726]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[731]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[736]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[741]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[746]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[751]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[756]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[761]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[766]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[771]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[776]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[781]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[786]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[791]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[796]*acadoWorkspace.QDy[164];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[322]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[327]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[332]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[337]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[342]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[347]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[352]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[357]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[362]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[367]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[372]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[377]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[382]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[387]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[392]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[397]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[402]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[407]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[412]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[417]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[422]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[427]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[432]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[437]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[442]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[447]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[452]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[457]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[462]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[467]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[472]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[477]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[482]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[487]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[492]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[497]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[502]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[507]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[512]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[517]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[522]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[527]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[532]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[537]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[542]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[547]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[552]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[557]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[562]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[567]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[572]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[577]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[582]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[587]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[592]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[597]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[602]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[607]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[612]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[617]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[622]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[627]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[632]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[637]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[642]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[647]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[652]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[657]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[662]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[667]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[672]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[677]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[682]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[687]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[692]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[697]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[702]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[707]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[712]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[717]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[722]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[727]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[732]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[737]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[742]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[747]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[752]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[757]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[762]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[767]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[772]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[777]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[782]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[787]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[792]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[797]*acadoWorkspace.QDy[164];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[323]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[328]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[333]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[338]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[343]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[348]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[353]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[358]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[363]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[368]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[373]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[378]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[383]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[388]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[393]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[398]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[403]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[408]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[413]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[418]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[423]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[428]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[433]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[438]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[443]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[448]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[453]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[458]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[463]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[468]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[473]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[478]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[483]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[488]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[493]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[498]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[503]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[508]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[513]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[518]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[523]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[528]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[533]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[538]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[543]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[548]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[553]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[558]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[563]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[568]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[573]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[578]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[583]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[588]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[593]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[598]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[603]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[608]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[613]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[618]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[623]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[628]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[633]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[638]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[643]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[648]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[653]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[658]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[663]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[668]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[673]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[678]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[683]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[688]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[693]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[698]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[703]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[708]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[713]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[718]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[723]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[728]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[733]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[738]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[743]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[748]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[753]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[758]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[763]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[768]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[773]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[778]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[783]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[788]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[793]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[798]*acadoWorkspace.QDy[164];
acadoWorkspace.g[4] = + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[324]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[329]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[334]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[339]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[344]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[349]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[354]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[359]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[364]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[369]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[374]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[379]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[384]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[389]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[394]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[399]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[404]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[409]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[414]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[419]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[424]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[429]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[434]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[439]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[444]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[449]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[454]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[459]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[464]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[469]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[474]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[479]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[484]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[489]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[494]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[499]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[504]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[509]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[514]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[519]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[524]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[529]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[534]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[539]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[544]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[549]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[554]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[559]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[564]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[569]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[574]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[579]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[584]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[589]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[594]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[599]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[604]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[609]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[614]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[619]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[624]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[629]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[634]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[639]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[644]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[649]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[654]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[659]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[664]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[669]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[674]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[679]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[684]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[689]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[694]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[699]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[704]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[709]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[714]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[719]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[724]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[729]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[734]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[739]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[744]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[749]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[754]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[759]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[764]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[769]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[774]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[779]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[784]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[789]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[794]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[799]*acadoWorkspace.QDy[164];


for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multEQDy( &(acadoWorkspace.E[ lRun3 * 15 ]), &(acadoWorkspace.QDy[ lRun2 * 5 + 5 ]), &(acadoWorkspace.g[ lRun1 * 3 + 5 ]) );
}
}

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
}

void acado_expand(  )
{
int lRun1;
int lRun2;
int lRun3;
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
acadoVariables.u[60] += acadoWorkspace.x[65];
acadoVariables.u[61] += acadoWorkspace.x[66];
acadoVariables.u[62] += acadoWorkspace.x[67];
acadoVariables.u[63] += acadoWorkspace.x[68];
acadoVariables.u[64] += acadoWorkspace.x[69];
acadoVariables.u[65] += acadoWorkspace.x[70];
acadoVariables.u[66] += acadoWorkspace.x[71];
acadoVariables.u[67] += acadoWorkspace.x[72];
acadoVariables.u[68] += acadoWorkspace.x[73];
acadoVariables.u[69] += acadoWorkspace.x[74];
acadoVariables.u[70] += acadoWorkspace.x[75];
acadoVariables.u[71] += acadoWorkspace.x[76];
acadoVariables.u[72] += acadoWorkspace.x[77];
acadoVariables.u[73] += acadoWorkspace.x[78];
acadoVariables.u[74] += acadoWorkspace.x[79];
acadoVariables.u[75] += acadoWorkspace.x[80];
acadoVariables.u[76] += acadoWorkspace.x[81];
acadoVariables.u[77] += acadoWorkspace.x[82];
acadoVariables.u[78] += acadoWorkspace.x[83];
acadoVariables.u[79] += acadoWorkspace.x[84];
acadoVariables.u[80] += acadoWorkspace.x[85];
acadoVariables.u[81] += acadoWorkspace.x[86];
acadoVariables.u[82] += acadoWorkspace.x[87];
acadoVariables.u[83] += acadoWorkspace.x[88];
acadoVariables.u[84] += acadoWorkspace.x[89];
acadoVariables.u[85] += acadoWorkspace.x[90];
acadoVariables.u[86] += acadoWorkspace.x[91];
acadoVariables.u[87] += acadoWorkspace.x[92];
acadoVariables.u[88] += acadoWorkspace.x[93];
acadoVariables.u[89] += acadoWorkspace.x[94];
acadoVariables.u[90] += acadoWorkspace.x[95];
acadoVariables.u[91] += acadoWorkspace.x[96];
acadoVariables.u[92] += acadoWorkspace.x[97];
acadoVariables.u[93] += acadoWorkspace.x[98];
acadoVariables.u[94] += acadoWorkspace.x[99];
acadoVariables.u[95] += acadoWorkspace.x[100];

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
acadoVariables.x[105] += + acadoWorkspace.evGx[500]*acadoWorkspace.x[0] + acadoWorkspace.evGx[501]*acadoWorkspace.x[1] + acadoWorkspace.evGx[502]*acadoWorkspace.x[2] + acadoWorkspace.evGx[503]*acadoWorkspace.x[3] + acadoWorkspace.evGx[504]*acadoWorkspace.x[4] + acadoWorkspace.d[100];
acadoVariables.x[106] += + acadoWorkspace.evGx[505]*acadoWorkspace.x[0] + acadoWorkspace.evGx[506]*acadoWorkspace.x[1] + acadoWorkspace.evGx[507]*acadoWorkspace.x[2] + acadoWorkspace.evGx[508]*acadoWorkspace.x[3] + acadoWorkspace.evGx[509]*acadoWorkspace.x[4] + acadoWorkspace.d[101];
acadoVariables.x[107] += + acadoWorkspace.evGx[510]*acadoWorkspace.x[0] + acadoWorkspace.evGx[511]*acadoWorkspace.x[1] + acadoWorkspace.evGx[512]*acadoWorkspace.x[2] + acadoWorkspace.evGx[513]*acadoWorkspace.x[3] + acadoWorkspace.evGx[514]*acadoWorkspace.x[4] + acadoWorkspace.d[102];
acadoVariables.x[108] += + acadoWorkspace.evGx[515]*acadoWorkspace.x[0] + acadoWorkspace.evGx[516]*acadoWorkspace.x[1] + acadoWorkspace.evGx[517]*acadoWorkspace.x[2] + acadoWorkspace.evGx[518]*acadoWorkspace.x[3] + acadoWorkspace.evGx[519]*acadoWorkspace.x[4] + acadoWorkspace.d[103];
acadoVariables.x[109] += + acadoWorkspace.evGx[520]*acadoWorkspace.x[0] + acadoWorkspace.evGx[521]*acadoWorkspace.x[1] + acadoWorkspace.evGx[522]*acadoWorkspace.x[2] + acadoWorkspace.evGx[523]*acadoWorkspace.x[3] + acadoWorkspace.evGx[524]*acadoWorkspace.x[4] + acadoWorkspace.d[104];
acadoVariables.x[110] += + acadoWorkspace.evGx[525]*acadoWorkspace.x[0] + acadoWorkspace.evGx[526]*acadoWorkspace.x[1] + acadoWorkspace.evGx[527]*acadoWorkspace.x[2] + acadoWorkspace.evGx[528]*acadoWorkspace.x[3] + acadoWorkspace.evGx[529]*acadoWorkspace.x[4] + acadoWorkspace.d[105];
acadoVariables.x[111] += + acadoWorkspace.evGx[530]*acadoWorkspace.x[0] + acadoWorkspace.evGx[531]*acadoWorkspace.x[1] + acadoWorkspace.evGx[532]*acadoWorkspace.x[2] + acadoWorkspace.evGx[533]*acadoWorkspace.x[3] + acadoWorkspace.evGx[534]*acadoWorkspace.x[4] + acadoWorkspace.d[106];
acadoVariables.x[112] += + acadoWorkspace.evGx[535]*acadoWorkspace.x[0] + acadoWorkspace.evGx[536]*acadoWorkspace.x[1] + acadoWorkspace.evGx[537]*acadoWorkspace.x[2] + acadoWorkspace.evGx[538]*acadoWorkspace.x[3] + acadoWorkspace.evGx[539]*acadoWorkspace.x[4] + acadoWorkspace.d[107];
acadoVariables.x[113] += + acadoWorkspace.evGx[540]*acadoWorkspace.x[0] + acadoWorkspace.evGx[541]*acadoWorkspace.x[1] + acadoWorkspace.evGx[542]*acadoWorkspace.x[2] + acadoWorkspace.evGx[543]*acadoWorkspace.x[3] + acadoWorkspace.evGx[544]*acadoWorkspace.x[4] + acadoWorkspace.d[108];
acadoVariables.x[114] += + acadoWorkspace.evGx[545]*acadoWorkspace.x[0] + acadoWorkspace.evGx[546]*acadoWorkspace.x[1] + acadoWorkspace.evGx[547]*acadoWorkspace.x[2] + acadoWorkspace.evGx[548]*acadoWorkspace.x[3] + acadoWorkspace.evGx[549]*acadoWorkspace.x[4] + acadoWorkspace.d[109];
acadoVariables.x[115] += + acadoWorkspace.evGx[550]*acadoWorkspace.x[0] + acadoWorkspace.evGx[551]*acadoWorkspace.x[1] + acadoWorkspace.evGx[552]*acadoWorkspace.x[2] + acadoWorkspace.evGx[553]*acadoWorkspace.x[3] + acadoWorkspace.evGx[554]*acadoWorkspace.x[4] + acadoWorkspace.d[110];
acadoVariables.x[116] += + acadoWorkspace.evGx[555]*acadoWorkspace.x[0] + acadoWorkspace.evGx[556]*acadoWorkspace.x[1] + acadoWorkspace.evGx[557]*acadoWorkspace.x[2] + acadoWorkspace.evGx[558]*acadoWorkspace.x[3] + acadoWorkspace.evGx[559]*acadoWorkspace.x[4] + acadoWorkspace.d[111];
acadoVariables.x[117] += + acadoWorkspace.evGx[560]*acadoWorkspace.x[0] + acadoWorkspace.evGx[561]*acadoWorkspace.x[1] + acadoWorkspace.evGx[562]*acadoWorkspace.x[2] + acadoWorkspace.evGx[563]*acadoWorkspace.x[3] + acadoWorkspace.evGx[564]*acadoWorkspace.x[4] + acadoWorkspace.d[112];
acadoVariables.x[118] += + acadoWorkspace.evGx[565]*acadoWorkspace.x[0] + acadoWorkspace.evGx[566]*acadoWorkspace.x[1] + acadoWorkspace.evGx[567]*acadoWorkspace.x[2] + acadoWorkspace.evGx[568]*acadoWorkspace.x[3] + acadoWorkspace.evGx[569]*acadoWorkspace.x[4] + acadoWorkspace.d[113];
acadoVariables.x[119] += + acadoWorkspace.evGx[570]*acadoWorkspace.x[0] + acadoWorkspace.evGx[571]*acadoWorkspace.x[1] + acadoWorkspace.evGx[572]*acadoWorkspace.x[2] + acadoWorkspace.evGx[573]*acadoWorkspace.x[3] + acadoWorkspace.evGx[574]*acadoWorkspace.x[4] + acadoWorkspace.d[114];
acadoVariables.x[120] += + acadoWorkspace.evGx[575]*acadoWorkspace.x[0] + acadoWorkspace.evGx[576]*acadoWorkspace.x[1] + acadoWorkspace.evGx[577]*acadoWorkspace.x[2] + acadoWorkspace.evGx[578]*acadoWorkspace.x[3] + acadoWorkspace.evGx[579]*acadoWorkspace.x[4] + acadoWorkspace.d[115];
acadoVariables.x[121] += + acadoWorkspace.evGx[580]*acadoWorkspace.x[0] + acadoWorkspace.evGx[581]*acadoWorkspace.x[1] + acadoWorkspace.evGx[582]*acadoWorkspace.x[2] + acadoWorkspace.evGx[583]*acadoWorkspace.x[3] + acadoWorkspace.evGx[584]*acadoWorkspace.x[4] + acadoWorkspace.d[116];
acadoVariables.x[122] += + acadoWorkspace.evGx[585]*acadoWorkspace.x[0] + acadoWorkspace.evGx[586]*acadoWorkspace.x[1] + acadoWorkspace.evGx[587]*acadoWorkspace.x[2] + acadoWorkspace.evGx[588]*acadoWorkspace.x[3] + acadoWorkspace.evGx[589]*acadoWorkspace.x[4] + acadoWorkspace.d[117];
acadoVariables.x[123] += + acadoWorkspace.evGx[590]*acadoWorkspace.x[0] + acadoWorkspace.evGx[591]*acadoWorkspace.x[1] + acadoWorkspace.evGx[592]*acadoWorkspace.x[2] + acadoWorkspace.evGx[593]*acadoWorkspace.x[3] + acadoWorkspace.evGx[594]*acadoWorkspace.x[4] + acadoWorkspace.d[118];
acadoVariables.x[124] += + acadoWorkspace.evGx[595]*acadoWorkspace.x[0] + acadoWorkspace.evGx[596]*acadoWorkspace.x[1] + acadoWorkspace.evGx[597]*acadoWorkspace.x[2] + acadoWorkspace.evGx[598]*acadoWorkspace.x[3] + acadoWorkspace.evGx[599]*acadoWorkspace.x[4] + acadoWorkspace.d[119];
acadoVariables.x[125] += + acadoWorkspace.evGx[600]*acadoWorkspace.x[0] + acadoWorkspace.evGx[601]*acadoWorkspace.x[1] + acadoWorkspace.evGx[602]*acadoWorkspace.x[2] + acadoWorkspace.evGx[603]*acadoWorkspace.x[3] + acadoWorkspace.evGx[604]*acadoWorkspace.x[4] + acadoWorkspace.d[120];
acadoVariables.x[126] += + acadoWorkspace.evGx[605]*acadoWorkspace.x[0] + acadoWorkspace.evGx[606]*acadoWorkspace.x[1] + acadoWorkspace.evGx[607]*acadoWorkspace.x[2] + acadoWorkspace.evGx[608]*acadoWorkspace.x[3] + acadoWorkspace.evGx[609]*acadoWorkspace.x[4] + acadoWorkspace.d[121];
acadoVariables.x[127] += + acadoWorkspace.evGx[610]*acadoWorkspace.x[0] + acadoWorkspace.evGx[611]*acadoWorkspace.x[1] + acadoWorkspace.evGx[612]*acadoWorkspace.x[2] + acadoWorkspace.evGx[613]*acadoWorkspace.x[3] + acadoWorkspace.evGx[614]*acadoWorkspace.x[4] + acadoWorkspace.d[122];
acadoVariables.x[128] += + acadoWorkspace.evGx[615]*acadoWorkspace.x[0] + acadoWorkspace.evGx[616]*acadoWorkspace.x[1] + acadoWorkspace.evGx[617]*acadoWorkspace.x[2] + acadoWorkspace.evGx[618]*acadoWorkspace.x[3] + acadoWorkspace.evGx[619]*acadoWorkspace.x[4] + acadoWorkspace.d[123];
acadoVariables.x[129] += + acadoWorkspace.evGx[620]*acadoWorkspace.x[0] + acadoWorkspace.evGx[621]*acadoWorkspace.x[1] + acadoWorkspace.evGx[622]*acadoWorkspace.x[2] + acadoWorkspace.evGx[623]*acadoWorkspace.x[3] + acadoWorkspace.evGx[624]*acadoWorkspace.x[4] + acadoWorkspace.d[124];
acadoVariables.x[130] += + acadoWorkspace.evGx[625]*acadoWorkspace.x[0] + acadoWorkspace.evGx[626]*acadoWorkspace.x[1] + acadoWorkspace.evGx[627]*acadoWorkspace.x[2] + acadoWorkspace.evGx[628]*acadoWorkspace.x[3] + acadoWorkspace.evGx[629]*acadoWorkspace.x[4] + acadoWorkspace.d[125];
acadoVariables.x[131] += + acadoWorkspace.evGx[630]*acadoWorkspace.x[0] + acadoWorkspace.evGx[631]*acadoWorkspace.x[1] + acadoWorkspace.evGx[632]*acadoWorkspace.x[2] + acadoWorkspace.evGx[633]*acadoWorkspace.x[3] + acadoWorkspace.evGx[634]*acadoWorkspace.x[4] + acadoWorkspace.d[126];
acadoVariables.x[132] += + acadoWorkspace.evGx[635]*acadoWorkspace.x[0] + acadoWorkspace.evGx[636]*acadoWorkspace.x[1] + acadoWorkspace.evGx[637]*acadoWorkspace.x[2] + acadoWorkspace.evGx[638]*acadoWorkspace.x[3] + acadoWorkspace.evGx[639]*acadoWorkspace.x[4] + acadoWorkspace.d[127];
acadoVariables.x[133] += + acadoWorkspace.evGx[640]*acadoWorkspace.x[0] + acadoWorkspace.evGx[641]*acadoWorkspace.x[1] + acadoWorkspace.evGx[642]*acadoWorkspace.x[2] + acadoWorkspace.evGx[643]*acadoWorkspace.x[3] + acadoWorkspace.evGx[644]*acadoWorkspace.x[4] + acadoWorkspace.d[128];
acadoVariables.x[134] += + acadoWorkspace.evGx[645]*acadoWorkspace.x[0] + acadoWorkspace.evGx[646]*acadoWorkspace.x[1] + acadoWorkspace.evGx[647]*acadoWorkspace.x[2] + acadoWorkspace.evGx[648]*acadoWorkspace.x[3] + acadoWorkspace.evGx[649]*acadoWorkspace.x[4] + acadoWorkspace.d[129];
acadoVariables.x[135] += + acadoWorkspace.evGx[650]*acadoWorkspace.x[0] + acadoWorkspace.evGx[651]*acadoWorkspace.x[1] + acadoWorkspace.evGx[652]*acadoWorkspace.x[2] + acadoWorkspace.evGx[653]*acadoWorkspace.x[3] + acadoWorkspace.evGx[654]*acadoWorkspace.x[4] + acadoWorkspace.d[130];
acadoVariables.x[136] += + acadoWorkspace.evGx[655]*acadoWorkspace.x[0] + acadoWorkspace.evGx[656]*acadoWorkspace.x[1] + acadoWorkspace.evGx[657]*acadoWorkspace.x[2] + acadoWorkspace.evGx[658]*acadoWorkspace.x[3] + acadoWorkspace.evGx[659]*acadoWorkspace.x[4] + acadoWorkspace.d[131];
acadoVariables.x[137] += + acadoWorkspace.evGx[660]*acadoWorkspace.x[0] + acadoWorkspace.evGx[661]*acadoWorkspace.x[1] + acadoWorkspace.evGx[662]*acadoWorkspace.x[2] + acadoWorkspace.evGx[663]*acadoWorkspace.x[3] + acadoWorkspace.evGx[664]*acadoWorkspace.x[4] + acadoWorkspace.d[132];
acadoVariables.x[138] += + acadoWorkspace.evGx[665]*acadoWorkspace.x[0] + acadoWorkspace.evGx[666]*acadoWorkspace.x[1] + acadoWorkspace.evGx[667]*acadoWorkspace.x[2] + acadoWorkspace.evGx[668]*acadoWorkspace.x[3] + acadoWorkspace.evGx[669]*acadoWorkspace.x[4] + acadoWorkspace.d[133];
acadoVariables.x[139] += + acadoWorkspace.evGx[670]*acadoWorkspace.x[0] + acadoWorkspace.evGx[671]*acadoWorkspace.x[1] + acadoWorkspace.evGx[672]*acadoWorkspace.x[2] + acadoWorkspace.evGx[673]*acadoWorkspace.x[3] + acadoWorkspace.evGx[674]*acadoWorkspace.x[4] + acadoWorkspace.d[134];
acadoVariables.x[140] += + acadoWorkspace.evGx[675]*acadoWorkspace.x[0] + acadoWorkspace.evGx[676]*acadoWorkspace.x[1] + acadoWorkspace.evGx[677]*acadoWorkspace.x[2] + acadoWorkspace.evGx[678]*acadoWorkspace.x[3] + acadoWorkspace.evGx[679]*acadoWorkspace.x[4] + acadoWorkspace.d[135];
acadoVariables.x[141] += + acadoWorkspace.evGx[680]*acadoWorkspace.x[0] + acadoWorkspace.evGx[681]*acadoWorkspace.x[1] + acadoWorkspace.evGx[682]*acadoWorkspace.x[2] + acadoWorkspace.evGx[683]*acadoWorkspace.x[3] + acadoWorkspace.evGx[684]*acadoWorkspace.x[4] + acadoWorkspace.d[136];
acadoVariables.x[142] += + acadoWorkspace.evGx[685]*acadoWorkspace.x[0] + acadoWorkspace.evGx[686]*acadoWorkspace.x[1] + acadoWorkspace.evGx[687]*acadoWorkspace.x[2] + acadoWorkspace.evGx[688]*acadoWorkspace.x[3] + acadoWorkspace.evGx[689]*acadoWorkspace.x[4] + acadoWorkspace.d[137];
acadoVariables.x[143] += + acadoWorkspace.evGx[690]*acadoWorkspace.x[0] + acadoWorkspace.evGx[691]*acadoWorkspace.x[1] + acadoWorkspace.evGx[692]*acadoWorkspace.x[2] + acadoWorkspace.evGx[693]*acadoWorkspace.x[3] + acadoWorkspace.evGx[694]*acadoWorkspace.x[4] + acadoWorkspace.d[138];
acadoVariables.x[144] += + acadoWorkspace.evGx[695]*acadoWorkspace.x[0] + acadoWorkspace.evGx[696]*acadoWorkspace.x[1] + acadoWorkspace.evGx[697]*acadoWorkspace.x[2] + acadoWorkspace.evGx[698]*acadoWorkspace.x[3] + acadoWorkspace.evGx[699]*acadoWorkspace.x[4] + acadoWorkspace.d[139];
acadoVariables.x[145] += + acadoWorkspace.evGx[700]*acadoWorkspace.x[0] + acadoWorkspace.evGx[701]*acadoWorkspace.x[1] + acadoWorkspace.evGx[702]*acadoWorkspace.x[2] + acadoWorkspace.evGx[703]*acadoWorkspace.x[3] + acadoWorkspace.evGx[704]*acadoWorkspace.x[4] + acadoWorkspace.d[140];
acadoVariables.x[146] += + acadoWorkspace.evGx[705]*acadoWorkspace.x[0] + acadoWorkspace.evGx[706]*acadoWorkspace.x[1] + acadoWorkspace.evGx[707]*acadoWorkspace.x[2] + acadoWorkspace.evGx[708]*acadoWorkspace.x[3] + acadoWorkspace.evGx[709]*acadoWorkspace.x[4] + acadoWorkspace.d[141];
acadoVariables.x[147] += + acadoWorkspace.evGx[710]*acadoWorkspace.x[0] + acadoWorkspace.evGx[711]*acadoWorkspace.x[1] + acadoWorkspace.evGx[712]*acadoWorkspace.x[2] + acadoWorkspace.evGx[713]*acadoWorkspace.x[3] + acadoWorkspace.evGx[714]*acadoWorkspace.x[4] + acadoWorkspace.d[142];
acadoVariables.x[148] += + acadoWorkspace.evGx[715]*acadoWorkspace.x[0] + acadoWorkspace.evGx[716]*acadoWorkspace.x[1] + acadoWorkspace.evGx[717]*acadoWorkspace.x[2] + acadoWorkspace.evGx[718]*acadoWorkspace.x[3] + acadoWorkspace.evGx[719]*acadoWorkspace.x[4] + acadoWorkspace.d[143];
acadoVariables.x[149] += + acadoWorkspace.evGx[720]*acadoWorkspace.x[0] + acadoWorkspace.evGx[721]*acadoWorkspace.x[1] + acadoWorkspace.evGx[722]*acadoWorkspace.x[2] + acadoWorkspace.evGx[723]*acadoWorkspace.x[3] + acadoWorkspace.evGx[724]*acadoWorkspace.x[4] + acadoWorkspace.d[144];
acadoVariables.x[150] += + acadoWorkspace.evGx[725]*acadoWorkspace.x[0] + acadoWorkspace.evGx[726]*acadoWorkspace.x[1] + acadoWorkspace.evGx[727]*acadoWorkspace.x[2] + acadoWorkspace.evGx[728]*acadoWorkspace.x[3] + acadoWorkspace.evGx[729]*acadoWorkspace.x[4] + acadoWorkspace.d[145];
acadoVariables.x[151] += + acadoWorkspace.evGx[730]*acadoWorkspace.x[0] + acadoWorkspace.evGx[731]*acadoWorkspace.x[1] + acadoWorkspace.evGx[732]*acadoWorkspace.x[2] + acadoWorkspace.evGx[733]*acadoWorkspace.x[3] + acadoWorkspace.evGx[734]*acadoWorkspace.x[4] + acadoWorkspace.d[146];
acadoVariables.x[152] += + acadoWorkspace.evGx[735]*acadoWorkspace.x[0] + acadoWorkspace.evGx[736]*acadoWorkspace.x[1] + acadoWorkspace.evGx[737]*acadoWorkspace.x[2] + acadoWorkspace.evGx[738]*acadoWorkspace.x[3] + acadoWorkspace.evGx[739]*acadoWorkspace.x[4] + acadoWorkspace.d[147];
acadoVariables.x[153] += + acadoWorkspace.evGx[740]*acadoWorkspace.x[0] + acadoWorkspace.evGx[741]*acadoWorkspace.x[1] + acadoWorkspace.evGx[742]*acadoWorkspace.x[2] + acadoWorkspace.evGx[743]*acadoWorkspace.x[3] + acadoWorkspace.evGx[744]*acadoWorkspace.x[4] + acadoWorkspace.d[148];
acadoVariables.x[154] += + acadoWorkspace.evGx[745]*acadoWorkspace.x[0] + acadoWorkspace.evGx[746]*acadoWorkspace.x[1] + acadoWorkspace.evGx[747]*acadoWorkspace.x[2] + acadoWorkspace.evGx[748]*acadoWorkspace.x[3] + acadoWorkspace.evGx[749]*acadoWorkspace.x[4] + acadoWorkspace.d[149];
acadoVariables.x[155] += + acadoWorkspace.evGx[750]*acadoWorkspace.x[0] + acadoWorkspace.evGx[751]*acadoWorkspace.x[1] + acadoWorkspace.evGx[752]*acadoWorkspace.x[2] + acadoWorkspace.evGx[753]*acadoWorkspace.x[3] + acadoWorkspace.evGx[754]*acadoWorkspace.x[4] + acadoWorkspace.d[150];
acadoVariables.x[156] += + acadoWorkspace.evGx[755]*acadoWorkspace.x[0] + acadoWorkspace.evGx[756]*acadoWorkspace.x[1] + acadoWorkspace.evGx[757]*acadoWorkspace.x[2] + acadoWorkspace.evGx[758]*acadoWorkspace.x[3] + acadoWorkspace.evGx[759]*acadoWorkspace.x[4] + acadoWorkspace.d[151];
acadoVariables.x[157] += + acadoWorkspace.evGx[760]*acadoWorkspace.x[0] + acadoWorkspace.evGx[761]*acadoWorkspace.x[1] + acadoWorkspace.evGx[762]*acadoWorkspace.x[2] + acadoWorkspace.evGx[763]*acadoWorkspace.x[3] + acadoWorkspace.evGx[764]*acadoWorkspace.x[4] + acadoWorkspace.d[152];
acadoVariables.x[158] += + acadoWorkspace.evGx[765]*acadoWorkspace.x[0] + acadoWorkspace.evGx[766]*acadoWorkspace.x[1] + acadoWorkspace.evGx[767]*acadoWorkspace.x[2] + acadoWorkspace.evGx[768]*acadoWorkspace.x[3] + acadoWorkspace.evGx[769]*acadoWorkspace.x[4] + acadoWorkspace.d[153];
acadoVariables.x[159] += + acadoWorkspace.evGx[770]*acadoWorkspace.x[0] + acadoWorkspace.evGx[771]*acadoWorkspace.x[1] + acadoWorkspace.evGx[772]*acadoWorkspace.x[2] + acadoWorkspace.evGx[773]*acadoWorkspace.x[3] + acadoWorkspace.evGx[774]*acadoWorkspace.x[4] + acadoWorkspace.d[154];
acadoVariables.x[160] += + acadoWorkspace.evGx[775]*acadoWorkspace.x[0] + acadoWorkspace.evGx[776]*acadoWorkspace.x[1] + acadoWorkspace.evGx[777]*acadoWorkspace.x[2] + acadoWorkspace.evGx[778]*acadoWorkspace.x[3] + acadoWorkspace.evGx[779]*acadoWorkspace.x[4] + acadoWorkspace.d[155];
acadoVariables.x[161] += + acadoWorkspace.evGx[780]*acadoWorkspace.x[0] + acadoWorkspace.evGx[781]*acadoWorkspace.x[1] + acadoWorkspace.evGx[782]*acadoWorkspace.x[2] + acadoWorkspace.evGx[783]*acadoWorkspace.x[3] + acadoWorkspace.evGx[784]*acadoWorkspace.x[4] + acadoWorkspace.d[156];
acadoVariables.x[162] += + acadoWorkspace.evGx[785]*acadoWorkspace.x[0] + acadoWorkspace.evGx[786]*acadoWorkspace.x[1] + acadoWorkspace.evGx[787]*acadoWorkspace.x[2] + acadoWorkspace.evGx[788]*acadoWorkspace.x[3] + acadoWorkspace.evGx[789]*acadoWorkspace.x[4] + acadoWorkspace.d[157];
acadoVariables.x[163] += + acadoWorkspace.evGx[790]*acadoWorkspace.x[0] + acadoWorkspace.evGx[791]*acadoWorkspace.x[1] + acadoWorkspace.evGx[792]*acadoWorkspace.x[2] + acadoWorkspace.evGx[793]*acadoWorkspace.x[3] + acadoWorkspace.evGx[794]*acadoWorkspace.x[4] + acadoWorkspace.d[158];
acadoVariables.x[164] += + acadoWorkspace.evGx[795]*acadoWorkspace.x[0] + acadoWorkspace.evGx[796]*acadoWorkspace.x[1] + acadoWorkspace.evGx[797]*acadoWorkspace.x[2] + acadoWorkspace.evGx[798]*acadoWorkspace.x[3] + acadoWorkspace.evGx[799]*acadoWorkspace.x[4] + acadoWorkspace.d[159];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multEDu( &(acadoWorkspace.E[ lRun3 * 15 ]), &(acadoWorkspace.x[ lRun2 * 3 + 5 ]), &(acadoVariables.x[ lRun1 * 5 + 5 ]) );
}
}
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
for (index = 0; index < 32; ++index)
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
for (index = 0; index < 32; ++index)
{
acadoVariables.x[index * 5] = acadoVariables.x[index * 5 + 5];
acadoVariables.x[index * 5 + 1] = acadoVariables.x[index * 5 + 6];
acadoVariables.x[index * 5 + 2] = acadoVariables.x[index * 5 + 7];
acadoVariables.x[index * 5 + 3] = acadoVariables.x[index * 5 + 8];
acadoVariables.x[index * 5 + 4] = acadoVariables.x[index * 5 + 9];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[160] = xEnd[0];
acadoVariables.x[161] = xEnd[1];
acadoVariables.x[162] = xEnd[2];
acadoVariables.x[163] = xEnd[3];
acadoVariables.x[164] = xEnd[4];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[160];
acadoWorkspace.state[1] = acadoVariables.x[161];
acadoWorkspace.state[2] = acadoVariables.x[162];
acadoWorkspace.state[3] = acadoVariables.x[163];
acadoWorkspace.state[4] = acadoVariables.x[164];
if (uEnd != 0)
{
acadoWorkspace.state[45] = uEnd[0];
acadoWorkspace.state[46] = uEnd[1];
acadoWorkspace.state[47] = uEnd[2];
}
else
{
acadoWorkspace.state[45] = acadoVariables.u[93];
acadoWorkspace.state[46] = acadoVariables.u[94];
acadoWorkspace.state[47] = acadoVariables.u[95];
}
acadoWorkspace.state[48] = acadoVariables.od[64];
acadoWorkspace.state[49] = acadoVariables.od[65];

acado_integrate(acadoWorkspace.state, 1, 31);

acadoVariables.x[160] = acadoWorkspace.state[0];
acadoVariables.x[161] = acadoWorkspace.state[1];
acadoVariables.x[162] = acadoWorkspace.state[2];
acadoVariables.x[163] = acadoWorkspace.state[3];
acadoVariables.x[164] = acadoWorkspace.state[4];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 31; ++index)
{
acadoVariables.u[index * 3] = acadoVariables.u[index * 3 + 3];
acadoVariables.u[index * 3 + 1] = acadoVariables.u[index * 3 + 4];
acadoVariables.u[index * 3 + 2] = acadoVariables.u[index * 3 + 5];
}

if (uEnd != 0)
{
acadoVariables.u[93] = uEnd[0];
acadoVariables.u[94] = uEnd[1];
acadoVariables.u[95] = uEnd[2];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43] + acadoWorkspace.g[44]*acadoWorkspace.x[44] + acadoWorkspace.g[45]*acadoWorkspace.x[45] + acadoWorkspace.g[46]*acadoWorkspace.x[46] + acadoWorkspace.g[47]*acadoWorkspace.x[47] + acadoWorkspace.g[48]*acadoWorkspace.x[48] + acadoWorkspace.g[49]*acadoWorkspace.x[49] + acadoWorkspace.g[50]*acadoWorkspace.x[50] + acadoWorkspace.g[51]*acadoWorkspace.x[51] + acadoWorkspace.g[52]*acadoWorkspace.x[52] + acadoWorkspace.g[53]*acadoWorkspace.x[53] + acadoWorkspace.g[54]*acadoWorkspace.x[54] + acadoWorkspace.g[55]*acadoWorkspace.x[55] + acadoWorkspace.g[56]*acadoWorkspace.x[56] + acadoWorkspace.g[57]*acadoWorkspace.x[57] + acadoWorkspace.g[58]*acadoWorkspace.x[58] + acadoWorkspace.g[59]*acadoWorkspace.x[59] + acadoWorkspace.g[60]*acadoWorkspace.x[60] + acadoWorkspace.g[61]*acadoWorkspace.x[61] + acadoWorkspace.g[62]*acadoWorkspace.x[62] + acadoWorkspace.g[63]*acadoWorkspace.x[63] + acadoWorkspace.g[64]*acadoWorkspace.x[64] + acadoWorkspace.g[65]*acadoWorkspace.x[65] + acadoWorkspace.g[66]*acadoWorkspace.x[66] + acadoWorkspace.g[67]*acadoWorkspace.x[67] + acadoWorkspace.g[68]*acadoWorkspace.x[68] + acadoWorkspace.g[69]*acadoWorkspace.x[69] + acadoWorkspace.g[70]*acadoWorkspace.x[70] + acadoWorkspace.g[71]*acadoWorkspace.x[71] + acadoWorkspace.g[72]*acadoWorkspace.x[72] + acadoWorkspace.g[73]*acadoWorkspace.x[73] + acadoWorkspace.g[74]*acadoWorkspace.x[74] + acadoWorkspace.g[75]*acadoWorkspace.x[75] + acadoWorkspace.g[76]*acadoWorkspace.x[76] + acadoWorkspace.g[77]*acadoWorkspace.x[77] + acadoWorkspace.g[78]*acadoWorkspace.x[78] + acadoWorkspace.g[79]*acadoWorkspace.x[79] + acadoWorkspace.g[80]*acadoWorkspace.x[80] + acadoWorkspace.g[81]*acadoWorkspace.x[81] + acadoWorkspace.g[82]*acadoWorkspace.x[82] + acadoWorkspace.g[83]*acadoWorkspace.x[83] + acadoWorkspace.g[84]*acadoWorkspace.x[84] + acadoWorkspace.g[85]*acadoWorkspace.x[85] + acadoWorkspace.g[86]*acadoWorkspace.x[86] + acadoWorkspace.g[87]*acadoWorkspace.x[87] + acadoWorkspace.g[88]*acadoWorkspace.x[88] + acadoWorkspace.g[89]*acadoWorkspace.x[89] + acadoWorkspace.g[90]*acadoWorkspace.x[90] + acadoWorkspace.g[91]*acadoWorkspace.x[91] + acadoWorkspace.g[92]*acadoWorkspace.x[92] + acadoWorkspace.g[93]*acadoWorkspace.x[93] + acadoWorkspace.g[94]*acadoWorkspace.x[94] + acadoWorkspace.g[95]*acadoWorkspace.x[95] + acadoWorkspace.g[96]*acadoWorkspace.x[96] + acadoWorkspace.g[97]*acadoWorkspace.x[97] + acadoWorkspace.g[98]*acadoWorkspace.x[98] + acadoWorkspace.g[99]*acadoWorkspace.x[99] + acadoWorkspace.g[100]*acadoWorkspace.x[100];
kkt = fabs( kkt );
for (index = 0; index < 101; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 96; ++index)
{
prd = acadoWorkspace.y[index + 101];
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
/** Row vector of size: 6 */
real_t tmpDy[ 6 ];

/** Row vector of size: 3 */
real_t tmpDyN[ 3 ];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
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
acadoWorkspace.Dy[lRun1 * 6] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 6];
acadoWorkspace.Dy[lRun1 * 6 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 6 + 1];
acadoWorkspace.Dy[lRun1 * 6 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 6 + 2];
acadoWorkspace.Dy[lRun1 * 6 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 6 + 3];
acadoWorkspace.Dy[lRun1 * 6 + 4] = acadoWorkspace.objValueOut[4] - acadoVariables.y[lRun1 * 6 + 4];
acadoWorkspace.Dy[lRun1 * 6 + 5] = acadoWorkspace.objValueOut[5] - acadoVariables.y[lRun1 * 6 + 5];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[160];
acadoWorkspace.objValueIn[1] = acadoVariables.x[161];
acadoWorkspace.objValueIn[2] = acadoVariables.x[162];
acadoWorkspace.objValueIn[3] = acadoVariables.x[163];
acadoWorkspace.objValueIn[4] = acadoVariables.x[164];
acadoWorkspace.objValueIn[5] = acadoVariables.od[64];
acadoWorkspace.objValueIn[6] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 6] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 12] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 18] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 24] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 30];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36 + 1] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 7] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 13] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 19] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 25] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 31];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36 + 2] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 8] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 14] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 20] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 26] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 32];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36 + 3] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 9] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 15] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 21] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 27] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 33];
tmpDy[4] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36 + 4] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 10] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 16] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 22] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 28] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 34];
tmpDy[5] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[lRun1 * 36 + 5] + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[lRun1 * 36 + 11] + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[lRun1 * 36 + 17] + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[lRun1 * 36 + 23] + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[lRun1 * 36 + 29] + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[lRun1 * 36 + 35];
objVal += + acadoWorkspace.Dy[lRun1 * 6]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 6 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 6 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 6 + 3]*tmpDy[3] + acadoWorkspace.Dy[lRun1 * 6 + 4]*tmpDy[4] + acadoWorkspace.Dy[lRun1 * 6 + 5]*tmpDy[5];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0] + acadoWorkspace.DyN[1]*acadoVariables.WN[3] + acadoWorkspace.DyN[2]*acadoVariables.WN[6];
tmpDyN[1] = + acadoWorkspace.DyN[0]*acadoVariables.WN[1] + acadoWorkspace.DyN[1]*acadoVariables.WN[4] + acadoWorkspace.DyN[2]*acadoVariables.WN[7];
tmpDyN[2] = + acadoWorkspace.DyN[0]*acadoVariables.WN[2] + acadoWorkspace.DyN[1]*acadoVariables.WN[5] + acadoWorkspace.DyN[2]*acadoVariables.WN[8];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2];

objVal *= 0.5;
return objVal;
}

