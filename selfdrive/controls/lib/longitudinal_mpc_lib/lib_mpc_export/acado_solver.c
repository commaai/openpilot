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
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 4 + 3];

acadoWorkspace.state[28] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.state[29] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.state[30] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.state[31] = acadoVariables.od[lRun1 * 2 + 1];

ret = acado_integrate(acadoWorkspace.state, 1, lRun1);

acadoWorkspace.d[lRun1 * 4] = acadoWorkspace.state[0] - acadoVariables.x[lRun1 * 4 + 4];
acadoWorkspace.d[lRun1 * 4 + 1] = acadoWorkspace.state[1] - acadoVariables.x[lRun1 * 4 + 5];
acadoWorkspace.d[lRun1 * 4 + 2] = acadoWorkspace.state[2] - acadoVariables.x[lRun1 * 4 + 6];
acadoWorkspace.d[lRun1 * 4 + 3] = acadoWorkspace.state[3] - acadoVariables.x[lRun1 * 4 + 7];

acadoWorkspace.evGx[lRun1 * 16] = acadoWorkspace.state[4];
acadoWorkspace.evGx[lRun1 * 16 + 1] = acadoWorkspace.state[5];
acadoWorkspace.evGx[lRun1 * 16 + 2] = acadoWorkspace.state[6];
acadoWorkspace.evGx[lRun1 * 16 + 3] = acadoWorkspace.state[7];
acadoWorkspace.evGx[lRun1 * 16 + 4] = acadoWorkspace.state[8];
acadoWorkspace.evGx[lRun1 * 16 + 5] = acadoWorkspace.state[9];
acadoWorkspace.evGx[lRun1 * 16 + 6] = acadoWorkspace.state[10];
acadoWorkspace.evGx[lRun1 * 16 + 7] = acadoWorkspace.state[11];
acadoWorkspace.evGx[lRun1 * 16 + 8] = acadoWorkspace.state[12];
acadoWorkspace.evGx[lRun1 * 16 + 9] = acadoWorkspace.state[13];
acadoWorkspace.evGx[lRun1 * 16 + 10] = acadoWorkspace.state[14];
acadoWorkspace.evGx[lRun1 * 16 + 11] = acadoWorkspace.state[15];
acadoWorkspace.evGx[lRun1 * 16 + 12] = acadoWorkspace.state[16];
acadoWorkspace.evGx[lRun1 * 16 + 13] = acadoWorkspace.state[17];
acadoWorkspace.evGx[lRun1 * 16 + 14] = acadoWorkspace.state[18];
acadoWorkspace.evGx[lRun1 * 16 + 15] = acadoWorkspace.state[19];

acadoWorkspace.evGu[lRun1 * 8] = acadoWorkspace.state[20];
acadoWorkspace.evGu[lRun1 * 8 + 1] = acadoWorkspace.state[21];
acadoWorkspace.evGu[lRun1 * 8 + 2] = acadoWorkspace.state[22];
acadoWorkspace.evGu[lRun1 * 8 + 3] = acadoWorkspace.state[23];
acadoWorkspace.evGu[lRun1 * 8 + 4] = acadoWorkspace.state[24];
acadoWorkspace.evGu[lRun1 * 8 + 5] = acadoWorkspace.state[25];
acadoWorkspace.evGu[lRun1 * 8 + 6] = acadoWorkspace.state[26];
acadoWorkspace.evGu[lRun1 * 8 + 7] = acadoWorkspace.state[27];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 4;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = u[0];
out[4] = u[1];
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
tmpQ2[15] = 0.0;
;
tmpQ2[16] = 0.0;
;
tmpQ2[17] = 0.0;
;
tmpQ2[18] = 0.0;
;
tmpQ2[19] = 0.0;
;
tmpQ1[0] = + tmpQ2[0];
tmpQ1[1] = + tmpQ2[1];
tmpQ1[2] = + tmpQ2[2];
tmpQ1[3] = 0.0;
;
tmpQ1[4] = + tmpQ2[5];
tmpQ1[5] = + tmpQ2[6];
tmpQ1[6] = + tmpQ2[7];
tmpQ1[7] = 0.0;
;
tmpQ1[8] = + tmpQ2[10];
tmpQ1[9] = + tmpQ2[11];
tmpQ1[10] = + tmpQ2[12];
tmpQ1[11] = 0.0;
;
tmpQ1[12] = + tmpQ2[15];
tmpQ1[13] = + tmpQ2[16];
tmpQ1[14] = + tmpQ2[17];
tmpQ1[15] = 0.0;
;
}

void acado_setObjR1R2( real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = +tmpObjS[15];
tmpR2[1] = +tmpObjS[16];
tmpR2[2] = +tmpObjS[17];
tmpR2[3] = +tmpObjS[18];
tmpR2[4] = +tmpObjS[19];
tmpR2[5] = +tmpObjS[20];
tmpR2[6] = +tmpObjS[21];
tmpR2[7] = +tmpObjS[22];
tmpR2[8] = +tmpObjS[23];
tmpR2[9] = +tmpObjS[24];
tmpR1[0] = + tmpR2[3];
tmpR1[1] = + tmpR2[4];
tmpR1[2] = + tmpR2[8];
tmpR1[3] = + tmpR2[9];
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
tmpQN1[0] = + tmpQN2[0];
tmpQN1[1] = + tmpQN2[1];
tmpQN1[2] = + tmpQN2[2];
tmpQN1[3] = 0.0;
;
tmpQN1[4] = + tmpQN2[3];
tmpQN1[5] = + tmpQN2[4];
tmpQN1[6] = + tmpQN2[5];
tmpQN1[7] = 0.0;
;
tmpQN1[8] = + tmpQN2[6];
tmpQN1[9] = + tmpQN2[7];
tmpQN1[10] = + tmpQN2[8];
tmpQN1[11] = 0.0;
;
tmpQN1[12] = + tmpQN2[9];
tmpQN1[13] = + tmpQN2[10];
tmpQN1[14] = + tmpQN2[11];
tmpQN1[15] = 0.0;
;
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 32; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[runObj * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[runObj * 2 + 1];
acadoWorkspace.objValueIn[6] = acadoVariables.od[runObj * 2];
acadoWorkspace.objValueIn[7] = acadoVariables.od[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 5] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 5 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 5 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 5 + 3] = acadoWorkspace.objValueOut[3];
acadoWorkspace.Dy[runObj * 5 + 4] = acadoWorkspace.objValueOut[4];

acado_setObjQ1Q2( &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 20 ]) );

acado_setObjR1R2( &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.R1[ runObj * 4 ]), &(acadoWorkspace.R2[ runObj * 10 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[128];
acadoWorkspace.objValueIn[1] = acadoVariables.x[129];
acadoWorkspace.objValueIn[2] = acadoVariables.x[130];
acadoWorkspace.objValueIn[3] = acadoVariables.x[131];
acadoWorkspace.objValueIn[4] = acadoVariables.od[64];
acadoWorkspace.objValueIn[5] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_multGxd( real_t* const dOld, real_t* const Gx1, real_t* const dNew )
{
dNew[0] += + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3];
dNew[1] += + Gx1[4]*dOld[0] + Gx1[5]*dOld[1] + Gx1[6]*dOld[2] + Gx1[7]*dOld[3];
dNew[2] += + Gx1[8]*dOld[0] + Gx1[9]*dOld[1] + Gx1[10]*dOld[2] + Gx1[11]*dOld[3];
dNew[3] += + Gx1[12]*dOld[0] + Gx1[13]*dOld[1] + Gx1[14]*dOld[2] + Gx1[15]*dOld[3];
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
}

void acado_multGxGx( real_t* const Gx1, real_t* const Gx2, real_t* const Gx3 )
{
Gx3[0] = + Gx1[0]*Gx2[0] + Gx1[1]*Gx2[4] + Gx1[2]*Gx2[8] + Gx1[3]*Gx2[12];
Gx3[1] = + Gx1[0]*Gx2[1] + Gx1[1]*Gx2[5] + Gx1[2]*Gx2[9] + Gx1[3]*Gx2[13];
Gx3[2] = + Gx1[0]*Gx2[2] + Gx1[1]*Gx2[6] + Gx1[2]*Gx2[10] + Gx1[3]*Gx2[14];
Gx3[3] = + Gx1[0]*Gx2[3] + Gx1[1]*Gx2[7] + Gx1[2]*Gx2[11] + Gx1[3]*Gx2[15];
Gx3[4] = + Gx1[4]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[6]*Gx2[8] + Gx1[7]*Gx2[12];
Gx3[5] = + Gx1[4]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[6]*Gx2[9] + Gx1[7]*Gx2[13];
Gx3[6] = + Gx1[4]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[6]*Gx2[10] + Gx1[7]*Gx2[14];
Gx3[7] = + Gx1[4]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[6]*Gx2[11] + Gx1[7]*Gx2[15];
Gx3[8] = + Gx1[8]*Gx2[0] + Gx1[9]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[11]*Gx2[12];
Gx3[9] = + Gx1[8]*Gx2[1] + Gx1[9]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[11]*Gx2[13];
Gx3[10] = + Gx1[8]*Gx2[2] + Gx1[9]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[11]*Gx2[14];
Gx3[11] = + Gx1[8]*Gx2[3] + Gx1[9]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[11]*Gx2[15];
Gx3[12] = + Gx1[12]*Gx2[0] + Gx1[13]*Gx2[4] + Gx1[14]*Gx2[8] + Gx1[15]*Gx2[12];
Gx3[13] = + Gx1[12]*Gx2[1] + Gx1[13]*Gx2[5] + Gx1[14]*Gx2[9] + Gx1[15]*Gx2[13];
Gx3[14] = + Gx1[12]*Gx2[2] + Gx1[13]*Gx2[6] + Gx1[14]*Gx2[10] + Gx1[15]*Gx2[14];
Gx3[15] = + Gx1[12]*Gx2[3] + Gx1[13]*Gx2[7] + Gx1[14]*Gx2[11] + Gx1[15]*Gx2[15];
}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[2] + Gx1[2]*Gu1[4] + Gx1[3]*Gu1[6];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[1]*Gu1[3] + Gx1[2]*Gu1[5] + Gx1[3]*Gu1[7];
Gu2[2] = + Gx1[4]*Gu1[0] + Gx1[5]*Gu1[2] + Gx1[6]*Gu1[4] + Gx1[7]*Gu1[6];
Gu2[3] = + Gx1[4]*Gu1[1] + Gx1[5]*Gu1[3] + Gx1[6]*Gu1[5] + Gx1[7]*Gu1[7];
Gu2[4] = + Gx1[8]*Gu1[0] + Gx1[9]*Gu1[2] + Gx1[10]*Gu1[4] + Gx1[11]*Gu1[6];
Gu2[5] = + Gx1[8]*Gu1[1] + Gx1[9]*Gu1[3] + Gx1[10]*Gu1[5] + Gx1[11]*Gu1[7];
Gu2[6] = + Gx1[12]*Gu1[0] + Gx1[13]*Gu1[2] + Gx1[14]*Gu1[4] + Gx1[15]*Gu1[6];
Gu2[7] = + Gx1[12]*Gu1[1] + Gx1[13]*Gu1[3] + Gx1[14]*Gu1[5] + Gx1[15]*Gu1[7];
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
}

void acado_setBlockH11( int iRow, int iCol, real_t* const Gu1, real_t* const Gu2 )
{
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 4)] += + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6];
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 5)] += + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 4)] += + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 5)] += + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 4)] = R11[0];
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 5)] = R11[1];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 4)] = R11[2];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 5)] = R11[3];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 4)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 4)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 5)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 4)] = acadoWorkspace.H[(iCol * 136 + 272) + (iRow * 2 + 4)];
acadoWorkspace.H[(iRow * 136 + 272) + (iCol * 2 + 5)] = acadoWorkspace.H[(iCol * 136 + 340) + (iRow * 2 + 4)];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 4)] = acadoWorkspace.H[(iCol * 136 + 272) + (iRow * 2 + 5)];
acadoWorkspace.H[(iRow * 136 + 340) + (iCol * 2 + 5)] = acadoWorkspace.H[(iCol * 136 + 340) + (iRow * 2 + 5)];
}

void acado_multQ1d( real_t* const Gx1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3];
dNew[1] = + Gx1[4]*dOld[0] + Gx1[5]*dOld[1] + Gx1[6]*dOld[2] + Gx1[7]*dOld[3];
dNew[2] = + Gx1[8]*dOld[0] + Gx1[9]*dOld[1] + Gx1[10]*dOld[2] + Gx1[11]*dOld[3];
dNew[3] = + Gx1[12]*dOld[0] + Gx1[13]*dOld[1] + Gx1[14]*dOld[2] + Gx1[15]*dOld[3];
}

void acado_multQN1d( real_t* const QN1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + acadoWorkspace.QN1[0]*dOld[0] + acadoWorkspace.QN1[1]*dOld[1] + acadoWorkspace.QN1[2]*dOld[2] + acadoWorkspace.QN1[3]*dOld[3];
dNew[1] = + acadoWorkspace.QN1[4]*dOld[0] + acadoWorkspace.QN1[5]*dOld[1] + acadoWorkspace.QN1[6]*dOld[2] + acadoWorkspace.QN1[7]*dOld[3];
dNew[2] = + acadoWorkspace.QN1[8]*dOld[0] + acadoWorkspace.QN1[9]*dOld[1] + acadoWorkspace.QN1[10]*dOld[2] + acadoWorkspace.QN1[11]*dOld[3];
dNew[3] = + acadoWorkspace.QN1[12]*dOld[0] + acadoWorkspace.QN1[13]*dOld[1] + acadoWorkspace.QN1[14]*dOld[2] + acadoWorkspace.QN1[15]*dOld[3];
}

void acado_multRDy( real_t* const R2, real_t* const Dy1, real_t* const RDy1 )
{
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2] + R2[3]*Dy1[3] + R2[4]*Dy1[4];
RDy1[1] = + R2[5]*Dy1[0] + R2[6]*Dy1[1] + R2[7]*Dy1[2] + R2[8]*Dy1[3] + R2[9]*Dy1[4];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3] + Q2[4]*Dy1[4];
QDy1[1] = + Q2[5]*Dy1[0] + Q2[6]*Dy1[1] + Q2[7]*Dy1[2] + Q2[8]*Dy1[3] + Q2[9]*Dy1[4];
QDy1[2] = + Q2[10]*Dy1[0] + Q2[11]*Dy1[1] + Q2[12]*Dy1[2] + Q2[13]*Dy1[3] + Q2[14]*Dy1[4];
QDy1[3] = + Q2[15]*Dy1[0] + Q2[16]*Dy1[1] + Q2[17]*Dy1[2] + Q2[18]*Dy1[3] + Q2[19]*Dy1[4];
}

void acado_multEQDy( real_t* const E1, real_t* const QDy1, real_t* const U1 )
{
U1[0] += + E1[0]*QDy1[0] + E1[2]*QDy1[1] + E1[4]*QDy1[2] + E1[6]*QDy1[3];
U1[1] += + E1[1]*QDy1[0] + E1[3]*QDy1[1] + E1[5]*QDy1[2] + E1[7]*QDy1[3];
}

void acado_multQETGx( real_t* const E1, real_t* const Gx1, real_t* const H101 )
{
H101[0] += + E1[0]*Gx1[0] + E1[2]*Gx1[4] + E1[4]*Gx1[8] + E1[6]*Gx1[12];
H101[1] += + E1[0]*Gx1[1] + E1[2]*Gx1[5] + E1[4]*Gx1[9] + E1[6]*Gx1[13];
H101[2] += + E1[0]*Gx1[2] + E1[2]*Gx1[6] + E1[4]*Gx1[10] + E1[6]*Gx1[14];
H101[3] += + E1[0]*Gx1[3] + E1[2]*Gx1[7] + E1[4]*Gx1[11] + E1[6]*Gx1[15];
H101[4] += + E1[1]*Gx1[0] + E1[3]*Gx1[4] + E1[5]*Gx1[8] + E1[7]*Gx1[12];
H101[5] += + E1[1]*Gx1[1] + E1[3]*Gx1[5] + E1[5]*Gx1[9] + E1[7]*Gx1[13];
H101[6] += + E1[1]*Gx1[2] + E1[3]*Gx1[6] + E1[5]*Gx1[10] + E1[7]*Gx1[14];
H101[7] += + E1[1]*Gx1[3] + E1[3]*Gx1[7] + E1[5]*Gx1[11] + E1[7]*Gx1[15];
}

void acado_zeroBlockH10( real_t* const H101 )
{
{ int lCopy; for (lCopy = 0; lCopy < 8; lCopy++) H101[ lCopy ] = 0; }
}

void acado_multEDu( real_t* const E1, real_t* const U1, real_t* const dNew )
{
dNew[0] += + E1[0]*U1[0] + E1[1]*U1[1];
dNew[1] += + E1[2]*U1[0] + E1[3]*U1[1];
dNew[2] += + E1[4]*U1[0] + E1[5]*U1[1];
dNew[3] += + E1[6]*U1[0] + E1[7]*U1[1];
}

void acado_zeroBlockH00(  )
{
acadoWorkspace.H[0] = 0.0000000000000000e+00;
acadoWorkspace.H[1] = 0.0000000000000000e+00;
acadoWorkspace.H[2] = 0.0000000000000000e+00;
acadoWorkspace.H[3] = 0.0000000000000000e+00;
acadoWorkspace.H[68] = 0.0000000000000000e+00;
acadoWorkspace.H[69] = 0.0000000000000000e+00;
acadoWorkspace.H[70] = 0.0000000000000000e+00;
acadoWorkspace.H[71] = 0.0000000000000000e+00;
acadoWorkspace.H[136] = 0.0000000000000000e+00;
acadoWorkspace.H[137] = 0.0000000000000000e+00;
acadoWorkspace.H[138] = 0.0000000000000000e+00;
acadoWorkspace.H[139] = 0.0000000000000000e+00;
acadoWorkspace.H[204] = 0.0000000000000000e+00;
acadoWorkspace.H[205] = 0.0000000000000000e+00;
acadoWorkspace.H[206] = 0.0000000000000000e+00;
acadoWorkspace.H[207] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[4]*Gx2[4] + Gx1[8]*Gx2[8] + Gx1[12]*Gx2[12];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[4]*Gx2[5] + Gx1[8]*Gx2[9] + Gx1[12]*Gx2[13];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[4]*Gx2[6] + Gx1[8]*Gx2[10] + Gx1[12]*Gx2[14];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[4]*Gx2[7] + Gx1[8]*Gx2[11] + Gx1[12]*Gx2[15];
acadoWorkspace.H[68] += + Gx1[1]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[9]*Gx2[8] + Gx1[13]*Gx2[12];
acadoWorkspace.H[69] += + Gx1[1]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[9]*Gx2[9] + Gx1[13]*Gx2[13];
acadoWorkspace.H[70] += + Gx1[1]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[9]*Gx2[10] + Gx1[13]*Gx2[14];
acadoWorkspace.H[71] += + Gx1[1]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[9]*Gx2[11] + Gx1[13]*Gx2[15];
acadoWorkspace.H[136] += + Gx1[2]*Gx2[0] + Gx1[6]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[14]*Gx2[12];
acadoWorkspace.H[137] += + Gx1[2]*Gx2[1] + Gx1[6]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[14]*Gx2[13];
acadoWorkspace.H[138] += + Gx1[2]*Gx2[2] + Gx1[6]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[14]*Gx2[14];
acadoWorkspace.H[139] += + Gx1[2]*Gx2[3] + Gx1[6]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[14]*Gx2[15];
acadoWorkspace.H[204] += + Gx1[3]*Gx2[0] + Gx1[7]*Gx2[4] + Gx1[11]*Gx2[8] + Gx1[15]*Gx2[12];
acadoWorkspace.H[205] += + Gx1[3]*Gx2[1] + Gx1[7]*Gx2[5] + Gx1[11]*Gx2[9] + Gx1[15]*Gx2[13];
acadoWorkspace.H[206] += + Gx1[3]*Gx2[2] + Gx1[7]*Gx2[6] + Gx1[11]*Gx2[10] + Gx1[15]*Gx2[14];
acadoWorkspace.H[207] += + Gx1[3]*Gx2[3] + Gx1[7]*Gx2[7] + Gx1[11]*Gx2[11] + Gx1[15]*Gx2[15];
}

void acado_multHxC( real_t* const Hx, real_t* const Gx, real_t* const A01 )
{
A01[0] = + Hx[0]*Gx[0] + Hx[1]*Gx[4] + Hx[2]*Gx[8] + Hx[3]*Gx[12];
A01[1] = + Hx[0]*Gx[1] + Hx[1]*Gx[5] + Hx[2]*Gx[9] + Hx[3]*Gx[13];
A01[2] = + Hx[0]*Gx[2] + Hx[1]*Gx[6] + Hx[2]*Gx[10] + Hx[3]*Gx[14];
A01[3] = + Hx[0]*Gx[3] + Hx[1]*Gx[7] + Hx[2]*Gx[11] + Hx[3]*Gx[15];
A01[68] = + Hx[4]*Gx[0] + Hx[5]*Gx[4] + Hx[6]*Gx[8] + Hx[7]*Gx[12];
A01[69] = + Hx[4]*Gx[1] + Hx[5]*Gx[5] + Hx[6]*Gx[9] + Hx[7]*Gx[13];
A01[70] = + Hx[4]*Gx[2] + Hx[5]*Gx[6] + Hx[6]*Gx[10] + Hx[7]*Gx[14];
A01[71] = + Hx[4]*Gx[3] + Hx[5]*Gx[7] + Hx[6]*Gx[11] + Hx[7]*Gx[15];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 136 + 2176) + (col * 2 + 4)] = + Hx[0]*E[0] + Hx[1]*E[2] + Hx[2]*E[4] + Hx[3]*E[6];
acadoWorkspace.A[(row * 136 + 2176) + (col * 2 + 5)] = + Hx[0]*E[1] + Hx[1]*E[3] + Hx[2]*E[5] + Hx[3]*E[7];
acadoWorkspace.A[(row * 136 + 2244) + (col * 2 + 4)] = + Hx[4]*E[0] + Hx[5]*E[2] + Hx[6]*E[4] + Hx[7]*E[6];
acadoWorkspace.A[(row * 136 + 2244) + (col * 2 + 5)] = + Hx[4]*E[1] + Hx[5]*E[3] + Hx[6]*E[5] + Hx[7]*E[7];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2] + Hx[3]*tmpd[3];
acadoWorkspace.evHxd[1] = + Hx[4]*tmpd[0] + Hx[5]*tmpd[1] + Hx[6]*tmpd[2] + Hx[7]*tmpd[3];
lbA[0] -= acadoWorkspace.evHxd[0];
lbA[1] -= acadoWorkspace.evHxd[1];
ubA[0] -= acadoWorkspace.evHxd[0];
ubA[1] -= acadoWorkspace.evHxd[1];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 4;
const real_t* od = in + 6;
/* Vector of auxiliary variables; number of elements: 12. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (real_t)(0.0000000000000000e+00);
a[1] = (real_t)(0.0000000000000000e+00);
a[2] = (real_t)(1.0000000000000000e+00);
a[3] = (real_t)(0.0000000000000000e+00);
a[4] = (real_t)(0.0000000000000000e+00);
a[5] = (real_t)(0.0000000000000000e+00);
a[6] = (real_t)(1.0000000000000000e+00);
a[7] = (real_t)(0.0000000000000000e+00);
a[8] = (real_t)(0.0000000000000000e+00);
a[9] = (real_t)(1.0000000000000000e+00);
a[10] = (real_t)(0.0000000000000000e+00);
a[11] = (real_t)(1.0000000000000000e+00);

/* Compute outputs: */
out[0] = ((xd[2]-od[0])+u[1]);
out[1] = ((xd[2]-od[1])+u[1]);
out[2] = a[0];
out[3] = a[1];
out[4] = a[2];
out[5] = a[3];
out[6] = a[4];
out[7] = a[5];
out[8] = a[6];
out[9] = a[7];
out[10] = a[8];
out[11] = a[9];
out[12] = a[10];
out[13] = a[11];
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
}

void acado_macETSlu( real_t* const E0, real_t* const g1 )
{
g1[0] += 0.0;
;
g1[1] += 0.0;
;
}

void acado_condensePrep(  )
{
int lRun1;
int lRun2;
int lRun3;
int lRun4;
int lRun5;
/** Row vector of size: 32 */
static const int xBoundIndices[ 32 ] = 
{ 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
for (lRun1 = 1; lRun1 < 32; ++lRun1)
{
acado_moveGxT( &(acadoWorkspace.evGx[ lRun1 * 16 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ lRun1 * 4-4 ]), &(acadoWorkspace.evGx[ lRun1 * 16 ]), &(acadoWorkspace.d[ lRun1 * 4 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ lRun1 * 16-16 ]), &(acadoWorkspace.evGx[ lRun1 * 16 ]) );
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
lRun4 = (((lRun1) * (lRun1-1)) / (2)) + (lRun2);
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ lRun4 * 8 ]), &(acadoWorkspace.E[ lRun3 * 8 ]) );
}
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_moveGuE( &(acadoWorkspace.evGu[ lRun1 * 8 ]), &(acadoWorkspace.E[ lRun3 * 8 ]) );
}

acado_multGxGx( &(acadoWorkspace.Q1[ 16 ]), acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multGxGx( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.QGx[ 16 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.QGx[ 32 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.QGx[ 48 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.QGx[ 64 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.QGx[ 80 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.QGx[ 96 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.QGx[ 112 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.QGx[ 128 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.QGx[ 160 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.QGx[ 176 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.QGx[ 192 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.QGx[ 208 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.QGx[ 224 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.QGx[ 240 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.QGx[ 256 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.QGx[ 272 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.QGx[ 288 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 320 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.QGx[ 304 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 336 ]), &(acadoWorkspace.evGx[ 320 ]), &(acadoWorkspace.QGx[ 320 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 352 ]), &(acadoWorkspace.evGx[ 336 ]), &(acadoWorkspace.QGx[ 336 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 368 ]), &(acadoWorkspace.evGx[ 352 ]), &(acadoWorkspace.QGx[ 352 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 384 ]), &(acadoWorkspace.evGx[ 368 ]), &(acadoWorkspace.QGx[ 368 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.evGx[ 384 ]), &(acadoWorkspace.QGx[ 384 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 416 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.QGx[ 400 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.evGx[ 416 ]), &(acadoWorkspace.QGx[ 416 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 448 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.QGx[ 432 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 464 ]), &(acadoWorkspace.evGx[ 448 ]), &(acadoWorkspace.QGx[ 448 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 480 ]), &(acadoWorkspace.evGx[ 464 ]), &(acadoWorkspace.QGx[ 464 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 496 ]), &(acadoWorkspace.evGx[ 480 ]), &(acadoWorkspace.QGx[ 480 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 496 ]), &(acadoWorkspace.QGx[ 496 ]) );

for (lRun1 = 0; lRun1 < 31; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( &(acadoWorkspace.Q1[ lRun1 * 16 + 16 ]), &(acadoWorkspace.E[ lRun3 * 8 ]), &(acadoWorkspace.QE[ lRun3 * 8 ]) );
}
}

for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ lRun3 * 8 ]), &(acadoWorkspace.QE[ lRun3 * 8 ]) );
}

acado_zeroBlockH00(  );
acado_multCTQC( acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multCTQC( &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.QGx[ 16 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.QGx[ 32 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.QGx[ 48 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.QGx[ 64 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.QGx[ 80 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.QGx[ 96 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.QGx[ 112 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.QGx[ 128 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.QGx[ 160 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.QGx[ 176 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.QGx[ 192 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.QGx[ 208 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.QGx[ 224 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.QGx[ 240 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.QGx[ 256 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.QGx[ 272 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.QGx[ 288 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.QGx[ 304 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 320 ]), &(acadoWorkspace.QGx[ 320 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 336 ]), &(acadoWorkspace.QGx[ 336 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 352 ]), &(acadoWorkspace.QGx[ 352 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 368 ]), &(acadoWorkspace.QGx[ 368 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 384 ]), &(acadoWorkspace.QGx[ 384 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.QGx[ 400 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 416 ]), &(acadoWorkspace.QGx[ 416 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.QGx[ 432 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 448 ]), &(acadoWorkspace.QGx[ 448 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 464 ]), &(acadoWorkspace.QGx[ 464 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 480 ]), &(acadoWorkspace.QGx[ 480 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 496 ]), &(acadoWorkspace.QGx[ 496 ]) );

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_zeroBlockH10( &(acadoWorkspace.H10[ lRun1 * 8 ]) );
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multQETGx( &(acadoWorkspace.QE[ lRun3 * 8 ]), &(acadoWorkspace.evGx[ lRun2 * 16 ]), &(acadoWorkspace.H10[ lRun1 * 8 ]) );
}
}

for (lRun2 = 0;lRun2 < 4; ++lRun2)
for (lRun3 = 0;lRun3 < 64; ++lRun3)
acadoWorkspace.H[(lRun2 * 68) + (lRun3 + 4)] = acadoWorkspace.H10[(lRun3 * 4) + (lRun2)];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_setBlockH11_R1( lRun1, lRun1, &(acadoWorkspace.R1[ lRun1 * 4 ]) );
lRun2 = lRun1;
for (lRun3 = lRun1; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 8 ]), &(acadoWorkspace.QE[ lRun5 * 8 ]) );
}
for (lRun2 = lRun1 + 1; lRun2 < 32; ++lRun2)
{
acado_zeroBlockH11( lRun1, lRun2 );
for (lRun3 = lRun2; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 8 ]), &(acadoWorkspace.QE[ lRun5 * 8 ]) );
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

for (lRun2 = 0;lRun2 < 64; ++lRun2)
for (lRun3 = 0;lRun3 < 4; ++lRun3)
acadoWorkspace.H[(lRun2 * 68 + 272) + (lRun3)] = acadoWorkspace.H10[(lRun2 * 4) + (lRun3)];

acado_multQ1d( &(acadoWorkspace.Q1[ 16 ]), acadoWorkspace.d, acadoWorkspace.Qd );
acado_multQ1d( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.d[ 4 ]), &(acadoWorkspace.Qd[ 4 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.d[ 8 ]), &(acadoWorkspace.Qd[ 8 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.Qd[ 12 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.d[ 16 ]), &(acadoWorkspace.Qd[ 16 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.Qd[ 20 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.Qd[ 24 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.d[ 28 ]), &(acadoWorkspace.Qd[ 28 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.d[ 32 ]), &(acadoWorkspace.Qd[ 32 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.Qd[ 36 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.Qd[ 40 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.d[ 44 ]), &(acadoWorkspace.Qd[ 44 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.Qd[ 48 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.d[ 52 ]), &(acadoWorkspace.Qd[ 52 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.d[ 56 ]), &(acadoWorkspace.Qd[ 56 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.Qd[ 60 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.d[ 64 ]), &(acadoWorkspace.Qd[ 64 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.d[ 68 ]), &(acadoWorkspace.Qd[ 68 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.Qd[ 72 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 320 ]), &(acadoWorkspace.d[ 76 ]), &(acadoWorkspace.Qd[ 76 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 336 ]), &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.Qd[ 80 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 352 ]), &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.Qd[ 84 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 368 ]), &(acadoWorkspace.d[ 88 ]), &(acadoWorkspace.Qd[ 88 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 384 ]), &(acadoWorkspace.d[ 92 ]), &(acadoWorkspace.Qd[ 92 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.d[ 96 ]), &(acadoWorkspace.Qd[ 96 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 416 ]), &(acadoWorkspace.d[ 100 ]), &(acadoWorkspace.Qd[ 100 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.d[ 104 ]), &(acadoWorkspace.Qd[ 104 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 448 ]), &(acadoWorkspace.d[ 108 ]), &(acadoWorkspace.Qd[ 108 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 464 ]), &(acadoWorkspace.d[ 112 ]), &(acadoWorkspace.Qd[ 112 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 480 ]), &(acadoWorkspace.d[ 116 ]), &(acadoWorkspace.Qd[ 116 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 496 ]), &(acadoWorkspace.d[ 120 ]), &(acadoWorkspace.Qd[ 120 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 124 ]), &(acadoWorkspace.Qd[ 124 ]) );

acado_macCTSlx( acadoWorkspace.evGx, acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 256 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 272 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 304 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 320 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 336 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 352 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 368 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 384 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 400 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 416 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 432 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 448 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 464 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 480 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 496 ]), acadoWorkspace.g );
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_macETSlu( &(acadoWorkspace.QE[ lRun3 * 8 ]), &(acadoWorkspace.g[ lRun1 * 2 + 4 ]) );
}
}
acadoWorkspace.lb[4] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.lb[5] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.lb[6] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.lb[10] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.lb[11] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.lb[12] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.lb[13] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.lb[14] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.lb[15] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.lb[16] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.lb[17] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.lb[18] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.lb[19] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.lb[20] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.lb[21] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.lb[22] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.lb[23] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.lb[24] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.lb[25] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.lb[26] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.lb[27] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.lb[28] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.lb[29] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.lb[30] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.lb[31] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.lb[32] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.lb[33] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.lb[34] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.lb[35] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[31];
acadoWorkspace.lb[36] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[32];
acadoWorkspace.lb[37] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[33];
acadoWorkspace.lb[38] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[34];
acadoWorkspace.lb[39] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[35];
acadoWorkspace.lb[40] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[36];
acadoWorkspace.lb[41] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[37];
acadoWorkspace.lb[42] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[38];
acadoWorkspace.lb[43] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[39];
acadoWorkspace.lb[44] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[40];
acadoWorkspace.lb[45] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[41];
acadoWorkspace.lb[46] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[42];
acadoWorkspace.lb[47] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[43];
acadoWorkspace.lb[48] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[44];
acadoWorkspace.lb[49] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[45];
acadoWorkspace.lb[50] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[46];
acadoWorkspace.lb[51] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[47];
acadoWorkspace.lb[52] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[48];
acadoWorkspace.lb[53] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[49];
acadoWorkspace.lb[54] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[50];
acadoWorkspace.lb[55] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[51];
acadoWorkspace.lb[56] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[52];
acadoWorkspace.lb[57] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[53];
acadoWorkspace.lb[58] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[54];
acadoWorkspace.lb[59] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[55];
acadoWorkspace.lb[60] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[56];
acadoWorkspace.lb[61] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[57];
acadoWorkspace.lb[62] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[58];
acadoWorkspace.lb[63] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[59];
acadoWorkspace.lb[64] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[60];
acadoWorkspace.lb[65] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[61];
acadoWorkspace.lb[66] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[62];
acadoWorkspace.lb[67] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[63];
acadoWorkspace.ub[4] = (real_t)1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.ub[5] = (real_t)1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.ub[6] = (real_t)1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.ub[7] = (real_t)1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.ub[8] = (real_t)1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.ub[9] = (real_t)1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.ub[10] = (real_t)1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.ub[11] = (real_t)1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.ub[12] = (real_t)1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.ub[13] = (real_t)1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.ub[14] = (real_t)1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.ub[15] = (real_t)1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.ub[16] = (real_t)1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.ub[17] = (real_t)1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.ub[18] = (real_t)1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.ub[19] = (real_t)1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.ub[20] = (real_t)1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.ub[21] = (real_t)1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.ub[22] = (real_t)1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.ub[23] = (real_t)1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.ub[24] = (real_t)1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.ub[25] = (real_t)1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.ub[26] = (real_t)1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.ub[27] = (real_t)1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.ub[28] = (real_t)1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.ub[29] = (real_t)1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.ub[30] = (real_t)1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.ub[31] = (real_t)1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.ub[32] = (real_t)1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.ub[33] = (real_t)1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.ub[34] = (real_t)1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.ub[35] = (real_t)1.0000000000000000e+12 - acadoVariables.u[31];
acadoWorkspace.ub[36] = (real_t)1.0000000000000000e+12 - acadoVariables.u[32];
acadoWorkspace.ub[37] = (real_t)1.0000000000000000e+12 - acadoVariables.u[33];
acadoWorkspace.ub[38] = (real_t)1.0000000000000000e+12 - acadoVariables.u[34];
acadoWorkspace.ub[39] = (real_t)1.0000000000000000e+12 - acadoVariables.u[35];
acadoWorkspace.ub[40] = (real_t)1.0000000000000000e+12 - acadoVariables.u[36];
acadoWorkspace.ub[41] = (real_t)1.0000000000000000e+12 - acadoVariables.u[37];
acadoWorkspace.ub[42] = (real_t)1.0000000000000000e+12 - acadoVariables.u[38];
acadoWorkspace.ub[43] = (real_t)1.0000000000000000e+12 - acadoVariables.u[39];
acadoWorkspace.ub[44] = (real_t)1.0000000000000000e+12 - acadoVariables.u[40];
acadoWorkspace.ub[45] = (real_t)1.0000000000000000e+12 - acadoVariables.u[41];
acadoWorkspace.ub[46] = (real_t)1.0000000000000000e+12 - acadoVariables.u[42];
acadoWorkspace.ub[47] = (real_t)1.0000000000000000e+12 - acadoVariables.u[43];
acadoWorkspace.ub[48] = (real_t)1.0000000000000000e+12 - acadoVariables.u[44];
acadoWorkspace.ub[49] = (real_t)1.0000000000000000e+12 - acadoVariables.u[45];
acadoWorkspace.ub[50] = (real_t)1.0000000000000000e+12 - acadoVariables.u[46];
acadoWorkspace.ub[51] = (real_t)1.0000000000000000e+12 - acadoVariables.u[47];
acadoWorkspace.ub[52] = (real_t)1.0000000000000000e+12 - acadoVariables.u[48];
acadoWorkspace.ub[53] = (real_t)1.0000000000000000e+12 - acadoVariables.u[49];
acadoWorkspace.ub[54] = (real_t)1.0000000000000000e+12 - acadoVariables.u[50];
acadoWorkspace.ub[55] = (real_t)1.0000000000000000e+12 - acadoVariables.u[51];
acadoWorkspace.ub[56] = (real_t)1.0000000000000000e+12 - acadoVariables.u[52];
acadoWorkspace.ub[57] = (real_t)1.0000000000000000e+12 - acadoVariables.u[53];
acadoWorkspace.ub[58] = (real_t)1.0000000000000000e+12 - acadoVariables.u[54];
acadoWorkspace.ub[59] = (real_t)1.0000000000000000e+12 - acadoVariables.u[55];
acadoWorkspace.ub[60] = (real_t)1.0000000000000000e+12 - acadoVariables.u[56];
acadoWorkspace.ub[61] = (real_t)1.0000000000000000e+12 - acadoVariables.u[57];
acadoWorkspace.ub[62] = (real_t)1.0000000000000000e+12 - acadoVariables.u[58];
acadoWorkspace.ub[63] = (real_t)1.0000000000000000e+12 - acadoVariables.u[59];
acadoWorkspace.ub[64] = (real_t)1.0000000000000000e+12 - acadoVariables.u[60];
acadoWorkspace.ub[65] = (real_t)1.0000000000000000e+12 - acadoVariables.u[61];
acadoWorkspace.ub[66] = (real_t)1.0000000000000000e+12 - acadoVariables.u[62];
acadoWorkspace.ub[67] = (real_t)1.0000000000000000e+12 - acadoVariables.u[63];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 4;
lRun4 = ((lRun3) / (4)) + (1);
acadoWorkspace.A[lRun1 * 68] = acadoWorkspace.evGx[lRun3 * 4];
acadoWorkspace.A[lRun1 * 68 + 1] = acadoWorkspace.evGx[lRun3 * 4 + 1];
acadoWorkspace.A[lRun1 * 68 + 2] = acadoWorkspace.evGx[lRun3 * 4 + 2];
acadoWorkspace.A[lRun1 * 68 + 3] = acadoWorkspace.evGx[lRun3 * 4 + 3];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (4)) + ((lRun3) % (4));
acadoWorkspace.A[(lRun1 * 68) + (lRun2 * 2 + 4)] = acadoWorkspace.E[lRun5 * 2];
acadoWorkspace.A[(lRun1 * 68) + (lRun2 * 2 + 5)] = acadoWorkspace.E[lRun5 * 2 + 1];
}
}

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acadoWorkspace.conValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.conValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.conValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.conValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.conValueIn[4] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.conValueIn[5] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.conValueIn[6] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.conValueIn[7] = acadoVariables.od[lRun1 * 2 + 1];
acado_evaluatePathConstraints( acadoWorkspace.conValueIn, acadoWorkspace.conValueOut );
acadoWorkspace.evH[lRun1 * 2] = acadoWorkspace.conValueOut[0];
acadoWorkspace.evH[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[1];

acadoWorkspace.evHx[lRun1 * 8] = acadoWorkspace.conValueOut[2];
acadoWorkspace.evHx[lRun1 * 8 + 1] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 8 + 2] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHx[lRun1 * 8 + 3] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHx[lRun1 * 8 + 4] = acadoWorkspace.conValueOut[6];
acadoWorkspace.evHx[lRun1 * 8 + 5] = acadoWorkspace.conValueOut[7];
acadoWorkspace.evHx[lRun1 * 8 + 6] = acadoWorkspace.conValueOut[8];
acadoWorkspace.evHx[lRun1 * 8 + 7] = acadoWorkspace.conValueOut[9];
acadoWorkspace.evHu[lRun1 * 4] = acadoWorkspace.conValueOut[10];
acadoWorkspace.evHu[lRun1 * 4 + 1] = acadoWorkspace.conValueOut[11];
acadoWorkspace.evHu[lRun1 * 4 + 2] = acadoWorkspace.conValueOut[12];
acadoWorkspace.evHu[lRun1 * 4 + 3] = acadoWorkspace.conValueOut[13];
}

acadoWorkspace.A[2176] = acadoWorkspace.evHx[0];
acadoWorkspace.A[2177] = acadoWorkspace.evHx[1];
acadoWorkspace.A[2178] = acadoWorkspace.evHx[2];
acadoWorkspace.A[2179] = acadoWorkspace.evHx[3];
acadoWorkspace.A[2244] = acadoWorkspace.evHx[4];
acadoWorkspace.A[2245] = acadoWorkspace.evHx[5];
acadoWorkspace.A[2246] = acadoWorkspace.evHx[6];
acadoWorkspace.A[2247] = acadoWorkspace.evHx[7];

acado_multHxC( &(acadoWorkspace.evHx[ 8 ]), acadoWorkspace.evGx, &(acadoWorkspace.A[ 2312 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.A[ 2448 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.A[ 2584 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.A[ 2720 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.A[ 2856 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.A[ 2992 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.A[ 3128 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.A[ 3264 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.A[ 3400 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.A[ 3536 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 88 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.A[ 3672 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 96 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.A[ 3808 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 104 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.A[ 3944 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 112 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.A[ 4080 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.A[ 4216 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 128 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.A[ 4352 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 136 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.A[ 4488 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 144 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.A[ 4624 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 152 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.A[ 4760 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.A[ 4896 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 168 ]), &(acadoWorkspace.evGx[ 320 ]), &(acadoWorkspace.A[ 5032 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 176 ]), &(acadoWorkspace.evGx[ 336 ]), &(acadoWorkspace.A[ 5168 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 184 ]), &(acadoWorkspace.evGx[ 352 ]), &(acadoWorkspace.A[ 5304 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 192 ]), &(acadoWorkspace.evGx[ 368 ]), &(acadoWorkspace.A[ 5440 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 200 ]), &(acadoWorkspace.evGx[ 384 ]), &(acadoWorkspace.A[ 5576 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 208 ]), &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.A[ 5712 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 216 ]), &(acadoWorkspace.evGx[ 416 ]), &(acadoWorkspace.A[ 5848 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 224 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.A[ 5984 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 232 ]), &(acadoWorkspace.evGx[ 448 ]), &(acadoWorkspace.A[ 6120 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 240 ]), &(acadoWorkspace.evGx[ 464 ]), &(acadoWorkspace.A[ 6256 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 248 ]), &(acadoWorkspace.evGx[ 480 ]), &(acadoWorkspace.A[ 6392 ]) );

for (lRun2 = 0; lRun2 < 31; ++lRun2)
{
for (lRun3 = 0; lRun3 < lRun2 + 1; ++lRun3)
{
lRun4 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun3);
lRun5 = lRun2 + 1;
acado_multHxE( &(acadoWorkspace.evHx[ lRun2 * 8 + 8 ]), &(acadoWorkspace.E[ lRun4 * 8 ]), lRun5, lRun3 );
}
}

acadoWorkspace.A[2180] = acadoWorkspace.evHu[0];
acadoWorkspace.A[2181] = acadoWorkspace.evHu[1];
acadoWorkspace.A[2248] = acadoWorkspace.evHu[2];
acadoWorkspace.A[2249] = acadoWorkspace.evHu[3];
acadoWorkspace.A[2318] = acadoWorkspace.evHu[4];
acadoWorkspace.A[2319] = acadoWorkspace.evHu[5];
acadoWorkspace.A[2386] = acadoWorkspace.evHu[6];
acadoWorkspace.A[2387] = acadoWorkspace.evHu[7];
acadoWorkspace.A[2456] = acadoWorkspace.evHu[8];
acadoWorkspace.A[2457] = acadoWorkspace.evHu[9];
acadoWorkspace.A[2524] = acadoWorkspace.evHu[10];
acadoWorkspace.A[2525] = acadoWorkspace.evHu[11];
acadoWorkspace.A[2594] = acadoWorkspace.evHu[12];
acadoWorkspace.A[2595] = acadoWorkspace.evHu[13];
acadoWorkspace.A[2662] = acadoWorkspace.evHu[14];
acadoWorkspace.A[2663] = acadoWorkspace.evHu[15];
acadoWorkspace.A[2732] = acadoWorkspace.evHu[16];
acadoWorkspace.A[2733] = acadoWorkspace.evHu[17];
acadoWorkspace.A[2800] = acadoWorkspace.evHu[18];
acadoWorkspace.A[2801] = acadoWorkspace.evHu[19];
acadoWorkspace.A[2870] = acadoWorkspace.evHu[20];
acadoWorkspace.A[2871] = acadoWorkspace.evHu[21];
acadoWorkspace.A[2938] = acadoWorkspace.evHu[22];
acadoWorkspace.A[2939] = acadoWorkspace.evHu[23];
acadoWorkspace.A[3008] = acadoWorkspace.evHu[24];
acadoWorkspace.A[3009] = acadoWorkspace.evHu[25];
acadoWorkspace.A[3076] = acadoWorkspace.evHu[26];
acadoWorkspace.A[3077] = acadoWorkspace.evHu[27];
acadoWorkspace.A[3146] = acadoWorkspace.evHu[28];
acadoWorkspace.A[3147] = acadoWorkspace.evHu[29];
acadoWorkspace.A[3214] = acadoWorkspace.evHu[30];
acadoWorkspace.A[3215] = acadoWorkspace.evHu[31];
acadoWorkspace.A[3284] = acadoWorkspace.evHu[32];
acadoWorkspace.A[3285] = acadoWorkspace.evHu[33];
acadoWorkspace.A[3352] = acadoWorkspace.evHu[34];
acadoWorkspace.A[3353] = acadoWorkspace.evHu[35];
acadoWorkspace.A[3422] = acadoWorkspace.evHu[36];
acadoWorkspace.A[3423] = acadoWorkspace.evHu[37];
acadoWorkspace.A[3490] = acadoWorkspace.evHu[38];
acadoWorkspace.A[3491] = acadoWorkspace.evHu[39];
acadoWorkspace.A[3560] = acadoWorkspace.evHu[40];
acadoWorkspace.A[3561] = acadoWorkspace.evHu[41];
acadoWorkspace.A[3628] = acadoWorkspace.evHu[42];
acadoWorkspace.A[3629] = acadoWorkspace.evHu[43];
acadoWorkspace.A[3698] = acadoWorkspace.evHu[44];
acadoWorkspace.A[3699] = acadoWorkspace.evHu[45];
acadoWorkspace.A[3766] = acadoWorkspace.evHu[46];
acadoWorkspace.A[3767] = acadoWorkspace.evHu[47];
acadoWorkspace.A[3836] = acadoWorkspace.evHu[48];
acadoWorkspace.A[3837] = acadoWorkspace.evHu[49];
acadoWorkspace.A[3904] = acadoWorkspace.evHu[50];
acadoWorkspace.A[3905] = acadoWorkspace.evHu[51];
acadoWorkspace.A[3974] = acadoWorkspace.evHu[52];
acadoWorkspace.A[3975] = acadoWorkspace.evHu[53];
acadoWorkspace.A[4042] = acadoWorkspace.evHu[54];
acadoWorkspace.A[4043] = acadoWorkspace.evHu[55];
acadoWorkspace.A[4112] = acadoWorkspace.evHu[56];
acadoWorkspace.A[4113] = acadoWorkspace.evHu[57];
acadoWorkspace.A[4180] = acadoWorkspace.evHu[58];
acadoWorkspace.A[4181] = acadoWorkspace.evHu[59];
acadoWorkspace.A[4250] = acadoWorkspace.evHu[60];
acadoWorkspace.A[4251] = acadoWorkspace.evHu[61];
acadoWorkspace.A[4318] = acadoWorkspace.evHu[62];
acadoWorkspace.A[4319] = acadoWorkspace.evHu[63];
acadoWorkspace.A[4388] = acadoWorkspace.evHu[64];
acadoWorkspace.A[4389] = acadoWorkspace.evHu[65];
acadoWorkspace.A[4456] = acadoWorkspace.evHu[66];
acadoWorkspace.A[4457] = acadoWorkspace.evHu[67];
acadoWorkspace.A[4526] = acadoWorkspace.evHu[68];
acadoWorkspace.A[4527] = acadoWorkspace.evHu[69];
acadoWorkspace.A[4594] = acadoWorkspace.evHu[70];
acadoWorkspace.A[4595] = acadoWorkspace.evHu[71];
acadoWorkspace.A[4664] = acadoWorkspace.evHu[72];
acadoWorkspace.A[4665] = acadoWorkspace.evHu[73];
acadoWorkspace.A[4732] = acadoWorkspace.evHu[74];
acadoWorkspace.A[4733] = acadoWorkspace.evHu[75];
acadoWorkspace.A[4802] = acadoWorkspace.evHu[76];
acadoWorkspace.A[4803] = acadoWorkspace.evHu[77];
acadoWorkspace.A[4870] = acadoWorkspace.evHu[78];
acadoWorkspace.A[4871] = acadoWorkspace.evHu[79];
acadoWorkspace.A[4940] = acadoWorkspace.evHu[80];
acadoWorkspace.A[4941] = acadoWorkspace.evHu[81];
acadoWorkspace.A[5008] = acadoWorkspace.evHu[82];
acadoWorkspace.A[5009] = acadoWorkspace.evHu[83];
acadoWorkspace.A[5078] = acadoWorkspace.evHu[84];
acadoWorkspace.A[5079] = acadoWorkspace.evHu[85];
acadoWorkspace.A[5146] = acadoWorkspace.evHu[86];
acadoWorkspace.A[5147] = acadoWorkspace.evHu[87];
acadoWorkspace.A[5216] = acadoWorkspace.evHu[88];
acadoWorkspace.A[5217] = acadoWorkspace.evHu[89];
acadoWorkspace.A[5284] = acadoWorkspace.evHu[90];
acadoWorkspace.A[5285] = acadoWorkspace.evHu[91];
acadoWorkspace.A[5354] = acadoWorkspace.evHu[92];
acadoWorkspace.A[5355] = acadoWorkspace.evHu[93];
acadoWorkspace.A[5422] = acadoWorkspace.evHu[94];
acadoWorkspace.A[5423] = acadoWorkspace.evHu[95];
acadoWorkspace.A[5492] = acadoWorkspace.evHu[96];
acadoWorkspace.A[5493] = acadoWorkspace.evHu[97];
acadoWorkspace.A[5560] = acadoWorkspace.evHu[98];
acadoWorkspace.A[5561] = acadoWorkspace.evHu[99];
acadoWorkspace.A[5630] = acadoWorkspace.evHu[100];
acadoWorkspace.A[5631] = acadoWorkspace.evHu[101];
acadoWorkspace.A[5698] = acadoWorkspace.evHu[102];
acadoWorkspace.A[5699] = acadoWorkspace.evHu[103];
acadoWorkspace.A[5768] = acadoWorkspace.evHu[104];
acadoWorkspace.A[5769] = acadoWorkspace.evHu[105];
acadoWorkspace.A[5836] = acadoWorkspace.evHu[106];
acadoWorkspace.A[5837] = acadoWorkspace.evHu[107];
acadoWorkspace.A[5906] = acadoWorkspace.evHu[108];
acadoWorkspace.A[5907] = acadoWorkspace.evHu[109];
acadoWorkspace.A[5974] = acadoWorkspace.evHu[110];
acadoWorkspace.A[5975] = acadoWorkspace.evHu[111];
acadoWorkspace.A[6044] = acadoWorkspace.evHu[112];
acadoWorkspace.A[6045] = acadoWorkspace.evHu[113];
acadoWorkspace.A[6112] = acadoWorkspace.evHu[114];
acadoWorkspace.A[6113] = acadoWorkspace.evHu[115];
acadoWorkspace.A[6182] = acadoWorkspace.evHu[116];
acadoWorkspace.A[6183] = acadoWorkspace.evHu[117];
acadoWorkspace.A[6250] = acadoWorkspace.evHu[118];
acadoWorkspace.A[6251] = acadoWorkspace.evHu[119];
acadoWorkspace.A[6320] = acadoWorkspace.evHu[120];
acadoWorkspace.A[6321] = acadoWorkspace.evHu[121];
acadoWorkspace.A[6388] = acadoWorkspace.evHu[122];
acadoWorkspace.A[6389] = acadoWorkspace.evHu[123];
acadoWorkspace.A[6458] = acadoWorkspace.evHu[124];
acadoWorkspace.A[6459] = acadoWorkspace.evHu[125];
acadoWorkspace.A[6526] = acadoWorkspace.evHu[126];
acadoWorkspace.A[6527] = acadoWorkspace.evHu[127];
acadoWorkspace.lbA[32] = - acadoWorkspace.evH[0];
acadoWorkspace.lbA[33] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[1];
acadoWorkspace.lbA[34] = - acadoWorkspace.evH[2];
acadoWorkspace.lbA[35] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[3];
acadoWorkspace.lbA[36] = - acadoWorkspace.evH[4];
acadoWorkspace.lbA[37] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[5];
acadoWorkspace.lbA[38] = - acadoWorkspace.evH[6];
acadoWorkspace.lbA[39] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[7];
acadoWorkspace.lbA[40] = - acadoWorkspace.evH[8];
acadoWorkspace.lbA[41] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[9];
acadoWorkspace.lbA[42] = - acadoWorkspace.evH[10];
acadoWorkspace.lbA[43] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[11];
acadoWorkspace.lbA[44] = - acadoWorkspace.evH[12];
acadoWorkspace.lbA[45] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[13];
acadoWorkspace.lbA[46] = - acadoWorkspace.evH[14];
acadoWorkspace.lbA[47] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[15];
acadoWorkspace.lbA[48] = - acadoWorkspace.evH[16];
acadoWorkspace.lbA[49] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[17];
acadoWorkspace.lbA[50] = - acadoWorkspace.evH[18];
acadoWorkspace.lbA[51] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[19];
acadoWorkspace.lbA[52] = - acadoWorkspace.evH[20];
acadoWorkspace.lbA[53] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[21];
acadoWorkspace.lbA[54] = - acadoWorkspace.evH[22];
acadoWorkspace.lbA[55] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[23];
acadoWorkspace.lbA[56] = - acadoWorkspace.evH[24];
acadoWorkspace.lbA[57] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[25];
acadoWorkspace.lbA[58] = - acadoWorkspace.evH[26];
acadoWorkspace.lbA[59] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[27];
acadoWorkspace.lbA[60] = - acadoWorkspace.evH[28];
acadoWorkspace.lbA[61] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[29];
acadoWorkspace.lbA[62] = - acadoWorkspace.evH[30];
acadoWorkspace.lbA[63] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[31];
acadoWorkspace.lbA[64] = - acadoWorkspace.evH[32];
acadoWorkspace.lbA[65] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[33];
acadoWorkspace.lbA[66] = - acadoWorkspace.evH[34];
acadoWorkspace.lbA[67] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[35];
acadoWorkspace.lbA[68] = - acadoWorkspace.evH[36];
acadoWorkspace.lbA[69] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[37];
acadoWorkspace.lbA[70] = - acadoWorkspace.evH[38];
acadoWorkspace.lbA[71] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[39];
acadoWorkspace.lbA[72] = - acadoWorkspace.evH[40];
acadoWorkspace.lbA[73] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[41];
acadoWorkspace.lbA[74] = - acadoWorkspace.evH[42];
acadoWorkspace.lbA[75] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[43];
acadoWorkspace.lbA[76] = - acadoWorkspace.evH[44];
acadoWorkspace.lbA[77] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[45];
acadoWorkspace.lbA[78] = - acadoWorkspace.evH[46];
acadoWorkspace.lbA[79] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[47];
acadoWorkspace.lbA[80] = - acadoWorkspace.evH[48];
acadoWorkspace.lbA[81] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[49];
acadoWorkspace.lbA[82] = - acadoWorkspace.evH[50];
acadoWorkspace.lbA[83] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[51];
acadoWorkspace.lbA[84] = - acadoWorkspace.evH[52];
acadoWorkspace.lbA[85] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[53];
acadoWorkspace.lbA[86] = - acadoWorkspace.evH[54];
acadoWorkspace.lbA[87] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[55];
acadoWorkspace.lbA[88] = - acadoWorkspace.evH[56];
acadoWorkspace.lbA[89] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[57];
acadoWorkspace.lbA[90] = - acadoWorkspace.evH[58];
acadoWorkspace.lbA[91] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[59];
acadoWorkspace.lbA[92] = - acadoWorkspace.evH[60];
acadoWorkspace.lbA[93] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[61];
acadoWorkspace.lbA[94] = - acadoWorkspace.evH[62];
acadoWorkspace.lbA[95] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[63];

acadoWorkspace.ubA[32] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[0];
acadoWorkspace.ubA[33] = - acadoWorkspace.evH[1];
acadoWorkspace.ubA[34] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[2];
acadoWorkspace.ubA[35] = - acadoWorkspace.evH[3];
acadoWorkspace.ubA[36] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[4];
acadoWorkspace.ubA[37] = - acadoWorkspace.evH[5];
acadoWorkspace.ubA[38] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[6];
acadoWorkspace.ubA[39] = - acadoWorkspace.evH[7];
acadoWorkspace.ubA[40] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[8];
acadoWorkspace.ubA[41] = - acadoWorkspace.evH[9];
acadoWorkspace.ubA[42] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[10];
acadoWorkspace.ubA[43] = - acadoWorkspace.evH[11];
acadoWorkspace.ubA[44] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[12];
acadoWorkspace.ubA[45] = - acadoWorkspace.evH[13];
acadoWorkspace.ubA[46] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[14];
acadoWorkspace.ubA[47] = - acadoWorkspace.evH[15];
acadoWorkspace.ubA[48] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[16];
acadoWorkspace.ubA[49] = - acadoWorkspace.evH[17];
acadoWorkspace.ubA[50] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[18];
acadoWorkspace.ubA[51] = - acadoWorkspace.evH[19];
acadoWorkspace.ubA[52] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[20];
acadoWorkspace.ubA[53] = - acadoWorkspace.evH[21];
acadoWorkspace.ubA[54] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[22];
acadoWorkspace.ubA[55] = - acadoWorkspace.evH[23];
acadoWorkspace.ubA[56] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[24];
acadoWorkspace.ubA[57] = - acadoWorkspace.evH[25];
acadoWorkspace.ubA[58] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[26];
acadoWorkspace.ubA[59] = - acadoWorkspace.evH[27];
acadoWorkspace.ubA[60] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[28];
acadoWorkspace.ubA[61] = - acadoWorkspace.evH[29];
acadoWorkspace.ubA[62] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[30];
acadoWorkspace.ubA[63] = - acadoWorkspace.evH[31];
acadoWorkspace.ubA[64] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[32];
acadoWorkspace.ubA[65] = - acadoWorkspace.evH[33];
acadoWorkspace.ubA[66] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[34];
acadoWorkspace.ubA[67] = - acadoWorkspace.evH[35];
acadoWorkspace.ubA[68] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[36];
acadoWorkspace.ubA[69] = - acadoWorkspace.evH[37];
acadoWorkspace.ubA[70] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[38];
acadoWorkspace.ubA[71] = - acadoWorkspace.evH[39];
acadoWorkspace.ubA[72] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[40];
acadoWorkspace.ubA[73] = - acadoWorkspace.evH[41];
acadoWorkspace.ubA[74] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[42];
acadoWorkspace.ubA[75] = - acadoWorkspace.evH[43];
acadoWorkspace.ubA[76] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[44];
acadoWorkspace.ubA[77] = - acadoWorkspace.evH[45];
acadoWorkspace.ubA[78] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[46];
acadoWorkspace.ubA[79] = - acadoWorkspace.evH[47];
acadoWorkspace.ubA[80] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[48];
acadoWorkspace.ubA[81] = - acadoWorkspace.evH[49];
acadoWorkspace.ubA[82] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[50];
acadoWorkspace.ubA[83] = - acadoWorkspace.evH[51];
acadoWorkspace.ubA[84] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[52];
acadoWorkspace.ubA[85] = - acadoWorkspace.evH[53];
acadoWorkspace.ubA[86] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[54];
acadoWorkspace.ubA[87] = - acadoWorkspace.evH[55];
acadoWorkspace.ubA[88] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[56];
acadoWorkspace.ubA[89] = - acadoWorkspace.evH[57];
acadoWorkspace.ubA[90] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[58];
acadoWorkspace.ubA[91] = - acadoWorkspace.evH[59];
acadoWorkspace.ubA[92] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[60];
acadoWorkspace.ubA[93] = - acadoWorkspace.evH[61];
acadoWorkspace.ubA[94] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[62];
acadoWorkspace.ubA[95] = - acadoWorkspace.evH[63];

acado_macHxd( &(acadoWorkspace.evHx[ 8 ]), acadoWorkspace.d, &(acadoWorkspace.lbA[ 34 ]), &(acadoWorkspace.ubA[ 34 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.d[ 4 ]), &(acadoWorkspace.lbA[ 36 ]), &(acadoWorkspace.ubA[ 36 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.d[ 8 ]), &(acadoWorkspace.lbA[ 38 ]), &(acadoWorkspace.ubA[ 38 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.lbA[ 40 ]), &(acadoWorkspace.ubA[ 40 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.d[ 16 ]), &(acadoWorkspace.lbA[ 42 ]), &(acadoWorkspace.ubA[ 42 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.lbA[ 44 ]), &(acadoWorkspace.ubA[ 44 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.lbA[ 46 ]), &(acadoWorkspace.ubA[ 46 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.d[ 28 ]), &(acadoWorkspace.lbA[ 48 ]), &(acadoWorkspace.ubA[ 48 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.d[ 32 ]), &(acadoWorkspace.lbA[ 50 ]), &(acadoWorkspace.ubA[ 50 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 80 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.lbA[ 52 ]), &(acadoWorkspace.ubA[ 52 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 88 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.lbA[ 54 ]), &(acadoWorkspace.ubA[ 54 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 96 ]), &(acadoWorkspace.d[ 44 ]), &(acadoWorkspace.lbA[ 56 ]), &(acadoWorkspace.ubA[ 56 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 104 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.lbA[ 58 ]), &(acadoWorkspace.ubA[ 58 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 112 ]), &(acadoWorkspace.d[ 52 ]), &(acadoWorkspace.lbA[ 60 ]), &(acadoWorkspace.ubA[ 60 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.d[ 56 ]), &(acadoWorkspace.lbA[ 62 ]), &(acadoWorkspace.ubA[ 62 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 128 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.lbA[ 64 ]), &(acadoWorkspace.ubA[ 64 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 136 ]), &(acadoWorkspace.d[ 64 ]), &(acadoWorkspace.lbA[ 66 ]), &(acadoWorkspace.ubA[ 66 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 144 ]), &(acadoWorkspace.d[ 68 ]), &(acadoWorkspace.lbA[ 68 ]), &(acadoWorkspace.ubA[ 68 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 152 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.lbA[ 70 ]), &(acadoWorkspace.ubA[ 70 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 160 ]), &(acadoWorkspace.d[ 76 ]), &(acadoWorkspace.lbA[ 72 ]), &(acadoWorkspace.ubA[ 72 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 168 ]), &(acadoWorkspace.d[ 80 ]), &(acadoWorkspace.lbA[ 74 ]), &(acadoWorkspace.ubA[ 74 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 176 ]), &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.lbA[ 76 ]), &(acadoWorkspace.ubA[ 76 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 184 ]), &(acadoWorkspace.d[ 88 ]), &(acadoWorkspace.lbA[ 78 ]), &(acadoWorkspace.ubA[ 78 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 192 ]), &(acadoWorkspace.d[ 92 ]), &(acadoWorkspace.lbA[ 80 ]), &(acadoWorkspace.ubA[ 80 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 200 ]), &(acadoWorkspace.d[ 96 ]), &(acadoWorkspace.lbA[ 82 ]), &(acadoWorkspace.ubA[ 82 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 208 ]), &(acadoWorkspace.d[ 100 ]), &(acadoWorkspace.lbA[ 84 ]), &(acadoWorkspace.ubA[ 84 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 216 ]), &(acadoWorkspace.d[ 104 ]), &(acadoWorkspace.lbA[ 86 ]), &(acadoWorkspace.ubA[ 86 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 224 ]), &(acadoWorkspace.d[ 108 ]), &(acadoWorkspace.lbA[ 88 ]), &(acadoWorkspace.ubA[ 88 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 232 ]), &(acadoWorkspace.d[ 112 ]), &(acadoWorkspace.lbA[ 90 ]), &(acadoWorkspace.ubA[ 90 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 240 ]), &(acadoWorkspace.d[ 116 ]), &(acadoWorkspace.lbA[ 92 ]), &(acadoWorkspace.ubA[ 92 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 248 ]), &(acadoWorkspace.d[ 120 ]), &(acadoWorkspace.lbA[ 94 ]), &(acadoWorkspace.ubA[ 94 ]) );

}

void acado_condenseFdb(  )
{
int lRun1;
int lRun2;
int lRun3;
real_t tmp;

acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];

for (lRun2 = 0; lRun2 < 160; ++lRun2)
acadoWorkspace.Dy[lRun2] -= acadoVariables.y[lRun2];

acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 10 ]), &(acadoWorkspace.Dy[ 5 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 20 ]), &(acadoWorkspace.Dy[ 10 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 30 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 40 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 50 ]), &(acadoWorkspace.Dy[ 25 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 60 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 70 ]), &(acadoWorkspace.Dy[ 35 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 80 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 90 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 100 ]), &(acadoWorkspace.Dy[ 50 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 110 ]), &(acadoWorkspace.Dy[ 55 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 120 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 130 ]), &(acadoWorkspace.Dy[ 65 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 140 ]), &(acadoWorkspace.Dy[ 70 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 150 ]), &(acadoWorkspace.Dy[ 75 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 160 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 170 ]), &(acadoWorkspace.Dy[ 85 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 180 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.g[ 40 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 190 ]), &(acadoWorkspace.Dy[ 95 ]), &(acadoWorkspace.g[ 42 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 200 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 210 ]), &(acadoWorkspace.Dy[ 105 ]), &(acadoWorkspace.g[ 46 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 220 ]), &(acadoWorkspace.Dy[ 110 ]), &(acadoWorkspace.g[ 48 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 230 ]), &(acadoWorkspace.Dy[ 115 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 240 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.g[ 52 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 250 ]), &(acadoWorkspace.Dy[ 125 ]), &(acadoWorkspace.g[ 54 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 260 ]), &(acadoWorkspace.Dy[ 130 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 270 ]), &(acadoWorkspace.Dy[ 135 ]), &(acadoWorkspace.g[ 58 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 280 ]), &(acadoWorkspace.Dy[ 140 ]), &(acadoWorkspace.g[ 60 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 290 ]), &(acadoWorkspace.Dy[ 145 ]), &(acadoWorkspace.g[ 62 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 300 ]), &(acadoWorkspace.Dy[ 150 ]), &(acadoWorkspace.g[ 64 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 310 ]), &(acadoWorkspace.Dy[ 155 ]), &(acadoWorkspace.g[ 66 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 20 ]), &(acadoWorkspace.Dy[ 5 ]), &(acadoWorkspace.QDy[ 4 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 40 ]), &(acadoWorkspace.Dy[ 10 ]), &(acadoWorkspace.QDy[ 8 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 60 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 80 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.QDy[ 16 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 100 ]), &(acadoWorkspace.Dy[ 25 ]), &(acadoWorkspace.QDy[ 20 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 120 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.QDy[ 24 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 140 ]), &(acadoWorkspace.Dy[ 35 ]), &(acadoWorkspace.QDy[ 28 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 160 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.QDy[ 32 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 180 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.QDy[ 36 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 200 ]), &(acadoWorkspace.Dy[ 50 ]), &(acadoWorkspace.QDy[ 40 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 220 ]), &(acadoWorkspace.Dy[ 55 ]), &(acadoWorkspace.QDy[ 44 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 240 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 48 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 260 ]), &(acadoWorkspace.Dy[ 65 ]), &(acadoWorkspace.QDy[ 52 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 280 ]), &(acadoWorkspace.Dy[ 70 ]), &(acadoWorkspace.QDy[ 56 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 300 ]), &(acadoWorkspace.Dy[ 75 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 320 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.QDy[ 64 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 340 ]), &(acadoWorkspace.Dy[ 85 ]), &(acadoWorkspace.QDy[ 68 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 360 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.QDy[ 72 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 380 ]), &(acadoWorkspace.Dy[ 95 ]), &(acadoWorkspace.QDy[ 76 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 400 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.QDy[ 80 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 420 ]), &(acadoWorkspace.Dy[ 105 ]), &(acadoWorkspace.QDy[ 84 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 440 ]), &(acadoWorkspace.Dy[ 110 ]), &(acadoWorkspace.QDy[ 88 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 460 ]), &(acadoWorkspace.Dy[ 115 ]), &(acadoWorkspace.QDy[ 92 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 480 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.QDy[ 96 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 500 ]), &(acadoWorkspace.Dy[ 125 ]), &(acadoWorkspace.QDy[ 100 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 520 ]), &(acadoWorkspace.Dy[ 130 ]), &(acadoWorkspace.QDy[ 104 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 540 ]), &(acadoWorkspace.Dy[ 135 ]), &(acadoWorkspace.QDy[ 108 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 560 ]), &(acadoWorkspace.Dy[ 140 ]), &(acadoWorkspace.QDy[ 112 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 580 ]), &(acadoWorkspace.Dy[ 145 ]), &(acadoWorkspace.QDy[ 116 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 600 ]), &(acadoWorkspace.Dy[ 150 ]), &(acadoWorkspace.QDy[ 120 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 620 ]), &(acadoWorkspace.Dy[ 155 ]), &(acadoWorkspace.QDy[ 124 ]) );

acadoWorkspace.QDy[128] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[129] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[130] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[131] = + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[2];

for (lRun2 = 0; lRun2 < 128; ++lRun2)
acadoWorkspace.QDy[lRun2 + 4] += acadoWorkspace.Qd[lRun2];


acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[320]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[324]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[328]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[332]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[336]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[340]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[344]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[348]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[352]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[356]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[360]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[364]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[368]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[372]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[376]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[380]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[384]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[388]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[392]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[396]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[400]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[404]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[408]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[412]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[416]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[420]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[424]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[428]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[432]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[436]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[440]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[444]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[448]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[452]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[456]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[460]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[464]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[468]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[472]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[476]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[480]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[484]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[488]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[492]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[496]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[500]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[504]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[508]*acadoWorkspace.QDy[131];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[321]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[325]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[329]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[333]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[337]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[341]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[345]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[349]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[353]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[357]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[361]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[365]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[369]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[373]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[377]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[381]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[385]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[389]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[393]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[397]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[401]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[405]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[409]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[413]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[417]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[421]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[425]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[429]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[433]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[437]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[441]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[445]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[449]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[453]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[457]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[461]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[465]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[469]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[473]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[477]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[481]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[485]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[489]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[493]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[497]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[501]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[505]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[509]*acadoWorkspace.QDy[131];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[322]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[326]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[330]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[334]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[338]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[342]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[346]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[350]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[354]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[358]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[362]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[366]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[370]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[374]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[378]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[382]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[386]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[390]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[394]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[398]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[402]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[406]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[410]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[414]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[418]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[422]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[426]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[430]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[434]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[438]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[442]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[446]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[450]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[454]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[458]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[462]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[466]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[470]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[474]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[478]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[482]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[486]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[490]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[494]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[498]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[502]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[506]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[510]*acadoWorkspace.QDy[131];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[323]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[327]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[331]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[335]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[339]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[343]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[347]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[351]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[355]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[359]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[363]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[367]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[371]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[375]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[379]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[383]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[387]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[391]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[395]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[399]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[403]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[407]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[411]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[415]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[419]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[423]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[427]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[431]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[435]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[439]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[443]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[447]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[451]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[455]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[459]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[463]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[467]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[471]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[475]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[479]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[483]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[487]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[491]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[495]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[499]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[503]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[507]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[511]*acadoWorkspace.QDy[131];


for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multEQDy( &(acadoWorkspace.E[ lRun3 * 8 ]), &(acadoWorkspace.QDy[ lRun2 * 4 + 4 ]), &(acadoWorkspace.g[ lRun1 * 2 + 4 ]) );
}
}

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.lb[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[3] = acadoWorkspace.Dx0[3];
tmp = acadoVariables.x[5] + acadoWorkspace.d[1];
acadoWorkspace.lbA[0] = - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[9] + acadoWorkspace.d[5];
acadoWorkspace.lbA[1] = - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[13] + acadoWorkspace.d[9];
acadoWorkspace.lbA[2] = - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[17] + acadoWorkspace.d[13];
acadoWorkspace.lbA[3] = - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[21] + acadoWorkspace.d[17];
acadoWorkspace.lbA[4] = - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[25] + acadoWorkspace.d[21];
acadoWorkspace.lbA[5] = - tmp;
acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[29] + acadoWorkspace.d[25];
acadoWorkspace.lbA[6] = - tmp;
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[33] + acadoWorkspace.d[29];
acadoWorkspace.lbA[7] = - tmp;
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[37] + acadoWorkspace.d[33];
acadoWorkspace.lbA[8] = - tmp;
acadoWorkspace.ubA[8] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[41] + acadoWorkspace.d[37];
acadoWorkspace.lbA[9] = - tmp;
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[45] + acadoWorkspace.d[41];
acadoWorkspace.lbA[10] = - tmp;
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[49] + acadoWorkspace.d[45];
acadoWorkspace.lbA[11] = - tmp;
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[53] + acadoWorkspace.d[49];
acadoWorkspace.lbA[12] = - tmp;
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[57] + acadoWorkspace.d[53];
acadoWorkspace.lbA[13] = - tmp;
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[61] + acadoWorkspace.d[57];
acadoWorkspace.lbA[14] = - tmp;
acadoWorkspace.ubA[14] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[65] + acadoWorkspace.d[61];
acadoWorkspace.lbA[15] = - tmp;
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[69] + acadoWorkspace.d[65];
acadoWorkspace.lbA[16] = - tmp;
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[73] + acadoWorkspace.d[69];
acadoWorkspace.lbA[17] = - tmp;
acadoWorkspace.ubA[17] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[77] + acadoWorkspace.d[73];
acadoWorkspace.lbA[18] = - tmp;
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[81] + acadoWorkspace.d[77];
acadoWorkspace.lbA[19] = - tmp;
acadoWorkspace.ubA[19] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[85] + acadoWorkspace.d[81];
acadoWorkspace.lbA[20] = - tmp;
acadoWorkspace.ubA[20] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[89] + acadoWorkspace.d[85];
acadoWorkspace.lbA[21] = - tmp;
acadoWorkspace.ubA[21] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[93] + acadoWorkspace.d[89];
acadoWorkspace.lbA[22] = - tmp;
acadoWorkspace.ubA[22] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[97] + acadoWorkspace.d[93];
acadoWorkspace.lbA[23] = - tmp;
acadoWorkspace.ubA[23] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[101] + acadoWorkspace.d[97];
acadoWorkspace.lbA[24] = - tmp;
acadoWorkspace.ubA[24] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[105] + acadoWorkspace.d[101];
acadoWorkspace.lbA[25] = - tmp;
acadoWorkspace.ubA[25] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[109] + acadoWorkspace.d[105];
acadoWorkspace.lbA[26] = - tmp;
acadoWorkspace.ubA[26] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[113] + acadoWorkspace.d[109];
acadoWorkspace.lbA[27] = - tmp;
acadoWorkspace.ubA[27] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[117] + acadoWorkspace.d[113];
acadoWorkspace.lbA[28] = - tmp;
acadoWorkspace.ubA[28] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[121] + acadoWorkspace.d[117];
acadoWorkspace.lbA[29] = - tmp;
acadoWorkspace.ubA[29] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[125] + acadoWorkspace.d[121];
acadoWorkspace.lbA[30] = - tmp;
acadoWorkspace.ubA[30] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[129] + acadoWorkspace.d[125];
acadoWorkspace.lbA[31] = - tmp;
acadoWorkspace.ubA[31] = (real_t)1.0000000000000000e+12 - tmp;

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

acadoVariables.u[0] += acadoWorkspace.x[4];
acadoVariables.u[1] += acadoWorkspace.x[5];
acadoVariables.u[2] += acadoWorkspace.x[6];
acadoVariables.u[3] += acadoWorkspace.x[7];
acadoVariables.u[4] += acadoWorkspace.x[8];
acadoVariables.u[5] += acadoWorkspace.x[9];
acadoVariables.u[6] += acadoWorkspace.x[10];
acadoVariables.u[7] += acadoWorkspace.x[11];
acadoVariables.u[8] += acadoWorkspace.x[12];
acadoVariables.u[9] += acadoWorkspace.x[13];
acadoVariables.u[10] += acadoWorkspace.x[14];
acadoVariables.u[11] += acadoWorkspace.x[15];
acadoVariables.u[12] += acadoWorkspace.x[16];
acadoVariables.u[13] += acadoWorkspace.x[17];
acadoVariables.u[14] += acadoWorkspace.x[18];
acadoVariables.u[15] += acadoWorkspace.x[19];
acadoVariables.u[16] += acadoWorkspace.x[20];
acadoVariables.u[17] += acadoWorkspace.x[21];
acadoVariables.u[18] += acadoWorkspace.x[22];
acadoVariables.u[19] += acadoWorkspace.x[23];
acadoVariables.u[20] += acadoWorkspace.x[24];
acadoVariables.u[21] += acadoWorkspace.x[25];
acadoVariables.u[22] += acadoWorkspace.x[26];
acadoVariables.u[23] += acadoWorkspace.x[27];
acadoVariables.u[24] += acadoWorkspace.x[28];
acadoVariables.u[25] += acadoWorkspace.x[29];
acadoVariables.u[26] += acadoWorkspace.x[30];
acadoVariables.u[27] += acadoWorkspace.x[31];
acadoVariables.u[28] += acadoWorkspace.x[32];
acadoVariables.u[29] += acadoWorkspace.x[33];
acadoVariables.u[30] += acadoWorkspace.x[34];
acadoVariables.u[31] += acadoWorkspace.x[35];
acadoVariables.u[32] += acadoWorkspace.x[36];
acadoVariables.u[33] += acadoWorkspace.x[37];
acadoVariables.u[34] += acadoWorkspace.x[38];
acadoVariables.u[35] += acadoWorkspace.x[39];
acadoVariables.u[36] += acadoWorkspace.x[40];
acadoVariables.u[37] += acadoWorkspace.x[41];
acadoVariables.u[38] += acadoWorkspace.x[42];
acadoVariables.u[39] += acadoWorkspace.x[43];
acadoVariables.u[40] += acadoWorkspace.x[44];
acadoVariables.u[41] += acadoWorkspace.x[45];
acadoVariables.u[42] += acadoWorkspace.x[46];
acadoVariables.u[43] += acadoWorkspace.x[47];
acadoVariables.u[44] += acadoWorkspace.x[48];
acadoVariables.u[45] += acadoWorkspace.x[49];
acadoVariables.u[46] += acadoWorkspace.x[50];
acadoVariables.u[47] += acadoWorkspace.x[51];
acadoVariables.u[48] += acadoWorkspace.x[52];
acadoVariables.u[49] += acadoWorkspace.x[53];
acadoVariables.u[50] += acadoWorkspace.x[54];
acadoVariables.u[51] += acadoWorkspace.x[55];
acadoVariables.u[52] += acadoWorkspace.x[56];
acadoVariables.u[53] += acadoWorkspace.x[57];
acadoVariables.u[54] += acadoWorkspace.x[58];
acadoVariables.u[55] += acadoWorkspace.x[59];
acadoVariables.u[56] += acadoWorkspace.x[60];
acadoVariables.u[57] += acadoWorkspace.x[61];
acadoVariables.u[58] += acadoWorkspace.x[62];
acadoVariables.u[59] += acadoWorkspace.x[63];
acadoVariables.u[60] += acadoWorkspace.x[64];
acadoVariables.u[61] += acadoWorkspace.x[65];
acadoVariables.u[62] += acadoWorkspace.x[66];
acadoVariables.u[63] += acadoWorkspace.x[67];

acadoVariables.x[4] += + acadoWorkspace.evGx[0]*acadoWorkspace.x[0] + acadoWorkspace.evGx[1]*acadoWorkspace.x[1] + acadoWorkspace.evGx[2]*acadoWorkspace.x[2] + acadoWorkspace.evGx[3]*acadoWorkspace.x[3] + acadoWorkspace.d[0];
acadoVariables.x[5] += + acadoWorkspace.evGx[4]*acadoWorkspace.x[0] + acadoWorkspace.evGx[5]*acadoWorkspace.x[1] + acadoWorkspace.evGx[6]*acadoWorkspace.x[2] + acadoWorkspace.evGx[7]*acadoWorkspace.x[3] + acadoWorkspace.d[1];
acadoVariables.x[6] += + acadoWorkspace.evGx[8]*acadoWorkspace.x[0] + acadoWorkspace.evGx[9]*acadoWorkspace.x[1] + acadoWorkspace.evGx[10]*acadoWorkspace.x[2] + acadoWorkspace.evGx[11]*acadoWorkspace.x[3] + acadoWorkspace.d[2];
acadoVariables.x[7] += + acadoWorkspace.evGx[12]*acadoWorkspace.x[0] + acadoWorkspace.evGx[13]*acadoWorkspace.x[1] + acadoWorkspace.evGx[14]*acadoWorkspace.x[2] + acadoWorkspace.evGx[15]*acadoWorkspace.x[3] + acadoWorkspace.d[3];
acadoVariables.x[8] += + acadoWorkspace.evGx[16]*acadoWorkspace.x[0] + acadoWorkspace.evGx[17]*acadoWorkspace.x[1] + acadoWorkspace.evGx[18]*acadoWorkspace.x[2] + acadoWorkspace.evGx[19]*acadoWorkspace.x[3] + acadoWorkspace.d[4];
acadoVariables.x[9] += + acadoWorkspace.evGx[20]*acadoWorkspace.x[0] + acadoWorkspace.evGx[21]*acadoWorkspace.x[1] + acadoWorkspace.evGx[22]*acadoWorkspace.x[2] + acadoWorkspace.evGx[23]*acadoWorkspace.x[3] + acadoWorkspace.d[5];
acadoVariables.x[10] += + acadoWorkspace.evGx[24]*acadoWorkspace.x[0] + acadoWorkspace.evGx[25]*acadoWorkspace.x[1] + acadoWorkspace.evGx[26]*acadoWorkspace.x[2] + acadoWorkspace.evGx[27]*acadoWorkspace.x[3] + acadoWorkspace.d[6];
acadoVariables.x[11] += + acadoWorkspace.evGx[28]*acadoWorkspace.x[0] + acadoWorkspace.evGx[29]*acadoWorkspace.x[1] + acadoWorkspace.evGx[30]*acadoWorkspace.x[2] + acadoWorkspace.evGx[31]*acadoWorkspace.x[3] + acadoWorkspace.d[7];
acadoVariables.x[12] += + acadoWorkspace.evGx[32]*acadoWorkspace.x[0] + acadoWorkspace.evGx[33]*acadoWorkspace.x[1] + acadoWorkspace.evGx[34]*acadoWorkspace.x[2] + acadoWorkspace.evGx[35]*acadoWorkspace.x[3] + acadoWorkspace.d[8];
acadoVariables.x[13] += + acadoWorkspace.evGx[36]*acadoWorkspace.x[0] + acadoWorkspace.evGx[37]*acadoWorkspace.x[1] + acadoWorkspace.evGx[38]*acadoWorkspace.x[2] + acadoWorkspace.evGx[39]*acadoWorkspace.x[3] + acadoWorkspace.d[9];
acadoVariables.x[14] += + acadoWorkspace.evGx[40]*acadoWorkspace.x[0] + acadoWorkspace.evGx[41]*acadoWorkspace.x[1] + acadoWorkspace.evGx[42]*acadoWorkspace.x[2] + acadoWorkspace.evGx[43]*acadoWorkspace.x[3] + acadoWorkspace.d[10];
acadoVariables.x[15] += + acadoWorkspace.evGx[44]*acadoWorkspace.x[0] + acadoWorkspace.evGx[45]*acadoWorkspace.x[1] + acadoWorkspace.evGx[46]*acadoWorkspace.x[2] + acadoWorkspace.evGx[47]*acadoWorkspace.x[3] + acadoWorkspace.d[11];
acadoVariables.x[16] += + acadoWorkspace.evGx[48]*acadoWorkspace.x[0] + acadoWorkspace.evGx[49]*acadoWorkspace.x[1] + acadoWorkspace.evGx[50]*acadoWorkspace.x[2] + acadoWorkspace.evGx[51]*acadoWorkspace.x[3] + acadoWorkspace.d[12];
acadoVariables.x[17] += + acadoWorkspace.evGx[52]*acadoWorkspace.x[0] + acadoWorkspace.evGx[53]*acadoWorkspace.x[1] + acadoWorkspace.evGx[54]*acadoWorkspace.x[2] + acadoWorkspace.evGx[55]*acadoWorkspace.x[3] + acadoWorkspace.d[13];
acadoVariables.x[18] += + acadoWorkspace.evGx[56]*acadoWorkspace.x[0] + acadoWorkspace.evGx[57]*acadoWorkspace.x[1] + acadoWorkspace.evGx[58]*acadoWorkspace.x[2] + acadoWorkspace.evGx[59]*acadoWorkspace.x[3] + acadoWorkspace.d[14];
acadoVariables.x[19] += + acadoWorkspace.evGx[60]*acadoWorkspace.x[0] + acadoWorkspace.evGx[61]*acadoWorkspace.x[1] + acadoWorkspace.evGx[62]*acadoWorkspace.x[2] + acadoWorkspace.evGx[63]*acadoWorkspace.x[3] + acadoWorkspace.d[15];
acadoVariables.x[20] += + acadoWorkspace.evGx[64]*acadoWorkspace.x[0] + acadoWorkspace.evGx[65]*acadoWorkspace.x[1] + acadoWorkspace.evGx[66]*acadoWorkspace.x[2] + acadoWorkspace.evGx[67]*acadoWorkspace.x[3] + acadoWorkspace.d[16];
acadoVariables.x[21] += + acadoWorkspace.evGx[68]*acadoWorkspace.x[0] + acadoWorkspace.evGx[69]*acadoWorkspace.x[1] + acadoWorkspace.evGx[70]*acadoWorkspace.x[2] + acadoWorkspace.evGx[71]*acadoWorkspace.x[3] + acadoWorkspace.d[17];
acadoVariables.x[22] += + acadoWorkspace.evGx[72]*acadoWorkspace.x[0] + acadoWorkspace.evGx[73]*acadoWorkspace.x[1] + acadoWorkspace.evGx[74]*acadoWorkspace.x[2] + acadoWorkspace.evGx[75]*acadoWorkspace.x[3] + acadoWorkspace.d[18];
acadoVariables.x[23] += + acadoWorkspace.evGx[76]*acadoWorkspace.x[0] + acadoWorkspace.evGx[77]*acadoWorkspace.x[1] + acadoWorkspace.evGx[78]*acadoWorkspace.x[2] + acadoWorkspace.evGx[79]*acadoWorkspace.x[3] + acadoWorkspace.d[19];
acadoVariables.x[24] += + acadoWorkspace.evGx[80]*acadoWorkspace.x[0] + acadoWorkspace.evGx[81]*acadoWorkspace.x[1] + acadoWorkspace.evGx[82]*acadoWorkspace.x[2] + acadoWorkspace.evGx[83]*acadoWorkspace.x[3] + acadoWorkspace.d[20];
acadoVariables.x[25] += + acadoWorkspace.evGx[84]*acadoWorkspace.x[0] + acadoWorkspace.evGx[85]*acadoWorkspace.x[1] + acadoWorkspace.evGx[86]*acadoWorkspace.x[2] + acadoWorkspace.evGx[87]*acadoWorkspace.x[3] + acadoWorkspace.d[21];
acadoVariables.x[26] += + acadoWorkspace.evGx[88]*acadoWorkspace.x[0] + acadoWorkspace.evGx[89]*acadoWorkspace.x[1] + acadoWorkspace.evGx[90]*acadoWorkspace.x[2] + acadoWorkspace.evGx[91]*acadoWorkspace.x[3] + acadoWorkspace.d[22];
acadoVariables.x[27] += + acadoWorkspace.evGx[92]*acadoWorkspace.x[0] + acadoWorkspace.evGx[93]*acadoWorkspace.x[1] + acadoWorkspace.evGx[94]*acadoWorkspace.x[2] + acadoWorkspace.evGx[95]*acadoWorkspace.x[3] + acadoWorkspace.d[23];
acadoVariables.x[28] += + acadoWorkspace.evGx[96]*acadoWorkspace.x[0] + acadoWorkspace.evGx[97]*acadoWorkspace.x[1] + acadoWorkspace.evGx[98]*acadoWorkspace.x[2] + acadoWorkspace.evGx[99]*acadoWorkspace.x[3] + acadoWorkspace.d[24];
acadoVariables.x[29] += + acadoWorkspace.evGx[100]*acadoWorkspace.x[0] + acadoWorkspace.evGx[101]*acadoWorkspace.x[1] + acadoWorkspace.evGx[102]*acadoWorkspace.x[2] + acadoWorkspace.evGx[103]*acadoWorkspace.x[3] + acadoWorkspace.d[25];
acadoVariables.x[30] += + acadoWorkspace.evGx[104]*acadoWorkspace.x[0] + acadoWorkspace.evGx[105]*acadoWorkspace.x[1] + acadoWorkspace.evGx[106]*acadoWorkspace.x[2] + acadoWorkspace.evGx[107]*acadoWorkspace.x[3] + acadoWorkspace.d[26];
acadoVariables.x[31] += + acadoWorkspace.evGx[108]*acadoWorkspace.x[0] + acadoWorkspace.evGx[109]*acadoWorkspace.x[1] + acadoWorkspace.evGx[110]*acadoWorkspace.x[2] + acadoWorkspace.evGx[111]*acadoWorkspace.x[3] + acadoWorkspace.d[27];
acadoVariables.x[32] += + acadoWorkspace.evGx[112]*acadoWorkspace.x[0] + acadoWorkspace.evGx[113]*acadoWorkspace.x[1] + acadoWorkspace.evGx[114]*acadoWorkspace.x[2] + acadoWorkspace.evGx[115]*acadoWorkspace.x[3] + acadoWorkspace.d[28];
acadoVariables.x[33] += + acadoWorkspace.evGx[116]*acadoWorkspace.x[0] + acadoWorkspace.evGx[117]*acadoWorkspace.x[1] + acadoWorkspace.evGx[118]*acadoWorkspace.x[2] + acadoWorkspace.evGx[119]*acadoWorkspace.x[3] + acadoWorkspace.d[29];
acadoVariables.x[34] += + acadoWorkspace.evGx[120]*acadoWorkspace.x[0] + acadoWorkspace.evGx[121]*acadoWorkspace.x[1] + acadoWorkspace.evGx[122]*acadoWorkspace.x[2] + acadoWorkspace.evGx[123]*acadoWorkspace.x[3] + acadoWorkspace.d[30];
acadoVariables.x[35] += + acadoWorkspace.evGx[124]*acadoWorkspace.x[0] + acadoWorkspace.evGx[125]*acadoWorkspace.x[1] + acadoWorkspace.evGx[126]*acadoWorkspace.x[2] + acadoWorkspace.evGx[127]*acadoWorkspace.x[3] + acadoWorkspace.d[31];
acadoVariables.x[36] += + acadoWorkspace.evGx[128]*acadoWorkspace.x[0] + acadoWorkspace.evGx[129]*acadoWorkspace.x[1] + acadoWorkspace.evGx[130]*acadoWorkspace.x[2] + acadoWorkspace.evGx[131]*acadoWorkspace.x[3] + acadoWorkspace.d[32];
acadoVariables.x[37] += + acadoWorkspace.evGx[132]*acadoWorkspace.x[0] + acadoWorkspace.evGx[133]*acadoWorkspace.x[1] + acadoWorkspace.evGx[134]*acadoWorkspace.x[2] + acadoWorkspace.evGx[135]*acadoWorkspace.x[3] + acadoWorkspace.d[33];
acadoVariables.x[38] += + acadoWorkspace.evGx[136]*acadoWorkspace.x[0] + acadoWorkspace.evGx[137]*acadoWorkspace.x[1] + acadoWorkspace.evGx[138]*acadoWorkspace.x[2] + acadoWorkspace.evGx[139]*acadoWorkspace.x[3] + acadoWorkspace.d[34];
acadoVariables.x[39] += + acadoWorkspace.evGx[140]*acadoWorkspace.x[0] + acadoWorkspace.evGx[141]*acadoWorkspace.x[1] + acadoWorkspace.evGx[142]*acadoWorkspace.x[2] + acadoWorkspace.evGx[143]*acadoWorkspace.x[3] + acadoWorkspace.d[35];
acadoVariables.x[40] += + acadoWorkspace.evGx[144]*acadoWorkspace.x[0] + acadoWorkspace.evGx[145]*acadoWorkspace.x[1] + acadoWorkspace.evGx[146]*acadoWorkspace.x[2] + acadoWorkspace.evGx[147]*acadoWorkspace.x[3] + acadoWorkspace.d[36];
acadoVariables.x[41] += + acadoWorkspace.evGx[148]*acadoWorkspace.x[0] + acadoWorkspace.evGx[149]*acadoWorkspace.x[1] + acadoWorkspace.evGx[150]*acadoWorkspace.x[2] + acadoWorkspace.evGx[151]*acadoWorkspace.x[3] + acadoWorkspace.d[37];
acadoVariables.x[42] += + acadoWorkspace.evGx[152]*acadoWorkspace.x[0] + acadoWorkspace.evGx[153]*acadoWorkspace.x[1] + acadoWorkspace.evGx[154]*acadoWorkspace.x[2] + acadoWorkspace.evGx[155]*acadoWorkspace.x[3] + acadoWorkspace.d[38];
acadoVariables.x[43] += + acadoWorkspace.evGx[156]*acadoWorkspace.x[0] + acadoWorkspace.evGx[157]*acadoWorkspace.x[1] + acadoWorkspace.evGx[158]*acadoWorkspace.x[2] + acadoWorkspace.evGx[159]*acadoWorkspace.x[3] + acadoWorkspace.d[39];
acadoVariables.x[44] += + acadoWorkspace.evGx[160]*acadoWorkspace.x[0] + acadoWorkspace.evGx[161]*acadoWorkspace.x[1] + acadoWorkspace.evGx[162]*acadoWorkspace.x[2] + acadoWorkspace.evGx[163]*acadoWorkspace.x[3] + acadoWorkspace.d[40];
acadoVariables.x[45] += + acadoWorkspace.evGx[164]*acadoWorkspace.x[0] + acadoWorkspace.evGx[165]*acadoWorkspace.x[1] + acadoWorkspace.evGx[166]*acadoWorkspace.x[2] + acadoWorkspace.evGx[167]*acadoWorkspace.x[3] + acadoWorkspace.d[41];
acadoVariables.x[46] += + acadoWorkspace.evGx[168]*acadoWorkspace.x[0] + acadoWorkspace.evGx[169]*acadoWorkspace.x[1] + acadoWorkspace.evGx[170]*acadoWorkspace.x[2] + acadoWorkspace.evGx[171]*acadoWorkspace.x[3] + acadoWorkspace.d[42];
acadoVariables.x[47] += + acadoWorkspace.evGx[172]*acadoWorkspace.x[0] + acadoWorkspace.evGx[173]*acadoWorkspace.x[1] + acadoWorkspace.evGx[174]*acadoWorkspace.x[2] + acadoWorkspace.evGx[175]*acadoWorkspace.x[3] + acadoWorkspace.d[43];
acadoVariables.x[48] += + acadoWorkspace.evGx[176]*acadoWorkspace.x[0] + acadoWorkspace.evGx[177]*acadoWorkspace.x[1] + acadoWorkspace.evGx[178]*acadoWorkspace.x[2] + acadoWorkspace.evGx[179]*acadoWorkspace.x[3] + acadoWorkspace.d[44];
acadoVariables.x[49] += + acadoWorkspace.evGx[180]*acadoWorkspace.x[0] + acadoWorkspace.evGx[181]*acadoWorkspace.x[1] + acadoWorkspace.evGx[182]*acadoWorkspace.x[2] + acadoWorkspace.evGx[183]*acadoWorkspace.x[3] + acadoWorkspace.d[45];
acadoVariables.x[50] += + acadoWorkspace.evGx[184]*acadoWorkspace.x[0] + acadoWorkspace.evGx[185]*acadoWorkspace.x[1] + acadoWorkspace.evGx[186]*acadoWorkspace.x[2] + acadoWorkspace.evGx[187]*acadoWorkspace.x[3] + acadoWorkspace.d[46];
acadoVariables.x[51] += + acadoWorkspace.evGx[188]*acadoWorkspace.x[0] + acadoWorkspace.evGx[189]*acadoWorkspace.x[1] + acadoWorkspace.evGx[190]*acadoWorkspace.x[2] + acadoWorkspace.evGx[191]*acadoWorkspace.x[3] + acadoWorkspace.d[47];
acadoVariables.x[52] += + acadoWorkspace.evGx[192]*acadoWorkspace.x[0] + acadoWorkspace.evGx[193]*acadoWorkspace.x[1] + acadoWorkspace.evGx[194]*acadoWorkspace.x[2] + acadoWorkspace.evGx[195]*acadoWorkspace.x[3] + acadoWorkspace.d[48];
acadoVariables.x[53] += + acadoWorkspace.evGx[196]*acadoWorkspace.x[0] + acadoWorkspace.evGx[197]*acadoWorkspace.x[1] + acadoWorkspace.evGx[198]*acadoWorkspace.x[2] + acadoWorkspace.evGx[199]*acadoWorkspace.x[3] + acadoWorkspace.d[49];
acadoVariables.x[54] += + acadoWorkspace.evGx[200]*acadoWorkspace.x[0] + acadoWorkspace.evGx[201]*acadoWorkspace.x[1] + acadoWorkspace.evGx[202]*acadoWorkspace.x[2] + acadoWorkspace.evGx[203]*acadoWorkspace.x[3] + acadoWorkspace.d[50];
acadoVariables.x[55] += + acadoWorkspace.evGx[204]*acadoWorkspace.x[0] + acadoWorkspace.evGx[205]*acadoWorkspace.x[1] + acadoWorkspace.evGx[206]*acadoWorkspace.x[2] + acadoWorkspace.evGx[207]*acadoWorkspace.x[3] + acadoWorkspace.d[51];
acadoVariables.x[56] += + acadoWorkspace.evGx[208]*acadoWorkspace.x[0] + acadoWorkspace.evGx[209]*acadoWorkspace.x[1] + acadoWorkspace.evGx[210]*acadoWorkspace.x[2] + acadoWorkspace.evGx[211]*acadoWorkspace.x[3] + acadoWorkspace.d[52];
acadoVariables.x[57] += + acadoWorkspace.evGx[212]*acadoWorkspace.x[0] + acadoWorkspace.evGx[213]*acadoWorkspace.x[1] + acadoWorkspace.evGx[214]*acadoWorkspace.x[2] + acadoWorkspace.evGx[215]*acadoWorkspace.x[3] + acadoWorkspace.d[53];
acadoVariables.x[58] += + acadoWorkspace.evGx[216]*acadoWorkspace.x[0] + acadoWorkspace.evGx[217]*acadoWorkspace.x[1] + acadoWorkspace.evGx[218]*acadoWorkspace.x[2] + acadoWorkspace.evGx[219]*acadoWorkspace.x[3] + acadoWorkspace.d[54];
acadoVariables.x[59] += + acadoWorkspace.evGx[220]*acadoWorkspace.x[0] + acadoWorkspace.evGx[221]*acadoWorkspace.x[1] + acadoWorkspace.evGx[222]*acadoWorkspace.x[2] + acadoWorkspace.evGx[223]*acadoWorkspace.x[3] + acadoWorkspace.d[55];
acadoVariables.x[60] += + acadoWorkspace.evGx[224]*acadoWorkspace.x[0] + acadoWorkspace.evGx[225]*acadoWorkspace.x[1] + acadoWorkspace.evGx[226]*acadoWorkspace.x[2] + acadoWorkspace.evGx[227]*acadoWorkspace.x[3] + acadoWorkspace.d[56];
acadoVariables.x[61] += + acadoWorkspace.evGx[228]*acadoWorkspace.x[0] + acadoWorkspace.evGx[229]*acadoWorkspace.x[1] + acadoWorkspace.evGx[230]*acadoWorkspace.x[2] + acadoWorkspace.evGx[231]*acadoWorkspace.x[3] + acadoWorkspace.d[57];
acadoVariables.x[62] += + acadoWorkspace.evGx[232]*acadoWorkspace.x[0] + acadoWorkspace.evGx[233]*acadoWorkspace.x[1] + acadoWorkspace.evGx[234]*acadoWorkspace.x[2] + acadoWorkspace.evGx[235]*acadoWorkspace.x[3] + acadoWorkspace.d[58];
acadoVariables.x[63] += + acadoWorkspace.evGx[236]*acadoWorkspace.x[0] + acadoWorkspace.evGx[237]*acadoWorkspace.x[1] + acadoWorkspace.evGx[238]*acadoWorkspace.x[2] + acadoWorkspace.evGx[239]*acadoWorkspace.x[3] + acadoWorkspace.d[59];
acadoVariables.x[64] += + acadoWorkspace.evGx[240]*acadoWorkspace.x[0] + acadoWorkspace.evGx[241]*acadoWorkspace.x[1] + acadoWorkspace.evGx[242]*acadoWorkspace.x[2] + acadoWorkspace.evGx[243]*acadoWorkspace.x[3] + acadoWorkspace.d[60];
acadoVariables.x[65] += + acadoWorkspace.evGx[244]*acadoWorkspace.x[0] + acadoWorkspace.evGx[245]*acadoWorkspace.x[1] + acadoWorkspace.evGx[246]*acadoWorkspace.x[2] + acadoWorkspace.evGx[247]*acadoWorkspace.x[3] + acadoWorkspace.d[61];
acadoVariables.x[66] += + acadoWorkspace.evGx[248]*acadoWorkspace.x[0] + acadoWorkspace.evGx[249]*acadoWorkspace.x[1] + acadoWorkspace.evGx[250]*acadoWorkspace.x[2] + acadoWorkspace.evGx[251]*acadoWorkspace.x[3] + acadoWorkspace.d[62];
acadoVariables.x[67] += + acadoWorkspace.evGx[252]*acadoWorkspace.x[0] + acadoWorkspace.evGx[253]*acadoWorkspace.x[1] + acadoWorkspace.evGx[254]*acadoWorkspace.x[2] + acadoWorkspace.evGx[255]*acadoWorkspace.x[3] + acadoWorkspace.d[63];
acadoVariables.x[68] += + acadoWorkspace.evGx[256]*acadoWorkspace.x[0] + acadoWorkspace.evGx[257]*acadoWorkspace.x[1] + acadoWorkspace.evGx[258]*acadoWorkspace.x[2] + acadoWorkspace.evGx[259]*acadoWorkspace.x[3] + acadoWorkspace.d[64];
acadoVariables.x[69] += + acadoWorkspace.evGx[260]*acadoWorkspace.x[0] + acadoWorkspace.evGx[261]*acadoWorkspace.x[1] + acadoWorkspace.evGx[262]*acadoWorkspace.x[2] + acadoWorkspace.evGx[263]*acadoWorkspace.x[3] + acadoWorkspace.d[65];
acadoVariables.x[70] += + acadoWorkspace.evGx[264]*acadoWorkspace.x[0] + acadoWorkspace.evGx[265]*acadoWorkspace.x[1] + acadoWorkspace.evGx[266]*acadoWorkspace.x[2] + acadoWorkspace.evGx[267]*acadoWorkspace.x[3] + acadoWorkspace.d[66];
acadoVariables.x[71] += + acadoWorkspace.evGx[268]*acadoWorkspace.x[0] + acadoWorkspace.evGx[269]*acadoWorkspace.x[1] + acadoWorkspace.evGx[270]*acadoWorkspace.x[2] + acadoWorkspace.evGx[271]*acadoWorkspace.x[3] + acadoWorkspace.d[67];
acadoVariables.x[72] += + acadoWorkspace.evGx[272]*acadoWorkspace.x[0] + acadoWorkspace.evGx[273]*acadoWorkspace.x[1] + acadoWorkspace.evGx[274]*acadoWorkspace.x[2] + acadoWorkspace.evGx[275]*acadoWorkspace.x[3] + acadoWorkspace.d[68];
acadoVariables.x[73] += + acadoWorkspace.evGx[276]*acadoWorkspace.x[0] + acadoWorkspace.evGx[277]*acadoWorkspace.x[1] + acadoWorkspace.evGx[278]*acadoWorkspace.x[2] + acadoWorkspace.evGx[279]*acadoWorkspace.x[3] + acadoWorkspace.d[69];
acadoVariables.x[74] += + acadoWorkspace.evGx[280]*acadoWorkspace.x[0] + acadoWorkspace.evGx[281]*acadoWorkspace.x[1] + acadoWorkspace.evGx[282]*acadoWorkspace.x[2] + acadoWorkspace.evGx[283]*acadoWorkspace.x[3] + acadoWorkspace.d[70];
acadoVariables.x[75] += + acadoWorkspace.evGx[284]*acadoWorkspace.x[0] + acadoWorkspace.evGx[285]*acadoWorkspace.x[1] + acadoWorkspace.evGx[286]*acadoWorkspace.x[2] + acadoWorkspace.evGx[287]*acadoWorkspace.x[3] + acadoWorkspace.d[71];
acadoVariables.x[76] += + acadoWorkspace.evGx[288]*acadoWorkspace.x[0] + acadoWorkspace.evGx[289]*acadoWorkspace.x[1] + acadoWorkspace.evGx[290]*acadoWorkspace.x[2] + acadoWorkspace.evGx[291]*acadoWorkspace.x[3] + acadoWorkspace.d[72];
acadoVariables.x[77] += + acadoWorkspace.evGx[292]*acadoWorkspace.x[0] + acadoWorkspace.evGx[293]*acadoWorkspace.x[1] + acadoWorkspace.evGx[294]*acadoWorkspace.x[2] + acadoWorkspace.evGx[295]*acadoWorkspace.x[3] + acadoWorkspace.d[73];
acadoVariables.x[78] += + acadoWorkspace.evGx[296]*acadoWorkspace.x[0] + acadoWorkspace.evGx[297]*acadoWorkspace.x[1] + acadoWorkspace.evGx[298]*acadoWorkspace.x[2] + acadoWorkspace.evGx[299]*acadoWorkspace.x[3] + acadoWorkspace.d[74];
acadoVariables.x[79] += + acadoWorkspace.evGx[300]*acadoWorkspace.x[0] + acadoWorkspace.evGx[301]*acadoWorkspace.x[1] + acadoWorkspace.evGx[302]*acadoWorkspace.x[2] + acadoWorkspace.evGx[303]*acadoWorkspace.x[3] + acadoWorkspace.d[75];
acadoVariables.x[80] += + acadoWorkspace.evGx[304]*acadoWorkspace.x[0] + acadoWorkspace.evGx[305]*acadoWorkspace.x[1] + acadoWorkspace.evGx[306]*acadoWorkspace.x[2] + acadoWorkspace.evGx[307]*acadoWorkspace.x[3] + acadoWorkspace.d[76];
acadoVariables.x[81] += + acadoWorkspace.evGx[308]*acadoWorkspace.x[0] + acadoWorkspace.evGx[309]*acadoWorkspace.x[1] + acadoWorkspace.evGx[310]*acadoWorkspace.x[2] + acadoWorkspace.evGx[311]*acadoWorkspace.x[3] + acadoWorkspace.d[77];
acadoVariables.x[82] += + acadoWorkspace.evGx[312]*acadoWorkspace.x[0] + acadoWorkspace.evGx[313]*acadoWorkspace.x[1] + acadoWorkspace.evGx[314]*acadoWorkspace.x[2] + acadoWorkspace.evGx[315]*acadoWorkspace.x[3] + acadoWorkspace.d[78];
acadoVariables.x[83] += + acadoWorkspace.evGx[316]*acadoWorkspace.x[0] + acadoWorkspace.evGx[317]*acadoWorkspace.x[1] + acadoWorkspace.evGx[318]*acadoWorkspace.x[2] + acadoWorkspace.evGx[319]*acadoWorkspace.x[3] + acadoWorkspace.d[79];
acadoVariables.x[84] += + acadoWorkspace.evGx[320]*acadoWorkspace.x[0] + acadoWorkspace.evGx[321]*acadoWorkspace.x[1] + acadoWorkspace.evGx[322]*acadoWorkspace.x[2] + acadoWorkspace.evGx[323]*acadoWorkspace.x[3] + acadoWorkspace.d[80];
acadoVariables.x[85] += + acadoWorkspace.evGx[324]*acadoWorkspace.x[0] + acadoWorkspace.evGx[325]*acadoWorkspace.x[1] + acadoWorkspace.evGx[326]*acadoWorkspace.x[2] + acadoWorkspace.evGx[327]*acadoWorkspace.x[3] + acadoWorkspace.d[81];
acadoVariables.x[86] += + acadoWorkspace.evGx[328]*acadoWorkspace.x[0] + acadoWorkspace.evGx[329]*acadoWorkspace.x[1] + acadoWorkspace.evGx[330]*acadoWorkspace.x[2] + acadoWorkspace.evGx[331]*acadoWorkspace.x[3] + acadoWorkspace.d[82];
acadoVariables.x[87] += + acadoWorkspace.evGx[332]*acadoWorkspace.x[0] + acadoWorkspace.evGx[333]*acadoWorkspace.x[1] + acadoWorkspace.evGx[334]*acadoWorkspace.x[2] + acadoWorkspace.evGx[335]*acadoWorkspace.x[3] + acadoWorkspace.d[83];
acadoVariables.x[88] += + acadoWorkspace.evGx[336]*acadoWorkspace.x[0] + acadoWorkspace.evGx[337]*acadoWorkspace.x[1] + acadoWorkspace.evGx[338]*acadoWorkspace.x[2] + acadoWorkspace.evGx[339]*acadoWorkspace.x[3] + acadoWorkspace.d[84];
acadoVariables.x[89] += + acadoWorkspace.evGx[340]*acadoWorkspace.x[0] + acadoWorkspace.evGx[341]*acadoWorkspace.x[1] + acadoWorkspace.evGx[342]*acadoWorkspace.x[2] + acadoWorkspace.evGx[343]*acadoWorkspace.x[3] + acadoWorkspace.d[85];
acadoVariables.x[90] += + acadoWorkspace.evGx[344]*acadoWorkspace.x[0] + acadoWorkspace.evGx[345]*acadoWorkspace.x[1] + acadoWorkspace.evGx[346]*acadoWorkspace.x[2] + acadoWorkspace.evGx[347]*acadoWorkspace.x[3] + acadoWorkspace.d[86];
acadoVariables.x[91] += + acadoWorkspace.evGx[348]*acadoWorkspace.x[0] + acadoWorkspace.evGx[349]*acadoWorkspace.x[1] + acadoWorkspace.evGx[350]*acadoWorkspace.x[2] + acadoWorkspace.evGx[351]*acadoWorkspace.x[3] + acadoWorkspace.d[87];
acadoVariables.x[92] += + acadoWorkspace.evGx[352]*acadoWorkspace.x[0] + acadoWorkspace.evGx[353]*acadoWorkspace.x[1] + acadoWorkspace.evGx[354]*acadoWorkspace.x[2] + acadoWorkspace.evGx[355]*acadoWorkspace.x[3] + acadoWorkspace.d[88];
acadoVariables.x[93] += + acadoWorkspace.evGx[356]*acadoWorkspace.x[0] + acadoWorkspace.evGx[357]*acadoWorkspace.x[1] + acadoWorkspace.evGx[358]*acadoWorkspace.x[2] + acadoWorkspace.evGx[359]*acadoWorkspace.x[3] + acadoWorkspace.d[89];
acadoVariables.x[94] += + acadoWorkspace.evGx[360]*acadoWorkspace.x[0] + acadoWorkspace.evGx[361]*acadoWorkspace.x[1] + acadoWorkspace.evGx[362]*acadoWorkspace.x[2] + acadoWorkspace.evGx[363]*acadoWorkspace.x[3] + acadoWorkspace.d[90];
acadoVariables.x[95] += + acadoWorkspace.evGx[364]*acadoWorkspace.x[0] + acadoWorkspace.evGx[365]*acadoWorkspace.x[1] + acadoWorkspace.evGx[366]*acadoWorkspace.x[2] + acadoWorkspace.evGx[367]*acadoWorkspace.x[3] + acadoWorkspace.d[91];
acadoVariables.x[96] += + acadoWorkspace.evGx[368]*acadoWorkspace.x[0] + acadoWorkspace.evGx[369]*acadoWorkspace.x[1] + acadoWorkspace.evGx[370]*acadoWorkspace.x[2] + acadoWorkspace.evGx[371]*acadoWorkspace.x[3] + acadoWorkspace.d[92];
acadoVariables.x[97] += + acadoWorkspace.evGx[372]*acadoWorkspace.x[0] + acadoWorkspace.evGx[373]*acadoWorkspace.x[1] + acadoWorkspace.evGx[374]*acadoWorkspace.x[2] + acadoWorkspace.evGx[375]*acadoWorkspace.x[3] + acadoWorkspace.d[93];
acadoVariables.x[98] += + acadoWorkspace.evGx[376]*acadoWorkspace.x[0] + acadoWorkspace.evGx[377]*acadoWorkspace.x[1] + acadoWorkspace.evGx[378]*acadoWorkspace.x[2] + acadoWorkspace.evGx[379]*acadoWorkspace.x[3] + acadoWorkspace.d[94];
acadoVariables.x[99] += + acadoWorkspace.evGx[380]*acadoWorkspace.x[0] + acadoWorkspace.evGx[381]*acadoWorkspace.x[1] + acadoWorkspace.evGx[382]*acadoWorkspace.x[2] + acadoWorkspace.evGx[383]*acadoWorkspace.x[3] + acadoWorkspace.d[95];
acadoVariables.x[100] += + acadoWorkspace.evGx[384]*acadoWorkspace.x[0] + acadoWorkspace.evGx[385]*acadoWorkspace.x[1] + acadoWorkspace.evGx[386]*acadoWorkspace.x[2] + acadoWorkspace.evGx[387]*acadoWorkspace.x[3] + acadoWorkspace.d[96];
acadoVariables.x[101] += + acadoWorkspace.evGx[388]*acadoWorkspace.x[0] + acadoWorkspace.evGx[389]*acadoWorkspace.x[1] + acadoWorkspace.evGx[390]*acadoWorkspace.x[2] + acadoWorkspace.evGx[391]*acadoWorkspace.x[3] + acadoWorkspace.d[97];
acadoVariables.x[102] += + acadoWorkspace.evGx[392]*acadoWorkspace.x[0] + acadoWorkspace.evGx[393]*acadoWorkspace.x[1] + acadoWorkspace.evGx[394]*acadoWorkspace.x[2] + acadoWorkspace.evGx[395]*acadoWorkspace.x[3] + acadoWorkspace.d[98];
acadoVariables.x[103] += + acadoWorkspace.evGx[396]*acadoWorkspace.x[0] + acadoWorkspace.evGx[397]*acadoWorkspace.x[1] + acadoWorkspace.evGx[398]*acadoWorkspace.x[2] + acadoWorkspace.evGx[399]*acadoWorkspace.x[3] + acadoWorkspace.d[99];
acadoVariables.x[104] += + acadoWorkspace.evGx[400]*acadoWorkspace.x[0] + acadoWorkspace.evGx[401]*acadoWorkspace.x[1] + acadoWorkspace.evGx[402]*acadoWorkspace.x[2] + acadoWorkspace.evGx[403]*acadoWorkspace.x[3] + acadoWorkspace.d[100];
acadoVariables.x[105] += + acadoWorkspace.evGx[404]*acadoWorkspace.x[0] + acadoWorkspace.evGx[405]*acadoWorkspace.x[1] + acadoWorkspace.evGx[406]*acadoWorkspace.x[2] + acadoWorkspace.evGx[407]*acadoWorkspace.x[3] + acadoWorkspace.d[101];
acadoVariables.x[106] += + acadoWorkspace.evGx[408]*acadoWorkspace.x[0] + acadoWorkspace.evGx[409]*acadoWorkspace.x[1] + acadoWorkspace.evGx[410]*acadoWorkspace.x[2] + acadoWorkspace.evGx[411]*acadoWorkspace.x[3] + acadoWorkspace.d[102];
acadoVariables.x[107] += + acadoWorkspace.evGx[412]*acadoWorkspace.x[0] + acadoWorkspace.evGx[413]*acadoWorkspace.x[1] + acadoWorkspace.evGx[414]*acadoWorkspace.x[2] + acadoWorkspace.evGx[415]*acadoWorkspace.x[3] + acadoWorkspace.d[103];
acadoVariables.x[108] += + acadoWorkspace.evGx[416]*acadoWorkspace.x[0] + acadoWorkspace.evGx[417]*acadoWorkspace.x[1] + acadoWorkspace.evGx[418]*acadoWorkspace.x[2] + acadoWorkspace.evGx[419]*acadoWorkspace.x[3] + acadoWorkspace.d[104];
acadoVariables.x[109] += + acadoWorkspace.evGx[420]*acadoWorkspace.x[0] + acadoWorkspace.evGx[421]*acadoWorkspace.x[1] + acadoWorkspace.evGx[422]*acadoWorkspace.x[2] + acadoWorkspace.evGx[423]*acadoWorkspace.x[3] + acadoWorkspace.d[105];
acadoVariables.x[110] += + acadoWorkspace.evGx[424]*acadoWorkspace.x[0] + acadoWorkspace.evGx[425]*acadoWorkspace.x[1] + acadoWorkspace.evGx[426]*acadoWorkspace.x[2] + acadoWorkspace.evGx[427]*acadoWorkspace.x[3] + acadoWorkspace.d[106];
acadoVariables.x[111] += + acadoWorkspace.evGx[428]*acadoWorkspace.x[0] + acadoWorkspace.evGx[429]*acadoWorkspace.x[1] + acadoWorkspace.evGx[430]*acadoWorkspace.x[2] + acadoWorkspace.evGx[431]*acadoWorkspace.x[3] + acadoWorkspace.d[107];
acadoVariables.x[112] += + acadoWorkspace.evGx[432]*acadoWorkspace.x[0] + acadoWorkspace.evGx[433]*acadoWorkspace.x[1] + acadoWorkspace.evGx[434]*acadoWorkspace.x[2] + acadoWorkspace.evGx[435]*acadoWorkspace.x[3] + acadoWorkspace.d[108];
acadoVariables.x[113] += + acadoWorkspace.evGx[436]*acadoWorkspace.x[0] + acadoWorkspace.evGx[437]*acadoWorkspace.x[1] + acadoWorkspace.evGx[438]*acadoWorkspace.x[2] + acadoWorkspace.evGx[439]*acadoWorkspace.x[3] + acadoWorkspace.d[109];
acadoVariables.x[114] += + acadoWorkspace.evGx[440]*acadoWorkspace.x[0] + acadoWorkspace.evGx[441]*acadoWorkspace.x[1] + acadoWorkspace.evGx[442]*acadoWorkspace.x[2] + acadoWorkspace.evGx[443]*acadoWorkspace.x[3] + acadoWorkspace.d[110];
acadoVariables.x[115] += + acadoWorkspace.evGx[444]*acadoWorkspace.x[0] + acadoWorkspace.evGx[445]*acadoWorkspace.x[1] + acadoWorkspace.evGx[446]*acadoWorkspace.x[2] + acadoWorkspace.evGx[447]*acadoWorkspace.x[3] + acadoWorkspace.d[111];
acadoVariables.x[116] += + acadoWorkspace.evGx[448]*acadoWorkspace.x[0] + acadoWorkspace.evGx[449]*acadoWorkspace.x[1] + acadoWorkspace.evGx[450]*acadoWorkspace.x[2] + acadoWorkspace.evGx[451]*acadoWorkspace.x[3] + acadoWorkspace.d[112];
acadoVariables.x[117] += + acadoWorkspace.evGx[452]*acadoWorkspace.x[0] + acadoWorkspace.evGx[453]*acadoWorkspace.x[1] + acadoWorkspace.evGx[454]*acadoWorkspace.x[2] + acadoWorkspace.evGx[455]*acadoWorkspace.x[3] + acadoWorkspace.d[113];
acadoVariables.x[118] += + acadoWorkspace.evGx[456]*acadoWorkspace.x[0] + acadoWorkspace.evGx[457]*acadoWorkspace.x[1] + acadoWorkspace.evGx[458]*acadoWorkspace.x[2] + acadoWorkspace.evGx[459]*acadoWorkspace.x[3] + acadoWorkspace.d[114];
acadoVariables.x[119] += + acadoWorkspace.evGx[460]*acadoWorkspace.x[0] + acadoWorkspace.evGx[461]*acadoWorkspace.x[1] + acadoWorkspace.evGx[462]*acadoWorkspace.x[2] + acadoWorkspace.evGx[463]*acadoWorkspace.x[3] + acadoWorkspace.d[115];
acadoVariables.x[120] += + acadoWorkspace.evGx[464]*acadoWorkspace.x[0] + acadoWorkspace.evGx[465]*acadoWorkspace.x[1] + acadoWorkspace.evGx[466]*acadoWorkspace.x[2] + acadoWorkspace.evGx[467]*acadoWorkspace.x[3] + acadoWorkspace.d[116];
acadoVariables.x[121] += + acadoWorkspace.evGx[468]*acadoWorkspace.x[0] + acadoWorkspace.evGx[469]*acadoWorkspace.x[1] + acadoWorkspace.evGx[470]*acadoWorkspace.x[2] + acadoWorkspace.evGx[471]*acadoWorkspace.x[3] + acadoWorkspace.d[117];
acadoVariables.x[122] += + acadoWorkspace.evGx[472]*acadoWorkspace.x[0] + acadoWorkspace.evGx[473]*acadoWorkspace.x[1] + acadoWorkspace.evGx[474]*acadoWorkspace.x[2] + acadoWorkspace.evGx[475]*acadoWorkspace.x[3] + acadoWorkspace.d[118];
acadoVariables.x[123] += + acadoWorkspace.evGx[476]*acadoWorkspace.x[0] + acadoWorkspace.evGx[477]*acadoWorkspace.x[1] + acadoWorkspace.evGx[478]*acadoWorkspace.x[2] + acadoWorkspace.evGx[479]*acadoWorkspace.x[3] + acadoWorkspace.d[119];
acadoVariables.x[124] += + acadoWorkspace.evGx[480]*acadoWorkspace.x[0] + acadoWorkspace.evGx[481]*acadoWorkspace.x[1] + acadoWorkspace.evGx[482]*acadoWorkspace.x[2] + acadoWorkspace.evGx[483]*acadoWorkspace.x[3] + acadoWorkspace.d[120];
acadoVariables.x[125] += + acadoWorkspace.evGx[484]*acadoWorkspace.x[0] + acadoWorkspace.evGx[485]*acadoWorkspace.x[1] + acadoWorkspace.evGx[486]*acadoWorkspace.x[2] + acadoWorkspace.evGx[487]*acadoWorkspace.x[3] + acadoWorkspace.d[121];
acadoVariables.x[126] += + acadoWorkspace.evGx[488]*acadoWorkspace.x[0] + acadoWorkspace.evGx[489]*acadoWorkspace.x[1] + acadoWorkspace.evGx[490]*acadoWorkspace.x[2] + acadoWorkspace.evGx[491]*acadoWorkspace.x[3] + acadoWorkspace.d[122];
acadoVariables.x[127] += + acadoWorkspace.evGx[492]*acadoWorkspace.x[0] + acadoWorkspace.evGx[493]*acadoWorkspace.x[1] + acadoWorkspace.evGx[494]*acadoWorkspace.x[2] + acadoWorkspace.evGx[495]*acadoWorkspace.x[3] + acadoWorkspace.d[123];
acadoVariables.x[128] += + acadoWorkspace.evGx[496]*acadoWorkspace.x[0] + acadoWorkspace.evGx[497]*acadoWorkspace.x[1] + acadoWorkspace.evGx[498]*acadoWorkspace.x[2] + acadoWorkspace.evGx[499]*acadoWorkspace.x[3] + acadoWorkspace.d[124];
acadoVariables.x[129] += + acadoWorkspace.evGx[500]*acadoWorkspace.x[0] + acadoWorkspace.evGx[501]*acadoWorkspace.x[1] + acadoWorkspace.evGx[502]*acadoWorkspace.x[2] + acadoWorkspace.evGx[503]*acadoWorkspace.x[3] + acadoWorkspace.d[125];
acadoVariables.x[130] += + acadoWorkspace.evGx[504]*acadoWorkspace.x[0] + acadoWorkspace.evGx[505]*acadoWorkspace.x[1] + acadoWorkspace.evGx[506]*acadoWorkspace.x[2] + acadoWorkspace.evGx[507]*acadoWorkspace.x[3] + acadoWorkspace.d[126];
acadoVariables.x[131] += + acadoWorkspace.evGx[508]*acadoWorkspace.x[0] + acadoWorkspace.evGx[509]*acadoWorkspace.x[1] + acadoWorkspace.evGx[510]*acadoWorkspace.x[2] + acadoWorkspace.evGx[511]*acadoWorkspace.x[3] + acadoWorkspace.d[127];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multEDu( &(acadoWorkspace.E[ lRun3 * 8 ]), &(acadoWorkspace.x[ lRun2 * 2 + 4 ]), &(acadoVariables.x[ lRun1 * 4 + 4 ]) );
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
acadoWorkspace.state[0] = acadoVariables.x[index * 4];
acadoWorkspace.state[1] = acadoVariables.x[index * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 4 + 3];
acadoWorkspace.state[28] = acadoVariables.u[index * 2];
acadoWorkspace.state[29] = acadoVariables.u[index * 2 + 1];
acadoWorkspace.state[30] = acadoVariables.od[index * 2];
acadoWorkspace.state[31] = acadoVariables.od[index * 2 + 1];

acado_integrate(acadoWorkspace.state, index == 0, index);

acadoVariables.x[index * 4 + 4] = acadoWorkspace.state[0];
acadoVariables.x[index * 4 + 5] = acadoWorkspace.state[1];
acadoVariables.x[index * 4 + 6] = acadoWorkspace.state[2];
acadoVariables.x[index * 4 + 7] = acadoWorkspace.state[3];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 32; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[128] = xEnd[0];
acadoVariables.x[129] = xEnd[1];
acadoVariables.x[130] = xEnd[2];
acadoVariables.x[131] = xEnd[3];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[128];
acadoWorkspace.state[1] = acadoVariables.x[129];
acadoWorkspace.state[2] = acadoVariables.x[130];
acadoWorkspace.state[3] = acadoVariables.x[131];
if (uEnd != 0)
{
acadoWorkspace.state[28] = uEnd[0];
acadoWorkspace.state[29] = uEnd[1];
}
else
{
acadoWorkspace.state[28] = acadoVariables.u[62];
acadoWorkspace.state[29] = acadoVariables.u[63];
}
acadoWorkspace.state[30] = acadoVariables.od[64];
acadoWorkspace.state[31] = acadoVariables.od[65];

acado_integrate(acadoWorkspace.state, 1, 31);

acadoVariables.x[128] = acadoWorkspace.state[0];
acadoVariables.x[129] = acadoWorkspace.state[1];
acadoVariables.x[130] = acadoWorkspace.state[2];
acadoVariables.x[131] = acadoWorkspace.state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 31; ++index)
{
acadoVariables.u[index * 2] = acadoVariables.u[index * 2 + 2];
acadoVariables.u[index * 2 + 1] = acadoVariables.u[index * 2 + 3];
}

if (uEnd != 0)
{
acadoVariables.u[62] = uEnd[0];
acadoVariables.u[63] = uEnd[1];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43] + acadoWorkspace.g[44]*acadoWorkspace.x[44] + acadoWorkspace.g[45]*acadoWorkspace.x[45] + acadoWorkspace.g[46]*acadoWorkspace.x[46] + acadoWorkspace.g[47]*acadoWorkspace.x[47] + acadoWorkspace.g[48]*acadoWorkspace.x[48] + acadoWorkspace.g[49]*acadoWorkspace.x[49] + acadoWorkspace.g[50]*acadoWorkspace.x[50] + acadoWorkspace.g[51]*acadoWorkspace.x[51] + acadoWorkspace.g[52]*acadoWorkspace.x[52] + acadoWorkspace.g[53]*acadoWorkspace.x[53] + acadoWorkspace.g[54]*acadoWorkspace.x[54] + acadoWorkspace.g[55]*acadoWorkspace.x[55] + acadoWorkspace.g[56]*acadoWorkspace.x[56] + acadoWorkspace.g[57]*acadoWorkspace.x[57] + acadoWorkspace.g[58]*acadoWorkspace.x[58] + acadoWorkspace.g[59]*acadoWorkspace.x[59] + acadoWorkspace.g[60]*acadoWorkspace.x[60] + acadoWorkspace.g[61]*acadoWorkspace.x[61] + acadoWorkspace.g[62]*acadoWorkspace.x[62] + acadoWorkspace.g[63]*acadoWorkspace.x[63] + acadoWorkspace.g[64]*acadoWorkspace.x[64] + acadoWorkspace.g[65]*acadoWorkspace.x[65] + acadoWorkspace.g[66]*acadoWorkspace.x[66] + acadoWorkspace.g[67]*acadoWorkspace.x[67];
kkt = fabs( kkt );
for (index = 0; index < 68; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 96; ++index)
{
prd = acadoWorkspace.y[index + 68];
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

/** Row vector of size: 3 */
real_t tmpDyN[ 3 ];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.objValueIn[6] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.objValueIn[7] = acadoVariables.od[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 5] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 5];
acadoWorkspace.Dy[lRun1 * 5 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 5 + 1];
acadoWorkspace.Dy[lRun1 * 5 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 5 + 2];
acadoWorkspace.Dy[lRun1 * 5 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 5 + 3];
acadoWorkspace.Dy[lRun1 * 5 + 4] = acadoWorkspace.objValueOut[4] - acadoVariables.y[lRun1 * 5 + 4];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[128];
acadoWorkspace.objValueIn[1] = acadoVariables.x[129];
acadoWorkspace.objValueIn[2] = acadoVariables.x[130];
acadoWorkspace.objValueIn[3] = acadoVariables.x[131];
acadoWorkspace.objValueIn[4] = acadoVariables.od[64];
acadoWorkspace.objValueIn[5] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 5] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 10] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 15] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 20];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 1] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 6] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 11] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 16] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 21];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 2] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 7] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 12] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 17] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 22];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 3] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 8] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 13] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 18] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 23];
tmpDy[4] = + acadoWorkspace.Dy[lRun1 * 5]*acadoVariables.W[lRun1 * 25 + 4] + acadoWorkspace.Dy[lRun1 * 5 + 1]*acadoVariables.W[lRun1 * 25 + 9] + acadoWorkspace.Dy[lRun1 * 5 + 2]*acadoVariables.W[lRun1 * 25 + 14] + acadoWorkspace.Dy[lRun1 * 5 + 3]*acadoVariables.W[lRun1 * 25 + 19] + acadoWorkspace.Dy[lRun1 * 5 + 4]*acadoVariables.W[lRun1 * 25 + 24];
objVal += + acadoWorkspace.Dy[lRun1 * 5]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 5 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 5 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 5 + 3]*tmpDy[3] + acadoWorkspace.Dy[lRun1 * 5 + 4]*tmpDy[4];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0] + acadoWorkspace.DyN[1]*acadoVariables.WN[3] + acadoWorkspace.DyN[2]*acadoVariables.WN[6];
tmpDyN[1] = + acadoWorkspace.DyN[0]*acadoVariables.WN[1] + acadoWorkspace.DyN[1]*acadoVariables.WN[4] + acadoWorkspace.DyN[2]*acadoVariables.WN[7];
tmpDyN[2] = + acadoWorkspace.DyN[0]*acadoVariables.WN[2] + acadoWorkspace.DyN[1]*acadoVariables.WN[5] + acadoWorkspace.DyN[2]*acadoVariables.WN[8];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2];

objVal *= 0.5;
return objVal;
}

