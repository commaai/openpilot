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
for (lRun1 = 0; lRun1 < 16; ++lRun1)
{
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 4 + 3];

acadoWorkspace.state[24] = acadoVariables.u[lRun1];
acadoWorkspace.state[25] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.state[26] = acadoVariables.od[lRun1 * 2 + 1];

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

acadoWorkspace.evGu[lRun1 * 4] = acadoWorkspace.state[20];
acadoWorkspace.evGu[lRun1 * 4 + 1] = acadoWorkspace.state[21];
acadoWorkspace.evGu[lRun1 * 4 + 2] = acadoWorkspace.state[22];
acadoWorkspace.evGu[lRun1 * 4 + 3] = acadoWorkspace.state[23];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 4;
const real_t* od = in + 5;

/* Compute outputs: */
out[0] = xd[1];
out[1] = ((od[0]+(real_t)(5.0000000000000000e+00))*xd[2]);
out[2] = (((od[0]+(real_t)(5.0000000000000000e+00))*(real_t)(4.0000000000000000e+00))*u[0]);
out[3] = (real_t)(0.0000000000000000e+00);
out[4] = (real_t)(1.0000000000000000e+00);
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (od[0]+(real_t)(5.0000000000000000e+00));
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = (real_t)(0.0000000000000000e+00);
out[14] = (real_t)(0.0000000000000000e+00);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = ((od[0]+(real_t)(5.0000000000000000e+00))*(real_t)(4.0000000000000000e+00));
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 4;

/* Compute outputs: */
out[0] = xd[1];
out[1] = ((((real_t)(2.0000000000000000e+00)*od[0])+(real_t)(5.0000000000000000e+00))*xd[2]);
out[2] = (real_t)(0.0000000000000000e+00);
out[3] = (real_t)(1.0000000000000000e+00);
out[4] = (real_t)(0.0000000000000000e+00);
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (((real_t)(2.0000000000000000e+00)*od[0])+(real_t)(5.0000000000000000e+00));
out[9] = (real_t)(0.0000000000000000e+00);
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0]*tmpObjS[0] + tmpFx[4]*tmpObjS[3] + tmpFx[8]*tmpObjS[6];
tmpQ2[1] = + tmpFx[0]*tmpObjS[1] + tmpFx[4]*tmpObjS[4] + tmpFx[8]*tmpObjS[7];
tmpQ2[2] = + tmpFx[0]*tmpObjS[2] + tmpFx[4]*tmpObjS[5] + tmpFx[8]*tmpObjS[8];
tmpQ2[3] = + tmpFx[1]*tmpObjS[0] + tmpFx[5]*tmpObjS[3] + tmpFx[9]*tmpObjS[6];
tmpQ2[4] = + tmpFx[1]*tmpObjS[1] + tmpFx[5]*tmpObjS[4] + tmpFx[9]*tmpObjS[7];
tmpQ2[5] = + tmpFx[1]*tmpObjS[2] + tmpFx[5]*tmpObjS[5] + tmpFx[9]*tmpObjS[8];
tmpQ2[6] = + tmpFx[2]*tmpObjS[0] + tmpFx[6]*tmpObjS[3] + tmpFx[10]*tmpObjS[6];
tmpQ2[7] = + tmpFx[2]*tmpObjS[1] + tmpFx[6]*tmpObjS[4] + tmpFx[10]*tmpObjS[7];
tmpQ2[8] = + tmpFx[2]*tmpObjS[2] + tmpFx[6]*tmpObjS[5] + tmpFx[10]*tmpObjS[8];
tmpQ2[9] = + tmpFx[3]*tmpObjS[0] + tmpFx[7]*tmpObjS[3] + tmpFx[11]*tmpObjS[6];
tmpQ2[10] = + tmpFx[3]*tmpObjS[1] + tmpFx[7]*tmpObjS[4] + tmpFx[11]*tmpObjS[7];
tmpQ2[11] = + tmpFx[3]*tmpObjS[2] + tmpFx[7]*tmpObjS[5] + tmpFx[11]*tmpObjS[8];
tmpQ1[0] = + tmpQ2[0]*tmpFx[0] + tmpQ2[1]*tmpFx[4] + tmpQ2[2]*tmpFx[8];
tmpQ1[1] = + tmpQ2[0]*tmpFx[1] + tmpQ2[1]*tmpFx[5] + tmpQ2[2]*tmpFx[9];
tmpQ1[2] = + tmpQ2[0]*tmpFx[2] + tmpQ2[1]*tmpFx[6] + tmpQ2[2]*tmpFx[10];
tmpQ1[3] = + tmpQ2[0]*tmpFx[3] + tmpQ2[1]*tmpFx[7] + tmpQ2[2]*tmpFx[11];
tmpQ1[4] = + tmpQ2[3]*tmpFx[0] + tmpQ2[4]*tmpFx[4] + tmpQ2[5]*tmpFx[8];
tmpQ1[5] = + tmpQ2[3]*tmpFx[1] + tmpQ2[4]*tmpFx[5] + tmpQ2[5]*tmpFx[9];
tmpQ1[6] = + tmpQ2[3]*tmpFx[2] + tmpQ2[4]*tmpFx[6] + tmpQ2[5]*tmpFx[10];
tmpQ1[7] = + tmpQ2[3]*tmpFx[3] + tmpQ2[4]*tmpFx[7] + tmpQ2[5]*tmpFx[11];
tmpQ1[8] = + tmpQ2[6]*tmpFx[0] + tmpQ2[7]*tmpFx[4] + tmpQ2[8]*tmpFx[8];
tmpQ1[9] = + tmpQ2[6]*tmpFx[1] + tmpQ2[7]*tmpFx[5] + tmpQ2[8]*tmpFx[9];
tmpQ1[10] = + tmpQ2[6]*tmpFx[2] + tmpQ2[7]*tmpFx[6] + tmpQ2[8]*tmpFx[10];
tmpQ1[11] = + tmpQ2[6]*tmpFx[3] + tmpQ2[7]*tmpFx[7] + tmpQ2[8]*tmpFx[11];
tmpQ1[12] = + tmpQ2[9]*tmpFx[0] + tmpQ2[10]*tmpFx[4] + tmpQ2[11]*tmpFx[8];
tmpQ1[13] = + tmpQ2[9]*tmpFx[1] + tmpQ2[10]*tmpFx[5] + tmpQ2[11]*tmpFx[9];
tmpQ1[14] = + tmpQ2[9]*tmpFx[2] + tmpQ2[10]*tmpFx[6] + tmpQ2[11]*tmpFx[10];
tmpQ1[15] = + tmpQ2[9]*tmpFx[3] + tmpQ2[10]*tmpFx[7] + tmpQ2[11]*tmpFx[11];
}

void acado_setObjR1R2( real_t* const tmpFu, real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = + tmpFu[0]*tmpObjS[0] + tmpFu[1]*tmpObjS[3] + tmpFu[2]*tmpObjS[6];
tmpR2[1] = + tmpFu[0]*tmpObjS[1] + tmpFu[1]*tmpObjS[4] + tmpFu[2]*tmpObjS[7];
tmpR2[2] = + tmpFu[0]*tmpObjS[2] + tmpFu[1]*tmpObjS[5] + tmpFu[2]*tmpObjS[8];
tmpR1[0] = + tmpR2[0]*tmpFu[0] + tmpR2[1]*tmpFu[1] + tmpR2[2]*tmpFu[2];
}

void acado_setObjQN1QN2( real_t* const tmpFx, real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = + tmpFx[0]*tmpObjSEndTerm[0] + tmpFx[4]*tmpObjSEndTerm[2];
tmpQN2[1] = + tmpFx[0]*tmpObjSEndTerm[1] + tmpFx[4]*tmpObjSEndTerm[3];
tmpQN2[2] = + tmpFx[1]*tmpObjSEndTerm[0] + tmpFx[5]*tmpObjSEndTerm[2];
tmpQN2[3] = + tmpFx[1]*tmpObjSEndTerm[1] + tmpFx[5]*tmpObjSEndTerm[3];
tmpQN2[4] = + tmpFx[2]*tmpObjSEndTerm[0] + tmpFx[6]*tmpObjSEndTerm[2];
tmpQN2[5] = + tmpFx[2]*tmpObjSEndTerm[1] + tmpFx[6]*tmpObjSEndTerm[3];
tmpQN2[6] = + tmpFx[3]*tmpObjSEndTerm[0] + tmpFx[7]*tmpObjSEndTerm[2];
tmpQN2[7] = + tmpFx[3]*tmpObjSEndTerm[1] + tmpFx[7]*tmpObjSEndTerm[3];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0] + tmpQN2[1]*tmpFx[4];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1] + tmpQN2[1]*tmpFx[5];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2] + tmpQN2[1]*tmpFx[6];
tmpQN1[3] = + tmpQN2[0]*tmpFx[3] + tmpQN2[1]*tmpFx[7];
tmpQN1[4] = + tmpQN2[2]*tmpFx[0] + tmpQN2[3]*tmpFx[4];
tmpQN1[5] = + tmpQN2[2]*tmpFx[1] + tmpQN2[3]*tmpFx[5];
tmpQN1[6] = + tmpQN2[2]*tmpFx[2] + tmpQN2[3]*tmpFx[6];
tmpQN1[7] = + tmpQN2[2]*tmpFx[3] + tmpQN2[3]*tmpFx[7];
tmpQN1[8] = + tmpQN2[4]*tmpFx[0] + tmpQN2[5]*tmpFx[4];
tmpQN1[9] = + tmpQN2[4]*tmpFx[1] + tmpQN2[5]*tmpFx[5];
tmpQN1[10] = + tmpQN2[4]*tmpFx[2] + tmpQN2[5]*tmpFx[6];
tmpQN1[11] = + tmpQN2[4]*tmpFx[3] + tmpQN2[5]*tmpFx[7];
tmpQN1[12] = + tmpQN2[6]*tmpFx[0] + tmpQN2[7]*tmpFx[4];
tmpQN1[13] = + tmpQN2[6]*tmpFx[1] + tmpQN2[7]*tmpFx[5];
tmpQN1[14] = + tmpQN2[6]*tmpFx[2] + tmpQN2[7]*tmpFx[6];
tmpQN1[15] = + tmpQN2[6]*tmpFx[3] + tmpQN2[7]*tmpFx[7];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 16; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[runObj];
acadoWorkspace.objValueIn[5] = acadoVariables.od[runObj * 2];
acadoWorkspace.objValueIn[6] = acadoVariables.od[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 3] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 3 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 3 + 2] = acadoWorkspace.objValueOut[2];

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 3 ]), &(acadoVariables.W[ runObj * 9 ]), &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 12 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 15 ]), &(acadoVariables.W[ runObj * 9 ]), &(acadoWorkspace.R1[ runObj ]), &(acadoWorkspace.R2[ runObj * 3 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[64];
acadoWorkspace.objValueIn[1] = acadoVariables.x[65];
acadoWorkspace.objValueIn[2] = acadoVariables.x[66];
acadoWorkspace.objValueIn[3] = acadoVariables.x[67];
acadoWorkspace.objValueIn[4] = acadoVariables.od[32];
acadoWorkspace.objValueIn[5] = acadoVariables.od[33];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 2 ]), acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

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
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[1] + Gx1[2]*Gu1[2] + Gx1[3]*Gu1[3];
Gu2[1] = + Gx1[4]*Gu1[0] + Gx1[5]*Gu1[1] + Gx1[6]*Gu1[2] + Gx1[7]*Gu1[3];
Gu2[2] = + Gx1[8]*Gu1[0] + Gx1[9]*Gu1[1] + Gx1[10]*Gu1[2] + Gx1[11]*Gu1[3];
Gu2[3] = + Gx1[12]*Gu1[0] + Gx1[13]*Gu1[1] + Gx1[14]*Gu1[2] + Gx1[15]*Gu1[3];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
Gu2[3] = Gu1[3];
}

void acado_setBlockH11( int iRow, int iCol, real_t* const Gu1, real_t* const Gu2 )
{
acadoWorkspace.H[(iRow * 20 + 80) + (iCol + 4)] += + Gu1[0]*Gu2[0] + Gu1[1]*Gu2[1] + Gu1[2]*Gu2[2] + Gu1[3]*Gu2[3];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 20 + 80) + (iCol + 4)] = R11[0];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 20 + 80) + (iCol + 4)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 20 + 80) + (iCol + 4)] = acadoWorkspace.H[(iCol * 20 + 80) + (iRow + 4)];
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
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2];
QDy1[1] = + Q2[3]*Dy1[0] + Q2[4]*Dy1[1] + Q2[5]*Dy1[2];
QDy1[2] = + Q2[6]*Dy1[0] + Q2[7]*Dy1[1] + Q2[8]*Dy1[2];
QDy1[3] = + Q2[9]*Dy1[0] + Q2[10]*Dy1[1] + Q2[11]*Dy1[2];
}

void acado_multEQDy( real_t* const E1, real_t* const QDy1, real_t* const U1 )
{
U1[0] += + E1[0]*QDy1[0] + E1[1]*QDy1[1] + E1[2]*QDy1[2] + E1[3]*QDy1[3];
}

void acado_multQETGx( real_t* const E1, real_t* const Gx1, real_t* const H101 )
{
H101[0] += + E1[0]*Gx1[0] + E1[1]*Gx1[4] + E1[2]*Gx1[8] + E1[3]*Gx1[12];
H101[1] += + E1[0]*Gx1[1] + E1[1]*Gx1[5] + E1[2]*Gx1[9] + E1[3]*Gx1[13];
H101[2] += + E1[0]*Gx1[2] + E1[1]*Gx1[6] + E1[2]*Gx1[10] + E1[3]*Gx1[14];
H101[3] += + E1[0]*Gx1[3] + E1[1]*Gx1[7] + E1[2]*Gx1[11] + E1[3]*Gx1[15];
}

void acado_zeroBlockH10( real_t* const H101 )
{
{ int lCopy; for (lCopy = 0; lCopy < 4; lCopy++) H101[ lCopy ] = 0; }
}

void acado_multEDu( real_t* const E1, real_t* const U1, real_t* const dNew )
{
dNew[0] += + E1[0]*U1[0];
dNew[1] += + E1[1]*U1[0];
dNew[2] += + E1[2]*U1[0];
dNew[3] += + E1[3]*U1[0];
}

void acado_zeroBlockH00(  )
{
acadoWorkspace.H[0] = 0.0000000000000000e+00;
acadoWorkspace.H[1] = 0.0000000000000000e+00;
acadoWorkspace.H[2] = 0.0000000000000000e+00;
acadoWorkspace.H[3] = 0.0000000000000000e+00;
acadoWorkspace.H[20] = 0.0000000000000000e+00;
acadoWorkspace.H[21] = 0.0000000000000000e+00;
acadoWorkspace.H[22] = 0.0000000000000000e+00;
acadoWorkspace.H[23] = 0.0000000000000000e+00;
acadoWorkspace.H[40] = 0.0000000000000000e+00;
acadoWorkspace.H[41] = 0.0000000000000000e+00;
acadoWorkspace.H[42] = 0.0000000000000000e+00;
acadoWorkspace.H[43] = 0.0000000000000000e+00;
acadoWorkspace.H[60] = 0.0000000000000000e+00;
acadoWorkspace.H[61] = 0.0000000000000000e+00;
acadoWorkspace.H[62] = 0.0000000000000000e+00;
acadoWorkspace.H[63] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[4]*Gx2[4] + Gx1[8]*Gx2[8] + Gx1[12]*Gx2[12];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[4]*Gx2[5] + Gx1[8]*Gx2[9] + Gx1[12]*Gx2[13];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[4]*Gx2[6] + Gx1[8]*Gx2[10] + Gx1[12]*Gx2[14];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[4]*Gx2[7] + Gx1[8]*Gx2[11] + Gx1[12]*Gx2[15];
acadoWorkspace.H[20] += + Gx1[1]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[9]*Gx2[8] + Gx1[13]*Gx2[12];
acadoWorkspace.H[21] += + Gx1[1]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[9]*Gx2[9] + Gx1[13]*Gx2[13];
acadoWorkspace.H[22] += + Gx1[1]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[9]*Gx2[10] + Gx1[13]*Gx2[14];
acadoWorkspace.H[23] += + Gx1[1]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[9]*Gx2[11] + Gx1[13]*Gx2[15];
acadoWorkspace.H[40] += + Gx1[2]*Gx2[0] + Gx1[6]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[14]*Gx2[12];
acadoWorkspace.H[41] += + Gx1[2]*Gx2[1] + Gx1[6]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[14]*Gx2[13];
acadoWorkspace.H[42] += + Gx1[2]*Gx2[2] + Gx1[6]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[14]*Gx2[14];
acadoWorkspace.H[43] += + Gx1[2]*Gx2[3] + Gx1[6]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[14]*Gx2[15];
acadoWorkspace.H[60] += + Gx1[3]*Gx2[0] + Gx1[7]*Gx2[4] + Gx1[11]*Gx2[8] + Gx1[15]*Gx2[12];
acadoWorkspace.H[61] += + Gx1[3]*Gx2[1] + Gx1[7]*Gx2[5] + Gx1[11]*Gx2[9] + Gx1[15]*Gx2[13];
acadoWorkspace.H[62] += + Gx1[3]*Gx2[2] + Gx1[7]*Gx2[6] + Gx1[11]*Gx2[10] + Gx1[15]*Gx2[14];
acadoWorkspace.H[63] += + Gx1[3]*Gx2[3] + Gx1[7]*Gx2[7] + Gx1[11]*Gx2[11] + Gx1[15]*Gx2[15];
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
{ 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63, 66, 67 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_moveGxT( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.T );
acado_multGxd( acadoWorkspace.d, &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.d[ 4 ]) );
acado_multGxGx( acadoWorkspace.T, acadoWorkspace.evGx, &(acadoWorkspace.evGx[ 16 ]) );

acado_multGxGu( acadoWorkspace.T, acadoWorkspace.E, &(acadoWorkspace.E[ 4 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 4 ]), &(acadoWorkspace.E[ 8 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 4 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.d[ 8 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.evGx[ 32 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.E[ 12 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.E[ 16 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 8 ]), &(acadoWorkspace.E[ 20 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 8 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.d[ 12 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.evGx[ 48 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.E[ 24 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.E[ 28 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 20 ]), &(acadoWorkspace.E[ 32 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 12 ]), &(acadoWorkspace.E[ 36 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.d[ 16 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.evGx[ 64 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.E[ 40 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.E[ 44 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.E[ 48 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.E[ 52 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 16 ]), &(acadoWorkspace.E[ 56 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 16 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.d[ 20 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.evGx[ 80 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.E[ 60 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.E[ 64 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.E[ 68 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.E[ 72 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.E[ 76 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 20 ]), &(acadoWorkspace.E[ 80 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.d[ 24 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.evGx[ 96 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.E[ 84 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.E[ 88 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.E[ 92 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.E[ 96 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.E[ 100 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.E[ 104 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.E[ 108 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.d[ 28 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.evGx[ 112 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.E[ 112 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.E[ 116 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.E[ 120 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.E[ 124 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.E[ 128 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.E[ 132 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.E[ 136 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 28 ]), &(acadoWorkspace.E[ 140 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 28 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.d[ 32 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.evGx[ 128 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.E[ 144 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.E[ 148 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.E[ 152 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.E[ 156 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.E[ 160 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.E[ 164 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.E[ 168 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 140 ]), &(acadoWorkspace.E[ 172 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 32 ]), &(acadoWorkspace.E[ 176 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 32 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.d[ 36 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.evGx[ 144 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.E[ 180 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.E[ 184 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.E[ 188 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.E[ 192 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.E[ 196 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.E[ 200 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.E[ 204 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.E[ 208 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.E[ 212 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 36 ]), &(acadoWorkspace.E[ 216 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.d[ 40 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.evGx[ 160 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.E[ 220 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.E[ 224 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.E[ 228 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.E[ 232 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.E[ 236 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.E[ 240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.E[ 244 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.E[ 248 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.E[ 252 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.E[ 256 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 40 ]), &(acadoWorkspace.E[ 260 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.d[ 44 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.evGx[ 176 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.E[ 264 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.E[ 268 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.E[ 272 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.E[ 276 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.E[ 280 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.E[ 284 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.E[ 288 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.E[ 292 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.E[ 296 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.E[ 300 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 260 ]), &(acadoWorkspace.E[ 304 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 44 ]), &(acadoWorkspace.E[ 308 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 44 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.d[ 48 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.evGx[ 192 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.E[ 312 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.E[ 316 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.E[ 320 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.E[ 324 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.E[ 328 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.E[ 332 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.E[ 336 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.E[ 340 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.E[ 344 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.E[ 348 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.E[ 352 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 308 ]), &(acadoWorkspace.E[ 356 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 48 ]), &(acadoWorkspace.E[ 360 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.d[ 52 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.evGx[ 208 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.E[ 364 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.E[ 368 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.E[ 372 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.E[ 376 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.E[ 380 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.E[ 384 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.E[ 388 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.E[ 392 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.E[ 396 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.E[ 400 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.E[ 404 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.E[ 408 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.E[ 412 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 52 ]), &(acadoWorkspace.E[ 416 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 52 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.d[ 56 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.evGx[ 224 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.E[ 420 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.E[ 424 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.E[ 428 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.E[ 432 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.E[ 436 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.E[ 440 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.E[ 444 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.E[ 448 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.E[ 452 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.E[ 456 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.E[ 460 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.E[ 464 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.E[ 468 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.E[ 472 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 56 ]), &(acadoWorkspace.E[ 476 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 56 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.d[ 60 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.evGx[ 240 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.E[ 480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.E[ 484 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.E[ 488 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.E[ 492 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.E[ 496 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.E[ 500 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.E[ 504 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.E[ 508 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.E[ 512 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.E[ 516 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.E[ 520 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.E[ 524 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.E[ 528 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.E[ 532 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 476 ]), &(acadoWorkspace.E[ 536 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 60 ]), &(acadoWorkspace.E[ 540 ]) );

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
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.QGx[ 240 ]) );

acado_multGxGu( &(acadoWorkspace.Q1[ 16 ]), acadoWorkspace.E, acadoWorkspace.QE );
acado_multGxGu( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.QE[ 4 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QE[ 8 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 20 ]), &(acadoWorkspace.QE[ 20 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.QE[ 28 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QE[ 44 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.QE[ 52 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QE[ 68 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 92 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 116 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 124 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 140 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 148 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 188 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 220 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 224 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 260 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 268 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 308 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 316 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 364 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 428 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 476 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 484 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 500 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 508 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 524 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.QE[ 532 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 536 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );

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

acado_zeroBlockH10( acadoWorkspace.H10 );
acado_multQETGx( acadoWorkspace.QE, acadoWorkspace.evGx, acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 4 ]), &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 40 ]), &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 112 ]), &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 220 ]), &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 364 ]), &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.H10 );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 8 ]), &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 16 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 28 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 44 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 64 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 88 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 116 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 148 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 184 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 224 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 268 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 316 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 368 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 424 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 484 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 4 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 20 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 32 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 68 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 92 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 152 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 188 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 272 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 320 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 428 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 488 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 52 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 124 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 232 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 376 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 56 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 76 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 100 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 128 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 160 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 196 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 236 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 280 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 328 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 380 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 436 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 496 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 80 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 104 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 164 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 200 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 284 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 332 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 440 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 500 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 20 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 136 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 244 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 388 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 140 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 172 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 208 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 248 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 292 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 340 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 392 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 448 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 508 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 28 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 176 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 212 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 296 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 344 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 452 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 512 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 256 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 400 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 260 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 304 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 352 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 404 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 460 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 520 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 44 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 308 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 44 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 356 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 44 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 44 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 464 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 44 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 524 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 44 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 412 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 52 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 416 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 52 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 472 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 52 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 532 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 52 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 476 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 536 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 60 ]) );

acadoWorkspace.H[4] = acadoWorkspace.H10[0];
acadoWorkspace.H[5] = acadoWorkspace.H10[4];
acadoWorkspace.H[6] = acadoWorkspace.H10[8];
acadoWorkspace.H[7] = acadoWorkspace.H10[12];
acadoWorkspace.H[8] = acadoWorkspace.H10[16];
acadoWorkspace.H[9] = acadoWorkspace.H10[20];
acadoWorkspace.H[10] = acadoWorkspace.H10[24];
acadoWorkspace.H[11] = acadoWorkspace.H10[28];
acadoWorkspace.H[12] = acadoWorkspace.H10[32];
acadoWorkspace.H[13] = acadoWorkspace.H10[36];
acadoWorkspace.H[14] = acadoWorkspace.H10[40];
acadoWorkspace.H[15] = acadoWorkspace.H10[44];
acadoWorkspace.H[16] = acadoWorkspace.H10[48];
acadoWorkspace.H[17] = acadoWorkspace.H10[52];
acadoWorkspace.H[18] = acadoWorkspace.H10[56];
acadoWorkspace.H[19] = acadoWorkspace.H10[60];
acadoWorkspace.H[24] = acadoWorkspace.H10[1];
acadoWorkspace.H[25] = acadoWorkspace.H10[5];
acadoWorkspace.H[26] = acadoWorkspace.H10[9];
acadoWorkspace.H[27] = acadoWorkspace.H10[13];
acadoWorkspace.H[28] = acadoWorkspace.H10[17];
acadoWorkspace.H[29] = acadoWorkspace.H10[21];
acadoWorkspace.H[30] = acadoWorkspace.H10[25];
acadoWorkspace.H[31] = acadoWorkspace.H10[29];
acadoWorkspace.H[32] = acadoWorkspace.H10[33];
acadoWorkspace.H[33] = acadoWorkspace.H10[37];
acadoWorkspace.H[34] = acadoWorkspace.H10[41];
acadoWorkspace.H[35] = acadoWorkspace.H10[45];
acadoWorkspace.H[36] = acadoWorkspace.H10[49];
acadoWorkspace.H[37] = acadoWorkspace.H10[53];
acadoWorkspace.H[38] = acadoWorkspace.H10[57];
acadoWorkspace.H[39] = acadoWorkspace.H10[61];
acadoWorkspace.H[44] = acadoWorkspace.H10[2];
acadoWorkspace.H[45] = acadoWorkspace.H10[6];
acadoWorkspace.H[46] = acadoWorkspace.H10[10];
acadoWorkspace.H[47] = acadoWorkspace.H10[14];
acadoWorkspace.H[48] = acadoWorkspace.H10[18];
acadoWorkspace.H[49] = acadoWorkspace.H10[22];
acadoWorkspace.H[50] = acadoWorkspace.H10[26];
acadoWorkspace.H[51] = acadoWorkspace.H10[30];
acadoWorkspace.H[52] = acadoWorkspace.H10[34];
acadoWorkspace.H[53] = acadoWorkspace.H10[38];
acadoWorkspace.H[54] = acadoWorkspace.H10[42];
acadoWorkspace.H[55] = acadoWorkspace.H10[46];
acadoWorkspace.H[56] = acadoWorkspace.H10[50];
acadoWorkspace.H[57] = acadoWorkspace.H10[54];
acadoWorkspace.H[58] = acadoWorkspace.H10[58];
acadoWorkspace.H[59] = acadoWorkspace.H10[62];
acadoWorkspace.H[64] = acadoWorkspace.H10[3];
acadoWorkspace.H[65] = acadoWorkspace.H10[7];
acadoWorkspace.H[66] = acadoWorkspace.H10[11];
acadoWorkspace.H[67] = acadoWorkspace.H10[15];
acadoWorkspace.H[68] = acadoWorkspace.H10[19];
acadoWorkspace.H[69] = acadoWorkspace.H10[23];
acadoWorkspace.H[70] = acadoWorkspace.H10[27];
acadoWorkspace.H[71] = acadoWorkspace.H10[31];
acadoWorkspace.H[72] = acadoWorkspace.H10[35];
acadoWorkspace.H[73] = acadoWorkspace.H10[39];
acadoWorkspace.H[74] = acadoWorkspace.H10[43];
acadoWorkspace.H[75] = acadoWorkspace.H10[47];
acadoWorkspace.H[76] = acadoWorkspace.H10[51];
acadoWorkspace.H[77] = acadoWorkspace.H10[55];
acadoWorkspace.H[78] = acadoWorkspace.H10[59];
acadoWorkspace.H[79] = acadoWorkspace.H10[63];

acado_setBlockH11_R1( 0, 0, acadoWorkspace.R1 );
acado_setBlockH11( 0, 0, acadoWorkspace.E, acadoWorkspace.QE );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.QE[ 4 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 220 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 364 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );

acado_zeroBlockH11( 0, 1 );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.QE[ 8 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 28 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 44 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 116 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 148 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 224 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 268 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 316 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 484 ]) );

acado_zeroBlockH11( 0, 2 );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 20 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 68 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 92 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 188 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 428 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 488 ]) );

acado_zeroBlockH11( 0, 3 );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 52 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 124 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 492 ]) );

acado_zeroBlockH11( 0, 4 );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 496 ]) );

acado_zeroBlockH11( 0, 5 );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 0, 6 );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 0, 7 );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 0, 8 );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 0, 9 );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 0, 10 );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 0, 11 );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 0, 12 );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 0, 13 );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 0, 14 );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 0, 15 );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 1, 1, &(acadoWorkspace.R1[ 1 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QE[ 8 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.QE[ 28 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QE[ 44 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 116 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 148 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 224 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 268 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 316 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 484 ]) );

acado_zeroBlockH11( 1, 2 );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QE[ 20 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 68 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 92 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 188 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 428 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 488 ]) );

acado_zeroBlockH11( 1, 3 );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QE[ 52 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 124 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 492 ]) );

acado_zeroBlockH11( 1, 4 );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 496 ]) );

acado_zeroBlockH11( 1, 5 );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 1, 6 );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 1, 7 );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 1, 8 );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 1, 9 );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 1, 10 );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 1, 11 );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 1, 12 );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 1, 13 );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 1, 14 );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 1, 15 );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 2, 2, &(acadoWorkspace.R1[ 2 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 20 ]), &(acadoWorkspace.QE[ 20 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QE[ 68 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 92 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 188 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 428 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 488 ]) );

acado_zeroBlockH11( 2, 3 );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 52 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 124 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 492 ]) );

acado_zeroBlockH11( 2, 4 );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 496 ]) );

acado_zeroBlockH11( 2, 5 );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 2, 6 );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 2, 7 );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 2, 8 );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 2, 9 );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 2, 10 );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 2, 11 );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 2, 12 );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 2, 13 );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 2, 14 );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 2, 15 );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 3, 3, &(acadoWorkspace.R1[ 3 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.QE[ 52 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 124 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );

acado_zeroBlockH11( 3, 4 );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 496 ]) );

acado_zeroBlockH11( 3, 5 );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 3, 6 );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 3, 7 );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 3, 8 );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 3, 9 );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 3, 10 );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 3, 11 );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 3, 12 );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 3, 13 );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 3, 14 );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 3, 15 );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 4, 4, &(acadoWorkspace.R1[ 4 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.QE[ 76 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.QE[ 100 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 196 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 236 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 380 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 436 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 496 ]) );

acado_zeroBlockH11( 4, 5 );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 4, 6 );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 4, 7 );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 4, 8 );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 4, 9 );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 4, 10 );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 4, 11 );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 4, 12 );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 4, 13 );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 4, 14 );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 4, 15 );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 5, 5, &(acadoWorkspace.R1[ 5 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QE[ 164 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 284 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 332 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 500 ]) );

acado_zeroBlockH11( 5, 6 );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 5, 7 );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 5, 8 );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 5, 9 );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 5, 10 );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 5, 11 );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 5, 12 );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 5, 13 );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 5, 14 );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 5, 15 );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 6, 6, &(acadoWorkspace.R1[ 6 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 244 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 388 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );

acado_zeroBlockH11( 6, 7 );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 6, 8 );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 6, 9 );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 6, 10 );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 6, 11 );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 6, 12 );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 6, 13 );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 6, 14 );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 6, 15 );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 7, 7, &(acadoWorkspace.R1[ 7 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 140 ]), &(acadoWorkspace.QE[ 140 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.QE[ 172 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 292 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 340 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 508 ]) );

acado_zeroBlockH11( 7, 8 );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 7, 9 );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 7, 10 );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 7, 11 );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 7, 12 );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 7, 13 );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 7, 14 );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 7, 15 );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 8, 8, &(acadoWorkspace.R1[ 8 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.QE[ 212 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 452 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 512 ]) );

acado_zeroBlockH11( 8, 9 );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 8, 10 );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 8, 11 );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 8, 12 );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 8, 13 );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 8, 14 );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 8, 15 );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 9, 9, &(acadoWorkspace.R1[ 9 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );

acado_zeroBlockH11( 9, 10 );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 9, 11 );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 9, 12 );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 9, 13 );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 9, 14 );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 9, 15 );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 10, 10, &(acadoWorkspace.R1[ 10 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 260 ]), &(acadoWorkspace.QE[ 260 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QE[ 404 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 460 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 520 ]) );

acado_zeroBlockH11( 10, 11 );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 10, 12 );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 10, 13 );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 10, 14 );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 10, 15 );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 11, 11, &(acadoWorkspace.R1[ 11 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 308 ]), &(acadoWorkspace.QE[ 308 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.QE[ 356 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 524 ]) );

acado_zeroBlockH11( 11, 12 );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 11, 13 );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 11, 14 );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 11, 15 );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 12, 12, &(acadoWorkspace.R1[ 12 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.QE[ 412 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );

acado_zeroBlockH11( 12, 13 );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 12, 14 );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 12, 15 );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 13, 13, &(acadoWorkspace.R1[ 13 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.QE[ 532 ]) );

acado_zeroBlockH11( 13, 14 );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 13, 15 );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 14, 14, &(acadoWorkspace.R1[ 14 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 476 ]), &(acadoWorkspace.QE[ 476 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 536 ]) );

acado_zeroBlockH11( 14, 15 );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 540 ]) );

acado_setBlockH11_R1( 15, 15, &(acadoWorkspace.R1[ 15 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );


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

acadoWorkspace.H[80] = acadoWorkspace.H10[0];
acadoWorkspace.H[81] = acadoWorkspace.H10[1];
acadoWorkspace.H[82] = acadoWorkspace.H10[2];
acadoWorkspace.H[83] = acadoWorkspace.H10[3];
acadoWorkspace.H[100] = acadoWorkspace.H10[4];
acadoWorkspace.H[101] = acadoWorkspace.H10[5];
acadoWorkspace.H[102] = acadoWorkspace.H10[6];
acadoWorkspace.H[103] = acadoWorkspace.H10[7];
acadoWorkspace.H[120] = acadoWorkspace.H10[8];
acadoWorkspace.H[121] = acadoWorkspace.H10[9];
acadoWorkspace.H[122] = acadoWorkspace.H10[10];
acadoWorkspace.H[123] = acadoWorkspace.H10[11];
acadoWorkspace.H[140] = acadoWorkspace.H10[12];
acadoWorkspace.H[141] = acadoWorkspace.H10[13];
acadoWorkspace.H[142] = acadoWorkspace.H10[14];
acadoWorkspace.H[143] = acadoWorkspace.H10[15];
acadoWorkspace.H[160] = acadoWorkspace.H10[16];
acadoWorkspace.H[161] = acadoWorkspace.H10[17];
acadoWorkspace.H[162] = acadoWorkspace.H10[18];
acadoWorkspace.H[163] = acadoWorkspace.H10[19];
acadoWorkspace.H[180] = acadoWorkspace.H10[20];
acadoWorkspace.H[181] = acadoWorkspace.H10[21];
acadoWorkspace.H[182] = acadoWorkspace.H10[22];
acadoWorkspace.H[183] = acadoWorkspace.H10[23];
acadoWorkspace.H[200] = acadoWorkspace.H10[24];
acadoWorkspace.H[201] = acadoWorkspace.H10[25];
acadoWorkspace.H[202] = acadoWorkspace.H10[26];
acadoWorkspace.H[203] = acadoWorkspace.H10[27];
acadoWorkspace.H[220] = acadoWorkspace.H10[28];
acadoWorkspace.H[221] = acadoWorkspace.H10[29];
acadoWorkspace.H[222] = acadoWorkspace.H10[30];
acadoWorkspace.H[223] = acadoWorkspace.H10[31];
acadoWorkspace.H[240] = acadoWorkspace.H10[32];
acadoWorkspace.H[241] = acadoWorkspace.H10[33];
acadoWorkspace.H[242] = acadoWorkspace.H10[34];
acadoWorkspace.H[243] = acadoWorkspace.H10[35];
acadoWorkspace.H[260] = acadoWorkspace.H10[36];
acadoWorkspace.H[261] = acadoWorkspace.H10[37];
acadoWorkspace.H[262] = acadoWorkspace.H10[38];
acadoWorkspace.H[263] = acadoWorkspace.H10[39];
acadoWorkspace.H[280] = acadoWorkspace.H10[40];
acadoWorkspace.H[281] = acadoWorkspace.H10[41];
acadoWorkspace.H[282] = acadoWorkspace.H10[42];
acadoWorkspace.H[283] = acadoWorkspace.H10[43];
acadoWorkspace.H[300] = acadoWorkspace.H10[44];
acadoWorkspace.H[301] = acadoWorkspace.H10[45];
acadoWorkspace.H[302] = acadoWorkspace.H10[46];
acadoWorkspace.H[303] = acadoWorkspace.H10[47];
acadoWorkspace.H[320] = acadoWorkspace.H10[48];
acadoWorkspace.H[321] = acadoWorkspace.H10[49];
acadoWorkspace.H[322] = acadoWorkspace.H10[50];
acadoWorkspace.H[323] = acadoWorkspace.H10[51];
acadoWorkspace.H[340] = acadoWorkspace.H10[52];
acadoWorkspace.H[341] = acadoWorkspace.H10[53];
acadoWorkspace.H[342] = acadoWorkspace.H10[54];
acadoWorkspace.H[343] = acadoWorkspace.H10[55];
acadoWorkspace.H[360] = acadoWorkspace.H10[56];
acadoWorkspace.H[361] = acadoWorkspace.H10[57];
acadoWorkspace.H[362] = acadoWorkspace.H10[58];
acadoWorkspace.H[363] = acadoWorkspace.H10[59];
acadoWorkspace.H[380] = acadoWorkspace.H10[60];
acadoWorkspace.H[381] = acadoWorkspace.H10[61];
acadoWorkspace.H[382] = acadoWorkspace.H10[62];
acadoWorkspace.H[383] = acadoWorkspace.H10[63];

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
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.Qd[ 60 ]) );

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
acado_macETSlu( acadoWorkspace.QE, &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 4 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 40 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 112 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 220 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 364 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 8 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 16 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 28 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 44 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 64 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 88 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 116 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 148 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 184 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 224 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 268 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 316 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 368 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 424 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 484 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 20 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 32 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 68 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 92 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 152 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 188 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 272 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 320 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 428 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 488 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 52 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 124 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 232 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 376 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 56 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 76 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 100 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 128 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 160 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 196 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 236 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 280 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 328 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 380 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 436 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 496 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 80 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 104 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 164 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 200 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 284 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 332 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 440 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 500 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 136 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 244 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 388 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 140 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 172 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 208 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 248 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 292 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 340 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 392 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 448 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 508 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 176 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 212 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 296 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 344 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 452 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 512 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 256 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 400 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 260 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 304 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 352 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 404 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 460 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 520 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 308 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 356 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 464 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 524 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 412 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 416 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 472 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 532 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 476 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 536 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.g[ 19 ]) );
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

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 4;
lRun4 = ((lRun3) / (4)) + (1);
acadoWorkspace.A[lRun1 * 20] = acadoWorkspace.evGx[lRun3 * 4];
acadoWorkspace.A[lRun1 * 20 + 1] = acadoWorkspace.evGx[lRun3 * 4 + 1];
acadoWorkspace.A[lRun1 * 20 + 2] = acadoWorkspace.evGx[lRun3 * 4 + 2];
acadoWorkspace.A[lRun1 * 20 + 3] = acadoWorkspace.evGx[lRun3 * 4 + 3];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (4)) + ((lRun3) % (4));
acadoWorkspace.A[(lRun1 * 20) + (lRun2 + 4)] = acadoWorkspace.E[lRun5];
}
}

}

void acado_condenseFdb(  )
{
real_t tmp;

acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];

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
acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 3 ]), &(acadoWorkspace.Dy[ 3 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 6 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 9 ]), &(acadoWorkspace.Dy[ 9 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 12 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 15 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 18 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 21 ]), &(acadoWorkspace.Dy[ 21 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 24 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 27 ]), &(acadoWorkspace.Dy[ 27 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 30 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 33 ]), &(acadoWorkspace.Dy[ 33 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 36 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 39 ]), &(acadoWorkspace.Dy[ 39 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 42 ]), &(acadoWorkspace.Dy[ 42 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 45 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.g[ 19 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 12 ]), &(acadoWorkspace.Dy[ 3 ]), &(acadoWorkspace.QDy[ 4 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 24 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.QDy[ 8 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 36 ]), &(acadoWorkspace.Dy[ 9 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 48 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 16 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 60 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.QDy[ 20 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 72 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.QDy[ 24 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 84 ]), &(acadoWorkspace.Dy[ 21 ]), &(acadoWorkspace.QDy[ 28 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 96 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 32 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 108 ]), &(acadoWorkspace.Dy[ 27 ]), &(acadoWorkspace.QDy[ 36 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 120 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.QDy[ 40 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 132 ]), &(acadoWorkspace.Dy[ 33 ]), &(acadoWorkspace.QDy[ 44 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 144 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.QDy[ 48 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 156 ]), &(acadoWorkspace.Dy[ 39 ]), &(acadoWorkspace.QDy[ 52 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 168 ]), &(acadoWorkspace.Dy[ 42 ]), &(acadoWorkspace.QDy[ 56 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 180 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.QDy[ 60 ]) );

acadoWorkspace.QDy[64] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1];
acadoWorkspace.QDy[65] = + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[1];
acadoWorkspace.QDy[66] = + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[1];
acadoWorkspace.QDy[67] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1];

acadoWorkspace.QDy[4] += acadoWorkspace.Qd[0];
acadoWorkspace.QDy[5] += acadoWorkspace.Qd[1];
acadoWorkspace.QDy[6] += acadoWorkspace.Qd[2];
acadoWorkspace.QDy[7] += acadoWorkspace.Qd[3];
acadoWorkspace.QDy[8] += acadoWorkspace.Qd[4];
acadoWorkspace.QDy[9] += acadoWorkspace.Qd[5];
acadoWorkspace.QDy[10] += acadoWorkspace.Qd[6];
acadoWorkspace.QDy[11] += acadoWorkspace.Qd[7];
acadoWorkspace.QDy[12] += acadoWorkspace.Qd[8];
acadoWorkspace.QDy[13] += acadoWorkspace.Qd[9];
acadoWorkspace.QDy[14] += acadoWorkspace.Qd[10];
acadoWorkspace.QDy[15] += acadoWorkspace.Qd[11];
acadoWorkspace.QDy[16] += acadoWorkspace.Qd[12];
acadoWorkspace.QDy[17] += acadoWorkspace.Qd[13];
acadoWorkspace.QDy[18] += acadoWorkspace.Qd[14];
acadoWorkspace.QDy[19] += acadoWorkspace.Qd[15];
acadoWorkspace.QDy[20] += acadoWorkspace.Qd[16];
acadoWorkspace.QDy[21] += acadoWorkspace.Qd[17];
acadoWorkspace.QDy[22] += acadoWorkspace.Qd[18];
acadoWorkspace.QDy[23] += acadoWorkspace.Qd[19];
acadoWorkspace.QDy[24] += acadoWorkspace.Qd[20];
acadoWorkspace.QDy[25] += acadoWorkspace.Qd[21];
acadoWorkspace.QDy[26] += acadoWorkspace.Qd[22];
acadoWorkspace.QDy[27] += acadoWorkspace.Qd[23];
acadoWorkspace.QDy[28] += acadoWorkspace.Qd[24];
acadoWorkspace.QDy[29] += acadoWorkspace.Qd[25];
acadoWorkspace.QDy[30] += acadoWorkspace.Qd[26];
acadoWorkspace.QDy[31] += acadoWorkspace.Qd[27];
acadoWorkspace.QDy[32] += acadoWorkspace.Qd[28];
acadoWorkspace.QDy[33] += acadoWorkspace.Qd[29];
acadoWorkspace.QDy[34] += acadoWorkspace.Qd[30];
acadoWorkspace.QDy[35] += acadoWorkspace.Qd[31];
acadoWorkspace.QDy[36] += acadoWorkspace.Qd[32];
acadoWorkspace.QDy[37] += acadoWorkspace.Qd[33];
acadoWorkspace.QDy[38] += acadoWorkspace.Qd[34];
acadoWorkspace.QDy[39] += acadoWorkspace.Qd[35];
acadoWorkspace.QDy[40] += acadoWorkspace.Qd[36];
acadoWorkspace.QDy[41] += acadoWorkspace.Qd[37];
acadoWorkspace.QDy[42] += acadoWorkspace.Qd[38];
acadoWorkspace.QDy[43] += acadoWorkspace.Qd[39];
acadoWorkspace.QDy[44] += acadoWorkspace.Qd[40];
acadoWorkspace.QDy[45] += acadoWorkspace.Qd[41];
acadoWorkspace.QDy[46] += acadoWorkspace.Qd[42];
acadoWorkspace.QDy[47] += acadoWorkspace.Qd[43];
acadoWorkspace.QDy[48] += acadoWorkspace.Qd[44];
acadoWorkspace.QDy[49] += acadoWorkspace.Qd[45];
acadoWorkspace.QDy[50] += acadoWorkspace.Qd[46];
acadoWorkspace.QDy[51] += acadoWorkspace.Qd[47];
acadoWorkspace.QDy[52] += acadoWorkspace.Qd[48];
acadoWorkspace.QDy[53] += acadoWorkspace.Qd[49];
acadoWorkspace.QDy[54] += acadoWorkspace.Qd[50];
acadoWorkspace.QDy[55] += acadoWorkspace.Qd[51];
acadoWorkspace.QDy[56] += acadoWorkspace.Qd[52];
acadoWorkspace.QDy[57] += acadoWorkspace.Qd[53];
acadoWorkspace.QDy[58] += acadoWorkspace.Qd[54];
acadoWorkspace.QDy[59] += acadoWorkspace.Qd[55];
acadoWorkspace.QDy[60] += acadoWorkspace.Qd[56];
acadoWorkspace.QDy[61] += acadoWorkspace.Qd[57];
acadoWorkspace.QDy[62] += acadoWorkspace.Qd[58];
acadoWorkspace.QDy[63] += acadoWorkspace.Qd[59];
acadoWorkspace.QDy[64] += acadoWorkspace.Qd[60];
acadoWorkspace.QDy[65] += acadoWorkspace.Qd[61];
acadoWorkspace.QDy[66] += acadoWorkspace.Qd[62];
acadoWorkspace.QDy[67] += acadoWorkspace.Qd[63];

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[67];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[67];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[67];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[67];


acado_multEQDy( acadoWorkspace.E, &(acadoWorkspace.QDy[ 4 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.QDy[ 8 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QDy[ 8 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 20 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 140 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 260 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 308 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 476 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 19 ]) );

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.lb[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[3] = acadoWorkspace.Dx0[3];
tmp = acadoVariables.x[6] + acadoWorkspace.d[2];
acadoWorkspace.lbA[0] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[0] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[7] + acadoWorkspace.d[3];
acadoWorkspace.lbA[1] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[1] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[10] + acadoWorkspace.d[6];
acadoWorkspace.lbA[2] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[2] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[11] + acadoWorkspace.d[7];
acadoWorkspace.lbA[3] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[3] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[14] + acadoWorkspace.d[10];
acadoWorkspace.lbA[4] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[4] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[15] + acadoWorkspace.d[11];
acadoWorkspace.lbA[5] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[5] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[18] + acadoWorkspace.d[14];
acadoWorkspace.lbA[6] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[6] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[19] + acadoWorkspace.d[15];
acadoWorkspace.lbA[7] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[7] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[22] + acadoWorkspace.d[18];
acadoWorkspace.lbA[8] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[8] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[23] + acadoWorkspace.d[19];
acadoWorkspace.lbA[9] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[9] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[26] + acadoWorkspace.d[22];
acadoWorkspace.lbA[10] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[10] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[27] + acadoWorkspace.d[23];
acadoWorkspace.lbA[11] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[11] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[30] + acadoWorkspace.d[26];
acadoWorkspace.lbA[12] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[12] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[27];
acadoWorkspace.lbA[13] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[13] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[34] + acadoWorkspace.d[30];
acadoWorkspace.lbA[14] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[14] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[35] + acadoWorkspace.d[31];
acadoWorkspace.lbA[15] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[15] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[38] + acadoWorkspace.d[34];
acadoWorkspace.lbA[16] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[16] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[39] + acadoWorkspace.d[35];
acadoWorkspace.lbA[17] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[17] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[42] + acadoWorkspace.d[38];
acadoWorkspace.lbA[18] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[18] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[43] + acadoWorkspace.d[39];
acadoWorkspace.lbA[19] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[19] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[46] + acadoWorkspace.d[42];
acadoWorkspace.lbA[20] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[20] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[47] + acadoWorkspace.d[43];
acadoWorkspace.lbA[21] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[21] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[50] + acadoWorkspace.d[46];
acadoWorkspace.lbA[22] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[22] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[51] + acadoWorkspace.d[47];
acadoWorkspace.lbA[23] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[23] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[54] + acadoWorkspace.d[50];
acadoWorkspace.lbA[24] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[24] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[55] + acadoWorkspace.d[51];
acadoWorkspace.lbA[25] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[25] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[58] + acadoWorkspace.d[54];
acadoWorkspace.lbA[26] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[26] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[59] + acadoWorkspace.d[55];
acadoWorkspace.lbA[27] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[27] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[62] + acadoWorkspace.d[58];
acadoWorkspace.lbA[28] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[28] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[63] + acadoWorkspace.d[59];
acadoWorkspace.lbA[29] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[29] = (real_t)8.7266462599716477e-01 - tmp;
tmp = acadoVariables.x[66] + acadoWorkspace.d[62];
acadoWorkspace.lbA[30] = (real_t)-1.5707963267948966e+00 - tmp;
acadoWorkspace.ubA[30] = (real_t)1.5707963267948966e+00 - tmp;
tmp = acadoVariables.x[67] + acadoWorkspace.d[63];
acadoWorkspace.lbA[31] = (real_t)-8.7266462599716477e-01 - tmp;
acadoWorkspace.ubA[31] = (real_t)8.7266462599716477e-01 - tmp;

}

void acado_expand(  )
{
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

acado_multEDu( acadoWorkspace.E, &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 4 ]) );
acado_multEDu( &(acadoWorkspace.E[ 4 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 8 ]) );
acado_multEDu( &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 8 ]) );
acado_multEDu( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 20 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 28 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 44 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 52 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 68 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 76 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 92 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 100 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 116 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 124 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 140 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 148 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 164 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 172 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 188 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 196 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 212 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 220 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 236 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 244 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 260 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 268 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 284 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 292 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 308 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 316 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 332 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 340 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 356 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 364 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 380 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 388 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 404 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 412 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 428 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 436 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 452 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 460 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 476 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 484 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 500 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 508 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 524 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 532 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 64 ]) );
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
for (index = 0; index < 16; ++index)
{
acadoWorkspace.state[0] = acadoVariables.x[index * 4];
acadoWorkspace.state[1] = acadoVariables.x[index * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 4 + 3];
acadoWorkspace.state[24] = acadoVariables.u[index];
acadoWorkspace.state[25] = acadoVariables.od[index * 2];
acadoWorkspace.state[26] = acadoVariables.od[index * 2 + 1];

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
for (index = 0; index < 16; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[64] = xEnd[0];
acadoVariables.x[65] = xEnd[1];
acadoVariables.x[66] = xEnd[2];
acadoVariables.x[67] = xEnd[3];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[64];
acadoWorkspace.state[1] = acadoVariables.x[65];
acadoWorkspace.state[2] = acadoVariables.x[66];
acadoWorkspace.state[3] = acadoVariables.x[67];
if (uEnd != 0)
{
acadoWorkspace.state[24] = uEnd[0];
}
else
{
acadoWorkspace.state[24] = acadoVariables.u[15];
}
acadoWorkspace.state[25] = acadoVariables.od[32];
acadoWorkspace.state[26] = acadoVariables.od[33];

acado_integrate(acadoWorkspace.state, 1, 15);

acadoVariables.x[64] = acadoWorkspace.state[0];
acadoVariables.x[65] = acadoWorkspace.state[1];
acadoVariables.x[66] = acadoWorkspace.state[2];
acadoVariables.x[67] = acadoWorkspace.state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 15; ++index)
{
acadoVariables.u[index] = acadoVariables.u[index + 1];
}

if (uEnd != 0)
{
acadoVariables.u[15] = uEnd[0];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19];
kkt = fabs( kkt );
for (index = 0; index < 20; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 32; ++index)
{
prd = acadoWorkspace.y[index + 20];
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
/** Row vector of size: 3 */
real_t tmpDy[ 3 ];

/** Row vector of size: 2 */
real_t tmpDyN[ 2 ];

for (lRun1 = 0; lRun1 < 16; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[lRun1];
acadoWorkspace.objValueIn[5] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.objValueIn[6] = acadoVariables.od[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 3] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 3];
acadoWorkspace.Dy[lRun1 * 3 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 3 + 1];
acadoWorkspace.Dy[lRun1 * 3 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 3 + 2];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[64];
acadoWorkspace.objValueIn[1] = acadoVariables.x[65];
acadoWorkspace.objValueIn[2] = acadoVariables.x[66];
acadoWorkspace.objValueIn[3] = acadoVariables.x[67];
acadoWorkspace.objValueIn[4] = acadoVariables.od[32];
acadoWorkspace.objValueIn[5] = acadoVariables.od[33];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 16; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 3]*acadoVariables.W[lRun1 * 9] + acadoWorkspace.Dy[lRun1 * 3 + 1]*acadoVariables.W[lRun1 * 9 + 3] + acadoWorkspace.Dy[lRun1 * 3 + 2]*acadoVariables.W[lRun1 * 9 + 6];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 3]*acadoVariables.W[lRun1 * 9 + 1] + acadoWorkspace.Dy[lRun1 * 3 + 1]*acadoVariables.W[lRun1 * 9 + 4] + acadoWorkspace.Dy[lRun1 * 3 + 2]*acadoVariables.W[lRun1 * 9 + 7];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 3]*acadoVariables.W[lRun1 * 9 + 2] + acadoWorkspace.Dy[lRun1 * 3 + 1]*acadoVariables.W[lRun1 * 9 + 5] + acadoWorkspace.Dy[lRun1 * 3 + 2]*acadoVariables.W[lRun1 * 9 + 8];
objVal += + acadoWorkspace.Dy[lRun1 * 3]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 3 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 3 + 2]*tmpDy[2];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0] + acadoWorkspace.DyN[1]*acadoVariables.WN[2];
tmpDyN[1] = + acadoWorkspace.DyN[0]*acadoVariables.WN[1] + acadoWorkspace.DyN[1]*acadoVariables.WN[3];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1];

objVal *= 0.5;
return objVal;
}

