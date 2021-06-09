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
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 3];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 3 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 3 + 2];

acadoWorkspace.state[15] = acadoVariables.u[lRun1];
acadoWorkspace.state[16] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.state[17] = acadoVariables.od[lRun1 * 2 + 1];

ret = acado_integrate(acadoWorkspace.state, 1, lRun1);

acadoWorkspace.d[lRun1 * 3] = acadoWorkspace.state[0] - acadoVariables.x[lRun1 * 3 + 3];
acadoWorkspace.d[lRun1 * 3 + 1] = acadoWorkspace.state[1] - acadoVariables.x[lRun1 * 3 + 4];
acadoWorkspace.d[lRun1 * 3 + 2] = acadoWorkspace.state[2] - acadoVariables.x[lRun1 * 3 + 5];

acadoWorkspace.evGx[lRun1 * 9] = acadoWorkspace.state[3];
acadoWorkspace.evGx[lRun1 * 9 + 1] = acadoWorkspace.state[4];
acadoWorkspace.evGx[lRun1 * 9 + 2] = acadoWorkspace.state[5];
acadoWorkspace.evGx[lRun1 * 9 + 3] = acadoWorkspace.state[6];
acadoWorkspace.evGx[lRun1 * 9 + 4] = acadoWorkspace.state[7];
acadoWorkspace.evGx[lRun1 * 9 + 5] = acadoWorkspace.state[8];
acadoWorkspace.evGx[lRun1 * 9 + 6] = acadoWorkspace.state[9];
acadoWorkspace.evGx[lRun1 * 9 + 7] = acadoWorkspace.state[10];
acadoWorkspace.evGx[lRun1 * 9 + 8] = acadoWorkspace.state[11];

acadoWorkspace.evGu[lRun1 * 3] = acadoWorkspace.state[12];
acadoWorkspace.evGu[lRun1 * 3 + 1] = acadoWorkspace.state[13];
acadoWorkspace.evGu[lRun1 * 3 + 2] = acadoWorkspace.state[14];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 3;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = u[0];
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
tmpQ1[0] = + tmpQ2[0];
tmpQ1[1] = + tmpQ2[1];
tmpQ1[2] = + tmpQ2[2];
tmpQ1[3] = + tmpQ2[4];
tmpQ1[4] = + tmpQ2[5];
tmpQ1[5] = + tmpQ2[6];
tmpQ1[6] = + tmpQ2[8];
tmpQ1[7] = + tmpQ2[9];
tmpQ1[8] = + tmpQ2[10];
}

void acado_setObjR1R2( real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = +tmpObjS[12];
tmpR2[1] = +tmpObjS[13];
tmpR2[2] = +tmpObjS[14];
tmpR2[3] = +tmpObjS[15];
tmpR1[0] = + tmpR2[3];
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
tmpQN1[0] = + tmpQN2[0];
tmpQN1[1] = + tmpQN2[1];
tmpQN1[2] = + tmpQN2[2];
tmpQN1[3] = + tmpQN2[3];
tmpQN1[4] = + tmpQN2[4];
tmpQN1[5] = + tmpQN2[5];
tmpQN1[6] = + tmpQN2[6];
tmpQN1[7] = + tmpQN2[7];
tmpQN1[8] = + tmpQN2[8];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 32; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 3];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 3 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 3 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.u[runObj];
acadoWorkspace.objValueIn[4] = acadoVariables.od[runObj * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.od[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 4] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 4 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 4 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 4 + 3] = acadoWorkspace.objValueOut[3];

acado_setObjQ1Q2( &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.Q1[ runObj * 9 ]), &(acadoWorkspace.Q2[ runObj * 12 ]) );

acado_setObjR1R2( &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.R1[ runObj ]), &(acadoWorkspace.R2[ runObj * 4 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[96];
acadoWorkspace.objValueIn[1] = acadoVariables.x[97];
acadoWorkspace.objValueIn[2] = acadoVariables.x[98];
acadoWorkspace.objValueIn[3] = acadoVariables.od[64];
acadoWorkspace.objValueIn[4] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_multGxd( real_t* const dOld, real_t* const Gx1, real_t* const dNew )
{
dNew[0] += + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2];
dNew[1] += + Gx1[3]*dOld[0] + Gx1[4]*dOld[1] + Gx1[5]*dOld[2];
dNew[2] += + Gx1[6]*dOld[0] + Gx1[7]*dOld[1] + Gx1[8]*dOld[2];
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
}

void acado_multGxGx( real_t* const Gx1, real_t* const Gx2, real_t* const Gx3 )
{
Gx3[0] = + Gx1[0]*Gx2[0] + Gx1[1]*Gx2[3] + Gx1[2]*Gx2[6];
Gx3[1] = + Gx1[0]*Gx2[1] + Gx1[1]*Gx2[4] + Gx1[2]*Gx2[7];
Gx3[2] = + Gx1[0]*Gx2[2] + Gx1[1]*Gx2[5] + Gx1[2]*Gx2[8];
Gx3[3] = + Gx1[3]*Gx2[0] + Gx1[4]*Gx2[3] + Gx1[5]*Gx2[6];
Gx3[4] = + Gx1[3]*Gx2[1] + Gx1[4]*Gx2[4] + Gx1[5]*Gx2[7];
Gx3[5] = + Gx1[3]*Gx2[2] + Gx1[4]*Gx2[5] + Gx1[5]*Gx2[8];
Gx3[6] = + Gx1[6]*Gx2[0] + Gx1[7]*Gx2[3] + Gx1[8]*Gx2[6];
Gx3[7] = + Gx1[6]*Gx2[1] + Gx1[7]*Gx2[4] + Gx1[8]*Gx2[7];
Gx3[8] = + Gx1[6]*Gx2[2] + Gx1[7]*Gx2[5] + Gx1[8]*Gx2[8];
}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[1] + Gx1[2]*Gu1[2];
Gu2[1] = + Gx1[3]*Gu1[0] + Gx1[4]*Gu1[1] + Gx1[5]*Gu1[2];
Gu2[2] = + Gx1[6]*Gu1[0] + Gx1[7]*Gu1[1] + Gx1[8]*Gu1[2];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
}

void acado_setBlockH11( int iRow, int iCol, real_t* const Gu1, real_t* const Gu2 )
{
acadoWorkspace.H[(iRow * 35 + 105) + (iCol + 3)] += + Gu1[0]*Gu2[0] + Gu1[1]*Gu2[1] + Gu1[2]*Gu2[2];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 35 + 105) + (iCol + 3)] = R11[0];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 35 + 105) + (iCol + 3)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 35 + 105) + (iCol + 3)] = acadoWorkspace.H[(iCol * 35 + 105) + (iRow + 3)];
}

void acado_multQ1d( real_t* const Gx1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2];
dNew[1] = + Gx1[3]*dOld[0] + Gx1[4]*dOld[1] + Gx1[5]*dOld[2];
dNew[2] = + Gx1[6]*dOld[0] + Gx1[7]*dOld[1] + Gx1[8]*dOld[2];
}

void acado_multQN1d( real_t* const QN1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + acadoWorkspace.QN1[0]*dOld[0] + acadoWorkspace.QN1[1]*dOld[1] + acadoWorkspace.QN1[2]*dOld[2];
dNew[1] = + acadoWorkspace.QN1[3]*dOld[0] + acadoWorkspace.QN1[4]*dOld[1] + acadoWorkspace.QN1[5]*dOld[2];
dNew[2] = + acadoWorkspace.QN1[6]*dOld[0] + acadoWorkspace.QN1[7]*dOld[1] + acadoWorkspace.QN1[8]*dOld[2];
}

void acado_multRDy( real_t* const R2, real_t* const Dy1, real_t* const RDy1 )
{
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2] + R2[3]*Dy1[3];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3];
QDy1[1] = + Q2[4]*Dy1[0] + Q2[5]*Dy1[1] + Q2[6]*Dy1[2] + Q2[7]*Dy1[3];
QDy1[2] = + Q2[8]*Dy1[0] + Q2[9]*Dy1[1] + Q2[10]*Dy1[2] + Q2[11]*Dy1[3];
}

void acado_multEQDy( real_t* const E1, real_t* const QDy1, real_t* const U1 )
{
U1[0] += + E1[0]*QDy1[0] + E1[1]*QDy1[1] + E1[2]*QDy1[2];
}

void acado_multQETGx( real_t* const E1, real_t* const Gx1, real_t* const H101 )
{
H101[0] += + E1[0]*Gx1[0] + E1[1]*Gx1[3] + E1[2]*Gx1[6];
H101[1] += + E1[0]*Gx1[1] + E1[1]*Gx1[4] + E1[2]*Gx1[7];
H101[2] += + E1[0]*Gx1[2] + E1[1]*Gx1[5] + E1[2]*Gx1[8];
}

void acado_zeroBlockH10( real_t* const H101 )
{
{ int lCopy; for (lCopy = 0; lCopy < 3; lCopy++) H101[ lCopy ] = 0; }
}

void acado_multEDu( real_t* const E1, real_t* const U1, real_t* const dNew )
{
dNew[0] += + E1[0]*U1[0];
dNew[1] += + E1[1]*U1[0];
dNew[2] += + E1[2]*U1[0];
}

void acado_zeroBlockH00(  )
{
acadoWorkspace.H[0] = 0.0000000000000000e+00;
acadoWorkspace.H[1] = 0.0000000000000000e+00;
acadoWorkspace.H[2] = 0.0000000000000000e+00;
acadoWorkspace.H[35] = 0.0000000000000000e+00;
acadoWorkspace.H[36] = 0.0000000000000000e+00;
acadoWorkspace.H[37] = 0.0000000000000000e+00;
acadoWorkspace.H[70] = 0.0000000000000000e+00;
acadoWorkspace.H[71] = 0.0000000000000000e+00;
acadoWorkspace.H[72] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[3]*Gx2[3] + Gx1[6]*Gx2[6];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[3]*Gx2[4] + Gx1[6]*Gx2[7];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[3]*Gx2[5] + Gx1[6]*Gx2[8];
acadoWorkspace.H[35] += + Gx1[1]*Gx2[0] + Gx1[4]*Gx2[3] + Gx1[7]*Gx2[6];
acadoWorkspace.H[36] += + Gx1[1]*Gx2[1] + Gx1[4]*Gx2[4] + Gx1[7]*Gx2[7];
acadoWorkspace.H[37] += + Gx1[1]*Gx2[2] + Gx1[4]*Gx2[5] + Gx1[7]*Gx2[8];
acadoWorkspace.H[70] += + Gx1[2]*Gx2[0] + Gx1[5]*Gx2[3] + Gx1[8]*Gx2[6];
acadoWorkspace.H[71] += + Gx1[2]*Gx2[1] + Gx1[5]*Gx2[4] + Gx1[8]*Gx2[7];
acadoWorkspace.H[72] += + Gx1[2]*Gx2[2] + Gx1[5]*Gx2[5] + Gx1[8]*Gx2[8];
}

void acado_multHxC( real_t* const Hx, real_t* const Gx, real_t* const A01 )
{
A01[0] = + Hx[0]*Gx[0] + Hx[1]*Gx[3] + Hx[2]*Gx[6];
A01[1] = + Hx[0]*Gx[1] + Hx[1]*Gx[4] + Hx[2]*Gx[7];
A01[2] = + Hx[0]*Gx[2] + Hx[1]*Gx[5] + Hx[2]*Gx[8];
A01[35] = + Hx[3]*Gx[0] + Hx[4]*Gx[3] + Hx[5]*Gx[6];
A01[36] = + Hx[3]*Gx[1] + Hx[4]*Gx[4] + Hx[5]*Gx[7];
A01[37] = + Hx[3]*Gx[2] + Hx[4]*Gx[5] + Hx[5]*Gx[8];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 70 + 1120) + (col + 3)] = + Hx[0]*E[0] + Hx[1]*E[1] + Hx[2]*E[2];
acadoWorkspace.A[(row * 70 + 1155) + (col + 3)] = + Hx[3]*E[0] + Hx[4]*E[1] + Hx[5]*E[2];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2];
acadoWorkspace.evHxd[1] = + Hx[3]*tmpd[0] + Hx[4]*tmpd[1] + Hx[5]*tmpd[2];
lbA[0] -= acadoWorkspace.evHxd[0];
lbA[1] -= acadoWorkspace.evHxd[1];
ubA[0] -= acadoWorkspace.evHxd[0];
ubA[1] -= acadoWorkspace.evHxd[1];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 4;
/* Vector of auxiliary variables; number of elements: 8. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (real_t)(0.0000000000000000e+00);
a[1] = (real_t)(0.0000000000000000e+00);
a[2] = (real_t)(1.0000000000000000e+00);
a[3] = (real_t)(0.0000000000000000e+00);
a[4] = (real_t)(0.0000000000000000e+00);
a[5] = (real_t)(1.0000000000000000e+00);
a[6] = (real_t)(0.0000000000000000e+00);
a[7] = (real_t)(0.0000000000000000e+00);

/* Compute outputs: */
out[0] = (xd[2]-od[0]);
out[1] = (xd[2]-od[1]);
out[2] = a[0];
out[3] = a[1];
out[4] = a[2];
out[5] = a[3];
out[6] = a[4];
out[7] = a[5];
out[8] = a[6];
out[9] = a[7];
}

void acado_macCTSlx( real_t* const C0, real_t* const g0 )
{
g0[0] += 0.0;
;
g0[1] += 0.0;
;
g0[2] += 0.0;
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
{ 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
for (lRun1 = 1; lRun1 < 32; ++lRun1)
{
acado_moveGxT( &(acadoWorkspace.evGx[ lRun1 * 9 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ lRun1 * 3-3 ]), &(acadoWorkspace.evGx[ lRun1 * 9 ]), &(acadoWorkspace.d[ lRun1 * 3 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ lRun1 * 9-9 ]), &(acadoWorkspace.evGx[ lRun1 * 9 ]) );
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
lRun4 = (((lRun1) * (lRun1-1)) / (2)) + (lRun2);
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ lRun4 * 3 ]), &(acadoWorkspace.E[ lRun3 * 3 ]) );
}
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_moveGuE( &(acadoWorkspace.evGu[ lRun1 * 3 ]), &(acadoWorkspace.E[ lRun3 * 3 ]) );
}

acado_multGxGx( &(acadoWorkspace.Q1[ 9 ]), acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multGxGx( &(acadoWorkspace.Q1[ 18 ]), &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.QGx[ 9 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 27 ]), &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.QGx[ 18 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.QGx[ 27 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.QGx[ 36 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.QGx[ 45 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.QGx[ 54 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.QGx[ 63 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.QGx[ 72 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.QGx[ 81 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.QGx[ 90 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.QGx[ 99 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.QGx[ 108 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.QGx[ 117 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.QGx[ 126 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.QGx[ 135 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.QGx[ 153 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.QGx[ 162 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.QGx[ 171 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 189 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.QGx[ 180 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 198 ]), &(acadoWorkspace.evGx[ 189 ]), &(acadoWorkspace.QGx[ 189 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 207 ]), &(acadoWorkspace.evGx[ 198 ]), &(acadoWorkspace.QGx[ 198 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.evGx[ 207 ]), &(acadoWorkspace.QGx[ 207 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.QGx[ 216 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 234 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.QGx[ 225 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 243 ]), &(acadoWorkspace.evGx[ 234 ]), &(acadoWorkspace.QGx[ 234 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.evGx[ 243 ]), &(acadoWorkspace.QGx[ 243 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 261 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.QGx[ 252 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 270 ]), &(acadoWorkspace.evGx[ 261 ]), &(acadoWorkspace.QGx[ 261 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 279 ]), &(acadoWorkspace.evGx[ 270 ]), &(acadoWorkspace.QGx[ 270 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 279 ]), &(acadoWorkspace.QGx[ 279 ]) );

for (lRun1 = 0; lRun1 < 31; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( &(acadoWorkspace.Q1[ lRun1 * 9 + 9 ]), &(acadoWorkspace.E[ lRun3 * 3 ]), &(acadoWorkspace.QE[ lRun3 * 3 ]) );
}
}

for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ lRun3 * 3 ]), &(acadoWorkspace.QE[ lRun3 * 3 ]) );
}

acado_zeroBlockH00(  );
acado_multCTQC( acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multCTQC( &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.QGx[ 9 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.QGx[ 18 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.QGx[ 27 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.QGx[ 36 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.QGx[ 45 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.QGx[ 54 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.QGx[ 63 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.QGx[ 72 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.QGx[ 81 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.QGx[ 90 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.QGx[ 99 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.QGx[ 108 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.QGx[ 117 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.QGx[ 126 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.QGx[ 135 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.QGx[ 153 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.QGx[ 162 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.QGx[ 171 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.QGx[ 180 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 189 ]), &(acadoWorkspace.QGx[ 189 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 198 ]), &(acadoWorkspace.QGx[ 198 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 207 ]), &(acadoWorkspace.QGx[ 207 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.QGx[ 216 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.QGx[ 225 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 234 ]), &(acadoWorkspace.QGx[ 234 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 243 ]), &(acadoWorkspace.QGx[ 243 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.QGx[ 252 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 261 ]), &(acadoWorkspace.QGx[ 261 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 270 ]), &(acadoWorkspace.QGx[ 270 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 279 ]), &(acadoWorkspace.QGx[ 279 ]) );

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_zeroBlockH10( &(acadoWorkspace.H10[ lRun1 * 3 ]) );
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multQETGx( &(acadoWorkspace.QE[ lRun3 * 3 ]), &(acadoWorkspace.evGx[ lRun2 * 9 ]), &(acadoWorkspace.H10[ lRun1 * 3 ]) );
}
}

acadoWorkspace.H[3] = acadoWorkspace.H10[0];
acadoWorkspace.H[4] = acadoWorkspace.H10[3];
acadoWorkspace.H[5] = acadoWorkspace.H10[6];
acadoWorkspace.H[6] = acadoWorkspace.H10[9];
acadoWorkspace.H[7] = acadoWorkspace.H10[12];
acadoWorkspace.H[8] = acadoWorkspace.H10[15];
acadoWorkspace.H[9] = acadoWorkspace.H10[18];
acadoWorkspace.H[10] = acadoWorkspace.H10[21];
acadoWorkspace.H[11] = acadoWorkspace.H10[24];
acadoWorkspace.H[12] = acadoWorkspace.H10[27];
acadoWorkspace.H[13] = acadoWorkspace.H10[30];
acadoWorkspace.H[14] = acadoWorkspace.H10[33];
acadoWorkspace.H[15] = acadoWorkspace.H10[36];
acadoWorkspace.H[16] = acadoWorkspace.H10[39];
acadoWorkspace.H[17] = acadoWorkspace.H10[42];
acadoWorkspace.H[18] = acadoWorkspace.H10[45];
acadoWorkspace.H[19] = acadoWorkspace.H10[48];
acadoWorkspace.H[20] = acadoWorkspace.H10[51];
acadoWorkspace.H[21] = acadoWorkspace.H10[54];
acadoWorkspace.H[22] = acadoWorkspace.H10[57];
acadoWorkspace.H[23] = acadoWorkspace.H10[60];
acadoWorkspace.H[24] = acadoWorkspace.H10[63];
acadoWorkspace.H[25] = acadoWorkspace.H10[66];
acadoWorkspace.H[26] = acadoWorkspace.H10[69];
acadoWorkspace.H[27] = acadoWorkspace.H10[72];
acadoWorkspace.H[28] = acadoWorkspace.H10[75];
acadoWorkspace.H[29] = acadoWorkspace.H10[78];
acadoWorkspace.H[30] = acadoWorkspace.H10[81];
acadoWorkspace.H[31] = acadoWorkspace.H10[84];
acadoWorkspace.H[32] = acadoWorkspace.H10[87];
acadoWorkspace.H[33] = acadoWorkspace.H10[90];
acadoWorkspace.H[34] = acadoWorkspace.H10[93];
acadoWorkspace.H[38] = acadoWorkspace.H10[1];
acadoWorkspace.H[39] = acadoWorkspace.H10[4];
acadoWorkspace.H[40] = acadoWorkspace.H10[7];
acadoWorkspace.H[41] = acadoWorkspace.H10[10];
acadoWorkspace.H[42] = acadoWorkspace.H10[13];
acadoWorkspace.H[43] = acadoWorkspace.H10[16];
acadoWorkspace.H[44] = acadoWorkspace.H10[19];
acadoWorkspace.H[45] = acadoWorkspace.H10[22];
acadoWorkspace.H[46] = acadoWorkspace.H10[25];
acadoWorkspace.H[47] = acadoWorkspace.H10[28];
acadoWorkspace.H[48] = acadoWorkspace.H10[31];
acadoWorkspace.H[49] = acadoWorkspace.H10[34];
acadoWorkspace.H[50] = acadoWorkspace.H10[37];
acadoWorkspace.H[51] = acadoWorkspace.H10[40];
acadoWorkspace.H[52] = acadoWorkspace.H10[43];
acadoWorkspace.H[53] = acadoWorkspace.H10[46];
acadoWorkspace.H[54] = acadoWorkspace.H10[49];
acadoWorkspace.H[55] = acadoWorkspace.H10[52];
acadoWorkspace.H[56] = acadoWorkspace.H10[55];
acadoWorkspace.H[57] = acadoWorkspace.H10[58];
acadoWorkspace.H[58] = acadoWorkspace.H10[61];
acadoWorkspace.H[59] = acadoWorkspace.H10[64];
acadoWorkspace.H[60] = acadoWorkspace.H10[67];
acadoWorkspace.H[61] = acadoWorkspace.H10[70];
acadoWorkspace.H[62] = acadoWorkspace.H10[73];
acadoWorkspace.H[63] = acadoWorkspace.H10[76];
acadoWorkspace.H[64] = acadoWorkspace.H10[79];
acadoWorkspace.H[65] = acadoWorkspace.H10[82];
acadoWorkspace.H[66] = acadoWorkspace.H10[85];
acadoWorkspace.H[67] = acadoWorkspace.H10[88];
acadoWorkspace.H[68] = acadoWorkspace.H10[91];
acadoWorkspace.H[69] = acadoWorkspace.H10[94];
acadoWorkspace.H[73] = acadoWorkspace.H10[2];
acadoWorkspace.H[74] = acadoWorkspace.H10[5];
acadoWorkspace.H[75] = acadoWorkspace.H10[8];
acadoWorkspace.H[76] = acadoWorkspace.H10[11];
acadoWorkspace.H[77] = acadoWorkspace.H10[14];
acadoWorkspace.H[78] = acadoWorkspace.H10[17];
acadoWorkspace.H[79] = acadoWorkspace.H10[20];
acadoWorkspace.H[80] = acadoWorkspace.H10[23];
acadoWorkspace.H[81] = acadoWorkspace.H10[26];
acadoWorkspace.H[82] = acadoWorkspace.H10[29];
acadoWorkspace.H[83] = acadoWorkspace.H10[32];
acadoWorkspace.H[84] = acadoWorkspace.H10[35];
acadoWorkspace.H[85] = acadoWorkspace.H10[38];
acadoWorkspace.H[86] = acadoWorkspace.H10[41];
acadoWorkspace.H[87] = acadoWorkspace.H10[44];
acadoWorkspace.H[88] = acadoWorkspace.H10[47];
acadoWorkspace.H[89] = acadoWorkspace.H10[50];
acadoWorkspace.H[90] = acadoWorkspace.H10[53];
acadoWorkspace.H[91] = acadoWorkspace.H10[56];
acadoWorkspace.H[92] = acadoWorkspace.H10[59];
acadoWorkspace.H[93] = acadoWorkspace.H10[62];
acadoWorkspace.H[94] = acadoWorkspace.H10[65];
acadoWorkspace.H[95] = acadoWorkspace.H10[68];
acadoWorkspace.H[96] = acadoWorkspace.H10[71];
acadoWorkspace.H[97] = acadoWorkspace.H10[74];
acadoWorkspace.H[98] = acadoWorkspace.H10[77];
acadoWorkspace.H[99] = acadoWorkspace.H10[80];
acadoWorkspace.H[100] = acadoWorkspace.H10[83];
acadoWorkspace.H[101] = acadoWorkspace.H10[86];
acadoWorkspace.H[102] = acadoWorkspace.H10[89];
acadoWorkspace.H[103] = acadoWorkspace.H10[92];
acadoWorkspace.H[104] = acadoWorkspace.H10[95];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acado_setBlockH11_R1( lRun1, lRun1, &(acadoWorkspace.R1[ lRun1 ]) );
lRun2 = lRun1;
for (lRun3 = lRun1; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 3 ]), &(acadoWorkspace.QE[ lRun5 * 3 ]) );
}
for (lRun2 = lRun1 + 1; lRun2 < 32; ++lRun2)
{
acado_zeroBlockH11( lRun1, lRun2 );
for (lRun3 = lRun2; lRun3 < 32; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 3 ]), &(acadoWorkspace.QE[ lRun5 * 3 ]) );
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

acadoWorkspace.H[105] = acadoWorkspace.H10[0];
acadoWorkspace.H[106] = acadoWorkspace.H10[1];
acadoWorkspace.H[107] = acadoWorkspace.H10[2];
acadoWorkspace.H[140] = acadoWorkspace.H10[3];
acadoWorkspace.H[141] = acadoWorkspace.H10[4];
acadoWorkspace.H[142] = acadoWorkspace.H10[5];
acadoWorkspace.H[175] = acadoWorkspace.H10[6];
acadoWorkspace.H[176] = acadoWorkspace.H10[7];
acadoWorkspace.H[177] = acadoWorkspace.H10[8];
acadoWorkspace.H[210] = acadoWorkspace.H10[9];
acadoWorkspace.H[211] = acadoWorkspace.H10[10];
acadoWorkspace.H[212] = acadoWorkspace.H10[11];
acadoWorkspace.H[245] = acadoWorkspace.H10[12];
acadoWorkspace.H[246] = acadoWorkspace.H10[13];
acadoWorkspace.H[247] = acadoWorkspace.H10[14];
acadoWorkspace.H[280] = acadoWorkspace.H10[15];
acadoWorkspace.H[281] = acadoWorkspace.H10[16];
acadoWorkspace.H[282] = acadoWorkspace.H10[17];
acadoWorkspace.H[315] = acadoWorkspace.H10[18];
acadoWorkspace.H[316] = acadoWorkspace.H10[19];
acadoWorkspace.H[317] = acadoWorkspace.H10[20];
acadoWorkspace.H[350] = acadoWorkspace.H10[21];
acadoWorkspace.H[351] = acadoWorkspace.H10[22];
acadoWorkspace.H[352] = acadoWorkspace.H10[23];
acadoWorkspace.H[385] = acadoWorkspace.H10[24];
acadoWorkspace.H[386] = acadoWorkspace.H10[25];
acadoWorkspace.H[387] = acadoWorkspace.H10[26];
acadoWorkspace.H[420] = acadoWorkspace.H10[27];
acadoWorkspace.H[421] = acadoWorkspace.H10[28];
acadoWorkspace.H[422] = acadoWorkspace.H10[29];
acadoWorkspace.H[455] = acadoWorkspace.H10[30];
acadoWorkspace.H[456] = acadoWorkspace.H10[31];
acadoWorkspace.H[457] = acadoWorkspace.H10[32];
acadoWorkspace.H[490] = acadoWorkspace.H10[33];
acadoWorkspace.H[491] = acadoWorkspace.H10[34];
acadoWorkspace.H[492] = acadoWorkspace.H10[35];
acadoWorkspace.H[525] = acadoWorkspace.H10[36];
acadoWorkspace.H[526] = acadoWorkspace.H10[37];
acadoWorkspace.H[527] = acadoWorkspace.H10[38];
acadoWorkspace.H[560] = acadoWorkspace.H10[39];
acadoWorkspace.H[561] = acadoWorkspace.H10[40];
acadoWorkspace.H[562] = acadoWorkspace.H10[41];
acadoWorkspace.H[595] = acadoWorkspace.H10[42];
acadoWorkspace.H[596] = acadoWorkspace.H10[43];
acadoWorkspace.H[597] = acadoWorkspace.H10[44];
acadoWorkspace.H[630] = acadoWorkspace.H10[45];
acadoWorkspace.H[631] = acadoWorkspace.H10[46];
acadoWorkspace.H[632] = acadoWorkspace.H10[47];
acadoWorkspace.H[665] = acadoWorkspace.H10[48];
acadoWorkspace.H[666] = acadoWorkspace.H10[49];
acadoWorkspace.H[667] = acadoWorkspace.H10[50];
acadoWorkspace.H[700] = acadoWorkspace.H10[51];
acadoWorkspace.H[701] = acadoWorkspace.H10[52];
acadoWorkspace.H[702] = acadoWorkspace.H10[53];
acadoWorkspace.H[735] = acadoWorkspace.H10[54];
acadoWorkspace.H[736] = acadoWorkspace.H10[55];
acadoWorkspace.H[737] = acadoWorkspace.H10[56];
acadoWorkspace.H[770] = acadoWorkspace.H10[57];
acadoWorkspace.H[771] = acadoWorkspace.H10[58];
acadoWorkspace.H[772] = acadoWorkspace.H10[59];
acadoWorkspace.H[805] = acadoWorkspace.H10[60];
acadoWorkspace.H[806] = acadoWorkspace.H10[61];
acadoWorkspace.H[807] = acadoWorkspace.H10[62];
acadoWorkspace.H[840] = acadoWorkspace.H10[63];
acadoWorkspace.H[841] = acadoWorkspace.H10[64];
acadoWorkspace.H[842] = acadoWorkspace.H10[65];
acadoWorkspace.H[875] = acadoWorkspace.H10[66];
acadoWorkspace.H[876] = acadoWorkspace.H10[67];
acadoWorkspace.H[877] = acadoWorkspace.H10[68];
acadoWorkspace.H[910] = acadoWorkspace.H10[69];
acadoWorkspace.H[911] = acadoWorkspace.H10[70];
acadoWorkspace.H[912] = acadoWorkspace.H10[71];
acadoWorkspace.H[945] = acadoWorkspace.H10[72];
acadoWorkspace.H[946] = acadoWorkspace.H10[73];
acadoWorkspace.H[947] = acadoWorkspace.H10[74];
acadoWorkspace.H[980] = acadoWorkspace.H10[75];
acadoWorkspace.H[981] = acadoWorkspace.H10[76];
acadoWorkspace.H[982] = acadoWorkspace.H10[77];
acadoWorkspace.H[1015] = acadoWorkspace.H10[78];
acadoWorkspace.H[1016] = acadoWorkspace.H10[79];
acadoWorkspace.H[1017] = acadoWorkspace.H10[80];
acadoWorkspace.H[1050] = acadoWorkspace.H10[81];
acadoWorkspace.H[1051] = acadoWorkspace.H10[82];
acadoWorkspace.H[1052] = acadoWorkspace.H10[83];
acadoWorkspace.H[1085] = acadoWorkspace.H10[84];
acadoWorkspace.H[1086] = acadoWorkspace.H10[85];
acadoWorkspace.H[1087] = acadoWorkspace.H10[86];
acadoWorkspace.H[1120] = acadoWorkspace.H10[87];
acadoWorkspace.H[1121] = acadoWorkspace.H10[88];
acadoWorkspace.H[1122] = acadoWorkspace.H10[89];
acadoWorkspace.H[1155] = acadoWorkspace.H10[90];
acadoWorkspace.H[1156] = acadoWorkspace.H10[91];
acadoWorkspace.H[1157] = acadoWorkspace.H10[92];
acadoWorkspace.H[1190] = acadoWorkspace.H10[93];
acadoWorkspace.H[1191] = acadoWorkspace.H10[94];
acadoWorkspace.H[1192] = acadoWorkspace.H10[95];

acado_multQ1d( &(acadoWorkspace.Q1[ 9 ]), acadoWorkspace.d, acadoWorkspace.Qd );
acado_multQ1d( &(acadoWorkspace.Q1[ 18 ]), &(acadoWorkspace.d[ 3 ]), &(acadoWorkspace.Qd[ 3 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 27 ]), &(acadoWorkspace.d[ 6 ]), &(acadoWorkspace.Qd[ 6 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.d[ 9 ]), &(acadoWorkspace.Qd[ 9 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.Qd[ 12 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.Qd[ 15 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.d[ 18 ]), &(acadoWorkspace.Qd[ 18 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.d[ 21 ]), &(acadoWorkspace.Qd[ 21 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.Qd[ 24 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.d[ 27 ]), &(acadoWorkspace.Qd[ 27 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.Qd[ 30 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.d[ 33 ]), &(acadoWorkspace.Qd[ 33 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.Qd[ 36 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.d[ 39 ]), &(acadoWorkspace.Qd[ 39 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.d[ 42 ]), &(acadoWorkspace.Qd[ 42 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.Qd[ 45 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.Qd[ 48 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.d[ 51 ]), &(acadoWorkspace.Qd[ 51 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.d[ 54 ]), &(acadoWorkspace.Qd[ 54 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.d[ 57 ]), &(acadoWorkspace.Qd[ 57 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 189 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.Qd[ 60 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 198 ]), &(acadoWorkspace.d[ 63 ]), &(acadoWorkspace.Qd[ 63 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 207 ]), &(acadoWorkspace.d[ 66 ]), &(acadoWorkspace.Qd[ 66 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.d[ 69 ]), &(acadoWorkspace.Qd[ 69 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 225 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.Qd[ 72 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 234 ]), &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.Qd[ 75 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 243 ]), &(acadoWorkspace.d[ 78 ]), &(acadoWorkspace.Qd[ 78 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.d[ 81 ]), &(acadoWorkspace.Qd[ 81 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 261 ]), &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.Qd[ 84 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 270 ]), &(acadoWorkspace.d[ 87 ]), &(acadoWorkspace.Qd[ 87 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 279 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.Qd[ 90 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 93 ]), &(acadoWorkspace.Qd[ 93 ]) );

acado_macCTSlx( acadoWorkspace.evGx, acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 9 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 18 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 27 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 45 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 54 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 63 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 81 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 90 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 99 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 117 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 126 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 135 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 153 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 162 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 171 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 180 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 189 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 198 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 207 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 216 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 225 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 234 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 243 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 252 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 261 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 270 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 279 ]), acadoWorkspace.g );
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_macETSlu( &(acadoWorkspace.QE[ lRun3 * 3 ]), &(acadoWorkspace.g[ lRun1 + 3 ]) );
}
}
acadoWorkspace.lb[3] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.lb[4] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.lb[5] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.lb[6] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.lb[10] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.lb[11] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.lb[12] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.lb[13] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.lb[14] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.lb[15] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.lb[16] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.lb[17] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.lb[18] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.lb[19] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.lb[20] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.lb[21] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.lb[22] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.lb[23] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.lb[24] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.lb[25] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.lb[26] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.lb[27] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.lb[28] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.lb[29] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.lb[30] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.lb[31] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.lb[32] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.lb[33] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.lb[34] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[31];
acadoWorkspace.ub[3] = (real_t)1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.ub[4] = (real_t)1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.ub[5] = (real_t)1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.ub[6] = (real_t)1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.ub[7] = (real_t)1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.ub[8] = (real_t)1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.ub[9] = (real_t)1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.ub[10] = (real_t)1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.ub[11] = (real_t)1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.ub[12] = (real_t)1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.ub[13] = (real_t)1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.ub[14] = (real_t)1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.ub[15] = (real_t)1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.ub[16] = (real_t)1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.ub[17] = (real_t)1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.ub[18] = (real_t)1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.ub[19] = (real_t)1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.ub[20] = (real_t)1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.ub[21] = (real_t)1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.ub[22] = (real_t)1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.ub[23] = (real_t)1.0000000000000000e+12 - acadoVariables.u[20];
acadoWorkspace.ub[24] = (real_t)1.0000000000000000e+12 - acadoVariables.u[21];
acadoWorkspace.ub[25] = (real_t)1.0000000000000000e+12 - acadoVariables.u[22];
acadoWorkspace.ub[26] = (real_t)1.0000000000000000e+12 - acadoVariables.u[23];
acadoWorkspace.ub[27] = (real_t)1.0000000000000000e+12 - acadoVariables.u[24];
acadoWorkspace.ub[28] = (real_t)1.0000000000000000e+12 - acadoVariables.u[25];
acadoWorkspace.ub[29] = (real_t)1.0000000000000000e+12 - acadoVariables.u[26];
acadoWorkspace.ub[30] = (real_t)1.0000000000000000e+12 - acadoVariables.u[27];
acadoWorkspace.ub[31] = (real_t)1.0000000000000000e+12 - acadoVariables.u[28];
acadoWorkspace.ub[32] = (real_t)1.0000000000000000e+12 - acadoVariables.u[29];
acadoWorkspace.ub[33] = (real_t)1.0000000000000000e+12 - acadoVariables.u[30];
acadoWorkspace.ub[34] = (real_t)1.0000000000000000e+12 - acadoVariables.u[31];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 3;
lRun4 = ((lRun3) / (3)) + (1);
acadoWorkspace.A[lRun1 * 35] = acadoWorkspace.evGx[lRun3 * 3];
acadoWorkspace.A[lRun1 * 35 + 1] = acadoWorkspace.evGx[lRun3 * 3 + 1];
acadoWorkspace.A[lRun1 * 35 + 2] = acadoWorkspace.evGx[lRun3 * 3 + 2];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (3)) + ((lRun3) % (3));
acadoWorkspace.A[(lRun1 * 35) + (lRun2 + 3)] = acadoWorkspace.E[lRun5];
}
}

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acadoWorkspace.conValueIn[0] = acadoVariables.x[lRun1 * 3];
acadoWorkspace.conValueIn[1] = acadoVariables.x[lRun1 * 3 + 1];
acadoWorkspace.conValueIn[2] = acadoVariables.x[lRun1 * 3 + 2];
acadoWorkspace.conValueIn[3] = acadoVariables.u[lRun1];
acadoWorkspace.conValueIn[4] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.conValueIn[5] = acadoVariables.od[lRun1 * 2 + 1];
acado_evaluatePathConstraints( acadoWorkspace.conValueIn, acadoWorkspace.conValueOut );
acadoWorkspace.evH[lRun1 * 2] = acadoWorkspace.conValueOut[0];
acadoWorkspace.evH[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[1];

acadoWorkspace.evHx[lRun1 * 6] = acadoWorkspace.conValueOut[2];
acadoWorkspace.evHx[lRun1 * 6 + 1] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 6 + 2] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHx[lRun1 * 6 + 3] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHx[lRun1 * 6 + 4] = acadoWorkspace.conValueOut[6];
acadoWorkspace.evHx[lRun1 * 6 + 5] = acadoWorkspace.conValueOut[7];
acadoWorkspace.evHu[lRun1 * 2] = acadoWorkspace.conValueOut[8];
acadoWorkspace.evHu[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[9];
}

acadoWorkspace.A[1120] = acadoWorkspace.evHx[0];
acadoWorkspace.A[1121] = acadoWorkspace.evHx[1];
acadoWorkspace.A[1122] = acadoWorkspace.evHx[2];
acadoWorkspace.A[1155] = acadoWorkspace.evHx[3];
acadoWorkspace.A[1156] = acadoWorkspace.evHx[4];
acadoWorkspace.A[1157] = acadoWorkspace.evHx[5];

acado_multHxC( &(acadoWorkspace.evHx[ 6 ]), acadoWorkspace.evGx, &(acadoWorkspace.A[ 1190 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.A[ 1260 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 18 ]), &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.A[ 1330 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.A[ 1400 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.A[ 1470 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.A[ 1540 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 42 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.A[ 1610 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.A[ 1680 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 54 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.A[ 1750 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.A[ 1820 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 66 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.A[ 1890 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.A[ 1960 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 78 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.A[ 2030 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 84 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.A[ 2100 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.A[ 2170 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 96 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.A[ 2240 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 102 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.A[ 2310 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 108 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.A[ 2380 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 114 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.A[ 2450 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.A[ 2520 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 126 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.A[ 2590 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 132 ]), &(acadoWorkspace.evGx[ 189 ]), &(acadoWorkspace.A[ 2660 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 138 ]), &(acadoWorkspace.evGx[ 198 ]), &(acadoWorkspace.A[ 2730 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 144 ]), &(acadoWorkspace.evGx[ 207 ]), &(acadoWorkspace.A[ 2800 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.A[ 2870 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 156 ]), &(acadoWorkspace.evGx[ 225 ]), &(acadoWorkspace.A[ 2940 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 162 ]), &(acadoWorkspace.evGx[ 234 ]), &(acadoWorkspace.A[ 3010 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 168 ]), &(acadoWorkspace.evGx[ 243 ]), &(acadoWorkspace.A[ 3080 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 174 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.A[ 3150 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.evGx[ 261 ]), &(acadoWorkspace.A[ 3220 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 186 ]), &(acadoWorkspace.evGx[ 270 ]), &(acadoWorkspace.A[ 3290 ]) );

for (lRun2 = 0; lRun2 < 31; ++lRun2)
{
for (lRun3 = 0; lRun3 < lRun2 + 1; ++lRun3)
{
lRun4 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun3);
lRun5 = lRun2 + 1;
acado_multHxE( &(acadoWorkspace.evHx[ lRun2 * 6 + 6 ]), &(acadoWorkspace.E[ lRun4 * 3 ]), lRun5, lRun3 );
}
}

acadoWorkspace.A[1123] = acadoWorkspace.evHu[0];
acadoWorkspace.A[1158] = acadoWorkspace.evHu[1];
acadoWorkspace.A[1194] = acadoWorkspace.evHu[2];
acadoWorkspace.A[1229] = acadoWorkspace.evHu[3];
acadoWorkspace.A[1265] = acadoWorkspace.evHu[4];
acadoWorkspace.A[1300] = acadoWorkspace.evHu[5];
acadoWorkspace.A[1336] = acadoWorkspace.evHu[6];
acadoWorkspace.A[1371] = acadoWorkspace.evHu[7];
acadoWorkspace.A[1407] = acadoWorkspace.evHu[8];
acadoWorkspace.A[1442] = acadoWorkspace.evHu[9];
acadoWorkspace.A[1478] = acadoWorkspace.evHu[10];
acadoWorkspace.A[1513] = acadoWorkspace.evHu[11];
acadoWorkspace.A[1549] = acadoWorkspace.evHu[12];
acadoWorkspace.A[1584] = acadoWorkspace.evHu[13];
acadoWorkspace.A[1620] = acadoWorkspace.evHu[14];
acadoWorkspace.A[1655] = acadoWorkspace.evHu[15];
acadoWorkspace.A[1691] = acadoWorkspace.evHu[16];
acadoWorkspace.A[1726] = acadoWorkspace.evHu[17];
acadoWorkspace.A[1762] = acadoWorkspace.evHu[18];
acadoWorkspace.A[1797] = acadoWorkspace.evHu[19];
acadoWorkspace.A[1833] = acadoWorkspace.evHu[20];
acadoWorkspace.A[1868] = acadoWorkspace.evHu[21];
acadoWorkspace.A[1904] = acadoWorkspace.evHu[22];
acadoWorkspace.A[1939] = acadoWorkspace.evHu[23];
acadoWorkspace.A[1975] = acadoWorkspace.evHu[24];
acadoWorkspace.A[2010] = acadoWorkspace.evHu[25];
acadoWorkspace.A[2046] = acadoWorkspace.evHu[26];
acadoWorkspace.A[2081] = acadoWorkspace.evHu[27];
acadoWorkspace.A[2117] = acadoWorkspace.evHu[28];
acadoWorkspace.A[2152] = acadoWorkspace.evHu[29];
acadoWorkspace.A[2188] = acadoWorkspace.evHu[30];
acadoWorkspace.A[2223] = acadoWorkspace.evHu[31];
acadoWorkspace.A[2259] = acadoWorkspace.evHu[32];
acadoWorkspace.A[2294] = acadoWorkspace.evHu[33];
acadoWorkspace.A[2330] = acadoWorkspace.evHu[34];
acadoWorkspace.A[2365] = acadoWorkspace.evHu[35];
acadoWorkspace.A[2401] = acadoWorkspace.evHu[36];
acadoWorkspace.A[2436] = acadoWorkspace.evHu[37];
acadoWorkspace.A[2472] = acadoWorkspace.evHu[38];
acadoWorkspace.A[2507] = acadoWorkspace.evHu[39];
acadoWorkspace.A[2543] = acadoWorkspace.evHu[40];
acadoWorkspace.A[2578] = acadoWorkspace.evHu[41];
acadoWorkspace.A[2614] = acadoWorkspace.evHu[42];
acadoWorkspace.A[2649] = acadoWorkspace.evHu[43];
acadoWorkspace.A[2685] = acadoWorkspace.evHu[44];
acadoWorkspace.A[2720] = acadoWorkspace.evHu[45];
acadoWorkspace.A[2756] = acadoWorkspace.evHu[46];
acadoWorkspace.A[2791] = acadoWorkspace.evHu[47];
acadoWorkspace.A[2827] = acadoWorkspace.evHu[48];
acadoWorkspace.A[2862] = acadoWorkspace.evHu[49];
acadoWorkspace.A[2898] = acadoWorkspace.evHu[50];
acadoWorkspace.A[2933] = acadoWorkspace.evHu[51];
acadoWorkspace.A[2969] = acadoWorkspace.evHu[52];
acadoWorkspace.A[3004] = acadoWorkspace.evHu[53];
acadoWorkspace.A[3040] = acadoWorkspace.evHu[54];
acadoWorkspace.A[3075] = acadoWorkspace.evHu[55];
acadoWorkspace.A[3111] = acadoWorkspace.evHu[56];
acadoWorkspace.A[3146] = acadoWorkspace.evHu[57];
acadoWorkspace.A[3182] = acadoWorkspace.evHu[58];
acadoWorkspace.A[3217] = acadoWorkspace.evHu[59];
acadoWorkspace.A[3253] = acadoWorkspace.evHu[60];
acadoWorkspace.A[3288] = acadoWorkspace.evHu[61];
acadoWorkspace.A[3324] = acadoWorkspace.evHu[62];
acadoWorkspace.A[3359] = acadoWorkspace.evHu[63];
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

acado_macHxd( &(acadoWorkspace.evHx[ 6 ]), acadoWorkspace.d, &(acadoWorkspace.lbA[ 34 ]), &(acadoWorkspace.ubA[ 34 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.d[ 3 ]), &(acadoWorkspace.lbA[ 36 ]), &(acadoWorkspace.ubA[ 36 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 18 ]), &(acadoWorkspace.d[ 6 ]), &(acadoWorkspace.lbA[ 38 ]), &(acadoWorkspace.ubA[ 38 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.d[ 9 ]), &(acadoWorkspace.lbA[ 40 ]), &(acadoWorkspace.ubA[ 40 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 30 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.lbA[ 42 ]), &(acadoWorkspace.ubA[ 42 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.lbA[ 44 ]), &(acadoWorkspace.ubA[ 44 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 42 ]), &(acadoWorkspace.d[ 18 ]), &(acadoWorkspace.lbA[ 46 ]), &(acadoWorkspace.ubA[ 46 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.d[ 21 ]), &(acadoWorkspace.lbA[ 48 ]), &(acadoWorkspace.ubA[ 48 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 54 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.lbA[ 50 ]), &(acadoWorkspace.ubA[ 50 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.d[ 27 ]), &(acadoWorkspace.lbA[ 52 ]), &(acadoWorkspace.ubA[ 52 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 66 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.lbA[ 54 ]), &(acadoWorkspace.ubA[ 54 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.d[ 33 ]), &(acadoWorkspace.lbA[ 56 ]), &(acadoWorkspace.ubA[ 56 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 78 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.lbA[ 58 ]), &(acadoWorkspace.ubA[ 58 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 84 ]), &(acadoWorkspace.d[ 39 ]), &(acadoWorkspace.lbA[ 60 ]), &(acadoWorkspace.ubA[ 60 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 90 ]), &(acadoWorkspace.d[ 42 ]), &(acadoWorkspace.lbA[ 62 ]), &(acadoWorkspace.ubA[ 62 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 96 ]), &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.lbA[ 64 ]), &(acadoWorkspace.ubA[ 64 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 102 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.lbA[ 66 ]), &(acadoWorkspace.ubA[ 66 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 108 ]), &(acadoWorkspace.d[ 51 ]), &(acadoWorkspace.lbA[ 68 ]), &(acadoWorkspace.ubA[ 68 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 114 ]), &(acadoWorkspace.d[ 54 ]), &(acadoWorkspace.lbA[ 70 ]), &(acadoWorkspace.ubA[ 70 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 120 ]), &(acadoWorkspace.d[ 57 ]), &(acadoWorkspace.lbA[ 72 ]), &(acadoWorkspace.ubA[ 72 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 126 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.lbA[ 74 ]), &(acadoWorkspace.ubA[ 74 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 132 ]), &(acadoWorkspace.d[ 63 ]), &(acadoWorkspace.lbA[ 76 ]), &(acadoWorkspace.ubA[ 76 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 138 ]), &(acadoWorkspace.d[ 66 ]), &(acadoWorkspace.lbA[ 78 ]), &(acadoWorkspace.ubA[ 78 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 144 ]), &(acadoWorkspace.d[ 69 ]), &(acadoWorkspace.lbA[ 80 ]), &(acadoWorkspace.ubA[ 80 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 150 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.lbA[ 82 ]), &(acadoWorkspace.ubA[ 82 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 156 ]), &(acadoWorkspace.d[ 75 ]), &(acadoWorkspace.lbA[ 84 ]), &(acadoWorkspace.ubA[ 84 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 162 ]), &(acadoWorkspace.d[ 78 ]), &(acadoWorkspace.lbA[ 86 ]), &(acadoWorkspace.ubA[ 86 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 168 ]), &(acadoWorkspace.d[ 81 ]), &(acadoWorkspace.lbA[ 88 ]), &(acadoWorkspace.ubA[ 88 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 174 ]), &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.lbA[ 90 ]), &(acadoWorkspace.ubA[ 90 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 180 ]), &(acadoWorkspace.d[ 87 ]), &(acadoWorkspace.lbA[ 92 ]), &(acadoWorkspace.ubA[ 92 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 186 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.lbA[ 94 ]), &(acadoWorkspace.ubA[ 94 ]) );

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

for (lRun2 = 0; lRun2 < 128; ++lRun2)
acadoWorkspace.Dy[lRun2] -= acadoVariables.y[lRun2];

acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 3 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 4 ]), &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 8 ]), &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 12 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 16 ]), &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 20 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 24 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 28 ]), &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 32 ]), &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 36 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 40 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 44 ]), &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 48 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 52 ]), &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 56 ]), &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 60 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 64 ]), &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 68 ]), &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 72 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 76 ]), &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 80 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 84 ]), &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 88 ]), &(acadoWorkspace.Dy[ 88 ]), &(acadoWorkspace.g[ 25 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 92 ]), &(acadoWorkspace.Dy[ 92 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 96 ]), &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.g[ 27 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 100 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 104 ]), &(acadoWorkspace.Dy[ 104 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 108 ]), &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 112 ]), &(acadoWorkspace.Dy[ 112 ]), &(acadoWorkspace.g[ 31 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 116 ]), &(acadoWorkspace.Dy[ 116 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 120 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.g[ 33 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 124 ]), &(acadoWorkspace.Dy[ 124 ]), &(acadoWorkspace.g[ 34 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 12 ]), &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.QDy[ 3 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 24 ]), &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.QDy[ 6 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 36 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 9 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 48 ]), &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 60 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.QDy[ 15 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 72 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 18 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 84 ]), &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.QDy[ 21 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 96 ]), &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.QDy[ 24 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 108 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.QDy[ 27 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 120 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.QDy[ 30 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 132 ]), &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.QDy[ 33 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 144 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.QDy[ 36 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 156 ]), &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.QDy[ 39 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 168 ]), &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.QDy[ 42 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 180 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 45 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 192 ]), &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.QDy[ 48 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 204 ]), &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.QDy[ 51 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 216 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.QDy[ 54 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 228 ]), &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.QDy[ 57 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 240 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 252 ]), &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.QDy[ 63 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 264 ]), &(acadoWorkspace.Dy[ 88 ]), &(acadoWorkspace.QDy[ 66 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 276 ]), &(acadoWorkspace.Dy[ 92 ]), &(acadoWorkspace.QDy[ 69 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 288 ]), &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.QDy[ 72 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 300 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.QDy[ 75 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 312 ]), &(acadoWorkspace.Dy[ 104 ]), &(acadoWorkspace.QDy[ 78 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 324 ]), &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.QDy[ 81 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 336 ]), &(acadoWorkspace.Dy[ 112 ]), &(acadoWorkspace.QDy[ 84 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 348 ]), &(acadoWorkspace.Dy[ 116 ]), &(acadoWorkspace.QDy[ 87 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 360 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.QDy[ 90 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 372 ]), &(acadoWorkspace.Dy[ 124 ]), &(acadoWorkspace.QDy[ 93 ]) );

acadoWorkspace.QDy[96] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[97] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[98] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];

acadoWorkspace.QDy[3] += acadoWorkspace.Qd[0];
acadoWorkspace.QDy[4] += acadoWorkspace.Qd[1];
acadoWorkspace.QDy[5] += acadoWorkspace.Qd[2];
acadoWorkspace.QDy[6] += acadoWorkspace.Qd[3];
acadoWorkspace.QDy[7] += acadoWorkspace.Qd[4];
acadoWorkspace.QDy[8] += acadoWorkspace.Qd[5];
acadoWorkspace.QDy[9] += acadoWorkspace.Qd[6];
acadoWorkspace.QDy[10] += acadoWorkspace.Qd[7];
acadoWorkspace.QDy[11] += acadoWorkspace.Qd[8];
acadoWorkspace.QDy[12] += acadoWorkspace.Qd[9];
acadoWorkspace.QDy[13] += acadoWorkspace.Qd[10];
acadoWorkspace.QDy[14] += acadoWorkspace.Qd[11];
acadoWorkspace.QDy[15] += acadoWorkspace.Qd[12];
acadoWorkspace.QDy[16] += acadoWorkspace.Qd[13];
acadoWorkspace.QDy[17] += acadoWorkspace.Qd[14];
acadoWorkspace.QDy[18] += acadoWorkspace.Qd[15];
acadoWorkspace.QDy[19] += acadoWorkspace.Qd[16];
acadoWorkspace.QDy[20] += acadoWorkspace.Qd[17];
acadoWorkspace.QDy[21] += acadoWorkspace.Qd[18];
acadoWorkspace.QDy[22] += acadoWorkspace.Qd[19];
acadoWorkspace.QDy[23] += acadoWorkspace.Qd[20];
acadoWorkspace.QDy[24] += acadoWorkspace.Qd[21];
acadoWorkspace.QDy[25] += acadoWorkspace.Qd[22];
acadoWorkspace.QDy[26] += acadoWorkspace.Qd[23];
acadoWorkspace.QDy[27] += acadoWorkspace.Qd[24];
acadoWorkspace.QDy[28] += acadoWorkspace.Qd[25];
acadoWorkspace.QDy[29] += acadoWorkspace.Qd[26];
acadoWorkspace.QDy[30] += acadoWorkspace.Qd[27];
acadoWorkspace.QDy[31] += acadoWorkspace.Qd[28];
acadoWorkspace.QDy[32] += acadoWorkspace.Qd[29];
acadoWorkspace.QDy[33] += acadoWorkspace.Qd[30];
acadoWorkspace.QDy[34] += acadoWorkspace.Qd[31];
acadoWorkspace.QDy[35] += acadoWorkspace.Qd[32];
acadoWorkspace.QDy[36] += acadoWorkspace.Qd[33];
acadoWorkspace.QDy[37] += acadoWorkspace.Qd[34];
acadoWorkspace.QDy[38] += acadoWorkspace.Qd[35];
acadoWorkspace.QDy[39] += acadoWorkspace.Qd[36];
acadoWorkspace.QDy[40] += acadoWorkspace.Qd[37];
acadoWorkspace.QDy[41] += acadoWorkspace.Qd[38];
acadoWorkspace.QDy[42] += acadoWorkspace.Qd[39];
acadoWorkspace.QDy[43] += acadoWorkspace.Qd[40];
acadoWorkspace.QDy[44] += acadoWorkspace.Qd[41];
acadoWorkspace.QDy[45] += acadoWorkspace.Qd[42];
acadoWorkspace.QDy[46] += acadoWorkspace.Qd[43];
acadoWorkspace.QDy[47] += acadoWorkspace.Qd[44];
acadoWorkspace.QDy[48] += acadoWorkspace.Qd[45];
acadoWorkspace.QDy[49] += acadoWorkspace.Qd[46];
acadoWorkspace.QDy[50] += acadoWorkspace.Qd[47];
acadoWorkspace.QDy[51] += acadoWorkspace.Qd[48];
acadoWorkspace.QDy[52] += acadoWorkspace.Qd[49];
acadoWorkspace.QDy[53] += acadoWorkspace.Qd[50];
acadoWorkspace.QDy[54] += acadoWorkspace.Qd[51];
acadoWorkspace.QDy[55] += acadoWorkspace.Qd[52];
acadoWorkspace.QDy[56] += acadoWorkspace.Qd[53];
acadoWorkspace.QDy[57] += acadoWorkspace.Qd[54];
acadoWorkspace.QDy[58] += acadoWorkspace.Qd[55];
acadoWorkspace.QDy[59] += acadoWorkspace.Qd[56];
acadoWorkspace.QDy[60] += acadoWorkspace.Qd[57];
acadoWorkspace.QDy[61] += acadoWorkspace.Qd[58];
acadoWorkspace.QDy[62] += acadoWorkspace.Qd[59];
acadoWorkspace.QDy[63] += acadoWorkspace.Qd[60];
acadoWorkspace.QDy[64] += acadoWorkspace.Qd[61];
acadoWorkspace.QDy[65] += acadoWorkspace.Qd[62];
acadoWorkspace.QDy[66] += acadoWorkspace.Qd[63];
acadoWorkspace.QDy[67] += acadoWorkspace.Qd[64];
acadoWorkspace.QDy[68] += acadoWorkspace.Qd[65];
acadoWorkspace.QDy[69] += acadoWorkspace.Qd[66];
acadoWorkspace.QDy[70] += acadoWorkspace.Qd[67];
acadoWorkspace.QDy[71] += acadoWorkspace.Qd[68];
acadoWorkspace.QDy[72] += acadoWorkspace.Qd[69];
acadoWorkspace.QDy[73] += acadoWorkspace.Qd[70];
acadoWorkspace.QDy[74] += acadoWorkspace.Qd[71];
acadoWorkspace.QDy[75] += acadoWorkspace.Qd[72];
acadoWorkspace.QDy[76] += acadoWorkspace.Qd[73];
acadoWorkspace.QDy[77] += acadoWorkspace.Qd[74];
acadoWorkspace.QDy[78] += acadoWorkspace.Qd[75];
acadoWorkspace.QDy[79] += acadoWorkspace.Qd[76];
acadoWorkspace.QDy[80] += acadoWorkspace.Qd[77];
acadoWorkspace.QDy[81] += acadoWorkspace.Qd[78];
acadoWorkspace.QDy[82] += acadoWorkspace.Qd[79];
acadoWorkspace.QDy[83] += acadoWorkspace.Qd[80];
acadoWorkspace.QDy[84] += acadoWorkspace.Qd[81];
acadoWorkspace.QDy[85] += acadoWorkspace.Qd[82];
acadoWorkspace.QDy[86] += acadoWorkspace.Qd[83];
acadoWorkspace.QDy[87] += acadoWorkspace.Qd[84];
acadoWorkspace.QDy[88] += acadoWorkspace.Qd[85];
acadoWorkspace.QDy[89] += acadoWorkspace.Qd[86];
acadoWorkspace.QDy[90] += acadoWorkspace.Qd[87];
acadoWorkspace.QDy[91] += acadoWorkspace.Qd[88];
acadoWorkspace.QDy[92] += acadoWorkspace.Qd[89];
acadoWorkspace.QDy[93] += acadoWorkspace.Qd[90];
acadoWorkspace.QDy[94] += acadoWorkspace.Qd[91];
acadoWorkspace.QDy[95] += acadoWorkspace.Qd[92];
acadoWorkspace.QDy[96] += acadoWorkspace.Qd[93];
acadoWorkspace.QDy[97] += acadoWorkspace.Qd[94];
acadoWorkspace.QDy[98] += acadoWorkspace.Qd[95];

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[98];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[98];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[98];


for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 32; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multEQDy( &(acadoWorkspace.E[ lRun3 * 3 ]), &(acadoWorkspace.QDy[ lRun2 * 3 + 3 ]), &(acadoWorkspace.g[ lRun1 + 3 ]) );
}
}

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
tmp = acadoVariables.x[4] + acadoWorkspace.d[1];
acadoWorkspace.lbA[0] = - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[7] + acadoWorkspace.d[4];
acadoWorkspace.lbA[1] = - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[10] + acadoWorkspace.d[7];
acadoWorkspace.lbA[2] = - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[13] + acadoWorkspace.d[10];
acadoWorkspace.lbA[3] = - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[16] + acadoWorkspace.d[13];
acadoWorkspace.lbA[4] = - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[19] + acadoWorkspace.d[16];
acadoWorkspace.lbA[5] = - tmp;
acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[22] + acadoWorkspace.d[19];
acadoWorkspace.lbA[6] = - tmp;
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[25] + acadoWorkspace.d[22];
acadoWorkspace.lbA[7] = - tmp;
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[28] + acadoWorkspace.d[25];
acadoWorkspace.lbA[8] = - tmp;
acadoWorkspace.ubA[8] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[28];
acadoWorkspace.lbA[9] = - tmp;
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[34] + acadoWorkspace.d[31];
acadoWorkspace.lbA[10] = - tmp;
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[37] + acadoWorkspace.d[34];
acadoWorkspace.lbA[11] = - tmp;
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[40] + acadoWorkspace.d[37];
acadoWorkspace.lbA[12] = - tmp;
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[43] + acadoWorkspace.d[40];
acadoWorkspace.lbA[13] = - tmp;
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[46] + acadoWorkspace.d[43];
acadoWorkspace.lbA[14] = - tmp;
acadoWorkspace.ubA[14] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[49] + acadoWorkspace.d[46];
acadoWorkspace.lbA[15] = - tmp;
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[52] + acadoWorkspace.d[49];
acadoWorkspace.lbA[16] = - tmp;
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[55] + acadoWorkspace.d[52];
acadoWorkspace.lbA[17] = - tmp;
acadoWorkspace.ubA[17] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[58] + acadoWorkspace.d[55];
acadoWorkspace.lbA[18] = - tmp;
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[61] + acadoWorkspace.d[58];
acadoWorkspace.lbA[19] = - tmp;
acadoWorkspace.ubA[19] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[64] + acadoWorkspace.d[61];
acadoWorkspace.lbA[20] = - tmp;
acadoWorkspace.ubA[20] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[67] + acadoWorkspace.d[64];
acadoWorkspace.lbA[21] = - tmp;
acadoWorkspace.ubA[21] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[70] + acadoWorkspace.d[67];
acadoWorkspace.lbA[22] = - tmp;
acadoWorkspace.ubA[22] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[73] + acadoWorkspace.d[70];
acadoWorkspace.lbA[23] = - tmp;
acadoWorkspace.ubA[23] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[76] + acadoWorkspace.d[73];
acadoWorkspace.lbA[24] = - tmp;
acadoWorkspace.ubA[24] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[79] + acadoWorkspace.d[76];
acadoWorkspace.lbA[25] = - tmp;
acadoWorkspace.ubA[25] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[82] + acadoWorkspace.d[79];
acadoWorkspace.lbA[26] = - tmp;
acadoWorkspace.ubA[26] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[85] + acadoWorkspace.d[82];
acadoWorkspace.lbA[27] = - tmp;
acadoWorkspace.ubA[27] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[88] + acadoWorkspace.d[85];
acadoWorkspace.lbA[28] = - tmp;
acadoWorkspace.ubA[28] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[91] + acadoWorkspace.d[88];
acadoWorkspace.lbA[29] = - tmp;
acadoWorkspace.ubA[29] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[94] + acadoWorkspace.d[91];
acadoWorkspace.lbA[30] = - tmp;
acadoWorkspace.ubA[30] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[97] + acadoWorkspace.d[94];
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

acadoVariables.u[0] += acadoWorkspace.x[3];
acadoVariables.u[1] += acadoWorkspace.x[4];
acadoVariables.u[2] += acadoWorkspace.x[5];
acadoVariables.u[3] += acadoWorkspace.x[6];
acadoVariables.u[4] += acadoWorkspace.x[7];
acadoVariables.u[5] += acadoWorkspace.x[8];
acadoVariables.u[6] += acadoWorkspace.x[9];
acadoVariables.u[7] += acadoWorkspace.x[10];
acadoVariables.u[8] += acadoWorkspace.x[11];
acadoVariables.u[9] += acadoWorkspace.x[12];
acadoVariables.u[10] += acadoWorkspace.x[13];
acadoVariables.u[11] += acadoWorkspace.x[14];
acadoVariables.u[12] += acadoWorkspace.x[15];
acadoVariables.u[13] += acadoWorkspace.x[16];
acadoVariables.u[14] += acadoWorkspace.x[17];
acadoVariables.u[15] += acadoWorkspace.x[18];
acadoVariables.u[16] += acadoWorkspace.x[19];
acadoVariables.u[17] += acadoWorkspace.x[20];
acadoVariables.u[18] += acadoWorkspace.x[21];
acadoVariables.u[19] += acadoWorkspace.x[22];
acadoVariables.u[20] += acadoWorkspace.x[23];
acadoVariables.u[21] += acadoWorkspace.x[24];
acadoVariables.u[22] += acadoWorkspace.x[25];
acadoVariables.u[23] += acadoWorkspace.x[26];
acadoVariables.u[24] += acadoWorkspace.x[27];
acadoVariables.u[25] += acadoWorkspace.x[28];
acadoVariables.u[26] += acadoWorkspace.x[29];
acadoVariables.u[27] += acadoWorkspace.x[30];
acadoVariables.u[28] += acadoWorkspace.x[31];
acadoVariables.u[29] += acadoWorkspace.x[32];
acadoVariables.u[30] += acadoWorkspace.x[33];
acadoVariables.u[31] += acadoWorkspace.x[34];

acadoVariables.x[3] += + acadoWorkspace.evGx[0]*acadoWorkspace.x[0] + acadoWorkspace.evGx[1]*acadoWorkspace.x[1] + acadoWorkspace.evGx[2]*acadoWorkspace.x[2] + acadoWorkspace.d[0];
acadoVariables.x[4] += + acadoWorkspace.evGx[3]*acadoWorkspace.x[0] + acadoWorkspace.evGx[4]*acadoWorkspace.x[1] + acadoWorkspace.evGx[5]*acadoWorkspace.x[2] + acadoWorkspace.d[1];
acadoVariables.x[5] += + acadoWorkspace.evGx[6]*acadoWorkspace.x[0] + acadoWorkspace.evGx[7]*acadoWorkspace.x[1] + acadoWorkspace.evGx[8]*acadoWorkspace.x[2] + acadoWorkspace.d[2];
acadoVariables.x[6] += + acadoWorkspace.evGx[9]*acadoWorkspace.x[0] + acadoWorkspace.evGx[10]*acadoWorkspace.x[1] + acadoWorkspace.evGx[11]*acadoWorkspace.x[2] + acadoWorkspace.d[3];
acadoVariables.x[7] += + acadoWorkspace.evGx[12]*acadoWorkspace.x[0] + acadoWorkspace.evGx[13]*acadoWorkspace.x[1] + acadoWorkspace.evGx[14]*acadoWorkspace.x[2] + acadoWorkspace.d[4];
acadoVariables.x[8] += + acadoWorkspace.evGx[15]*acadoWorkspace.x[0] + acadoWorkspace.evGx[16]*acadoWorkspace.x[1] + acadoWorkspace.evGx[17]*acadoWorkspace.x[2] + acadoWorkspace.d[5];
acadoVariables.x[9] += + acadoWorkspace.evGx[18]*acadoWorkspace.x[0] + acadoWorkspace.evGx[19]*acadoWorkspace.x[1] + acadoWorkspace.evGx[20]*acadoWorkspace.x[2] + acadoWorkspace.d[6];
acadoVariables.x[10] += + acadoWorkspace.evGx[21]*acadoWorkspace.x[0] + acadoWorkspace.evGx[22]*acadoWorkspace.x[1] + acadoWorkspace.evGx[23]*acadoWorkspace.x[2] + acadoWorkspace.d[7];
acadoVariables.x[11] += + acadoWorkspace.evGx[24]*acadoWorkspace.x[0] + acadoWorkspace.evGx[25]*acadoWorkspace.x[1] + acadoWorkspace.evGx[26]*acadoWorkspace.x[2] + acadoWorkspace.d[8];
acadoVariables.x[12] += + acadoWorkspace.evGx[27]*acadoWorkspace.x[0] + acadoWorkspace.evGx[28]*acadoWorkspace.x[1] + acadoWorkspace.evGx[29]*acadoWorkspace.x[2] + acadoWorkspace.d[9];
acadoVariables.x[13] += + acadoWorkspace.evGx[30]*acadoWorkspace.x[0] + acadoWorkspace.evGx[31]*acadoWorkspace.x[1] + acadoWorkspace.evGx[32]*acadoWorkspace.x[2] + acadoWorkspace.d[10];
acadoVariables.x[14] += + acadoWorkspace.evGx[33]*acadoWorkspace.x[0] + acadoWorkspace.evGx[34]*acadoWorkspace.x[1] + acadoWorkspace.evGx[35]*acadoWorkspace.x[2] + acadoWorkspace.d[11];
acadoVariables.x[15] += + acadoWorkspace.evGx[36]*acadoWorkspace.x[0] + acadoWorkspace.evGx[37]*acadoWorkspace.x[1] + acadoWorkspace.evGx[38]*acadoWorkspace.x[2] + acadoWorkspace.d[12];
acadoVariables.x[16] += + acadoWorkspace.evGx[39]*acadoWorkspace.x[0] + acadoWorkspace.evGx[40]*acadoWorkspace.x[1] + acadoWorkspace.evGx[41]*acadoWorkspace.x[2] + acadoWorkspace.d[13];
acadoVariables.x[17] += + acadoWorkspace.evGx[42]*acadoWorkspace.x[0] + acadoWorkspace.evGx[43]*acadoWorkspace.x[1] + acadoWorkspace.evGx[44]*acadoWorkspace.x[2] + acadoWorkspace.d[14];
acadoVariables.x[18] += + acadoWorkspace.evGx[45]*acadoWorkspace.x[0] + acadoWorkspace.evGx[46]*acadoWorkspace.x[1] + acadoWorkspace.evGx[47]*acadoWorkspace.x[2] + acadoWorkspace.d[15];
acadoVariables.x[19] += + acadoWorkspace.evGx[48]*acadoWorkspace.x[0] + acadoWorkspace.evGx[49]*acadoWorkspace.x[1] + acadoWorkspace.evGx[50]*acadoWorkspace.x[2] + acadoWorkspace.d[16];
acadoVariables.x[20] += + acadoWorkspace.evGx[51]*acadoWorkspace.x[0] + acadoWorkspace.evGx[52]*acadoWorkspace.x[1] + acadoWorkspace.evGx[53]*acadoWorkspace.x[2] + acadoWorkspace.d[17];
acadoVariables.x[21] += + acadoWorkspace.evGx[54]*acadoWorkspace.x[0] + acadoWorkspace.evGx[55]*acadoWorkspace.x[1] + acadoWorkspace.evGx[56]*acadoWorkspace.x[2] + acadoWorkspace.d[18];
acadoVariables.x[22] += + acadoWorkspace.evGx[57]*acadoWorkspace.x[0] + acadoWorkspace.evGx[58]*acadoWorkspace.x[1] + acadoWorkspace.evGx[59]*acadoWorkspace.x[2] + acadoWorkspace.d[19];
acadoVariables.x[23] += + acadoWorkspace.evGx[60]*acadoWorkspace.x[0] + acadoWorkspace.evGx[61]*acadoWorkspace.x[1] + acadoWorkspace.evGx[62]*acadoWorkspace.x[2] + acadoWorkspace.d[20];
acadoVariables.x[24] += + acadoWorkspace.evGx[63]*acadoWorkspace.x[0] + acadoWorkspace.evGx[64]*acadoWorkspace.x[1] + acadoWorkspace.evGx[65]*acadoWorkspace.x[2] + acadoWorkspace.d[21];
acadoVariables.x[25] += + acadoWorkspace.evGx[66]*acadoWorkspace.x[0] + acadoWorkspace.evGx[67]*acadoWorkspace.x[1] + acadoWorkspace.evGx[68]*acadoWorkspace.x[2] + acadoWorkspace.d[22];
acadoVariables.x[26] += + acadoWorkspace.evGx[69]*acadoWorkspace.x[0] + acadoWorkspace.evGx[70]*acadoWorkspace.x[1] + acadoWorkspace.evGx[71]*acadoWorkspace.x[2] + acadoWorkspace.d[23];
acadoVariables.x[27] += + acadoWorkspace.evGx[72]*acadoWorkspace.x[0] + acadoWorkspace.evGx[73]*acadoWorkspace.x[1] + acadoWorkspace.evGx[74]*acadoWorkspace.x[2] + acadoWorkspace.d[24];
acadoVariables.x[28] += + acadoWorkspace.evGx[75]*acadoWorkspace.x[0] + acadoWorkspace.evGx[76]*acadoWorkspace.x[1] + acadoWorkspace.evGx[77]*acadoWorkspace.x[2] + acadoWorkspace.d[25];
acadoVariables.x[29] += + acadoWorkspace.evGx[78]*acadoWorkspace.x[0] + acadoWorkspace.evGx[79]*acadoWorkspace.x[1] + acadoWorkspace.evGx[80]*acadoWorkspace.x[2] + acadoWorkspace.d[26];
acadoVariables.x[30] += + acadoWorkspace.evGx[81]*acadoWorkspace.x[0] + acadoWorkspace.evGx[82]*acadoWorkspace.x[1] + acadoWorkspace.evGx[83]*acadoWorkspace.x[2] + acadoWorkspace.d[27];
acadoVariables.x[31] += + acadoWorkspace.evGx[84]*acadoWorkspace.x[0] + acadoWorkspace.evGx[85]*acadoWorkspace.x[1] + acadoWorkspace.evGx[86]*acadoWorkspace.x[2] + acadoWorkspace.d[28];
acadoVariables.x[32] += + acadoWorkspace.evGx[87]*acadoWorkspace.x[0] + acadoWorkspace.evGx[88]*acadoWorkspace.x[1] + acadoWorkspace.evGx[89]*acadoWorkspace.x[2] + acadoWorkspace.d[29];
acadoVariables.x[33] += + acadoWorkspace.evGx[90]*acadoWorkspace.x[0] + acadoWorkspace.evGx[91]*acadoWorkspace.x[1] + acadoWorkspace.evGx[92]*acadoWorkspace.x[2] + acadoWorkspace.d[30];
acadoVariables.x[34] += + acadoWorkspace.evGx[93]*acadoWorkspace.x[0] + acadoWorkspace.evGx[94]*acadoWorkspace.x[1] + acadoWorkspace.evGx[95]*acadoWorkspace.x[2] + acadoWorkspace.d[31];
acadoVariables.x[35] += + acadoWorkspace.evGx[96]*acadoWorkspace.x[0] + acadoWorkspace.evGx[97]*acadoWorkspace.x[1] + acadoWorkspace.evGx[98]*acadoWorkspace.x[2] + acadoWorkspace.d[32];
acadoVariables.x[36] += + acadoWorkspace.evGx[99]*acadoWorkspace.x[0] + acadoWorkspace.evGx[100]*acadoWorkspace.x[1] + acadoWorkspace.evGx[101]*acadoWorkspace.x[2] + acadoWorkspace.d[33];
acadoVariables.x[37] += + acadoWorkspace.evGx[102]*acadoWorkspace.x[0] + acadoWorkspace.evGx[103]*acadoWorkspace.x[1] + acadoWorkspace.evGx[104]*acadoWorkspace.x[2] + acadoWorkspace.d[34];
acadoVariables.x[38] += + acadoWorkspace.evGx[105]*acadoWorkspace.x[0] + acadoWorkspace.evGx[106]*acadoWorkspace.x[1] + acadoWorkspace.evGx[107]*acadoWorkspace.x[2] + acadoWorkspace.d[35];
acadoVariables.x[39] += + acadoWorkspace.evGx[108]*acadoWorkspace.x[0] + acadoWorkspace.evGx[109]*acadoWorkspace.x[1] + acadoWorkspace.evGx[110]*acadoWorkspace.x[2] + acadoWorkspace.d[36];
acadoVariables.x[40] += + acadoWorkspace.evGx[111]*acadoWorkspace.x[0] + acadoWorkspace.evGx[112]*acadoWorkspace.x[1] + acadoWorkspace.evGx[113]*acadoWorkspace.x[2] + acadoWorkspace.d[37];
acadoVariables.x[41] += + acadoWorkspace.evGx[114]*acadoWorkspace.x[0] + acadoWorkspace.evGx[115]*acadoWorkspace.x[1] + acadoWorkspace.evGx[116]*acadoWorkspace.x[2] + acadoWorkspace.d[38];
acadoVariables.x[42] += + acadoWorkspace.evGx[117]*acadoWorkspace.x[0] + acadoWorkspace.evGx[118]*acadoWorkspace.x[1] + acadoWorkspace.evGx[119]*acadoWorkspace.x[2] + acadoWorkspace.d[39];
acadoVariables.x[43] += + acadoWorkspace.evGx[120]*acadoWorkspace.x[0] + acadoWorkspace.evGx[121]*acadoWorkspace.x[1] + acadoWorkspace.evGx[122]*acadoWorkspace.x[2] + acadoWorkspace.d[40];
acadoVariables.x[44] += + acadoWorkspace.evGx[123]*acadoWorkspace.x[0] + acadoWorkspace.evGx[124]*acadoWorkspace.x[1] + acadoWorkspace.evGx[125]*acadoWorkspace.x[2] + acadoWorkspace.d[41];
acadoVariables.x[45] += + acadoWorkspace.evGx[126]*acadoWorkspace.x[0] + acadoWorkspace.evGx[127]*acadoWorkspace.x[1] + acadoWorkspace.evGx[128]*acadoWorkspace.x[2] + acadoWorkspace.d[42];
acadoVariables.x[46] += + acadoWorkspace.evGx[129]*acadoWorkspace.x[0] + acadoWorkspace.evGx[130]*acadoWorkspace.x[1] + acadoWorkspace.evGx[131]*acadoWorkspace.x[2] + acadoWorkspace.d[43];
acadoVariables.x[47] += + acadoWorkspace.evGx[132]*acadoWorkspace.x[0] + acadoWorkspace.evGx[133]*acadoWorkspace.x[1] + acadoWorkspace.evGx[134]*acadoWorkspace.x[2] + acadoWorkspace.d[44];
acadoVariables.x[48] += + acadoWorkspace.evGx[135]*acadoWorkspace.x[0] + acadoWorkspace.evGx[136]*acadoWorkspace.x[1] + acadoWorkspace.evGx[137]*acadoWorkspace.x[2] + acadoWorkspace.d[45];
acadoVariables.x[49] += + acadoWorkspace.evGx[138]*acadoWorkspace.x[0] + acadoWorkspace.evGx[139]*acadoWorkspace.x[1] + acadoWorkspace.evGx[140]*acadoWorkspace.x[2] + acadoWorkspace.d[46];
acadoVariables.x[50] += + acadoWorkspace.evGx[141]*acadoWorkspace.x[0] + acadoWorkspace.evGx[142]*acadoWorkspace.x[1] + acadoWorkspace.evGx[143]*acadoWorkspace.x[2] + acadoWorkspace.d[47];
acadoVariables.x[51] += + acadoWorkspace.evGx[144]*acadoWorkspace.x[0] + acadoWorkspace.evGx[145]*acadoWorkspace.x[1] + acadoWorkspace.evGx[146]*acadoWorkspace.x[2] + acadoWorkspace.d[48];
acadoVariables.x[52] += + acadoWorkspace.evGx[147]*acadoWorkspace.x[0] + acadoWorkspace.evGx[148]*acadoWorkspace.x[1] + acadoWorkspace.evGx[149]*acadoWorkspace.x[2] + acadoWorkspace.d[49];
acadoVariables.x[53] += + acadoWorkspace.evGx[150]*acadoWorkspace.x[0] + acadoWorkspace.evGx[151]*acadoWorkspace.x[1] + acadoWorkspace.evGx[152]*acadoWorkspace.x[2] + acadoWorkspace.d[50];
acadoVariables.x[54] += + acadoWorkspace.evGx[153]*acadoWorkspace.x[0] + acadoWorkspace.evGx[154]*acadoWorkspace.x[1] + acadoWorkspace.evGx[155]*acadoWorkspace.x[2] + acadoWorkspace.d[51];
acadoVariables.x[55] += + acadoWorkspace.evGx[156]*acadoWorkspace.x[0] + acadoWorkspace.evGx[157]*acadoWorkspace.x[1] + acadoWorkspace.evGx[158]*acadoWorkspace.x[2] + acadoWorkspace.d[52];
acadoVariables.x[56] += + acadoWorkspace.evGx[159]*acadoWorkspace.x[0] + acadoWorkspace.evGx[160]*acadoWorkspace.x[1] + acadoWorkspace.evGx[161]*acadoWorkspace.x[2] + acadoWorkspace.d[53];
acadoVariables.x[57] += + acadoWorkspace.evGx[162]*acadoWorkspace.x[0] + acadoWorkspace.evGx[163]*acadoWorkspace.x[1] + acadoWorkspace.evGx[164]*acadoWorkspace.x[2] + acadoWorkspace.d[54];
acadoVariables.x[58] += + acadoWorkspace.evGx[165]*acadoWorkspace.x[0] + acadoWorkspace.evGx[166]*acadoWorkspace.x[1] + acadoWorkspace.evGx[167]*acadoWorkspace.x[2] + acadoWorkspace.d[55];
acadoVariables.x[59] += + acadoWorkspace.evGx[168]*acadoWorkspace.x[0] + acadoWorkspace.evGx[169]*acadoWorkspace.x[1] + acadoWorkspace.evGx[170]*acadoWorkspace.x[2] + acadoWorkspace.d[56];
acadoVariables.x[60] += + acadoWorkspace.evGx[171]*acadoWorkspace.x[0] + acadoWorkspace.evGx[172]*acadoWorkspace.x[1] + acadoWorkspace.evGx[173]*acadoWorkspace.x[2] + acadoWorkspace.d[57];
acadoVariables.x[61] += + acadoWorkspace.evGx[174]*acadoWorkspace.x[0] + acadoWorkspace.evGx[175]*acadoWorkspace.x[1] + acadoWorkspace.evGx[176]*acadoWorkspace.x[2] + acadoWorkspace.d[58];
acadoVariables.x[62] += + acadoWorkspace.evGx[177]*acadoWorkspace.x[0] + acadoWorkspace.evGx[178]*acadoWorkspace.x[1] + acadoWorkspace.evGx[179]*acadoWorkspace.x[2] + acadoWorkspace.d[59];
acadoVariables.x[63] += + acadoWorkspace.evGx[180]*acadoWorkspace.x[0] + acadoWorkspace.evGx[181]*acadoWorkspace.x[1] + acadoWorkspace.evGx[182]*acadoWorkspace.x[2] + acadoWorkspace.d[60];
acadoVariables.x[64] += + acadoWorkspace.evGx[183]*acadoWorkspace.x[0] + acadoWorkspace.evGx[184]*acadoWorkspace.x[1] + acadoWorkspace.evGx[185]*acadoWorkspace.x[2] + acadoWorkspace.d[61];
acadoVariables.x[65] += + acadoWorkspace.evGx[186]*acadoWorkspace.x[0] + acadoWorkspace.evGx[187]*acadoWorkspace.x[1] + acadoWorkspace.evGx[188]*acadoWorkspace.x[2] + acadoWorkspace.d[62];
acadoVariables.x[66] += + acadoWorkspace.evGx[189]*acadoWorkspace.x[0] + acadoWorkspace.evGx[190]*acadoWorkspace.x[1] + acadoWorkspace.evGx[191]*acadoWorkspace.x[2] + acadoWorkspace.d[63];
acadoVariables.x[67] += + acadoWorkspace.evGx[192]*acadoWorkspace.x[0] + acadoWorkspace.evGx[193]*acadoWorkspace.x[1] + acadoWorkspace.evGx[194]*acadoWorkspace.x[2] + acadoWorkspace.d[64];
acadoVariables.x[68] += + acadoWorkspace.evGx[195]*acadoWorkspace.x[0] + acadoWorkspace.evGx[196]*acadoWorkspace.x[1] + acadoWorkspace.evGx[197]*acadoWorkspace.x[2] + acadoWorkspace.d[65];
acadoVariables.x[69] += + acadoWorkspace.evGx[198]*acadoWorkspace.x[0] + acadoWorkspace.evGx[199]*acadoWorkspace.x[1] + acadoWorkspace.evGx[200]*acadoWorkspace.x[2] + acadoWorkspace.d[66];
acadoVariables.x[70] += + acadoWorkspace.evGx[201]*acadoWorkspace.x[0] + acadoWorkspace.evGx[202]*acadoWorkspace.x[1] + acadoWorkspace.evGx[203]*acadoWorkspace.x[2] + acadoWorkspace.d[67];
acadoVariables.x[71] += + acadoWorkspace.evGx[204]*acadoWorkspace.x[0] + acadoWorkspace.evGx[205]*acadoWorkspace.x[1] + acadoWorkspace.evGx[206]*acadoWorkspace.x[2] + acadoWorkspace.d[68];
acadoVariables.x[72] += + acadoWorkspace.evGx[207]*acadoWorkspace.x[0] + acadoWorkspace.evGx[208]*acadoWorkspace.x[1] + acadoWorkspace.evGx[209]*acadoWorkspace.x[2] + acadoWorkspace.d[69];
acadoVariables.x[73] += + acadoWorkspace.evGx[210]*acadoWorkspace.x[0] + acadoWorkspace.evGx[211]*acadoWorkspace.x[1] + acadoWorkspace.evGx[212]*acadoWorkspace.x[2] + acadoWorkspace.d[70];
acadoVariables.x[74] += + acadoWorkspace.evGx[213]*acadoWorkspace.x[0] + acadoWorkspace.evGx[214]*acadoWorkspace.x[1] + acadoWorkspace.evGx[215]*acadoWorkspace.x[2] + acadoWorkspace.d[71];
acadoVariables.x[75] += + acadoWorkspace.evGx[216]*acadoWorkspace.x[0] + acadoWorkspace.evGx[217]*acadoWorkspace.x[1] + acadoWorkspace.evGx[218]*acadoWorkspace.x[2] + acadoWorkspace.d[72];
acadoVariables.x[76] += + acadoWorkspace.evGx[219]*acadoWorkspace.x[0] + acadoWorkspace.evGx[220]*acadoWorkspace.x[1] + acadoWorkspace.evGx[221]*acadoWorkspace.x[2] + acadoWorkspace.d[73];
acadoVariables.x[77] += + acadoWorkspace.evGx[222]*acadoWorkspace.x[0] + acadoWorkspace.evGx[223]*acadoWorkspace.x[1] + acadoWorkspace.evGx[224]*acadoWorkspace.x[2] + acadoWorkspace.d[74];
acadoVariables.x[78] += + acadoWorkspace.evGx[225]*acadoWorkspace.x[0] + acadoWorkspace.evGx[226]*acadoWorkspace.x[1] + acadoWorkspace.evGx[227]*acadoWorkspace.x[2] + acadoWorkspace.d[75];
acadoVariables.x[79] += + acadoWorkspace.evGx[228]*acadoWorkspace.x[0] + acadoWorkspace.evGx[229]*acadoWorkspace.x[1] + acadoWorkspace.evGx[230]*acadoWorkspace.x[2] + acadoWorkspace.d[76];
acadoVariables.x[80] += + acadoWorkspace.evGx[231]*acadoWorkspace.x[0] + acadoWorkspace.evGx[232]*acadoWorkspace.x[1] + acadoWorkspace.evGx[233]*acadoWorkspace.x[2] + acadoWorkspace.d[77];
acadoVariables.x[81] += + acadoWorkspace.evGx[234]*acadoWorkspace.x[0] + acadoWorkspace.evGx[235]*acadoWorkspace.x[1] + acadoWorkspace.evGx[236]*acadoWorkspace.x[2] + acadoWorkspace.d[78];
acadoVariables.x[82] += + acadoWorkspace.evGx[237]*acadoWorkspace.x[0] + acadoWorkspace.evGx[238]*acadoWorkspace.x[1] + acadoWorkspace.evGx[239]*acadoWorkspace.x[2] + acadoWorkspace.d[79];
acadoVariables.x[83] += + acadoWorkspace.evGx[240]*acadoWorkspace.x[0] + acadoWorkspace.evGx[241]*acadoWorkspace.x[1] + acadoWorkspace.evGx[242]*acadoWorkspace.x[2] + acadoWorkspace.d[80];
acadoVariables.x[84] += + acadoWorkspace.evGx[243]*acadoWorkspace.x[0] + acadoWorkspace.evGx[244]*acadoWorkspace.x[1] + acadoWorkspace.evGx[245]*acadoWorkspace.x[2] + acadoWorkspace.d[81];
acadoVariables.x[85] += + acadoWorkspace.evGx[246]*acadoWorkspace.x[0] + acadoWorkspace.evGx[247]*acadoWorkspace.x[1] + acadoWorkspace.evGx[248]*acadoWorkspace.x[2] + acadoWorkspace.d[82];
acadoVariables.x[86] += + acadoWorkspace.evGx[249]*acadoWorkspace.x[0] + acadoWorkspace.evGx[250]*acadoWorkspace.x[1] + acadoWorkspace.evGx[251]*acadoWorkspace.x[2] + acadoWorkspace.d[83];
acadoVariables.x[87] += + acadoWorkspace.evGx[252]*acadoWorkspace.x[0] + acadoWorkspace.evGx[253]*acadoWorkspace.x[1] + acadoWorkspace.evGx[254]*acadoWorkspace.x[2] + acadoWorkspace.d[84];
acadoVariables.x[88] += + acadoWorkspace.evGx[255]*acadoWorkspace.x[0] + acadoWorkspace.evGx[256]*acadoWorkspace.x[1] + acadoWorkspace.evGx[257]*acadoWorkspace.x[2] + acadoWorkspace.d[85];
acadoVariables.x[89] += + acadoWorkspace.evGx[258]*acadoWorkspace.x[0] + acadoWorkspace.evGx[259]*acadoWorkspace.x[1] + acadoWorkspace.evGx[260]*acadoWorkspace.x[2] + acadoWorkspace.d[86];
acadoVariables.x[90] += + acadoWorkspace.evGx[261]*acadoWorkspace.x[0] + acadoWorkspace.evGx[262]*acadoWorkspace.x[1] + acadoWorkspace.evGx[263]*acadoWorkspace.x[2] + acadoWorkspace.d[87];
acadoVariables.x[91] += + acadoWorkspace.evGx[264]*acadoWorkspace.x[0] + acadoWorkspace.evGx[265]*acadoWorkspace.x[1] + acadoWorkspace.evGx[266]*acadoWorkspace.x[2] + acadoWorkspace.d[88];
acadoVariables.x[92] += + acadoWorkspace.evGx[267]*acadoWorkspace.x[0] + acadoWorkspace.evGx[268]*acadoWorkspace.x[1] + acadoWorkspace.evGx[269]*acadoWorkspace.x[2] + acadoWorkspace.d[89];
acadoVariables.x[93] += + acadoWorkspace.evGx[270]*acadoWorkspace.x[0] + acadoWorkspace.evGx[271]*acadoWorkspace.x[1] + acadoWorkspace.evGx[272]*acadoWorkspace.x[2] + acadoWorkspace.d[90];
acadoVariables.x[94] += + acadoWorkspace.evGx[273]*acadoWorkspace.x[0] + acadoWorkspace.evGx[274]*acadoWorkspace.x[1] + acadoWorkspace.evGx[275]*acadoWorkspace.x[2] + acadoWorkspace.d[91];
acadoVariables.x[95] += + acadoWorkspace.evGx[276]*acadoWorkspace.x[0] + acadoWorkspace.evGx[277]*acadoWorkspace.x[1] + acadoWorkspace.evGx[278]*acadoWorkspace.x[2] + acadoWorkspace.d[92];
acadoVariables.x[96] += + acadoWorkspace.evGx[279]*acadoWorkspace.x[0] + acadoWorkspace.evGx[280]*acadoWorkspace.x[1] + acadoWorkspace.evGx[281]*acadoWorkspace.x[2] + acadoWorkspace.d[93];
acadoVariables.x[97] += + acadoWorkspace.evGx[282]*acadoWorkspace.x[0] + acadoWorkspace.evGx[283]*acadoWorkspace.x[1] + acadoWorkspace.evGx[284]*acadoWorkspace.x[2] + acadoWorkspace.d[94];
acadoVariables.x[98] += + acadoWorkspace.evGx[285]*acadoWorkspace.x[0] + acadoWorkspace.evGx[286]*acadoWorkspace.x[1] + acadoWorkspace.evGx[287]*acadoWorkspace.x[2] + acadoWorkspace.d[95];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multEDu( &(acadoWorkspace.E[ lRun3 * 3 ]), &(acadoWorkspace.x[ lRun2 + 3 ]), &(acadoVariables.x[ lRun1 * 3 + 3 ]) );
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
acadoWorkspace.state[0] = acadoVariables.x[index * 3];
acadoWorkspace.state[1] = acadoVariables.x[index * 3 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 3 + 2];
acadoWorkspace.state[15] = acadoVariables.u[index];
acadoWorkspace.state[16] = acadoVariables.od[index * 2];
acadoWorkspace.state[17] = acadoVariables.od[index * 2 + 1];

acado_integrate(acadoWorkspace.state, index == 0, index);

acadoVariables.x[index * 3 + 3] = acadoWorkspace.state[0];
acadoVariables.x[index * 3 + 4] = acadoWorkspace.state[1];
acadoVariables.x[index * 3 + 5] = acadoWorkspace.state[2];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 32; ++index)
{
acadoVariables.x[index * 3] = acadoVariables.x[index * 3 + 3];
acadoVariables.x[index * 3 + 1] = acadoVariables.x[index * 3 + 4];
acadoVariables.x[index * 3 + 2] = acadoVariables.x[index * 3 + 5];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[96] = xEnd[0];
acadoVariables.x[97] = xEnd[1];
acadoVariables.x[98] = xEnd[2];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[96];
acadoWorkspace.state[1] = acadoVariables.x[97];
acadoWorkspace.state[2] = acadoVariables.x[98];
if (uEnd != 0)
{
acadoWorkspace.state[15] = uEnd[0];
}
else
{
acadoWorkspace.state[15] = acadoVariables.u[31];
}
acadoWorkspace.state[16] = acadoVariables.od[64];
acadoWorkspace.state[17] = acadoVariables.od[65];

acado_integrate(acadoWorkspace.state, 1, 31);

acadoVariables.x[96] = acadoWorkspace.state[0];
acadoVariables.x[97] = acadoWorkspace.state[1];
acadoVariables.x[98] = acadoWorkspace.state[2];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 31; ++index)
{
acadoVariables.u[index] = acadoVariables.u[index + 1];
}

if (uEnd != 0)
{
acadoVariables.u[31] = uEnd[0];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34];
kkt = fabs( kkt );
for (index = 0; index < 35; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 96; ++index)
{
prd = acadoWorkspace.y[index + 35];
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
/** Row vector of size: 4 */
real_t tmpDy[ 4 ];

/** Row vector of size: 3 */
real_t tmpDyN[ 3 ];

for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 3];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 3 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 3 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.u[lRun1];
acadoWorkspace.objValueIn[4] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.od[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 4] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 4];
acadoWorkspace.Dy[lRun1 * 4 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 4 + 1];
acadoWorkspace.Dy[lRun1 * 4 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 4 + 2];
acadoWorkspace.Dy[lRun1 * 4 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 4 + 3];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[96];
acadoWorkspace.objValueIn[1] = acadoVariables.x[97];
acadoWorkspace.objValueIn[2] = acadoVariables.x[98];
acadoWorkspace.objValueIn[3] = acadoVariables.od[64];
acadoWorkspace.objValueIn[4] = acadoVariables.od[65];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 32; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 4] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 8] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 12];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 1] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 5] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 9] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 13];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 2] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 6] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 10] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 14];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 3] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 7] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 11] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 15];
objVal += + acadoWorkspace.Dy[lRun1 * 4]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 4 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 4 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 4 + 3]*tmpDy[3];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0] + acadoWorkspace.DyN[1]*acadoVariables.WN[3] + acadoWorkspace.DyN[2]*acadoVariables.WN[6];
tmpDyN[1] = + acadoWorkspace.DyN[0]*acadoVariables.WN[1] + acadoWorkspace.DyN[1]*acadoVariables.WN[4] + acadoWorkspace.DyN[2]*acadoVariables.WN[7];
tmpDyN[2] = + acadoWorkspace.DyN[0]*acadoVariables.WN[2] + acadoWorkspace.DyN[1]*acadoVariables.WN[5] + acadoWorkspace.DyN[2]*acadoVariables.WN[8];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2];

objVal *= 0.5;
return objVal;
}

