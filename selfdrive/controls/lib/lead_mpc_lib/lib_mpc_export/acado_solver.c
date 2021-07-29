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
const real_t* od = in + 4;
/* Vector of auxiliary variables; number of elements: 13. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[1] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[2] = ((real_t)(1.0000000000000000e+00)/(a[0]+(real_t)(1.0000000000000001e-01)));
a[3] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[4] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[2]))*a[3]);
a[5] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[6] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[7] = (a[6]*(real_t)(5.0000000000000000e-01));
a[8] = (a[2]*a[2]);
a[9] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[5]))*a[2])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))*a[7])*a[8])))*a[3]);
a[10] = ((real_t)(1.0000000000000000e+00)/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
a[11] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[12] = (a[10]*a[10]);

/* Compute outputs: */
out[0] = (a[1]-(real_t)(1.0000000000000000e+00));
out[1] = (((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))))/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = (u[0]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[4] = a[4];
out[5] = a[9];
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[10]);
out[8] = ((((real_t)(0.0000000000000000e+00)-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[11])))*a[10])-((((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))))*(real_t)(5.0000000000000003e-02))*a[12]));
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[12] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[13] = (real_t)(0.0000000000000000e+00);
out[14] = (u[0]*(real_t)(1.0000000000000001e-01));
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = (real_t)(0.0000000000000000e+00);
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 3;
/* Vector of auxiliary variables; number of elements: 13. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[1] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[2] = ((real_t)(1.0000000000000000e+00)/(a[0]+(real_t)(1.0000000000000001e-01)));
a[3] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[4] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[2]))*a[3]);
a[5] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[6] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[7] = (a[6]*(real_t)(5.0000000000000000e-01));
a[8] = (a[2]*a[2]);
a[9] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[5]))*a[2])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(od[0]-xd[0]))*a[7])*a[8])))*a[3]);
a[10] = ((real_t)(1.0000000000000000e+00)/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
a[11] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[12] = (a[10]*a[10]);

/* Compute outputs: */
out[0] = (a[1]-(real_t)(1.0000000000000000e+00));
out[1] = (((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))))/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = a[4];
out[4] = a[9];
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[10]);
out[7] = ((((real_t)(0.0000000000000000e+00)-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[11])))*a[10])-((((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01)))))*(real_t)(5.0000000000000003e-02))*a[12]));
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[11] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0]*tmpObjS[0] + tmpFx[3]*tmpObjS[4] + tmpFx[6]*tmpObjS[8] + tmpFx[9]*tmpObjS[12];
tmpQ2[1] = + tmpFx[0]*tmpObjS[1] + tmpFx[3]*tmpObjS[5] + tmpFx[6]*tmpObjS[9] + tmpFx[9]*tmpObjS[13];
tmpQ2[2] = + tmpFx[0]*tmpObjS[2] + tmpFx[3]*tmpObjS[6] + tmpFx[6]*tmpObjS[10] + tmpFx[9]*tmpObjS[14];
tmpQ2[3] = + tmpFx[0]*tmpObjS[3] + tmpFx[3]*tmpObjS[7] + tmpFx[6]*tmpObjS[11] + tmpFx[9]*tmpObjS[15];
tmpQ2[4] = + tmpFx[1]*tmpObjS[0] + tmpFx[4]*tmpObjS[4] + tmpFx[7]*tmpObjS[8] + tmpFx[10]*tmpObjS[12];
tmpQ2[5] = + tmpFx[1]*tmpObjS[1] + tmpFx[4]*tmpObjS[5] + tmpFx[7]*tmpObjS[9] + tmpFx[10]*tmpObjS[13];
tmpQ2[6] = + tmpFx[1]*tmpObjS[2] + tmpFx[4]*tmpObjS[6] + tmpFx[7]*tmpObjS[10] + tmpFx[10]*tmpObjS[14];
tmpQ2[7] = + tmpFx[1]*tmpObjS[3] + tmpFx[4]*tmpObjS[7] + tmpFx[7]*tmpObjS[11] + tmpFx[10]*tmpObjS[15];
tmpQ2[8] = + tmpFx[2]*tmpObjS[0] + tmpFx[5]*tmpObjS[4] + tmpFx[8]*tmpObjS[8] + tmpFx[11]*tmpObjS[12];
tmpQ2[9] = + tmpFx[2]*tmpObjS[1] + tmpFx[5]*tmpObjS[5] + tmpFx[8]*tmpObjS[9] + tmpFx[11]*tmpObjS[13];
tmpQ2[10] = + tmpFx[2]*tmpObjS[2] + tmpFx[5]*tmpObjS[6] + tmpFx[8]*tmpObjS[10] + tmpFx[11]*tmpObjS[14];
tmpQ2[11] = + tmpFx[2]*tmpObjS[3] + tmpFx[5]*tmpObjS[7] + tmpFx[8]*tmpObjS[11] + tmpFx[11]*tmpObjS[15];
tmpQ1[0] = + tmpQ2[0]*tmpFx[0] + tmpQ2[1]*tmpFx[3] + tmpQ2[2]*tmpFx[6] + tmpQ2[3]*tmpFx[9];
tmpQ1[1] = + tmpQ2[0]*tmpFx[1] + tmpQ2[1]*tmpFx[4] + tmpQ2[2]*tmpFx[7] + tmpQ2[3]*tmpFx[10];
tmpQ1[2] = + tmpQ2[0]*tmpFx[2] + tmpQ2[1]*tmpFx[5] + tmpQ2[2]*tmpFx[8] + tmpQ2[3]*tmpFx[11];
tmpQ1[3] = + tmpQ2[4]*tmpFx[0] + tmpQ2[5]*tmpFx[3] + tmpQ2[6]*tmpFx[6] + tmpQ2[7]*tmpFx[9];
tmpQ1[4] = + tmpQ2[4]*tmpFx[1] + tmpQ2[5]*tmpFx[4] + tmpQ2[6]*tmpFx[7] + tmpQ2[7]*tmpFx[10];
tmpQ1[5] = + tmpQ2[4]*tmpFx[2] + tmpQ2[5]*tmpFx[5] + tmpQ2[6]*tmpFx[8] + tmpQ2[7]*tmpFx[11];
tmpQ1[6] = + tmpQ2[8]*tmpFx[0] + tmpQ2[9]*tmpFx[3] + tmpQ2[10]*tmpFx[6] + tmpQ2[11]*tmpFx[9];
tmpQ1[7] = + tmpQ2[8]*tmpFx[1] + tmpQ2[9]*tmpFx[4] + tmpQ2[10]*tmpFx[7] + tmpQ2[11]*tmpFx[10];
tmpQ1[8] = + tmpQ2[8]*tmpFx[2] + tmpQ2[9]*tmpFx[5] + tmpQ2[10]*tmpFx[8] + tmpQ2[11]*tmpFx[11];
}

void acado_setObjR1R2( real_t* const tmpFu, real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = + tmpFu[0]*tmpObjS[0] + tmpFu[1]*tmpObjS[4] + tmpFu[2]*tmpObjS[8] + tmpFu[3]*tmpObjS[12];
tmpR2[1] = + tmpFu[0]*tmpObjS[1] + tmpFu[1]*tmpObjS[5] + tmpFu[2]*tmpObjS[9] + tmpFu[3]*tmpObjS[13];
tmpR2[2] = + tmpFu[0]*tmpObjS[2] + tmpFu[1]*tmpObjS[6] + tmpFu[2]*tmpObjS[10] + tmpFu[3]*tmpObjS[14];
tmpR2[3] = + tmpFu[0]*tmpObjS[3] + tmpFu[1]*tmpObjS[7] + tmpFu[2]*tmpObjS[11] + tmpFu[3]*tmpObjS[15];
tmpR1[0] = + tmpR2[0]*tmpFu[0] + tmpR2[1]*tmpFu[1] + tmpR2[2]*tmpFu[2] + tmpR2[3]*tmpFu[3];
}

void acado_setObjQN1QN2( real_t* const tmpFx, real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = + tmpFx[0]*tmpObjSEndTerm[0] + tmpFx[3]*tmpObjSEndTerm[3] + tmpFx[6]*tmpObjSEndTerm[6];
tmpQN2[1] = + tmpFx[0]*tmpObjSEndTerm[1] + tmpFx[3]*tmpObjSEndTerm[4] + tmpFx[6]*tmpObjSEndTerm[7];
tmpQN2[2] = + tmpFx[0]*tmpObjSEndTerm[2] + tmpFx[3]*tmpObjSEndTerm[5] + tmpFx[6]*tmpObjSEndTerm[8];
tmpQN2[3] = + tmpFx[1]*tmpObjSEndTerm[0] + tmpFx[4]*tmpObjSEndTerm[3] + tmpFx[7]*tmpObjSEndTerm[6];
tmpQN2[4] = + tmpFx[1]*tmpObjSEndTerm[1] + tmpFx[4]*tmpObjSEndTerm[4] + tmpFx[7]*tmpObjSEndTerm[7];
tmpQN2[5] = + tmpFx[1]*tmpObjSEndTerm[2] + tmpFx[4]*tmpObjSEndTerm[5] + tmpFx[7]*tmpObjSEndTerm[8];
tmpQN2[6] = + tmpFx[2]*tmpObjSEndTerm[0] + tmpFx[5]*tmpObjSEndTerm[3] + tmpFx[8]*tmpObjSEndTerm[6];
tmpQN2[7] = + tmpFx[2]*tmpObjSEndTerm[1] + tmpFx[5]*tmpObjSEndTerm[4] + tmpFx[8]*tmpObjSEndTerm[7];
tmpQN2[8] = + tmpFx[2]*tmpObjSEndTerm[2] + tmpFx[5]*tmpObjSEndTerm[5] + tmpFx[8]*tmpObjSEndTerm[8];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0] + tmpQN2[1]*tmpFx[3] + tmpQN2[2]*tmpFx[6];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1] + tmpQN2[1]*tmpFx[4] + tmpQN2[2]*tmpFx[7];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2] + tmpQN2[1]*tmpFx[5] + tmpQN2[2]*tmpFx[8];
tmpQN1[3] = + tmpQN2[3]*tmpFx[0] + tmpQN2[4]*tmpFx[3] + tmpQN2[5]*tmpFx[6];
tmpQN1[4] = + tmpQN2[3]*tmpFx[1] + tmpQN2[4]*tmpFx[4] + tmpQN2[5]*tmpFx[7];
tmpQN1[5] = + tmpQN2[3]*tmpFx[2] + tmpQN2[4]*tmpFx[5] + tmpQN2[5]*tmpFx[8];
tmpQN1[6] = + tmpQN2[6]*tmpFx[0] + tmpQN2[7]*tmpFx[3] + tmpQN2[8]*tmpFx[6];
tmpQN1[7] = + tmpQN2[6]*tmpFx[1] + tmpQN2[7]*tmpFx[4] + tmpQN2[8]*tmpFx[7];
tmpQN1[8] = + tmpQN2[6]*tmpFx[2] + tmpQN2[7]*tmpFx[5] + tmpQN2[8]*tmpFx[8];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 20; ++runObj)
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

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 4 ]), &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.Q1[ runObj * 9 ]), &(acadoWorkspace.Q2[ runObj * 12 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 16 ]), &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.R1[ runObj ]), &(acadoWorkspace.R2[ runObj * 4 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[60];
acadoWorkspace.objValueIn[1] = acadoVariables.x[61];
acadoWorkspace.objValueIn[2] = acadoVariables.x[62];
acadoWorkspace.objValueIn[3] = acadoVariables.od[40];
acadoWorkspace.objValueIn[4] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 3 ]), acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

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
acadoWorkspace.H[(iRow * 23 + 69) + (iCol + 3)] += + Gu1[0]*Gu2[0] + Gu1[1]*Gu2[1] + Gu1[2]*Gu2[2];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 23 + 69) + (iCol + 3)] = R11[0];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 23 + 69) + (iCol + 3)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 23 + 69) + (iCol + 3)] = acadoWorkspace.H[(iCol * 23 + 69) + (iRow + 3)];
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
acadoWorkspace.H[23] = 0.0000000000000000e+00;
acadoWorkspace.H[24] = 0.0000000000000000e+00;
acadoWorkspace.H[25] = 0.0000000000000000e+00;
acadoWorkspace.H[46] = 0.0000000000000000e+00;
acadoWorkspace.H[47] = 0.0000000000000000e+00;
acadoWorkspace.H[48] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[3]*Gx2[3] + Gx1[6]*Gx2[6];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[3]*Gx2[4] + Gx1[6]*Gx2[7];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[3]*Gx2[5] + Gx1[6]*Gx2[8];
acadoWorkspace.H[23] += + Gx1[1]*Gx2[0] + Gx1[4]*Gx2[3] + Gx1[7]*Gx2[6];
acadoWorkspace.H[24] += + Gx1[1]*Gx2[1] + Gx1[4]*Gx2[4] + Gx1[7]*Gx2[7];
acadoWorkspace.H[25] += + Gx1[1]*Gx2[2] + Gx1[4]*Gx2[5] + Gx1[7]*Gx2[8];
acadoWorkspace.H[46] += + Gx1[2]*Gx2[0] + Gx1[5]*Gx2[3] + Gx1[8]*Gx2[6];
acadoWorkspace.H[47] += + Gx1[2]*Gx2[1] + Gx1[5]*Gx2[4] + Gx1[8]*Gx2[7];
acadoWorkspace.H[48] += + Gx1[2]*Gx2[2] + Gx1[5]*Gx2[5] + Gx1[8]*Gx2[8];
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
/** Row vector of size: 20 */
static const int xBoundIndices[ 20 ] = 
{ 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_moveGxT( &(acadoWorkspace.evGx[ 9 ]), acadoWorkspace.T );
acado_multGxd( acadoWorkspace.d, &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.d[ 3 ]) );
acado_multGxGx( acadoWorkspace.T, acadoWorkspace.evGx, &(acadoWorkspace.evGx[ 9 ]) );

acado_multGxGu( acadoWorkspace.T, acadoWorkspace.E, &(acadoWorkspace.E[ 3 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 3 ]), &(acadoWorkspace.E[ 6 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 18 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 3 ]), &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.d[ 6 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.evGx[ 18 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.E[ 9 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.E[ 12 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 6 ]), &(acadoWorkspace.E[ 15 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 27 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 6 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.d[ 9 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.evGx[ 27 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.E[ 18 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.E[ 21 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.E[ 24 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 9 ]), &(acadoWorkspace.E[ 27 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 9 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.d[ 12 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.evGx[ 36 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.E[ 30 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.E[ 33 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.E[ 36 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 27 ]), &(acadoWorkspace.E[ 39 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 12 ]), &(acadoWorkspace.E[ 42 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 45 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.d[ 15 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.evGx[ 45 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.E[ 45 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.E[ 48 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.E[ 51 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.E[ 54 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.E[ 57 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 15 ]), &(acadoWorkspace.E[ 60 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 54 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 15 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.d[ 18 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.evGx[ 54 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.E[ 63 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.E[ 66 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.E[ 69 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.E[ 72 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.E[ 75 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.E[ 78 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 18 ]), &(acadoWorkspace.E[ 81 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 63 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 18 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.d[ 21 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.evGx[ 63 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.E[ 84 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.E[ 87 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.E[ 90 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.E[ 93 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.E[ 96 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.E[ 99 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 81 ]), &(acadoWorkspace.E[ 102 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 21 ]), &(acadoWorkspace.E[ 105 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 21 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.d[ 24 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.evGx[ 72 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.E[ 108 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.E[ 111 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.E[ 114 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.E[ 117 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.E[ 120 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.E[ 123 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.E[ 126 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.E[ 129 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.E[ 132 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 81 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.d[ 27 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.evGx[ 81 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.E[ 135 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.E[ 138 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.E[ 141 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.E[ 144 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.E[ 147 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.E[ 150 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.E[ 153 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.E[ 156 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.E[ 159 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 27 ]), &(acadoWorkspace.E[ 162 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 90 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 27 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.d[ 30 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.evGx[ 90 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.E[ 165 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.E[ 168 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.E[ 171 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.E[ 174 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.E[ 177 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.E[ 180 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.E[ 183 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.E[ 186 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.E[ 189 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.E[ 192 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 30 ]), &(acadoWorkspace.E[ 195 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 99 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.d[ 33 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.evGx[ 99 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.E[ 198 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.E[ 201 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.E[ 204 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.E[ 207 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.E[ 210 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.E[ 213 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.E[ 216 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.E[ 219 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.E[ 222 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.E[ 225 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.E[ 228 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 33 ]), &(acadoWorkspace.E[ 231 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 33 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.d[ 36 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.evGx[ 108 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.E[ 234 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.E[ 237 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.E[ 240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.E[ 243 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.E[ 246 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.E[ 249 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.E[ 252 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.E[ 255 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.E[ 258 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.E[ 261 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.E[ 264 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 231 ]), &(acadoWorkspace.E[ 267 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 36 ]), &(acadoWorkspace.E[ 270 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 117 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.d[ 39 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.evGx[ 117 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.E[ 273 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.E[ 276 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.E[ 279 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.E[ 282 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.E[ 285 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.E[ 288 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.E[ 291 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.E[ 294 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.E[ 297 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.E[ 300 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.E[ 303 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.E[ 306 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.E[ 309 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 39 ]), &(acadoWorkspace.E[ 312 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 126 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 39 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.d[ 42 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.evGx[ 126 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.E[ 315 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.E[ 318 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.E[ 321 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.E[ 324 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.E[ 327 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.E[ 330 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.E[ 333 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.E[ 336 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.E[ 339 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.E[ 342 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.E[ 345 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.E[ 348 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.E[ 351 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.E[ 354 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 42 ]), &(acadoWorkspace.E[ 357 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 135 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 42 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.d[ 45 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.evGx[ 135 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.E[ 360 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.E[ 363 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.E[ 366 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.E[ 369 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.E[ 372 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.E[ 375 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.E[ 378 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.E[ 381 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.E[ 384 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.E[ 387 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.E[ 390 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.E[ 393 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.E[ 396 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.E[ 399 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 357 ]), &(acadoWorkspace.E[ 402 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 45 ]), &(acadoWorkspace.E[ 405 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 45 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.d[ 48 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.evGx[ 144 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.E[ 408 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.E[ 411 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.E[ 414 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.E[ 417 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.E[ 420 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.E[ 423 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.E[ 426 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.E[ 429 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.E[ 432 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.E[ 435 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.E[ 438 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.E[ 441 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.E[ 444 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.E[ 447 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.E[ 450 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.E[ 453 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 48 ]), &(acadoWorkspace.E[ 456 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 153 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.d[ 51 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.evGx[ 153 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.E[ 459 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.E[ 462 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.E[ 465 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.E[ 468 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.E[ 471 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.E[ 474 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.E[ 477 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.E[ 480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.E[ 483 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.E[ 486 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.E[ 489 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.E[ 492 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.E[ 495 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.E[ 498 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.E[ 501 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.E[ 504 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.E[ 507 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 51 ]), &(acadoWorkspace.E[ 510 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 162 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 51 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.d[ 54 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.evGx[ 162 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.E[ 513 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.E[ 516 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.E[ 519 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.E[ 522 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.E[ 525 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.E[ 528 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.E[ 531 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.E[ 534 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.E[ 537 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.E[ 540 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.E[ 543 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.E[ 546 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.E[ 549 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.E[ 552 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.E[ 555 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.E[ 558 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.E[ 561 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.E[ 564 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 54 ]), &(acadoWorkspace.E[ 567 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 171 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 54 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.d[ 57 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.evGx[ 171 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.E[ 570 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.E[ 573 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.E[ 576 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.E[ 579 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.E[ 582 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.E[ 585 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.E[ 588 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.E[ 591 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.E[ 594 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.E[ 597 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.E[ 600 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.E[ 603 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.E[ 606 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.E[ 609 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.E[ 612 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.E[ 615 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.E[ 618 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.E[ 621 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 567 ]), &(acadoWorkspace.E[ 624 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 57 ]), &(acadoWorkspace.E[ 627 ]) );

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
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.QGx[ 171 ]) );

acado_multGxGu( &(acadoWorkspace.Q1[ 9 ]), acadoWorkspace.E, acadoWorkspace.QE );
acado_multGxGu( &(acadoWorkspace.Q1[ 18 ]), &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.QE[ 3 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 18 ]), &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QE[ 6 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 27 ]), &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.QE[ 9 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 27 ]), &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 27 ]), &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 18 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.QE[ 21 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 36 ]), &(acadoWorkspace.E[ 27 ]), &(acadoWorkspace.QE[ 27 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QE[ 33 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.QE[ 39 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 45 ]), &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 45 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QE[ 51 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 54 ]), &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 63 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 69 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 63 ]), &(acadoWorkspace.E[ 81 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 87 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 93 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 111 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 117 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 81 ]), &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 141 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 90 ]), &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 165 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 171 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 99 ]), &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 201 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 207 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 231 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 237 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 243 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 117 ]), &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 273 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 279 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 126 ]), &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 315 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 321 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 135 ]), &(acadoWorkspace.E[ 357 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 363 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 369 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 411 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 417 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 153 ]), &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 459 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 162 ]), &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 513 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 519 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 171 ]), &(acadoWorkspace.E[ 567 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 573 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 579 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 585 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 591 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 597 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 603 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 609 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 615 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.QE[ 621 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 627 ]), &(acadoWorkspace.QE[ 627 ]) );

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

acado_zeroBlockH10( acadoWorkspace.H10 );
acado_multQETGx( acadoWorkspace.QE, acadoWorkspace.evGx, acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 3 ]), &(acadoWorkspace.evGx[ 9 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 9 ]), &(acadoWorkspace.evGx[ 18 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 18 ]), &(acadoWorkspace.evGx[ 27 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 45 ]), &(acadoWorkspace.evGx[ 45 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 63 ]), &(acadoWorkspace.evGx[ 54 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.evGx[ 63 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 135 ]), &(acadoWorkspace.evGx[ 81 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 165 ]), &(acadoWorkspace.evGx[ 90 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 198 ]), &(acadoWorkspace.evGx[ 99 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 234 ]), &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 273 ]), &(acadoWorkspace.evGx[ 117 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 315 ]), &(acadoWorkspace.evGx[ 126 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.evGx[ 135 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 459 ]), &(acadoWorkspace.evGx[ 153 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 513 ]), &(acadoWorkspace.evGx[ 162 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.evGx[ 171 ]), acadoWorkspace.H10 );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 6 ]), &(acadoWorkspace.evGx[ 9 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 21 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 33 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 66 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 87 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 111 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 138 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 201 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 237 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 318 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 363 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 411 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 462 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 573 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 3 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 15 ]), &(acadoWorkspace.evGx[ 18 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 51 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 69 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 114 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 141 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 171 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 279 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 321 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 366 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 414 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 465 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 519 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 27 ]), &(acadoWorkspace.evGx[ 27 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 39 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 54 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 93 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 117 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 174 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 207 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 243 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 282 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 369 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 417 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 522 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 579 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 9 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 42 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 57 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 75 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 147 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 177 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 246 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 285 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 327 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 471 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 525 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 582 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.evGx[ 45 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 78 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 99 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 123 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 213 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 249 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 375 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 423 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 474 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 585 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 15 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 81 ]), &(acadoWorkspace.evGx[ 54 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 102 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 126 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 153 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 183 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 291 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 333 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 378 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 426 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 477 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 531 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 588 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 105 ]), &(acadoWorkspace.evGx[ 63 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 129 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 186 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 219 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 255 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 294 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 381 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 429 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 534 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 591 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 21 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 159 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 189 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 222 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 258 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 297 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 339 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 483 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 537 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 594 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 162 ]), &(acadoWorkspace.evGx[ 81 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 225 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 261 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 342 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 387 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 435 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 486 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 597 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 27 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 195 ]), &(acadoWorkspace.evGx[ 90 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 303 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 345 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 438 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 489 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 543 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 231 ]), &(acadoWorkspace.evGx[ 99 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 267 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 306 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 393 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 441 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 546 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 603 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 33 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 309 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 351 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 495 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 549 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 606 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.evGx[ 117 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 354 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 399 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 447 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 498 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 609 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 39 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 357 ]), &(acadoWorkspace.evGx[ 126 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 402 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 501 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 555 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 612 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 405 ]), &(acadoWorkspace.evGx[ 135 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 453 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 558 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 615 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 45 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 507 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 561 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 618 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 51 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.evGx[ 153 ]), &(acadoWorkspace.H10[ 51 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 564 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 51 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 621 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 51 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 567 ]), &(acadoWorkspace.evGx[ 162 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 57 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 627 ]), &(acadoWorkspace.evGx[ 171 ]), &(acadoWorkspace.H10[ 57 ]) );

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
acadoWorkspace.H[26] = acadoWorkspace.H10[1];
acadoWorkspace.H[27] = acadoWorkspace.H10[4];
acadoWorkspace.H[28] = acadoWorkspace.H10[7];
acadoWorkspace.H[29] = acadoWorkspace.H10[10];
acadoWorkspace.H[30] = acadoWorkspace.H10[13];
acadoWorkspace.H[31] = acadoWorkspace.H10[16];
acadoWorkspace.H[32] = acadoWorkspace.H10[19];
acadoWorkspace.H[33] = acadoWorkspace.H10[22];
acadoWorkspace.H[34] = acadoWorkspace.H10[25];
acadoWorkspace.H[35] = acadoWorkspace.H10[28];
acadoWorkspace.H[36] = acadoWorkspace.H10[31];
acadoWorkspace.H[37] = acadoWorkspace.H10[34];
acadoWorkspace.H[38] = acadoWorkspace.H10[37];
acadoWorkspace.H[39] = acadoWorkspace.H10[40];
acadoWorkspace.H[40] = acadoWorkspace.H10[43];
acadoWorkspace.H[41] = acadoWorkspace.H10[46];
acadoWorkspace.H[42] = acadoWorkspace.H10[49];
acadoWorkspace.H[43] = acadoWorkspace.H10[52];
acadoWorkspace.H[44] = acadoWorkspace.H10[55];
acadoWorkspace.H[45] = acadoWorkspace.H10[58];
acadoWorkspace.H[49] = acadoWorkspace.H10[2];
acadoWorkspace.H[50] = acadoWorkspace.H10[5];
acadoWorkspace.H[51] = acadoWorkspace.H10[8];
acadoWorkspace.H[52] = acadoWorkspace.H10[11];
acadoWorkspace.H[53] = acadoWorkspace.H10[14];
acadoWorkspace.H[54] = acadoWorkspace.H10[17];
acadoWorkspace.H[55] = acadoWorkspace.H10[20];
acadoWorkspace.H[56] = acadoWorkspace.H10[23];
acadoWorkspace.H[57] = acadoWorkspace.H10[26];
acadoWorkspace.H[58] = acadoWorkspace.H10[29];
acadoWorkspace.H[59] = acadoWorkspace.H10[32];
acadoWorkspace.H[60] = acadoWorkspace.H10[35];
acadoWorkspace.H[61] = acadoWorkspace.H10[38];
acadoWorkspace.H[62] = acadoWorkspace.H10[41];
acadoWorkspace.H[63] = acadoWorkspace.H10[44];
acadoWorkspace.H[64] = acadoWorkspace.H10[47];
acadoWorkspace.H[65] = acadoWorkspace.H10[50];
acadoWorkspace.H[66] = acadoWorkspace.H10[53];
acadoWorkspace.H[67] = acadoWorkspace.H10[56];
acadoWorkspace.H[68] = acadoWorkspace.H10[59];

acado_setBlockH11_R1( 0, 0, acadoWorkspace.R1 );
acado_setBlockH11( 0, 0, acadoWorkspace.E, acadoWorkspace.QE );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.QE[ 3 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.QE[ 9 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 18 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 45 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 63 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 135 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 165 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 273 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 315 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 459 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 513 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );

acado_zeroBlockH11( 0, 1 );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.QE[ 6 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 21 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 33 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 87 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 111 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 201 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 237 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 363 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 411 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 573 ]) );

acado_zeroBlockH11( 0, 2 );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 51 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 69 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 141 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 171 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 279 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 321 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 519 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 576 ]) );

acado_zeroBlockH11( 0, 3 );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 27 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 39 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 93 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 117 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 207 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 243 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 369 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 417 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 579 ]) );

acado_zeroBlockH11( 0, 4 );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 582 ]) );

acado_zeroBlockH11( 0, 5 );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 0, 6 );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 0, 7 );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 0, 8 );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 0, 9 );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 0, 10 );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 0, 11 );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 0, 12 );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 0, 13 );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 0, 14 );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 0, 15 );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 0, 16 );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 0, 17 );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 0, 18 );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 0, 19 );
acado_setBlockH11( 0, 19, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 1, 1, &(acadoWorkspace.R1[ 1 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QE[ 6 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.QE[ 21 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QE[ 33 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 87 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 111 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 201 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 237 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 363 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 411 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 573 ]) );

acado_zeroBlockH11( 1, 2 );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 51 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 69 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 141 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 171 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 279 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 321 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 519 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 576 ]) );

acado_zeroBlockH11( 1, 3 );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.QE[ 27 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QE[ 39 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 93 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 117 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 207 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 243 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 369 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 417 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 579 ]) );

acado_zeroBlockH11( 1, 4 );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 582 ]) );

acado_zeroBlockH11( 1, 5 );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 1, 6 );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 1, 7 );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 1, 8 );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 1, 9 );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 1, 10 );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 1, 11 );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 1, 12 );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 1, 13 );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 1, 14 );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 1, 15 );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 1, 16 );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 1, 17 );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 1, 18 );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 1, 19 );
acado_setBlockH11( 1, 19, &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 2, 2, &(acadoWorkspace.R1[ 2 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QE[ 15 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QE[ 51 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 69 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 141 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 171 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 279 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 321 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 465 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 519 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );

acado_zeroBlockH11( 2, 3 );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 27 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 39 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 93 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 117 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 207 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 243 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 369 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 417 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 579 ]) );

acado_zeroBlockH11( 2, 4 );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 582 ]) );

acado_zeroBlockH11( 2, 5 );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 2, 6 );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 2, 7 );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 2, 8 );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 2, 9 );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 2, 10 );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 2, 11 );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 2, 12 );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 2, 13 );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 2, 14 );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 2, 15 );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 2, 16 );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 2, 17 );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 2, 18 );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 2, 19 );
acado_setBlockH11( 2, 19, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 3, 3, &(acadoWorkspace.R1[ 3 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 27 ]), &(acadoWorkspace.QE[ 27 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.QE[ 39 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 93 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 117 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 207 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 243 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 369 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 417 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 579 ]) );

acado_zeroBlockH11( 3, 4 );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 582 ]) );

acado_zeroBlockH11( 3, 5 );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 3, 6 );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 3, 7 );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 3, 8 );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 3, 9 );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 3, 10 );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 3, 11 );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 3, 12 );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 3, 13 );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 3, 14 );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 3, 15 );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 3, 16 );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 3, 17 );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 3, 18 );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 3, 19 );
acado_setBlockH11( 3, 19, &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 4, 4, &(acadoWorkspace.R1[ 4 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.QE[ 57 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 75 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 147 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 177 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 285 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 327 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 471 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 525 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 582 ]) );

acado_zeroBlockH11( 4, 5 );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 4, 6 );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 4, 7 );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 4, 8 );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 4, 9 );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 4, 10 );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 4, 11 );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 4, 12 );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 4, 13 );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 4, 14 );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 4, 15 );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 4, 16 );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 4, 17 );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 4, 18 );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 4, 19 );
acado_setBlockH11( 4, 19, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 5, 5, &(acadoWorkspace.R1[ 5 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.QE[ 99 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QE[ 123 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 213 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 249 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 375 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 423 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 585 ]) );

acado_zeroBlockH11( 5, 6 );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 5, 7 );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 5, 8 );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 5, 9 );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 5, 10 );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 5, 11 );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 5, 12 );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 5, 13 );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 5, 14 );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 5, 15 );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 5, 16 );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 5, 17 );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 5, 18 );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 5, 19 );
acado_setBlockH11( 5, 19, &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 6, 6, &(acadoWorkspace.R1[ 6 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 81 ]), &(acadoWorkspace.QE[ 81 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QE[ 153 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 183 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 291 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 333 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 477 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 531 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 588 ]) );

acado_zeroBlockH11( 6, 7 );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 6, 8 );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 6, 9 );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 6, 10 );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 6, 11 );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 6, 12 );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 6, 13 );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 6, 14 );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 6, 15 );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 6, 16 );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 6, 17 );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 6, 18 );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 6, 19 );
acado_setBlockH11( 6, 19, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 7, 7, &(acadoWorkspace.R1[ 7 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QE[ 105 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.QE[ 129 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 219 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 255 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 381 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 429 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 591 ]) );

acado_zeroBlockH11( 7, 8 );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 7, 9 );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 7, 10 );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 7, 11 );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 7, 12 );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 7, 13 );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 7, 14 );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 7, 15 );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 7, 16 );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 7, 17 );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 7, 18 );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 7, 19 );
acado_setBlockH11( 7, 19, &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 8, 8, &(acadoWorkspace.R1[ 8 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.QE[ 159 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.QE[ 189 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 297 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 339 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 483 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 537 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 594 ]) );

acado_zeroBlockH11( 8, 9 );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 8, 10 );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 8, 11 );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 8, 12 );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 8, 13 );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 8, 14 );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 8, 15 );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 8, 16 );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 8, 17 );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 8, 18 );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 8, 19 );
acado_setBlockH11( 8, 19, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 9, 9, &(acadoWorkspace.R1[ 9 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 225 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QE[ 261 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 387 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 435 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 597 ]) );

acado_zeroBlockH11( 9, 10 );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 9, 11 );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 9, 12 );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 9, 13 );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 9, 14 );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 9, 15 );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 9, 16 );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 9, 17 );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 9, 18 );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 9, 19 );
acado_setBlockH11( 9, 19, &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 10, 10, &(acadoWorkspace.R1[ 10 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QE[ 195 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QE[ 303 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 345 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 489 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 543 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );

acado_zeroBlockH11( 10, 11 );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 10, 12 );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 10, 13 );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 10, 14 );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 10, 15 );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 10, 16 );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 10, 17 );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 10, 18 );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 10, 19 );
acado_setBlockH11( 10, 19, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 11, 11, &(acadoWorkspace.R1[ 11 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 231 ]), &(acadoWorkspace.QE[ 231 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.QE[ 267 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 393 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 441 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 603 ]) );

acado_zeroBlockH11( 11, 12 );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 11, 13 );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 11, 14 );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 11, 15 );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 11, 16 );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 11, 17 );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 11, 18 );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 11, 19 );
acado_setBlockH11( 11, 19, &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 12, 12, &(acadoWorkspace.R1[ 12 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.QE[ 309 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.QE[ 351 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 495 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 549 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 606 ]) );

acado_zeroBlockH11( 12, 13 );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 12, 14 );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 12, 15 );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 12, 16 );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 12, 17 );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 12, 18 );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 12, 19 );
acado_setBlockH11( 12, 19, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 13, 13, &(acadoWorkspace.R1[ 13 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.QE[ 399 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QE[ 447 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 609 ]) );

acado_zeroBlockH11( 13, 14 );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 13, 15 );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 13, 16 );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 13, 17 );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 13, 18 );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 13, 19 );
acado_setBlockH11( 13, 19, &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 14, 14, &(acadoWorkspace.R1[ 14 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 357 ]), &(acadoWorkspace.QE[ 357 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QE[ 501 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 555 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 612 ]) );

acado_zeroBlockH11( 14, 15 );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 14, 16 );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 14, 17 );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 14, 18 );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 14, 19 );
acado_setBlockH11( 14, 19, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 15, 15, &(acadoWorkspace.R1[ 15 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QE[ 405 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.QE[ 453 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 615 ]) );

acado_zeroBlockH11( 15, 16 );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 15, 17 );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 15, 18 );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 15, 19 );
acado_setBlockH11( 15, 19, &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 16, 16, &(acadoWorkspace.R1[ 16 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.QE[ 507 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.QE[ 561 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 618 ]) );

acado_zeroBlockH11( 16, 17 );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 16, 18 );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 16, 19 );
acado_setBlockH11( 16, 19, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 17, 17, &(acadoWorkspace.R1[ 17 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.QE[ 621 ]) );

acado_zeroBlockH11( 17, 18 );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 17, 19 );
acado_setBlockH11( 17, 19, &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 18, 18, &(acadoWorkspace.R1[ 18 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 567 ]), &(acadoWorkspace.QE[ 567 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );

acado_zeroBlockH11( 18, 19 );
acado_setBlockH11( 18, 19, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 627 ]) );

acado_setBlockH11_R1( 19, 19, &(acadoWorkspace.R1[ 19 ]) );
acado_setBlockH11( 19, 19, &(acadoWorkspace.E[ 627 ]), &(acadoWorkspace.QE[ 627 ]) );


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

acadoWorkspace.H[69] = acadoWorkspace.H10[0];
acadoWorkspace.H[70] = acadoWorkspace.H10[1];
acadoWorkspace.H[71] = acadoWorkspace.H10[2];
acadoWorkspace.H[92] = acadoWorkspace.H10[3];
acadoWorkspace.H[93] = acadoWorkspace.H10[4];
acadoWorkspace.H[94] = acadoWorkspace.H10[5];
acadoWorkspace.H[115] = acadoWorkspace.H10[6];
acadoWorkspace.H[116] = acadoWorkspace.H10[7];
acadoWorkspace.H[117] = acadoWorkspace.H10[8];
acadoWorkspace.H[138] = acadoWorkspace.H10[9];
acadoWorkspace.H[139] = acadoWorkspace.H10[10];
acadoWorkspace.H[140] = acadoWorkspace.H10[11];
acadoWorkspace.H[161] = acadoWorkspace.H10[12];
acadoWorkspace.H[162] = acadoWorkspace.H10[13];
acadoWorkspace.H[163] = acadoWorkspace.H10[14];
acadoWorkspace.H[184] = acadoWorkspace.H10[15];
acadoWorkspace.H[185] = acadoWorkspace.H10[16];
acadoWorkspace.H[186] = acadoWorkspace.H10[17];
acadoWorkspace.H[207] = acadoWorkspace.H10[18];
acadoWorkspace.H[208] = acadoWorkspace.H10[19];
acadoWorkspace.H[209] = acadoWorkspace.H10[20];
acadoWorkspace.H[230] = acadoWorkspace.H10[21];
acadoWorkspace.H[231] = acadoWorkspace.H10[22];
acadoWorkspace.H[232] = acadoWorkspace.H10[23];
acadoWorkspace.H[253] = acadoWorkspace.H10[24];
acadoWorkspace.H[254] = acadoWorkspace.H10[25];
acadoWorkspace.H[255] = acadoWorkspace.H10[26];
acadoWorkspace.H[276] = acadoWorkspace.H10[27];
acadoWorkspace.H[277] = acadoWorkspace.H10[28];
acadoWorkspace.H[278] = acadoWorkspace.H10[29];
acadoWorkspace.H[299] = acadoWorkspace.H10[30];
acadoWorkspace.H[300] = acadoWorkspace.H10[31];
acadoWorkspace.H[301] = acadoWorkspace.H10[32];
acadoWorkspace.H[322] = acadoWorkspace.H10[33];
acadoWorkspace.H[323] = acadoWorkspace.H10[34];
acadoWorkspace.H[324] = acadoWorkspace.H10[35];
acadoWorkspace.H[345] = acadoWorkspace.H10[36];
acadoWorkspace.H[346] = acadoWorkspace.H10[37];
acadoWorkspace.H[347] = acadoWorkspace.H10[38];
acadoWorkspace.H[368] = acadoWorkspace.H10[39];
acadoWorkspace.H[369] = acadoWorkspace.H10[40];
acadoWorkspace.H[370] = acadoWorkspace.H10[41];
acadoWorkspace.H[391] = acadoWorkspace.H10[42];
acadoWorkspace.H[392] = acadoWorkspace.H10[43];
acadoWorkspace.H[393] = acadoWorkspace.H10[44];
acadoWorkspace.H[414] = acadoWorkspace.H10[45];
acadoWorkspace.H[415] = acadoWorkspace.H10[46];
acadoWorkspace.H[416] = acadoWorkspace.H10[47];
acadoWorkspace.H[437] = acadoWorkspace.H10[48];
acadoWorkspace.H[438] = acadoWorkspace.H10[49];
acadoWorkspace.H[439] = acadoWorkspace.H10[50];
acadoWorkspace.H[460] = acadoWorkspace.H10[51];
acadoWorkspace.H[461] = acadoWorkspace.H10[52];
acadoWorkspace.H[462] = acadoWorkspace.H10[53];
acadoWorkspace.H[483] = acadoWorkspace.H10[54];
acadoWorkspace.H[484] = acadoWorkspace.H10[55];
acadoWorkspace.H[485] = acadoWorkspace.H10[56];
acadoWorkspace.H[506] = acadoWorkspace.H10[57];
acadoWorkspace.H[507] = acadoWorkspace.H10[58];
acadoWorkspace.H[508] = acadoWorkspace.H10[59];

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
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 57 ]), &(acadoWorkspace.Qd[ 57 ]) );

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
acado_macETSlu( acadoWorkspace.QE, &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 3 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 9 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 18 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 45 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 63 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 135 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 165 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 198 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 234 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 273 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 315 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 459 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 513 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.g[ 3 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 6 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 21 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 33 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 66 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 87 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 111 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 138 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 201 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 237 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 318 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 363 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 411 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 462 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 573 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 15 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 51 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 69 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 114 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 141 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 171 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 279 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 321 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 366 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 414 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 465 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 519 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.g[ 5 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 27 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 39 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 54 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 93 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 117 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 174 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 207 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 243 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 282 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 369 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 417 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 522 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 579 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 42 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 57 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 75 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 147 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 177 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 246 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 285 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 327 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 471 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 525 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 582 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 78 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 99 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 123 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 213 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 249 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 375 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 423 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 474 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 585 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 81 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 102 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 126 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 153 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 183 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 291 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 333 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 378 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 426 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 477 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 531 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 588 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 105 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 129 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 186 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 219 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 255 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 294 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 381 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 429 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 534 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 591 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 159 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 189 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 222 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 258 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 297 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 339 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 483 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 537 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 594 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 162 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 225 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 261 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 342 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 387 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 435 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 486 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 597 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 195 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 303 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 345 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 438 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 489 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 543 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 231 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 267 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 306 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 393 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 441 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 546 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 603 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 309 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 351 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 495 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 549 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 606 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 354 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 399 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 447 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 498 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 609 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 357 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 402 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 501 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 555 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 612 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 405 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 453 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 558 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 615 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 507 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 561 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 618 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 564 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 621 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 567 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 627 ]), &(acadoWorkspace.g[ 22 ]) );
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

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 3;
lRun4 = ((lRun3) / (3)) + (1);
acadoWorkspace.A[lRun1 * 23] = acadoWorkspace.evGx[lRun3 * 3];
acadoWorkspace.A[lRun1 * 23 + 1] = acadoWorkspace.evGx[lRun3 * 3 + 1];
acadoWorkspace.A[lRun1 * 23 + 2] = acadoWorkspace.evGx[lRun3 * 3 + 2];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (3)) + ((lRun3) % (3));
acadoWorkspace.A[(lRun1 * 23) + (lRun2 + 3)] = acadoWorkspace.E[lRun5];
}
}

}

void acado_condenseFdb(  )
{
real_t tmp;

acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];

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

acadoWorkspace.QDy[60] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[61] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[62] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];

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

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[62];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[62];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[3] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[62];


acado_multEQDy( acadoWorkspace.E, &(acadoWorkspace.QDy[ 3 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.QDy[ 6 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.QDy[ 9 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 3 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QDy[ 6 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QDy[ 9 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.QDy[ 9 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 27 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QDy[ 15 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 81 ]), &(acadoWorkspace.QDy[ 21 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QDy[ 27 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.QDy[ 33 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 231 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QDy[ 39 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 357 ]), &(acadoWorkspace.QDy[ 45 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QDy[ 51 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 567 ]), &(acadoWorkspace.QDy[ 57 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 627 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 22 ]) );

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
tmp = acadoVariables.x[4] + acadoWorkspace.d[1];
acadoWorkspace.lbA[0] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[7] + acadoWorkspace.d[4];
acadoWorkspace.lbA[1] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[10] + acadoWorkspace.d[7];
acadoWorkspace.lbA[2] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[13] + acadoWorkspace.d[10];
acadoWorkspace.lbA[3] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[16] + acadoWorkspace.d[13];
acadoWorkspace.lbA[4] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[19] + acadoWorkspace.d[16];
acadoWorkspace.lbA[5] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[22] + acadoWorkspace.d[19];
acadoWorkspace.lbA[6] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[25] + acadoWorkspace.d[22];
acadoWorkspace.lbA[7] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[28] + acadoWorkspace.d[25];
acadoWorkspace.lbA[8] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[8] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[28];
acadoWorkspace.lbA[9] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[34] + acadoWorkspace.d[31];
acadoWorkspace.lbA[10] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[37] + acadoWorkspace.d[34];
acadoWorkspace.lbA[11] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[40] + acadoWorkspace.d[37];
acadoWorkspace.lbA[12] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[43] + acadoWorkspace.d[40];
acadoWorkspace.lbA[13] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[46] + acadoWorkspace.d[43];
acadoWorkspace.lbA[14] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[14] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[49] + acadoWorkspace.d[46];
acadoWorkspace.lbA[15] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[52] + acadoWorkspace.d[49];
acadoWorkspace.lbA[16] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[55] + acadoWorkspace.d[52];
acadoWorkspace.lbA[17] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[17] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[58] + acadoWorkspace.d[55];
acadoWorkspace.lbA[18] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[61] + acadoWorkspace.d[58];
acadoWorkspace.lbA[19] = (real_t)-1.0000000000000001e-01 - tmp;
acadoWorkspace.ubA[19] = (real_t)1.0000000000000000e+12 - tmp;

}

void acado_expand(  )
{
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

acado_multEDu( acadoWorkspace.E, &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 3 ]) );
acado_multEDu( &(acadoWorkspace.E[ 3 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 6 ]) );
acado_multEDu( &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 6 ]) );
acado_multEDu( &(acadoWorkspace.E[ 9 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 9 ]) );
acado_multEDu( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 9 ]) );
acado_multEDu( &(acadoWorkspace.E[ 15 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 9 ]) );
acado_multEDu( &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 21 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 27 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 33 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 39 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 15 ]) );
acado_multEDu( &(acadoWorkspace.E[ 45 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 51 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 57 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 63 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 69 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 75 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 81 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 21 ]) );
acado_multEDu( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 87 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 93 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 99 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 105 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 111 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 117 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 123 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 129 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 27 ]) );
acado_multEDu( &(acadoWorkspace.E[ 135 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 141 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 147 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 153 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 159 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 165 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 171 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 177 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 183 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 189 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 195 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 33 ]) );
acado_multEDu( &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 201 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 207 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 213 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 219 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 225 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 231 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 237 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 243 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 249 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 255 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 261 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 267 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 39 ]) );
acado_multEDu( &(acadoWorkspace.E[ 273 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 279 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 285 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 291 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 297 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 303 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 309 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 315 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 321 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 327 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 333 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 339 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 345 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 351 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 357 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 45 ]) );
acado_multEDu( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 363 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 369 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 375 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 381 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 387 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 393 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 399 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 405 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 411 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 417 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 423 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 429 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 435 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 441 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 447 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 453 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 51 ]) );
acado_multEDu( &(acadoWorkspace.E[ 459 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 465 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 471 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 477 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 483 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 489 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 495 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 501 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 507 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 513 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 519 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 525 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 531 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 537 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 543 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 549 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 555 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 561 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 567 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 57 ]) );
acado_multEDu( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.x[ 3 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 573 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.x[ 5 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 579 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 585 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 591 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 597 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 603 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 609 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 615 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 621 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 627 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 60 ]) );
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
for (index = 0; index < 20; ++index)
{
acadoVariables.x[index * 3] = acadoVariables.x[index * 3 + 3];
acadoVariables.x[index * 3 + 1] = acadoVariables.x[index * 3 + 4];
acadoVariables.x[index * 3 + 2] = acadoVariables.x[index * 3 + 5];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[60] = xEnd[0];
acadoVariables.x[61] = xEnd[1];
acadoVariables.x[62] = xEnd[2];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[60];
acadoWorkspace.state[1] = acadoVariables.x[61];
acadoWorkspace.state[2] = acadoVariables.x[62];
if (uEnd != 0)
{
acadoWorkspace.state[15] = uEnd[0];
}
else
{
acadoWorkspace.state[15] = acadoVariables.u[19];
}
acadoWorkspace.state[16] = acadoVariables.od[40];
acadoWorkspace.state[17] = acadoVariables.od[41];

acado_integrate(acadoWorkspace.state, 1, 19);

acadoVariables.x[60] = acadoWorkspace.state[0];
acadoVariables.x[61] = acadoWorkspace.state[1];
acadoVariables.x[62] = acadoWorkspace.state[2];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 19; ++index)
{
acadoVariables.u[index] = acadoVariables.u[index + 1];
}

if (uEnd != 0)
{
acadoVariables.u[19] = uEnd[0];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22];
kkt = fabs( kkt );
for (index = 0; index < 23; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 20; ++index)
{
prd = acadoWorkspace.y[index + 23];
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

for (lRun1 = 0; lRun1 < 20; ++lRun1)
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
acadoWorkspace.objValueIn[0] = acadoVariables.x[60];
acadoWorkspace.objValueIn[1] = acadoVariables.x[61];
acadoWorkspace.objValueIn[2] = acadoVariables.x[62];
acadoWorkspace.objValueIn[3] = acadoVariables.od[40];
acadoWorkspace.objValueIn[4] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 20; ++lRun1)
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

