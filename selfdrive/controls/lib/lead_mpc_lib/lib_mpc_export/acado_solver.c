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
const real_t* od = in + 6;
/* Vector of auxiliary variables; number of elements: 10. */
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

/* Compute outputs: */
out[0] = (a[1]-(real_t)(1.0000000000000000e+00));
out[1] = ((real_t)(0.0000000000000000e+00)/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = (u[0]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[4] = u[1];
out[5] = a[4];
out[6] = a[9];
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = (real_t)(0.0000000000000000e+00);
out[14] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[15] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = (real_t)(0.0000000000000000e+00);
out[18] = (u[0]*(real_t)(1.0000000000000001e-01));
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (real_t)(0.0000000000000000e+00);
out[21] = (real_t)(0.0000000000000000e+00);
out[22] = (real_t)(0.0000000000000000e+00);
out[23] = (real_t)(0.0000000000000000e+00);
out[24] = (real_t)(0.0000000000000000e+00);
out[25] = (real_t)(0.0000000000000000e+00);
out[26] = (real_t)(0.0000000000000000e+00);
out[27] = (real_t)(0.0000000000000000e+00);
out[28] = (real_t)(0.0000000000000000e+00);
out[29] = (real_t)(0.0000000000000000e+00);
out[30] = (real_t)(0.0000000000000000e+00);
out[31] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[32] = (real_t)(0.0000000000000000e+00);
out[33] = (real_t)(0.0000000000000000e+00);
out[34] = (real_t)(1.0000000000000000e+00);
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
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
out[1] = (((real_t)(2.0000000000000000e+00)*((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01))))))/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = a[4];
out[4] = a[9];
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (((real_t)(-2.0000000000000000e+00))*a[10]);
out[8] = ((((real_t)(2.0000000000000000e+00)*((real_t)(0.0000000000000000e+00)-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[11]))))*a[10])-((((real_t)(2.0000000000000000e+00)*((od[0]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((od[1]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((od[1]*od[1])/(real_t)(1.9620000000000001e+01))))))*(real_t)(5.0000000000000003e-02))*a[12]));
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[13] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[14] = (real_t)(0.0000000000000000e+00);
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0]*tmpObjS[0] + tmpFx[4]*tmpObjS[5] + tmpFx[8]*tmpObjS[10] + tmpFx[12]*tmpObjS[15] + tmpFx[16]*tmpObjS[20];
tmpQ2[1] = + tmpFx[0]*tmpObjS[1] + tmpFx[4]*tmpObjS[6] + tmpFx[8]*tmpObjS[11] + tmpFx[12]*tmpObjS[16] + tmpFx[16]*tmpObjS[21];
tmpQ2[2] = + tmpFx[0]*tmpObjS[2] + tmpFx[4]*tmpObjS[7] + tmpFx[8]*tmpObjS[12] + tmpFx[12]*tmpObjS[17] + tmpFx[16]*tmpObjS[22];
tmpQ2[3] = + tmpFx[0]*tmpObjS[3] + tmpFx[4]*tmpObjS[8] + tmpFx[8]*tmpObjS[13] + tmpFx[12]*tmpObjS[18] + tmpFx[16]*tmpObjS[23];
tmpQ2[4] = + tmpFx[0]*tmpObjS[4] + tmpFx[4]*tmpObjS[9] + tmpFx[8]*tmpObjS[14] + tmpFx[12]*tmpObjS[19] + tmpFx[16]*tmpObjS[24];
tmpQ2[5] = + tmpFx[1]*tmpObjS[0] + tmpFx[5]*tmpObjS[5] + tmpFx[9]*tmpObjS[10] + tmpFx[13]*tmpObjS[15] + tmpFx[17]*tmpObjS[20];
tmpQ2[6] = + tmpFx[1]*tmpObjS[1] + tmpFx[5]*tmpObjS[6] + tmpFx[9]*tmpObjS[11] + tmpFx[13]*tmpObjS[16] + tmpFx[17]*tmpObjS[21];
tmpQ2[7] = + tmpFx[1]*tmpObjS[2] + tmpFx[5]*tmpObjS[7] + tmpFx[9]*tmpObjS[12] + tmpFx[13]*tmpObjS[17] + tmpFx[17]*tmpObjS[22];
tmpQ2[8] = + tmpFx[1]*tmpObjS[3] + tmpFx[5]*tmpObjS[8] + tmpFx[9]*tmpObjS[13] + tmpFx[13]*tmpObjS[18] + tmpFx[17]*tmpObjS[23];
tmpQ2[9] = + tmpFx[1]*tmpObjS[4] + tmpFx[5]*tmpObjS[9] + tmpFx[9]*tmpObjS[14] + tmpFx[13]*tmpObjS[19] + tmpFx[17]*tmpObjS[24];
tmpQ2[10] = + tmpFx[2]*tmpObjS[0] + tmpFx[6]*tmpObjS[5] + tmpFx[10]*tmpObjS[10] + tmpFx[14]*tmpObjS[15] + tmpFx[18]*tmpObjS[20];
tmpQ2[11] = + tmpFx[2]*tmpObjS[1] + tmpFx[6]*tmpObjS[6] + tmpFx[10]*tmpObjS[11] + tmpFx[14]*tmpObjS[16] + tmpFx[18]*tmpObjS[21];
tmpQ2[12] = + tmpFx[2]*tmpObjS[2] + tmpFx[6]*tmpObjS[7] + tmpFx[10]*tmpObjS[12] + tmpFx[14]*tmpObjS[17] + tmpFx[18]*tmpObjS[22];
tmpQ2[13] = + tmpFx[2]*tmpObjS[3] + tmpFx[6]*tmpObjS[8] + tmpFx[10]*tmpObjS[13] + tmpFx[14]*tmpObjS[18] + tmpFx[18]*tmpObjS[23];
tmpQ2[14] = + tmpFx[2]*tmpObjS[4] + tmpFx[6]*tmpObjS[9] + tmpFx[10]*tmpObjS[14] + tmpFx[14]*tmpObjS[19] + tmpFx[18]*tmpObjS[24];
tmpQ2[15] = + tmpFx[3]*tmpObjS[0] + tmpFx[7]*tmpObjS[5] + tmpFx[11]*tmpObjS[10] + tmpFx[15]*tmpObjS[15] + tmpFx[19]*tmpObjS[20];
tmpQ2[16] = + tmpFx[3]*tmpObjS[1] + tmpFx[7]*tmpObjS[6] + tmpFx[11]*tmpObjS[11] + tmpFx[15]*tmpObjS[16] + tmpFx[19]*tmpObjS[21];
tmpQ2[17] = + tmpFx[3]*tmpObjS[2] + tmpFx[7]*tmpObjS[7] + tmpFx[11]*tmpObjS[12] + tmpFx[15]*tmpObjS[17] + tmpFx[19]*tmpObjS[22];
tmpQ2[18] = + tmpFx[3]*tmpObjS[3] + tmpFx[7]*tmpObjS[8] + tmpFx[11]*tmpObjS[13] + tmpFx[15]*tmpObjS[18] + tmpFx[19]*tmpObjS[23];
tmpQ2[19] = + tmpFx[3]*tmpObjS[4] + tmpFx[7]*tmpObjS[9] + tmpFx[11]*tmpObjS[14] + tmpFx[15]*tmpObjS[19] + tmpFx[19]*tmpObjS[24];
tmpQ1[0] = + tmpQ2[0]*tmpFx[0] + tmpQ2[1]*tmpFx[4] + tmpQ2[2]*tmpFx[8] + tmpQ2[3]*tmpFx[12] + tmpQ2[4]*tmpFx[16];
tmpQ1[1] = + tmpQ2[0]*tmpFx[1] + tmpQ2[1]*tmpFx[5] + tmpQ2[2]*tmpFx[9] + tmpQ2[3]*tmpFx[13] + tmpQ2[4]*tmpFx[17];
tmpQ1[2] = + tmpQ2[0]*tmpFx[2] + tmpQ2[1]*tmpFx[6] + tmpQ2[2]*tmpFx[10] + tmpQ2[3]*tmpFx[14] + tmpQ2[4]*tmpFx[18];
tmpQ1[3] = + tmpQ2[0]*tmpFx[3] + tmpQ2[1]*tmpFx[7] + tmpQ2[2]*tmpFx[11] + tmpQ2[3]*tmpFx[15] + tmpQ2[4]*tmpFx[19];
tmpQ1[4] = + tmpQ2[5]*tmpFx[0] + tmpQ2[6]*tmpFx[4] + tmpQ2[7]*tmpFx[8] + tmpQ2[8]*tmpFx[12] + tmpQ2[9]*tmpFx[16];
tmpQ1[5] = + tmpQ2[5]*tmpFx[1] + tmpQ2[6]*tmpFx[5] + tmpQ2[7]*tmpFx[9] + tmpQ2[8]*tmpFx[13] + tmpQ2[9]*tmpFx[17];
tmpQ1[6] = + tmpQ2[5]*tmpFx[2] + tmpQ2[6]*tmpFx[6] + tmpQ2[7]*tmpFx[10] + tmpQ2[8]*tmpFx[14] + tmpQ2[9]*tmpFx[18];
tmpQ1[7] = + tmpQ2[5]*tmpFx[3] + tmpQ2[6]*tmpFx[7] + tmpQ2[7]*tmpFx[11] + tmpQ2[8]*tmpFx[15] + tmpQ2[9]*tmpFx[19];
tmpQ1[8] = + tmpQ2[10]*tmpFx[0] + tmpQ2[11]*tmpFx[4] + tmpQ2[12]*tmpFx[8] + tmpQ2[13]*tmpFx[12] + tmpQ2[14]*tmpFx[16];
tmpQ1[9] = + tmpQ2[10]*tmpFx[1] + tmpQ2[11]*tmpFx[5] + tmpQ2[12]*tmpFx[9] + tmpQ2[13]*tmpFx[13] + tmpQ2[14]*tmpFx[17];
tmpQ1[10] = + tmpQ2[10]*tmpFx[2] + tmpQ2[11]*tmpFx[6] + tmpQ2[12]*tmpFx[10] + tmpQ2[13]*tmpFx[14] + tmpQ2[14]*tmpFx[18];
tmpQ1[11] = + tmpQ2[10]*tmpFx[3] + tmpQ2[11]*tmpFx[7] + tmpQ2[12]*tmpFx[11] + tmpQ2[13]*tmpFx[15] + tmpQ2[14]*tmpFx[19];
tmpQ1[12] = + tmpQ2[15]*tmpFx[0] + tmpQ2[16]*tmpFx[4] + tmpQ2[17]*tmpFx[8] + tmpQ2[18]*tmpFx[12] + tmpQ2[19]*tmpFx[16];
tmpQ1[13] = + tmpQ2[15]*tmpFx[1] + tmpQ2[16]*tmpFx[5] + tmpQ2[17]*tmpFx[9] + tmpQ2[18]*tmpFx[13] + tmpQ2[19]*tmpFx[17];
tmpQ1[14] = + tmpQ2[15]*tmpFx[2] + tmpQ2[16]*tmpFx[6] + tmpQ2[17]*tmpFx[10] + tmpQ2[18]*tmpFx[14] + tmpQ2[19]*tmpFx[18];
tmpQ1[15] = + tmpQ2[15]*tmpFx[3] + tmpQ2[16]*tmpFx[7] + tmpQ2[17]*tmpFx[11] + tmpQ2[18]*tmpFx[15] + tmpQ2[19]*tmpFx[19];
}

void acado_setObjR1R2( real_t* const tmpFu, real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = + tmpFu[0]*tmpObjS[0] + tmpFu[2]*tmpObjS[5] + tmpFu[4]*tmpObjS[10] + tmpFu[6]*tmpObjS[15] + tmpFu[8]*tmpObjS[20];
tmpR2[1] = + tmpFu[0]*tmpObjS[1] + tmpFu[2]*tmpObjS[6] + tmpFu[4]*tmpObjS[11] + tmpFu[6]*tmpObjS[16] + tmpFu[8]*tmpObjS[21];
tmpR2[2] = + tmpFu[0]*tmpObjS[2] + tmpFu[2]*tmpObjS[7] + tmpFu[4]*tmpObjS[12] + tmpFu[6]*tmpObjS[17] + tmpFu[8]*tmpObjS[22];
tmpR2[3] = + tmpFu[0]*tmpObjS[3] + tmpFu[2]*tmpObjS[8] + tmpFu[4]*tmpObjS[13] + tmpFu[6]*tmpObjS[18] + tmpFu[8]*tmpObjS[23];
tmpR2[4] = + tmpFu[0]*tmpObjS[4] + tmpFu[2]*tmpObjS[9] + tmpFu[4]*tmpObjS[14] + tmpFu[6]*tmpObjS[19] + tmpFu[8]*tmpObjS[24];
tmpR2[5] = + tmpFu[1]*tmpObjS[0] + tmpFu[3]*tmpObjS[5] + tmpFu[5]*tmpObjS[10] + tmpFu[7]*tmpObjS[15] + tmpFu[9]*tmpObjS[20];
tmpR2[6] = + tmpFu[1]*tmpObjS[1] + tmpFu[3]*tmpObjS[6] + tmpFu[5]*tmpObjS[11] + tmpFu[7]*tmpObjS[16] + tmpFu[9]*tmpObjS[21];
tmpR2[7] = + tmpFu[1]*tmpObjS[2] + tmpFu[3]*tmpObjS[7] + tmpFu[5]*tmpObjS[12] + tmpFu[7]*tmpObjS[17] + tmpFu[9]*tmpObjS[22];
tmpR2[8] = + tmpFu[1]*tmpObjS[3] + tmpFu[3]*tmpObjS[8] + tmpFu[5]*tmpObjS[13] + tmpFu[7]*tmpObjS[18] + tmpFu[9]*tmpObjS[23];
tmpR2[9] = + tmpFu[1]*tmpObjS[4] + tmpFu[3]*tmpObjS[9] + tmpFu[5]*tmpObjS[14] + tmpFu[7]*tmpObjS[19] + tmpFu[9]*tmpObjS[24];
tmpR1[0] = + tmpR2[0]*tmpFu[0] + tmpR2[1]*tmpFu[2] + tmpR2[2]*tmpFu[4] + tmpR2[3]*tmpFu[6] + tmpR2[4]*tmpFu[8];
tmpR1[1] = + tmpR2[0]*tmpFu[1] + tmpR2[1]*tmpFu[3] + tmpR2[2]*tmpFu[5] + tmpR2[3]*tmpFu[7] + tmpR2[4]*tmpFu[9];
tmpR1[2] = + tmpR2[5]*tmpFu[0] + tmpR2[6]*tmpFu[2] + tmpR2[7]*tmpFu[4] + tmpR2[8]*tmpFu[6] + tmpR2[9]*tmpFu[8];
tmpR1[3] = + tmpR2[5]*tmpFu[1] + tmpR2[6]*tmpFu[3] + tmpR2[7]*tmpFu[5] + tmpR2[8]*tmpFu[7] + tmpR2[9]*tmpFu[9];
}

void acado_setObjQN1QN2( real_t* const tmpFx, real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = + tmpFx[0]*tmpObjSEndTerm[0] + tmpFx[4]*tmpObjSEndTerm[3] + tmpFx[8]*tmpObjSEndTerm[6];
tmpQN2[1] = + tmpFx[0]*tmpObjSEndTerm[1] + tmpFx[4]*tmpObjSEndTerm[4] + tmpFx[8]*tmpObjSEndTerm[7];
tmpQN2[2] = + tmpFx[0]*tmpObjSEndTerm[2] + tmpFx[4]*tmpObjSEndTerm[5] + tmpFx[8]*tmpObjSEndTerm[8];
tmpQN2[3] = + tmpFx[1]*tmpObjSEndTerm[0] + tmpFx[5]*tmpObjSEndTerm[3] + tmpFx[9]*tmpObjSEndTerm[6];
tmpQN2[4] = + tmpFx[1]*tmpObjSEndTerm[1] + tmpFx[5]*tmpObjSEndTerm[4] + tmpFx[9]*tmpObjSEndTerm[7];
tmpQN2[5] = + tmpFx[1]*tmpObjSEndTerm[2] + tmpFx[5]*tmpObjSEndTerm[5] + tmpFx[9]*tmpObjSEndTerm[8];
tmpQN2[6] = + tmpFx[2]*tmpObjSEndTerm[0] + tmpFx[6]*tmpObjSEndTerm[3] + tmpFx[10]*tmpObjSEndTerm[6];
tmpQN2[7] = + tmpFx[2]*tmpObjSEndTerm[1] + tmpFx[6]*tmpObjSEndTerm[4] + tmpFx[10]*tmpObjSEndTerm[7];
tmpQN2[8] = + tmpFx[2]*tmpObjSEndTerm[2] + tmpFx[6]*tmpObjSEndTerm[5] + tmpFx[10]*tmpObjSEndTerm[8];
tmpQN2[9] = + tmpFx[3]*tmpObjSEndTerm[0] + tmpFx[7]*tmpObjSEndTerm[3] + tmpFx[11]*tmpObjSEndTerm[6];
tmpQN2[10] = + tmpFx[3]*tmpObjSEndTerm[1] + tmpFx[7]*tmpObjSEndTerm[4] + tmpFx[11]*tmpObjSEndTerm[7];
tmpQN2[11] = + tmpFx[3]*tmpObjSEndTerm[2] + tmpFx[7]*tmpObjSEndTerm[5] + tmpFx[11]*tmpObjSEndTerm[8];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0] + tmpQN2[1]*tmpFx[4] + tmpQN2[2]*tmpFx[8];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1] + tmpQN2[1]*tmpFx[5] + tmpQN2[2]*tmpFx[9];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2] + tmpQN2[1]*tmpFx[6] + tmpQN2[2]*tmpFx[10];
tmpQN1[3] = + tmpQN2[0]*tmpFx[3] + tmpQN2[1]*tmpFx[7] + tmpQN2[2]*tmpFx[11];
tmpQN1[4] = + tmpQN2[3]*tmpFx[0] + tmpQN2[4]*tmpFx[4] + tmpQN2[5]*tmpFx[8];
tmpQN1[5] = + tmpQN2[3]*tmpFx[1] + tmpQN2[4]*tmpFx[5] + tmpQN2[5]*tmpFx[9];
tmpQN1[6] = + tmpQN2[3]*tmpFx[2] + tmpQN2[4]*tmpFx[6] + tmpQN2[5]*tmpFx[10];
tmpQN1[7] = + tmpQN2[3]*tmpFx[3] + tmpQN2[4]*tmpFx[7] + tmpQN2[5]*tmpFx[11];
tmpQN1[8] = + tmpQN2[6]*tmpFx[0] + tmpQN2[7]*tmpFx[4] + tmpQN2[8]*tmpFx[8];
tmpQN1[9] = + tmpQN2[6]*tmpFx[1] + tmpQN2[7]*tmpFx[5] + tmpQN2[8]*tmpFx[9];
tmpQN1[10] = + tmpQN2[6]*tmpFx[2] + tmpQN2[7]*tmpFx[6] + tmpQN2[8]*tmpFx[10];
tmpQN1[11] = + tmpQN2[6]*tmpFx[3] + tmpQN2[7]*tmpFx[7] + tmpQN2[8]*tmpFx[11];
tmpQN1[12] = + tmpQN2[9]*tmpFx[0] + tmpQN2[10]*tmpFx[4] + tmpQN2[11]*tmpFx[8];
tmpQN1[13] = + tmpQN2[9]*tmpFx[1] + tmpQN2[10]*tmpFx[5] + tmpQN2[11]*tmpFx[9];
tmpQN1[14] = + tmpQN2[9]*tmpFx[2] + tmpQN2[10]*tmpFx[6] + tmpQN2[11]*tmpFx[10];
tmpQN1[15] = + tmpQN2[9]*tmpFx[3] + tmpQN2[10]*tmpFx[7] + tmpQN2[11]*tmpFx[11];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 20; ++runObj)
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

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 5 ]), &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 20 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 25 ]), &(acadoVariables.W[ runObj * 25 ]), &(acadoWorkspace.R1[ runObj * 4 ]), &(acadoWorkspace.R2[ runObj * 10 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[80];
acadoWorkspace.objValueIn[1] = acadoVariables.x[81];
acadoWorkspace.objValueIn[2] = acadoVariables.x[82];
acadoWorkspace.objValueIn[3] = acadoVariables.x[83];
acadoWorkspace.objValueIn[4] = acadoVariables.od[40];
acadoWorkspace.objValueIn[5] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 3 ]), acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

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
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 4)] += + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6];
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 5)] += + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 4)] += + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 5)] += + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 4)] = R11[0];
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 5)] = R11[1];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 4)] = R11[2];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 5)] = R11[3];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 4)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 5)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 4)] = 0.0000000000000000e+00;
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 5)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 4)] = acadoWorkspace.H[(iCol * 88 + 176) + (iRow * 2 + 4)];
acadoWorkspace.H[(iRow * 88 + 176) + (iCol * 2 + 5)] = acadoWorkspace.H[(iCol * 88 + 220) + (iRow * 2 + 4)];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 4)] = acadoWorkspace.H[(iCol * 88 + 176) + (iRow * 2 + 5)];
acadoWorkspace.H[(iRow * 88 + 220) + (iCol * 2 + 5)] = acadoWorkspace.H[(iCol * 88 + 220) + (iRow * 2 + 5)];
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
acadoWorkspace.H[44] = 0.0000000000000000e+00;
acadoWorkspace.H[45] = 0.0000000000000000e+00;
acadoWorkspace.H[46] = 0.0000000000000000e+00;
acadoWorkspace.H[47] = 0.0000000000000000e+00;
acadoWorkspace.H[88] = 0.0000000000000000e+00;
acadoWorkspace.H[89] = 0.0000000000000000e+00;
acadoWorkspace.H[90] = 0.0000000000000000e+00;
acadoWorkspace.H[91] = 0.0000000000000000e+00;
acadoWorkspace.H[132] = 0.0000000000000000e+00;
acadoWorkspace.H[133] = 0.0000000000000000e+00;
acadoWorkspace.H[134] = 0.0000000000000000e+00;
acadoWorkspace.H[135] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[4]*Gx2[4] + Gx1[8]*Gx2[8] + Gx1[12]*Gx2[12];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[4]*Gx2[5] + Gx1[8]*Gx2[9] + Gx1[12]*Gx2[13];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[4]*Gx2[6] + Gx1[8]*Gx2[10] + Gx1[12]*Gx2[14];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[4]*Gx2[7] + Gx1[8]*Gx2[11] + Gx1[12]*Gx2[15];
acadoWorkspace.H[44] += + Gx1[1]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[9]*Gx2[8] + Gx1[13]*Gx2[12];
acadoWorkspace.H[45] += + Gx1[1]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[9]*Gx2[9] + Gx1[13]*Gx2[13];
acadoWorkspace.H[46] += + Gx1[1]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[9]*Gx2[10] + Gx1[13]*Gx2[14];
acadoWorkspace.H[47] += + Gx1[1]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[9]*Gx2[11] + Gx1[13]*Gx2[15];
acadoWorkspace.H[88] += + Gx1[2]*Gx2[0] + Gx1[6]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[14]*Gx2[12];
acadoWorkspace.H[89] += + Gx1[2]*Gx2[1] + Gx1[6]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[14]*Gx2[13];
acadoWorkspace.H[90] += + Gx1[2]*Gx2[2] + Gx1[6]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[14]*Gx2[14];
acadoWorkspace.H[91] += + Gx1[2]*Gx2[3] + Gx1[6]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[14]*Gx2[15];
acadoWorkspace.H[132] += + Gx1[3]*Gx2[0] + Gx1[7]*Gx2[4] + Gx1[11]*Gx2[8] + Gx1[15]*Gx2[12];
acadoWorkspace.H[133] += + Gx1[3]*Gx2[1] + Gx1[7]*Gx2[5] + Gx1[11]*Gx2[9] + Gx1[15]*Gx2[13];
acadoWorkspace.H[134] += + Gx1[3]*Gx2[2] + Gx1[7]*Gx2[6] + Gx1[11]*Gx2[10] + Gx1[15]*Gx2[14];
acadoWorkspace.H[135] += + Gx1[3]*Gx2[3] + Gx1[7]*Gx2[7] + Gx1[11]*Gx2[11] + Gx1[15]*Gx2[15];
}

void acado_multHxC( real_t* const Hx, real_t* const Gx, real_t* const A01 )
{
A01[0] = + Hx[0]*Gx[0] + Hx[1]*Gx[4] + Hx[2]*Gx[8] + Hx[3]*Gx[12];
A01[1] = + Hx[0]*Gx[1] + Hx[1]*Gx[5] + Hx[2]*Gx[9] + Hx[3]*Gx[13];
A01[2] = + Hx[0]*Gx[2] + Hx[1]*Gx[6] + Hx[2]*Gx[10] + Hx[3]*Gx[14];
A01[3] = + Hx[0]*Gx[3] + Hx[1]*Gx[7] + Hx[2]*Gx[11] + Hx[3]*Gx[15];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 44 + 880) + (col * 2 + 4)] = + Hx[0]*E[0] + Hx[1]*E[2] + Hx[2]*E[4] + Hx[3]*E[6];
acadoWorkspace.A[(row * 44 + 880) + (col * 2 + 5)] = + Hx[0]*E[1] + Hx[1]*E[3] + Hx[2]*E[5] + Hx[3]*E[7];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2] + Hx[3]*tmpd[3];
lbA[0] -= acadoWorkspace.evHxd[0];
ubA[0] -= acadoWorkspace.evHxd[0];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 4;
/* Vector of auxiliary variables; number of elements: 6. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (real_t)(0.0000000000000000e+00);
a[1] = (real_t)(0.0000000000000000e+00);
a[2] = (real_t)(1.0000000000000000e+00);
a[3] = (real_t)(0.0000000000000000e+00);
a[4] = (real_t)(0.0000000000000000e+00);
a[5] = (real_t)(1.0000000000000000e+00);

/* Compute outputs: */
out[0] = (xd[2]+u[1]);
out[1] = a[0];
out[2] = a[1];
out[3] = a[2];
out[4] = a[3];
out[5] = a[4];
out[6] = a[5];
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
/** Row vector of size: 20 */
static const int xBoundIndices[ 20 ] = 
{ 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_moveGxT( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.T );
acado_multGxd( acadoWorkspace.d, &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.d[ 4 ]) );
acado_multGxGx( acadoWorkspace.T, acadoWorkspace.evGx, &(acadoWorkspace.evGx[ 16 ]) );

acado_multGxGu( acadoWorkspace.T, acadoWorkspace.E, &(acadoWorkspace.E[ 8 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 8 ]), &(acadoWorkspace.E[ 16 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 4 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.d[ 8 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.evGx[ 32 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.E[ 24 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.E[ 32 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 16 ]), &(acadoWorkspace.E[ 40 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 8 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.d[ 12 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.evGx[ 48 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.E[ 48 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.E[ 56 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.E[ 64 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.E[ 72 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.d[ 16 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.evGx[ 64 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.E[ 80 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.E[ 88 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.E[ 96 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.E[ 104 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 32 ]), &(acadoWorkspace.E[ 112 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 16 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.d[ 20 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.evGx[ 80 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.E[ 120 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.E[ 128 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.E[ 136 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.E[ 144 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.E[ 152 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 40 ]), &(acadoWorkspace.E[ 160 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.d[ 24 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.evGx[ 96 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.E[ 168 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.E[ 176 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.E[ 184 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.E[ 192 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.E[ 200 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.E[ 208 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 48 ]), &(acadoWorkspace.E[ 216 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.d[ 28 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.evGx[ 112 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.E[ 224 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.E[ 232 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.E[ 240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.E[ 248 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.E[ 256 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.E[ 264 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.E[ 272 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 56 ]), &(acadoWorkspace.E[ 280 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 28 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.d[ 32 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.evGx[ 128 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.E[ 288 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.E[ 296 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.E[ 304 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.E[ 312 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.E[ 320 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.E[ 328 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.E[ 336 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.E[ 344 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 64 ]), &(acadoWorkspace.E[ 352 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 32 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.d[ 36 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.evGx[ 144 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.E[ 360 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.E[ 368 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.E[ 376 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.E[ 384 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.E[ 392 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.E[ 400 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.E[ 408 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.E[ 416 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.E[ 424 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 72 ]), &(acadoWorkspace.E[ 432 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.d[ 40 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.evGx[ 160 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.E[ 440 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.E[ 448 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.E[ 456 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.E[ 464 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.E[ 472 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.E[ 480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.E[ 488 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.E[ 496 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.E[ 504 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.E[ 512 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 80 ]), &(acadoWorkspace.E[ 520 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.d[ 44 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.evGx[ 176 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.E[ 528 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.E[ 536 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.E[ 544 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.E[ 552 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.E[ 560 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.E[ 568 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.E[ 576 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.E[ 584 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.E[ 592 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.E[ 600 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.E[ 608 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 88 ]), &(acadoWorkspace.E[ 616 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 44 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.d[ 48 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.evGx[ 192 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.E[ 624 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.E[ 632 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.E[ 640 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.E[ 648 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.E[ 656 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.E[ 664 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.E[ 672 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.E[ 680 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.E[ 688 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.E[ 696 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.E[ 704 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 616 ]), &(acadoWorkspace.E[ 712 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 96 ]), &(acadoWorkspace.E[ 720 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.d[ 52 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.evGx[ 208 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.E[ 728 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.E[ 736 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.E[ 744 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.E[ 752 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.E[ 760 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.E[ 768 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.E[ 776 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.E[ 784 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.E[ 792 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.E[ 800 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.E[ 808 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.E[ 816 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.E[ 824 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 104 ]), &(acadoWorkspace.E[ 832 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 52 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.d[ 56 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.evGx[ 224 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.E[ 840 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.E[ 848 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.E[ 856 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.E[ 864 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.E[ 872 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.E[ 880 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.E[ 888 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.E[ 896 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.E[ 904 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.E[ 912 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.E[ 920 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.E[ 928 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.E[ 936 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 832 ]), &(acadoWorkspace.E[ 944 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 112 ]), &(acadoWorkspace.E[ 952 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 56 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.d[ 60 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.evGx[ 240 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.E[ 960 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.E[ 968 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.E[ 976 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.E[ 984 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.E[ 992 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.E[ 1000 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.E[ 1008 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.E[ 1016 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.E[ 1024 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.E[ 1032 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.E[ 1040 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.E[ 1048 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.E[ 1056 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.E[ 1064 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 952 ]), &(acadoWorkspace.E[ 1072 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 120 ]), &(acadoWorkspace.E[ 1080 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 256 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.d[ 64 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.evGx[ 256 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.E[ 1088 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.E[ 1096 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.E[ 1104 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.E[ 1112 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.E[ 1120 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.E[ 1128 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.E[ 1136 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.E[ 1144 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.E[ 1152 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.E[ 1160 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.E[ 1168 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.E[ 1176 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.E[ 1184 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.E[ 1192 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.E[ 1200 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.E[ 1208 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 128 ]), &(acadoWorkspace.E[ 1216 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 272 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 64 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.d[ 68 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.evGx[ 272 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.E[ 1224 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.E[ 1232 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.E[ 1240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.E[ 1248 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.E[ 1256 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.E[ 1264 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.E[ 1272 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.E[ 1280 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.E[ 1288 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.E[ 1296 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.E[ 1304 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.E[ 1312 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.E[ 1320 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.E[ 1328 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.E[ 1336 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.E[ 1344 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1216 ]), &(acadoWorkspace.E[ 1352 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 136 ]), &(acadoWorkspace.E[ 1360 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 68 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.d[ 72 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.evGx[ 288 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.E[ 1368 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.E[ 1376 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.E[ 1384 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.E[ 1392 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.E[ 1400 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.E[ 1408 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.E[ 1416 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.E[ 1424 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.E[ 1432 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.E[ 1440 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.E[ 1448 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.E[ 1456 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.E[ 1464 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.E[ 1472 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.E[ 1480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.E[ 1488 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.E[ 1496 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1360 ]), &(acadoWorkspace.E[ 1504 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 144 ]), &(acadoWorkspace.E[ 1512 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 304 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.d[ 76 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.evGx[ 304 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.E[ 1520 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.E[ 1528 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.E[ 1536 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.E[ 1544 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.E[ 1552 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.E[ 1560 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.E[ 1568 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.E[ 1576 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.E[ 1584 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.E[ 1592 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.E[ 1600 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.E[ 1608 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.E[ 1616 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.E[ 1624 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.E[ 1632 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.E[ 1640 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.E[ 1648 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.E[ 1656 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1512 ]), &(acadoWorkspace.E[ 1664 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 152 ]), &(acadoWorkspace.E[ 1672 ]) );

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
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.QGx[ 304 ]) );

acado_multGxGu( &(acadoWorkspace.Q1[ 16 ]), acadoWorkspace.E, acadoWorkspace.QE );
acado_multGxGu( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QE[ 8 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 224 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 536 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 544 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.E[ 616 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 632 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 640 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 728 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 736 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 752 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.E[ 832 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 848 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 856 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.E[ 952 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 968 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 976 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1088 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1096 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1112 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.E[ 1216 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1224 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1232 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 1360 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1368 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1376 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1384 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1392 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.E[ 1512 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1520 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1528 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1536 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1544 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1552 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1560 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1568 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1576 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1584 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1592 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1600 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1608 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1616 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1624 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1632 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1640 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QE[ 1648 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.QE[ 1656 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1664 ]), &(acadoWorkspace.QE[ 1664 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1672 ]), &(acadoWorkspace.QE[ 1672 ]) );

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

acado_zeroBlockH10( acadoWorkspace.H10 );
acado_multQETGx( acadoWorkspace.QE, acadoWorkspace.evGx, acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 8 ]), &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 80 ]), &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 224 ]), &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 440 ]), &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 728 ]), &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1088 ]), &(acadoWorkspace.evGx[ 256 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1224 ]), &(acadoWorkspace.evGx[ 272 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1368 ]), &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1520 ]), &(acadoWorkspace.evGx[ 304 ]), acadoWorkspace.H10 );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 16 ]), &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 32 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 56 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 88 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 128 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 176 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 232 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 296 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 368 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 448 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 536 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 632 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 736 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 848 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 968 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1096 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1232 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1376 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1528 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 8 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 40 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 64 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 136 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 184 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 304 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 376 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 544 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 640 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 744 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 856 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 976 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1104 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1240 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1384 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1536 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 16 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 104 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 248 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 464 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 648 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 752 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 864 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 984 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1112 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1248 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1392 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1544 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 112 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 152 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 200 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 256 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 320 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 392 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 472 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 560 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 656 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 760 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 872 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 992 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1120 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1256 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1400 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1552 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 32 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 160 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 208 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 328 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 400 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 568 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 664 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 768 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 880 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1000 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1128 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1264 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1408 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1560 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 40 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 272 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 488 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 672 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 776 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 888 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1008 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1136 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1272 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1416 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1568 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 280 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 344 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 416 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 496 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 584 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 680 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 784 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 896 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1016 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1144 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1280 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1424 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1576 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 56 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 352 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 424 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 592 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 688 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 792 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 904 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1024 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1152 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1288 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1432 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1584 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 64 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 512 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 696 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 800 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 912 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1032 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1160 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1296 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1440 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1592 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 520 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 608 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 704 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 808 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 920 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1040 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1168 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1304 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1448 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1600 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 80 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 616 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 712 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 816 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 928 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1048 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1176 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1312 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1456 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1608 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 88 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 824 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 936 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1056 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1184 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1320 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1464 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1616 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 832 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 944 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1064 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1192 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1328 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1472 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1624 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 104 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 952 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1072 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1336 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1480 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1632 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 112 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1208 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1344 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1488 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1640 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 120 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 128 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1216 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.H10[ 128 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1352 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 128 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1496 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 128 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1648 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 128 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 136 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1360 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.H10[ 136 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1504 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 136 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1656 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 136 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 144 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1512 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 144 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1664 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 144 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 152 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1672 ]), &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.H10[ 152 ]) );

for (lRun2 = 0;lRun2 < 4; ++lRun2)
for (lRun3 = 0;lRun3 < 40; ++lRun3)
acadoWorkspace.H[(lRun2 * 44) + (lRun3 + 4)] = acadoWorkspace.H10[(lRun3 * 4) + (lRun2)];

acado_setBlockH11_R1( 0, 0, acadoWorkspace.R1 );
acado_setBlockH11( 0, 0, acadoWorkspace.E, acadoWorkspace.QE );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QE[ 8 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 80 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 224 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 440 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 728 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1088 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1224 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1368 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1520 ]) );

acado_zeroBlockH11( 0, 1 );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 536 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 632 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 736 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 848 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 968 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1096 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1232 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1376 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1528 ]) );

acado_zeroBlockH11( 0, 2 );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 544 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 640 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 856 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 976 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1240 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1384 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1536 ]) );

acado_zeroBlockH11( 0, 3 );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 752 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1112 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1392 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1544 ]) );

acado_zeroBlockH11( 0, 4 );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1552 ]) );

acado_zeroBlockH11( 0, 5 );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 0, 6 );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 0, 7 );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 0, 8 );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 0, 9 );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 0, 10 );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 0, 11 );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 0, 12 );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 0, 13 );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 0, 14 );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 0, 15 );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 0, 16 );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 0, 17 );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 0, 18 );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 0, 19 );
acado_setBlockH11( 0, 19, &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 1, 1, &(acadoWorkspace.R1[ 4 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QE[ 16 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 32 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 56 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 88 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 128 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 176 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 232 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 296 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 368 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 448 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 536 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 632 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 736 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 848 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 968 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1096 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1232 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1376 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1528 ]) );

acado_zeroBlockH11( 1, 2 );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 544 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 640 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 856 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 976 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1240 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1384 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1536 ]) );

acado_zeroBlockH11( 1, 3 );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 752 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1112 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1392 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1544 ]) );

acado_zeroBlockH11( 1, 4 );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1552 ]) );

acado_zeroBlockH11( 1, 5 );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 1, 6 );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 1, 7 );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 1, 8 );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 1, 9 );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 1, 10 );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 1, 11 );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 1, 12 );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 1, 13 );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 1, 14 );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 1, 15 );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 1, 16 );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 1, 17 );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 1, 18 );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 1, 19 );
acado_setBlockH11( 1, 19, &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 2, 2, &(acadoWorkspace.R1[ 8 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QE[ 40 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 64 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 136 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 184 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 304 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 376 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 544 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 640 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 856 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 976 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1240 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1384 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1536 ]) );

acado_zeroBlockH11( 2, 3 );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 752 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1112 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1392 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1544 ]) );

acado_zeroBlockH11( 2, 4 );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1552 ]) );

acado_zeroBlockH11( 2, 5 );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 2, 6 );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 2, 7 );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 2, 8 );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 2, 9 );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 2, 10 );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 2, 11 );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 2, 12 );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 2, 13 );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 2, 14 );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 2, 15 );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 2, 16 );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 2, 17 );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 2, 18 );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 2, 19 );
acado_setBlockH11( 2, 19, &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 3, 3, &(acadoWorkspace.R1[ 12 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 104 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 248 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 464 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 752 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1112 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1392 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1544 ]) );

acado_zeroBlockH11( 3, 4 );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1552 ]) );

acado_zeroBlockH11( 3, 5 );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 3, 6 );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 3, 7 );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 3, 8 );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 3, 9 );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 3, 10 );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 3, 11 );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 3, 12 );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 3, 13 );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 3, 14 );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 3, 15 );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 3, 16 );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 3, 17 );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 3, 18 );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 3, 19 );
acado_setBlockH11( 3, 19, &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 4, 4, &(acadoWorkspace.R1[ 16 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QE[ 112 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 152 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 200 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 256 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 320 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 392 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 472 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 560 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 656 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 760 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 872 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 992 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1120 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1256 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1400 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1552 ]) );

acado_zeroBlockH11( 4, 5 );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 4, 6 );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 4, 7 );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 4, 8 );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 4, 9 );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 4, 10 );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 4, 11 );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 4, 12 );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 4, 13 );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 4, 14 );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 4, 15 );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 4, 16 );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 4, 17 );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 4, 18 );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 4, 19 );
acado_setBlockH11( 4, 19, &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 5, 5, &(acadoWorkspace.R1[ 20 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QE[ 160 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 208 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 328 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 400 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 568 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 664 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 880 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1000 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1264 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1408 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1560 ]) );

acado_zeroBlockH11( 5, 6 );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 5, 7 );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 5, 8 );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 5, 9 );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 5, 10 );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 5, 11 );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 5, 12 );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 5, 13 );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 5, 14 );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 5, 15 );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 5, 16 );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 5, 17 );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 5, 18 );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 5, 19 );
acado_setBlockH11( 5, 19, &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 6, 6, &(acadoWorkspace.R1[ 24 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 272 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 488 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 776 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1136 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1272 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1416 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1568 ]) );

acado_zeroBlockH11( 6, 7 );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 6, 8 );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 6, 9 );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 6, 10 );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 6, 11 );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 6, 12 );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 6, 13 );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 6, 14 );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 6, 15 );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 6, 16 );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 6, 17 );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 6, 18 );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 6, 19 );
acado_setBlockH11( 6, 19, &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 7, 7, &(acadoWorkspace.R1[ 28 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QE[ 280 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 344 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 416 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 496 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 584 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 680 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 784 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 896 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1016 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1144 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1280 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1424 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1576 ]) );

acado_zeroBlockH11( 7, 8 );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 7, 9 );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 7, 10 );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 7, 11 );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 7, 12 );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 7, 13 );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 7, 14 );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 7, 15 );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 7, 16 );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 7, 17 );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 7, 18 );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 7, 19 );
acado_setBlockH11( 7, 19, &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 8, 8, &(acadoWorkspace.R1[ 32 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QE[ 352 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 424 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QE[ 592 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 688 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 904 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1024 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1288 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1432 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1584 ]) );

acado_zeroBlockH11( 8, 9 );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 8, 10 );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 8, 11 );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 8, 12 );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 8, 13 );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 8, 14 );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 8, 15 );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 8, 16 );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 8, 17 );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 8, 18 );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 8, 19 );
acado_setBlockH11( 8, 19, &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 9, 9, &(acadoWorkspace.R1[ 36 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 512 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 800 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1160 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1296 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1440 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1592 ]) );

acado_zeroBlockH11( 9, 10 );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 9, 11 );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 9, 12 );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 9, 13 );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 9, 14 );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 9, 15 );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 9, 16 );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 9, 17 );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 9, 18 );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 9, 19 );
acado_setBlockH11( 9, 19, &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 10, 10, &(acadoWorkspace.R1[ 40 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QE[ 520 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.QE[ 608 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.QE[ 704 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QE[ 808 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 920 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1040 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1168 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1304 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1448 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1600 ]) );

acado_zeroBlockH11( 10, 11 );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 10, 12 );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 10, 13 );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 10, 14 );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 10, 15 );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 10, 16 );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 10, 17 );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 10, 18 );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 10, 19 );
acado_setBlockH11( 10, 19, &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 11, 11, &(acadoWorkspace.R1[ 44 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 616 ]), &(acadoWorkspace.QE[ 616 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.QE[ 712 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QE[ 928 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1048 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1312 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1456 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1608 ]) );

acado_zeroBlockH11( 11, 12 );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 11, 13 );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 11, 14 );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 11, 15 );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 11, 16 );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 11, 17 );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 11, 18 );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 11, 19 );
acado_setBlockH11( 11, 19, &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 12, 12, &(acadoWorkspace.R1[ 48 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.QE[ 824 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1184 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1320 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1464 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1616 ]) );

acado_zeroBlockH11( 12, 13 );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 12, 14 );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 12, 15 );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 12, 16 );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 12, 17 );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 12, 18 );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 12, 19 );
acado_setBlockH11( 12, 19, &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 13, 13, &(acadoWorkspace.R1[ 52 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 832 ]), &(acadoWorkspace.QE[ 832 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.QE[ 944 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.QE[ 1064 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QE[ 1192 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1328 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1472 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1624 ]) );

acado_zeroBlockH11( 13, 14 );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 13, 15 );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 13, 16 );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 13, 17 );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 13, 18 );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 13, 19 );
acado_setBlockH11( 13, 19, &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 14, 14, &(acadoWorkspace.R1[ 56 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 952 ]), &(acadoWorkspace.QE[ 952 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.QE[ 1072 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QE[ 1336 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1480 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1632 ]) );

acado_zeroBlockH11( 14, 15 );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 14, 16 );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 14, 17 );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 14, 18 );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 14, 19 );
acado_setBlockH11( 14, 19, &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 15, 15, &(acadoWorkspace.R1[ 60 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.QE[ 1208 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.QE[ 1344 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QE[ 1488 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1640 ]) );

acado_zeroBlockH11( 15, 16 );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 15, 17 );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 15, 18 );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 15, 19 );
acado_setBlockH11( 15, 19, &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 16, 16, &(acadoWorkspace.R1[ 64 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1216 ]), &(acadoWorkspace.QE[ 1216 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.QE[ 1352 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.QE[ 1496 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QE[ 1648 ]) );

acado_zeroBlockH11( 16, 17 );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 16, 18 );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 16, 19 );
acado_setBlockH11( 16, 19, &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 17, 17, &(acadoWorkspace.R1[ 68 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1360 ]), &(acadoWorkspace.QE[ 1360 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.QE[ 1504 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.QE[ 1656 ]) );

acado_zeroBlockH11( 17, 18 );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 17, 19 );
acado_setBlockH11( 17, 19, &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 18, 18, &(acadoWorkspace.R1[ 72 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 1512 ]), &(acadoWorkspace.QE[ 1512 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 1664 ]), &(acadoWorkspace.QE[ 1664 ]) );

acado_zeroBlockH11( 18, 19 );
acado_setBlockH11( 18, 19, &(acadoWorkspace.E[ 1664 ]), &(acadoWorkspace.QE[ 1672 ]) );

acado_setBlockH11_R1( 19, 19, &(acadoWorkspace.R1[ 76 ]) );
acado_setBlockH11( 19, 19, &(acadoWorkspace.E[ 1672 ]), &(acadoWorkspace.QE[ 1672 ]) );


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

for (lRun2 = 0;lRun2 < 40; ++lRun2)
for (lRun3 = 0;lRun3 < 4; ++lRun3)
acadoWorkspace.H[(lRun2 * 44 + 176) + (lRun3)] = acadoWorkspace.H10[(lRun2 * 4) + (lRun3)];

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
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 76 ]), &(acadoWorkspace.Qd[ 76 ]) );

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
acado_macETSlu( acadoWorkspace.QE, &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 8 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 80 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 224 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 440 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 728 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1088 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1224 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1368 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1520 ]), &(acadoWorkspace.g[ 4 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 16 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 32 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 56 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 88 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 128 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 176 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 232 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 296 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 368 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 448 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 536 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 632 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 736 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 848 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 968 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1096 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1232 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1376 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1528 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 40 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 64 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 136 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 184 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 304 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 376 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 544 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 640 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 744 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 856 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 976 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1104 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1240 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1384 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1536 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 104 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 248 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 464 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 648 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 752 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 864 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 984 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1112 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1248 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1392 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1544 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 112 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 152 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 200 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 256 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 320 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 392 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 472 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 560 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 656 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 760 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 872 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 992 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1120 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1256 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1400 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1552 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 160 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 208 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 328 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 400 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 568 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 664 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 768 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 880 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1000 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1128 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1264 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1408 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1560 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 272 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 488 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 672 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 776 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 888 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1008 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1136 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1272 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1416 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1568 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 280 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 344 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 416 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 496 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 584 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 680 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 784 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 896 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1016 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1144 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1280 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1424 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1576 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 352 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 424 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 592 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 688 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 792 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 904 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1024 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1152 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1288 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1432 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1584 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 512 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 696 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 800 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 912 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1032 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1160 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1296 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1440 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1592 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 520 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 608 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 704 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 808 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 920 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1040 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1168 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1304 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1448 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1600 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 616 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 712 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 816 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 928 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1048 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1176 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1312 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1456 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1608 ]), &(acadoWorkspace.g[ 26 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 824 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 936 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1056 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1184 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1320 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1464 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1616 ]), &(acadoWorkspace.g[ 28 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 832 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 944 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1064 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1192 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1328 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1472 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1624 ]), &(acadoWorkspace.g[ 30 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 952 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1072 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1336 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1480 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1632 ]), &(acadoWorkspace.g[ 32 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.g[ 34 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1208 ]), &(acadoWorkspace.g[ 34 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1344 ]), &(acadoWorkspace.g[ 34 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1488 ]), &(acadoWorkspace.g[ 34 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1640 ]), &(acadoWorkspace.g[ 34 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1216 ]), &(acadoWorkspace.g[ 36 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1352 ]), &(acadoWorkspace.g[ 36 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1496 ]), &(acadoWorkspace.g[ 36 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1648 ]), &(acadoWorkspace.g[ 36 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1360 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1504 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1656 ]), &(acadoWorkspace.g[ 38 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1512 ]), &(acadoWorkspace.g[ 40 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1664 ]), &(acadoWorkspace.g[ 40 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1672 ]), &(acadoWorkspace.g[ 42 ]) );
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

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 4;
lRun4 = ((lRun3) / (4)) + (1);
acadoWorkspace.A[lRun1 * 44] = acadoWorkspace.evGx[lRun3 * 4];
acadoWorkspace.A[lRun1 * 44 + 1] = acadoWorkspace.evGx[lRun3 * 4 + 1];
acadoWorkspace.A[lRun1 * 44 + 2] = acadoWorkspace.evGx[lRun3 * 4 + 2];
acadoWorkspace.A[lRun1 * 44 + 3] = acadoWorkspace.evGx[lRun3 * 4 + 3];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (4)) + ((lRun3) % (4));
acadoWorkspace.A[(lRun1 * 44) + (lRun2 * 2 + 4)] = acadoWorkspace.E[lRun5 * 2];
acadoWorkspace.A[(lRun1 * 44) + (lRun2 * 2 + 5)] = acadoWorkspace.E[lRun5 * 2 + 1];
}
}

for (lRun1 = 0; lRun1 < 20; ++lRun1)
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
acadoWorkspace.evH[lRun1] = acadoWorkspace.conValueOut[0];

acadoWorkspace.evHx[lRun1 * 4] = acadoWorkspace.conValueOut[1];
acadoWorkspace.evHx[lRun1 * 4 + 1] = acadoWorkspace.conValueOut[2];
acadoWorkspace.evHx[lRun1 * 4 + 2] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 4 + 3] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHu[lRun1 * 2] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHu[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[6];
}

acadoWorkspace.A[880] = acadoWorkspace.evHx[0];
acadoWorkspace.A[881] = acadoWorkspace.evHx[1];
acadoWorkspace.A[882] = acadoWorkspace.evHx[2];
acadoWorkspace.A[883] = acadoWorkspace.evHx[3];

acado_multHxC( &(acadoWorkspace.evHx[ 4 ]), acadoWorkspace.evGx, &(acadoWorkspace.A[ 924 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 8 ]), &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.A[ 968 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.A[ 1012 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.A[ 1056 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.A[ 1100 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.A[ 1144 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.A[ 1188 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.A[ 1232 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.A[ 1276 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.A[ 1320 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.A[ 1364 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.A[ 1408 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.A[ 1452 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.A[ 1496 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.A[ 1540 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.A[ 1584 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.A[ 1628 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.A[ 1672 ]) );
acado_multHxC( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.A[ 1716 ]) );

acado_multHxE( &(acadoWorkspace.evHx[ 4 ]), acadoWorkspace.E, 1, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 8 ]), &(acadoWorkspace.E[ 8 ]), 2, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 8 ]), &(acadoWorkspace.E[ 16 ]), 2, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.E[ 24 ]), 3, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.E[ 32 ]), 3, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.E[ 40 ]), 3, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 48 ]), 4, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 56 ]), 4, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 64 ]), 4, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 72 ]), 4, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 80 ]), 5, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 88 ]), 5, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 96 ]), 5, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 104 ]), 5, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.E[ 112 ]), 5, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 120 ]), 6, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 128 ]), 6, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 136 ]), 6, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 144 ]), 6, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 152 ]), 6, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 160 ]), 6, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 168 ]), 7, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 176 ]), 7, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 184 ]), 7, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 192 ]), 7, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 200 ]), 7, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 208 ]), 7, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.E[ 216 ]), 7, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 224 ]), 8, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 232 ]), 8, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 240 ]), 8, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 248 ]), 8, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 256 ]), 8, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 264 ]), 8, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 272 ]), 8, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 280 ]), 8, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 288 ]), 9, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 296 ]), 9, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 304 ]), 9, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 312 ]), 9, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 320 ]), 9, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 328 ]), 9, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 336 ]), 9, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 344 ]), 9, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.E[ 352 ]), 9, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 360 ]), 10, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 368 ]), 10, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 376 ]), 10, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 384 ]), 10, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 392 ]), 10, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 400 ]), 10, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 408 ]), 10, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 416 ]), 10, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 424 ]), 10, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.E[ 432 ]), 10, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 440 ]), 11, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 448 ]), 11, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 456 ]), 11, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 464 ]), 11, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 472 ]), 11, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 480 ]), 11, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 488 ]), 11, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 496 ]), 11, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 504 ]), 11, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 512 ]), 11, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.E[ 520 ]), 11, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 528 ]), 12, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 536 ]), 12, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 544 ]), 12, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 552 ]), 12, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 560 ]), 12, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 568 ]), 12, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 576 ]), 12, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 584 ]), 12, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 592 ]), 12, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 600 ]), 12, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 608 ]), 12, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.E[ 616 ]), 12, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 624 ]), 13, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 632 ]), 13, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 640 ]), 13, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 648 ]), 13, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 656 ]), 13, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 664 ]), 13, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 672 ]), 13, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 680 ]), 13, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 688 ]), 13, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 696 ]), 13, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 704 ]), 13, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 712 ]), 13, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.E[ 720 ]), 13, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 728 ]), 14, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 736 ]), 14, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 744 ]), 14, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 752 ]), 14, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 760 ]), 14, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 768 ]), 14, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 776 ]), 14, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 784 ]), 14, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 792 ]), 14, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 800 ]), 14, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 808 ]), 14, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 816 ]), 14, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 824 ]), 14, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.E[ 832 ]), 14, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 840 ]), 15, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 848 ]), 15, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 856 ]), 15, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 864 ]), 15, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 872 ]), 15, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 880 ]), 15, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 888 ]), 15, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 896 ]), 15, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 904 ]), 15, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 912 ]), 15, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 920 ]), 15, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 928 ]), 15, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 936 ]), 15, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 944 ]), 15, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.E[ 952 ]), 15, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 960 ]), 16, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 968 ]), 16, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 976 ]), 16, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 984 ]), 16, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 992 ]), 16, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1000 ]), 16, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1008 ]), 16, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1016 ]), 16, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1024 ]), 16, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1032 ]), 16, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1040 ]), 16, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1048 ]), 16, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1056 ]), 16, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1064 ]), 16, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1072 ]), 16, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.E[ 1080 ]), 16, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1088 ]), 17, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1096 ]), 17, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1104 ]), 17, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1112 ]), 17, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1120 ]), 17, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1128 ]), 17, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1136 ]), 17, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1144 ]), 17, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1152 ]), 17, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1160 ]), 17, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1168 ]), 17, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1176 ]), 17, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1184 ]), 17, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1192 ]), 17, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1200 ]), 17, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1208 ]), 17, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.E[ 1216 ]), 17, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1224 ]), 18, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1232 ]), 18, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1240 ]), 18, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1248 ]), 18, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1256 ]), 18, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1264 ]), 18, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1272 ]), 18, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1280 ]), 18, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1288 ]), 18, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1296 ]), 18, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1304 ]), 18, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1312 ]), 18, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1320 ]), 18, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1328 ]), 18, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1336 ]), 18, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1344 ]), 18, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1352 ]), 18, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.E[ 1360 ]), 18, 17 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1368 ]), 19, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1376 ]), 19, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1384 ]), 19, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1392 ]), 19, 3 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1400 ]), 19, 4 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1408 ]), 19, 5 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1416 ]), 19, 6 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1424 ]), 19, 7 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1432 ]), 19, 8 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1440 ]), 19, 9 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1448 ]), 19, 10 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1456 ]), 19, 11 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1464 ]), 19, 12 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1472 ]), 19, 13 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1480 ]), 19, 14 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1488 ]), 19, 15 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1496 ]), 19, 16 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1504 ]), 19, 17 );
acado_multHxE( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.E[ 1512 ]), 19, 18 );

acadoWorkspace.A[884] = acadoWorkspace.evHu[0];
acadoWorkspace.A[885] = acadoWorkspace.evHu[1];
acadoWorkspace.A[930] = acadoWorkspace.evHu[2];
acadoWorkspace.A[931] = acadoWorkspace.evHu[3];
acadoWorkspace.A[976] = acadoWorkspace.evHu[4];
acadoWorkspace.A[977] = acadoWorkspace.evHu[5];
acadoWorkspace.A[1022] = acadoWorkspace.evHu[6];
acadoWorkspace.A[1023] = acadoWorkspace.evHu[7];
acadoWorkspace.A[1068] = acadoWorkspace.evHu[8];
acadoWorkspace.A[1069] = acadoWorkspace.evHu[9];
acadoWorkspace.A[1114] = acadoWorkspace.evHu[10];
acadoWorkspace.A[1115] = acadoWorkspace.evHu[11];
acadoWorkspace.A[1160] = acadoWorkspace.evHu[12];
acadoWorkspace.A[1161] = acadoWorkspace.evHu[13];
acadoWorkspace.A[1206] = acadoWorkspace.evHu[14];
acadoWorkspace.A[1207] = acadoWorkspace.evHu[15];
acadoWorkspace.A[1252] = acadoWorkspace.evHu[16];
acadoWorkspace.A[1253] = acadoWorkspace.evHu[17];
acadoWorkspace.A[1298] = acadoWorkspace.evHu[18];
acadoWorkspace.A[1299] = acadoWorkspace.evHu[19];
acadoWorkspace.A[1344] = acadoWorkspace.evHu[20];
acadoWorkspace.A[1345] = acadoWorkspace.evHu[21];
acadoWorkspace.A[1390] = acadoWorkspace.evHu[22];
acadoWorkspace.A[1391] = acadoWorkspace.evHu[23];
acadoWorkspace.A[1436] = acadoWorkspace.evHu[24];
acadoWorkspace.A[1437] = acadoWorkspace.evHu[25];
acadoWorkspace.A[1482] = acadoWorkspace.evHu[26];
acadoWorkspace.A[1483] = acadoWorkspace.evHu[27];
acadoWorkspace.A[1528] = acadoWorkspace.evHu[28];
acadoWorkspace.A[1529] = acadoWorkspace.evHu[29];
acadoWorkspace.A[1574] = acadoWorkspace.evHu[30];
acadoWorkspace.A[1575] = acadoWorkspace.evHu[31];
acadoWorkspace.A[1620] = acadoWorkspace.evHu[32];
acadoWorkspace.A[1621] = acadoWorkspace.evHu[33];
acadoWorkspace.A[1666] = acadoWorkspace.evHu[34];
acadoWorkspace.A[1667] = acadoWorkspace.evHu[35];
acadoWorkspace.A[1712] = acadoWorkspace.evHu[36];
acadoWorkspace.A[1713] = acadoWorkspace.evHu[37];
acadoWorkspace.A[1758] = acadoWorkspace.evHu[38];
acadoWorkspace.A[1759] = acadoWorkspace.evHu[39];
acadoWorkspace.lbA[20] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[0];
acadoWorkspace.lbA[21] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[1];
acadoWorkspace.lbA[22] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[2];
acadoWorkspace.lbA[23] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[3];
acadoWorkspace.lbA[24] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[4];
acadoWorkspace.lbA[25] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[5];
acadoWorkspace.lbA[26] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[6];
acadoWorkspace.lbA[27] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[7];
acadoWorkspace.lbA[28] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[8];
acadoWorkspace.lbA[29] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[9];
acadoWorkspace.lbA[30] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[10];
acadoWorkspace.lbA[31] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[11];
acadoWorkspace.lbA[32] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[12];
acadoWorkspace.lbA[33] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[13];
acadoWorkspace.lbA[34] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[14];
acadoWorkspace.lbA[35] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[15];
acadoWorkspace.lbA[36] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[16];
acadoWorkspace.lbA[37] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[17];
acadoWorkspace.lbA[38] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[18];
acadoWorkspace.lbA[39] = (real_t)-3.0000000000000000e+00 - acadoWorkspace.evH[19];

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

acado_macHxd( &(acadoWorkspace.evHx[ 4 ]), acadoWorkspace.d, &(acadoWorkspace.lbA[ 21 ]), &(acadoWorkspace.ubA[ 21 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 8 ]), &(acadoWorkspace.d[ 4 ]), &(acadoWorkspace.lbA[ 22 ]), &(acadoWorkspace.ubA[ 22 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 12 ]), &(acadoWorkspace.d[ 8 ]), &(acadoWorkspace.lbA[ 23 ]), &(acadoWorkspace.ubA[ 23 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.lbA[ 24 ]), &(acadoWorkspace.ubA[ 24 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 20 ]), &(acadoWorkspace.d[ 16 ]), &(acadoWorkspace.lbA[ 25 ]), &(acadoWorkspace.ubA[ 25 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.d[ 20 ]), &(acadoWorkspace.lbA[ 26 ]), &(acadoWorkspace.ubA[ 26 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 28 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.lbA[ 27 ]), &(acadoWorkspace.ubA[ 27 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.d[ 28 ]), &(acadoWorkspace.lbA[ 28 ]), &(acadoWorkspace.ubA[ 28 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 36 ]), &(acadoWorkspace.d[ 32 ]), &(acadoWorkspace.lbA[ 29 ]), &(acadoWorkspace.ubA[ 29 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 40 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.lbA[ 30 ]), &(acadoWorkspace.ubA[ 30 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 44 ]), &(acadoWorkspace.d[ 40 ]), &(acadoWorkspace.lbA[ 31 ]), &(acadoWorkspace.ubA[ 31 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 48 ]), &(acadoWorkspace.d[ 44 ]), &(acadoWorkspace.lbA[ 32 ]), &(acadoWorkspace.ubA[ 32 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 52 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.lbA[ 33 ]), &(acadoWorkspace.ubA[ 33 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 56 ]), &(acadoWorkspace.d[ 52 ]), &(acadoWorkspace.lbA[ 34 ]), &(acadoWorkspace.ubA[ 34 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 60 ]), &(acadoWorkspace.d[ 56 ]), &(acadoWorkspace.lbA[ 35 ]), &(acadoWorkspace.ubA[ 35 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 64 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.lbA[ 36 ]), &(acadoWorkspace.ubA[ 36 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 68 ]), &(acadoWorkspace.d[ 64 ]), &(acadoWorkspace.lbA[ 37 ]), &(acadoWorkspace.ubA[ 37 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 72 ]), &(acadoWorkspace.d[ 68 ]), &(acadoWorkspace.lbA[ 38 ]), &(acadoWorkspace.ubA[ 38 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 76 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.lbA[ 39 ]), &(acadoWorkspace.ubA[ 39 ]) );

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

acadoWorkspace.QDy[80] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[81] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[82] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[83] = + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[2];

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
acadoWorkspace.QDy[68] += acadoWorkspace.Qd[64];
acadoWorkspace.QDy[69] += acadoWorkspace.Qd[65];
acadoWorkspace.QDy[70] += acadoWorkspace.Qd[66];
acadoWorkspace.QDy[71] += acadoWorkspace.Qd[67];
acadoWorkspace.QDy[72] += acadoWorkspace.Qd[68];
acadoWorkspace.QDy[73] += acadoWorkspace.Qd[69];
acadoWorkspace.QDy[74] += acadoWorkspace.Qd[70];
acadoWorkspace.QDy[75] += acadoWorkspace.Qd[71];
acadoWorkspace.QDy[76] += acadoWorkspace.Qd[72];
acadoWorkspace.QDy[77] += acadoWorkspace.Qd[73];
acadoWorkspace.QDy[78] += acadoWorkspace.Qd[74];
acadoWorkspace.QDy[79] += acadoWorkspace.Qd[75];
acadoWorkspace.QDy[80] += acadoWorkspace.Qd[76];
acadoWorkspace.QDy[81] += acadoWorkspace.Qd[77];
acadoWorkspace.QDy[82] += acadoWorkspace.Qd[78];
acadoWorkspace.QDy[83] += acadoWorkspace.Qd[79];

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[83];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[83];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[83];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[83];


acado_multEQDy( acadoWorkspace.E, &(acadoWorkspace.QDy[ 4 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.QDy[ 8 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.QDy[ 8 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QDy[ 16 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.QDy[ 20 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QDy[ 28 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.QDy[ 32 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QDy[ 40 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.QDy[ 44 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 616 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QDy[ 52 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 832 ]), &(acadoWorkspace.QDy[ 56 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 952 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QDy[ 64 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1216 ]), &(acadoWorkspace.QDy[ 68 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1360 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1512 ]), &(acadoWorkspace.QDy[ 76 ]), &(acadoWorkspace.g[ 40 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1664 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 40 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1672 ]), &(acadoWorkspace.QDy[ 80 ]), &(acadoWorkspace.g[ 42 ]) );

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

acado_multEDu( acadoWorkspace.E, &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 4 ]) );
acado_multEDu( &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 8 ]) );
acado_multEDu( &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 8 ]) );
acado_multEDu( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 32 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 64 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 16 ]) );
acado_multEDu( &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 88 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 104 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 112 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 20 ]) );
acado_multEDu( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 128 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 136 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 152 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 160 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 176 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 184 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 200 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 208 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 28 ]) );
acado_multEDu( &(acadoWorkspace.E[ 224 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 232 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 248 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 256 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 272 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 280 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 32 ]) );
acado_multEDu( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 296 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 304 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 320 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 328 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 344 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 352 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 368 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 376 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 392 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 400 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 416 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 424 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 40 ]) );
acado_multEDu( &(acadoWorkspace.E[ 440 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 448 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 464 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 472 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 488 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 496 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 512 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 520 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 44 ]) );
acado_multEDu( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 536 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 544 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 560 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 568 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 584 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 592 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 608 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 616 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 632 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 640 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 656 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 664 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 680 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 688 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 704 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 712 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 52 ]) );
acado_multEDu( &(acadoWorkspace.E[ 728 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 736 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 752 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 760 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 776 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 784 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 800 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 808 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 824 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 832 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 56 ]) );
acado_multEDu( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 848 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 856 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 872 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 880 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 896 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 904 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 920 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 928 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 944 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 952 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 968 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 976 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 992 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1000 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1016 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1024 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1040 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1048 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1064 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1072 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.x[ 34 ]), &(acadoVariables.x[ 64 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1088 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1096 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1112 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1120 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1136 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1144 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1160 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1168 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1184 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1192 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1208 ]), &(acadoWorkspace.x[ 34 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1216 ]), &(acadoWorkspace.x[ 36 ]), &(acadoVariables.x[ 68 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1232 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1240 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1256 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1264 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1272 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1280 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1288 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1296 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1304 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1312 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1320 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1328 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1336 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1344 ]), &(acadoWorkspace.x[ 34 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1352 ]), &(acadoWorkspace.x[ 36 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1360 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1368 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1376 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1384 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1392 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1400 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1408 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1416 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1424 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1432 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1440 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1448 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1456 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1464 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1472 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1480 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1488 ]), &(acadoWorkspace.x[ 34 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1496 ]), &(acadoWorkspace.x[ 36 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1504 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1512 ]), &(acadoWorkspace.x[ 40 ]), &(acadoVariables.x[ 76 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1520 ]), &(acadoWorkspace.x[ 4 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1528 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1536 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1544 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1552 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1560 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1568 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1576 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1584 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1592 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1600 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1608 ]), &(acadoWorkspace.x[ 26 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1616 ]), &(acadoWorkspace.x[ 28 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1624 ]), &(acadoWorkspace.x[ 30 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1632 ]), &(acadoWorkspace.x[ 32 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1640 ]), &(acadoWorkspace.x[ 34 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1648 ]), &(acadoWorkspace.x[ 36 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1656 ]), &(acadoWorkspace.x[ 38 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1664 ]), &(acadoWorkspace.x[ 40 ]), &(acadoVariables.x[ 80 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1672 ]), &(acadoWorkspace.x[ 42 ]), &(acadoVariables.x[ 80 ]) );
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
for (index = 0; index < 20; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[80] = xEnd[0];
acadoVariables.x[81] = xEnd[1];
acadoVariables.x[82] = xEnd[2];
acadoVariables.x[83] = xEnd[3];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[80];
acadoWorkspace.state[1] = acadoVariables.x[81];
acadoWorkspace.state[2] = acadoVariables.x[82];
acadoWorkspace.state[3] = acadoVariables.x[83];
if (uEnd != 0)
{
acadoWorkspace.state[28] = uEnd[0];
acadoWorkspace.state[29] = uEnd[1];
}
else
{
acadoWorkspace.state[28] = acadoVariables.u[38];
acadoWorkspace.state[29] = acadoVariables.u[39];
}
acadoWorkspace.state[30] = acadoVariables.od[40];
acadoWorkspace.state[31] = acadoVariables.od[41];

acado_integrate(acadoWorkspace.state, 1, 19);

acadoVariables.x[80] = acadoWorkspace.state[0];
acadoVariables.x[81] = acadoWorkspace.state[1];
acadoVariables.x[82] = acadoWorkspace.state[2];
acadoVariables.x[83] = acadoWorkspace.state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 19; ++index)
{
acadoVariables.u[index * 2] = acadoVariables.u[index * 2 + 2];
acadoVariables.u[index * 2 + 1] = acadoVariables.u[index * 2 + 3];
}

if (uEnd != 0)
{
acadoVariables.u[38] = uEnd[0];
acadoVariables.u[39] = uEnd[1];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43];
kkt = fabs( kkt );
for (index = 0; index < 44; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 40; ++index)
{
prd = acadoWorkspace.y[index + 44];
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

for (lRun1 = 0; lRun1 < 20; ++lRun1)
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
acadoWorkspace.objValueIn[0] = acadoVariables.x[80];
acadoWorkspace.objValueIn[1] = acadoVariables.x[81];
acadoWorkspace.objValueIn[2] = acadoVariables.x[82];
acadoWorkspace.objValueIn[3] = acadoVariables.x[83];
acadoWorkspace.objValueIn[4] = acadoVariables.od[40];
acadoWorkspace.objValueIn[5] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
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

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0] + acadoWorkspace.DyN[1]*acadoVariables.WN[3] + acadoWorkspace.DyN[2]*acadoVariables.WN[6];
tmpDyN[1] = + acadoWorkspace.DyN[0]*acadoVariables.WN[1] + acadoWorkspace.DyN[1]*acadoVariables.WN[4] + acadoWorkspace.DyN[2]*acadoVariables.WN[7];
tmpDyN[2] = + acadoWorkspace.DyN[0]*acadoVariables.WN[2] + acadoWorkspace.DyN[1]*acadoVariables.WN[5] + acadoWorkspace.DyN[2]*acadoVariables.WN[8];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2];

objVal *= 0.5;
return objVal;
}

