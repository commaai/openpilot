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
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 6];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 6 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 6 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 6 + 3];
acadoWorkspace.state[4] = acadoVariables.x[lRun1 * 6 + 4];
acadoWorkspace.state[5] = acadoVariables.x[lRun1 * 6 + 5];

acadoWorkspace.state[48] = acadoVariables.u[lRun1];
acadoWorkspace.state[49] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.state[50] = acadoVariables.od[lRun1 * 2 + 1];

ret = acado_integrate(acadoWorkspace.state, 1, lRun1);

acadoWorkspace.d[lRun1 * 6] = acadoWorkspace.state[0] - acadoVariables.x[lRun1 * 6 + 6];
acadoWorkspace.d[lRun1 * 6 + 1] = acadoWorkspace.state[1] - acadoVariables.x[lRun1 * 6 + 7];
acadoWorkspace.d[lRun1 * 6 + 2] = acadoWorkspace.state[2] - acadoVariables.x[lRun1 * 6 + 8];
acadoWorkspace.d[lRun1 * 6 + 3] = acadoWorkspace.state[3] - acadoVariables.x[lRun1 * 6 + 9];
acadoWorkspace.d[lRun1 * 6 + 4] = acadoWorkspace.state[4] - acadoVariables.x[lRun1 * 6 + 10];
acadoWorkspace.d[lRun1 * 6 + 5] = acadoWorkspace.state[5] - acadoVariables.x[lRun1 * 6 + 11];

acadoWorkspace.evGx[lRun1 * 36] = acadoWorkspace.state[6];
acadoWorkspace.evGx[lRun1 * 36 + 1] = acadoWorkspace.state[7];
acadoWorkspace.evGx[lRun1 * 36 + 2] = acadoWorkspace.state[8];
acadoWorkspace.evGx[lRun1 * 36 + 3] = acadoWorkspace.state[9];
acadoWorkspace.evGx[lRun1 * 36 + 4] = acadoWorkspace.state[10];
acadoWorkspace.evGx[lRun1 * 36 + 5] = acadoWorkspace.state[11];
acadoWorkspace.evGx[lRun1 * 36 + 6] = acadoWorkspace.state[12];
acadoWorkspace.evGx[lRun1 * 36 + 7] = acadoWorkspace.state[13];
acadoWorkspace.evGx[lRun1 * 36 + 8] = acadoWorkspace.state[14];
acadoWorkspace.evGx[lRun1 * 36 + 9] = acadoWorkspace.state[15];
acadoWorkspace.evGx[lRun1 * 36 + 10] = acadoWorkspace.state[16];
acadoWorkspace.evGx[lRun1 * 36 + 11] = acadoWorkspace.state[17];
acadoWorkspace.evGx[lRun1 * 36 + 12] = acadoWorkspace.state[18];
acadoWorkspace.evGx[lRun1 * 36 + 13] = acadoWorkspace.state[19];
acadoWorkspace.evGx[lRun1 * 36 + 14] = acadoWorkspace.state[20];
acadoWorkspace.evGx[lRun1 * 36 + 15] = acadoWorkspace.state[21];
acadoWorkspace.evGx[lRun1 * 36 + 16] = acadoWorkspace.state[22];
acadoWorkspace.evGx[lRun1 * 36 + 17] = acadoWorkspace.state[23];
acadoWorkspace.evGx[lRun1 * 36 + 18] = acadoWorkspace.state[24];
acadoWorkspace.evGx[lRun1 * 36 + 19] = acadoWorkspace.state[25];
acadoWorkspace.evGx[lRun1 * 36 + 20] = acadoWorkspace.state[26];
acadoWorkspace.evGx[lRun1 * 36 + 21] = acadoWorkspace.state[27];
acadoWorkspace.evGx[lRun1 * 36 + 22] = acadoWorkspace.state[28];
acadoWorkspace.evGx[lRun1 * 36 + 23] = acadoWorkspace.state[29];
acadoWorkspace.evGx[lRun1 * 36 + 24] = acadoWorkspace.state[30];
acadoWorkspace.evGx[lRun1 * 36 + 25] = acadoWorkspace.state[31];
acadoWorkspace.evGx[lRun1 * 36 + 26] = acadoWorkspace.state[32];
acadoWorkspace.evGx[lRun1 * 36 + 27] = acadoWorkspace.state[33];
acadoWorkspace.evGx[lRun1 * 36 + 28] = acadoWorkspace.state[34];
acadoWorkspace.evGx[lRun1 * 36 + 29] = acadoWorkspace.state[35];
acadoWorkspace.evGx[lRun1 * 36 + 30] = acadoWorkspace.state[36];
acadoWorkspace.evGx[lRun1 * 36 + 31] = acadoWorkspace.state[37];
acadoWorkspace.evGx[lRun1 * 36 + 32] = acadoWorkspace.state[38];
acadoWorkspace.evGx[lRun1 * 36 + 33] = acadoWorkspace.state[39];
acadoWorkspace.evGx[lRun1 * 36 + 34] = acadoWorkspace.state[40];
acadoWorkspace.evGx[lRun1 * 36 + 35] = acadoWorkspace.state[41];

acadoWorkspace.evGu[lRun1 * 6] = acadoWorkspace.state[42];
acadoWorkspace.evGu[lRun1 * 6 + 1] = acadoWorkspace.state[43];
acadoWorkspace.evGu[lRun1 * 6 + 2] = acadoWorkspace.state[44];
acadoWorkspace.evGu[lRun1 * 6 + 3] = acadoWorkspace.state[45];
acadoWorkspace.evGu[lRun1 * 6 + 4] = acadoWorkspace.state[46];
acadoWorkspace.evGu[lRun1 * 6 + 5] = acadoWorkspace.state[47];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 6;
/* Vector of auxiliary variables; number of elements: 30. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[1] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[2] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[3] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(a[2]+(real_t)(1.0000000000000001e-01))))));
a[4] = ((real_t)(1.0000000000000000e+00)/(a[0]+(real_t)(1.0000000000000001e-01)));
a[5] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[6] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[4]))*a[5]);
a[7] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[8] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[9] = (a[8]*(real_t)(5.0000000000000000e-01));
a[10] = (a[4]*a[4]);
a[11] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[7]))*a[4])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))*a[9])*a[10])))*a[5]);
a[12] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[13] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[14] = ((real_t)(1.0000000000000000e+00)/(a[2]+(real_t)(1.0000000000000001e-01)));
a[15] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[16] = (a[15]*(real_t)(5.0000000000000000e-01));
a[17] = (a[14]*a[14]);
a[18] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(a[2]+(real_t)(1.0000000000000001e-01))))));
a[19] = (((real_t)(2.9999999999999999e-01)*((((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[12]))-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[13])))*a[14])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))*a[16])*a[17])))*a[18]);
a[20] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[4]))*a[5]);
a[21] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[22] = (((real_t)(2.9999999999999999e-01)*((((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[21]))*a[4]))*a[5]);
a[23] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[24] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[25] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[23]))-(((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[24])))*a[14]))*a[18]);
a[26] = ((real_t)(1.0000000000000000e+00)/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
a[27] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[28] = (a[26]*a[26]);
a[29] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));

/* Compute outputs: */
out[0] = (a[1]-a[3]);
out[1] = (((xd[3]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = (u[0]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[4] = a[6];
out[5] = (a[11]-a[19]);
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = a[20];
out[8] = (a[22]-a[25]);
out[9] = (real_t)(0.0000000000000000e+00);
out[10] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[26]);
out[11] = ((((real_t)(0.0000000000000000e+00)-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[27])))*a[26])-((((xd[3]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))*(real_t)(5.0000000000000003e-02))*a[28]));
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = a[26];
out[14] = (((real_t)(0.0000000000000000e+00)-(((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[29])))*a[26]);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[18] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (real_t)(0.0000000000000000e+00);
out[21] = (real_t)(0.0000000000000000e+00);
out[22] = (real_t)(0.0000000000000000e+00);
out[23] = (u[0]*(real_t)(1.0000000000000001e-01));
out[24] = (real_t)(0.0000000000000000e+00);
out[25] = (real_t)(0.0000000000000000e+00);
out[26] = (real_t)(0.0000000000000000e+00);
out[27] = (real_t)(0.0000000000000000e+00);
out[28] = (real_t)(0.0000000000000000e+00);
out[29] = (real_t)(0.0000000000000000e+00);
out[30] = (real_t)(0.0000000000000000e+00);
out[31] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
/* Vector of auxiliary variables; number of elements: 30. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[1] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[2] = (sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[3] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(a[2]+(real_t)(1.0000000000000001e-01))))));
a[4] = ((real_t)(1.0000000000000000e+00)/(a[0]+(real_t)(1.0000000000000001e-01)));
a[5] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))/(a[0]+(real_t)(1.0000000000000001e-01))))));
a[6] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[4]))*a[5]);
a[7] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[8] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[9] = (a[8]*(real_t)(5.0000000000000000e-01));
a[10] = (a[4]*a[4]);
a[11] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[7]))*a[4])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-(xd[3]-xd[0]))*a[9])*a[10])))*a[5]);
a[12] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[13] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[14] = ((real_t)(1.0000000000000000e+00)/(a[2]+(real_t)(1.0000000000000001e-01)));
a[15] = (1.0/sqrt((xd[1]+(real_t)(5.0000000000000000e-01))));
a[16] = (a[15]*(real_t)(5.0000000000000000e-01));
a[17] = (a[14]*a[14]);
a[18] = (exp(((real_t)(2.9999999999999999e-01)*(((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(a[2]+(real_t)(1.0000000000000001e-01))))));
a[19] = (((real_t)(2.9999999999999999e-01)*((((((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[12]))-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[13])))*a[14])-((((((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))+(real_t)(4.0000000000000000e+00))-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))*a[16])*a[17])))*a[18]);
a[20] = (((real_t)(2.9999999999999999e-01)*(((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[4]))*a[5]);
a[21] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[22] = (((real_t)(2.9999999999999999e-01)*((((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[21]))*a[4]))*a[5]);
a[23] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[24] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[25] = (((real_t)(2.9999999999999999e-01)*(((((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[23]))-(((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[24])))*a[14]))*a[18]);
a[26] = ((real_t)(1.0000000000000000e+00)/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
a[27] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));
a[28] = (a[26]*a[26]);
a[29] = ((real_t)(1.0000000000000000e+00)/(real_t)(1.9620000000000001e+01));

/* Compute outputs: */
out[0] = (a[1]-a[3]);
out[1] = (((xd[3]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))/(((real_t)(5.0000000000000003e-02)*xd[1])+(real_t)(5.0000000000000000e-01)));
out[2] = (xd[2]*(((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00)));
out[3] = a[6];
out[4] = (a[11]-a[19]);
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = a[20];
out[7] = (a[22]-a[25]);
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[26]);
out[10] = ((((real_t)(0.0000000000000000e+00)-(((real_t)(1.8000000000000000e+00)-((real_t)(-1.8000000000000000e+00)))+((xd[1]+xd[1])*a[27])))*a[26])-((((xd[3]-xd[0])-((real_t)(4.0000000000000000e+00)+((((xd[1]*(real_t)(1.8000000000000000e+00))-((xd[4]-xd[1])*(real_t)(1.8000000000000000e+00)))+((xd[1]*xd[1])/(real_t)(1.9620000000000001e+01)))-((xd[4]*xd[4])/(real_t)(1.9620000000000001e+01)))))*(real_t)(5.0000000000000003e-02))*a[28]));
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = a[26];
out[13] = (((real_t)(0.0000000000000000e+00)-(((real_t)(0.0000000000000000e+00)-(real_t)(1.8000000000000000e+00))-((xd[4]+xd[4])*a[29])))*a[26]);
out[14] = (real_t)(0.0000000000000000e+00);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (xd[2]*(real_t)(1.0000000000000001e-01));
out[17] = (((real_t)(1.0000000000000001e-01)*xd[1])+(real_t)(1.0000000000000000e+00));
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = (real_t)(0.0000000000000000e+00);
out[20] = (real_t)(0.0000000000000000e+00);
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0]*tmpObjS[0] + tmpFx[6]*tmpObjS[4] + tmpFx[12]*tmpObjS[8] + tmpFx[18]*tmpObjS[12];
tmpQ2[1] = + tmpFx[0]*tmpObjS[1] + tmpFx[6]*tmpObjS[5] + tmpFx[12]*tmpObjS[9] + tmpFx[18]*tmpObjS[13];
tmpQ2[2] = + tmpFx[0]*tmpObjS[2] + tmpFx[6]*tmpObjS[6] + tmpFx[12]*tmpObjS[10] + tmpFx[18]*tmpObjS[14];
tmpQ2[3] = + tmpFx[0]*tmpObjS[3] + tmpFx[6]*tmpObjS[7] + tmpFx[12]*tmpObjS[11] + tmpFx[18]*tmpObjS[15];
tmpQ2[4] = + tmpFx[1]*tmpObjS[0] + tmpFx[7]*tmpObjS[4] + tmpFx[13]*tmpObjS[8] + tmpFx[19]*tmpObjS[12];
tmpQ2[5] = + tmpFx[1]*tmpObjS[1] + tmpFx[7]*tmpObjS[5] + tmpFx[13]*tmpObjS[9] + tmpFx[19]*tmpObjS[13];
tmpQ2[6] = + tmpFx[1]*tmpObjS[2] + tmpFx[7]*tmpObjS[6] + tmpFx[13]*tmpObjS[10] + tmpFx[19]*tmpObjS[14];
tmpQ2[7] = + tmpFx[1]*tmpObjS[3] + tmpFx[7]*tmpObjS[7] + tmpFx[13]*tmpObjS[11] + tmpFx[19]*tmpObjS[15];
tmpQ2[8] = + tmpFx[2]*tmpObjS[0] + tmpFx[8]*tmpObjS[4] + tmpFx[14]*tmpObjS[8] + tmpFx[20]*tmpObjS[12];
tmpQ2[9] = + tmpFx[2]*tmpObjS[1] + tmpFx[8]*tmpObjS[5] + tmpFx[14]*tmpObjS[9] + tmpFx[20]*tmpObjS[13];
tmpQ2[10] = + tmpFx[2]*tmpObjS[2] + tmpFx[8]*tmpObjS[6] + tmpFx[14]*tmpObjS[10] + tmpFx[20]*tmpObjS[14];
tmpQ2[11] = + tmpFx[2]*tmpObjS[3] + tmpFx[8]*tmpObjS[7] + tmpFx[14]*tmpObjS[11] + tmpFx[20]*tmpObjS[15];
tmpQ2[12] = + tmpFx[3]*tmpObjS[0] + tmpFx[9]*tmpObjS[4] + tmpFx[15]*tmpObjS[8] + tmpFx[21]*tmpObjS[12];
tmpQ2[13] = + tmpFx[3]*tmpObjS[1] + tmpFx[9]*tmpObjS[5] + tmpFx[15]*tmpObjS[9] + tmpFx[21]*tmpObjS[13];
tmpQ2[14] = + tmpFx[3]*tmpObjS[2] + tmpFx[9]*tmpObjS[6] + tmpFx[15]*tmpObjS[10] + tmpFx[21]*tmpObjS[14];
tmpQ2[15] = + tmpFx[3]*tmpObjS[3] + tmpFx[9]*tmpObjS[7] + tmpFx[15]*tmpObjS[11] + tmpFx[21]*tmpObjS[15];
tmpQ2[16] = + tmpFx[4]*tmpObjS[0] + tmpFx[10]*tmpObjS[4] + tmpFx[16]*tmpObjS[8] + tmpFx[22]*tmpObjS[12];
tmpQ2[17] = + tmpFx[4]*tmpObjS[1] + tmpFx[10]*tmpObjS[5] + tmpFx[16]*tmpObjS[9] + tmpFx[22]*tmpObjS[13];
tmpQ2[18] = + tmpFx[4]*tmpObjS[2] + tmpFx[10]*tmpObjS[6] + tmpFx[16]*tmpObjS[10] + tmpFx[22]*tmpObjS[14];
tmpQ2[19] = + tmpFx[4]*tmpObjS[3] + tmpFx[10]*tmpObjS[7] + tmpFx[16]*tmpObjS[11] + tmpFx[22]*tmpObjS[15];
tmpQ2[20] = + tmpFx[5]*tmpObjS[0] + tmpFx[11]*tmpObjS[4] + tmpFx[17]*tmpObjS[8] + tmpFx[23]*tmpObjS[12];
tmpQ2[21] = + tmpFx[5]*tmpObjS[1] + tmpFx[11]*tmpObjS[5] + tmpFx[17]*tmpObjS[9] + tmpFx[23]*tmpObjS[13];
tmpQ2[22] = + tmpFx[5]*tmpObjS[2] + tmpFx[11]*tmpObjS[6] + tmpFx[17]*tmpObjS[10] + tmpFx[23]*tmpObjS[14];
tmpQ2[23] = + tmpFx[5]*tmpObjS[3] + tmpFx[11]*tmpObjS[7] + tmpFx[17]*tmpObjS[11] + tmpFx[23]*tmpObjS[15];
tmpQ1[0] = + tmpQ2[0]*tmpFx[0] + tmpQ2[1]*tmpFx[6] + tmpQ2[2]*tmpFx[12] + tmpQ2[3]*tmpFx[18];
tmpQ1[1] = + tmpQ2[0]*tmpFx[1] + tmpQ2[1]*tmpFx[7] + tmpQ2[2]*tmpFx[13] + tmpQ2[3]*tmpFx[19];
tmpQ1[2] = + tmpQ2[0]*tmpFx[2] + tmpQ2[1]*tmpFx[8] + tmpQ2[2]*tmpFx[14] + tmpQ2[3]*tmpFx[20];
tmpQ1[3] = + tmpQ2[0]*tmpFx[3] + tmpQ2[1]*tmpFx[9] + tmpQ2[2]*tmpFx[15] + tmpQ2[3]*tmpFx[21];
tmpQ1[4] = + tmpQ2[0]*tmpFx[4] + tmpQ2[1]*tmpFx[10] + tmpQ2[2]*tmpFx[16] + tmpQ2[3]*tmpFx[22];
tmpQ1[5] = + tmpQ2[0]*tmpFx[5] + tmpQ2[1]*tmpFx[11] + tmpQ2[2]*tmpFx[17] + tmpQ2[3]*tmpFx[23];
tmpQ1[6] = + tmpQ2[4]*tmpFx[0] + tmpQ2[5]*tmpFx[6] + tmpQ2[6]*tmpFx[12] + tmpQ2[7]*tmpFx[18];
tmpQ1[7] = + tmpQ2[4]*tmpFx[1] + tmpQ2[5]*tmpFx[7] + tmpQ2[6]*tmpFx[13] + tmpQ2[7]*tmpFx[19];
tmpQ1[8] = + tmpQ2[4]*tmpFx[2] + tmpQ2[5]*tmpFx[8] + tmpQ2[6]*tmpFx[14] + tmpQ2[7]*tmpFx[20];
tmpQ1[9] = + tmpQ2[4]*tmpFx[3] + tmpQ2[5]*tmpFx[9] + tmpQ2[6]*tmpFx[15] + tmpQ2[7]*tmpFx[21];
tmpQ1[10] = + tmpQ2[4]*tmpFx[4] + tmpQ2[5]*tmpFx[10] + tmpQ2[6]*tmpFx[16] + tmpQ2[7]*tmpFx[22];
tmpQ1[11] = + tmpQ2[4]*tmpFx[5] + tmpQ2[5]*tmpFx[11] + tmpQ2[6]*tmpFx[17] + tmpQ2[7]*tmpFx[23];
tmpQ1[12] = + tmpQ2[8]*tmpFx[0] + tmpQ2[9]*tmpFx[6] + tmpQ2[10]*tmpFx[12] + tmpQ2[11]*tmpFx[18];
tmpQ1[13] = + tmpQ2[8]*tmpFx[1] + tmpQ2[9]*tmpFx[7] + tmpQ2[10]*tmpFx[13] + tmpQ2[11]*tmpFx[19];
tmpQ1[14] = + tmpQ2[8]*tmpFx[2] + tmpQ2[9]*tmpFx[8] + tmpQ2[10]*tmpFx[14] + tmpQ2[11]*tmpFx[20];
tmpQ1[15] = + tmpQ2[8]*tmpFx[3] + tmpQ2[9]*tmpFx[9] + tmpQ2[10]*tmpFx[15] + tmpQ2[11]*tmpFx[21];
tmpQ1[16] = + tmpQ2[8]*tmpFx[4] + tmpQ2[9]*tmpFx[10] + tmpQ2[10]*tmpFx[16] + tmpQ2[11]*tmpFx[22];
tmpQ1[17] = + tmpQ2[8]*tmpFx[5] + tmpQ2[9]*tmpFx[11] + tmpQ2[10]*tmpFx[17] + tmpQ2[11]*tmpFx[23];
tmpQ1[18] = + tmpQ2[12]*tmpFx[0] + tmpQ2[13]*tmpFx[6] + tmpQ2[14]*tmpFx[12] + tmpQ2[15]*tmpFx[18];
tmpQ1[19] = + tmpQ2[12]*tmpFx[1] + tmpQ2[13]*tmpFx[7] + tmpQ2[14]*tmpFx[13] + tmpQ2[15]*tmpFx[19];
tmpQ1[20] = + tmpQ2[12]*tmpFx[2] + tmpQ2[13]*tmpFx[8] + tmpQ2[14]*tmpFx[14] + tmpQ2[15]*tmpFx[20];
tmpQ1[21] = + tmpQ2[12]*tmpFx[3] + tmpQ2[13]*tmpFx[9] + tmpQ2[14]*tmpFx[15] + tmpQ2[15]*tmpFx[21];
tmpQ1[22] = + tmpQ2[12]*tmpFx[4] + tmpQ2[13]*tmpFx[10] + tmpQ2[14]*tmpFx[16] + tmpQ2[15]*tmpFx[22];
tmpQ1[23] = + tmpQ2[12]*tmpFx[5] + tmpQ2[13]*tmpFx[11] + tmpQ2[14]*tmpFx[17] + tmpQ2[15]*tmpFx[23];
tmpQ1[24] = + tmpQ2[16]*tmpFx[0] + tmpQ2[17]*tmpFx[6] + tmpQ2[18]*tmpFx[12] + tmpQ2[19]*tmpFx[18];
tmpQ1[25] = + tmpQ2[16]*tmpFx[1] + tmpQ2[17]*tmpFx[7] + tmpQ2[18]*tmpFx[13] + tmpQ2[19]*tmpFx[19];
tmpQ1[26] = + tmpQ2[16]*tmpFx[2] + tmpQ2[17]*tmpFx[8] + tmpQ2[18]*tmpFx[14] + tmpQ2[19]*tmpFx[20];
tmpQ1[27] = + tmpQ2[16]*tmpFx[3] + tmpQ2[17]*tmpFx[9] + tmpQ2[18]*tmpFx[15] + tmpQ2[19]*tmpFx[21];
tmpQ1[28] = + tmpQ2[16]*tmpFx[4] + tmpQ2[17]*tmpFx[10] + tmpQ2[18]*tmpFx[16] + tmpQ2[19]*tmpFx[22];
tmpQ1[29] = + tmpQ2[16]*tmpFx[5] + tmpQ2[17]*tmpFx[11] + tmpQ2[18]*tmpFx[17] + tmpQ2[19]*tmpFx[23];
tmpQ1[30] = + tmpQ2[20]*tmpFx[0] + tmpQ2[21]*tmpFx[6] + tmpQ2[22]*tmpFx[12] + tmpQ2[23]*tmpFx[18];
tmpQ1[31] = + tmpQ2[20]*tmpFx[1] + tmpQ2[21]*tmpFx[7] + tmpQ2[22]*tmpFx[13] + tmpQ2[23]*tmpFx[19];
tmpQ1[32] = + tmpQ2[20]*tmpFx[2] + tmpQ2[21]*tmpFx[8] + tmpQ2[22]*tmpFx[14] + tmpQ2[23]*tmpFx[20];
tmpQ1[33] = + tmpQ2[20]*tmpFx[3] + tmpQ2[21]*tmpFx[9] + tmpQ2[22]*tmpFx[15] + tmpQ2[23]*tmpFx[21];
tmpQ1[34] = + tmpQ2[20]*tmpFx[4] + tmpQ2[21]*tmpFx[10] + tmpQ2[22]*tmpFx[16] + tmpQ2[23]*tmpFx[22];
tmpQ1[35] = + tmpQ2[20]*tmpFx[5] + tmpQ2[21]*tmpFx[11] + tmpQ2[22]*tmpFx[17] + tmpQ2[23]*tmpFx[23];
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
tmpQN2[0] = + tmpFx[0]*tmpObjSEndTerm[0] + tmpFx[6]*tmpObjSEndTerm[3] + tmpFx[12]*tmpObjSEndTerm[6];
tmpQN2[1] = + tmpFx[0]*tmpObjSEndTerm[1] + tmpFx[6]*tmpObjSEndTerm[4] + tmpFx[12]*tmpObjSEndTerm[7];
tmpQN2[2] = + tmpFx[0]*tmpObjSEndTerm[2] + tmpFx[6]*tmpObjSEndTerm[5] + tmpFx[12]*tmpObjSEndTerm[8];
tmpQN2[3] = + tmpFx[1]*tmpObjSEndTerm[0] + tmpFx[7]*tmpObjSEndTerm[3] + tmpFx[13]*tmpObjSEndTerm[6];
tmpQN2[4] = + tmpFx[1]*tmpObjSEndTerm[1] + tmpFx[7]*tmpObjSEndTerm[4] + tmpFx[13]*tmpObjSEndTerm[7];
tmpQN2[5] = + tmpFx[1]*tmpObjSEndTerm[2] + tmpFx[7]*tmpObjSEndTerm[5] + tmpFx[13]*tmpObjSEndTerm[8];
tmpQN2[6] = + tmpFx[2]*tmpObjSEndTerm[0] + tmpFx[8]*tmpObjSEndTerm[3] + tmpFx[14]*tmpObjSEndTerm[6];
tmpQN2[7] = + tmpFx[2]*tmpObjSEndTerm[1] + tmpFx[8]*tmpObjSEndTerm[4] + tmpFx[14]*tmpObjSEndTerm[7];
tmpQN2[8] = + tmpFx[2]*tmpObjSEndTerm[2] + tmpFx[8]*tmpObjSEndTerm[5] + tmpFx[14]*tmpObjSEndTerm[8];
tmpQN2[9] = + tmpFx[3]*tmpObjSEndTerm[0] + tmpFx[9]*tmpObjSEndTerm[3] + tmpFx[15]*tmpObjSEndTerm[6];
tmpQN2[10] = + tmpFx[3]*tmpObjSEndTerm[1] + tmpFx[9]*tmpObjSEndTerm[4] + tmpFx[15]*tmpObjSEndTerm[7];
tmpQN2[11] = + tmpFx[3]*tmpObjSEndTerm[2] + tmpFx[9]*tmpObjSEndTerm[5] + tmpFx[15]*tmpObjSEndTerm[8];
tmpQN2[12] = + tmpFx[4]*tmpObjSEndTerm[0] + tmpFx[10]*tmpObjSEndTerm[3] + tmpFx[16]*tmpObjSEndTerm[6];
tmpQN2[13] = + tmpFx[4]*tmpObjSEndTerm[1] + tmpFx[10]*tmpObjSEndTerm[4] + tmpFx[16]*tmpObjSEndTerm[7];
tmpQN2[14] = + tmpFx[4]*tmpObjSEndTerm[2] + tmpFx[10]*tmpObjSEndTerm[5] + tmpFx[16]*tmpObjSEndTerm[8];
tmpQN2[15] = + tmpFx[5]*tmpObjSEndTerm[0] + tmpFx[11]*tmpObjSEndTerm[3] + tmpFx[17]*tmpObjSEndTerm[6];
tmpQN2[16] = + tmpFx[5]*tmpObjSEndTerm[1] + tmpFx[11]*tmpObjSEndTerm[4] + tmpFx[17]*tmpObjSEndTerm[7];
tmpQN2[17] = + tmpFx[5]*tmpObjSEndTerm[2] + tmpFx[11]*tmpObjSEndTerm[5] + tmpFx[17]*tmpObjSEndTerm[8];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0] + tmpQN2[1]*tmpFx[6] + tmpQN2[2]*tmpFx[12];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1] + tmpQN2[1]*tmpFx[7] + tmpQN2[2]*tmpFx[13];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2] + tmpQN2[1]*tmpFx[8] + tmpQN2[2]*tmpFx[14];
tmpQN1[3] = + tmpQN2[0]*tmpFx[3] + tmpQN2[1]*tmpFx[9] + tmpQN2[2]*tmpFx[15];
tmpQN1[4] = + tmpQN2[0]*tmpFx[4] + tmpQN2[1]*tmpFx[10] + tmpQN2[2]*tmpFx[16];
tmpQN1[5] = + tmpQN2[0]*tmpFx[5] + tmpQN2[1]*tmpFx[11] + tmpQN2[2]*tmpFx[17];
tmpQN1[6] = + tmpQN2[3]*tmpFx[0] + tmpQN2[4]*tmpFx[6] + tmpQN2[5]*tmpFx[12];
tmpQN1[7] = + tmpQN2[3]*tmpFx[1] + tmpQN2[4]*tmpFx[7] + tmpQN2[5]*tmpFx[13];
tmpQN1[8] = + tmpQN2[3]*tmpFx[2] + tmpQN2[4]*tmpFx[8] + tmpQN2[5]*tmpFx[14];
tmpQN1[9] = + tmpQN2[3]*tmpFx[3] + tmpQN2[4]*tmpFx[9] + tmpQN2[5]*tmpFx[15];
tmpQN1[10] = + tmpQN2[3]*tmpFx[4] + tmpQN2[4]*tmpFx[10] + tmpQN2[5]*tmpFx[16];
tmpQN1[11] = + tmpQN2[3]*tmpFx[5] + tmpQN2[4]*tmpFx[11] + tmpQN2[5]*tmpFx[17];
tmpQN1[12] = + tmpQN2[6]*tmpFx[0] + tmpQN2[7]*tmpFx[6] + tmpQN2[8]*tmpFx[12];
tmpQN1[13] = + tmpQN2[6]*tmpFx[1] + tmpQN2[7]*tmpFx[7] + tmpQN2[8]*tmpFx[13];
tmpQN1[14] = + tmpQN2[6]*tmpFx[2] + tmpQN2[7]*tmpFx[8] + tmpQN2[8]*tmpFx[14];
tmpQN1[15] = + tmpQN2[6]*tmpFx[3] + tmpQN2[7]*tmpFx[9] + tmpQN2[8]*tmpFx[15];
tmpQN1[16] = + tmpQN2[6]*tmpFx[4] + tmpQN2[7]*tmpFx[10] + tmpQN2[8]*tmpFx[16];
tmpQN1[17] = + tmpQN2[6]*tmpFx[5] + tmpQN2[7]*tmpFx[11] + tmpQN2[8]*tmpFx[17];
tmpQN1[18] = + tmpQN2[9]*tmpFx[0] + tmpQN2[10]*tmpFx[6] + tmpQN2[11]*tmpFx[12];
tmpQN1[19] = + tmpQN2[9]*tmpFx[1] + tmpQN2[10]*tmpFx[7] + tmpQN2[11]*tmpFx[13];
tmpQN1[20] = + tmpQN2[9]*tmpFx[2] + tmpQN2[10]*tmpFx[8] + tmpQN2[11]*tmpFx[14];
tmpQN1[21] = + tmpQN2[9]*tmpFx[3] + tmpQN2[10]*tmpFx[9] + tmpQN2[11]*tmpFx[15];
tmpQN1[22] = + tmpQN2[9]*tmpFx[4] + tmpQN2[10]*tmpFx[10] + tmpQN2[11]*tmpFx[16];
tmpQN1[23] = + tmpQN2[9]*tmpFx[5] + tmpQN2[10]*tmpFx[11] + tmpQN2[11]*tmpFx[17];
tmpQN1[24] = + tmpQN2[12]*tmpFx[0] + tmpQN2[13]*tmpFx[6] + tmpQN2[14]*tmpFx[12];
tmpQN1[25] = + tmpQN2[12]*tmpFx[1] + tmpQN2[13]*tmpFx[7] + tmpQN2[14]*tmpFx[13];
tmpQN1[26] = + tmpQN2[12]*tmpFx[2] + tmpQN2[13]*tmpFx[8] + tmpQN2[14]*tmpFx[14];
tmpQN1[27] = + tmpQN2[12]*tmpFx[3] + tmpQN2[13]*tmpFx[9] + tmpQN2[14]*tmpFx[15];
tmpQN1[28] = + tmpQN2[12]*tmpFx[4] + tmpQN2[13]*tmpFx[10] + tmpQN2[14]*tmpFx[16];
tmpQN1[29] = + tmpQN2[12]*tmpFx[5] + tmpQN2[13]*tmpFx[11] + tmpQN2[14]*tmpFx[17];
tmpQN1[30] = + tmpQN2[15]*tmpFx[0] + tmpQN2[16]*tmpFx[6] + tmpQN2[17]*tmpFx[12];
tmpQN1[31] = + tmpQN2[15]*tmpFx[1] + tmpQN2[16]*tmpFx[7] + tmpQN2[17]*tmpFx[13];
tmpQN1[32] = + tmpQN2[15]*tmpFx[2] + tmpQN2[16]*tmpFx[8] + tmpQN2[17]*tmpFx[14];
tmpQN1[33] = + tmpQN2[15]*tmpFx[3] + tmpQN2[16]*tmpFx[9] + tmpQN2[17]*tmpFx[15];
tmpQN1[34] = + tmpQN2[15]*tmpFx[4] + tmpQN2[16]*tmpFx[10] + tmpQN2[17]*tmpFx[16];
tmpQN1[35] = + tmpQN2[15]*tmpFx[5] + tmpQN2[16]*tmpFx[11] + tmpQN2[17]*tmpFx[17];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 20; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 6];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 6 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 6 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 6 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.x[runObj * 6 + 4];
acadoWorkspace.objValueIn[5] = acadoVariables.x[runObj * 6 + 5];
acadoWorkspace.objValueIn[6] = acadoVariables.u[runObj];
acadoWorkspace.objValueIn[7] = acadoVariables.od[runObj * 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 4] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 4 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 4 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 4 + 3] = acadoWorkspace.objValueOut[3];

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 4 ]), &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.Q1[ runObj * 36 ]), &(acadoWorkspace.Q2[ runObj * 24 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 28 ]), &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.R1[ runObj ]), &(acadoWorkspace.R2[ runObj * 4 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[120];
acadoWorkspace.objValueIn[1] = acadoVariables.x[121];
acadoWorkspace.objValueIn[2] = acadoVariables.x[122];
acadoWorkspace.objValueIn[3] = acadoVariables.x[123];
acadoWorkspace.objValueIn[4] = acadoVariables.x[124];
acadoWorkspace.objValueIn[5] = acadoVariables.x[125];
acadoWorkspace.objValueIn[6] = acadoVariables.od[40];
acadoWorkspace.objValueIn[7] = acadoVariables.od[41];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 3 ]), acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_multGxd( real_t* const dOld, real_t* const Gx1, real_t* const dNew )
{
dNew[0] += + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3] + Gx1[4]*dOld[4] + Gx1[5]*dOld[5];
dNew[1] += + Gx1[6]*dOld[0] + Gx1[7]*dOld[1] + Gx1[8]*dOld[2] + Gx1[9]*dOld[3] + Gx1[10]*dOld[4] + Gx1[11]*dOld[5];
dNew[2] += + Gx1[12]*dOld[0] + Gx1[13]*dOld[1] + Gx1[14]*dOld[2] + Gx1[15]*dOld[3] + Gx1[16]*dOld[4] + Gx1[17]*dOld[5];
dNew[3] += + Gx1[18]*dOld[0] + Gx1[19]*dOld[1] + Gx1[20]*dOld[2] + Gx1[21]*dOld[3] + Gx1[22]*dOld[4] + Gx1[23]*dOld[5];
dNew[4] += + Gx1[24]*dOld[0] + Gx1[25]*dOld[1] + Gx1[26]*dOld[2] + Gx1[27]*dOld[3] + Gx1[28]*dOld[4] + Gx1[29]*dOld[5];
dNew[5] += + Gx1[30]*dOld[0] + Gx1[31]*dOld[1] + Gx1[32]*dOld[2] + Gx1[33]*dOld[3] + Gx1[34]*dOld[4] + Gx1[35]*dOld[5];
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
Gx2[25] = Gx1[25];
Gx2[26] = Gx1[26];
Gx2[27] = Gx1[27];
Gx2[28] = Gx1[28];
Gx2[29] = Gx1[29];
Gx2[30] = Gx1[30];
Gx2[31] = Gx1[31];
Gx2[32] = Gx1[32];
Gx2[33] = Gx1[33];
Gx2[34] = Gx1[34];
Gx2[35] = Gx1[35];
}

void acado_multGxGx( real_t* const Gx1, real_t* const Gx2, real_t* const Gx3 )
{
Gx3[0] = + Gx1[0]*Gx2[0] + Gx1[1]*Gx2[6] + Gx1[2]*Gx2[12] + Gx1[3]*Gx2[18] + Gx1[4]*Gx2[24] + Gx1[5]*Gx2[30];
Gx3[1] = + Gx1[0]*Gx2[1] + Gx1[1]*Gx2[7] + Gx1[2]*Gx2[13] + Gx1[3]*Gx2[19] + Gx1[4]*Gx2[25] + Gx1[5]*Gx2[31];
Gx3[2] = + Gx1[0]*Gx2[2] + Gx1[1]*Gx2[8] + Gx1[2]*Gx2[14] + Gx1[3]*Gx2[20] + Gx1[4]*Gx2[26] + Gx1[5]*Gx2[32];
Gx3[3] = + Gx1[0]*Gx2[3] + Gx1[1]*Gx2[9] + Gx1[2]*Gx2[15] + Gx1[3]*Gx2[21] + Gx1[4]*Gx2[27] + Gx1[5]*Gx2[33];
Gx3[4] = + Gx1[0]*Gx2[4] + Gx1[1]*Gx2[10] + Gx1[2]*Gx2[16] + Gx1[3]*Gx2[22] + Gx1[4]*Gx2[28] + Gx1[5]*Gx2[34];
Gx3[5] = + Gx1[0]*Gx2[5] + Gx1[1]*Gx2[11] + Gx1[2]*Gx2[17] + Gx1[3]*Gx2[23] + Gx1[4]*Gx2[29] + Gx1[5]*Gx2[35];
Gx3[6] = + Gx1[6]*Gx2[0] + Gx1[7]*Gx2[6] + Gx1[8]*Gx2[12] + Gx1[9]*Gx2[18] + Gx1[10]*Gx2[24] + Gx1[11]*Gx2[30];
Gx3[7] = + Gx1[6]*Gx2[1] + Gx1[7]*Gx2[7] + Gx1[8]*Gx2[13] + Gx1[9]*Gx2[19] + Gx1[10]*Gx2[25] + Gx1[11]*Gx2[31];
Gx3[8] = + Gx1[6]*Gx2[2] + Gx1[7]*Gx2[8] + Gx1[8]*Gx2[14] + Gx1[9]*Gx2[20] + Gx1[10]*Gx2[26] + Gx1[11]*Gx2[32];
Gx3[9] = + Gx1[6]*Gx2[3] + Gx1[7]*Gx2[9] + Gx1[8]*Gx2[15] + Gx1[9]*Gx2[21] + Gx1[10]*Gx2[27] + Gx1[11]*Gx2[33];
Gx3[10] = + Gx1[6]*Gx2[4] + Gx1[7]*Gx2[10] + Gx1[8]*Gx2[16] + Gx1[9]*Gx2[22] + Gx1[10]*Gx2[28] + Gx1[11]*Gx2[34];
Gx3[11] = + Gx1[6]*Gx2[5] + Gx1[7]*Gx2[11] + Gx1[8]*Gx2[17] + Gx1[9]*Gx2[23] + Gx1[10]*Gx2[29] + Gx1[11]*Gx2[35];
Gx3[12] = + Gx1[12]*Gx2[0] + Gx1[13]*Gx2[6] + Gx1[14]*Gx2[12] + Gx1[15]*Gx2[18] + Gx1[16]*Gx2[24] + Gx1[17]*Gx2[30];
Gx3[13] = + Gx1[12]*Gx2[1] + Gx1[13]*Gx2[7] + Gx1[14]*Gx2[13] + Gx1[15]*Gx2[19] + Gx1[16]*Gx2[25] + Gx1[17]*Gx2[31];
Gx3[14] = + Gx1[12]*Gx2[2] + Gx1[13]*Gx2[8] + Gx1[14]*Gx2[14] + Gx1[15]*Gx2[20] + Gx1[16]*Gx2[26] + Gx1[17]*Gx2[32];
Gx3[15] = + Gx1[12]*Gx2[3] + Gx1[13]*Gx2[9] + Gx1[14]*Gx2[15] + Gx1[15]*Gx2[21] + Gx1[16]*Gx2[27] + Gx1[17]*Gx2[33];
Gx3[16] = + Gx1[12]*Gx2[4] + Gx1[13]*Gx2[10] + Gx1[14]*Gx2[16] + Gx1[15]*Gx2[22] + Gx1[16]*Gx2[28] + Gx1[17]*Gx2[34];
Gx3[17] = + Gx1[12]*Gx2[5] + Gx1[13]*Gx2[11] + Gx1[14]*Gx2[17] + Gx1[15]*Gx2[23] + Gx1[16]*Gx2[29] + Gx1[17]*Gx2[35];
Gx3[18] = + Gx1[18]*Gx2[0] + Gx1[19]*Gx2[6] + Gx1[20]*Gx2[12] + Gx1[21]*Gx2[18] + Gx1[22]*Gx2[24] + Gx1[23]*Gx2[30];
Gx3[19] = + Gx1[18]*Gx2[1] + Gx1[19]*Gx2[7] + Gx1[20]*Gx2[13] + Gx1[21]*Gx2[19] + Gx1[22]*Gx2[25] + Gx1[23]*Gx2[31];
Gx3[20] = + Gx1[18]*Gx2[2] + Gx1[19]*Gx2[8] + Gx1[20]*Gx2[14] + Gx1[21]*Gx2[20] + Gx1[22]*Gx2[26] + Gx1[23]*Gx2[32];
Gx3[21] = + Gx1[18]*Gx2[3] + Gx1[19]*Gx2[9] + Gx1[20]*Gx2[15] + Gx1[21]*Gx2[21] + Gx1[22]*Gx2[27] + Gx1[23]*Gx2[33];
Gx3[22] = + Gx1[18]*Gx2[4] + Gx1[19]*Gx2[10] + Gx1[20]*Gx2[16] + Gx1[21]*Gx2[22] + Gx1[22]*Gx2[28] + Gx1[23]*Gx2[34];
Gx3[23] = + Gx1[18]*Gx2[5] + Gx1[19]*Gx2[11] + Gx1[20]*Gx2[17] + Gx1[21]*Gx2[23] + Gx1[22]*Gx2[29] + Gx1[23]*Gx2[35];
Gx3[24] = + Gx1[24]*Gx2[0] + Gx1[25]*Gx2[6] + Gx1[26]*Gx2[12] + Gx1[27]*Gx2[18] + Gx1[28]*Gx2[24] + Gx1[29]*Gx2[30];
Gx3[25] = + Gx1[24]*Gx2[1] + Gx1[25]*Gx2[7] + Gx1[26]*Gx2[13] + Gx1[27]*Gx2[19] + Gx1[28]*Gx2[25] + Gx1[29]*Gx2[31];
Gx3[26] = + Gx1[24]*Gx2[2] + Gx1[25]*Gx2[8] + Gx1[26]*Gx2[14] + Gx1[27]*Gx2[20] + Gx1[28]*Gx2[26] + Gx1[29]*Gx2[32];
Gx3[27] = + Gx1[24]*Gx2[3] + Gx1[25]*Gx2[9] + Gx1[26]*Gx2[15] + Gx1[27]*Gx2[21] + Gx1[28]*Gx2[27] + Gx1[29]*Gx2[33];
Gx3[28] = + Gx1[24]*Gx2[4] + Gx1[25]*Gx2[10] + Gx1[26]*Gx2[16] + Gx1[27]*Gx2[22] + Gx1[28]*Gx2[28] + Gx1[29]*Gx2[34];
Gx3[29] = + Gx1[24]*Gx2[5] + Gx1[25]*Gx2[11] + Gx1[26]*Gx2[17] + Gx1[27]*Gx2[23] + Gx1[28]*Gx2[29] + Gx1[29]*Gx2[35];
Gx3[30] = + Gx1[30]*Gx2[0] + Gx1[31]*Gx2[6] + Gx1[32]*Gx2[12] + Gx1[33]*Gx2[18] + Gx1[34]*Gx2[24] + Gx1[35]*Gx2[30];
Gx3[31] = + Gx1[30]*Gx2[1] + Gx1[31]*Gx2[7] + Gx1[32]*Gx2[13] + Gx1[33]*Gx2[19] + Gx1[34]*Gx2[25] + Gx1[35]*Gx2[31];
Gx3[32] = + Gx1[30]*Gx2[2] + Gx1[31]*Gx2[8] + Gx1[32]*Gx2[14] + Gx1[33]*Gx2[20] + Gx1[34]*Gx2[26] + Gx1[35]*Gx2[32];
Gx3[33] = + Gx1[30]*Gx2[3] + Gx1[31]*Gx2[9] + Gx1[32]*Gx2[15] + Gx1[33]*Gx2[21] + Gx1[34]*Gx2[27] + Gx1[35]*Gx2[33];
Gx3[34] = + Gx1[30]*Gx2[4] + Gx1[31]*Gx2[10] + Gx1[32]*Gx2[16] + Gx1[33]*Gx2[22] + Gx1[34]*Gx2[28] + Gx1[35]*Gx2[34];
Gx3[35] = + Gx1[30]*Gx2[5] + Gx1[31]*Gx2[11] + Gx1[32]*Gx2[17] + Gx1[33]*Gx2[23] + Gx1[34]*Gx2[29] + Gx1[35]*Gx2[35];
}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[1] + Gx1[2]*Gu1[2] + Gx1[3]*Gu1[3] + Gx1[4]*Gu1[4] + Gx1[5]*Gu1[5];
Gu2[1] = + Gx1[6]*Gu1[0] + Gx1[7]*Gu1[1] + Gx1[8]*Gu1[2] + Gx1[9]*Gu1[3] + Gx1[10]*Gu1[4] + Gx1[11]*Gu1[5];
Gu2[2] = + Gx1[12]*Gu1[0] + Gx1[13]*Gu1[1] + Gx1[14]*Gu1[2] + Gx1[15]*Gu1[3] + Gx1[16]*Gu1[4] + Gx1[17]*Gu1[5];
Gu2[3] = + Gx1[18]*Gu1[0] + Gx1[19]*Gu1[1] + Gx1[20]*Gu1[2] + Gx1[21]*Gu1[3] + Gx1[22]*Gu1[4] + Gx1[23]*Gu1[5];
Gu2[4] = + Gx1[24]*Gu1[0] + Gx1[25]*Gu1[1] + Gx1[26]*Gu1[2] + Gx1[27]*Gu1[3] + Gx1[28]*Gu1[4] + Gx1[29]*Gu1[5];
Gu2[5] = + Gx1[30]*Gu1[0] + Gx1[31]*Gu1[1] + Gx1[32]*Gu1[2] + Gx1[33]*Gu1[3] + Gx1[34]*Gu1[4] + Gx1[35]*Gu1[5];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
Gu2[3] = Gu1[3];
Gu2[4] = Gu1[4];
Gu2[5] = Gu1[5];
}

void acado_setBlockH11( int iRow, int iCol, real_t* const Gu1, real_t* const Gu2 )
{
acadoWorkspace.H[(iRow * 26 + 156) + (iCol + 6)] += + Gu1[0]*Gu2[0] + Gu1[1]*Gu2[1] + Gu1[2]*Gu2[2] + Gu1[3]*Gu2[3] + Gu1[4]*Gu2[4] + Gu1[5]*Gu2[5];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 26 + 156) + (iCol + 6)] = R11[0];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 26 + 156) + (iCol + 6)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 26 + 156) + (iCol + 6)] = acadoWorkspace.H[(iCol * 26 + 156) + (iRow + 6)];
}

void acado_multQ1d( real_t* const Gx1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + Gx1[0]*dOld[0] + Gx1[1]*dOld[1] + Gx1[2]*dOld[2] + Gx1[3]*dOld[3] + Gx1[4]*dOld[4] + Gx1[5]*dOld[5];
dNew[1] = + Gx1[6]*dOld[0] + Gx1[7]*dOld[1] + Gx1[8]*dOld[2] + Gx1[9]*dOld[3] + Gx1[10]*dOld[4] + Gx1[11]*dOld[5];
dNew[2] = + Gx1[12]*dOld[0] + Gx1[13]*dOld[1] + Gx1[14]*dOld[2] + Gx1[15]*dOld[3] + Gx1[16]*dOld[4] + Gx1[17]*dOld[5];
dNew[3] = + Gx1[18]*dOld[0] + Gx1[19]*dOld[1] + Gx1[20]*dOld[2] + Gx1[21]*dOld[3] + Gx1[22]*dOld[4] + Gx1[23]*dOld[5];
dNew[4] = + Gx1[24]*dOld[0] + Gx1[25]*dOld[1] + Gx1[26]*dOld[2] + Gx1[27]*dOld[3] + Gx1[28]*dOld[4] + Gx1[29]*dOld[5];
dNew[5] = + Gx1[30]*dOld[0] + Gx1[31]*dOld[1] + Gx1[32]*dOld[2] + Gx1[33]*dOld[3] + Gx1[34]*dOld[4] + Gx1[35]*dOld[5];
}

void acado_multQN1d( real_t* const QN1, real_t* const dOld, real_t* const dNew )
{
dNew[0] = + acadoWorkspace.QN1[0]*dOld[0] + acadoWorkspace.QN1[1]*dOld[1] + acadoWorkspace.QN1[2]*dOld[2] + acadoWorkspace.QN1[3]*dOld[3] + acadoWorkspace.QN1[4]*dOld[4] + acadoWorkspace.QN1[5]*dOld[5];
dNew[1] = + acadoWorkspace.QN1[6]*dOld[0] + acadoWorkspace.QN1[7]*dOld[1] + acadoWorkspace.QN1[8]*dOld[2] + acadoWorkspace.QN1[9]*dOld[3] + acadoWorkspace.QN1[10]*dOld[4] + acadoWorkspace.QN1[11]*dOld[5];
dNew[2] = + acadoWorkspace.QN1[12]*dOld[0] + acadoWorkspace.QN1[13]*dOld[1] + acadoWorkspace.QN1[14]*dOld[2] + acadoWorkspace.QN1[15]*dOld[3] + acadoWorkspace.QN1[16]*dOld[4] + acadoWorkspace.QN1[17]*dOld[5];
dNew[3] = + acadoWorkspace.QN1[18]*dOld[0] + acadoWorkspace.QN1[19]*dOld[1] + acadoWorkspace.QN1[20]*dOld[2] + acadoWorkspace.QN1[21]*dOld[3] + acadoWorkspace.QN1[22]*dOld[4] + acadoWorkspace.QN1[23]*dOld[5];
dNew[4] = + acadoWorkspace.QN1[24]*dOld[0] + acadoWorkspace.QN1[25]*dOld[1] + acadoWorkspace.QN1[26]*dOld[2] + acadoWorkspace.QN1[27]*dOld[3] + acadoWorkspace.QN1[28]*dOld[4] + acadoWorkspace.QN1[29]*dOld[5];
dNew[5] = + acadoWorkspace.QN1[30]*dOld[0] + acadoWorkspace.QN1[31]*dOld[1] + acadoWorkspace.QN1[32]*dOld[2] + acadoWorkspace.QN1[33]*dOld[3] + acadoWorkspace.QN1[34]*dOld[4] + acadoWorkspace.QN1[35]*dOld[5];
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
QDy1[3] = + Q2[12]*Dy1[0] + Q2[13]*Dy1[1] + Q2[14]*Dy1[2] + Q2[15]*Dy1[3];
QDy1[4] = + Q2[16]*Dy1[0] + Q2[17]*Dy1[1] + Q2[18]*Dy1[2] + Q2[19]*Dy1[3];
QDy1[5] = + Q2[20]*Dy1[0] + Q2[21]*Dy1[1] + Q2[22]*Dy1[2] + Q2[23]*Dy1[3];
}

void acado_multEQDy( real_t* const E1, real_t* const QDy1, real_t* const U1 )
{
U1[0] += + E1[0]*QDy1[0] + E1[1]*QDy1[1] + E1[2]*QDy1[2] + E1[3]*QDy1[3] + E1[4]*QDy1[4] + E1[5]*QDy1[5];
}

void acado_multQETGx( real_t* const E1, real_t* const Gx1, real_t* const H101 )
{
H101[0] += + E1[0]*Gx1[0] + E1[1]*Gx1[6] + E1[2]*Gx1[12] + E1[3]*Gx1[18] + E1[4]*Gx1[24] + E1[5]*Gx1[30];
H101[1] += + E1[0]*Gx1[1] + E1[1]*Gx1[7] + E1[2]*Gx1[13] + E1[3]*Gx1[19] + E1[4]*Gx1[25] + E1[5]*Gx1[31];
H101[2] += + E1[0]*Gx1[2] + E1[1]*Gx1[8] + E1[2]*Gx1[14] + E1[3]*Gx1[20] + E1[4]*Gx1[26] + E1[5]*Gx1[32];
H101[3] += + E1[0]*Gx1[3] + E1[1]*Gx1[9] + E1[2]*Gx1[15] + E1[3]*Gx1[21] + E1[4]*Gx1[27] + E1[5]*Gx1[33];
H101[4] += + E1[0]*Gx1[4] + E1[1]*Gx1[10] + E1[2]*Gx1[16] + E1[3]*Gx1[22] + E1[4]*Gx1[28] + E1[5]*Gx1[34];
H101[5] += + E1[0]*Gx1[5] + E1[1]*Gx1[11] + E1[2]*Gx1[17] + E1[3]*Gx1[23] + E1[4]*Gx1[29] + E1[5]*Gx1[35];
}

void acado_zeroBlockH10( real_t* const H101 )
{
{ int lCopy; for (lCopy = 0; lCopy < 6; lCopy++) H101[ lCopy ] = 0; }
}

void acado_multEDu( real_t* const E1, real_t* const U1, real_t* const dNew )
{
dNew[0] += + E1[0]*U1[0];
dNew[1] += + E1[1]*U1[0];
dNew[2] += + E1[2]*U1[0];
dNew[3] += + E1[3]*U1[0];
dNew[4] += + E1[4]*U1[0];
dNew[5] += + E1[5]*U1[0];
}

void acado_zeroBlockH00(  )
{
acadoWorkspace.H[0] = 0.0000000000000000e+00;
acadoWorkspace.H[1] = 0.0000000000000000e+00;
acadoWorkspace.H[2] = 0.0000000000000000e+00;
acadoWorkspace.H[3] = 0.0000000000000000e+00;
acadoWorkspace.H[4] = 0.0000000000000000e+00;
acadoWorkspace.H[5] = 0.0000000000000000e+00;
acadoWorkspace.H[26] = 0.0000000000000000e+00;
acadoWorkspace.H[27] = 0.0000000000000000e+00;
acadoWorkspace.H[28] = 0.0000000000000000e+00;
acadoWorkspace.H[29] = 0.0000000000000000e+00;
acadoWorkspace.H[30] = 0.0000000000000000e+00;
acadoWorkspace.H[31] = 0.0000000000000000e+00;
acadoWorkspace.H[52] = 0.0000000000000000e+00;
acadoWorkspace.H[53] = 0.0000000000000000e+00;
acadoWorkspace.H[54] = 0.0000000000000000e+00;
acadoWorkspace.H[55] = 0.0000000000000000e+00;
acadoWorkspace.H[56] = 0.0000000000000000e+00;
acadoWorkspace.H[57] = 0.0000000000000000e+00;
acadoWorkspace.H[78] = 0.0000000000000000e+00;
acadoWorkspace.H[79] = 0.0000000000000000e+00;
acadoWorkspace.H[80] = 0.0000000000000000e+00;
acadoWorkspace.H[81] = 0.0000000000000000e+00;
acadoWorkspace.H[82] = 0.0000000000000000e+00;
acadoWorkspace.H[83] = 0.0000000000000000e+00;
acadoWorkspace.H[104] = 0.0000000000000000e+00;
acadoWorkspace.H[105] = 0.0000000000000000e+00;
acadoWorkspace.H[106] = 0.0000000000000000e+00;
acadoWorkspace.H[107] = 0.0000000000000000e+00;
acadoWorkspace.H[108] = 0.0000000000000000e+00;
acadoWorkspace.H[109] = 0.0000000000000000e+00;
acadoWorkspace.H[130] = 0.0000000000000000e+00;
acadoWorkspace.H[131] = 0.0000000000000000e+00;
acadoWorkspace.H[132] = 0.0000000000000000e+00;
acadoWorkspace.H[133] = 0.0000000000000000e+00;
acadoWorkspace.H[134] = 0.0000000000000000e+00;
acadoWorkspace.H[135] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[6]*Gx2[6] + Gx1[12]*Gx2[12] + Gx1[18]*Gx2[18] + Gx1[24]*Gx2[24] + Gx1[30]*Gx2[30];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[6]*Gx2[7] + Gx1[12]*Gx2[13] + Gx1[18]*Gx2[19] + Gx1[24]*Gx2[25] + Gx1[30]*Gx2[31];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[6]*Gx2[8] + Gx1[12]*Gx2[14] + Gx1[18]*Gx2[20] + Gx1[24]*Gx2[26] + Gx1[30]*Gx2[32];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[6]*Gx2[9] + Gx1[12]*Gx2[15] + Gx1[18]*Gx2[21] + Gx1[24]*Gx2[27] + Gx1[30]*Gx2[33];
acadoWorkspace.H[4] += + Gx1[0]*Gx2[4] + Gx1[6]*Gx2[10] + Gx1[12]*Gx2[16] + Gx1[18]*Gx2[22] + Gx1[24]*Gx2[28] + Gx1[30]*Gx2[34];
acadoWorkspace.H[5] += + Gx1[0]*Gx2[5] + Gx1[6]*Gx2[11] + Gx1[12]*Gx2[17] + Gx1[18]*Gx2[23] + Gx1[24]*Gx2[29] + Gx1[30]*Gx2[35];
acadoWorkspace.H[26] += + Gx1[1]*Gx2[0] + Gx1[7]*Gx2[6] + Gx1[13]*Gx2[12] + Gx1[19]*Gx2[18] + Gx1[25]*Gx2[24] + Gx1[31]*Gx2[30];
acadoWorkspace.H[27] += + Gx1[1]*Gx2[1] + Gx1[7]*Gx2[7] + Gx1[13]*Gx2[13] + Gx1[19]*Gx2[19] + Gx1[25]*Gx2[25] + Gx1[31]*Gx2[31];
acadoWorkspace.H[28] += + Gx1[1]*Gx2[2] + Gx1[7]*Gx2[8] + Gx1[13]*Gx2[14] + Gx1[19]*Gx2[20] + Gx1[25]*Gx2[26] + Gx1[31]*Gx2[32];
acadoWorkspace.H[29] += + Gx1[1]*Gx2[3] + Gx1[7]*Gx2[9] + Gx1[13]*Gx2[15] + Gx1[19]*Gx2[21] + Gx1[25]*Gx2[27] + Gx1[31]*Gx2[33];
acadoWorkspace.H[30] += + Gx1[1]*Gx2[4] + Gx1[7]*Gx2[10] + Gx1[13]*Gx2[16] + Gx1[19]*Gx2[22] + Gx1[25]*Gx2[28] + Gx1[31]*Gx2[34];
acadoWorkspace.H[31] += + Gx1[1]*Gx2[5] + Gx1[7]*Gx2[11] + Gx1[13]*Gx2[17] + Gx1[19]*Gx2[23] + Gx1[25]*Gx2[29] + Gx1[31]*Gx2[35];
acadoWorkspace.H[52] += + Gx1[2]*Gx2[0] + Gx1[8]*Gx2[6] + Gx1[14]*Gx2[12] + Gx1[20]*Gx2[18] + Gx1[26]*Gx2[24] + Gx1[32]*Gx2[30];
acadoWorkspace.H[53] += + Gx1[2]*Gx2[1] + Gx1[8]*Gx2[7] + Gx1[14]*Gx2[13] + Gx1[20]*Gx2[19] + Gx1[26]*Gx2[25] + Gx1[32]*Gx2[31];
acadoWorkspace.H[54] += + Gx1[2]*Gx2[2] + Gx1[8]*Gx2[8] + Gx1[14]*Gx2[14] + Gx1[20]*Gx2[20] + Gx1[26]*Gx2[26] + Gx1[32]*Gx2[32];
acadoWorkspace.H[55] += + Gx1[2]*Gx2[3] + Gx1[8]*Gx2[9] + Gx1[14]*Gx2[15] + Gx1[20]*Gx2[21] + Gx1[26]*Gx2[27] + Gx1[32]*Gx2[33];
acadoWorkspace.H[56] += + Gx1[2]*Gx2[4] + Gx1[8]*Gx2[10] + Gx1[14]*Gx2[16] + Gx1[20]*Gx2[22] + Gx1[26]*Gx2[28] + Gx1[32]*Gx2[34];
acadoWorkspace.H[57] += + Gx1[2]*Gx2[5] + Gx1[8]*Gx2[11] + Gx1[14]*Gx2[17] + Gx1[20]*Gx2[23] + Gx1[26]*Gx2[29] + Gx1[32]*Gx2[35];
acadoWorkspace.H[78] += + Gx1[3]*Gx2[0] + Gx1[9]*Gx2[6] + Gx1[15]*Gx2[12] + Gx1[21]*Gx2[18] + Gx1[27]*Gx2[24] + Gx1[33]*Gx2[30];
acadoWorkspace.H[79] += + Gx1[3]*Gx2[1] + Gx1[9]*Gx2[7] + Gx1[15]*Gx2[13] + Gx1[21]*Gx2[19] + Gx1[27]*Gx2[25] + Gx1[33]*Gx2[31];
acadoWorkspace.H[80] += + Gx1[3]*Gx2[2] + Gx1[9]*Gx2[8] + Gx1[15]*Gx2[14] + Gx1[21]*Gx2[20] + Gx1[27]*Gx2[26] + Gx1[33]*Gx2[32];
acadoWorkspace.H[81] += + Gx1[3]*Gx2[3] + Gx1[9]*Gx2[9] + Gx1[15]*Gx2[15] + Gx1[21]*Gx2[21] + Gx1[27]*Gx2[27] + Gx1[33]*Gx2[33];
acadoWorkspace.H[82] += + Gx1[3]*Gx2[4] + Gx1[9]*Gx2[10] + Gx1[15]*Gx2[16] + Gx1[21]*Gx2[22] + Gx1[27]*Gx2[28] + Gx1[33]*Gx2[34];
acadoWorkspace.H[83] += + Gx1[3]*Gx2[5] + Gx1[9]*Gx2[11] + Gx1[15]*Gx2[17] + Gx1[21]*Gx2[23] + Gx1[27]*Gx2[29] + Gx1[33]*Gx2[35];
acadoWorkspace.H[104] += + Gx1[4]*Gx2[0] + Gx1[10]*Gx2[6] + Gx1[16]*Gx2[12] + Gx1[22]*Gx2[18] + Gx1[28]*Gx2[24] + Gx1[34]*Gx2[30];
acadoWorkspace.H[105] += + Gx1[4]*Gx2[1] + Gx1[10]*Gx2[7] + Gx1[16]*Gx2[13] + Gx1[22]*Gx2[19] + Gx1[28]*Gx2[25] + Gx1[34]*Gx2[31];
acadoWorkspace.H[106] += + Gx1[4]*Gx2[2] + Gx1[10]*Gx2[8] + Gx1[16]*Gx2[14] + Gx1[22]*Gx2[20] + Gx1[28]*Gx2[26] + Gx1[34]*Gx2[32];
acadoWorkspace.H[107] += + Gx1[4]*Gx2[3] + Gx1[10]*Gx2[9] + Gx1[16]*Gx2[15] + Gx1[22]*Gx2[21] + Gx1[28]*Gx2[27] + Gx1[34]*Gx2[33];
acadoWorkspace.H[108] += + Gx1[4]*Gx2[4] + Gx1[10]*Gx2[10] + Gx1[16]*Gx2[16] + Gx1[22]*Gx2[22] + Gx1[28]*Gx2[28] + Gx1[34]*Gx2[34];
acadoWorkspace.H[109] += + Gx1[4]*Gx2[5] + Gx1[10]*Gx2[11] + Gx1[16]*Gx2[17] + Gx1[22]*Gx2[23] + Gx1[28]*Gx2[29] + Gx1[34]*Gx2[35];
acadoWorkspace.H[130] += + Gx1[5]*Gx2[0] + Gx1[11]*Gx2[6] + Gx1[17]*Gx2[12] + Gx1[23]*Gx2[18] + Gx1[29]*Gx2[24] + Gx1[35]*Gx2[30];
acadoWorkspace.H[131] += + Gx1[5]*Gx2[1] + Gx1[11]*Gx2[7] + Gx1[17]*Gx2[13] + Gx1[23]*Gx2[19] + Gx1[29]*Gx2[25] + Gx1[35]*Gx2[31];
acadoWorkspace.H[132] += + Gx1[5]*Gx2[2] + Gx1[11]*Gx2[8] + Gx1[17]*Gx2[14] + Gx1[23]*Gx2[20] + Gx1[29]*Gx2[26] + Gx1[35]*Gx2[32];
acadoWorkspace.H[133] += + Gx1[5]*Gx2[3] + Gx1[11]*Gx2[9] + Gx1[17]*Gx2[15] + Gx1[23]*Gx2[21] + Gx1[29]*Gx2[27] + Gx1[35]*Gx2[33];
acadoWorkspace.H[134] += + Gx1[5]*Gx2[4] + Gx1[11]*Gx2[10] + Gx1[17]*Gx2[16] + Gx1[23]*Gx2[22] + Gx1[29]*Gx2[28] + Gx1[35]*Gx2[34];
acadoWorkspace.H[135] += + Gx1[5]*Gx2[5] + Gx1[11]*Gx2[11] + Gx1[17]*Gx2[17] + Gx1[23]*Gx2[23] + Gx1[29]*Gx2[29] + Gx1[35]*Gx2[35];
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
g0[5] += 0.0;
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
{ 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115, 121 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_moveGxT( &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.T );
acado_multGxd( acadoWorkspace.d, &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.d[ 6 ]) );
acado_multGxGx( acadoWorkspace.T, acadoWorkspace.evGx, &(acadoWorkspace.evGx[ 36 ]) );

acado_multGxGu( acadoWorkspace.T, acadoWorkspace.E, &(acadoWorkspace.E[ 6 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 6 ]), &(acadoWorkspace.E[ 12 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 6 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.d[ 12 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.evGx[ 72 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.E[ 18 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.E[ 24 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 12 ]), &(acadoWorkspace.E[ 30 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.d[ 18 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.evGx[ 108 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.E[ 36 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.E[ 42 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.E[ 48 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 18 ]), &(acadoWorkspace.E[ 54 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 18 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.d[ 24 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.evGx[ 144 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.E[ 60 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.E[ 66 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.E[ 72 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.E[ 78 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.E[ 84 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 180 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.d[ 30 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.evGx[ 180 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.E[ 90 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.E[ 96 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.E[ 102 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.E[ 108 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.E[ 114 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 30 ]), &(acadoWorkspace.E[ 120 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 216 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.d[ 36 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.evGx[ 216 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.E[ 126 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.E[ 132 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.E[ 138 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.E[ 144 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.E[ 150 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.E[ 156 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 36 ]), &(acadoWorkspace.E[ 162 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 252 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.d[ 42 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.evGx[ 252 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.E[ 168 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.E[ 174 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.E[ 180 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.E[ 186 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.E[ 192 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.E[ 198 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.E[ 204 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 42 ]), &(acadoWorkspace.E[ 210 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 42 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.d[ 48 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.evGx[ 288 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.E[ 216 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.E[ 222 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.E[ 228 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.E[ 234 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.E[ 240 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.E[ 246 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.E[ 252 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.E[ 258 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 48 ]), &(acadoWorkspace.E[ 264 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 324 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.d[ 54 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.evGx[ 324 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.E[ 270 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.E[ 276 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.E[ 282 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.E[ 288 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.E[ 294 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.E[ 300 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.E[ 306 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.E[ 312 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.E[ 318 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 54 ]), &(acadoWorkspace.E[ 324 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 360 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 54 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.d[ 60 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.evGx[ 360 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.E[ 330 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.E[ 336 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.E[ 342 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.E[ 348 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.E[ 354 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.E[ 360 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.E[ 366 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.E[ 372 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.E[ 378 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.E[ 384 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 60 ]), &(acadoWorkspace.E[ 390 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 396 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.d[ 66 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.evGx[ 396 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.E[ 396 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.E[ 402 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.E[ 408 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.E[ 414 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.E[ 420 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.E[ 426 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.E[ 432 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.E[ 438 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.E[ 444 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.E[ 450 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.E[ 456 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 66 ]), &(acadoWorkspace.E[ 462 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 432 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 66 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.d[ 72 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.evGx[ 432 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.E[ 468 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.E[ 474 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.E[ 480 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.E[ 486 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.E[ 492 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.E[ 498 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.E[ 504 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.E[ 510 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.E[ 516 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.E[ 522 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.E[ 528 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.E[ 534 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 72 ]), &(acadoWorkspace.E[ 540 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 468 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.d[ 78 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.evGx[ 468 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.E[ 546 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.E[ 552 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.E[ 558 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.E[ 564 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.E[ 570 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.E[ 576 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.E[ 582 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.E[ 588 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.E[ 594 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.E[ 600 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.E[ 606 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.E[ 612 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.E[ 618 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 78 ]), &(acadoWorkspace.E[ 624 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 504 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 78 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.d[ 84 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.evGx[ 504 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.E[ 630 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.E[ 636 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.E[ 642 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.E[ 648 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.E[ 654 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.E[ 660 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.E[ 666 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.E[ 672 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.E[ 678 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.E[ 684 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.E[ 690 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.E[ 696 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.E[ 702 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.E[ 708 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 84 ]), &(acadoWorkspace.E[ 714 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 540 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.d[ 90 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.evGx[ 540 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.E[ 720 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.E[ 726 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.E[ 732 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.E[ 738 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.E[ 744 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.E[ 750 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.E[ 756 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.E[ 762 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.E[ 768 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.E[ 774 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.E[ 780 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.E[ 786 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.E[ 792 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.E[ 798 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 714 ]), &(acadoWorkspace.E[ 804 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 90 ]), &(acadoWorkspace.E[ 810 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 576 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.d[ 96 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.evGx[ 576 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.E[ 816 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.E[ 822 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.E[ 828 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.E[ 834 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.E[ 840 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.E[ 846 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.E[ 852 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.E[ 858 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.E[ 864 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.E[ 870 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.E[ 876 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.E[ 882 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.E[ 888 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.E[ 894 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.E[ 900 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.E[ 906 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 96 ]), &(acadoWorkspace.E[ 912 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 612 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 96 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.d[ 102 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.evGx[ 612 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.E[ 918 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.E[ 924 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.E[ 930 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.E[ 936 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.E[ 942 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.E[ 948 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.E[ 954 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.E[ 960 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.E[ 966 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.E[ 972 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.E[ 978 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.E[ 984 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.E[ 990 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.E[ 996 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.E[ 1002 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.E[ 1008 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.E[ 1014 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 102 ]), &(acadoWorkspace.E[ 1020 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 648 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 102 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.d[ 108 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.evGx[ 648 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.E[ 1026 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.E[ 1032 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.E[ 1038 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.E[ 1044 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.E[ 1050 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.E[ 1056 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.E[ 1062 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.E[ 1068 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.E[ 1074 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.E[ 1080 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.E[ 1086 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.E[ 1092 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.E[ 1098 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.E[ 1104 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.E[ 1110 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.E[ 1116 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.E[ 1122 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.E[ 1128 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 108 ]), &(acadoWorkspace.E[ 1134 ]) );

acado_moveGxT( &(acadoWorkspace.evGx[ 684 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ 108 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.d[ 114 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.evGx[ 684 ]) );

acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.E[ 1140 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.E[ 1146 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.E[ 1152 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.E[ 1158 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.E[ 1164 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.E[ 1170 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.E[ 1176 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.E[ 1182 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.E[ 1188 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.E[ 1194 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.E[ 1200 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.E[ 1206 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.E[ 1212 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.E[ 1218 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.E[ 1224 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.E[ 1230 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.E[ 1236 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.E[ 1242 ]) );
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ 1134 ]), &(acadoWorkspace.E[ 1248 ]) );

acado_moveGuE( &(acadoWorkspace.evGu[ 114 ]), &(acadoWorkspace.E[ 1254 ]) );

acado_multGxGx( &(acadoWorkspace.Q1[ 36 ]), acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multGxGx( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.QGx[ 36 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.QGx[ 72 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.QGx[ 108 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.QGx[ 180 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.QGx[ 216 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.QGx[ 252 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.QGx[ 288 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.QGx[ 324 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.QGx[ 360 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.QGx[ 396 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.QGx[ 432 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.QGx[ 468 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.QGx[ 504 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.QGx[ 540 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.QGx[ 576 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.QGx[ 612 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.QGx[ 648 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.QGx[ 684 ]) );

acado_multGxGu( &(acadoWorkspace.Q1[ 36 ]), acadoWorkspace.E, acadoWorkspace.QE );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QE[ 6 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 18 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 636 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 642 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.E[ 714 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 726 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 732 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 738 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 822 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 828 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 834 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 918 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 924 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1026 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1038 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1044 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_multGxGu( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.E[ 1134 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1140 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1146 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1152 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1158 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1164 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1170 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1176 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1182 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1188 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1194 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1206 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1212 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1218 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1224 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1230 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QE[ 1236 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.QE[ 1242 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1248 ]) );
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 1254 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_zeroBlockH00(  );
acado_multCTQC( acadoWorkspace.evGx, acadoWorkspace.QGx );
acado_multCTQC( &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.QGx[ 36 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.QGx[ 72 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.QGx[ 108 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.QGx[ 144 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.QGx[ 180 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.QGx[ 216 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.QGx[ 252 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.QGx[ 288 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.QGx[ 324 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.QGx[ 360 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.QGx[ 396 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.QGx[ 432 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.QGx[ 468 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.QGx[ 504 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.QGx[ 540 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.QGx[ 576 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.QGx[ 612 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.QGx[ 648 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.QGx[ 684 ]) );

acado_zeroBlockH10( acadoWorkspace.H10 );
acado_multQETGx( acadoWorkspace.QE, acadoWorkspace.evGx, acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 6 ]), &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 18 ]), &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.evGx[ 180 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 126 ]), &(acadoWorkspace.evGx[ 216 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.evGx[ 252 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.evGx[ 324 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.evGx[ 360 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.evGx[ 396 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.evGx[ 432 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 546 ]), &(acadoWorkspace.evGx[ 468 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 630 ]), &(acadoWorkspace.evGx[ 504 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.evGx[ 540 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 816 ]), &(acadoWorkspace.evGx[ 576 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 918 ]), &(acadoWorkspace.evGx[ 612 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1026 ]), &(acadoWorkspace.evGx[ 648 ]), acadoWorkspace.H10 );
acado_multQETGx( &(acadoWorkspace.QE[ 1140 ]), &(acadoWorkspace.evGx[ 684 ]), acadoWorkspace.H10 );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.evGx[ 36 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 42 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 66 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 174 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 222 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 402 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 474 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 636 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 726 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 822 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 924 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1032 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1146 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 6 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.evGx[ 72 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 102 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 138 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 282 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 342 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 558 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 642 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 732 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 828 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 930 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1038 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1152 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 12 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 54 ]), &(acadoWorkspace.evGx[ 108 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 78 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 186 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 234 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 414 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 486 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 564 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 648 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 738 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 834 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 936 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1044 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1158 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 18 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 114 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 294 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 354 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 654 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 744 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 942 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1050 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1164 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 24 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.evGx[ 180 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 198 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 246 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 426 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 498 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 660 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 750 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 846 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 948 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1056 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1170 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 30 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 162 ]), &(acadoWorkspace.evGx[ 216 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 306 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 366 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 582 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 666 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 756 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 852 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 954 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1062 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1176 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 36 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.evGx[ 252 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 258 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 438 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 588 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 672 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 762 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 858 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1068 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1182 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 42 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 318 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 378 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 594 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 678 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 768 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 864 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 966 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1074 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1188 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 48 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.evGx[ 324 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 522 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 684 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 774 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 870 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 972 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1194 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 54 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.evGx[ 360 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 606 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 690 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 780 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 876 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 978 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1086 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 60 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 462 ]), &(acadoWorkspace.evGx[ 396 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 534 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 612 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 696 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 786 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 882 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 984 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1092 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1206 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 66 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 618 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 702 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 792 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 888 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 990 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1098 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1212 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 72 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.evGx[ 468 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 708 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 798 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 894 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 996 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1104 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1218 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 78 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 714 ]), &(acadoWorkspace.evGx[ 504 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 804 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 900 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1002 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1110 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1224 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 84 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 810 ]), &(acadoWorkspace.evGx[ 540 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 906 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1008 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1116 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1230 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 90 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 912 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1014 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1122 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1236 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 96 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 102 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1020 ]), &(acadoWorkspace.evGx[ 612 ]), &(acadoWorkspace.H10[ 102 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1128 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 102 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1242 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 102 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 108 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1134 ]), &(acadoWorkspace.evGx[ 648 ]), &(acadoWorkspace.H10[ 108 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1248 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 108 ]) );
acado_zeroBlockH10( &(acadoWorkspace.H10[ 114 ]) );
acado_multQETGx( &(acadoWorkspace.QE[ 1254 ]), &(acadoWorkspace.evGx[ 684 ]), &(acadoWorkspace.H10[ 114 ]) );

acadoWorkspace.H[6] = acadoWorkspace.H10[0];
acadoWorkspace.H[7] = acadoWorkspace.H10[6];
acadoWorkspace.H[8] = acadoWorkspace.H10[12];
acadoWorkspace.H[9] = acadoWorkspace.H10[18];
acadoWorkspace.H[10] = acadoWorkspace.H10[24];
acadoWorkspace.H[11] = acadoWorkspace.H10[30];
acadoWorkspace.H[12] = acadoWorkspace.H10[36];
acadoWorkspace.H[13] = acadoWorkspace.H10[42];
acadoWorkspace.H[14] = acadoWorkspace.H10[48];
acadoWorkspace.H[15] = acadoWorkspace.H10[54];
acadoWorkspace.H[16] = acadoWorkspace.H10[60];
acadoWorkspace.H[17] = acadoWorkspace.H10[66];
acadoWorkspace.H[18] = acadoWorkspace.H10[72];
acadoWorkspace.H[19] = acadoWorkspace.H10[78];
acadoWorkspace.H[20] = acadoWorkspace.H10[84];
acadoWorkspace.H[21] = acadoWorkspace.H10[90];
acadoWorkspace.H[22] = acadoWorkspace.H10[96];
acadoWorkspace.H[23] = acadoWorkspace.H10[102];
acadoWorkspace.H[24] = acadoWorkspace.H10[108];
acadoWorkspace.H[25] = acadoWorkspace.H10[114];
acadoWorkspace.H[32] = acadoWorkspace.H10[1];
acadoWorkspace.H[33] = acadoWorkspace.H10[7];
acadoWorkspace.H[34] = acadoWorkspace.H10[13];
acadoWorkspace.H[35] = acadoWorkspace.H10[19];
acadoWorkspace.H[36] = acadoWorkspace.H10[25];
acadoWorkspace.H[37] = acadoWorkspace.H10[31];
acadoWorkspace.H[38] = acadoWorkspace.H10[37];
acadoWorkspace.H[39] = acadoWorkspace.H10[43];
acadoWorkspace.H[40] = acadoWorkspace.H10[49];
acadoWorkspace.H[41] = acadoWorkspace.H10[55];
acadoWorkspace.H[42] = acadoWorkspace.H10[61];
acadoWorkspace.H[43] = acadoWorkspace.H10[67];
acadoWorkspace.H[44] = acadoWorkspace.H10[73];
acadoWorkspace.H[45] = acadoWorkspace.H10[79];
acadoWorkspace.H[46] = acadoWorkspace.H10[85];
acadoWorkspace.H[47] = acadoWorkspace.H10[91];
acadoWorkspace.H[48] = acadoWorkspace.H10[97];
acadoWorkspace.H[49] = acadoWorkspace.H10[103];
acadoWorkspace.H[50] = acadoWorkspace.H10[109];
acadoWorkspace.H[51] = acadoWorkspace.H10[115];
acadoWorkspace.H[58] = acadoWorkspace.H10[2];
acadoWorkspace.H[59] = acadoWorkspace.H10[8];
acadoWorkspace.H[60] = acadoWorkspace.H10[14];
acadoWorkspace.H[61] = acadoWorkspace.H10[20];
acadoWorkspace.H[62] = acadoWorkspace.H10[26];
acadoWorkspace.H[63] = acadoWorkspace.H10[32];
acadoWorkspace.H[64] = acadoWorkspace.H10[38];
acadoWorkspace.H[65] = acadoWorkspace.H10[44];
acadoWorkspace.H[66] = acadoWorkspace.H10[50];
acadoWorkspace.H[67] = acadoWorkspace.H10[56];
acadoWorkspace.H[68] = acadoWorkspace.H10[62];
acadoWorkspace.H[69] = acadoWorkspace.H10[68];
acadoWorkspace.H[70] = acadoWorkspace.H10[74];
acadoWorkspace.H[71] = acadoWorkspace.H10[80];
acadoWorkspace.H[72] = acadoWorkspace.H10[86];
acadoWorkspace.H[73] = acadoWorkspace.H10[92];
acadoWorkspace.H[74] = acadoWorkspace.H10[98];
acadoWorkspace.H[75] = acadoWorkspace.H10[104];
acadoWorkspace.H[76] = acadoWorkspace.H10[110];
acadoWorkspace.H[77] = acadoWorkspace.H10[116];
acadoWorkspace.H[84] = acadoWorkspace.H10[3];
acadoWorkspace.H[85] = acadoWorkspace.H10[9];
acadoWorkspace.H[86] = acadoWorkspace.H10[15];
acadoWorkspace.H[87] = acadoWorkspace.H10[21];
acadoWorkspace.H[88] = acadoWorkspace.H10[27];
acadoWorkspace.H[89] = acadoWorkspace.H10[33];
acadoWorkspace.H[90] = acadoWorkspace.H10[39];
acadoWorkspace.H[91] = acadoWorkspace.H10[45];
acadoWorkspace.H[92] = acadoWorkspace.H10[51];
acadoWorkspace.H[93] = acadoWorkspace.H10[57];
acadoWorkspace.H[94] = acadoWorkspace.H10[63];
acadoWorkspace.H[95] = acadoWorkspace.H10[69];
acadoWorkspace.H[96] = acadoWorkspace.H10[75];
acadoWorkspace.H[97] = acadoWorkspace.H10[81];
acadoWorkspace.H[98] = acadoWorkspace.H10[87];
acadoWorkspace.H[99] = acadoWorkspace.H10[93];
acadoWorkspace.H[100] = acadoWorkspace.H10[99];
acadoWorkspace.H[101] = acadoWorkspace.H10[105];
acadoWorkspace.H[102] = acadoWorkspace.H10[111];
acadoWorkspace.H[103] = acadoWorkspace.H10[117];
acadoWorkspace.H[110] = acadoWorkspace.H10[4];
acadoWorkspace.H[111] = acadoWorkspace.H10[10];
acadoWorkspace.H[112] = acadoWorkspace.H10[16];
acadoWorkspace.H[113] = acadoWorkspace.H10[22];
acadoWorkspace.H[114] = acadoWorkspace.H10[28];
acadoWorkspace.H[115] = acadoWorkspace.H10[34];
acadoWorkspace.H[116] = acadoWorkspace.H10[40];
acadoWorkspace.H[117] = acadoWorkspace.H10[46];
acadoWorkspace.H[118] = acadoWorkspace.H10[52];
acadoWorkspace.H[119] = acadoWorkspace.H10[58];
acadoWorkspace.H[120] = acadoWorkspace.H10[64];
acadoWorkspace.H[121] = acadoWorkspace.H10[70];
acadoWorkspace.H[122] = acadoWorkspace.H10[76];
acadoWorkspace.H[123] = acadoWorkspace.H10[82];
acadoWorkspace.H[124] = acadoWorkspace.H10[88];
acadoWorkspace.H[125] = acadoWorkspace.H10[94];
acadoWorkspace.H[126] = acadoWorkspace.H10[100];
acadoWorkspace.H[127] = acadoWorkspace.H10[106];
acadoWorkspace.H[128] = acadoWorkspace.H10[112];
acadoWorkspace.H[129] = acadoWorkspace.H10[118];
acadoWorkspace.H[136] = acadoWorkspace.H10[5];
acadoWorkspace.H[137] = acadoWorkspace.H10[11];
acadoWorkspace.H[138] = acadoWorkspace.H10[17];
acadoWorkspace.H[139] = acadoWorkspace.H10[23];
acadoWorkspace.H[140] = acadoWorkspace.H10[29];
acadoWorkspace.H[141] = acadoWorkspace.H10[35];
acadoWorkspace.H[142] = acadoWorkspace.H10[41];
acadoWorkspace.H[143] = acadoWorkspace.H10[47];
acadoWorkspace.H[144] = acadoWorkspace.H10[53];
acadoWorkspace.H[145] = acadoWorkspace.H10[59];
acadoWorkspace.H[146] = acadoWorkspace.H10[65];
acadoWorkspace.H[147] = acadoWorkspace.H10[71];
acadoWorkspace.H[148] = acadoWorkspace.H10[77];
acadoWorkspace.H[149] = acadoWorkspace.H10[83];
acadoWorkspace.H[150] = acadoWorkspace.H10[89];
acadoWorkspace.H[151] = acadoWorkspace.H10[95];
acadoWorkspace.H[152] = acadoWorkspace.H10[101];
acadoWorkspace.H[153] = acadoWorkspace.H10[107];
acadoWorkspace.H[154] = acadoWorkspace.H10[113];
acadoWorkspace.H[155] = acadoWorkspace.H10[119];

acado_setBlockH11_R1( 0, 0, acadoWorkspace.R1 );
acado_setBlockH11( 0, 0, acadoWorkspace.E, acadoWorkspace.QE );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QE[ 6 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 18 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 36 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 60 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 90 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 126 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 168 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 216 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 270 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 330 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 396 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 468 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 546 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 630 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 720 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 816 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 918 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1026 ]) );
acado_setBlockH11( 0, 0, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1140 ]) );

acado_zeroBlockH11( 0, 1 );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 636 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 726 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 822 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 924 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 0, 1, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1146 ]) );

acado_zeroBlockH11( 0, 2 );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 642 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 732 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 828 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1038 ]) );
acado_setBlockH11( 0, 2, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1152 ]) );

acado_zeroBlockH11( 0, 3 );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 738 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 834 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1044 ]) );
acado_setBlockH11( 0, 3, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1158 ]) );

acado_zeroBlockH11( 0, 4 );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 0, 4, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1164 ]) );

acado_zeroBlockH11( 0, 5 );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 0, 5, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 0, 6 );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 0, 6, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 0, 7 );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 0, 7, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 0, 8 );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 0, 8, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 0, 9 );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 0, 9, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 0, 10 );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 0, 10, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 0, 11 );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 0, 11, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 0, 12 );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 0, 12, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 0, 13 );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 0, 13, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 0, 14 );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 0, 14, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 0, 15 );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 0, 15, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 0, 16 );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 0, 16, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 0, 17 );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 0, 17, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 0, 18 );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 0, 18, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 0, 19 );
acado_setBlockH11( 0, 19, &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 1, 1, &(acadoWorkspace.R1[ 1 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QE[ 12 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 24 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 42 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 66 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 96 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 132 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 174 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 222 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 276 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 336 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 402 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 474 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 552 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 636 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 726 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 822 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 924 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1032 ]) );
acado_setBlockH11( 1, 1, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1146 ]) );

acado_zeroBlockH11( 1, 2 );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 642 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 732 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 828 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1038 ]) );
acado_setBlockH11( 1, 2, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1152 ]) );

acado_zeroBlockH11( 1, 3 );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 738 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 834 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1044 ]) );
acado_setBlockH11( 1, 3, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1158 ]) );

acado_zeroBlockH11( 1, 4 );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 1, 4, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1164 ]) );

acado_zeroBlockH11( 1, 5 );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 1, 5, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 1, 6 );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 1, 6, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 1, 7 );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 1, 7, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 1, 8 );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 1, 8, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 1, 9 );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 1, 9, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 1, 10 );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 1, 10, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 1, 11 );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 1, 11, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 1, 12 );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 1, 12, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 1, 13 );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 1, 13, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 1, 14 );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 1, 14, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 1, 15 );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 1, 15, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 1, 16 );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 1, 16, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 1, 17 );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 1, 17, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 1, 18 );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 1, 18, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 1, 19 );
acado_setBlockH11( 1, 19, &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 2, 2, &(acadoWorkspace.R1[ 2 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QE[ 30 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 48 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 72 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 102 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 138 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 180 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 228 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 282 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 342 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 408 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 480 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 558 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 642 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 732 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 828 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 930 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1038 ]) );
acado_setBlockH11( 2, 2, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1152 ]) );

acado_zeroBlockH11( 2, 3 );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 738 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 834 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1044 ]) );
acado_setBlockH11( 2, 3, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1158 ]) );

acado_zeroBlockH11( 2, 4 );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 2, 4, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1164 ]) );

acado_zeroBlockH11( 2, 5 );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 2, 5, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 2, 6 );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 2, 6, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 2, 7 );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 2, 7, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 2, 8 );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 2, 8, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 2, 9 );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 2, 9, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 2, 10 );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 2, 10, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 2, 11 );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 2, 11, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 2, 12 );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 2, 12, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 2, 13 );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 2, 13, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 2, 14 );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 2, 14, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 2, 15 );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 2, 15, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 2, 16 );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 2, 16, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 2, 17 );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 2, 17, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 2, 18 );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 2, 18, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 2, 19 );
acado_setBlockH11( 2, 19, &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 3, 3, &(acadoWorkspace.R1[ 3 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QE[ 54 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 78 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 108 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 144 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 186 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 234 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 288 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 348 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 414 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 486 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 564 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 648 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 738 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 834 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 936 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1044 ]) );
acado_setBlockH11( 3, 3, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1158 ]) );

acado_zeroBlockH11( 3, 4 );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 3, 4, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1164 ]) );

acado_zeroBlockH11( 3, 5 );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 3, 5, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 3, 6 );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 3, 6, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 3, 7 );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 3, 7, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 3, 8 );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 3, 8, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 3, 9 );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 3, 9, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 3, 10 );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 3, 10, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 3, 11 );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 3, 11, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 3, 12 );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 3, 12, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 3, 13 );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 3, 13, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 3, 14 );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 3, 14, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 3, 15 );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 3, 15, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 3, 16 );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 3, 16, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 3, 17 );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 3, 17, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 3, 18 );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 3, 18, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 3, 19 );
acado_setBlockH11( 3, 19, &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 4, 4, &(acadoWorkspace.R1[ 4 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QE[ 84 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 114 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 150 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 192 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 240 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 294 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 354 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 420 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 492 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 570 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 654 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 744 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 840 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 942 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1050 ]) );
acado_setBlockH11( 4, 4, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1164 ]) );

acado_zeroBlockH11( 4, 5 );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 4, 5, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 4, 6 );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 4, 6, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 4, 7 );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 4, 7, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 4, 8 );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 4, 8, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 4, 9 );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 4, 9, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 4, 10 );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 4, 10, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 4, 11 );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 4, 11, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 4, 12 );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 4, 12, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 4, 13 );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 4, 13, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 4, 14 );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 4, 14, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 4, 15 );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 4, 15, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 4, 16 );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 4, 16, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 4, 17 );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 4, 17, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 4, 18 );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 4, 18, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 4, 19 );
acado_setBlockH11( 4, 19, &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 5, 5, &(acadoWorkspace.R1[ 5 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QE[ 120 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 156 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 198 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 246 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 300 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 360 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 426 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 498 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 576 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 660 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 750 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 846 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 948 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1056 ]) );
acado_setBlockH11( 5, 5, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1170 ]) );

acado_zeroBlockH11( 5, 6 );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 5, 6, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 5, 7 );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 5, 7, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 5, 8 );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 5, 8, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 5, 9 );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 5, 9, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 5, 10 );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 5, 10, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 5, 11 );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 5, 11, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 5, 12 );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 5, 12, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 5, 13 );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 5, 13, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 5, 14 );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 5, 14, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 5, 15 );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 5, 15, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 5, 16 );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 5, 16, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 5, 17 );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 5, 17, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 5, 18 );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 5, 18, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 5, 19 );
acado_setBlockH11( 5, 19, &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 6, 6, &(acadoWorkspace.R1[ 6 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QE[ 162 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 204 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 252 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 306 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 366 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 432 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 504 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 582 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 666 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 756 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 852 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 954 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1062 ]) );
acado_setBlockH11( 6, 6, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1176 ]) );

acado_zeroBlockH11( 6, 7 );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 6, 7, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 6, 8 );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 6, 8, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 6, 9 );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 6, 9, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 6, 10 );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 6, 10, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 6, 11 );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 6, 11, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 6, 12 );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 6, 12, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 6, 13 );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 6, 13, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 6, 14 );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 6, 14, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 6, 15 );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 6, 15, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 6, 16 );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 6, 16, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 6, 17 );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 6, 17, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 6, 18 );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 6, 18, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 6, 19 );
acado_setBlockH11( 6, 19, &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 7, 7, &(acadoWorkspace.R1[ 7 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QE[ 210 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 258 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 312 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 372 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 438 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 510 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 588 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 672 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 762 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 858 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 960 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1068 ]) );
acado_setBlockH11( 7, 7, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1182 ]) );

acado_zeroBlockH11( 7, 8 );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 7, 8, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 7, 9 );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 7, 9, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 7, 10 );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 7, 10, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 7, 11 );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 7, 11, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 7, 12 );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 7, 12, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 7, 13 );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 7, 13, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 7, 14 );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 7, 14, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 7, 15 );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 7, 15, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 7, 16 );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 7, 16, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 7, 17 );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 7, 17, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 7, 18 );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 7, 18, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 7, 19 );
acado_setBlockH11( 7, 19, &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 8, 8, &(acadoWorkspace.R1[ 8 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QE[ 264 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 318 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 378 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 444 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 516 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 594 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 678 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 768 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 864 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 966 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1074 ]) );
acado_setBlockH11( 8, 8, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1188 ]) );

acado_zeroBlockH11( 8, 9 );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 8, 9, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 8, 10 );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 8, 10, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 8, 11 );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 8, 11, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 8, 12 );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 8, 12, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 8, 13 );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 8, 13, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 8, 14 );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 8, 14, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 8, 15 );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 8, 15, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 8, 16 );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 8, 16, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 8, 17 );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 8, 17, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 8, 18 );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 8, 18, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 8, 19 );
acado_setBlockH11( 8, 19, &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 9, 9, &(acadoWorkspace.R1[ 9 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QE[ 324 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 384 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 450 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 522 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 600 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 684 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 774 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 870 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 972 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1080 ]) );
acado_setBlockH11( 9, 9, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1194 ]) );

acado_zeroBlockH11( 9, 10 );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 9, 10, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 9, 11 );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 9, 11, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 9, 12 );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 9, 12, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 9, 13 );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 9, 13, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 9, 14 );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 9, 14, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 9, 15 );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 9, 15, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 9, 16 );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 9, 16, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 9, 17 );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 9, 17, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 9, 18 );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 9, 18, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 9, 19 );
acado_setBlockH11( 9, 19, &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 10, 10, &(acadoWorkspace.R1[ 10 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QE[ 390 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 456 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 528 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 606 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 690 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 780 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 876 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 978 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1086 ]) );
acado_setBlockH11( 10, 10, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1200 ]) );

acado_zeroBlockH11( 10, 11 );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 10, 11, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 10, 12 );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 10, 12, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 10, 13 );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 10, 13, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 10, 14 );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 10, 14, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 10, 15 );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 10, 15, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 10, 16 );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 10, 16, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 10, 17 );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 10, 17, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 10, 18 );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 10, 18, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 10, 19 );
acado_setBlockH11( 10, 19, &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 11, 11, &(acadoWorkspace.R1[ 11 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QE[ 462 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 534 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 612 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 696 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 786 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 882 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 984 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1092 ]) );
acado_setBlockH11( 11, 11, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1206 ]) );

acado_zeroBlockH11( 11, 12 );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 11, 12, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 11, 13 );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 11, 13, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 11, 14 );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 11, 14, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 11, 15 );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 11, 15, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 11, 16 );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 11, 16, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 11, 17 );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 11, 17, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 11, 18 );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 11, 18, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 11, 19 );
acado_setBlockH11( 11, 19, &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 12, 12, &(acadoWorkspace.R1[ 12 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QE[ 540 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 618 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.QE[ 702 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 792 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 888 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 990 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1098 ]) );
acado_setBlockH11( 12, 12, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1212 ]) );

acado_zeroBlockH11( 12, 13 );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 12, 13, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 12, 14 );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 12, 14, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 12, 15 );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 12, 15, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 12, 16 );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 12, 16, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 12, 17 );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 12, 17, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 12, 18 );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 12, 18, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 12, 19 );
acado_setBlockH11( 12, 19, &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 13, 13, &(acadoWorkspace.R1[ 13 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QE[ 624 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.QE[ 708 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.QE[ 798 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QE[ 894 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 996 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1104 ]) );
acado_setBlockH11( 13, 13, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1218 ]) );

acado_zeroBlockH11( 13, 14 );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 13, 14, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 13, 15 );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 13, 15, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 13, 16 );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 13, 16, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 13, 17 );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 13, 17, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 13, 18 );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 13, 18, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 13, 19 );
acado_setBlockH11( 13, 19, &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 14, 14, &(acadoWorkspace.R1[ 14 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 714 ]), &(acadoWorkspace.QE[ 714 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.QE[ 804 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 900 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QE[ 1002 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1110 ]) );
acado_setBlockH11( 14, 14, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1224 ]) );

acado_zeroBlockH11( 14, 15 );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 14, 15, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 14, 16 );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 14, 16, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 14, 17 );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 14, 17, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 14, 18 );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 14, 18, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 14, 19 );
acado_setBlockH11( 14, 19, &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 15, 15, &(acadoWorkspace.R1[ 15 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QE[ 810 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.QE[ 906 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1008 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QE[ 1116 ]) );
acado_setBlockH11( 15, 15, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1230 ]) );

acado_zeroBlockH11( 15, 16 );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 15, 16, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 15, 17 );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 15, 17, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 15, 18 );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 15, 18, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 15, 19 );
acado_setBlockH11( 15, 19, &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 16, 16, &(acadoWorkspace.R1[ 16 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QE[ 912 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.QE[ 1014 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.QE[ 1122 ]) );
acado_setBlockH11( 16, 16, &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QE[ 1236 ]) );

acado_zeroBlockH11( 16, 17 );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 16, 17, &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 16, 18 );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 16, 18, &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 16, 19 );
acado_setBlockH11( 16, 19, &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 17, 17, &(acadoWorkspace.R1[ 17 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QE[ 1020 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1128 ]) );
acado_setBlockH11( 17, 17, &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.QE[ 1242 ]) );

acado_zeroBlockH11( 17, 18 );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 17, 18, &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 17, 19 );
acado_setBlockH11( 17, 19, &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 18, 18, &(acadoWorkspace.R1[ 18 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 1134 ]), &(acadoWorkspace.QE[ 1134 ]) );
acado_setBlockH11( 18, 18, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1248 ]) );

acado_zeroBlockH11( 18, 19 );
acado_setBlockH11( 18, 19, &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QE[ 1254 ]) );

acado_setBlockH11_R1( 19, 19, &(acadoWorkspace.R1[ 19 ]) );
acado_setBlockH11( 19, 19, &(acadoWorkspace.E[ 1254 ]), &(acadoWorkspace.QE[ 1254 ]) );


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

acadoWorkspace.H[156] = acadoWorkspace.H10[0];
acadoWorkspace.H[157] = acadoWorkspace.H10[1];
acadoWorkspace.H[158] = acadoWorkspace.H10[2];
acadoWorkspace.H[159] = acadoWorkspace.H10[3];
acadoWorkspace.H[160] = acadoWorkspace.H10[4];
acadoWorkspace.H[161] = acadoWorkspace.H10[5];
acadoWorkspace.H[182] = acadoWorkspace.H10[6];
acadoWorkspace.H[183] = acadoWorkspace.H10[7];
acadoWorkspace.H[184] = acadoWorkspace.H10[8];
acadoWorkspace.H[185] = acadoWorkspace.H10[9];
acadoWorkspace.H[186] = acadoWorkspace.H10[10];
acadoWorkspace.H[187] = acadoWorkspace.H10[11];
acadoWorkspace.H[208] = acadoWorkspace.H10[12];
acadoWorkspace.H[209] = acadoWorkspace.H10[13];
acadoWorkspace.H[210] = acadoWorkspace.H10[14];
acadoWorkspace.H[211] = acadoWorkspace.H10[15];
acadoWorkspace.H[212] = acadoWorkspace.H10[16];
acadoWorkspace.H[213] = acadoWorkspace.H10[17];
acadoWorkspace.H[234] = acadoWorkspace.H10[18];
acadoWorkspace.H[235] = acadoWorkspace.H10[19];
acadoWorkspace.H[236] = acadoWorkspace.H10[20];
acadoWorkspace.H[237] = acadoWorkspace.H10[21];
acadoWorkspace.H[238] = acadoWorkspace.H10[22];
acadoWorkspace.H[239] = acadoWorkspace.H10[23];
acadoWorkspace.H[260] = acadoWorkspace.H10[24];
acadoWorkspace.H[261] = acadoWorkspace.H10[25];
acadoWorkspace.H[262] = acadoWorkspace.H10[26];
acadoWorkspace.H[263] = acadoWorkspace.H10[27];
acadoWorkspace.H[264] = acadoWorkspace.H10[28];
acadoWorkspace.H[265] = acadoWorkspace.H10[29];
acadoWorkspace.H[286] = acadoWorkspace.H10[30];
acadoWorkspace.H[287] = acadoWorkspace.H10[31];
acadoWorkspace.H[288] = acadoWorkspace.H10[32];
acadoWorkspace.H[289] = acadoWorkspace.H10[33];
acadoWorkspace.H[290] = acadoWorkspace.H10[34];
acadoWorkspace.H[291] = acadoWorkspace.H10[35];
acadoWorkspace.H[312] = acadoWorkspace.H10[36];
acadoWorkspace.H[313] = acadoWorkspace.H10[37];
acadoWorkspace.H[314] = acadoWorkspace.H10[38];
acadoWorkspace.H[315] = acadoWorkspace.H10[39];
acadoWorkspace.H[316] = acadoWorkspace.H10[40];
acadoWorkspace.H[317] = acadoWorkspace.H10[41];
acadoWorkspace.H[338] = acadoWorkspace.H10[42];
acadoWorkspace.H[339] = acadoWorkspace.H10[43];
acadoWorkspace.H[340] = acadoWorkspace.H10[44];
acadoWorkspace.H[341] = acadoWorkspace.H10[45];
acadoWorkspace.H[342] = acadoWorkspace.H10[46];
acadoWorkspace.H[343] = acadoWorkspace.H10[47];
acadoWorkspace.H[364] = acadoWorkspace.H10[48];
acadoWorkspace.H[365] = acadoWorkspace.H10[49];
acadoWorkspace.H[366] = acadoWorkspace.H10[50];
acadoWorkspace.H[367] = acadoWorkspace.H10[51];
acadoWorkspace.H[368] = acadoWorkspace.H10[52];
acadoWorkspace.H[369] = acadoWorkspace.H10[53];
acadoWorkspace.H[390] = acadoWorkspace.H10[54];
acadoWorkspace.H[391] = acadoWorkspace.H10[55];
acadoWorkspace.H[392] = acadoWorkspace.H10[56];
acadoWorkspace.H[393] = acadoWorkspace.H10[57];
acadoWorkspace.H[394] = acadoWorkspace.H10[58];
acadoWorkspace.H[395] = acadoWorkspace.H10[59];
acadoWorkspace.H[416] = acadoWorkspace.H10[60];
acadoWorkspace.H[417] = acadoWorkspace.H10[61];
acadoWorkspace.H[418] = acadoWorkspace.H10[62];
acadoWorkspace.H[419] = acadoWorkspace.H10[63];
acadoWorkspace.H[420] = acadoWorkspace.H10[64];
acadoWorkspace.H[421] = acadoWorkspace.H10[65];
acadoWorkspace.H[442] = acadoWorkspace.H10[66];
acadoWorkspace.H[443] = acadoWorkspace.H10[67];
acadoWorkspace.H[444] = acadoWorkspace.H10[68];
acadoWorkspace.H[445] = acadoWorkspace.H10[69];
acadoWorkspace.H[446] = acadoWorkspace.H10[70];
acadoWorkspace.H[447] = acadoWorkspace.H10[71];
acadoWorkspace.H[468] = acadoWorkspace.H10[72];
acadoWorkspace.H[469] = acadoWorkspace.H10[73];
acadoWorkspace.H[470] = acadoWorkspace.H10[74];
acadoWorkspace.H[471] = acadoWorkspace.H10[75];
acadoWorkspace.H[472] = acadoWorkspace.H10[76];
acadoWorkspace.H[473] = acadoWorkspace.H10[77];
acadoWorkspace.H[494] = acadoWorkspace.H10[78];
acadoWorkspace.H[495] = acadoWorkspace.H10[79];
acadoWorkspace.H[496] = acadoWorkspace.H10[80];
acadoWorkspace.H[497] = acadoWorkspace.H10[81];
acadoWorkspace.H[498] = acadoWorkspace.H10[82];
acadoWorkspace.H[499] = acadoWorkspace.H10[83];
acadoWorkspace.H[520] = acadoWorkspace.H10[84];
acadoWorkspace.H[521] = acadoWorkspace.H10[85];
acadoWorkspace.H[522] = acadoWorkspace.H10[86];
acadoWorkspace.H[523] = acadoWorkspace.H10[87];
acadoWorkspace.H[524] = acadoWorkspace.H10[88];
acadoWorkspace.H[525] = acadoWorkspace.H10[89];
acadoWorkspace.H[546] = acadoWorkspace.H10[90];
acadoWorkspace.H[547] = acadoWorkspace.H10[91];
acadoWorkspace.H[548] = acadoWorkspace.H10[92];
acadoWorkspace.H[549] = acadoWorkspace.H10[93];
acadoWorkspace.H[550] = acadoWorkspace.H10[94];
acadoWorkspace.H[551] = acadoWorkspace.H10[95];
acadoWorkspace.H[572] = acadoWorkspace.H10[96];
acadoWorkspace.H[573] = acadoWorkspace.H10[97];
acadoWorkspace.H[574] = acadoWorkspace.H10[98];
acadoWorkspace.H[575] = acadoWorkspace.H10[99];
acadoWorkspace.H[576] = acadoWorkspace.H10[100];
acadoWorkspace.H[577] = acadoWorkspace.H10[101];
acadoWorkspace.H[598] = acadoWorkspace.H10[102];
acadoWorkspace.H[599] = acadoWorkspace.H10[103];
acadoWorkspace.H[600] = acadoWorkspace.H10[104];
acadoWorkspace.H[601] = acadoWorkspace.H10[105];
acadoWorkspace.H[602] = acadoWorkspace.H10[106];
acadoWorkspace.H[603] = acadoWorkspace.H10[107];
acadoWorkspace.H[624] = acadoWorkspace.H10[108];
acadoWorkspace.H[625] = acadoWorkspace.H10[109];
acadoWorkspace.H[626] = acadoWorkspace.H10[110];
acadoWorkspace.H[627] = acadoWorkspace.H10[111];
acadoWorkspace.H[628] = acadoWorkspace.H10[112];
acadoWorkspace.H[629] = acadoWorkspace.H10[113];
acadoWorkspace.H[650] = acadoWorkspace.H10[114];
acadoWorkspace.H[651] = acadoWorkspace.H10[115];
acadoWorkspace.H[652] = acadoWorkspace.H10[116];
acadoWorkspace.H[653] = acadoWorkspace.H10[117];
acadoWorkspace.H[654] = acadoWorkspace.H10[118];
acadoWorkspace.H[655] = acadoWorkspace.H10[119];

acado_multQ1d( &(acadoWorkspace.Q1[ 36 ]), acadoWorkspace.d, acadoWorkspace.Qd );
acado_multQ1d( &(acadoWorkspace.Q1[ 72 ]), &(acadoWorkspace.d[ 6 ]), &(acadoWorkspace.Qd[ 6 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 108 ]), &(acadoWorkspace.d[ 12 ]), &(acadoWorkspace.Qd[ 12 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.d[ 18 ]), &(acadoWorkspace.Qd[ 18 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 180 ]), &(acadoWorkspace.d[ 24 ]), &(acadoWorkspace.Qd[ 24 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 216 ]), &(acadoWorkspace.d[ 30 ]), &(acadoWorkspace.Qd[ 30 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 252 ]), &(acadoWorkspace.d[ 36 ]), &(acadoWorkspace.Qd[ 36 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.d[ 42 ]), &(acadoWorkspace.Qd[ 42 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 324 ]), &(acadoWorkspace.d[ 48 ]), &(acadoWorkspace.Qd[ 48 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 360 ]), &(acadoWorkspace.d[ 54 ]), &(acadoWorkspace.Qd[ 54 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 396 ]), &(acadoWorkspace.d[ 60 ]), &(acadoWorkspace.Qd[ 60 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.d[ 66 ]), &(acadoWorkspace.Qd[ 66 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 468 ]), &(acadoWorkspace.d[ 72 ]), &(acadoWorkspace.Qd[ 72 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 504 ]), &(acadoWorkspace.d[ 78 ]), &(acadoWorkspace.Qd[ 78 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 540 ]), &(acadoWorkspace.d[ 84 ]), &(acadoWorkspace.Qd[ 84 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.d[ 90 ]), &(acadoWorkspace.Qd[ 90 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 612 ]), &(acadoWorkspace.d[ 96 ]), &(acadoWorkspace.Qd[ 96 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 648 ]), &(acadoWorkspace.d[ 102 ]), &(acadoWorkspace.Qd[ 102 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 684 ]), &(acadoWorkspace.d[ 108 ]), &(acadoWorkspace.Qd[ 108 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 114 ]), &(acadoWorkspace.Qd[ 114 ]) );

acado_macCTSlx( acadoWorkspace.evGx, acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 36 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 72 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 108 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 180 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 216 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 252 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 324 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 360 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 396 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 432 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 468 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 504 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 540 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 576 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 612 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 648 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 684 ]), acadoWorkspace.g );
acado_macETSlu( acadoWorkspace.QE, &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 6 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 18 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 36 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 60 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 90 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 126 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 168 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 216 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 270 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 330 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 396 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 468 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 546 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 630 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 720 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 816 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 918 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1026 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1140 ]), &(acadoWorkspace.g[ 6 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 12 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 24 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 42 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 66 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 96 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 132 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 174 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 222 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 276 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 336 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 402 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 474 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 552 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 636 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 726 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 822 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 924 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1032 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1146 ]), &(acadoWorkspace.g[ 7 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 30 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 48 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 72 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 102 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 138 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 180 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 228 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 282 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 342 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 408 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 480 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 558 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 642 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 732 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 828 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 930 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1038 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1152 ]), &(acadoWorkspace.g[ 8 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 54 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 78 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 108 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 144 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 186 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 234 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 288 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 348 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 414 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 486 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 564 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 648 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 738 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 834 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 936 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1044 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1158 ]), &(acadoWorkspace.g[ 9 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 84 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 114 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 150 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 192 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 240 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 294 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 354 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 420 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 492 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 570 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 654 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 744 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 840 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 942 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1050 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1164 ]), &(acadoWorkspace.g[ 10 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 120 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 156 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 198 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 246 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 300 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 360 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 426 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 498 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 576 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 660 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 750 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 846 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 948 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1056 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1170 ]), &(acadoWorkspace.g[ 11 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 162 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 204 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 252 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 306 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 366 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 432 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 504 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 582 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 666 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 756 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 852 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 954 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1062 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1176 ]), &(acadoWorkspace.g[ 12 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 210 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 258 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 312 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 372 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 438 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 510 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 588 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 672 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 762 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 858 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 960 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1068 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1182 ]), &(acadoWorkspace.g[ 13 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 264 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 318 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 378 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 444 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 516 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 594 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 678 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 768 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 864 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 966 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1074 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1188 ]), &(acadoWorkspace.g[ 14 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 324 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 384 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 450 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 522 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 600 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 684 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 774 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 870 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 972 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1080 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1194 ]), &(acadoWorkspace.g[ 15 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 390 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 456 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 528 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 606 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 690 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 780 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 876 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 978 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1086 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1200 ]), &(acadoWorkspace.g[ 16 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 462 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 534 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 612 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 696 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 786 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 882 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 984 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1092 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1206 ]), &(acadoWorkspace.g[ 17 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 540 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 618 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 702 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 792 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 888 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 990 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1098 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1212 ]), &(acadoWorkspace.g[ 18 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 624 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 708 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 798 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 894 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 996 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1104 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1218 ]), &(acadoWorkspace.g[ 19 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 714 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 804 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 900 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1002 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1110 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1224 ]), &(acadoWorkspace.g[ 20 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 810 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 906 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1008 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1116 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1230 ]), &(acadoWorkspace.g[ 21 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 912 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1014 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1122 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1236 ]), &(acadoWorkspace.g[ 22 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1020 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1128 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1242 ]), &(acadoWorkspace.g[ 23 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1134 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1248 ]), &(acadoWorkspace.g[ 24 ]) );
acado_macETSlu( &(acadoWorkspace.QE[ 1254 ]), &(acadoWorkspace.g[ 25 ]) );
acadoWorkspace.lb[6] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.lb[10] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.lb[11] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.lb[12] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.lb[13] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.lb[14] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.lb[15] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.lb[16] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.lb[17] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.lb[18] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.lb[19] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.lb[20] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.lb[21] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.lb[22] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.lb[23] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.lb[24] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.lb[25] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[19];
acadoWorkspace.ub[6] = (real_t)1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.ub[7] = (real_t)1.0000000000000000e+12 - acadoVariables.u[1];
acadoWorkspace.ub[8] = (real_t)1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.ub[9] = (real_t)1.0000000000000000e+12 - acadoVariables.u[3];
acadoWorkspace.ub[10] = (real_t)1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.ub[11] = (real_t)1.0000000000000000e+12 - acadoVariables.u[5];
acadoWorkspace.ub[12] = (real_t)1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.ub[13] = (real_t)1.0000000000000000e+12 - acadoVariables.u[7];
acadoWorkspace.ub[14] = (real_t)1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.ub[15] = (real_t)1.0000000000000000e+12 - acadoVariables.u[9];
acadoWorkspace.ub[16] = (real_t)1.0000000000000000e+12 - acadoVariables.u[10];
acadoWorkspace.ub[17] = (real_t)1.0000000000000000e+12 - acadoVariables.u[11];
acadoWorkspace.ub[18] = (real_t)1.0000000000000000e+12 - acadoVariables.u[12];
acadoWorkspace.ub[19] = (real_t)1.0000000000000000e+12 - acadoVariables.u[13];
acadoWorkspace.ub[20] = (real_t)1.0000000000000000e+12 - acadoVariables.u[14];
acadoWorkspace.ub[21] = (real_t)1.0000000000000000e+12 - acadoVariables.u[15];
acadoWorkspace.ub[22] = (real_t)1.0000000000000000e+12 - acadoVariables.u[16];
acadoWorkspace.ub[23] = (real_t)1.0000000000000000e+12 - acadoVariables.u[17];
acadoWorkspace.ub[24] = (real_t)1.0000000000000000e+12 - acadoVariables.u[18];
acadoWorkspace.ub[25] = (real_t)1.0000000000000000e+12 - acadoVariables.u[19];

for (lRun1 = 0; lRun1 < 20; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 6;
lRun4 = ((lRun3) / (6)) + (1);
acadoWorkspace.A[lRun1 * 26] = acadoWorkspace.evGx[lRun3 * 6];
acadoWorkspace.A[lRun1 * 26 + 1] = acadoWorkspace.evGx[lRun3 * 6 + 1];
acadoWorkspace.A[lRun1 * 26 + 2] = acadoWorkspace.evGx[lRun3 * 6 + 2];
acadoWorkspace.A[lRun1 * 26 + 3] = acadoWorkspace.evGx[lRun3 * 6 + 3];
acadoWorkspace.A[lRun1 * 26 + 4] = acadoWorkspace.evGx[lRun3 * 6 + 4];
acadoWorkspace.A[lRun1 * 26 + 5] = acadoWorkspace.evGx[lRun3 * 6 + 5];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (6)) + ((lRun3) % (6));
acadoWorkspace.A[(lRun1 * 26) + (lRun2 + 6)] = acadoWorkspace.E[lRun5];
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
acadoWorkspace.Dx0[4] = acadoVariables.x0[4] - acadoVariables.x[4];
acadoWorkspace.Dx0[5] = acadoVariables.x0[5] - acadoVariables.x[5];

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

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 4 ]), &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 8 ]), &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 12 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 16 ]), &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 20 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 24 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 28 ]), &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 32 ]), &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 36 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 40 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 44 ]), &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 48 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 52 ]), &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 56 ]), &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 60 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 64 ]), &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 68 ]), &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 72 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 76 ]), &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.g[ 25 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 24 ]), &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.QDy[ 6 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 48 ]), &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 72 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 18 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 96 ]), &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.QDy[ 24 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 120 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.QDy[ 30 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 144 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 36 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 168 ]), &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.QDy[ 42 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 192 ]), &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.QDy[ 48 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 216 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.QDy[ 54 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 240 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 264 ]), &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.QDy[ 66 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 288 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.QDy[ 72 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 312 ]), &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.QDy[ 78 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 336 ]), &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.QDy[ 84 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 360 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 90 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 384 ]), &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.QDy[ 96 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 408 ]), &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.QDy[ 102 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 432 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.QDy[ 108 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 456 ]), &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.QDy[ 114 ]) );

acadoWorkspace.QDy[120] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[121] = + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[122] = + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[123] = + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[124] = + acadoWorkspace.QN2[12]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[13]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[14]*acadoWorkspace.DyN[2];
acadoWorkspace.QDy[125] = + acadoWorkspace.QN2[15]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[16]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[17]*acadoWorkspace.DyN[2];

acadoWorkspace.QDy[6] += acadoWorkspace.Qd[0];
acadoWorkspace.QDy[7] += acadoWorkspace.Qd[1];
acadoWorkspace.QDy[8] += acadoWorkspace.Qd[2];
acadoWorkspace.QDy[9] += acadoWorkspace.Qd[3];
acadoWorkspace.QDy[10] += acadoWorkspace.Qd[4];
acadoWorkspace.QDy[11] += acadoWorkspace.Qd[5];
acadoWorkspace.QDy[12] += acadoWorkspace.Qd[6];
acadoWorkspace.QDy[13] += acadoWorkspace.Qd[7];
acadoWorkspace.QDy[14] += acadoWorkspace.Qd[8];
acadoWorkspace.QDy[15] += acadoWorkspace.Qd[9];
acadoWorkspace.QDy[16] += acadoWorkspace.Qd[10];
acadoWorkspace.QDy[17] += acadoWorkspace.Qd[11];
acadoWorkspace.QDy[18] += acadoWorkspace.Qd[12];
acadoWorkspace.QDy[19] += acadoWorkspace.Qd[13];
acadoWorkspace.QDy[20] += acadoWorkspace.Qd[14];
acadoWorkspace.QDy[21] += acadoWorkspace.Qd[15];
acadoWorkspace.QDy[22] += acadoWorkspace.Qd[16];
acadoWorkspace.QDy[23] += acadoWorkspace.Qd[17];
acadoWorkspace.QDy[24] += acadoWorkspace.Qd[18];
acadoWorkspace.QDy[25] += acadoWorkspace.Qd[19];
acadoWorkspace.QDy[26] += acadoWorkspace.Qd[20];
acadoWorkspace.QDy[27] += acadoWorkspace.Qd[21];
acadoWorkspace.QDy[28] += acadoWorkspace.Qd[22];
acadoWorkspace.QDy[29] += acadoWorkspace.Qd[23];
acadoWorkspace.QDy[30] += acadoWorkspace.Qd[24];
acadoWorkspace.QDy[31] += acadoWorkspace.Qd[25];
acadoWorkspace.QDy[32] += acadoWorkspace.Qd[26];
acadoWorkspace.QDy[33] += acadoWorkspace.Qd[27];
acadoWorkspace.QDy[34] += acadoWorkspace.Qd[28];
acadoWorkspace.QDy[35] += acadoWorkspace.Qd[29];
acadoWorkspace.QDy[36] += acadoWorkspace.Qd[30];
acadoWorkspace.QDy[37] += acadoWorkspace.Qd[31];
acadoWorkspace.QDy[38] += acadoWorkspace.Qd[32];
acadoWorkspace.QDy[39] += acadoWorkspace.Qd[33];
acadoWorkspace.QDy[40] += acadoWorkspace.Qd[34];
acadoWorkspace.QDy[41] += acadoWorkspace.Qd[35];
acadoWorkspace.QDy[42] += acadoWorkspace.Qd[36];
acadoWorkspace.QDy[43] += acadoWorkspace.Qd[37];
acadoWorkspace.QDy[44] += acadoWorkspace.Qd[38];
acadoWorkspace.QDy[45] += acadoWorkspace.Qd[39];
acadoWorkspace.QDy[46] += acadoWorkspace.Qd[40];
acadoWorkspace.QDy[47] += acadoWorkspace.Qd[41];
acadoWorkspace.QDy[48] += acadoWorkspace.Qd[42];
acadoWorkspace.QDy[49] += acadoWorkspace.Qd[43];
acadoWorkspace.QDy[50] += acadoWorkspace.Qd[44];
acadoWorkspace.QDy[51] += acadoWorkspace.Qd[45];
acadoWorkspace.QDy[52] += acadoWorkspace.Qd[46];
acadoWorkspace.QDy[53] += acadoWorkspace.Qd[47];
acadoWorkspace.QDy[54] += acadoWorkspace.Qd[48];
acadoWorkspace.QDy[55] += acadoWorkspace.Qd[49];
acadoWorkspace.QDy[56] += acadoWorkspace.Qd[50];
acadoWorkspace.QDy[57] += acadoWorkspace.Qd[51];
acadoWorkspace.QDy[58] += acadoWorkspace.Qd[52];
acadoWorkspace.QDy[59] += acadoWorkspace.Qd[53];
acadoWorkspace.QDy[60] += acadoWorkspace.Qd[54];
acadoWorkspace.QDy[61] += acadoWorkspace.Qd[55];
acadoWorkspace.QDy[62] += acadoWorkspace.Qd[56];
acadoWorkspace.QDy[63] += acadoWorkspace.Qd[57];
acadoWorkspace.QDy[64] += acadoWorkspace.Qd[58];
acadoWorkspace.QDy[65] += acadoWorkspace.Qd[59];
acadoWorkspace.QDy[66] += acadoWorkspace.Qd[60];
acadoWorkspace.QDy[67] += acadoWorkspace.Qd[61];
acadoWorkspace.QDy[68] += acadoWorkspace.Qd[62];
acadoWorkspace.QDy[69] += acadoWorkspace.Qd[63];
acadoWorkspace.QDy[70] += acadoWorkspace.Qd[64];
acadoWorkspace.QDy[71] += acadoWorkspace.Qd[65];
acadoWorkspace.QDy[72] += acadoWorkspace.Qd[66];
acadoWorkspace.QDy[73] += acadoWorkspace.Qd[67];
acadoWorkspace.QDy[74] += acadoWorkspace.Qd[68];
acadoWorkspace.QDy[75] += acadoWorkspace.Qd[69];
acadoWorkspace.QDy[76] += acadoWorkspace.Qd[70];
acadoWorkspace.QDy[77] += acadoWorkspace.Qd[71];
acadoWorkspace.QDy[78] += acadoWorkspace.Qd[72];
acadoWorkspace.QDy[79] += acadoWorkspace.Qd[73];
acadoWorkspace.QDy[80] += acadoWorkspace.Qd[74];
acadoWorkspace.QDy[81] += acadoWorkspace.Qd[75];
acadoWorkspace.QDy[82] += acadoWorkspace.Qd[76];
acadoWorkspace.QDy[83] += acadoWorkspace.Qd[77];
acadoWorkspace.QDy[84] += acadoWorkspace.Qd[78];
acadoWorkspace.QDy[85] += acadoWorkspace.Qd[79];
acadoWorkspace.QDy[86] += acadoWorkspace.Qd[80];
acadoWorkspace.QDy[87] += acadoWorkspace.Qd[81];
acadoWorkspace.QDy[88] += acadoWorkspace.Qd[82];
acadoWorkspace.QDy[89] += acadoWorkspace.Qd[83];
acadoWorkspace.QDy[90] += acadoWorkspace.Qd[84];
acadoWorkspace.QDy[91] += acadoWorkspace.Qd[85];
acadoWorkspace.QDy[92] += acadoWorkspace.Qd[86];
acadoWorkspace.QDy[93] += acadoWorkspace.Qd[87];
acadoWorkspace.QDy[94] += acadoWorkspace.Qd[88];
acadoWorkspace.QDy[95] += acadoWorkspace.Qd[89];
acadoWorkspace.QDy[96] += acadoWorkspace.Qd[90];
acadoWorkspace.QDy[97] += acadoWorkspace.Qd[91];
acadoWorkspace.QDy[98] += acadoWorkspace.Qd[92];
acadoWorkspace.QDy[99] += acadoWorkspace.Qd[93];
acadoWorkspace.QDy[100] += acadoWorkspace.Qd[94];
acadoWorkspace.QDy[101] += acadoWorkspace.Qd[95];
acadoWorkspace.QDy[102] += acadoWorkspace.Qd[96];
acadoWorkspace.QDy[103] += acadoWorkspace.Qd[97];
acadoWorkspace.QDy[104] += acadoWorkspace.Qd[98];
acadoWorkspace.QDy[105] += acadoWorkspace.Qd[99];
acadoWorkspace.QDy[106] += acadoWorkspace.Qd[100];
acadoWorkspace.QDy[107] += acadoWorkspace.Qd[101];
acadoWorkspace.QDy[108] += acadoWorkspace.Qd[102];
acadoWorkspace.QDy[109] += acadoWorkspace.Qd[103];
acadoWorkspace.QDy[110] += acadoWorkspace.Qd[104];
acadoWorkspace.QDy[111] += acadoWorkspace.Qd[105];
acadoWorkspace.QDy[112] += acadoWorkspace.Qd[106];
acadoWorkspace.QDy[113] += acadoWorkspace.Qd[107];
acadoWorkspace.QDy[114] += acadoWorkspace.Qd[108];
acadoWorkspace.QDy[115] += acadoWorkspace.Qd[109];
acadoWorkspace.QDy[116] += acadoWorkspace.Qd[110];
acadoWorkspace.QDy[117] += acadoWorkspace.Qd[111];
acadoWorkspace.QDy[118] += acadoWorkspace.Qd[112];
acadoWorkspace.QDy[119] += acadoWorkspace.Qd[113];
acadoWorkspace.QDy[120] += acadoWorkspace.Qd[114];
acadoWorkspace.QDy[121] += acadoWorkspace.Qd[115];
acadoWorkspace.QDy[122] += acadoWorkspace.Qd[116];
acadoWorkspace.QDy[123] += acadoWorkspace.Qd[117];
acadoWorkspace.QDy[124] += acadoWorkspace.Qd[118];
acadoWorkspace.QDy[125] += acadoWorkspace.Qd[119];

acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[324]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[330]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[336]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[342]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[348]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[354]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[360]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[366]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[372]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[378]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[384]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[390]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[396]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[402]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[408]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[414]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[420]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[426]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[432]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[438]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[444]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[450]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[456]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[462]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[468]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[474]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[480]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[486]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[492]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[498]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[504]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[510]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[516]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[522]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[528]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[534]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[540]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[546]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[552]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[558]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[564]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[570]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[576]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[582]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[588]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[594]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[600]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[606]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[612]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[618]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[624]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[630]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[636]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[642]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[648]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[654]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[660]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[666]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[672]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[678]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[684]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[690]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[696]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[702]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[708]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[714]*acadoWorkspace.QDy[125];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[325]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[331]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[337]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[343]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[349]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[355]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[361]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[367]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[373]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[379]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[385]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[391]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[397]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[403]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[409]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[415]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[421]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[427]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[433]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[439]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[445]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[451]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[457]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[463]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[469]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[475]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[481]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[487]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[493]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[499]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[505]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[511]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[517]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[523]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[529]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[535]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[541]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[547]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[553]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[559]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[565]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[571]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[577]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[583]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[589]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[595]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[601]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[607]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[613]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[619]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[625]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[631]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[637]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[643]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[649]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[655]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[661]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[667]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[673]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[679]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[685]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[691]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[697]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[703]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[709]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[715]*acadoWorkspace.QDy[125];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[320]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[326]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[332]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[338]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[344]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[350]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[356]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[362]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[368]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[374]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[380]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[386]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[392]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[398]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[404]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[410]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[416]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[422]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[428]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[434]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[440]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[446]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[452]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[458]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[464]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[470]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[476]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[482]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[488]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[494]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[500]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[506]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[512]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[518]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[524]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[530]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[536]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[542]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[548]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[554]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[560]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[566]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[572]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[578]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[584]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[590]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[596]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[602]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[608]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[614]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[620]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[626]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[632]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[638]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[644]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[650]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[656]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[662]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[668]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[674]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[680]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[686]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[692]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[698]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[704]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[710]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[716]*acadoWorkspace.QDy[125];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[321]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[327]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[333]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[339]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[345]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[351]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[357]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[363]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[369]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[375]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[381]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[387]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[393]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[399]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[405]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[411]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[417]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[423]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[429]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[435]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[441]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[447]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[453]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[459]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[465]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[471]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[477]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[483]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[489]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[495]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[501]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[507]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[513]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[519]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[525]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[531]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[537]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[543]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[549]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[555]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[561]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[567]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[573]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[579]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[585]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[591]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[597]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[603]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[609]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[615]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[621]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[627]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[633]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[639]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[645]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[651]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[657]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[663]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[669]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[675]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[681]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[687]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[693]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[699]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[705]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[711]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[717]*acadoWorkspace.QDy[125];
acadoWorkspace.g[4] = + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[322]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[328]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[334]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[340]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[346]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[352]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[358]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[364]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[370]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[376]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[382]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[388]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[394]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[400]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[406]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[412]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[418]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[424]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[430]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[436]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[442]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[448]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[454]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[460]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[466]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[472]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[478]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[484]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[490]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[496]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[502]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[508]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[514]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[520]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[526]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[532]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[538]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[544]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[550]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[556]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[562]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[568]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[574]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[580]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[586]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[592]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[598]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[604]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[610]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[616]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[622]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[628]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[634]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[640]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[646]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[652]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[658]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[664]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[670]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[676]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[682]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[688]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[694]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[700]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[706]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[712]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[718]*acadoWorkspace.QDy[125];
acadoWorkspace.g[5] = + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[323]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[329]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[335]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[341]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[347]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[353]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[359]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[365]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[371]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[377]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[383]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[389]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[395]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[401]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[407]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[413]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[419]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[425]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[431]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[437]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[443]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[449]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[455]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[461]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[467]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[473]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[479]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[485]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[491]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[497]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[503]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[509]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[515]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[521]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[527]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[533]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[539]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[545]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[551]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[557]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[563]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[569]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[575]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[581]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[587]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[593]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[599]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[605]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[611]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[617]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[623]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[629]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[635]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[641]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[647]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[653]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[659]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[665]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[671]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[677]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[683]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[689]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[695]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[701]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[707]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[713]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[719]*acadoWorkspace.QDy[125];


acado_multEQDy( acadoWorkspace.E, &(acadoWorkspace.QDy[ 6 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.QDy[ 12 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.QDy[ 18 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.QDy[ 24 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.QDy[ 30 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.QDy[ 36 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.QDy[ 42 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.QDy[ 48 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.QDy[ 54 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.QDy[ 60 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.QDy[ 66 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.QDy[ 72 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.QDy[ 78 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.QDy[ 84 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 714 ]), &(acadoWorkspace.QDy[ 90 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.QDy[ 96 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.QDy[ 102 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.QDy[ 108 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1134 ]), &(acadoWorkspace.QDy[ 114 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multEQDy( &(acadoWorkspace.E[ 1254 ]), &(acadoWorkspace.QDy[ 120 ]), &(acadoWorkspace.g[ 25 ]) );

acadoWorkspace.lb[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.lb[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.lb[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.lb[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.lb[4] = acadoWorkspace.Dx0[4];
acadoWorkspace.lb[5] = acadoWorkspace.Dx0[5];
acadoWorkspace.ub[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.ub[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.ub[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.ub[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.ub[4] = acadoWorkspace.Dx0[4];
acadoWorkspace.ub[5] = acadoWorkspace.Dx0[5];
tmp = acadoVariables.x[7] + acadoWorkspace.d[1];
acadoWorkspace.lbA[0] = - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[13] + acadoWorkspace.d[7];
acadoWorkspace.lbA[1] = - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[19] + acadoWorkspace.d[13];
acadoWorkspace.lbA[2] = - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[25] + acadoWorkspace.d[19];
acadoWorkspace.lbA[3] = - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[25];
acadoWorkspace.lbA[4] = - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[37] + acadoWorkspace.d[31];
acadoWorkspace.lbA[5] = - tmp;
acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[43] + acadoWorkspace.d[37];
acadoWorkspace.lbA[6] = - tmp;
acadoWorkspace.ubA[6] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[49] + acadoWorkspace.d[43];
acadoWorkspace.lbA[7] = - tmp;
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[55] + acadoWorkspace.d[49];
acadoWorkspace.lbA[8] = - tmp;
acadoWorkspace.ubA[8] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[61] + acadoWorkspace.d[55];
acadoWorkspace.lbA[9] = - tmp;
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[67] + acadoWorkspace.d[61];
acadoWorkspace.lbA[10] = - tmp;
acadoWorkspace.ubA[10] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[73] + acadoWorkspace.d[67];
acadoWorkspace.lbA[11] = - tmp;
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[79] + acadoWorkspace.d[73];
acadoWorkspace.lbA[12] = - tmp;
acadoWorkspace.ubA[12] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[85] + acadoWorkspace.d[79];
acadoWorkspace.lbA[13] = - tmp;
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[91] + acadoWorkspace.d[85];
acadoWorkspace.lbA[14] = - tmp;
acadoWorkspace.ubA[14] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[97] + acadoWorkspace.d[91];
acadoWorkspace.lbA[15] = - tmp;
acadoWorkspace.ubA[15] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[103] + acadoWorkspace.d[97];
acadoWorkspace.lbA[16] = - tmp;
acadoWorkspace.ubA[16] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[109] + acadoWorkspace.d[103];
acadoWorkspace.lbA[17] = - tmp;
acadoWorkspace.ubA[17] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[115] + acadoWorkspace.d[109];
acadoWorkspace.lbA[18] = - tmp;
acadoWorkspace.ubA[18] = (real_t)1.0000000000000000e+12 - tmp;
tmp = acadoVariables.x[121] + acadoWorkspace.d[115];
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
acadoVariables.x[5] += acadoWorkspace.x[5];

acadoVariables.u[0] += acadoWorkspace.x[6];
acadoVariables.u[1] += acadoWorkspace.x[7];
acadoVariables.u[2] += acadoWorkspace.x[8];
acadoVariables.u[3] += acadoWorkspace.x[9];
acadoVariables.u[4] += acadoWorkspace.x[10];
acadoVariables.u[5] += acadoWorkspace.x[11];
acadoVariables.u[6] += acadoWorkspace.x[12];
acadoVariables.u[7] += acadoWorkspace.x[13];
acadoVariables.u[8] += acadoWorkspace.x[14];
acadoVariables.u[9] += acadoWorkspace.x[15];
acadoVariables.u[10] += acadoWorkspace.x[16];
acadoVariables.u[11] += acadoWorkspace.x[17];
acadoVariables.u[12] += acadoWorkspace.x[18];
acadoVariables.u[13] += acadoWorkspace.x[19];
acadoVariables.u[14] += acadoWorkspace.x[20];
acadoVariables.u[15] += acadoWorkspace.x[21];
acadoVariables.u[16] += acadoWorkspace.x[22];
acadoVariables.u[17] += acadoWorkspace.x[23];
acadoVariables.u[18] += acadoWorkspace.x[24];
acadoVariables.u[19] += acadoWorkspace.x[25];

acadoVariables.x[6] += + acadoWorkspace.evGx[0]*acadoWorkspace.x[0] + acadoWorkspace.evGx[1]*acadoWorkspace.x[1] + acadoWorkspace.evGx[2]*acadoWorkspace.x[2] + acadoWorkspace.evGx[3]*acadoWorkspace.x[3] + acadoWorkspace.evGx[4]*acadoWorkspace.x[4] + acadoWorkspace.evGx[5]*acadoWorkspace.x[5] + acadoWorkspace.d[0];
acadoVariables.x[7] += + acadoWorkspace.evGx[6]*acadoWorkspace.x[0] + acadoWorkspace.evGx[7]*acadoWorkspace.x[1] + acadoWorkspace.evGx[8]*acadoWorkspace.x[2] + acadoWorkspace.evGx[9]*acadoWorkspace.x[3] + acadoWorkspace.evGx[10]*acadoWorkspace.x[4] + acadoWorkspace.evGx[11]*acadoWorkspace.x[5] + acadoWorkspace.d[1];
acadoVariables.x[8] += + acadoWorkspace.evGx[12]*acadoWorkspace.x[0] + acadoWorkspace.evGx[13]*acadoWorkspace.x[1] + acadoWorkspace.evGx[14]*acadoWorkspace.x[2] + acadoWorkspace.evGx[15]*acadoWorkspace.x[3] + acadoWorkspace.evGx[16]*acadoWorkspace.x[4] + acadoWorkspace.evGx[17]*acadoWorkspace.x[5] + acadoWorkspace.d[2];
acadoVariables.x[9] += + acadoWorkspace.evGx[18]*acadoWorkspace.x[0] + acadoWorkspace.evGx[19]*acadoWorkspace.x[1] + acadoWorkspace.evGx[20]*acadoWorkspace.x[2] + acadoWorkspace.evGx[21]*acadoWorkspace.x[3] + acadoWorkspace.evGx[22]*acadoWorkspace.x[4] + acadoWorkspace.evGx[23]*acadoWorkspace.x[5] + acadoWorkspace.d[3];
acadoVariables.x[10] += + acadoWorkspace.evGx[24]*acadoWorkspace.x[0] + acadoWorkspace.evGx[25]*acadoWorkspace.x[1] + acadoWorkspace.evGx[26]*acadoWorkspace.x[2] + acadoWorkspace.evGx[27]*acadoWorkspace.x[3] + acadoWorkspace.evGx[28]*acadoWorkspace.x[4] + acadoWorkspace.evGx[29]*acadoWorkspace.x[5] + acadoWorkspace.d[4];
acadoVariables.x[11] += + acadoWorkspace.evGx[30]*acadoWorkspace.x[0] + acadoWorkspace.evGx[31]*acadoWorkspace.x[1] + acadoWorkspace.evGx[32]*acadoWorkspace.x[2] + acadoWorkspace.evGx[33]*acadoWorkspace.x[3] + acadoWorkspace.evGx[34]*acadoWorkspace.x[4] + acadoWorkspace.evGx[35]*acadoWorkspace.x[5] + acadoWorkspace.d[5];
acadoVariables.x[12] += + acadoWorkspace.evGx[36]*acadoWorkspace.x[0] + acadoWorkspace.evGx[37]*acadoWorkspace.x[1] + acadoWorkspace.evGx[38]*acadoWorkspace.x[2] + acadoWorkspace.evGx[39]*acadoWorkspace.x[3] + acadoWorkspace.evGx[40]*acadoWorkspace.x[4] + acadoWorkspace.evGx[41]*acadoWorkspace.x[5] + acadoWorkspace.d[6];
acadoVariables.x[13] += + acadoWorkspace.evGx[42]*acadoWorkspace.x[0] + acadoWorkspace.evGx[43]*acadoWorkspace.x[1] + acadoWorkspace.evGx[44]*acadoWorkspace.x[2] + acadoWorkspace.evGx[45]*acadoWorkspace.x[3] + acadoWorkspace.evGx[46]*acadoWorkspace.x[4] + acadoWorkspace.evGx[47]*acadoWorkspace.x[5] + acadoWorkspace.d[7];
acadoVariables.x[14] += + acadoWorkspace.evGx[48]*acadoWorkspace.x[0] + acadoWorkspace.evGx[49]*acadoWorkspace.x[1] + acadoWorkspace.evGx[50]*acadoWorkspace.x[2] + acadoWorkspace.evGx[51]*acadoWorkspace.x[3] + acadoWorkspace.evGx[52]*acadoWorkspace.x[4] + acadoWorkspace.evGx[53]*acadoWorkspace.x[5] + acadoWorkspace.d[8];
acadoVariables.x[15] += + acadoWorkspace.evGx[54]*acadoWorkspace.x[0] + acadoWorkspace.evGx[55]*acadoWorkspace.x[1] + acadoWorkspace.evGx[56]*acadoWorkspace.x[2] + acadoWorkspace.evGx[57]*acadoWorkspace.x[3] + acadoWorkspace.evGx[58]*acadoWorkspace.x[4] + acadoWorkspace.evGx[59]*acadoWorkspace.x[5] + acadoWorkspace.d[9];
acadoVariables.x[16] += + acadoWorkspace.evGx[60]*acadoWorkspace.x[0] + acadoWorkspace.evGx[61]*acadoWorkspace.x[1] + acadoWorkspace.evGx[62]*acadoWorkspace.x[2] + acadoWorkspace.evGx[63]*acadoWorkspace.x[3] + acadoWorkspace.evGx[64]*acadoWorkspace.x[4] + acadoWorkspace.evGx[65]*acadoWorkspace.x[5] + acadoWorkspace.d[10];
acadoVariables.x[17] += + acadoWorkspace.evGx[66]*acadoWorkspace.x[0] + acadoWorkspace.evGx[67]*acadoWorkspace.x[1] + acadoWorkspace.evGx[68]*acadoWorkspace.x[2] + acadoWorkspace.evGx[69]*acadoWorkspace.x[3] + acadoWorkspace.evGx[70]*acadoWorkspace.x[4] + acadoWorkspace.evGx[71]*acadoWorkspace.x[5] + acadoWorkspace.d[11];
acadoVariables.x[18] += + acadoWorkspace.evGx[72]*acadoWorkspace.x[0] + acadoWorkspace.evGx[73]*acadoWorkspace.x[1] + acadoWorkspace.evGx[74]*acadoWorkspace.x[2] + acadoWorkspace.evGx[75]*acadoWorkspace.x[3] + acadoWorkspace.evGx[76]*acadoWorkspace.x[4] + acadoWorkspace.evGx[77]*acadoWorkspace.x[5] + acadoWorkspace.d[12];
acadoVariables.x[19] += + acadoWorkspace.evGx[78]*acadoWorkspace.x[0] + acadoWorkspace.evGx[79]*acadoWorkspace.x[1] + acadoWorkspace.evGx[80]*acadoWorkspace.x[2] + acadoWorkspace.evGx[81]*acadoWorkspace.x[3] + acadoWorkspace.evGx[82]*acadoWorkspace.x[4] + acadoWorkspace.evGx[83]*acadoWorkspace.x[5] + acadoWorkspace.d[13];
acadoVariables.x[20] += + acadoWorkspace.evGx[84]*acadoWorkspace.x[0] + acadoWorkspace.evGx[85]*acadoWorkspace.x[1] + acadoWorkspace.evGx[86]*acadoWorkspace.x[2] + acadoWorkspace.evGx[87]*acadoWorkspace.x[3] + acadoWorkspace.evGx[88]*acadoWorkspace.x[4] + acadoWorkspace.evGx[89]*acadoWorkspace.x[5] + acadoWorkspace.d[14];
acadoVariables.x[21] += + acadoWorkspace.evGx[90]*acadoWorkspace.x[0] + acadoWorkspace.evGx[91]*acadoWorkspace.x[1] + acadoWorkspace.evGx[92]*acadoWorkspace.x[2] + acadoWorkspace.evGx[93]*acadoWorkspace.x[3] + acadoWorkspace.evGx[94]*acadoWorkspace.x[4] + acadoWorkspace.evGx[95]*acadoWorkspace.x[5] + acadoWorkspace.d[15];
acadoVariables.x[22] += + acadoWorkspace.evGx[96]*acadoWorkspace.x[0] + acadoWorkspace.evGx[97]*acadoWorkspace.x[1] + acadoWorkspace.evGx[98]*acadoWorkspace.x[2] + acadoWorkspace.evGx[99]*acadoWorkspace.x[3] + acadoWorkspace.evGx[100]*acadoWorkspace.x[4] + acadoWorkspace.evGx[101]*acadoWorkspace.x[5] + acadoWorkspace.d[16];
acadoVariables.x[23] += + acadoWorkspace.evGx[102]*acadoWorkspace.x[0] + acadoWorkspace.evGx[103]*acadoWorkspace.x[1] + acadoWorkspace.evGx[104]*acadoWorkspace.x[2] + acadoWorkspace.evGx[105]*acadoWorkspace.x[3] + acadoWorkspace.evGx[106]*acadoWorkspace.x[4] + acadoWorkspace.evGx[107]*acadoWorkspace.x[5] + acadoWorkspace.d[17];
acadoVariables.x[24] += + acadoWorkspace.evGx[108]*acadoWorkspace.x[0] + acadoWorkspace.evGx[109]*acadoWorkspace.x[1] + acadoWorkspace.evGx[110]*acadoWorkspace.x[2] + acadoWorkspace.evGx[111]*acadoWorkspace.x[3] + acadoWorkspace.evGx[112]*acadoWorkspace.x[4] + acadoWorkspace.evGx[113]*acadoWorkspace.x[5] + acadoWorkspace.d[18];
acadoVariables.x[25] += + acadoWorkspace.evGx[114]*acadoWorkspace.x[0] + acadoWorkspace.evGx[115]*acadoWorkspace.x[1] + acadoWorkspace.evGx[116]*acadoWorkspace.x[2] + acadoWorkspace.evGx[117]*acadoWorkspace.x[3] + acadoWorkspace.evGx[118]*acadoWorkspace.x[4] + acadoWorkspace.evGx[119]*acadoWorkspace.x[5] + acadoWorkspace.d[19];
acadoVariables.x[26] += + acadoWorkspace.evGx[120]*acadoWorkspace.x[0] + acadoWorkspace.evGx[121]*acadoWorkspace.x[1] + acadoWorkspace.evGx[122]*acadoWorkspace.x[2] + acadoWorkspace.evGx[123]*acadoWorkspace.x[3] + acadoWorkspace.evGx[124]*acadoWorkspace.x[4] + acadoWorkspace.evGx[125]*acadoWorkspace.x[5] + acadoWorkspace.d[20];
acadoVariables.x[27] += + acadoWorkspace.evGx[126]*acadoWorkspace.x[0] + acadoWorkspace.evGx[127]*acadoWorkspace.x[1] + acadoWorkspace.evGx[128]*acadoWorkspace.x[2] + acadoWorkspace.evGx[129]*acadoWorkspace.x[3] + acadoWorkspace.evGx[130]*acadoWorkspace.x[4] + acadoWorkspace.evGx[131]*acadoWorkspace.x[5] + acadoWorkspace.d[21];
acadoVariables.x[28] += + acadoWorkspace.evGx[132]*acadoWorkspace.x[0] + acadoWorkspace.evGx[133]*acadoWorkspace.x[1] + acadoWorkspace.evGx[134]*acadoWorkspace.x[2] + acadoWorkspace.evGx[135]*acadoWorkspace.x[3] + acadoWorkspace.evGx[136]*acadoWorkspace.x[4] + acadoWorkspace.evGx[137]*acadoWorkspace.x[5] + acadoWorkspace.d[22];
acadoVariables.x[29] += + acadoWorkspace.evGx[138]*acadoWorkspace.x[0] + acadoWorkspace.evGx[139]*acadoWorkspace.x[1] + acadoWorkspace.evGx[140]*acadoWorkspace.x[2] + acadoWorkspace.evGx[141]*acadoWorkspace.x[3] + acadoWorkspace.evGx[142]*acadoWorkspace.x[4] + acadoWorkspace.evGx[143]*acadoWorkspace.x[5] + acadoWorkspace.d[23];
acadoVariables.x[30] += + acadoWorkspace.evGx[144]*acadoWorkspace.x[0] + acadoWorkspace.evGx[145]*acadoWorkspace.x[1] + acadoWorkspace.evGx[146]*acadoWorkspace.x[2] + acadoWorkspace.evGx[147]*acadoWorkspace.x[3] + acadoWorkspace.evGx[148]*acadoWorkspace.x[4] + acadoWorkspace.evGx[149]*acadoWorkspace.x[5] + acadoWorkspace.d[24];
acadoVariables.x[31] += + acadoWorkspace.evGx[150]*acadoWorkspace.x[0] + acadoWorkspace.evGx[151]*acadoWorkspace.x[1] + acadoWorkspace.evGx[152]*acadoWorkspace.x[2] + acadoWorkspace.evGx[153]*acadoWorkspace.x[3] + acadoWorkspace.evGx[154]*acadoWorkspace.x[4] + acadoWorkspace.evGx[155]*acadoWorkspace.x[5] + acadoWorkspace.d[25];
acadoVariables.x[32] += + acadoWorkspace.evGx[156]*acadoWorkspace.x[0] + acadoWorkspace.evGx[157]*acadoWorkspace.x[1] + acadoWorkspace.evGx[158]*acadoWorkspace.x[2] + acadoWorkspace.evGx[159]*acadoWorkspace.x[3] + acadoWorkspace.evGx[160]*acadoWorkspace.x[4] + acadoWorkspace.evGx[161]*acadoWorkspace.x[5] + acadoWorkspace.d[26];
acadoVariables.x[33] += + acadoWorkspace.evGx[162]*acadoWorkspace.x[0] + acadoWorkspace.evGx[163]*acadoWorkspace.x[1] + acadoWorkspace.evGx[164]*acadoWorkspace.x[2] + acadoWorkspace.evGx[165]*acadoWorkspace.x[3] + acadoWorkspace.evGx[166]*acadoWorkspace.x[4] + acadoWorkspace.evGx[167]*acadoWorkspace.x[5] + acadoWorkspace.d[27];
acadoVariables.x[34] += + acadoWorkspace.evGx[168]*acadoWorkspace.x[0] + acadoWorkspace.evGx[169]*acadoWorkspace.x[1] + acadoWorkspace.evGx[170]*acadoWorkspace.x[2] + acadoWorkspace.evGx[171]*acadoWorkspace.x[3] + acadoWorkspace.evGx[172]*acadoWorkspace.x[4] + acadoWorkspace.evGx[173]*acadoWorkspace.x[5] + acadoWorkspace.d[28];
acadoVariables.x[35] += + acadoWorkspace.evGx[174]*acadoWorkspace.x[0] + acadoWorkspace.evGx[175]*acadoWorkspace.x[1] + acadoWorkspace.evGx[176]*acadoWorkspace.x[2] + acadoWorkspace.evGx[177]*acadoWorkspace.x[3] + acadoWorkspace.evGx[178]*acadoWorkspace.x[4] + acadoWorkspace.evGx[179]*acadoWorkspace.x[5] + acadoWorkspace.d[29];
acadoVariables.x[36] += + acadoWorkspace.evGx[180]*acadoWorkspace.x[0] + acadoWorkspace.evGx[181]*acadoWorkspace.x[1] + acadoWorkspace.evGx[182]*acadoWorkspace.x[2] + acadoWorkspace.evGx[183]*acadoWorkspace.x[3] + acadoWorkspace.evGx[184]*acadoWorkspace.x[4] + acadoWorkspace.evGx[185]*acadoWorkspace.x[5] + acadoWorkspace.d[30];
acadoVariables.x[37] += + acadoWorkspace.evGx[186]*acadoWorkspace.x[0] + acadoWorkspace.evGx[187]*acadoWorkspace.x[1] + acadoWorkspace.evGx[188]*acadoWorkspace.x[2] + acadoWorkspace.evGx[189]*acadoWorkspace.x[3] + acadoWorkspace.evGx[190]*acadoWorkspace.x[4] + acadoWorkspace.evGx[191]*acadoWorkspace.x[5] + acadoWorkspace.d[31];
acadoVariables.x[38] += + acadoWorkspace.evGx[192]*acadoWorkspace.x[0] + acadoWorkspace.evGx[193]*acadoWorkspace.x[1] + acadoWorkspace.evGx[194]*acadoWorkspace.x[2] + acadoWorkspace.evGx[195]*acadoWorkspace.x[3] + acadoWorkspace.evGx[196]*acadoWorkspace.x[4] + acadoWorkspace.evGx[197]*acadoWorkspace.x[5] + acadoWorkspace.d[32];
acadoVariables.x[39] += + acadoWorkspace.evGx[198]*acadoWorkspace.x[0] + acadoWorkspace.evGx[199]*acadoWorkspace.x[1] + acadoWorkspace.evGx[200]*acadoWorkspace.x[2] + acadoWorkspace.evGx[201]*acadoWorkspace.x[3] + acadoWorkspace.evGx[202]*acadoWorkspace.x[4] + acadoWorkspace.evGx[203]*acadoWorkspace.x[5] + acadoWorkspace.d[33];
acadoVariables.x[40] += + acadoWorkspace.evGx[204]*acadoWorkspace.x[0] + acadoWorkspace.evGx[205]*acadoWorkspace.x[1] + acadoWorkspace.evGx[206]*acadoWorkspace.x[2] + acadoWorkspace.evGx[207]*acadoWorkspace.x[3] + acadoWorkspace.evGx[208]*acadoWorkspace.x[4] + acadoWorkspace.evGx[209]*acadoWorkspace.x[5] + acadoWorkspace.d[34];
acadoVariables.x[41] += + acadoWorkspace.evGx[210]*acadoWorkspace.x[0] + acadoWorkspace.evGx[211]*acadoWorkspace.x[1] + acadoWorkspace.evGx[212]*acadoWorkspace.x[2] + acadoWorkspace.evGx[213]*acadoWorkspace.x[3] + acadoWorkspace.evGx[214]*acadoWorkspace.x[4] + acadoWorkspace.evGx[215]*acadoWorkspace.x[5] + acadoWorkspace.d[35];
acadoVariables.x[42] += + acadoWorkspace.evGx[216]*acadoWorkspace.x[0] + acadoWorkspace.evGx[217]*acadoWorkspace.x[1] + acadoWorkspace.evGx[218]*acadoWorkspace.x[2] + acadoWorkspace.evGx[219]*acadoWorkspace.x[3] + acadoWorkspace.evGx[220]*acadoWorkspace.x[4] + acadoWorkspace.evGx[221]*acadoWorkspace.x[5] + acadoWorkspace.d[36];
acadoVariables.x[43] += + acadoWorkspace.evGx[222]*acadoWorkspace.x[0] + acadoWorkspace.evGx[223]*acadoWorkspace.x[1] + acadoWorkspace.evGx[224]*acadoWorkspace.x[2] + acadoWorkspace.evGx[225]*acadoWorkspace.x[3] + acadoWorkspace.evGx[226]*acadoWorkspace.x[4] + acadoWorkspace.evGx[227]*acadoWorkspace.x[5] + acadoWorkspace.d[37];
acadoVariables.x[44] += + acadoWorkspace.evGx[228]*acadoWorkspace.x[0] + acadoWorkspace.evGx[229]*acadoWorkspace.x[1] + acadoWorkspace.evGx[230]*acadoWorkspace.x[2] + acadoWorkspace.evGx[231]*acadoWorkspace.x[3] + acadoWorkspace.evGx[232]*acadoWorkspace.x[4] + acadoWorkspace.evGx[233]*acadoWorkspace.x[5] + acadoWorkspace.d[38];
acadoVariables.x[45] += + acadoWorkspace.evGx[234]*acadoWorkspace.x[0] + acadoWorkspace.evGx[235]*acadoWorkspace.x[1] + acadoWorkspace.evGx[236]*acadoWorkspace.x[2] + acadoWorkspace.evGx[237]*acadoWorkspace.x[3] + acadoWorkspace.evGx[238]*acadoWorkspace.x[4] + acadoWorkspace.evGx[239]*acadoWorkspace.x[5] + acadoWorkspace.d[39];
acadoVariables.x[46] += + acadoWorkspace.evGx[240]*acadoWorkspace.x[0] + acadoWorkspace.evGx[241]*acadoWorkspace.x[1] + acadoWorkspace.evGx[242]*acadoWorkspace.x[2] + acadoWorkspace.evGx[243]*acadoWorkspace.x[3] + acadoWorkspace.evGx[244]*acadoWorkspace.x[4] + acadoWorkspace.evGx[245]*acadoWorkspace.x[5] + acadoWorkspace.d[40];
acadoVariables.x[47] += + acadoWorkspace.evGx[246]*acadoWorkspace.x[0] + acadoWorkspace.evGx[247]*acadoWorkspace.x[1] + acadoWorkspace.evGx[248]*acadoWorkspace.x[2] + acadoWorkspace.evGx[249]*acadoWorkspace.x[3] + acadoWorkspace.evGx[250]*acadoWorkspace.x[4] + acadoWorkspace.evGx[251]*acadoWorkspace.x[5] + acadoWorkspace.d[41];
acadoVariables.x[48] += + acadoWorkspace.evGx[252]*acadoWorkspace.x[0] + acadoWorkspace.evGx[253]*acadoWorkspace.x[1] + acadoWorkspace.evGx[254]*acadoWorkspace.x[2] + acadoWorkspace.evGx[255]*acadoWorkspace.x[3] + acadoWorkspace.evGx[256]*acadoWorkspace.x[4] + acadoWorkspace.evGx[257]*acadoWorkspace.x[5] + acadoWorkspace.d[42];
acadoVariables.x[49] += + acadoWorkspace.evGx[258]*acadoWorkspace.x[0] + acadoWorkspace.evGx[259]*acadoWorkspace.x[1] + acadoWorkspace.evGx[260]*acadoWorkspace.x[2] + acadoWorkspace.evGx[261]*acadoWorkspace.x[3] + acadoWorkspace.evGx[262]*acadoWorkspace.x[4] + acadoWorkspace.evGx[263]*acadoWorkspace.x[5] + acadoWorkspace.d[43];
acadoVariables.x[50] += + acadoWorkspace.evGx[264]*acadoWorkspace.x[0] + acadoWorkspace.evGx[265]*acadoWorkspace.x[1] + acadoWorkspace.evGx[266]*acadoWorkspace.x[2] + acadoWorkspace.evGx[267]*acadoWorkspace.x[3] + acadoWorkspace.evGx[268]*acadoWorkspace.x[4] + acadoWorkspace.evGx[269]*acadoWorkspace.x[5] + acadoWorkspace.d[44];
acadoVariables.x[51] += + acadoWorkspace.evGx[270]*acadoWorkspace.x[0] + acadoWorkspace.evGx[271]*acadoWorkspace.x[1] + acadoWorkspace.evGx[272]*acadoWorkspace.x[2] + acadoWorkspace.evGx[273]*acadoWorkspace.x[3] + acadoWorkspace.evGx[274]*acadoWorkspace.x[4] + acadoWorkspace.evGx[275]*acadoWorkspace.x[5] + acadoWorkspace.d[45];
acadoVariables.x[52] += + acadoWorkspace.evGx[276]*acadoWorkspace.x[0] + acadoWorkspace.evGx[277]*acadoWorkspace.x[1] + acadoWorkspace.evGx[278]*acadoWorkspace.x[2] + acadoWorkspace.evGx[279]*acadoWorkspace.x[3] + acadoWorkspace.evGx[280]*acadoWorkspace.x[4] + acadoWorkspace.evGx[281]*acadoWorkspace.x[5] + acadoWorkspace.d[46];
acadoVariables.x[53] += + acadoWorkspace.evGx[282]*acadoWorkspace.x[0] + acadoWorkspace.evGx[283]*acadoWorkspace.x[1] + acadoWorkspace.evGx[284]*acadoWorkspace.x[2] + acadoWorkspace.evGx[285]*acadoWorkspace.x[3] + acadoWorkspace.evGx[286]*acadoWorkspace.x[4] + acadoWorkspace.evGx[287]*acadoWorkspace.x[5] + acadoWorkspace.d[47];
acadoVariables.x[54] += + acadoWorkspace.evGx[288]*acadoWorkspace.x[0] + acadoWorkspace.evGx[289]*acadoWorkspace.x[1] + acadoWorkspace.evGx[290]*acadoWorkspace.x[2] + acadoWorkspace.evGx[291]*acadoWorkspace.x[3] + acadoWorkspace.evGx[292]*acadoWorkspace.x[4] + acadoWorkspace.evGx[293]*acadoWorkspace.x[5] + acadoWorkspace.d[48];
acadoVariables.x[55] += + acadoWorkspace.evGx[294]*acadoWorkspace.x[0] + acadoWorkspace.evGx[295]*acadoWorkspace.x[1] + acadoWorkspace.evGx[296]*acadoWorkspace.x[2] + acadoWorkspace.evGx[297]*acadoWorkspace.x[3] + acadoWorkspace.evGx[298]*acadoWorkspace.x[4] + acadoWorkspace.evGx[299]*acadoWorkspace.x[5] + acadoWorkspace.d[49];
acadoVariables.x[56] += + acadoWorkspace.evGx[300]*acadoWorkspace.x[0] + acadoWorkspace.evGx[301]*acadoWorkspace.x[1] + acadoWorkspace.evGx[302]*acadoWorkspace.x[2] + acadoWorkspace.evGx[303]*acadoWorkspace.x[3] + acadoWorkspace.evGx[304]*acadoWorkspace.x[4] + acadoWorkspace.evGx[305]*acadoWorkspace.x[5] + acadoWorkspace.d[50];
acadoVariables.x[57] += + acadoWorkspace.evGx[306]*acadoWorkspace.x[0] + acadoWorkspace.evGx[307]*acadoWorkspace.x[1] + acadoWorkspace.evGx[308]*acadoWorkspace.x[2] + acadoWorkspace.evGx[309]*acadoWorkspace.x[3] + acadoWorkspace.evGx[310]*acadoWorkspace.x[4] + acadoWorkspace.evGx[311]*acadoWorkspace.x[5] + acadoWorkspace.d[51];
acadoVariables.x[58] += + acadoWorkspace.evGx[312]*acadoWorkspace.x[0] + acadoWorkspace.evGx[313]*acadoWorkspace.x[1] + acadoWorkspace.evGx[314]*acadoWorkspace.x[2] + acadoWorkspace.evGx[315]*acadoWorkspace.x[3] + acadoWorkspace.evGx[316]*acadoWorkspace.x[4] + acadoWorkspace.evGx[317]*acadoWorkspace.x[5] + acadoWorkspace.d[52];
acadoVariables.x[59] += + acadoWorkspace.evGx[318]*acadoWorkspace.x[0] + acadoWorkspace.evGx[319]*acadoWorkspace.x[1] + acadoWorkspace.evGx[320]*acadoWorkspace.x[2] + acadoWorkspace.evGx[321]*acadoWorkspace.x[3] + acadoWorkspace.evGx[322]*acadoWorkspace.x[4] + acadoWorkspace.evGx[323]*acadoWorkspace.x[5] + acadoWorkspace.d[53];
acadoVariables.x[60] += + acadoWorkspace.evGx[324]*acadoWorkspace.x[0] + acadoWorkspace.evGx[325]*acadoWorkspace.x[1] + acadoWorkspace.evGx[326]*acadoWorkspace.x[2] + acadoWorkspace.evGx[327]*acadoWorkspace.x[3] + acadoWorkspace.evGx[328]*acadoWorkspace.x[4] + acadoWorkspace.evGx[329]*acadoWorkspace.x[5] + acadoWorkspace.d[54];
acadoVariables.x[61] += + acadoWorkspace.evGx[330]*acadoWorkspace.x[0] + acadoWorkspace.evGx[331]*acadoWorkspace.x[1] + acadoWorkspace.evGx[332]*acadoWorkspace.x[2] + acadoWorkspace.evGx[333]*acadoWorkspace.x[3] + acadoWorkspace.evGx[334]*acadoWorkspace.x[4] + acadoWorkspace.evGx[335]*acadoWorkspace.x[5] + acadoWorkspace.d[55];
acadoVariables.x[62] += + acadoWorkspace.evGx[336]*acadoWorkspace.x[0] + acadoWorkspace.evGx[337]*acadoWorkspace.x[1] + acadoWorkspace.evGx[338]*acadoWorkspace.x[2] + acadoWorkspace.evGx[339]*acadoWorkspace.x[3] + acadoWorkspace.evGx[340]*acadoWorkspace.x[4] + acadoWorkspace.evGx[341]*acadoWorkspace.x[5] + acadoWorkspace.d[56];
acadoVariables.x[63] += + acadoWorkspace.evGx[342]*acadoWorkspace.x[0] + acadoWorkspace.evGx[343]*acadoWorkspace.x[1] + acadoWorkspace.evGx[344]*acadoWorkspace.x[2] + acadoWorkspace.evGx[345]*acadoWorkspace.x[3] + acadoWorkspace.evGx[346]*acadoWorkspace.x[4] + acadoWorkspace.evGx[347]*acadoWorkspace.x[5] + acadoWorkspace.d[57];
acadoVariables.x[64] += + acadoWorkspace.evGx[348]*acadoWorkspace.x[0] + acadoWorkspace.evGx[349]*acadoWorkspace.x[1] + acadoWorkspace.evGx[350]*acadoWorkspace.x[2] + acadoWorkspace.evGx[351]*acadoWorkspace.x[3] + acadoWorkspace.evGx[352]*acadoWorkspace.x[4] + acadoWorkspace.evGx[353]*acadoWorkspace.x[5] + acadoWorkspace.d[58];
acadoVariables.x[65] += + acadoWorkspace.evGx[354]*acadoWorkspace.x[0] + acadoWorkspace.evGx[355]*acadoWorkspace.x[1] + acadoWorkspace.evGx[356]*acadoWorkspace.x[2] + acadoWorkspace.evGx[357]*acadoWorkspace.x[3] + acadoWorkspace.evGx[358]*acadoWorkspace.x[4] + acadoWorkspace.evGx[359]*acadoWorkspace.x[5] + acadoWorkspace.d[59];
acadoVariables.x[66] += + acadoWorkspace.evGx[360]*acadoWorkspace.x[0] + acadoWorkspace.evGx[361]*acadoWorkspace.x[1] + acadoWorkspace.evGx[362]*acadoWorkspace.x[2] + acadoWorkspace.evGx[363]*acadoWorkspace.x[3] + acadoWorkspace.evGx[364]*acadoWorkspace.x[4] + acadoWorkspace.evGx[365]*acadoWorkspace.x[5] + acadoWorkspace.d[60];
acadoVariables.x[67] += + acadoWorkspace.evGx[366]*acadoWorkspace.x[0] + acadoWorkspace.evGx[367]*acadoWorkspace.x[1] + acadoWorkspace.evGx[368]*acadoWorkspace.x[2] + acadoWorkspace.evGx[369]*acadoWorkspace.x[3] + acadoWorkspace.evGx[370]*acadoWorkspace.x[4] + acadoWorkspace.evGx[371]*acadoWorkspace.x[5] + acadoWorkspace.d[61];
acadoVariables.x[68] += + acadoWorkspace.evGx[372]*acadoWorkspace.x[0] + acadoWorkspace.evGx[373]*acadoWorkspace.x[1] + acadoWorkspace.evGx[374]*acadoWorkspace.x[2] + acadoWorkspace.evGx[375]*acadoWorkspace.x[3] + acadoWorkspace.evGx[376]*acadoWorkspace.x[4] + acadoWorkspace.evGx[377]*acadoWorkspace.x[5] + acadoWorkspace.d[62];
acadoVariables.x[69] += + acadoWorkspace.evGx[378]*acadoWorkspace.x[0] + acadoWorkspace.evGx[379]*acadoWorkspace.x[1] + acadoWorkspace.evGx[380]*acadoWorkspace.x[2] + acadoWorkspace.evGx[381]*acadoWorkspace.x[3] + acadoWorkspace.evGx[382]*acadoWorkspace.x[4] + acadoWorkspace.evGx[383]*acadoWorkspace.x[5] + acadoWorkspace.d[63];
acadoVariables.x[70] += + acadoWorkspace.evGx[384]*acadoWorkspace.x[0] + acadoWorkspace.evGx[385]*acadoWorkspace.x[1] + acadoWorkspace.evGx[386]*acadoWorkspace.x[2] + acadoWorkspace.evGx[387]*acadoWorkspace.x[3] + acadoWorkspace.evGx[388]*acadoWorkspace.x[4] + acadoWorkspace.evGx[389]*acadoWorkspace.x[5] + acadoWorkspace.d[64];
acadoVariables.x[71] += + acadoWorkspace.evGx[390]*acadoWorkspace.x[0] + acadoWorkspace.evGx[391]*acadoWorkspace.x[1] + acadoWorkspace.evGx[392]*acadoWorkspace.x[2] + acadoWorkspace.evGx[393]*acadoWorkspace.x[3] + acadoWorkspace.evGx[394]*acadoWorkspace.x[4] + acadoWorkspace.evGx[395]*acadoWorkspace.x[5] + acadoWorkspace.d[65];
acadoVariables.x[72] += + acadoWorkspace.evGx[396]*acadoWorkspace.x[0] + acadoWorkspace.evGx[397]*acadoWorkspace.x[1] + acadoWorkspace.evGx[398]*acadoWorkspace.x[2] + acadoWorkspace.evGx[399]*acadoWorkspace.x[3] + acadoWorkspace.evGx[400]*acadoWorkspace.x[4] + acadoWorkspace.evGx[401]*acadoWorkspace.x[5] + acadoWorkspace.d[66];
acadoVariables.x[73] += + acadoWorkspace.evGx[402]*acadoWorkspace.x[0] + acadoWorkspace.evGx[403]*acadoWorkspace.x[1] + acadoWorkspace.evGx[404]*acadoWorkspace.x[2] + acadoWorkspace.evGx[405]*acadoWorkspace.x[3] + acadoWorkspace.evGx[406]*acadoWorkspace.x[4] + acadoWorkspace.evGx[407]*acadoWorkspace.x[5] + acadoWorkspace.d[67];
acadoVariables.x[74] += + acadoWorkspace.evGx[408]*acadoWorkspace.x[0] + acadoWorkspace.evGx[409]*acadoWorkspace.x[1] + acadoWorkspace.evGx[410]*acadoWorkspace.x[2] + acadoWorkspace.evGx[411]*acadoWorkspace.x[3] + acadoWorkspace.evGx[412]*acadoWorkspace.x[4] + acadoWorkspace.evGx[413]*acadoWorkspace.x[5] + acadoWorkspace.d[68];
acadoVariables.x[75] += + acadoWorkspace.evGx[414]*acadoWorkspace.x[0] + acadoWorkspace.evGx[415]*acadoWorkspace.x[1] + acadoWorkspace.evGx[416]*acadoWorkspace.x[2] + acadoWorkspace.evGx[417]*acadoWorkspace.x[3] + acadoWorkspace.evGx[418]*acadoWorkspace.x[4] + acadoWorkspace.evGx[419]*acadoWorkspace.x[5] + acadoWorkspace.d[69];
acadoVariables.x[76] += + acadoWorkspace.evGx[420]*acadoWorkspace.x[0] + acadoWorkspace.evGx[421]*acadoWorkspace.x[1] + acadoWorkspace.evGx[422]*acadoWorkspace.x[2] + acadoWorkspace.evGx[423]*acadoWorkspace.x[3] + acadoWorkspace.evGx[424]*acadoWorkspace.x[4] + acadoWorkspace.evGx[425]*acadoWorkspace.x[5] + acadoWorkspace.d[70];
acadoVariables.x[77] += + acadoWorkspace.evGx[426]*acadoWorkspace.x[0] + acadoWorkspace.evGx[427]*acadoWorkspace.x[1] + acadoWorkspace.evGx[428]*acadoWorkspace.x[2] + acadoWorkspace.evGx[429]*acadoWorkspace.x[3] + acadoWorkspace.evGx[430]*acadoWorkspace.x[4] + acadoWorkspace.evGx[431]*acadoWorkspace.x[5] + acadoWorkspace.d[71];
acadoVariables.x[78] += + acadoWorkspace.evGx[432]*acadoWorkspace.x[0] + acadoWorkspace.evGx[433]*acadoWorkspace.x[1] + acadoWorkspace.evGx[434]*acadoWorkspace.x[2] + acadoWorkspace.evGx[435]*acadoWorkspace.x[3] + acadoWorkspace.evGx[436]*acadoWorkspace.x[4] + acadoWorkspace.evGx[437]*acadoWorkspace.x[5] + acadoWorkspace.d[72];
acadoVariables.x[79] += + acadoWorkspace.evGx[438]*acadoWorkspace.x[0] + acadoWorkspace.evGx[439]*acadoWorkspace.x[1] + acadoWorkspace.evGx[440]*acadoWorkspace.x[2] + acadoWorkspace.evGx[441]*acadoWorkspace.x[3] + acadoWorkspace.evGx[442]*acadoWorkspace.x[4] + acadoWorkspace.evGx[443]*acadoWorkspace.x[5] + acadoWorkspace.d[73];
acadoVariables.x[80] += + acadoWorkspace.evGx[444]*acadoWorkspace.x[0] + acadoWorkspace.evGx[445]*acadoWorkspace.x[1] + acadoWorkspace.evGx[446]*acadoWorkspace.x[2] + acadoWorkspace.evGx[447]*acadoWorkspace.x[3] + acadoWorkspace.evGx[448]*acadoWorkspace.x[4] + acadoWorkspace.evGx[449]*acadoWorkspace.x[5] + acadoWorkspace.d[74];
acadoVariables.x[81] += + acadoWorkspace.evGx[450]*acadoWorkspace.x[0] + acadoWorkspace.evGx[451]*acadoWorkspace.x[1] + acadoWorkspace.evGx[452]*acadoWorkspace.x[2] + acadoWorkspace.evGx[453]*acadoWorkspace.x[3] + acadoWorkspace.evGx[454]*acadoWorkspace.x[4] + acadoWorkspace.evGx[455]*acadoWorkspace.x[5] + acadoWorkspace.d[75];
acadoVariables.x[82] += + acadoWorkspace.evGx[456]*acadoWorkspace.x[0] + acadoWorkspace.evGx[457]*acadoWorkspace.x[1] + acadoWorkspace.evGx[458]*acadoWorkspace.x[2] + acadoWorkspace.evGx[459]*acadoWorkspace.x[3] + acadoWorkspace.evGx[460]*acadoWorkspace.x[4] + acadoWorkspace.evGx[461]*acadoWorkspace.x[5] + acadoWorkspace.d[76];
acadoVariables.x[83] += + acadoWorkspace.evGx[462]*acadoWorkspace.x[0] + acadoWorkspace.evGx[463]*acadoWorkspace.x[1] + acadoWorkspace.evGx[464]*acadoWorkspace.x[2] + acadoWorkspace.evGx[465]*acadoWorkspace.x[3] + acadoWorkspace.evGx[466]*acadoWorkspace.x[4] + acadoWorkspace.evGx[467]*acadoWorkspace.x[5] + acadoWorkspace.d[77];
acadoVariables.x[84] += + acadoWorkspace.evGx[468]*acadoWorkspace.x[0] + acadoWorkspace.evGx[469]*acadoWorkspace.x[1] + acadoWorkspace.evGx[470]*acadoWorkspace.x[2] + acadoWorkspace.evGx[471]*acadoWorkspace.x[3] + acadoWorkspace.evGx[472]*acadoWorkspace.x[4] + acadoWorkspace.evGx[473]*acadoWorkspace.x[5] + acadoWorkspace.d[78];
acadoVariables.x[85] += + acadoWorkspace.evGx[474]*acadoWorkspace.x[0] + acadoWorkspace.evGx[475]*acadoWorkspace.x[1] + acadoWorkspace.evGx[476]*acadoWorkspace.x[2] + acadoWorkspace.evGx[477]*acadoWorkspace.x[3] + acadoWorkspace.evGx[478]*acadoWorkspace.x[4] + acadoWorkspace.evGx[479]*acadoWorkspace.x[5] + acadoWorkspace.d[79];
acadoVariables.x[86] += + acadoWorkspace.evGx[480]*acadoWorkspace.x[0] + acadoWorkspace.evGx[481]*acadoWorkspace.x[1] + acadoWorkspace.evGx[482]*acadoWorkspace.x[2] + acadoWorkspace.evGx[483]*acadoWorkspace.x[3] + acadoWorkspace.evGx[484]*acadoWorkspace.x[4] + acadoWorkspace.evGx[485]*acadoWorkspace.x[5] + acadoWorkspace.d[80];
acadoVariables.x[87] += + acadoWorkspace.evGx[486]*acadoWorkspace.x[0] + acadoWorkspace.evGx[487]*acadoWorkspace.x[1] + acadoWorkspace.evGx[488]*acadoWorkspace.x[2] + acadoWorkspace.evGx[489]*acadoWorkspace.x[3] + acadoWorkspace.evGx[490]*acadoWorkspace.x[4] + acadoWorkspace.evGx[491]*acadoWorkspace.x[5] + acadoWorkspace.d[81];
acadoVariables.x[88] += + acadoWorkspace.evGx[492]*acadoWorkspace.x[0] + acadoWorkspace.evGx[493]*acadoWorkspace.x[1] + acadoWorkspace.evGx[494]*acadoWorkspace.x[2] + acadoWorkspace.evGx[495]*acadoWorkspace.x[3] + acadoWorkspace.evGx[496]*acadoWorkspace.x[4] + acadoWorkspace.evGx[497]*acadoWorkspace.x[5] + acadoWorkspace.d[82];
acadoVariables.x[89] += + acadoWorkspace.evGx[498]*acadoWorkspace.x[0] + acadoWorkspace.evGx[499]*acadoWorkspace.x[1] + acadoWorkspace.evGx[500]*acadoWorkspace.x[2] + acadoWorkspace.evGx[501]*acadoWorkspace.x[3] + acadoWorkspace.evGx[502]*acadoWorkspace.x[4] + acadoWorkspace.evGx[503]*acadoWorkspace.x[5] + acadoWorkspace.d[83];
acadoVariables.x[90] += + acadoWorkspace.evGx[504]*acadoWorkspace.x[0] + acadoWorkspace.evGx[505]*acadoWorkspace.x[1] + acadoWorkspace.evGx[506]*acadoWorkspace.x[2] + acadoWorkspace.evGx[507]*acadoWorkspace.x[3] + acadoWorkspace.evGx[508]*acadoWorkspace.x[4] + acadoWorkspace.evGx[509]*acadoWorkspace.x[5] + acadoWorkspace.d[84];
acadoVariables.x[91] += + acadoWorkspace.evGx[510]*acadoWorkspace.x[0] + acadoWorkspace.evGx[511]*acadoWorkspace.x[1] + acadoWorkspace.evGx[512]*acadoWorkspace.x[2] + acadoWorkspace.evGx[513]*acadoWorkspace.x[3] + acadoWorkspace.evGx[514]*acadoWorkspace.x[4] + acadoWorkspace.evGx[515]*acadoWorkspace.x[5] + acadoWorkspace.d[85];
acadoVariables.x[92] += + acadoWorkspace.evGx[516]*acadoWorkspace.x[0] + acadoWorkspace.evGx[517]*acadoWorkspace.x[1] + acadoWorkspace.evGx[518]*acadoWorkspace.x[2] + acadoWorkspace.evGx[519]*acadoWorkspace.x[3] + acadoWorkspace.evGx[520]*acadoWorkspace.x[4] + acadoWorkspace.evGx[521]*acadoWorkspace.x[5] + acadoWorkspace.d[86];
acadoVariables.x[93] += + acadoWorkspace.evGx[522]*acadoWorkspace.x[0] + acadoWorkspace.evGx[523]*acadoWorkspace.x[1] + acadoWorkspace.evGx[524]*acadoWorkspace.x[2] + acadoWorkspace.evGx[525]*acadoWorkspace.x[3] + acadoWorkspace.evGx[526]*acadoWorkspace.x[4] + acadoWorkspace.evGx[527]*acadoWorkspace.x[5] + acadoWorkspace.d[87];
acadoVariables.x[94] += + acadoWorkspace.evGx[528]*acadoWorkspace.x[0] + acadoWorkspace.evGx[529]*acadoWorkspace.x[1] + acadoWorkspace.evGx[530]*acadoWorkspace.x[2] + acadoWorkspace.evGx[531]*acadoWorkspace.x[3] + acadoWorkspace.evGx[532]*acadoWorkspace.x[4] + acadoWorkspace.evGx[533]*acadoWorkspace.x[5] + acadoWorkspace.d[88];
acadoVariables.x[95] += + acadoWorkspace.evGx[534]*acadoWorkspace.x[0] + acadoWorkspace.evGx[535]*acadoWorkspace.x[1] + acadoWorkspace.evGx[536]*acadoWorkspace.x[2] + acadoWorkspace.evGx[537]*acadoWorkspace.x[3] + acadoWorkspace.evGx[538]*acadoWorkspace.x[4] + acadoWorkspace.evGx[539]*acadoWorkspace.x[5] + acadoWorkspace.d[89];
acadoVariables.x[96] += + acadoWorkspace.evGx[540]*acadoWorkspace.x[0] + acadoWorkspace.evGx[541]*acadoWorkspace.x[1] + acadoWorkspace.evGx[542]*acadoWorkspace.x[2] + acadoWorkspace.evGx[543]*acadoWorkspace.x[3] + acadoWorkspace.evGx[544]*acadoWorkspace.x[4] + acadoWorkspace.evGx[545]*acadoWorkspace.x[5] + acadoWorkspace.d[90];
acadoVariables.x[97] += + acadoWorkspace.evGx[546]*acadoWorkspace.x[0] + acadoWorkspace.evGx[547]*acadoWorkspace.x[1] + acadoWorkspace.evGx[548]*acadoWorkspace.x[2] + acadoWorkspace.evGx[549]*acadoWorkspace.x[3] + acadoWorkspace.evGx[550]*acadoWorkspace.x[4] + acadoWorkspace.evGx[551]*acadoWorkspace.x[5] + acadoWorkspace.d[91];
acadoVariables.x[98] += + acadoWorkspace.evGx[552]*acadoWorkspace.x[0] + acadoWorkspace.evGx[553]*acadoWorkspace.x[1] + acadoWorkspace.evGx[554]*acadoWorkspace.x[2] + acadoWorkspace.evGx[555]*acadoWorkspace.x[3] + acadoWorkspace.evGx[556]*acadoWorkspace.x[4] + acadoWorkspace.evGx[557]*acadoWorkspace.x[5] + acadoWorkspace.d[92];
acadoVariables.x[99] += + acadoWorkspace.evGx[558]*acadoWorkspace.x[0] + acadoWorkspace.evGx[559]*acadoWorkspace.x[1] + acadoWorkspace.evGx[560]*acadoWorkspace.x[2] + acadoWorkspace.evGx[561]*acadoWorkspace.x[3] + acadoWorkspace.evGx[562]*acadoWorkspace.x[4] + acadoWorkspace.evGx[563]*acadoWorkspace.x[5] + acadoWorkspace.d[93];
acadoVariables.x[100] += + acadoWorkspace.evGx[564]*acadoWorkspace.x[0] + acadoWorkspace.evGx[565]*acadoWorkspace.x[1] + acadoWorkspace.evGx[566]*acadoWorkspace.x[2] + acadoWorkspace.evGx[567]*acadoWorkspace.x[3] + acadoWorkspace.evGx[568]*acadoWorkspace.x[4] + acadoWorkspace.evGx[569]*acadoWorkspace.x[5] + acadoWorkspace.d[94];
acadoVariables.x[101] += + acadoWorkspace.evGx[570]*acadoWorkspace.x[0] + acadoWorkspace.evGx[571]*acadoWorkspace.x[1] + acadoWorkspace.evGx[572]*acadoWorkspace.x[2] + acadoWorkspace.evGx[573]*acadoWorkspace.x[3] + acadoWorkspace.evGx[574]*acadoWorkspace.x[4] + acadoWorkspace.evGx[575]*acadoWorkspace.x[5] + acadoWorkspace.d[95];
acadoVariables.x[102] += + acadoWorkspace.evGx[576]*acadoWorkspace.x[0] + acadoWorkspace.evGx[577]*acadoWorkspace.x[1] + acadoWorkspace.evGx[578]*acadoWorkspace.x[2] + acadoWorkspace.evGx[579]*acadoWorkspace.x[3] + acadoWorkspace.evGx[580]*acadoWorkspace.x[4] + acadoWorkspace.evGx[581]*acadoWorkspace.x[5] + acadoWorkspace.d[96];
acadoVariables.x[103] += + acadoWorkspace.evGx[582]*acadoWorkspace.x[0] + acadoWorkspace.evGx[583]*acadoWorkspace.x[1] + acadoWorkspace.evGx[584]*acadoWorkspace.x[2] + acadoWorkspace.evGx[585]*acadoWorkspace.x[3] + acadoWorkspace.evGx[586]*acadoWorkspace.x[4] + acadoWorkspace.evGx[587]*acadoWorkspace.x[5] + acadoWorkspace.d[97];
acadoVariables.x[104] += + acadoWorkspace.evGx[588]*acadoWorkspace.x[0] + acadoWorkspace.evGx[589]*acadoWorkspace.x[1] + acadoWorkspace.evGx[590]*acadoWorkspace.x[2] + acadoWorkspace.evGx[591]*acadoWorkspace.x[3] + acadoWorkspace.evGx[592]*acadoWorkspace.x[4] + acadoWorkspace.evGx[593]*acadoWorkspace.x[5] + acadoWorkspace.d[98];
acadoVariables.x[105] += + acadoWorkspace.evGx[594]*acadoWorkspace.x[0] + acadoWorkspace.evGx[595]*acadoWorkspace.x[1] + acadoWorkspace.evGx[596]*acadoWorkspace.x[2] + acadoWorkspace.evGx[597]*acadoWorkspace.x[3] + acadoWorkspace.evGx[598]*acadoWorkspace.x[4] + acadoWorkspace.evGx[599]*acadoWorkspace.x[5] + acadoWorkspace.d[99];
acadoVariables.x[106] += + acadoWorkspace.evGx[600]*acadoWorkspace.x[0] + acadoWorkspace.evGx[601]*acadoWorkspace.x[1] + acadoWorkspace.evGx[602]*acadoWorkspace.x[2] + acadoWorkspace.evGx[603]*acadoWorkspace.x[3] + acadoWorkspace.evGx[604]*acadoWorkspace.x[4] + acadoWorkspace.evGx[605]*acadoWorkspace.x[5] + acadoWorkspace.d[100];
acadoVariables.x[107] += + acadoWorkspace.evGx[606]*acadoWorkspace.x[0] + acadoWorkspace.evGx[607]*acadoWorkspace.x[1] + acadoWorkspace.evGx[608]*acadoWorkspace.x[2] + acadoWorkspace.evGx[609]*acadoWorkspace.x[3] + acadoWorkspace.evGx[610]*acadoWorkspace.x[4] + acadoWorkspace.evGx[611]*acadoWorkspace.x[5] + acadoWorkspace.d[101];
acadoVariables.x[108] += + acadoWorkspace.evGx[612]*acadoWorkspace.x[0] + acadoWorkspace.evGx[613]*acadoWorkspace.x[1] + acadoWorkspace.evGx[614]*acadoWorkspace.x[2] + acadoWorkspace.evGx[615]*acadoWorkspace.x[3] + acadoWorkspace.evGx[616]*acadoWorkspace.x[4] + acadoWorkspace.evGx[617]*acadoWorkspace.x[5] + acadoWorkspace.d[102];
acadoVariables.x[109] += + acadoWorkspace.evGx[618]*acadoWorkspace.x[0] + acadoWorkspace.evGx[619]*acadoWorkspace.x[1] + acadoWorkspace.evGx[620]*acadoWorkspace.x[2] + acadoWorkspace.evGx[621]*acadoWorkspace.x[3] + acadoWorkspace.evGx[622]*acadoWorkspace.x[4] + acadoWorkspace.evGx[623]*acadoWorkspace.x[5] + acadoWorkspace.d[103];
acadoVariables.x[110] += + acadoWorkspace.evGx[624]*acadoWorkspace.x[0] + acadoWorkspace.evGx[625]*acadoWorkspace.x[1] + acadoWorkspace.evGx[626]*acadoWorkspace.x[2] + acadoWorkspace.evGx[627]*acadoWorkspace.x[3] + acadoWorkspace.evGx[628]*acadoWorkspace.x[4] + acadoWorkspace.evGx[629]*acadoWorkspace.x[5] + acadoWorkspace.d[104];
acadoVariables.x[111] += + acadoWorkspace.evGx[630]*acadoWorkspace.x[0] + acadoWorkspace.evGx[631]*acadoWorkspace.x[1] + acadoWorkspace.evGx[632]*acadoWorkspace.x[2] + acadoWorkspace.evGx[633]*acadoWorkspace.x[3] + acadoWorkspace.evGx[634]*acadoWorkspace.x[4] + acadoWorkspace.evGx[635]*acadoWorkspace.x[5] + acadoWorkspace.d[105];
acadoVariables.x[112] += + acadoWorkspace.evGx[636]*acadoWorkspace.x[0] + acadoWorkspace.evGx[637]*acadoWorkspace.x[1] + acadoWorkspace.evGx[638]*acadoWorkspace.x[2] + acadoWorkspace.evGx[639]*acadoWorkspace.x[3] + acadoWorkspace.evGx[640]*acadoWorkspace.x[4] + acadoWorkspace.evGx[641]*acadoWorkspace.x[5] + acadoWorkspace.d[106];
acadoVariables.x[113] += + acadoWorkspace.evGx[642]*acadoWorkspace.x[0] + acadoWorkspace.evGx[643]*acadoWorkspace.x[1] + acadoWorkspace.evGx[644]*acadoWorkspace.x[2] + acadoWorkspace.evGx[645]*acadoWorkspace.x[3] + acadoWorkspace.evGx[646]*acadoWorkspace.x[4] + acadoWorkspace.evGx[647]*acadoWorkspace.x[5] + acadoWorkspace.d[107];
acadoVariables.x[114] += + acadoWorkspace.evGx[648]*acadoWorkspace.x[0] + acadoWorkspace.evGx[649]*acadoWorkspace.x[1] + acadoWorkspace.evGx[650]*acadoWorkspace.x[2] + acadoWorkspace.evGx[651]*acadoWorkspace.x[3] + acadoWorkspace.evGx[652]*acadoWorkspace.x[4] + acadoWorkspace.evGx[653]*acadoWorkspace.x[5] + acadoWorkspace.d[108];
acadoVariables.x[115] += + acadoWorkspace.evGx[654]*acadoWorkspace.x[0] + acadoWorkspace.evGx[655]*acadoWorkspace.x[1] + acadoWorkspace.evGx[656]*acadoWorkspace.x[2] + acadoWorkspace.evGx[657]*acadoWorkspace.x[3] + acadoWorkspace.evGx[658]*acadoWorkspace.x[4] + acadoWorkspace.evGx[659]*acadoWorkspace.x[5] + acadoWorkspace.d[109];
acadoVariables.x[116] += + acadoWorkspace.evGx[660]*acadoWorkspace.x[0] + acadoWorkspace.evGx[661]*acadoWorkspace.x[1] + acadoWorkspace.evGx[662]*acadoWorkspace.x[2] + acadoWorkspace.evGx[663]*acadoWorkspace.x[3] + acadoWorkspace.evGx[664]*acadoWorkspace.x[4] + acadoWorkspace.evGx[665]*acadoWorkspace.x[5] + acadoWorkspace.d[110];
acadoVariables.x[117] += + acadoWorkspace.evGx[666]*acadoWorkspace.x[0] + acadoWorkspace.evGx[667]*acadoWorkspace.x[1] + acadoWorkspace.evGx[668]*acadoWorkspace.x[2] + acadoWorkspace.evGx[669]*acadoWorkspace.x[3] + acadoWorkspace.evGx[670]*acadoWorkspace.x[4] + acadoWorkspace.evGx[671]*acadoWorkspace.x[5] + acadoWorkspace.d[111];
acadoVariables.x[118] += + acadoWorkspace.evGx[672]*acadoWorkspace.x[0] + acadoWorkspace.evGx[673]*acadoWorkspace.x[1] + acadoWorkspace.evGx[674]*acadoWorkspace.x[2] + acadoWorkspace.evGx[675]*acadoWorkspace.x[3] + acadoWorkspace.evGx[676]*acadoWorkspace.x[4] + acadoWorkspace.evGx[677]*acadoWorkspace.x[5] + acadoWorkspace.d[112];
acadoVariables.x[119] += + acadoWorkspace.evGx[678]*acadoWorkspace.x[0] + acadoWorkspace.evGx[679]*acadoWorkspace.x[1] + acadoWorkspace.evGx[680]*acadoWorkspace.x[2] + acadoWorkspace.evGx[681]*acadoWorkspace.x[3] + acadoWorkspace.evGx[682]*acadoWorkspace.x[4] + acadoWorkspace.evGx[683]*acadoWorkspace.x[5] + acadoWorkspace.d[113];
acadoVariables.x[120] += + acadoWorkspace.evGx[684]*acadoWorkspace.x[0] + acadoWorkspace.evGx[685]*acadoWorkspace.x[1] + acadoWorkspace.evGx[686]*acadoWorkspace.x[2] + acadoWorkspace.evGx[687]*acadoWorkspace.x[3] + acadoWorkspace.evGx[688]*acadoWorkspace.x[4] + acadoWorkspace.evGx[689]*acadoWorkspace.x[5] + acadoWorkspace.d[114];
acadoVariables.x[121] += + acadoWorkspace.evGx[690]*acadoWorkspace.x[0] + acadoWorkspace.evGx[691]*acadoWorkspace.x[1] + acadoWorkspace.evGx[692]*acadoWorkspace.x[2] + acadoWorkspace.evGx[693]*acadoWorkspace.x[3] + acadoWorkspace.evGx[694]*acadoWorkspace.x[4] + acadoWorkspace.evGx[695]*acadoWorkspace.x[5] + acadoWorkspace.d[115];
acadoVariables.x[122] += + acadoWorkspace.evGx[696]*acadoWorkspace.x[0] + acadoWorkspace.evGx[697]*acadoWorkspace.x[1] + acadoWorkspace.evGx[698]*acadoWorkspace.x[2] + acadoWorkspace.evGx[699]*acadoWorkspace.x[3] + acadoWorkspace.evGx[700]*acadoWorkspace.x[4] + acadoWorkspace.evGx[701]*acadoWorkspace.x[5] + acadoWorkspace.d[116];
acadoVariables.x[123] += + acadoWorkspace.evGx[702]*acadoWorkspace.x[0] + acadoWorkspace.evGx[703]*acadoWorkspace.x[1] + acadoWorkspace.evGx[704]*acadoWorkspace.x[2] + acadoWorkspace.evGx[705]*acadoWorkspace.x[3] + acadoWorkspace.evGx[706]*acadoWorkspace.x[4] + acadoWorkspace.evGx[707]*acadoWorkspace.x[5] + acadoWorkspace.d[117];
acadoVariables.x[124] += + acadoWorkspace.evGx[708]*acadoWorkspace.x[0] + acadoWorkspace.evGx[709]*acadoWorkspace.x[1] + acadoWorkspace.evGx[710]*acadoWorkspace.x[2] + acadoWorkspace.evGx[711]*acadoWorkspace.x[3] + acadoWorkspace.evGx[712]*acadoWorkspace.x[4] + acadoWorkspace.evGx[713]*acadoWorkspace.x[5] + acadoWorkspace.d[118];
acadoVariables.x[125] += + acadoWorkspace.evGx[714]*acadoWorkspace.x[0] + acadoWorkspace.evGx[715]*acadoWorkspace.x[1] + acadoWorkspace.evGx[716]*acadoWorkspace.x[2] + acadoWorkspace.evGx[717]*acadoWorkspace.x[3] + acadoWorkspace.evGx[718]*acadoWorkspace.x[4] + acadoWorkspace.evGx[719]*acadoWorkspace.x[5] + acadoWorkspace.d[119];

acado_multEDu( acadoWorkspace.E, &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 6 ]) );
acado_multEDu( &(acadoWorkspace.E[ 6 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 12 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 12 ]) );
acado_multEDu( &(acadoWorkspace.E[ 18 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 30 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 18 ]) );
acado_multEDu( &(acadoWorkspace.E[ 36 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 42 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 54 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 24 ]) );
acado_multEDu( &(acadoWorkspace.E[ 60 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 66 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 78 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 84 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 30 ]) );
acado_multEDu( &(acadoWorkspace.E[ 90 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 102 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 108 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 114 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 120 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 36 ]) );
acado_multEDu( &(acadoWorkspace.E[ 126 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 132 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 138 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 144 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 150 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 156 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 162 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 42 ]) );
acado_multEDu( &(acadoWorkspace.E[ 168 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 174 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 180 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 186 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 192 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 198 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 204 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 210 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 48 ]) );
acado_multEDu( &(acadoWorkspace.E[ 216 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 222 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 228 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 234 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 240 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 246 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 252 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 258 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 264 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 54 ]) );
acado_multEDu( &(acadoWorkspace.E[ 270 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 276 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 282 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 288 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 294 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 300 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 306 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 312 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 318 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 324 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 60 ]) );
acado_multEDu( &(acadoWorkspace.E[ 330 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 336 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 342 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 348 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 354 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 360 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 366 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 372 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 378 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 384 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 390 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 66 ]) );
acado_multEDu( &(acadoWorkspace.E[ 396 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 402 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 408 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 414 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 420 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 426 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 432 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 438 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 444 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 450 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 456 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 462 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 72 ]) );
acado_multEDu( &(acadoWorkspace.E[ 468 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 474 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 480 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 486 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 492 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 498 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 504 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 510 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 516 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 522 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 528 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 534 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 540 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 78 ]) );
acado_multEDu( &(acadoWorkspace.E[ 546 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 552 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 558 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 564 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 570 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 576 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 582 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 588 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 594 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 600 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 606 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 612 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 618 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 624 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 84 ]) );
acado_multEDu( &(acadoWorkspace.E[ 630 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 636 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 642 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 648 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 654 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 660 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 666 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 672 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 678 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 684 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 690 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 696 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 702 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 708 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 714 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 90 ]) );
acado_multEDu( &(acadoWorkspace.E[ 720 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 726 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 732 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 738 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 744 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 750 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 756 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 762 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 768 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 774 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 780 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 786 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 792 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 798 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 804 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 810 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 96 ]) );
acado_multEDu( &(acadoWorkspace.E[ 816 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 822 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 828 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 834 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 840 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 846 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 852 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 858 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 864 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 870 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 876 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 882 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 888 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 894 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 900 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 906 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 912 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 102 ]) );
acado_multEDu( &(acadoWorkspace.E[ 918 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 924 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 930 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 936 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 942 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 948 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 954 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 960 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 966 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 972 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 978 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 984 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 990 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 996 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1002 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1008 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1014 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1020 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 108 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1026 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1032 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1038 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1044 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1050 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1056 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1062 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1068 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1074 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1080 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1086 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1092 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1098 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1104 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1110 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1116 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1122 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1128 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1134 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 114 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1140 ]), &(acadoWorkspace.x[ 6 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1146 ]), &(acadoWorkspace.x[ 7 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1152 ]), &(acadoWorkspace.x[ 8 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1158 ]), &(acadoWorkspace.x[ 9 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1164 ]), &(acadoWorkspace.x[ 10 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1170 ]), &(acadoWorkspace.x[ 11 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1176 ]), &(acadoWorkspace.x[ 12 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1182 ]), &(acadoWorkspace.x[ 13 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1188 ]), &(acadoWorkspace.x[ 14 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1194 ]), &(acadoWorkspace.x[ 15 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1200 ]), &(acadoWorkspace.x[ 16 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1206 ]), &(acadoWorkspace.x[ 17 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1212 ]), &(acadoWorkspace.x[ 18 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1218 ]), &(acadoWorkspace.x[ 19 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1224 ]), &(acadoWorkspace.x[ 20 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1230 ]), &(acadoWorkspace.x[ 21 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1236 ]), &(acadoWorkspace.x[ 22 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1242 ]), &(acadoWorkspace.x[ 23 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1248 ]), &(acadoWorkspace.x[ 24 ]), &(acadoVariables.x[ 120 ]) );
acado_multEDu( &(acadoWorkspace.E[ 1254 ]), &(acadoWorkspace.x[ 25 ]), &(acadoVariables.x[ 120 ]) );
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
acadoWorkspace.state[0] = acadoVariables.x[index * 6];
acadoWorkspace.state[1] = acadoVariables.x[index * 6 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 6 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 6 + 3];
acadoWorkspace.state[4] = acadoVariables.x[index * 6 + 4];
acadoWorkspace.state[5] = acadoVariables.x[index * 6 + 5];
acadoWorkspace.state[48] = acadoVariables.u[index];
acadoWorkspace.state[49] = acadoVariables.od[index * 2];
acadoWorkspace.state[50] = acadoVariables.od[index * 2 + 1];

acado_integrate(acadoWorkspace.state, index == 0, index);

acadoVariables.x[index * 6 + 6] = acadoWorkspace.state[0];
acadoVariables.x[index * 6 + 7] = acadoWorkspace.state[1];
acadoVariables.x[index * 6 + 8] = acadoWorkspace.state[2];
acadoVariables.x[index * 6 + 9] = acadoWorkspace.state[3];
acadoVariables.x[index * 6 + 10] = acadoWorkspace.state[4];
acadoVariables.x[index * 6 + 11] = acadoWorkspace.state[5];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 20; ++index)
{
acadoVariables.x[index * 6] = acadoVariables.x[index * 6 + 6];
acadoVariables.x[index * 6 + 1] = acadoVariables.x[index * 6 + 7];
acadoVariables.x[index * 6 + 2] = acadoVariables.x[index * 6 + 8];
acadoVariables.x[index * 6 + 3] = acadoVariables.x[index * 6 + 9];
acadoVariables.x[index * 6 + 4] = acadoVariables.x[index * 6 + 10];
acadoVariables.x[index * 6 + 5] = acadoVariables.x[index * 6 + 11];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[120] = xEnd[0];
acadoVariables.x[121] = xEnd[1];
acadoVariables.x[122] = xEnd[2];
acadoVariables.x[123] = xEnd[3];
acadoVariables.x[124] = xEnd[4];
acadoVariables.x[125] = xEnd[5];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[120];
acadoWorkspace.state[1] = acadoVariables.x[121];
acadoWorkspace.state[2] = acadoVariables.x[122];
acadoWorkspace.state[3] = acadoVariables.x[123];
acadoWorkspace.state[4] = acadoVariables.x[124];
acadoWorkspace.state[5] = acadoVariables.x[125];
if (uEnd != 0)
{
acadoWorkspace.state[48] = uEnd[0];
}
else
{
acadoWorkspace.state[48] = acadoVariables.u[19];
}
acadoWorkspace.state[49] = acadoVariables.od[40];
acadoWorkspace.state[50] = acadoVariables.od[41];

acado_integrate(acadoWorkspace.state, 1, 19);

acadoVariables.x[120] = acadoWorkspace.state[0];
acadoVariables.x[121] = acadoWorkspace.state[1];
acadoVariables.x[122] = acadoWorkspace.state[2];
acadoVariables.x[123] = acadoWorkspace.state[3];
acadoVariables.x[124] = acadoWorkspace.state[4];
acadoVariables.x[125] = acadoWorkspace.state[5];
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

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25];
kkt = fabs( kkt );
for (index = 0; index < 26; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 20; ++index)
{
prd = acadoWorkspace.y[index + 26];
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
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 6];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 6 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 6 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 6 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.x[lRun1 * 6 + 4];
acadoWorkspace.objValueIn[5] = acadoVariables.x[lRun1 * 6 + 5];
acadoWorkspace.objValueIn[6] = acadoVariables.u[lRun1];
acadoWorkspace.objValueIn[7] = acadoVariables.od[lRun1 * 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 4] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 4];
acadoWorkspace.Dy[lRun1 * 4 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 4 + 1];
acadoWorkspace.Dy[lRun1 * 4 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 4 + 2];
acadoWorkspace.Dy[lRun1 * 4 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 4 + 3];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[120];
acadoWorkspace.objValueIn[1] = acadoVariables.x[121];
acadoWorkspace.objValueIn[2] = acadoVariables.x[122];
acadoWorkspace.objValueIn[3] = acadoVariables.x[123];
acadoWorkspace.objValueIn[4] = acadoVariables.x[124];
acadoWorkspace.objValueIn[5] = acadoVariables.x[125];
acadoWorkspace.objValueIn[6] = acadoVariables.od[40];
acadoWorkspace.objValueIn[7] = acadoVariables.od[41];
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

