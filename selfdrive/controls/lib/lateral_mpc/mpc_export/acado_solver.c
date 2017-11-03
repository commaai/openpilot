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
for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 4 + 3];

acadoWorkspace.state[24] = acadoVariables.u[lRun1];
acadoWorkspace.state[25] = acadoVariables.od[lRun1 * 18];
acadoWorkspace.state[26] = acadoVariables.od[lRun1 * 18 + 1];
acadoWorkspace.state[27] = acadoVariables.od[lRun1 * 18 + 2];
acadoWorkspace.state[28] = acadoVariables.od[lRun1 * 18 + 3];
acadoWorkspace.state[29] = acadoVariables.od[lRun1 * 18 + 4];
acadoWorkspace.state[30] = acadoVariables.od[lRun1 * 18 + 5];
acadoWorkspace.state[31] = acadoVariables.od[lRun1 * 18 + 6];
acadoWorkspace.state[32] = acadoVariables.od[lRun1 * 18 + 7];
acadoWorkspace.state[33] = acadoVariables.od[lRun1 * 18 + 8];
acadoWorkspace.state[34] = acadoVariables.od[lRun1 * 18 + 9];
acadoWorkspace.state[35] = acadoVariables.od[lRun1 * 18 + 10];
acadoWorkspace.state[36] = acadoVariables.od[lRun1 * 18 + 11];
acadoWorkspace.state[37] = acadoVariables.od[lRun1 * 18 + 12];
acadoWorkspace.state[38] = acadoVariables.od[lRun1 * 18 + 13];
acadoWorkspace.state[39] = acadoVariables.od[lRun1 * 18 + 14];
acadoWorkspace.state[40] = acadoVariables.od[lRun1 * 18 + 15];
acadoWorkspace.state[41] = acadoVariables.od[lRun1 * 18 + 16];
acadoWorkspace.state[42] = acadoVariables.od[lRun1 * 18 + 17];

ret = acado_integrate(acadoWorkspace.state, 1);

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
/* Vector of auxiliary variables; number of elements: 19. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (exp(((real_t)(0.0000000000000000e+00)-(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-xd[1]))));
a[1] = (exp((((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])-xd[1])));
a[2] = (atan(((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[3])*xd[0]))+od[4])));
a[3] = (atan(((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[7])*xd[0]))+od[8])));
a[4] = (atan(((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[11])*xd[0]))+od[12])));
a[5] = ((real_t)(1.0000000000000000e+00)/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)));
a[6] = (exp(((real_t)(0.0000000000000000e+00)-(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-xd[1]))));
a[7] = (((real_t)(0.0000000000000000e+00)-(((od[2]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[3]*(xd[0]+xd[0])))+od[4]))*a[6]);
a[8] = (((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[6]);
a[9] = (exp((((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])-xd[1])));
a[10] = ((((od[6]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[7]*(xd[0]+xd[0])))+od[8])*a[9]);
a[11] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[9]);
a[12] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[3])*xd[0]))+od[4]),2))));
a[13] = ((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[2])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[3]))*a[12]);
a[14] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[7])*xd[0]))+od[8]),2))));
a[15] = ((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[6])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[7]))*a[14]);
a[16] = ((real_t)(1.0000000000000000e+00)/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)));
a[17] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[11])*xd[0]))+od[12]),2))));
a[18] = ((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[10])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[11]))*a[17]);

/* Compute outputs: */
out[0] = ((((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-(od[17]/(real_t)(2.0000000000000000e+00))))+(od[15]*(((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])+(od[17]/(real_t)(2.0000000000000000e+00))))))/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)))+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*((((od[10]*((xd[0]*xd[0])*xd[0]))+(od[11]*(xd[0]*xd[0])))+(od[12]*xd[0]))+od[13])))-xd[1]);
out[1] = (od[14]*a[0]);
out[2] = (od[15]*a[1]);
out[3] = ((od[1]+(real_t)(1.0000000000000000e+00))*((((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*a[2])+(od[15]*a[3])))/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)))+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*a[4]))-xd[2]));
out[4] = ((od[1]+(real_t)(1.0000000000000000e+00))*u[0]);
out[5] = (((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*(((od[2]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[3]*(xd[0]+xd[0])))+od[4]))+(od[15]*(((od[6]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[7]*(xd[0]+xd[0])))+od[8]))))*a[5])+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*(((od[10]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[11]*(xd[0]+xd[0])))+od[12])));
out[6] = ((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00));
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (real_t)(0.0000000000000000e+00);
out[9] = (od[14]*a[7]);
out[10] = (od[14]*a[8]);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = (od[15]*a[10]);
out[14] = (od[15]*a[11]);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = (real_t)(0.0000000000000000e+00);
out[17] = ((od[1]+(real_t)(1.0000000000000000e+00))*(((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*a[13])+(od[15]*a[15])))*a[16])+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*a[18])));
out[18] = (real_t)(0.0000000000000000e+00);
out[19] = ((od[1]+(real_t)(1.0000000000000000e+00))*((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)));
out[20] = (real_t)(0.0000000000000000e+00);
out[21] = (real_t)(0.0000000000000000e+00);
out[22] = (real_t)(0.0000000000000000e+00);
out[23] = (real_t)(0.0000000000000000e+00);
out[24] = (real_t)(0.0000000000000000e+00);
out[25] = (real_t)(0.0000000000000000e+00);
out[26] = (real_t)(0.0000000000000000e+00);
out[27] = (real_t)(0.0000000000000000e+00);
out[28] = (real_t)(0.0000000000000000e+00);
out[29] = (od[1]+(real_t)(1.0000000000000000e+00));
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 4;
/* Vector of auxiliary variables; number of elements: 19. */
real_t* a = acadoWorkspace.objAuxVar;

/* Compute intermediate quantities: */
a[0] = (exp(((real_t)(0.0000000000000000e+00)-(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-xd[1]))));
a[1] = (exp((((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])-xd[1])));
a[2] = (atan(((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[3])*xd[0]))+od[4])));
a[3] = (atan(((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[7])*xd[0]))+od[8])));
a[4] = (atan(((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[11])*xd[0]))+od[12])));
a[5] = ((real_t)(1.0000000000000000e+00)/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)));
a[6] = (exp(((real_t)(0.0000000000000000e+00)-(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-xd[1]))));
a[7] = (((real_t)(0.0000000000000000e+00)-(((od[2]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[3]*(xd[0]+xd[0])))+od[4]))*a[6]);
a[8] = (((real_t)(0.0000000000000000e+00)-((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)))*a[6]);
a[9] = (exp((((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])-xd[1])));
a[10] = ((((od[6]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[7]*(xd[0]+xd[0])))+od[8])*a[9]);
a[11] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[9]);
a[12] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[3])*xd[0]))+od[4]),2))));
a[13] = ((((((real_t)(3.0000000000000000e+00)*od[2])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[2])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[3]))*a[12]);
a[14] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[7])*xd[0]))+od[8]),2))));
a[15] = ((((((real_t)(3.0000000000000000e+00)*od[6])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[6])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[7]))*a[14]);
a[16] = ((real_t)(1.0000000000000000e+00)/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)));
a[17] = ((real_t)(1.0000000000000000e+00)/((real_t)(1.0000000000000000e+00)+(pow(((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])*xd[0])+(((real_t)(2.0000000000000000e+00)*od[11])*xd[0]))+od[12]),2))));
a[18] = ((((((real_t)(3.0000000000000000e+00)*od[10])*xd[0])+(((real_t)(3.0000000000000000e+00)*od[10])*xd[0]))+((real_t)(2.0000000000000000e+00)*od[11]))*a[17]);

/* Compute outputs: */
out[0] = ((((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*(((((od[2]*((xd[0]*xd[0])*xd[0]))+(od[3]*(xd[0]*xd[0])))+(od[4]*xd[0]))+od[5])-(od[17]/(real_t)(2.0000000000000000e+00))))+(od[15]*(((((od[6]*((xd[0]*xd[0])*xd[0]))+(od[7]*(xd[0]*xd[0])))+(od[8]*xd[0]))+od[9])+(od[17]/(real_t)(2.0000000000000000e+00))))))/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)))+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*((((od[10]*((xd[0]*xd[0])*xd[0]))+(od[11]*(xd[0]*xd[0])))+(od[12]*xd[0]))+od[13])))-xd[1]);
out[1] = (od[14]*a[0]);
out[2] = (od[15]*a[1]);
out[3] = ((((real_t)(2.0000000000000000e+00)*od[1])+(real_t)(1.0000000000000000e+00))*((((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*a[2])+(od[15]*a[3])))/((od[14]+od[15])+(real_t)(1.0000000000000000e-04)))+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*a[4]))-xd[2]));
out[4] = (((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*(((od[2]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[3]*(xd[0]+xd[0])))+od[4]))+(od[15]*(((od[6]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[7]*(xd[0]+xd[0])))+od[8]))))*a[5])+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*(((od[10]*(((xd[0]+xd[0])*xd[0])+(xd[0]*xd[0])))+(od[11]*(xd[0]+xd[0])))+od[12])));
out[5] = ((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00));
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (real_t)(0.0000000000000000e+00);
out[8] = (od[14]*a[7]);
out[9] = (od[14]*a[8]);
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (od[15]*a[10]);
out[13] = (od[15]*a[11]);
out[14] = (real_t)(0.0000000000000000e+00);
out[15] = (real_t)(0.0000000000000000e+00);
out[16] = ((((real_t)(2.0000000000000000e+00)*od[1])+(real_t)(1.0000000000000000e+00))*(((((od[14]+od[15])-(od[14]*od[15]))*((od[14]*a[13])+(od[15]*a[15])))*a[16])+(((real_t)(1.0000000000000000e+00)-((od[14]+od[15])-(od[14]*od[15])))*a[18])));
out[17] = (real_t)(0.0000000000000000e+00);
out[18] = ((((real_t)(2.0000000000000000e+00)*od[1])+(real_t)(1.0000000000000000e+00))*((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00)));
out[19] = (real_t)(0.0000000000000000e+00);
}

void acado_setObjQ1Q2( real_t* const tmpFx, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = + tmpFx[0];
tmpQ2[1] = + tmpFx[4];
tmpQ2[2] = + tmpFx[8];
tmpQ2[3] = + tmpFx[12];
tmpQ2[4] = + tmpFx[16]*(real_t)5.0000000000000000e-01;
tmpQ2[5] = + tmpFx[1];
tmpQ2[6] = + tmpFx[5];
tmpQ2[7] = + tmpFx[9];
tmpQ2[8] = + tmpFx[13];
tmpQ2[9] = + tmpFx[17]*(real_t)5.0000000000000000e-01;
tmpQ2[10] = + tmpFx[2];
tmpQ2[11] = + tmpFx[6];
tmpQ2[12] = + tmpFx[10];
tmpQ2[13] = + tmpFx[14];
tmpQ2[14] = + tmpFx[18]*(real_t)5.0000000000000000e-01;
tmpQ2[15] = + tmpFx[3];
tmpQ2[16] = + tmpFx[7];
tmpQ2[17] = + tmpFx[11];
tmpQ2[18] = + tmpFx[15];
tmpQ2[19] = + tmpFx[19]*(real_t)5.0000000000000000e-01;
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

void acado_setObjR1R2( real_t* const tmpFu, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = + tmpFu[0];
tmpR2[1] = + tmpFu[1];
tmpR2[2] = + tmpFu[2];
tmpR2[3] = + tmpFu[3];
tmpR2[4] = + tmpFu[4]*(real_t)5.0000000000000000e-01;
tmpR1[0] = + tmpR2[0]*tmpFu[0] + tmpR2[1]*tmpFu[1] + tmpR2[2]*tmpFu[2] + tmpR2[3]*tmpFu[3] + tmpR2[4]*tmpFu[4];
}

void acado_setObjQN1QN2( real_t* const tmpFx, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = + tmpFx[0];
tmpQN2[1] = + tmpFx[4];
tmpQN2[2] = + tmpFx[8];
tmpQN2[3] = + tmpFx[12];
tmpQN2[4] = + tmpFx[1];
tmpQN2[5] = + tmpFx[5];
tmpQN2[6] = + tmpFx[9];
tmpQN2[7] = + tmpFx[13];
tmpQN2[8] = + tmpFx[2];
tmpQN2[9] = + tmpFx[6];
tmpQN2[10] = + tmpFx[10];
tmpQN2[11] = + tmpFx[14];
tmpQN2[12] = + tmpFx[3];
tmpQN2[13] = + tmpFx[7];
tmpQN2[14] = + tmpFx[11];
tmpQN2[15] = + tmpFx[15];
tmpQN1[0] = + tmpQN2[0]*tmpFx[0] + tmpQN2[1]*tmpFx[4] + tmpQN2[2]*tmpFx[8] + tmpQN2[3]*tmpFx[12];
tmpQN1[1] = + tmpQN2[0]*tmpFx[1] + tmpQN2[1]*tmpFx[5] + tmpQN2[2]*tmpFx[9] + tmpQN2[3]*tmpFx[13];
tmpQN1[2] = + tmpQN2[0]*tmpFx[2] + tmpQN2[1]*tmpFx[6] + tmpQN2[2]*tmpFx[10] + tmpQN2[3]*tmpFx[14];
tmpQN1[3] = + tmpQN2[0]*tmpFx[3] + tmpQN2[1]*tmpFx[7] + tmpQN2[2]*tmpFx[11] + tmpQN2[3]*tmpFx[15];
tmpQN1[4] = + tmpQN2[4]*tmpFx[0] + tmpQN2[5]*tmpFx[4] + tmpQN2[6]*tmpFx[8] + tmpQN2[7]*tmpFx[12];
tmpQN1[5] = + tmpQN2[4]*tmpFx[1] + tmpQN2[5]*tmpFx[5] + tmpQN2[6]*tmpFx[9] + tmpQN2[7]*tmpFx[13];
tmpQN1[6] = + tmpQN2[4]*tmpFx[2] + tmpQN2[5]*tmpFx[6] + tmpQN2[6]*tmpFx[10] + tmpQN2[7]*tmpFx[14];
tmpQN1[7] = + tmpQN2[4]*tmpFx[3] + tmpQN2[5]*tmpFx[7] + tmpQN2[6]*tmpFx[11] + tmpQN2[7]*tmpFx[15];
tmpQN1[8] = + tmpQN2[8]*tmpFx[0] + tmpQN2[9]*tmpFx[4] + tmpQN2[10]*tmpFx[8] + tmpQN2[11]*tmpFx[12];
tmpQN1[9] = + tmpQN2[8]*tmpFx[1] + tmpQN2[9]*tmpFx[5] + tmpQN2[10]*tmpFx[9] + tmpQN2[11]*tmpFx[13];
tmpQN1[10] = + tmpQN2[8]*tmpFx[2] + tmpQN2[9]*tmpFx[6] + tmpQN2[10]*tmpFx[10] + tmpQN2[11]*tmpFx[14];
tmpQN1[11] = + tmpQN2[8]*tmpFx[3] + tmpQN2[9]*tmpFx[7] + tmpQN2[10]*tmpFx[11] + tmpQN2[11]*tmpFx[15];
tmpQN1[12] = + tmpQN2[12]*tmpFx[0] + tmpQN2[13]*tmpFx[4] + tmpQN2[14]*tmpFx[8] + tmpQN2[15]*tmpFx[12];
tmpQN1[13] = + tmpQN2[12]*tmpFx[1] + tmpQN2[13]*tmpFx[5] + tmpQN2[14]*tmpFx[9] + tmpQN2[15]*tmpFx[13];
tmpQN1[14] = + tmpQN2[12]*tmpFx[2] + tmpQN2[13]*tmpFx[6] + tmpQN2[14]*tmpFx[10] + tmpQN2[15]*tmpFx[14];
tmpQN1[15] = + tmpQN2[12]*tmpFx[3] + tmpQN2[13]*tmpFx[7] + tmpQN2[14]*tmpFx[11] + tmpQN2[15]*tmpFx[15];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 50; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[runObj];
acadoWorkspace.objValueIn[5] = acadoVariables.od[runObj * 18];
acadoWorkspace.objValueIn[6] = acadoVariables.od[runObj * 18 + 1];
acadoWorkspace.objValueIn[7] = acadoVariables.od[runObj * 18 + 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[runObj * 18 + 3];
acadoWorkspace.objValueIn[9] = acadoVariables.od[runObj * 18 + 4];
acadoWorkspace.objValueIn[10] = acadoVariables.od[runObj * 18 + 5];
acadoWorkspace.objValueIn[11] = acadoVariables.od[runObj * 18 + 6];
acadoWorkspace.objValueIn[12] = acadoVariables.od[runObj * 18 + 7];
acadoWorkspace.objValueIn[13] = acadoVariables.od[runObj * 18 + 8];
acadoWorkspace.objValueIn[14] = acadoVariables.od[runObj * 18 + 9];
acadoWorkspace.objValueIn[15] = acadoVariables.od[runObj * 18 + 10];
acadoWorkspace.objValueIn[16] = acadoVariables.od[runObj * 18 + 11];
acadoWorkspace.objValueIn[17] = acadoVariables.od[runObj * 18 + 12];
acadoWorkspace.objValueIn[18] = acadoVariables.od[runObj * 18 + 13];
acadoWorkspace.objValueIn[19] = acadoVariables.od[runObj * 18 + 14];
acadoWorkspace.objValueIn[20] = acadoVariables.od[runObj * 18 + 15];
acadoWorkspace.objValueIn[21] = acadoVariables.od[runObj * 18 + 16];
acadoWorkspace.objValueIn[22] = acadoVariables.od[runObj * 18 + 17];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 5] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 5 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 5 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 5 + 3] = acadoWorkspace.objValueOut[3];
acadoWorkspace.Dy[runObj * 5 + 4] = acadoWorkspace.objValueOut[4];

acado_setObjQ1Q2( &(acadoWorkspace.objValueOut[ 5 ]), &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 20 ]) );

acado_setObjR1R2( &(acadoWorkspace.objValueOut[ 25 ]), &(acadoWorkspace.R1[ runObj ]), &(acadoWorkspace.R2[ runObj * 5 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[200];
acadoWorkspace.objValueIn[1] = acadoVariables.x[201];
acadoWorkspace.objValueIn[2] = acadoVariables.x[202];
acadoWorkspace.objValueIn[3] = acadoVariables.x[203];
acadoWorkspace.objValueIn[4] = acadoVariables.od[900];
acadoWorkspace.objValueIn[5] = acadoVariables.od[901];
acadoWorkspace.objValueIn[6] = acadoVariables.od[902];
acadoWorkspace.objValueIn[7] = acadoVariables.od[903];
acadoWorkspace.objValueIn[8] = acadoVariables.od[904];
acadoWorkspace.objValueIn[9] = acadoVariables.od[905];
acadoWorkspace.objValueIn[10] = acadoVariables.od[906];
acadoWorkspace.objValueIn[11] = acadoVariables.od[907];
acadoWorkspace.objValueIn[12] = acadoVariables.od[908];
acadoWorkspace.objValueIn[13] = acadoVariables.od[909];
acadoWorkspace.objValueIn[14] = acadoVariables.od[910];
acadoWorkspace.objValueIn[15] = acadoVariables.od[911];
acadoWorkspace.objValueIn[16] = acadoVariables.od[912];
acadoWorkspace.objValueIn[17] = acadoVariables.od[913];
acadoWorkspace.objValueIn[18] = acadoVariables.od[914];
acadoWorkspace.objValueIn[19] = acadoVariables.od[915];
acadoWorkspace.objValueIn[20] = acadoVariables.od[916];
acadoWorkspace.objValueIn[21] = acadoVariables.od[917];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3];

acado_setObjQN1QN2( &(acadoWorkspace.objValueOut[ 4 ]), acadoWorkspace.QN1, acadoWorkspace.QN2 );

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
acadoWorkspace.H[(iRow * 54 + 216) + (iCol + 4)] += + Gu1[0]*Gu2[0] + Gu1[1]*Gu2[1] + Gu1[2]*Gu2[2] + Gu1[3]*Gu2[3];
}

void acado_setBlockH11_R1( int iRow, int iCol, real_t* const R11 )
{
acadoWorkspace.H[(iRow * 54 + 216) + (iCol + 4)] = R11[0];
}

void acado_zeroBlockH11( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 54 + 216) + (iCol + 4)] = 0.0000000000000000e+00;
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 54 + 216) + (iCol + 4)] = acadoWorkspace.H[(iCol * 54 + 216) + (iRow + 4)];
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
acadoWorkspace.H[54] = 0.0000000000000000e+00;
acadoWorkspace.H[55] = 0.0000000000000000e+00;
acadoWorkspace.H[56] = 0.0000000000000000e+00;
acadoWorkspace.H[57] = 0.0000000000000000e+00;
acadoWorkspace.H[108] = 0.0000000000000000e+00;
acadoWorkspace.H[109] = 0.0000000000000000e+00;
acadoWorkspace.H[110] = 0.0000000000000000e+00;
acadoWorkspace.H[111] = 0.0000000000000000e+00;
acadoWorkspace.H[162] = 0.0000000000000000e+00;
acadoWorkspace.H[163] = 0.0000000000000000e+00;
acadoWorkspace.H[164] = 0.0000000000000000e+00;
acadoWorkspace.H[165] = 0.0000000000000000e+00;
}

void acado_multCTQC( real_t* const Gx1, real_t* const Gx2 )
{
acadoWorkspace.H[0] += + Gx1[0]*Gx2[0] + Gx1[4]*Gx2[4] + Gx1[8]*Gx2[8] + Gx1[12]*Gx2[12];
acadoWorkspace.H[1] += + Gx1[0]*Gx2[1] + Gx1[4]*Gx2[5] + Gx1[8]*Gx2[9] + Gx1[12]*Gx2[13];
acadoWorkspace.H[2] += + Gx1[0]*Gx2[2] + Gx1[4]*Gx2[6] + Gx1[8]*Gx2[10] + Gx1[12]*Gx2[14];
acadoWorkspace.H[3] += + Gx1[0]*Gx2[3] + Gx1[4]*Gx2[7] + Gx1[8]*Gx2[11] + Gx1[12]*Gx2[15];
acadoWorkspace.H[54] += + Gx1[1]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[9]*Gx2[8] + Gx1[13]*Gx2[12];
acadoWorkspace.H[55] += + Gx1[1]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[9]*Gx2[9] + Gx1[13]*Gx2[13];
acadoWorkspace.H[56] += + Gx1[1]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[9]*Gx2[10] + Gx1[13]*Gx2[14];
acadoWorkspace.H[57] += + Gx1[1]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[9]*Gx2[11] + Gx1[13]*Gx2[15];
acadoWorkspace.H[108] += + Gx1[2]*Gx2[0] + Gx1[6]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[14]*Gx2[12];
acadoWorkspace.H[109] += + Gx1[2]*Gx2[1] + Gx1[6]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[14]*Gx2[13];
acadoWorkspace.H[110] += + Gx1[2]*Gx2[2] + Gx1[6]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[14]*Gx2[14];
acadoWorkspace.H[111] += + Gx1[2]*Gx2[3] + Gx1[6]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[14]*Gx2[15];
acadoWorkspace.H[162] += + Gx1[3]*Gx2[0] + Gx1[7]*Gx2[4] + Gx1[11]*Gx2[8] + Gx1[15]*Gx2[12];
acadoWorkspace.H[163] += + Gx1[3]*Gx2[1] + Gx1[7]*Gx2[5] + Gx1[11]*Gx2[9] + Gx1[15]*Gx2[13];
acadoWorkspace.H[164] += + Gx1[3]*Gx2[2] + Gx1[7]*Gx2[6] + Gx1[11]*Gx2[10] + Gx1[15]*Gx2[14];
acadoWorkspace.H[165] += + Gx1[3]*Gx2[3] + Gx1[7]*Gx2[7] + Gx1[11]*Gx2[11] + Gx1[15]*Gx2[15];
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
/** Row vector of size: 100 */
static const int xBoundIndices[ 100 ] = 
{ 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63, 66, 67, 70, 71, 74, 75, 78, 79, 82, 83, 86, 87, 90, 91, 94, 95, 98, 99, 102, 103, 106, 107, 110, 111, 114, 115, 118, 119, 122, 123, 126, 127, 130, 131, 134, 135, 138, 139, 142, 143, 146, 147, 150, 151, 154, 155, 158, 159, 162, 163, 166, 167, 170, 171, 174, 175, 178, 179, 182, 183, 186, 187, 190, 191, 194, 195, 198, 199, 202, 203 };
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
for (lRun1 = 1; lRun1 < 50; ++lRun1)
{
acado_moveGxT( &(acadoWorkspace.evGx[ lRun1 * 16 ]), acadoWorkspace.T );
acado_multGxd( &(acadoWorkspace.d[ lRun1 * 4-4 ]), &(acadoWorkspace.evGx[ lRun1 * 16 ]), &(acadoWorkspace.d[ lRun1 * 4 ]) );
acado_multGxGx( acadoWorkspace.T, &(acadoWorkspace.evGx[ lRun1 * 16-16 ]), &(acadoWorkspace.evGx[ lRun1 * 16 ]) );
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
lRun4 = (((lRun1) * (lRun1-1)) / (2)) + (lRun2);
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.T, &(acadoWorkspace.E[ lRun4 * 4 ]), &(acadoWorkspace.E[ lRun3 * 4 ]) );
}
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_moveGuE( &(acadoWorkspace.evGu[ lRun1 * 4 ]), &(acadoWorkspace.E[ lRun3 * 4 ]) );
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
acado_multGxGx( &(acadoWorkspace.Q1[ 512 ]), &(acadoWorkspace.evGx[ 496 ]), &(acadoWorkspace.QGx[ 496 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 528 ]), &(acadoWorkspace.evGx[ 512 ]), &(acadoWorkspace.QGx[ 512 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 544 ]), &(acadoWorkspace.evGx[ 528 ]), &(acadoWorkspace.QGx[ 528 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 560 ]), &(acadoWorkspace.evGx[ 544 ]), &(acadoWorkspace.QGx[ 544 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.evGx[ 560 ]), &(acadoWorkspace.QGx[ 560 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 592 ]), &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.QGx[ 576 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 608 ]), &(acadoWorkspace.evGx[ 592 ]), &(acadoWorkspace.QGx[ 592 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 624 ]), &(acadoWorkspace.evGx[ 608 ]), &(acadoWorkspace.QGx[ 608 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 640 ]), &(acadoWorkspace.evGx[ 624 ]), &(acadoWorkspace.QGx[ 624 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 656 ]), &(acadoWorkspace.evGx[ 640 ]), &(acadoWorkspace.QGx[ 640 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 672 ]), &(acadoWorkspace.evGx[ 656 ]), &(acadoWorkspace.QGx[ 656 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 688 ]), &(acadoWorkspace.evGx[ 672 ]), &(acadoWorkspace.QGx[ 672 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 704 ]), &(acadoWorkspace.evGx[ 688 ]), &(acadoWorkspace.QGx[ 688 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 720 ]), &(acadoWorkspace.evGx[ 704 ]), &(acadoWorkspace.QGx[ 704 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 736 ]), &(acadoWorkspace.evGx[ 720 ]), &(acadoWorkspace.QGx[ 720 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 752 ]), &(acadoWorkspace.evGx[ 736 ]), &(acadoWorkspace.QGx[ 736 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 768 ]), &(acadoWorkspace.evGx[ 752 ]), &(acadoWorkspace.QGx[ 752 ]) );
acado_multGxGx( &(acadoWorkspace.Q1[ 784 ]), &(acadoWorkspace.evGx[ 768 ]), &(acadoWorkspace.QGx[ 768 ]) );
acado_multGxGx( acadoWorkspace.QN1, &(acadoWorkspace.evGx[ 784 ]), &(acadoWorkspace.QGx[ 784 ]) );

for (lRun1 = 0; lRun1 < 49; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( &(acadoWorkspace.Q1[ lRun1 * 16 + 16 ]), &(acadoWorkspace.E[ lRun3 * 4 ]), &(acadoWorkspace.QE[ lRun3 * 4 ]) );
}
}

for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ lRun3 * 4 ]), &(acadoWorkspace.QE[ lRun3 * 4 ]) );
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
acado_multCTQC( &(acadoWorkspace.evGx[ 512 ]), &(acadoWorkspace.QGx[ 512 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 528 ]), &(acadoWorkspace.QGx[ 528 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 544 ]), &(acadoWorkspace.QGx[ 544 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 560 ]), &(acadoWorkspace.QGx[ 560 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.QGx[ 576 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 592 ]), &(acadoWorkspace.QGx[ 592 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 608 ]), &(acadoWorkspace.QGx[ 608 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 624 ]), &(acadoWorkspace.QGx[ 624 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 640 ]), &(acadoWorkspace.QGx[ 640 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 656 ]), &(acadoWorkspace.QGx[ 656 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 672 ]), &(acadoWorkspace.QGx[ 672 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 688 ]), &(acadoWorkspace.QGx[ 688 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 704 ]), &(acadoWorkspace.QGx[ 704 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 720 ]), &(acadoWorkspace.QGx[ 720 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 736 ]), &(acadoWorkspace.QGx[ 736 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 752 ]), &(acadoWorkspace.QGx[ 752 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 768 ]), &(acadoWorkspace.QGx[ 768 ]) );
acado_multCTQC( &(acadoWorkspace.evGx[ 784 ]), &(acadoWorkspace.QGx[ 784 ]) );

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
acado_zeroBlockH10( &(acadoWorkspace.H10[ lRun1 * 4 ]) );
for (lRun2 = lRun1; lRun2 < 50; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multQETGx( &(acadoWorkspace.QE[ lRun3 * 4 ]), &(acadoWorkspace.evGx[ lRun2 * 16 ]), &(acadoWorkspace.H10[ lRun1 * 4 ]) );
}
}

for (lRun1 = 0;lRun1 < 4; ++lRun1)
for (lRun2 = 0;lRun2 < 50; ++lRun2)
acadoWorkspace.H[(lRun1 * 54) + (lRun2 + 4)] = acadoWorkspace.H10[(lRun2 * 4) + (lRun1)];

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
acado_setBlockH11_R1( lRun1, lRun1, &(acadoWorkspace.R1[ lRun1 ]) );
lRun2 = lRun1;
for (lRun3 = lRun1; lRun3 < 50; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 4 ]), &(acadoWorkspace.QE[ lRun5 * 4 ]) );
}
for (lRun2 = lRun1 + 1; lRun2 < 50; ++lRun2)
{
acado_zeroBlockH11( lRun1, lRun2 );
for (lRun3 = lRun2; lRun3 < 50; ++lRun3)
{
lRun4 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun1);
lRun5 = (((lRun3 + 1) * (lRun3)) / (2)) + (lRun2);
acado_setBlockH11( lRun1, lRun2, &(acadoWorkspace.E[ lRun4 * 4 ]), &(acadoWorkspace.QE[ lRun5 * 4 ]) );
}
}
}

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
acado_copyHTH( lRun1, lRun2 );
}
}

for (lRun1 = 0;lRun1 < 50; ++lRun1)
for (lRun2 = 0;lRun2 < 4; ++lRun2)
acadoWorkspace.H[(lRun1 * 54 + 216) + (lRun2)] = acadoWorkspace.H10[(lRun1 * 4) + (lRun2)];

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
acado_multQ1d( &(acadoWorkspace.Q1[ 512 ]), &(acadoWorkspace.d[ 124 ]), &(acadoWorkspace.Qd[ 124 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 528 ]), &(acadoWorkspace.d[ 128 ]), &(acadoWorkspace.Qd[ 128 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 544 ]), &(acadoWorkspace.d[ 132 ]), &(acadoWorkspace.Qd[ 132 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 560 ]), &(acadoWorkspace.d[ 136 ]), &(acadoWorkspace.Qd[ 136 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.d[ 140 ]), &(acadoWorkspace.Qd[ 140 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 592 ]), &(acadoWorkspace.d[ 144 ]), &(acadoWorkspace.Qd[ 144 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 608 ]), &(acadoWorkspace.d[ 148 ]), &(acadoWorkspace.Qd[ 148 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 624 ]), &(acadoWorkspace.d[ 152 ]), &(acadoWorkspace.Qd[ 152 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 640 ]), &(acadoWorkspace.d[ 156 ]), &(acadoWorkspace.Qd[ 156 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 656 ]), &(acadoWorkspace.d[ 160 ]), &(acadoWorkspace.Qd[ 160 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 672 ]), &(acadoWorkspace.d[ 164 ]), &(acadoWorkspace.Qd[ 164 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 688 ]), &(acadoWorkspace.d[ 168 ]), &(acadoWorkspace.Qd[ 168 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 704 ]), &(acadoWorkspace.d[ 172 ]), &(acadoWorkspace.Qd[ 172 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 720 ]), &(acadoWorkspace.d[ 176 ]), &(acadoWorkspace.Qd[ 176 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 736 ]), &(acadoWorkspace.d[ 180 ]), &(acadoWorkspace.Qd[ 180 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 752 ]), &(acadoWorkspace.d[ 184 ]), &(acadoWorkspace.Qd[ 184 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 768 ]), &(acadoWorkspace.d[ 188 ]), &(acadoWorkspace.Qd[ 188 ]) );
acado_multQ1d( &(acadoWorkspace.Q1[ 784 ]), &(acadoWorkspace.d[ 192 ]), &(acadoWorkspace.Qd[ 192 ]) );
acado_multQN1d( acadoWorkspace.QN1, &(acadoWorkspace.d[ 196 ]), &(acadoWorkspace.Qd[ 196 ]) );

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
acado_macCTSlx( &(acadoWorkspace.evGx[ 512 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 528 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 544 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 560 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 576 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 592 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 608 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 624 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 640 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 656 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 672 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 688 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 704 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 720 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 736 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 752 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 768 ]), acadoWorkspace.g );
acado_macCTSlx( &(acadoWorkspace.evGx[ 784 ]), acadoWorkspace.g );
for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 50; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_macETSlu( &(acadoWorkspace.QE[ lRun3 * 4 ]), &(acadoWorkspace.g[ lRun1 + 4 ]) );
}
}
acadoWorkspace.lb[4] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[0];
acadoWorkspace.lb[5] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[1];
acadoWorkspace.lb[6] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[2];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[3];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[4];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[5];
acadoWorkspace.lb[10] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[6];
acadoWorkspace.lb[11] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[7];
acadoWorkspace.lb[12] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[8];
acadoWorkspace.lb[13] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[9];
acadoWorkspace.lb[14] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[10];
acadoWorkspace.lb[15] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[11];
acadoWorkspace.lb[16] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[12];
acadoWorkspace.lb[17] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[13];
acadoWorkspace.lb[18] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[14];
acadoWorkspace.lb[19] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[15];
acadoWorkspace.lb[20] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[16];
acadoWorkspace.lb[21] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[17];
acadoWorkspace.lb[22] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[18];
acadoWorkspace.lb[23] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[19];
acadoWorkspace.lb[24] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[20];
acadoWorkspace.lb[25] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[21];
acadoWorkspace.lb[26] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[22];
acadoWorkspace.lb[27] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[23];
acadoWorkspace.lb[28] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[24];
acadoWorkspace.lb[29] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[25];
acadoWorkspace.lb[30] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[26];
acadoWorkspace.lb[31] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[27];
acadoWorkspace.lb[32] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[28];
acadoWorkspace.lb[33] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[29];
acadoWorkspace.lb[34] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[30];
acadoWorkspace.lb[35] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[31];
acadoWorkspace.lb[36] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[32];
acadoWorkspace.lb[37] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[33];
acadoWorkspace.lb[38] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[34];
acadoWorkspace.lb[39] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[35];
acadoWorkspace.lb[40] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[36];
acadoWorkspace.lb[41] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[37];
acadoWorkspace.lb[42] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[38];
acadoWorkspace.lb[43] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[39];
acadoWorkspace.lb[44] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[40];
acadoWorkspace.lb[45] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[41];
acadoWorkspace.lb[46] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[42];
acadoWorkspace.lb[47] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[43];
acadoWorkspace.lb[48] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[44];
acadoWorkspace.lb[49] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[45];
acadoWorkspace.lb[50] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[46];
acadoWorkspace.lb[51] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[47];
acadoWorkspace.lb[52] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[48];
acadoWorkspace.lb[53] = (real_t)-1.0000000000000001e-01 - acadoVariables.u[49];
acadoWorkspace.ub[4] = (real_t)1.0000000000000001e-01 - acadoVariables.u[0];
acadoWorkspace.ub[5] = (real_t)1.0000000000000001e-01 - acadoVariables.u[1];
acadoWorkspace.ub[6] = (real_t)1.0000000000000001e-01 - acadoVariables.u[2];
acadoWorkspace.ub[7] = (real_t)1.0000000000000001e-01 - acadoVariables.u[3];
acadoWorkspace.ub[8] = (real_t)1.0000000000000001e-01 - acadoVariables.u[4];
acadoWorkspace.ub[9] = (real_t)1.0000000000000001e-01 - acadoVariables.u[5];
acadoWorkspace.ub[10] = (real_t)1.0000000000000001e-01 - acadoVariables.u[6];
acadoWorkspace.ub[11] = (real_t)1.0000000000000001e-01 - acadoVariables.u[7];
acadoWorkspace.ub[12] = (real_t)1.0000000000000001e-01 - acadoVariables.u[8];
acadoWorkspace.ub[13] = (real_t)1.0000000000000001e-01 - acadoVariables.u[9];
acadoWorkspace.ub[14] = (real_t)1.0000000000000001e-01 - acadoVariables.u[10];
acadoWorkspace.ub[15] = (real_t)1.0000000000000001e-01 - acadoVariables.u[11];
acadoWorkspace.ub[16] = (real_t)1.0000000000000001e-01 - acadoVariables.u[12];
acadoWorkspace.ub[17] = (real_t)1.0000000000000001e-01 - acadoVariables.u[13];
acadoWorkspace.ub[18] = (real_t)1.0000000000000001e-01 - acadoVariables.u[14];
acadoWorkspace.ub[19] = (real_t)1.0000000000000001e-01 - acadoVariables.u[15];
acadoWorkspace.ub[20] = (real_t)1.0000000000000001e-01 - acadoVariables.u[16];
acadoWorkspace.ub[21] = (real_t)1.0000000000000001e-01 - acadoVariables.u[17];
acadoWorkspace.ub[22] = (real_t)1.0000000000000001e-01 - acadoVariables.u[18];
acadoWorkspace.ub[23] = (real_t)1.0000000000000001e-01 - acadoVariables.u[19];
acadoWorkspace.ub[24] = (real_t)1.0000000000000001e-01 - acadoVariables.u[20];
acadoWorkspace.ub[25] = (real_t)1.0000000000000001e-01 - acadoVariables.u[21];
acadoWorkspace.ub[26] = (real_t)1.0000000000000001e-01 - acadoVariables.u[22];
acadoWorkspace.ub[27] = (real_t)1.0000000000000001e-01 - acadoVariables.u[23];
acadoWorkspace.ub[28] = (real_t)1.0000000000000001e-01 - acadoVariables.u[24];
acadoWorkspace.ub[29] = (real_t)1.0000000000000001e-01 - acadoVariables.u[25];
acadoWorkspace.ub[30] = (real_t)1.0000000000000001e-01 - acadoVariables.u[26];
acadoWorkspace.ub[31] = (real_t)1.0000000000000001e-01 - acadoVariables.u[27];
acadoWorkspace.ub[32] = (real_t)1.0000000000000001e-01 - acadoVariables.u[28];
acadoWorkspace.ub[33] = (real_t)1.0000000000000001e-01 - acadoVariables.u[29];
acadoWorkspace.ub[34] = (real_t)1.0000000000000001e-01 - acadoVariables.u[30];
acadoWorkspace.ub[35] = (real_t)1.0000000000000001e-01 - acadoVariables.u[31];
acadoWorkspace.ub[36] = (real_t)1.0000000000000001e-01 - acadoVariables.u[32];
acadoWorkspace.ub[37] = (real_t)1.0000000000000001e-01 - acadoVariables.u[33];
acadoWorkspace.ub[38] = (real_t)1.0000000000000001e-01 - acadoVariables.u[34];
acadoWorkspace.ub[39] = (real_t)1.0000000000000001e-01 - acadoVariables.u[35];
acadoWorkspace.ub[40] = (real_t)1.0000000000000001e-01 - acadoVariables.u[36];
acadoWorkspace.ub[41] = (real_t)1.0000000000000001e-01 - acadoVariables.u[37];
acadoWorkspace.ub[42] = (real_t)1.0000000000000001e-01 - acadoVariables.u[38];
acadoWorkspace.ub[43] = (real_t)1.0000000000000001e-01 - acadoVariables.u[39];
acadoWorkspace.ub[44] = (real_t)1.0000000000000001e-01 - acadoVariables.u[40];
acadoWorkspace.ub[45] = (real_t)1.0000000000000001e-01 - acadoVariables.u[41];
acadoWorkspace.ub[46] = (real_t)1.0000000000000001e-01 - acadoVariables.u[42];
acadoWorkspace.ub[47] = (real_t)1.0000000000000001e-01 - acadoVariables.u[43];
acadoWorkspace.ub[48] = (real_t)1.0000000000000001e-01 - acadoVariables.u[44];
acadoWorkspace.ub[49] = (real_t)1.0000000000000001e-01 - acadoVariables.u[45];
acadoWorkspace.ub[50] = (real_t)1.0000000000000001e-01 - acadoVariables.u[46];
acadoWorkspace.ub[51] = (real_t)1.0000000000000001e-01 - acadoVariables.u[47];
acadoWorkspace.ub[52] = (real_t)1.0000000000000001e-01 - acadoVariables.u[48];
acadoWorkspace.ub[53] = (real_t)1.0000000000000001e-01 - acadoVariables.u[49];

for (lRun1 = 0; lRun1 < 100; ++lRun1)
{
lRun3 = xBoundIndices[ lRun1 ] - 4;
lRun4 = ((lRun3) / (4)) + (1);
acadoWorkspace.A[lRun1 * 54] = acadoWorkspace.evGx[lRun3 * 4];
acadoWorkspace.A[lRun1 * 54 + 1] = acadoWorkspace.evGx[lRun3 * 4 + 1];
acadoWorkspace.A[lRun1 * 54 + 2] = acadoWorkspace.evGx[lRun3 * 4 + 2];
acadoWorkspace.A[lRun1 * 54 + 3] = acadoWorkspace.evGx[lRun3 * 4 + 3];
for (lRun2 = 0; lRun2 < lRun4; ++lRun2)
{
lRun5 = (((((lRun4) * (lRun4-1)) / (2)) + (lRun2)) * (4)) + ((lRun3) % (4));
acadoWorkspace.A[(lRun1 * 54) + (lRun2 + 4)] = acadoWorkspace.E[lRun5];
}
}

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

for (lRun2 = 0; lRun2 < 250; ++lRun2)
acadoWorkspace.Dy[lRun2] -= acadoVariables.y[lRun2];

acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];
acadoWorkspace.DyN[3] -= acadoVariables.yN[3];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 5 ]), &(acadoWorkspace.Dy[ 5 ]), &(acadoWorkspace.g[ 5 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 10 ]), &(acadoWorkspace.Dy[ 10 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 15 ]), &(acadoWorkspace.Dy[ 15 ]), &(acadoWorkspace.g[ 7 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 20 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 25 ]), &(acadoWorkspace.Dy[ 25 ]), &(acadoWorkspace.g[ 9 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 30 ]), &(acadoWorkspace.Dy[ 30 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 35 ]), &(acadoWorkspace.Dy[ 35 ]), &(acadoWorkspace.g[ 11 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 40 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 45 ]), &(acadoWorkspace.Dy[ 45 ]), &(acadoWorkspace.g[ 13 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 50 ]), &(acadoWorkspace.Dy[ 50 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 55 ]), &(acadoWorkspace.Dy[ 55 ]), &(acadoWorkspace.g[ 15 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 60 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 65 ]), &(acadoWorkspace.Dy[ 65 ]), &(acadoWorkspace.g[ 17 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 70 ]), &(acadoWorkspace.Dy[ 70 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 75 ]), &(acadoWorkspace.Dy[ 75 ]), &(acadoWorkspace.g[ 19 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 80 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 85 ]), &(acadoWorkspace.Dy[ 85 ]), &(acadoWorkspace.g[ 21 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 90 ]), &(acadoWorkspace.Dy[ 90 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 95 ]), &(acadoWorkspace.Dy[ 95 ]), &(acadoWorkspace.g[ 23 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 100 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 105 ]), &(acadoWorkspace.Dy[ 105 ]), &(acadoWorkspace.g[ 25 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 110 ]), &(acadoWorkspace.Dy[ 110 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 115 ]), &(acadoWorkspace.Dy[ 115 ]), &(acadoWorkspace.g[ 27 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 120 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 125 ]), &(acadoWorkspace.Dy[ 125 ]), &(acadoWorkspace.g[ 29 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 130 ]), &(acadoWorkspace.Dy[ 130 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 135 ]), &(acadoWorkspace.Dy[ 135 ]), &(acadoWorkspace.g[ 31 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 140 ]), &(acadoWorkspace.Dy[ 140 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 145 ]), &(acadoWorkspace.Dy[ 145 ]), &(acadoWorkspace.g[ 33 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 150 ]), &(acadoWorkspace.Dy[ 150 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 155 ]), &(acadoWorkspace.Dy[ 155 ]), &(acadoWorkspace.g[ 35 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 160 ]), &(acadoWorkspace.Dy[ 160 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 165 ]), &(acadoWorkspace.Dy[ 165 ]), &(acadoWorkspace.g[ 37 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 170 ]), &(acadoWorkspace.Dy[ 170 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 175 ]), &(acadoWorkspace.Dy[ 175 ]), &(acadoWorkspace.g[ 39 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 180 ]), &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.g[ 40 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 185 ]), &(acadoWorkspace.Dy[ 185 ]), &(acadoWorkspace.g[ 41 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 190 ]), &(acadoWorkspace.Dy[ 190 ]), &(acadoWorkspace.g[ 42 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 195 ]), &(acadoWorkspace.Dy[ 195 ]), &(acadoWorkspace.g[ 43 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 200 ]), &(acadoWorkspace.Dy[ 200 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 205 ]), &(acadoWorkspace.Dy[ 205 ]), &(acadoWorkspace.g[ 45 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 210 ]), &(acadoWorkspace.Dy[ 210 ]), &(acadoWorkspace.g[ 46 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 215 ]), &(acadoWorkspace.Dy[ 215 ]), &(acadoWorkspace.g[ 47 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 220 ]), &(acadoWorkspace.Dy[ 220 ]), &(acadoWorkspace.g[ 48 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 225 ]), &(acadoWorkspace.Dy[ 225 ]), &(acadoWorkspace.g[ 49 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 230 ]), &(acadoWorkspace.Dy[ 230 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 235 ]), &(acadoWorkspace.Dy[ 235 ]), &(acadoWorkspace.g[ 51 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 240 ]), &(acadoWorkspace.Dy[ 240 ]), &(acadoWorkspace.g[ 52 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 245 ]), &(acadoWorkspace.Dy[ 245 ]), &(acadoWorkspace.g[ 53 ]) );

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
acado_multQDy( &(acadoWorkspace.Q2[ 640 ]), &(acadoWorkspace.Dy[ 160 ]), &(acadoWorkspace.QDy[ 128 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 660 ]), &(acadoWorkspace.Dy[ 165 ]), &(acadoWorkspace.QDy[ 132 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 680 ]), &(acadoWorkspace.Dy[ 170 ]), &(acadoWorkspace.QDy[ 136 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 700 ]), &(acadoWorkspace.Dy[ 175 ]), &(acadoWorkspace.QDy[ 140 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 720 ]), &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.QDy[ 144 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 740 ]), &(acadoWorkspace.Dy[ 185 ]), &(acadoWorkspace.QDy[ 148 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 760 ]), &(acadoWorkspace.Dy[ 190 ]), &(acadoWorkspace.QDy[ 152 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 780 ]), &(acadoWorkspace.Dy[ 195 ]), &(acadoWorkspace.QDy[ 156 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 800 ]), &(acadoWorkspace.Dy[ 200 ]), &(acadoWorkspace.QDy[ 160 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 820 ]), &(acadoWorkspace.Dy[ 205 ]), &(acadoWorkspace.QDy[ 164 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 840 ]), &(acadoWorkspace.Dy[ 210 ]), &(acadoWorkspace.QDy[ 168 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 860 ]), &(acadoWorkspace.Dy[ 215 ]), &(acadoWorkspace.QDy[ 172 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 880 ]), &(acadoWorkspace.Dy[ 220 ]), &(acadoWorkspace.QDy[ 176 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 900 ]), &(acadoWorkspace.Dy[ 225 ]), &(acadoWorkspace.QDy[ 180 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 920 ]), &(acadoWorkspace.Dy[ 230 ]), &(acadoWorkspace.QDy[ 184 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 940 ]), &(acadoWorkspace.Dy[ 235 ]), &(acadoWorkspace.QDy[ 188 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 960 ]), &(acadoWorkspace.Dy[ 240 ]), &(acadoWorkspace.QDy[ 192 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 980 ]), &(acadoWorkspace.Dy[ 245 ]), &(acadoWorkspace.QDy[ 196 ]) );

acadoWorkspace.QDy[200] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[201] = + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[202] = + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[203] = + acadoWorkspace.QN2[12]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[13]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[14]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[15]*acadoWorkspace.DyN[3];

for (lRun2 = 0; lRun2 < 200; ++lRun2)
acadoWorkspace.QDy[lRun2 + 4] += acadoWorkspace.Qd[lRun2];


acadoWorkspace.g[0] = + acadoWorkspace.evGx[0]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[4]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[8]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[12]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[16]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[20]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[24]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[28]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[32]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[36]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[40]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[44]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[48]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[52]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[56]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[60]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[64]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[68]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[72]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[76]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[80]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[84]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[88]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[92]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[96]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[100]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[104]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[108]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[112]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[116]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[120]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[124]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[128]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[132]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[136]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[140]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[144]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[148]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[152]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[156]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[160]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[164]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[168]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[172]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[176]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[180]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[184]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[188]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[192]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[196]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[200]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[204]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[208]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[212]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[216]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[220]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[224]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[228]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[232]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[236]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[240]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[244]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[248]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[252]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[256]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[260]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[264]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[268]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[272]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[276]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[280]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[284]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[288]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[292]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[296]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[300]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[304]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[308]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[312]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[316]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[320]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[324]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[328]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[332]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[336]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[340]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[344]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[348]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[352]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[356]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[360]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[364]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[368]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[372]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[376]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[380]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[384]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[388]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[392]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[396]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[400]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[404]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[408]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[412]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[416]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[420]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[424]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[428]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[432]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[436]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[440]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[444]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[448]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[452]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[456]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[460]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[464]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[468]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[472]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[476]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[480]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[484]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[488]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[492]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[496]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[500]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[504]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[508]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[512]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[516]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[520]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[524]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[528]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[532]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[536]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[540]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[544]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[548]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[552]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[556]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[560]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[564]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[568]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[572]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[576]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[580]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[584]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[588]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[592]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[596]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[600]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[604]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[608]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[612]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[616]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[620]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[624]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[628]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[632]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[636]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[640]*acadoWorkspace.QDy[164] + acadoWorkspace.evGx[644]*acadoWorkspace.QDy[165] + acadoWorkspace.evGx[648]*acadoWorkspace.QDy[166] + acadoWorkspace.evGx[652]*acadoWorkspace.QDy[167] + acadoWorkspace.evGx[656]*acadoWorkspace.QDy[168] + acadoWorkspace.evGx[660]*acadoWorkspace.QDy[169] + acadoWorkspace.evGx[664]*acadoWorkspace.QDy[170] + acadoWorkspace.evGx[668]*acadoWorkspace.QDy[171] + acadoWorkspace.evGx[672]*acadoWorkspace.QDy[172] + acadoWorkspace.evGx[676]*acadoWorkspace.QDy[173] + acadoWorkspace.evGx[680]*acadoWorkspace.QDy[174] + acadoWorkspace.evGx[684]*acadoWorkspace.QDy[175] + acadoWorkspace.evGx[688]*acadoWorkspace.QDy[176] + acadoWorkspace.evGx[692]*acadoWorkspace.QDy[177] + acadoWorkspace.evGx[696]*acadoWorkspace.QDy[178] + acadoWorkspace.evGx[700]*acadoWorkspace.QDy[179] + acadoWorkspace.evGx[704]*acadoWorkspace.QDy[180] + acadoWorkspace.evGx[708]*acadoWorkspace.QDy[181] + acadoWorkspace.evGx[712]*acadoWorkspace.QDy[182] + acadoWorkspace.evGx[716]*acadoWorkspace.QDy[183] + acadoWorkspace.evGx[720]*acadoWorkspace.QDy[184] + acadoWorkspace.evGx[724]*acadoWorkspace.QDy[185] + acadoWorkspace.evGx[728]*acadoWorkspace.QDy[186] + acadoWorkspace.evGx[732]*acadoWorkspace.QDy[187] + acadoWorkspace.evGx[736]*acadoWorkspace.QDy[188] + acadoWorkspace.evGx[740]*acadoWorkspace.QDy[189] + acadoWorkspace.evGx[744]*acadoWorkspace.QDy[190] + acadoWorkspace.evGx[748]*acadoWorkspace.QDy[191] + acadoWorkspace.evGx[752]*acadoWorkspace.QDy[192] + acadoWorkspace.evGx[756]*acadoWorkspace.QDy[193] + acadoWorkspace.evGx[760]*acadoWorkspace.QDy[194] + acadoWorkspace.evGx[764]*acadoWorkspace.QDy[195] + acadoWorkspace.evGx[768]*acadoWorkspace.QDy[196] + acadoWorkspace.evGx[772]*acadoWorkspace.QDy[197] + acadoWorkspace.evGx[776]*acadoWorkspace.QDy[198] + acadoWorkspace.evGx[780]*acadoWorkspace.QDy[199] + acadoWorkspace.evGx[784]*acadoWorkspace.QDy[200] + acadoWorkspace.evGx[788]*acadoWorkspace.QDy[201] + acadoWorkspace.evGx[792]*acadoWorkspace.QDy[202] + acadoWorkspace.evGx[796]*acadoWorkspace.QDy[203];
acadoWorkspace.g[1] = + acadoWorkspace.evGx[1]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[5]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[9]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[13]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[17]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[21]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[25]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[29]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[33]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[37]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[41]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[45]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[49]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[53]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[57]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[61]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[65]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[69]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[73]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[77]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[81]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[85]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[89]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[93]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[97]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[101]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[105]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[109]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[113]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[117]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[121]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[125]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[129]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[133]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[137]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[141]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[145]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[149]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[153]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[157]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[161]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[165]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[169]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[173]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[177]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[181]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[185]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[189]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[193]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[197]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[201]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[205]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[209]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[213]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[217]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[221]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[225]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[229]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[233]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[237]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[241]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[245]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[249]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[253]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[257]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[261]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[265]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[269]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[273]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[277]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[281]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[285]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[289]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[293]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[297]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[301]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[305]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[309]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[313]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[317]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[321]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[325]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[329]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[333]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[337]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[341]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[345]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[349]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[353]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[357]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[361]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[365]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[369]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[373]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[377]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[381]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[385]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[389]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[393]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[397]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[401]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[405]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[409]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[413]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[417]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[421]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[425]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[429]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[433]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[437]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[441]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[445]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[449]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[453]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[457]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[461]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[465]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[469]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[473]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[477]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[481]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[485]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[489]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[493]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[497]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[501]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[505]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[509]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[513]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[517]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[521]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[525]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[529]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[533]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[537]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[541]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[545]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[549]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[553]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[557]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[561]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[565]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[569]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[573]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[577]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[581]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[585]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[589]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[593]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[597]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[601]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[605]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[609]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[613]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[617]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[621]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[625]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[629]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[633]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[637]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[641]*acadoWorkspace.QDy[164] + acadoWorkspace.evGx[645]*acadoWorkspace.QDy[165] + acadoWorkspace.evGx[649]*acadoWorkspace.QDy[166] + acadoWorkspace.evGx[653]*acadoWorkspace.QDy[167] + acadoWorkspace.evGx[657]*acadoWorkspace.QDy[168] + acadoWorkspace.evGx[661]*acadoWorkspace.QDy[169] + acadoWorkspace.evGx[665]*acadoWorkspace.QDy[170] + acadoWorkspace.evGx[669]*acadoWorkspace.QDy[171] + acadoWorkspace.evGx[673]*acadoWorkspace.QDy[172] + acadoWorkspace.evGx[677]*acadoWorkspace.QDy[173] + acadoWorkspace.evGx[681]*acadoWorkspace.QDy[174] + acadoWorkspace.evGx[685]*acadoWorkspace.QDy[175] + acadoWorkspace.evGx[689]*acadoWorkspace.QDy[176] + acadoWorkspace.evGx[693]*acadoWorkspace.QDy[177] + acadoWorkspace.evGx[697]*acadoWorkspace.QDy[178] + acadoWorkspace.evGx[701]*acadoWorkspace.QDy[179] + acadoWorkspace.evGx[705]*acadoWorkspace.QDy[180] + acadoWorkspace.evGx[709]*acadoWorkspace.QDy[181] + acadoWorkspace.evGx[713]*acadoWorkspace.QDy[182] + acadoWorkspace.evGx[717]*acadoWorkspace.QDy[183] + acadoWorkspace.evGx[721]*acadoWorkspace.QDy[184] + acadoWorkspace.evGx[725]*acadoWorkspace.QDy[185] + acadoWorkspace.evGx[729]*acadoWorkspace.QDy[186] + acadoWorkspace.evGx[733]*acadoWorkspace.QDy[187] + acadoWorkspace.evGx[737]*acadoWorkspace.QDy[188] + acadoWorkspace.evGx[741]*acadoWorkspace.QDy[189] + acadoWorkspace.evGx[745]*acadoWorkspace.QDy[190] + acadoWorkspace.evGx[749]*acadoWorkspace.QDy[191] + acadoWorkspace.evGx[753]*acadoWorkspace.QDy[192] + acadoWorkspace.evGx[757]*acadoWorkspace.QDy[193] + acadoWorkspace.evGx[761]*acadoWorkspace.QDy[194] + acadoWorkspace.evGx[765]*acadoWorkspace.QDy[195] + acadoWorkspace.evGx[769]*acadoWorkspace.QDy[196] + acadoWorkspace.evGx[773]*acadoWorkspace.QDy[197] + acadoWorkspace.evGx[777]*acadoWorkspace.QDy[198] + acadoWorkspace.evGx[781]*acadoWorkspace.QDy[199] + acadoWorkspace.evGx[785]*acadoWorkspace.QDy[200] + acadoWorkspace.evGx[789]*acadoWorkspace.QDy[201] + acadoWorkspace.evGx[793]*acadoWorkspace.QDy[202] + acadoWorkspace.evGx[797]*acadoWorkspace.QDy[203];
acadoWorkspace.g[2] = + acadoWorkspace.evGx[2]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[6]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[10]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[14]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[18]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[22]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[26]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[30]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[34]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[38]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[42]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[46]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[50]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[54]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[58]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[62]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[66]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[70]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[74]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[78]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[82]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[86]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[90]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[94]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[98]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[102]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[106]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[110]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[114]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[118]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[122]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[126]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[130]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[134]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[138]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[142]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[146]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[150]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[154]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[158]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[162]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[166]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[170]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[174]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[178]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[182]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[186]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[190]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[194]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[198]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[202]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[206]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[210]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[214]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[218]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[222]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[226]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[230]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[234]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[238]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[242]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[246]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[250]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[254]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[258]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[262]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[266]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[270]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[274]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[278]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[282]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[286]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[290]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[294]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[298]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[302]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[306]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[310]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[314]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[318]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[322]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[326]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[330]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[334]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[338]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[342]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[346]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[350]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[354]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[358]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[362]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[366]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[370]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[374]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[378]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[382]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[386]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[390]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[394]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[398]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[402]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[406]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[410]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[414]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[418]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[422]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[426]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[430]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[434]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[438]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[442]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[446]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[450]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[454]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[458]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[462]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[466]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[470]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[474]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[478]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[482]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[486]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[490]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[494]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[498]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[502]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[506]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[510]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[514]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[518]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[522]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[526]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[530]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[534]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[538]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[542]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[546]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[550]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[554]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[558]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[562]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[566]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[570]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[574]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[578]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[582]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[586]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[590]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[594]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[598]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[602]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[606]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[610]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[614]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[618]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[622]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[626]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[630]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[634]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[638]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[642]*acadoWorkspace.QDy[164] + acadoWorkspace.evGx[646]*acadoWorkspace.QDy[165] + acadoWorkspace.evGx[650]*acadoWorkspace.QDy[166] + acadoWorkspace.evGx[654]*acadoWorkspace.QDy[167] + acadoWorkspace.evGx[658]*acadoWorkspace.QDy[168] + acadoWorkspace.evGx[662]*acadoWorkspace.QDy[169] + acadoWorkspace.evGx[666]*acadoWorkspace.QDy[170] + acadoWorkspace.evGx[670]*acadoWorkspace.QDy[171] + acadoWorkspace.evGx[674]*acadoWorkspace.QDy[172] + acadoWorkspace.evGx[678]*acadoWorkspace.QDy[173] + acadoWorkspace.evGx[682]*acadoWorkspace.QDy[174] + acadoWorkspace.evGx[686]*acadoWorkspace.QDy[175] + acadoWorkspace.evGx[690]*acadoWorkspace.QDy[176] + acadoWorkspace.evGx[694]*acadoWorkspace.QDy[177] + acadoWorkspace.evGx[698]*acadoWorkspace.QDy[178] + acadoWorkspace.evGx[702]*acadoWorkspace.QDy[179] + acadoWorkspace.evGx[706]*acadoWorkspace.QDy[180] + acadoWorkspace.evGx[710]*acadoWorkspace.QDy[181] + acadoWorkspace.evGx[714]*acadoWorkspace.QDy[182] + acadoWorkspace.evGx[718]*acadoWorkspace.QDy[183] + acadoWorkspace.evGx[722]*acadoWorkspace.QDy[184] + acadoWorkspace.evGx[726]*acadoWorkspace.QDy[185] + acadoWorkspace.evGx[730]*acadoWorkspace.QDy[186] + acadoWorkspace.evGx[734]*acadoWorkspace.QDy[187] + acadoWorkspace.evGx[738]*acadoWorkspace.QDy[188] + acadoWorkspace.evGx[742]*acadoWorkspace.QDy[189] + acadoWorkspace.evGx[746]*acadoWorkspace.QDy[190] + acadoWorkspace.evGx[750]*acadoWorkspace.QDy[191] + acadoWorkspace.evGx[754]*acadoWorkspace.QDy[192] + acadoWorkspace.evGx[758]*acadoWorkspace.QDy[193] + acadoWorkspace.evGx[762]*acadoWorkspace.QDy[194] + acadoWorkspace.evGx[766]*acadoWorkspace.QDy[195] + acadoWorkspace.evGx[770]*acadoWorkspace.QDy[196] + acadoWorkspace.evGx[774]*acadoWorkspace.QDy[197] + acadoWorkspace.evGx[778]*acadoWorkspace.QDy[198] + acadoWorkspace.evGx[782]*acadoWorkspace.QDy[199] + acadoWorkspace.evGx[786]*acadoWorkspace.QDy[200] + acadoWorkspace.evGx[790]*acadoWorkspace.QDy[201] + acadoWorkspace.evGx[794]*acadoWorkspace.QDy[202] + acadoWorkspace.evGx[798]*acadoWorkspace.QDy[203];
acadoWorkspace.g[3] = + acadoWorkspace.evGx[3]*acadoWorkspace.QDy[4] + acadoWorkspace.evGx[7]*acadoWorkspace.QDy[5] + acadoWorkspace.evGx[11]*acadoWorkspace.QDy[6] + acadoWorkspace.evGx[15]*acadoWorkspace.QDy[7] + acadoWorkspace.evGx[19]*acadoWorkspace.QDy[8] + acadoWorkspace.evGx[23]*acadoWorkspace.QDy[9] + acadoWorkspace.evGx[27]*acadoWorkspace.QDy[10] + acadoWorkspace.evGx[31]*acadoWorkspace.QDy[11] + acadoWorkspace.evGx[35]*acadoWorkspace.QDy[12] + acadoWorkspace.evGx[39]*acadoWorkspace.QDy[13] + acadoWorkspace.evGx[43]*acadoWorkspace.QDy[14] + acadoWorkspace.evGx[47]*acadoWorkspace.QDy[15] + acadoWorkspace.evGx[51]*acadoWorkspace.QDy[16] + acadoWorkspace.evGx[55]*acadoWorkspace.QDy[17] + acadoWorkspace.evGx[59]*acadoWorkspace.QDy[18] + acadoWorkspace.evGx[63]*acadoWorkspace.QDy[19] + acadoWorkspace.evGx[67]*acadoWorkspace.QDy[20] + acadoWorkspace.evGx[71]*acadoWorkspace.QDy[21] + acadoWorkspace.evGx[75]*acadoWorkspace.QDy[22] + acadoWorkspace.evGx[79]*acadoWorkspace.QDy[23] + acadoWorkspace.evGx[83]*acadoWorkspace.QDy[24] + acadoWorkspace.evGx[87]*acadoWorkspace.QDy[25] + acadoWorkspace.evGx[91]*acadoWorkspace.QDy[26] + acadoWorkspace.evGx[95]*acadoWorkspace.QDy[27] + acadoWorkspace.evGx[99]*acadoWorkspace.QDy[28] + acadoWorkspace.evGx[103]*acadoWorkspace.QDy[29] + acadoWorkspace.evGx[107]*acadoWorkspace.QDy[30] + acadoWorkspace.evGx[111]*acadoWorkspace.QDy[31] + acadoWorkspace.evGx[115]*acadoWorkspace.QDy[32] + acadoWorkspace.evGx[119]*acadoWorkspace.QDy[33] + acadoWorkspace.evGx[123]*acadoWorkspace.QDy[34] + acadoWorkspace.evGx[127]*acadoWorkspace.QDy[35] + acadoWorkspace.evGx[131]*acadoWorkspace.QDy[36] + acadoWorkspace.evGx[135]*acadoWorkspace.QDy[37] + acadoWorkspace.evGx[139]*acadoWorkspace.QDy[38] + acadoWorkspace.evGx[143]*acadoWorkspace.QDy[39] + acadoWorkspace.evGx[147]*acadoWorkspace.QDy[40] + acadoWorkspace.evGx[151]*acadoWorkspace.QDy[41] + acadoWorkspace.evGx[155]*acadoWorkspace.QDy[42] + acadoWorkspace.evGx[159]*acadoWorkspace.QDy[43] + acadoWorkspace.evGx[163]*acadoWorkspace.QDy[44] + acadoWorkspace.evGx[167]*acadoWorkspace.QDy[45] + acadoWorkspace.evGx[171]*acadoWorkspace.QDy[46] + acadoWorkspace.evGx[175]*acadoWorkspace.QDy[47] + acadoWorkspace.evGx[179]*acadoWorkspace.QDy[48] + acadoWorkspace.evGx[183]*acadoWorkspace.QDy[49] + acadoWorkspace.evGx[187]*acadoWorkspace.QDy[50] + acadoWorkspace.evGx[191]*acadoWorkspace.QDy[51] + acadoWorkspace.evGx[195]*acadoWorkspace.QDy[52] + acadoWorkspace.evGx[199]*acadoWorkspace.QDy[53] + acadoWorkspace.evGx[203]*acadoWorkspace.QDy[54] + acadoWorkspace.evGx[207]*acadoWorkspace.QDy[55] + acadoWorkspace.evGx[211]*acadoWorkspace.QDy[56] + acadoWorkspace.evGx[215]*acadoWorkspace.QDy[57] + acadoWorkspace.evGx[219]*acadoWorkspace.QDy[58] + acadoWorkspace.evGx[223]*acadoWorkspace.QDy[59] + acadoWorkspace.evGx[227]*acadoWorkspace.QDy[60] + acadoWorkspace.evGx[231]*acadoWorkspace.QDy[61] + acadoWorkspace.evGx[235]*acadoWorkspace.QDy[62] + acadoWorkspace.evGx[239]*acadoWorkspace.QDy[63] + acadoWorkspace.evGx[243]*acadoWorkspace.QDy[64] + acadoWorkspace.evGx[247]*acadoWorkspace.QDy[65] + acadoWorkspace.evGx[251]*acadoWorkspace.QDy[66] + acadoWorkspace.evGx[255]*acadoWorkspace.QDy[67] + acadoWorkspace.evGx[259]*acadoWorkspace.QDy[68] + acadoWorkspace.evGx[263]*acadoWorkspace.QDy[69] + acadoWorkspace.evGx[267]*acadoWorkspace.QDy[70] + acadoWorkspace.evGx[271]*acadoWorkspace.QDy[71] + acadoWorkspace.evGx[275]*acadoWorkspace.QDy[72] + acadoWorkspace.evGx[279]*acadoWorkspace.QDy[73] + acadoWorkspace.evGx[283]*acadoWorkspace.QDy[74] + acadoWorkspace.evGx[287]*acadoWorkspace.QDy[75] + acadoWorkspace.evGx[291]*acadoWorkspace.QDy[76] + acadoWorkspace.evGx[295]*acadoWorkspace.QDy[77] + acadoWorkspace.evGx[299]*acadoWorkspace.QDy[78] + acadoWorkspace.evGx[303]*acadoWorkspace.QDy[79] + acadoWorkspace.evGx[307]*acadoWorkspace.QDy[80] + acadoWorkspace.evGx[311]*acadoWorkspace.QDy[81] + acadoWorkspace.evGx[315]*acadoWorkspace.QDy[82] + acadoWorkspace.evGx[319]*acadoWorkspace.QDy[83] + acadoWorkspace.evGx[323]*acadoWorkspace.QDy[84] + acadoWorkspace.evGx[327]*acadoWorkspace.QDy[85] + acadoWorkspace.evGx[331]*acadoWorkspace.QDy[86] + acadoWorkspace.evGx[335]*acadoWorkspace.QDy[87] + acadoWorkspace.evGx[339]*acadoWorkspace.QDy[88] + acadoWorkspace.evGx[343]*acadoWorkspace.QDy[89] + acadoWorkspace.evGx[347]*acadoWorkspace.QDy[90] + acadoWorkspace.evGx[351]*acadoWorkspace.QDy[91] + acadoWorkspace.evGx[355]*acadoWorkspace.QDy[92] + acadoWorkspace.evGx[359]*acadoWorkspace.QDy[93] + acadoWorkspace.evGx[363]*acadoWorkspace.QDy[94] + acadoWorkspace.evGx[367]*acadoWorkspace.QDy[95] + acadoWorkspace.evGx[371]*acadoWorkspace.QDy[96] + acadoWorkspace.evGx[375]*acadoWorkspace.QDy[97] + acadoWorkspace.evGx[379]*acadoWorkspace.QDy[98] + acadoWorkspace.evGx[383]*acadoWorkspace.QDy[99] + acadoWorkspace.evGx[387]*acadoWorkspace.QDy[100] + acadoWorkspace.evGx[391]*acadoWorkspace.QDy[101] + acadoWorkspace.evGx[395]*acadoWorkspace.QDy[102] + acadoWorkspace.evGx[399]*acadoWorkspace.QDy[103] + acadoWorkspace.evGx[403]*acadoWorkspace.QDy[104] + acadoWorkspace.evGx[407]*acadoWorkspace.QDy[105] + acadoWorkspace.evGx[411]*acadoWorkspace.QDy[106] + acadoWorkspace.evGx[415]*acadoWorkspace.QDy[107] + acadoWorkspace.evGx[419]*acadoWorkspace.QDy[108] + acadoWorkspace.evGx[423]*acadoWorkspace.QDy[109] + acadoWorkspace.evGx[427]*acadoWorkspace.QDy[110] + acadoWorkspace.evGx[431]*acadoWorkspace.QDy[111] + acadoWorkspace.evGx[435]*acadoWorkspace.QDy[112] + acadoWorkspace.evGx[439]*acadoWorkspace.QDy[113] + acadoWorkspace.evGx[443]*acadoWorkspace.QDy[114] + acadoWorkspace.evGx[447]*acadoWorkspace.QDy[115] + acadoWorkspace.evGx[451]*acadoWorkspace.QDy[116] + acadoWorkspace.evGx[455]*acadoWorkspace.QDy[117] + acadoWorkspace.evGx[459]*acadoWorkspace.QDy[118] + acadoWorkspace.evGx[463]*acadoWorkspace.QDy[119] + acadoWorkspace.evGx[467]*acadoWorkspace.QDy[120] + acadoWorkspace.evGx[471]*acadoWorkspace.QDy[121] + acadoWorkspace.evGx[475]*acadoWorkspace.QDy[122] + acadoWorkspace.evGx[479]*acadoWorkspace.QDy[123] + acadoWorkspace.evGx[483]*acadoWorkspace.QDy[124] + acadoWorkspace.evGx[487]*acadoWorkspace.QDy[125] + acadoWorkspace.evGx[491]*acadoWorkspace.QDy[126] + acadoWorkspace.evGx[495]*acadoWorkspace.QDy[127] + acadoWorkspace.evGx[499]*acadoWorkspace.QDy[128] + acadoWorkspace.evGx[503]*acadoWorkspace.QDy[129] + acadoWorkspace.evGx[507]*acadoWorkspace.QDy[130] + acadoWorkspace.evGx[511]*acadoWorkspace.QDy[131] + acadoWorkspace.evGx[515]*acadoWorkspace.QDy[132] + acadoWorkspace.evGx[519]*acadoWorkspace.QDy[133] + acadoWorkspace.evGx[523]*acadoWorkspace.QDy[134] + acadoWorkspace.evGx[527]*acadoWorkspace.QDy[135] + acadoWorkspace.evGx[531]*acadoWorkspace.QDy[136] + acadoWorkspace.evGx[535]*acadoWorkspace.QDy[137] + acadoWorkspace.evGx[539]*acadoWorkspace.QDy[138] + acadoWorkspace.evGx[543]*acadoWorkspace.QDy[139] + acadoWorkspace.evGx[547]*acadoWorkspace.QDy[140] + acadoWorkspace.evGx[551]*acadoWorkspace.QDy[141] + acadoWorkspace.evGx[555]*acadoWorkspace.QDy[142] + acadoWorkspace.evGx[559]*acadoWorkspace.QDy[143] + acadoWorkspace.evGx[563]*acadoWorkspace.QDy[144] + acadoWorkspace.evGx[567]*acadoWorkspace.QDy[145] + acadoWorkspace.evGx[571]*acadoWorkspace.QDy[146] + acadoWorkspace.evGx[575]*acadoWorkspace.QDy[147] + acadoWorkspace.evGx[579]*acadoWorkspace.QDy[148] + acadoWorkspace.evGx[583]*acadoWorkspace.QDy[149] + acadoWorkspace.evGx[587]*acadoWorkspace.QDy[150] + acadoWorkspace.evGx[591]*acadoWorkspace.QDy[151] + acadoWorkspace.evGx[595]*acadoWorkspace.QDy[152] + acadoWorkspace.evGx[599]*acadoWorkspace.QDy[153] + acadoWorkspace.evGx[603]*acadoWorkspace.QDy[154] + acadoWorkspace.evGx[607]*acadoWorkspace.QDy[155] + acadoWorkspace.evGx[611]*acadoWorkspace.QDy[156] + acadoWorkspace.evGx[615]*acadoWorkspace.QDy[157] + acadoWorkspace.evGx[619]*acadoWorkspace.QDy[158] + acadoWorkspace.evGx[623]*acadoWorkspace.QDy[159] + acadoWorkspace.evGx[627]*acadoWorkspace.QDy[160] + acadoWorkspace.evGx[631]*acadoWorkspace.QDy[161] + acadoWorkspace.evGx[635]*acadoWorkspace.QDy[162] + acadoWorkspace.evGx[639]*acadoWorkspace.QDy[163] + acadoWorkspace.evGx[643]*acadoWorkspace.QDy[164] + acadoWorkspace.evGx[647]*acadoWorkspace.QDy[165] + acadoWorkspace.evGx[651]*acadoWorkspace.QDy[166] + acadoWorkspace.evGx[655]*acadoWorkspace.QDy[167] + acadoWorkspace.evGx[659]*acadoWorkspace.QDy[168] + acadoWorkspace.evGx[663]*acadoWorkspace.QDy[169] + acadoWorkspace.evGx[667]*acadoWorkspace.QDy[170] + acadoWorkspace.evGx[671]*acadoWorkspace.QDy[171] + acadoWorkspace.evGx[675]*acadoWorkspace.QDy[172] + acadoWorkspace.evGx[679]*acadoWorkspace.QDy[173] + acadoWorkspace.evGx[683]*acadoWorkspace.QDy[174] + acadoWorkspace.evGx[687]*acadoWorkspace.QDy[175] + acadoWorkspace.evGx[691]*acadoWorkspace.QDy[176] + acadoWorkspace.evGx[695]*acadoWorkspace.QDy[177] + acadoWorkspace.evGx[699]*acadoWorkspace.QDy[178] + acadoWorkspace.evGx[703]*acadoWorkspace.QDy[179] + acadoWorkspace.evGx[707]*acadoWorkspace.QDy[180] + acadoWorkspace.evGx[711]*acadoWorkspace.QDy[181] + acadoWorkspace.evGx[715]*acadoWorkspace.QDy[182] + acadoWorkspace.evGx[719]*acadoWorkspace.QDy[183] + acadoWorkspace.evGx[723]*acadoWorkspace.QDy[184] + acadoWorkspace.evGx[727]*acadoWorkspace.QDy[185] + acadoWorkspace.evGx[731]*acadoWorkspace.QDy[186] + acadoWorkspace.evGx[735]*acadoWorkspace.QDy[187] + acadoWorkspace.evGx[739]*acadoWorkspace.QDy[188] + acadoWorkspace.evGx[743]*acadoWorkspace.QDy[189] + acadoWorkspace.evGx[747]*acadoWorkspace.QDy[190] + acadoWorkspace.evGx[751]*acadoWorkspace.QDy[191] + acadoWorkspace.evGx[755]*acadoWorkspace.QDy[192] + acadoWorkspace.evGx[759]*acadoWorkspace.QDy[193] + acadoWorkspace.evGx[763]*acadoWorkspace.QDy[194] + acadoWorkspace.evGx[767]*acadoWorkspace.QDy[195] + acadoWorkspace.evGx[771]*acadoWorkspace.QDy[196] + acadoWorkspace.evGx[775]*acadoWorkspace.QDy[197] + acadoWorkspace.evGx[779]*acadoWorkspace.QDy[198] + acadoWorkspace.evGx[783]*acadoWorkspace.QDy[199] + acadoWorkspace.evGx[787]*acadoWorkspace.QDy[200] + acadoWorkspace.evGx[791]*acadoWorkspace.QDy[201] + acadoWorkspace.evGx[795]*acadoWorkspace.QDy[202] + acadoWorkspace.evGx[799]*acadoWorkspace.QDy[203];


for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
for (lRun2 = lRun1; lRun2 < 50; ++lRun2)
{
lRun3 = (((lRun2 + 1) * (lRun2)) / (2)) + (lRun1);
acado_multEQDy( &(acadoWorkspace.E[ lRun3 * 4 ]), &(acadoWorkspace.QDy[ lRun2 * 4 + 4 ]), &(acadoWorkspace.g[ lRun1 + 4 ]) );
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
tmp = acadoVariables.x[6] + acadoWorkspace.d[2];
acadoWorkspace.lbA[0] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[0] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[7] + acadoWorkspace.d[3];
acadoWorkspace.lbA[1] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[1] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[10] + acadoWorkspace.d[6];
acadoWorkspace.lbA[2] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[2] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[11] + acadoWorkspace.d[7];
acadoWorkspace.lbA[3] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[3] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[14] + acadoWorkspace.d[10];
acadoWorkspace.lbA[4] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[4] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[15] + acadoWorkspace.d[11];
acadoWorkspace.lbA[5] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[5] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[18] + acadoWorkspace.d[14];
acadoWorkspace.lbA[6] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[6] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[19] + acadoWorkspace.d[15];
acadoWorkspace.lbA[7] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[7] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[22] + acadoWorkspace.d[18];
acadoWorkspace.lbA[8] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[8] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[23] + acadoWorkspace.d[19];
acadoWorkspace.lbA[9] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[9] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[26] + acadoWorkspace.d[22];
acadoWorkspace.lbA[10] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[10] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[27] + acadoWorkspace.d[23];
acadoWorkspace.lbA[11] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[11] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[30] + acadoWorkspace.d[26];
acadoWorkspace.lbA[12] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[12] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[31] + acadoWorkspace.d[27];
acadoWorkspace.lbA[13] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[13] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[34] + acadoWorkspace.d[30];
acadoWorkspace.lbA[14] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[14] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[35] + acadoWorkspace.d[31];
acadoWorkspace.lbA[15] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[15] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[38] + acadoWorkspace.d[34];
acadoWorkspace.lbA[16] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[16] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[39] + acadoWorkspace.d[35];
acadoWorkspace.lbA[17] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[17] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[42] + acadoWorkspace.d[38];
acadoWorkspace.lbA[18] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[18] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[43] + acadoWorkspace.d[39];
acadoWorkspace.lbA[19] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[19] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[46] + acadoWorkspace.d[42];
acadoWorkspace.lbA[20] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[20] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[47] + acadoWorkspace.d[43];
acadoWorkspace.lbA[21] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[21] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[50] + acadoWorkspace.d[46];
acadoWorkspace.lbA[22] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[22] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[51] + acadoWorkspace.d[47];
acadoWorkspace.lbA[23] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[23] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[54] + acadoWorkspace.d[50];
acadoWorkspace.lbA[24] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[24] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[55] + acadoWorkspace.d[51];
acadoWorkspace.lbA[25] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[25] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[58] + acadoWorkspace.d[54];
acadoWorkspace.lbA[26] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[26] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[59] + acadoWorkspace.d[55];
acadoWorkspace.lbA[27] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[27] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[62] + acadoWorkspace.d[58];
acadoWorkspace.lbA[28] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[28] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[63] + acadoWorkspace.d[59];
acadoWorkspace.lbA[29] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[29] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[66] + acadoWorkspace.d[62];
acadoWorkspace.lbA[30] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[30] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[67] + acadoWorkspace.d[63];
acadoWorkspace.lbA[31] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[31] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[70] + acadoWorkspace.d[66];
acadoWorkspace.lbA[32] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[32] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[71] + acadoWorkspace.d[67];
acadoWorkspace.lbA[33] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[33] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[74] + acadoWorkspace.d[70];
acadoWorkspace.lbA[34] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[34] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[75] + acadoWorkspace.d[71];
acadoWorkspace.lbA[35] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[35] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[78] + acadoWorkspace.d[74];
acadoWorkspace.lbA[36] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[36] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[79] + acadoWorkspace.d[75];
acadoWorkspace.lbA[37] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[37] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[82] + acadoWorkspace.d[78];
acadoWorkspace.lbA[38] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[38] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[83] + acadoWorkspace.d[79];
acadoWorkspace.lbA[39] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[39] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[86] + acadoWorkspace.d[82];
acadoWorkspace.lbA[40] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[40] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[87] + acadoWorkspace.d[83];
acadoWorkspace.lbA[41] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[41] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[90] + acadoWorkspace.d[86];
acadoWorkspace.lbA[42] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[42] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[91] + acadoWorkspace.d[87];
acadoWorkspace.lbA[43] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[43] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[94] + acadoWorkspace.d[90];
acadoWorkspace.lbA[44] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[44] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[95] + acadoWorkspace.d[91];
acadoWorkspace.lbA[45] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[45] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[98] + acadoWorkspace.d[94];
acadoWorkspace.lbA[46] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[46] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[99] + acadoWorkspace.d[95];
acadoWorkspace.lbA[47] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[47] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[102] + acadoWorkspace.d[98];
acadoWorkspace.lbA[48] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[48] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[103] + acadoWorkspace.d[99];
acadoWorkspace.lbA[49] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[49] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[106] + acadoWorkspace.d[102];
acadoWorkspace.lbA[50] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[50] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[107] + acadoWorkspace.d[103];
acadoWorkspace.lbA[51] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[51] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[110] + acadoWorkspace.d[106];
acadoWorkspace.lbA[52] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[52] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[111] + acadoWorkspace.d[107];
acadoWorkspace.lbA[53] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[53] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[114] + acadoWorkspace.d[110];
acadoWorkspace.lbA[54] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[54] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[115] + acadoWorkspace.d[111];
acadoWorkspace.lbA[55] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[55] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[118] + acadoWorkspace.d[114];
acadoWorkspace.lbA[56] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[56] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[119] + acadoWorkspace.d[115];
acadoWorkspace.lbA[57] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[57] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[122] + acadoWorkspace.d[118];
acadoWorkspace.lbA[58] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[58] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[123] + acadoWorkspace.d[119];
acadoWorkspace.lbA[59] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[59] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[126] + acadoWorkspace.d[122];
acadoWorkspace.lbA[60] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[60] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[127] + acadoWorkspace.d[123];
acadoWorkspace.lbA[61] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[61] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[130] + acadoWorkspace.d[126];
acadoWorkspace.lbA[62] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[62] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[131] + acadoWorkspace.d[127];
acadoWorkspace.lbA[63] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[63] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[134] + acadoWorkspace.d[130];
acadoWorkspace.lbA[64] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[64] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[135] + acadoWorkspace.d[131];
acadoWorkspace.lbA[65] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[65] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[138] + acadoWorkspace.d[134];
acadoWorkspace.lbA[66] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[66] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[139] + acadoWorkspace.d[135];
acadoWorkspace.lbA[67] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[67] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[142] + acadoWorkspace.d[138];
acadoWorkspace.lbA[68] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[68] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[143] + acadoWorkspace.d[139];
acadoWorkspace.lbA[69] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[69] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[146] + acadoWorkspace.d[142];
acadoWorkspace.lbA[70] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[70] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[147] + acadoWorkspace.d[143];
acadoWorkspace.lbA[71] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[71] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[150] + acadoWorkspace.d[146];
acadoWorkspace.lbA[72] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[72] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[151] + acadoWorkspace.d[147];
acadoWorkspace.lbA[73] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[73] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[154] + acadoWorkspace.d[150];
acadoWorkspace.lbA[74] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[74] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[155] + acadoWorkspace.d[151];
acadoWorkspace.lbA[75] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[75] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[158] + acadoWorkspace.d[154];
acadoWorkspace.lbA[76] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[76] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[159] + acadoWorkspace.d[155];
acadoWorkspace.lbA[77] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[77] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[162] + acadoWorkspace.d[158];
acadoWorkspace.lbA[78] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[78] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[163] + acadoWorkspace.d[159];
acadoWorkspace.lbA[79] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[79] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[166] + acadoWorkspace.d[162];
acadoWorkspace.lbA[80] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[80] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[167] + acadoWorkspace.d[163];
acadoWorkspace.lbA[81] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[81] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[170] + acadoWorkspace.d[166];
acadoWorkspace.lbA[82] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[82] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[171] + acadoWorkspace.d[167];
acadoWorkspace.lbA[83] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[83] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[174] + acadoWorkspace.d[170];
acadoWorkspace.lbA[84] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[84] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[175] + acadoWorkspace.d[171];
acadoWorkspace.lbA[85] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[85] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[178] + acadoWorkspace.d[174];
acadoWorkspace.lbA[86] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[86] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[179] + acadoWorkspace.d[175];
acadoWorkspace.lbA[87] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[87] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[182] + acadoWorkspace.d[178];
acadoWorkspace.lbA[88] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[88] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[183] + acadoWorkspace.d[179];
acadoWorkspace.lbA[89] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[89] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[186] + acadoWorkspace.d[182];
acadoWorkspace.lbA[90] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[90] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[187] + acadoWorkspace.d[183];
acadoWorkspace.lbA[91] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[91] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[190] + acadoWorkspace.d[186];
acadoWorkspace.lbA[92] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[92] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[191] + acadoWorkspace.d[187];
acadoWorkspace.lbA[93] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[93] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[194] + acadoWorkspace.d[190];
acadoWorkspace.lbA[94] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[94] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[195] + acadoWorkspace.d[191];
acadoWorkspace.lbA[95] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[95] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[198] + acadoWorkspace.d[194];
acadoWorkspace.lbA[96] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[96] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[199] + acadoWorkspace.d[195];
acadoWorkspace.lbA[97] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[97] = (real_t)4.3633231300000003e-01 - tmp;
tmp = acadoVariables.x[202] + acadoWorkspace.d[198];
acadoWorkspace.lbA[98] = (real_t)-1.5707963268000000e+00 - tmp;
acadoWorkspace.ubA[98] = (real_t)1.5707963268000000e+00 - tmp;
tmp = acadoVariables.x[203] + acadoWorkspace.d[199];
acadoWorkspace.lbA[99] = (real_t)-4.3633231300000003e-01 - tmp;
acadoWorkspace.ubA[99] = (real_t)4.3633231300000003e-01 - tmp;

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
acadoVariables.x[132] += + acadoWorkspace.evGx[512]*acadoWorkspace.x[0] + acadoWorkspace.evGx[513]*acadoWorkspace.x[1] + acadoWorkspace.evGx[514]*acadoWorkspace.x[2] + acadoWorkspace.evGx[515]*acadoWorkspace.x[3] + acadoWorkspace.d[128];
acadoVariables.x[133] += + acadoWorkspace.evGx[516]*acadoWorkspace.x[0] + acadoWorkspace.evGx[517]*acadoWorkspace.x[1] + acadoWorkspace.evGx[518]*acadoWorkspace.x[2] + acadoWorkspace.evGx[519]*acadoWorkspace.x[3] + acadoWorkspace.d[129];
acadoVariables.x[134] += + acadoWorkspace.evGx[520]*acadoWorkspace.x[0] + acadoWorkspace.evGx[521]*acadoWorkspace.x[1] + acadoWorkspace.evGx[522]*acadoWorkspace.x[2] + acadoWorkspace.evGx[523]*acadoWorkspace.x[3] + acadoWorkspace.d[130];
acadoVariables.x[135] += + acadoWorkspace.evGx[524]*acadoWorkspace.x[0] + acadoWorkspace.evGx[525]*acadoWorkspace.x[1] + acadoWorkspace.evGx[526]*acadoWorkspace.x[2] + acadoWorkspace.evGx[527]*acadoWorkspace.x[3] + acadoWorkspace.d[131];
acadoVariables.x[136] += + acadoWorkspace.evGx[528]*acadoWorkspace.x[0] + acadoWorkspace.evGx[529]*acadoWorkspace.x[1] + acadoWorkspace.evGx[530]*acadoWorkspace.x[2] + acadoWorkspace.evGx[531]*acadoWorkspace.x[3] + acadoWorkspace.d[132];
acadoVariables.x[137] += + acadoWorkspace.evGx[532]*acadoWorkspace.x[0] + acadoWorkspace.evGx[533]*acadoWorkspace.x[1] + acadoWorkspace.evGx[534]*acadoWorkspace.x[2] + acadoWorkspace.evGx[535]*acadoWorkspace.x[3] + acadoWorkspace.d[133];
acadoVariables.x[138] += + acadoWorkspace.evGx[536]*acadoWorkspace.x[0] + acadoWorkspace.evGx[537]*acadoWorkspace.x[1] + acadoWorkspace.evGx[538]*acadoWorkspace.x[2] + acadoWorkspace.evGx[539]*acadoWorkspace.x[3] + acadoWorkspace.d[134];
acadoVariables.x[139] += + acadoWorkspace.evGx[540]*acadoWorkspace.x[0] + acadoWorkspace.evGx[541]*acadoWorkspace.x[1] + acadoWorkspace.evGx[542]*acadoWorkspace.x[2] + acadoWorkspace.evGx[543]*acadoWorkspace.x[3] + acadoWorkspace.d[135];
acadoVariables.x[140] += + acadoWorkspace.evGx[544]*acadoWorkspace.x[0] + acadoWorkspace.evGx[545]*acadoWorkspace.x[1] + acadoWorkspace.evGx[546]*acadoWorkspace.x[2] + acadoWorkspace.evGx[547]*acadoWorkspace.x[3] + acadoWorkspace.d[136];
acadoVariables.x[141] += + acadoWorkspace.evGx[548]*acadoWorkspace.x[0] + acadoWorkspace.evGx[549]*acadoWorkspace.x[1] + acadoWorkspace.evGx[550]*acadoWorkspace.x[2] + acadoWorkspace.evGx[551]*acadoWorkspace.x[3] + acadoWorkspace.d[137];
acadoVariables.x[142] += + acadoWorkspace.evGx[552]*acadoWorkspace.x[0] + acadoWorkspace.evGx[553]*acadoWorkspace.x[1] + acadoWorkspace.evGx[554]*acadoWorkspace.x[2] + acadoWorkspace.evGx[555]*acadoWorkspace.x[3] + acadoWorkspace.d[138];
acadoVariables.x[143] += + acadoWorkspace.evGx[556]*acadoWorkspace.x[0] + acadoWorkspace.evGx[557]*acadoWorkspace.x[1] + acadoWorkspace.evGx[558]*acadoWorkspace.x[2] + acadoWorkspace.evGx[559]*acadoWorkspace.x[3] + acadoWorkspace.d[139];
acadoVariables.x[144] += + acadoWorkspace.evGx[560]*acadoWorkspace.x[0] + acadoWorkspace.evGx[561]*acadoWorkspace.x[1] + acadoWorkspace.evGx[562]*acadoWorkspace.x[2] + acadoWorkspace.evGx[563]*acadoWorkspace.x[3] + acadoWorkspace.d[140];
acadoVariables.x[145] += + acadoWorkspace.evGx[564]*acadoWorkspace.x[0] + acadoWorkspace.evGx[565]*acadoWorkspace.x[1] + acadoWorkspace.evGx[566]*acadoWorkspace.x[2] + acadoWorkspace.evGx[567]*acadoWorkspace.x[3] + acadoWorkspace.d[141];
acadoVariables.x[146] += + acadoWorkspace.evGx[568]*acadoWorkspace.x[0] + acadoWorkspace.evGx[569]*acadoWorkspace.x[1] + acadoWorkspace.evGx[570]*acadoWorkspace.x[2] + acadoWorkspace.evGx[571]*acadoWorkspace.x[3] + acadoWorkspace.d[142];
acadoVariables.x[147] += + acadoWorkspace.evGx[572]*acadoWorkspace.x[0] + acadoWorkspace.evGx[573]*acadoWorkspace.x[1] + acadoWorkspace.evGx[574]*acadoWorkspace.x[2] + acadoWorkspace.evGx[575]*acadoWorkspace.x[3] + acadoWorkspace.d[143];
acadoVariables.x[148] += + acadoWorkspace.evGx[576]*acadoWorkspace.x[0] + acadoWorkspace.evGx[577]*acadoWorkspace.x[1] + acadoWorkspace.evGx[578]*acadoWorkspace.x[2] + acadoWorkspace.evGx[579]*acadoWorkspace.x[3] + acadoWorkspace.d[144];
acadoVariables.x[149] += + acadoWorkspace.evGx[580]*acadoWorkspace.x[0] + acadoWorkspace.evGx[581]*acadoWorkspace.x[1] + acadoWorkspace.evGx[582]*acadoWorkspace.x[2] + acadoWorkspace.evGx[583]*acadoWorkspace.x[3] + acadoWorkspace.d[145];
acadoVariables.x[150] += + acadoWorkspace.evGx[584]*acadoWorkspace.x[0] + acadoWorkspace.evGx[585]*acadoWorkspace.x[1] + acadoWorkspace.evGx[586]*acadoWorkspace.x[2] + acadoWorkspace.evGx[587]*acadoWorkspace.x[3] + acadoWorkspace.d[146];
acadoVariables.x[151] += + acadoWorkspace.evGx[588]*acadoWorkspace.x[0] + acadoWorkspace.evGx[589]*acadoWorkspace.x[1] + acadoWorkspace.evGx[590]*acadoWorkspace.x[2] + acadoWorkspace.evGx[591]*acadoWorkspace.x[3] + acadoWorkspace.d[147];
acadoVariables.x[152] += + acadoWorkspace.evGx[592]*acadoWorkspace.x[0] + acadoWorkspace.evGx[593]*acadoWorkspace.x[1] + acadoWorkspace.evGx[594]*acadoWorkspace.x[2] + acadoWorkspace.evGx[595]*acadoWorkspace.x[3] + acadoWorkspace.d[148];
acadoVariables.x[153] += + acadoWorkspace.evGx[596]*acadoWorkspace.x[0] + acadoWorkspace.evGx[597]*acadoWorkspace.x[1] + acadoWorkspace.evGx[598]*acadoWorkspace.x[2] + acadoWorkspace.evGx[599]*acadoWorkspace.x[3] + acadoWorkspace.d[149];
acadoVariables.x[154] += + acadoWorkspace.evGx[600]*acadoWorkspace.x[0] + acadoWorkspace.evGx[601]*acadoWorkspace.x[1] + acadoWorkspace.evGx[602]*acadoWorkspace.x[2] + acadoWorkspace.evGx[603]*acadoWorkspace.x[3] + acadoWorkspace.d[150];
acadoVariables.x[155] += + acadoWorkspace.evGx[604]*acadoWorkspace.x[0] + acadoWorkspace.evGx[605]*acadoWorkspace.x[1] + acadoWorkspace.evGx[606]*acadoWorkspace.x[2] + acadoWorkspace.evGx[607]*acadoWorkspace.x[3] + acadoWorkspace.d[151];
acadoVariables.x[156] += + acadoWorkspace.evGx[608]*acadoWorkspace.x[0] + acadoWorkspace.evGx[609]*acadoWorkspace.x[1] + acadoWorkspace.evGx[610]*acadoWorkspace.x[2] + acadoWorkspace.evGx[611]*acadoWorkspace.x[3] + acadoWorkspace.d[152];
acadoVariables.x[157] += + acadoWorkspace.evGx[612]*acadoWorkspace.x[0] + acadoWorkspace.evGx[613]*acadoWorkspace.x[1] + acadoWorkspace.evGx[614]*acadoWorkspace.x[2] + acadoWorkspace.evGx[615]*acadoWorkspace.x[3] + acadoWorkspace.d[153];
acadoVariables.x[158] += + acadoWorkspace.evGx[616]*acadoWorkspace.x[0] + acadoWorkspace.evGx[617]*acadoWorkspace.x[1] + acadoWorkspace.evGx[618]*acadoWorkspace.x[2] + acadoWorkspace.evGx[619]*acadoWorkspace.x[3] + acadoWorkspace.d[154];
acadoVariables.x[159] += + acadoWorkspace.evGx[620]*acadoWorkspace.x[0] + acadoWorkspace.evGx[621]*acadoWorkspace.x[1] + acadoWorkspace.evGx[622]*acadoWorkspace.x[2] + acadoWorkspace.evGx[623]*acadoWorkspace.x[3] + acadoWorkspace.d[155];
acadoVariables.x[160] += + acadoWorkspace.evGx[624]*acadoWorkspace.x[0] + acadoWorkspace.evGx[625]*acadoWorkspace.x[1] + acadoWorkspace.evGx[626]*acadoWorkspace.x[2] + acadoWorkspace.evGx[627]*acadoWorkspace.x[3] + acadoWorkspace.d[156];
acadoVariables.x[161] += + acadoWorkspace.evGx[628]*acadoWorkspace.x[0] + acadoWorkspace.evGx[629]*acadoWorkspace.x[1] + acadoWorkspace.evGx[630]*acadoWorkspace.x[2] + acadoWorkspace.evGx[631]*acadoWorkspace.x[3] + acadoWorkspace.d[157];
acadoVariables.x[162] += + acadoWorkspace.evGx[632]*acadoWorkspace.x[0] + acadoWorkspace.evGx[633]*acadoWorkspace.x[1] + acadoWorkspace.evGx[634]*acadoWorkspace.x[2] + acadoWorkspace.evGx[635]*acadoWorkspace.x[3] + acadoWorkspace.d[158];
acadoVariables.x[163] += + acadoWorkspace.evGx[636]*acadoWorkspace.x[0] + acadoWorkspace.evGx[637]*acadoWorkspace.x[1] + acadoWorkspace.evGx[638]*acadoWorkspace.x[2] + acadoWorkspace.evGx[639]*acadoWorkspace.x[3] + acadoWorkspace.d[159];
acadoVariables.x[164] += + acadoWorkspace.evGx[640]*acadoWorkspace.x[0] + acadoWorkspace.evGx[641]*acadoWorkspace.x[1] + acadoWorkspace.evGx[642]*acadoWorkspace.x[2] + acadoWorkspace.evGx[643]*acadoWorkspace.x[3] + acadoWorkspace.d[160];
acadoVariables.x[165] += + acadoWorkspace.evGx[644]*acadoWorkspace.x[0] + acadoWorkspace.evGx[645]*acadoWorkspace.x[1] + acadoWorkspace.evGx[646]*acadoWorkspace.x[2] + acadoWorkspace.evGx[647]*acadoWorkspace.x[3] + acadoWorkspace.d[161];
acadoVariables.x[166] += + acadoWorkspace.evGx[648]*acadoWorkspace.x[0] + acadoWorkspace.evGx[649]*acadoWorkspace.x[1] + acadoWorkspace.evGx[650]*acadoWorkspace.x[2] + acadoWorkspace.evGx[651]*acadoWorkspace.x[3] + acadoWorkspace.d[162];
acadoVariables.x[167] += + acadoWorkspace.evGx[652]*acadoWorkspace.x[0] + acadoWorkspace.evGx[653]*acadoWorkspace.x[1] + acadoWorkspace.evGx[654]*acadoWorkspace.x[2] + acadoWorkspace.evGx[655]*acadoWorkspace.x[3] + acadoWorkspace.d[163];
acadoVariables.x[168] += + acadoWorkspace.evGx[656]*acadoWorkspace.x[0] + acadoWorkspace.evGx[657]*acadoWorkspace.x[1] + acadoWorkspace.evGx[658]*acadoWorkspace.x[2] + acadoWorkspace.evGx[659]*acadoWorkspace.x[3] + acadoWorkspace.d[164];
acadoVariables.x[169] += + acadoWorkspace.evGx[660]*acadoWorkspace.x[0] + acadoWorkspace.evGx[661]*acadoWorkspace.x[1] + acadoWorkspace.evGx[662]*acadoWorkspace.x[2] + acadoWorkspace.evGx[663]*acadoWorkspace.x[3] + acadoWorkspace.d[165];
acadoVariables.x[170] += + acadoWorkspace.evGx[664]*acadoWorkspace.x[0] + acadoWorkspace.evGx[665]*acadoWorkspace.x[1] + acadoWorkspace.evGx[666]*acadoWorkspace.x[2] + acadoWorkspace.evGx[667]*acadoWorkspace.x[3] + acadoWorkspace.d[166];
acadoVariables.x[171] += + acadoWorkspace.evGx[668]*acadoWorkspace.x[0] + acadoWorkspace.evGx[669]*acadoWorkspace.x[1] + acadoWorkspace.evGx[670]*acadoWorkspace.x[2] + acadoWorkspace.evGx[671]*acadoWorkspace.x[3] + acadoWorkspace.d[167];
acadoVariables.x[172] += + acadoWorkspace.evGx[672]*acadoWorkspace.x[0] + acadoWorkspace.evGx[673]*acadoWorkspace.x[1] + acadoWorkspace.evGx[674]*acadoWorkspace.x[2] + acadoWorkspace.evGx[675]*acadoWorkspace.x[3] + acadoWorkspace.d[168];
acadoVariables.x[173] += + acadoWorkspace.evGx[676]*acadoWorkspace.x[0] + acadoWorkspace.evGx[677]*acadoWorkspace.x[1] + acadoWorkspace.evGx[678]*acadoWorkspace.x[2] + acadoWorkspace.evGx[679]*acadoWorkspace.x[3] + acadoWorkspace.d[169];
acadoVariables.x[174] += + acadoWorkspace.evGx[680]*acadoWorkspace.x[0] + acadoWorkspace.evGx[681]*acadoWorkspace.x[1] + acadoWorkspace.evGx[682]*acadoWorkspace.x[2] + acadoWorkspace.evGx[683]*acadoWorkspace.x[3] + acadoWorkspace.d[170];
acadoVariables.x[175] += + acadoWorkspace.evGx[684]*acadoWorkspace.x[0] + acadoWorkspace.evGx[685]*acadoWorkspace.x[1] + acadoWorkspace.evGx[686]*acadoWorkspace.x[2] + acadoWorkspace.evGx[687]*acadoWorkspace.x[3] + acadoWorkspace.d[171];
acadoVariables.x[176] += + acadoWorkspace.evGx[688]*acadoWorkspace.x[0] + acadoWorkspace.evGx[689]*acadoWorkspace.x[1] + acadoWorkspace.evGx[690]*acadoWorkspace.x[2] + acadoWorkspace.evGx[691]*acadoWorkspace.x[3] + acadoWorkspace.d[172];
acadoVariables.x[177] += + acadoWorkspace.evGx[692]*acadoWorkspace.x[0] + acadoWorkspace.evGx[693]*acadoWorkspace.x[1] + acadoWorkspace.evGx[694]*acadoWorkspace.x[2] + acadoWorkspace.evGx[695]*acadoWorkspace.x[3] + acadoWorkspace.d[173];
acadoVariables.x[178] += + acadoWorkspace.evGx[696]*acadoWorkspace.x[0] + acadoWorkspace.evGx[697]*acadoWorkspace.x[1] + acadoWorkspace.evGx[698]*acadoWorkspace.x[2] + acadoWorkspace.evGx[699]*acadoWorkspace.x[3] + acadoWorkspace.d[174];
acadoVariables.x[179] += + acadoWorkspace.evGx[700]*acadoWorkspace.x[0] + acadoWorkspace.evGx[701]*acadoWorkspace.x[1] + acadoWorkspace.evGx[702]*acadoWorkspace.x[2] + acadoWorkspace.evGx[703]*acadoWorkspace.x[3] + acadoWorkspace.d[175];
acadoVariables.x[180] += + acadoWorkspace.evGx[704]*acadoWorkspace.x[0] + acadoWorkspace.evGx[705]*acadoWorkspace.x[1] + acadoWorkspace.evGx[706]*acadoWorkspace.x[2] + acadoWorkspace.evGx[707]*acadoWorkspace.x[3] + acadoWorkspace.d[176];
acadoVariables.x[181] += + acadoWorkspace.evGx[708]*acadoWorkspace.x[0] + acadoWorkspace.evGx[709]*acadoWorkspace.x[1] + acadoWorkspace.evGx[710]*acadoWorkspace.x[2] + acadoWorkspace.evGx[711]*acadoWorkspace.x[3] + acadoWorkspace.d[177];
acadoVariables.x[182] += + acadoWorkspace.evGx[712]*acadoWorkspace.x[0] + acadoWorkspace.evGx[713]*acadoWorkspace.x[1] + acadoWorkspace.evGx[714]*acadoWorkspace.x[2] + acadoWorkspace.evGx[715]*acadoWorkspace.x[3] + acadoWorkspace.d[178];
acadoVariables.x[183] += + acadoWorkspace.evGx[716]*acadoWorkspace.x[0] + acadoWorkspace.evGx[717]*acadoWorkspace.x[1] + acadoWorkspace.evGx[718]*acadoWorkspace.x[2] + acadoWorkspace.evGx[719]*acadoWorkspace.x[3] + acadoWorkspace.d[179];
acadoVariables.x[184] += + acadoWorkspace.evGx[720]*acadoWorkspace.x[0] + acadoWorkspace.evGx[721]*acadoWorkspace.x[1] + acadoWorkspace.evGx[722]*acadoWorkspace.x[2] + acadoWorkspace.evGx[723]*acadoWorkspace.x[3] + acadoWorkspace.d[180];
acadoVariables.x[185] += + acadoWorkspace.evGx[724]*acadoWorkspace.x[0] + acadoWorkspace.evGx[725]*acadoWorkspace.x[1] + acadoWorkspace.evGx[726]*acadoWorkspace.x[2] + acadoWorkspace.evGx[727]*acadoWorkspace.x[3] + acadoWorkspace.d[181];
acadoVariables.x[186] += + acadoWorkspace.evGx[728]*acadoWorkspace.x[0] + acadoWorkspace.evGx[729]*acadoWorkspace.x[1] + acadoWorkspace.evGx[730]*acadoWorkspace.x[2] + acadoWorkspace.evGx[731]*acadoWorkspace.x[3] + acadoWorkspace.d[182];
acadoVariables.x[187] += + acadoWorkspace.evGx[732]*acadoWorkspace.x[0] + acadoWorkspace.evGx[733]*acadoWorkspace.x[1] + acadoWorkspace.evGx[734]*acadoWorkspace.x[2] + acadoWorkspace.evGx[735]*acadoWorkspace.x[3] + acadoWorkspace.d[183];
acadoVariables.x[188] += + acadoWorkspace.evGx[736]*acadoWorkspace.x[0] + acadoWorkspace.evGx[737]*acadoWorkspace.x[1] + acadoWorkspace.evGx[738]*acadoWorkspace.x[2] + acadoWorkspace.evGx[739]*acadoWorkspace.x[3] + acadoWorkspace.d[184];
acadoVariables.x[189] += + acadoWorkspace.evGx[740]*acadoWorkspace.x[0] + acadoWorkspace.evGx[741]*acadoWorkspace.x[1] + acadoWorkspace.evGx[742]*acadoWorkspace.x[2] + acadoWorkspace.evGx[743]*acadoWorkspace.x[3] + acadoWorkspace.d[185];
acadoVariables.x[190] += + acadoWorkspace.evGx[744]*acadoWorkspace.x[0] + acadoWorkspace.evGx[745]*acadoWorkspace.x[1] + acadoWorkspace.evGx[746]*acadoWorkspace.x[2] + acadoWorkspace.evGx[747]*acadoWorkspace.x[3] + acadoWorkspace.d[186];
acadoVariables.x[191] += + acadoWorkspace.evGx[748]*acadoWorkspace.x[0] + acadoWorkspace.evGx[749]*acadoWorkspace.x[1] + acadoWorkspace.evGx[750]*acadoWorkspace.x[2] + acadoWorkspace.evGx[751]*acadoWorkspace.x[3] + acadoWorkspace.d[187];
acadoVariables.x[192] += + acadoWorkspace.evGx[752]*acadoWorkspace.x[0] + acadoWorkspace.evGx[753]*acadoWorkspace.x[1] + acadoWorkspace.evGx[754]*acadoWorkspace.x[2] + acadoWorkspace.evGx[755]*acadoWorkspace.x[3] + acadoWorkspace.d[188];
acadoVariables.x[193] += + acadoWorkspace.evGx[756]*acadoWorkspace.x[0] + acadoWorkspace.evGx[757]*acadoWorkspace.x[1] + acadoWorkspace.evGx[758]*acadoWorkspace.x[2] + acadoWorkspace.evGx[759]*acadoWorkspace.x[3] + acadoWorkspace.d[189];
acadoVariables.x[194] += + acadoWorkspace.evGx[760]*acadoWorkspace.x[0] + acadoWorkspace.evGx[761]*acadoWorkspace.x[1] + acadoWorkspace.evGx[762]*acadoWorkspace.x[2] + acadoWorkspace.evGx[763]*acadoWorkspace.x[3] + acadoWorkspace.d[190];
acadoVariables.x[195] += + acadoWorkspace.evGx[764]*acadoWorkspace.x[0] + acadoWorkspace.evGx[765]*acadoWorkspace.x[1] + acadoWorkspace.evGx[766]*acadoWorkspace.x[2] + acadoWorkspace.evGx[767]*acadoWorkspace.x[3] + acadoWorkspace.d[191];
acadoVariables.x[196] += + acadoWorkspace.evGx[768]*acadoWorkspace.x[0] + acadoWorkspace.evGx[769]*acadoWorkspace.x[1] + acadoWorkspace.evGx[770]*acadoWorkspace.x[2] + acadoWorkspace.evGx[771]*acadoWorkspace.x[3] + acadoWorkspace.d[192];
acadoVariables.x[197] += + acadoWorkspace.evGx[772]*acadoWorkspace.x[0] + acadoWorkspace.evGx[773]*acadoWorkspace.x[1] + acadoWorkspace.evGx[774]*acadoWorkspace.x[2] + acadoWorkspace.evGx[775]*acadoWorkspace.x[3] + acadoWorkspace.d[193];
acadoVariables.x[198] += + acadoWorkspace.evGx[776]*acadoWorkspace.x[0] + acadoWorkspace.evGx[777]*acadoWorkspace.x[1] + acadoWorkspace.evGx[778]*acadoWorkspace.x[2] + acadoWorkspace.evGx[779]*acadoWorkspace.x[3] + acadoWorkspace.d[194];
acadoVariables.x[199] += + acadoWorkspace.evGx[780]*acadoWorkspace.x[0] + acadoWorkspace.evGx[781]*acadoWorkspace.x[1] + acadoWorkspace.evGx[782]*acadoWorkspace.x[2] + acadoWorkspace.evGx[783]*acadoWorkspace.x[3] + acadoWorkspace.d[195];
acadoVariables.x[200] += + acadoWorkspace.evGx[784]*acadoWorkspace.x[0] + acadoWorkspace.evGx[785]*acadoWorkspace.x[1] + acadoWorkspace.evGx[786]*acadoWorkspace.x[2] + acadoWorkspace.evGx[787]*acadoWorkspace.x[3] + acadoWorkspace.d[196];
acadoVariables.x[201] += + acadoWorkspace.evGx[788]*acadoWorkspace.x[0] + acadoWorkspace.evGx[789]*acadoWorkspace.x[1] + acadoWorkspace.evGx[790]*acadoWorkspace.x[2] + acadoWorkspace.evGx[791]*acadoWorkspace.x[3] + acadoWorkspace.d[197];
acadoVariables.x[202] += + acadoWorkspace.evGx[792]*acadoWorkspace.x[0] + acadoWorkspace.evGx[793]*acadoWorkspace.x[1] + acadoWorkspace.evGx[794]*acadoWorkspace.x[2] + acadoWorkspace.evGx[795]*acadoWorkspace.x[3] + acadoWorkspace.d[198];
acadoVariables.x[203] += + acadoWorkspace.evGx[796]*acadoWorkspace.x[0] + acadoWorkspace.evGx[797]*acadoWorkspace.x[1] + acadoWorkspace.evGx[798]*acadoWorkspace.x[2] + acadoWorkspace.evGx[799]*acadoWorkspace.x[3] + acadoWorkspace.d[199];

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1 + 1; ++lRun2)
{
lRun3 = (((lRun1 + 1) * (lRun1)) / (2)) + (lRun2);
acado_multEDu( &(acadoWorkspace.E[ lRun3 * 4 ]), &(acadoWorkspace.x[ lRun2 + 4 ]), &(acadoVariables.x[ lRun1 * 4 + 4 ]) );
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
for (index = 0; index < 50; ++index)
{
acadoWorkspace.state[0] = acadoVariables.x[index * 4];
acadoWorkspace.state[1] = acadoVariables.x[index * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 4 + 3];
acadoWorkspace.state[24] = acadoVariables.u[index];
acadoWorkspace.state[25] = acadoVariables.od[index * 18];
acadoWorkspace.state[26] = acadoVariables.od[index * 18 + 1];
acadoWorkspace.state[27] = acadoVariables.od[index * 18 + 2];
acadoWorkspace.state[28] = acadoVariables.od[index * 18 + 3];
acadoWorkspace.state[29] = acadoVariables.od[index * 18 + 4];
acadoWorkspace.state[30] = acadoVariables.od[index * 18 + 5];
acadoWorkspace.state[31] = acadoVariables.od[index * 18 + 6];
acadoWorkspace.state[32] = acadoVariables.od[index * 18 + 7];
acadoWorkspace.state[33] = acadoVariables.od[index * 18 + 8];
acadoWorkspace.state[34] = acadoVariables.od[index * 18 + 9];
acadoWorkspace.state[35] = acadoVariables.od[index * 18 + 10];
acadoWorkspace.state[36] = acadoVariables.od[index * 18 + 11];
acadoWorkspace.state[37] = acadoVariables.od[index * 18 + 12];
acadoWorkspace.state[38] = acadoVariables.od[index * 18 + 13];
acadoWorkspace.state[39] = acadoVariables.od[index * 18 + 14];
acadoWorkspace.state[40] = acadoVariables.od[index * 18 + 15];
acadoWorkspace.state[41] = acadoVariables.od[index * 18 + 16];
acadoWorkspace.state[42] = acadoVariables.od[index * 18 + 17];

acado_integrate(acadoWorkspace.state, index == 0);

acadoVariables.x[index * 4 + 4] = acadoWorkspace.state[0];
acadoVariables.x[index * 4 + 5] = acadoWorkspace.state[1];
acadoVariables.x[index * 4 + 6] = acadoWorkspace.state[2];
acadoVariables.x[index * 4 + 7] = acadoWorkspace.state[3];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 50; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[200] = xEnd[0];
acadoVariables.x[201] = xEnd[1];
acadoVariables.x[202] = xEnd[2];
acadoVariables.x[203] = xEnd[3];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[200];
acadoWorkspace.state[1] = acadoVariables.x[201];
acadoWorkspace.state[2] = acadoVariables.x[202];
acadoWorkspace.state[3] = acadoVariables.x[203];
if (uEnd != 0)
{
acadoWorkspace.state[24] = uEnd[0];
}
else
{
acadoWorkspace.state[24] = acadoVariables.u[49];
}
acadoWorkspace.state[25] = acadoVariables.od[900];
acadoWorkspace.state[26] = acadoVariables.od[901];
acadoWorkspace.state[27] = acadoVariables.od[902];
acadoWorkspace.state[28] = acadoVariables.od[903];
acadoWorkspace.state[29] = acadoVariables.od[904];
acadoWorkspace.state[30] = acadoVariables.od[905];
acadoWorkspace.state[31] = acadoVariables.od[906];
acadoWorkspace.state[32] = acadoVariables.od[907];
acadoWorkspace.state[33] = acadoVariables.od[908];
acadoWorkspace.state[34] = acadoVariables.od[909];
acadoWorkspace.state[35] = acadoVariables.od[910];
acadoWorkspace.state[36] = acadoVariables.od[911];
acadoWorkspace.state[37] = acadoVariables.od[912];
acadoWorkspace.state[38] = acadoVariables.od[913];
acadoWorkspace.state[39] = acadoVariables.od[914];
acadoWorkspace.state[40] = acadoVariables.od[915];
acadoWorkspace.state[41] = acadoVariables.od[916];
acadoWorkspace.state[42] = acadoVariables.od[917];

acado_integrate(acadoWorkspace.state, 1);

acadoVariables.x[200] = acadoWorkspace.state[0];
acadoVariables.x[201] = acadoWorkspace.state[1];
acadoVariables.x[202] = acadoWorkspace.state[2];
acadoVariables.x[203] = acadoWorkspace.state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 49; ++index)
{
acadoVariables.u[index] = acadoVariables.u[index + 1];
}

if (uEnd != 0)
{
acadoVariables.u[49] = uEnd[0];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43] + acadoWorkspace.g[44]*acadoWorkspace.x[44] + acadoWorkspace.g[45]*acadoWorkspace.x[45] + acadoWorkspace.g[46]*acadoWorkspace.x[46] + acadoWorkspace.g[47]*acadoWorkspace.x[47] + acadoWorkspace.g[48]*acadoWorkspace.x[48] + acadoWorkspace.g[49]*acadoWorkspace.x[49] + acadoWorkspace.g[50]*acadoWorkspace.x[50] + acadoWorkspace.g[51]*acadoWorkspace.x[51] + acadoWorkspace.g[52]*acadoWorkspace.x[52] + acadoWorkspace.g[53]*acadoWorkspace.x[53];
kkt = fabs( kkt );
for (index = 0; index < 54; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 100; ++index)
{
prd = acadoWorkspace.y[index + 54];
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

/** Row vector of size: 4 */
real_t tmpDyN[ 4 ];

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[lRun1];
acadoWorkspace.objValueIn[5] = acadoVariables.od[lRun1 * 18];
acadoWorkspace.objValueIn[6] = acadoVariables.od[lRun1 * 18 + 1];
acadoWorkspace.objValueIn[7] = acadoVariables.od[lRun1 * 18 + 2];
acadoWorkspace.objValueIn[8] = acadoVariables.od[lRun1 * 18 + 3];
acadoWorkspace.objValueIn[9] = acadoVariables.od[lRun1 * 18 + 4];
acadoWorkspace.objValueIn[10] = acadoVariables.od[lRun1 * 18 + 5];
acadoWorkspace.objValueIn[11] = acadoVariables.od[lRun1 * 18 + 6];
acadoWorkspace.objValueIn[12] = acadoVariables.od[lRun1 * 18 + 7];
acadoWorkspace.objValueIn[13] = acadoVariables.od[lRun1 * 18 + 8];
acadoWorkspace.objValueIn[14] = acadoVariables.od[lRun1 * 18 + 9];
acadoWorkspace.objValueIn[15] = acadoVariables.od[lRun1 * 18 + 10];
acadoWorkspace.objValueIn[16] = acadoVariables.od[lRun1 * 18 + 11];
acadoWorkspace.objValueIn[17] = acadoVariables.od[lRun1 * 18 + 12];
acadoWorkspace.objValueIn[18] = acadoVariables.od[lRun1 * 18 + 13];
acadoWorkspace.objValueIn[19] = acadoVariables.od[lRun1 * 18 + 14];
acadoWorkspace.objValueIn[20] = acadoVariables.od[lRun1 * 18 + 15];
acadoWorkspace.objValueIn[21] = acadoVariables.od[lRun1 * 18 + 16];
acadoWorkspace.objValueIn[22] = acadoVariables.od[lRun1 * 18 + 17];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 5] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 5];
acadoWorkspace.Dy[lRun1 * 5 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 5 + 1];
acadoWorkspace.Dy[lRun1 * 5 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 5 + 2];
acadoWorkspace.Dy[lRun1 * 5 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 5 + 3];
acadoWorkspace.Dy[lRun1 * 5 + 4] = acadoWorkspace.objValueOut[4] - acadoVariables.y[lRun1 * 5 + 4];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[200];
acadoWorkspace.objValueIn[1] = acadoVariables.x[201];
acadoWorkspace.objValueIn[2] = acadoVariables.x[202];
acadoWorkspace.objValueIn[3] = acadoVariables.x[203];
acadoWorkspace.objValueIn[4] = acadoVariables.od[900];
acadoWorkspace.objValueIn[5] = acadoVariables.od[901];
acadoWorkspace.objValueIn[6] = acadoVariables.od[902];
acadoWorkspace.objValueIn[7] = acadoVariables.od[903];
acadoWorkspace.objValueIn[8] = acadoVariables.od[904];
acadoWorkspace.objValueIn[9] = acadoVariables.od[905];
acadoWorkspace.objValueIn[10] = acadoVariables.od[906];
acadoWorkspace.objValueIn[11] = acadoVariables.od[907];
acadoWorkspace.objValueIn[12] = acadoVariables.od[908];
acadoWorkspace.objValueIn[13] = acadoVariables.od[909];
acadoWorkspace.objValueIn[14] = acadoVariables.od[910];
acadoWorkspace.objValueIn[15] = acadoVariables.od[911];
acadoWorkspace.objValueIn[16] = acadoVariables.od[912];
acadoWorkspace.objValueIn[17] = acadoVariables.od[913];
acadoWorkspace.objValueIn[18] = acadoVariables.od[914];
acadoWorkspace.objValueIn[19] = acadoVariables.od[915];
acadoWorkspace.objValueIn[20] = acadoVariables.od[916];
acadoWorkspace.objValueIn[21] = acadoVariables.od[917];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3] - acadoVariables.yN[3];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 5];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 5 + 1];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 5 + 2];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 5 + 3];
tmpDy[4] = + acadoWorkspace.Dy[lRun1 * 5 + 4]*(real_t)5.0000000000000000e-01;
objVal += + acadoWorkspace.Dy[lRun1 * 5]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 5 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 5 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 5 + 3]*tmpDy[3] + acadoWorkspace.Dy[lRun1 * 5 + 4]*tmpDy[4];
}

tmpDyN[0] = + acadoWorkspace.DyN[0];
tmpDyN[1] = + acadoWorkspace.DyN[1];
tmpDyN[2] = + acadoWorkspace.DyN[2];
tmpDyN[3] = + acadoWorkspace.DyN[3];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2] + acadoWorkspace.DyN[3]*tmpDyN[3];

objVal *= 0.5;
return objVal;
}

