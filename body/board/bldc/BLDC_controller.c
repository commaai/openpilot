/*
 * File: BLDC_controller.c
 *
 * Code generated for Simulink model 'BLDC_controller'.
 *
 * Model version                  : 1.1297
 * Simulink Coder version         : 8.13 (R2017b) 24-Jul-2017
 * C/C++ source code generated on : Sun Mar  6 11:02:11 2022
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: ARM Compatible->ARM Cortex
 * Emulation hardware selection:
 *    Differs from embedded hardware (MATLAB Host)
 * Code generation objectives:
 *    1. Execution efficiency
 *    2. RAM efficiency
 * Validation result: Not run
 */

#include "BLDC_controller.h"

/* Named constants for Chart: '<S5>/F03_02_Control_Mode_Manager' */
#define IN_ACTIVE                      ((uint8_T)1U)
#define IN_NO_ACTIVE_CHILD             ((uint8_T)0U)
#define IN_OPEN                        ((uint8_T)2U)
#define IN_SPEED_MODE                  ((uint8_T)1U)
#define IN_TORQUE_MODE                 ((uint8_T)2U)
#define IN_VOLTAGE_MODE                ((uint8_T)3U)
#define OPEN_MODE                      ((uint8_T)0U)
#define SPD_MODE                       ((uint8_T)2U)
#define TRQ_MODE                       ((uint8_T)3U)
#define VLT_MODE                       ((uint8_T)1U)
#ifndef UCHAR_MAX
#include <limits.h>
#endif

#if ( UCHAR_MAX != (0xFFU) ) || ( SCHAR_MAX != (0x7F) )
#error Code was generated for compiler with different sized uchar/char. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if ( USHRT_MAX != (0xFFFFU) ) || ( SHRT_MAX != (0x7FFF) )
#error Code was generated for compiler with different sized ushort/short. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if ( UINT_MAX != (0xFFFFFFFFU) ) || ( INT_MAX != (0x7FFFFFFF) )
#error Code was generated for compiler with different sized uint/int. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if ( ULONG_MAX != (0xFFFFFFFFU) ) || ( LONG_MAX != (0x7FFFFFFF) )
#error Code was generated for compiler with different sized ulong/long. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif

#if 0

/* Skip this size verification because of preprocessor limitation */
#if ( ULLONG_MAX != (0xFFFFFFFFFFFFFFFFULL) ) || ( LLONG_MAX != (0x7FFFFFFFFFFFFFFFLL) )
#error Code was generated for compiler with different sized ulong_long/long_long. \
Consider adjusting Test hardware word size settings on the \
Hardware Implementation pane to match your compiler word sizes as \
defined in limits.h of the compiler. Alternatively, you can \
select the Test hardware is the same as production hardware option and \
select the Enable portable word sizes option on the Code Generation > \
Verification pane for ERT based targets, which will disable the \
preprocessor word size checks.
#endif
#endif

uint8_T plook_u8s16_evencka(int16_T u, int16_T bp0, uint16_T bpSpace, uint32_T
  maxIndex);
uint8_T plook_u8u16_evencka(uint16_T u, uint16_T bp0, uint16_T bpSpace, uint32_T
  maxIndex);
int32_T div_nde_s32_floor(int32_T numerator, int32_T denominator);
extern void Counter_Init(DW_Counter *localDW, int16_T rtp_z_cntInit);
extern int16_T Counter(int16_T rtu_inc, int16_T rtu_max, boolean_T rtu_rst,
  DW_Counter *localDW);
extern void Low_Pass_Filter_Reset(DW_Low_Pass_Filter *localDW);
extern void Low_Pass_Filter(const int16_T rtu_u[2], uint16_T rtu_coef, int16_T
  rty_y[2], DW_Low_Pass_Filter *localDW);
extern void Counter_b_Init(DW_Counter_b *localDW, uint16_T rtp_z_cntInit);
extern void Counter_n(uint16_T rtu_inc, uint16_T rtu_max, boolean_T rtu_rst,
                      uint16_T *rty_cnt, DW_Counter_b *localDW);
extern void either_edge(boolean_T rtu_u, boolean_T *rty_y, DW_either_edge
  *localDW);
extern void Debounce_Filter_Init(DW_Debounce_Filter *localDW);
extern void Debounce_Filter(boolean_T rtu_u, uint16_T rtu_tAcv, uint16_T
  rtu_tDeacv, boolean_T *rty_y, DW_Debounce_Filter *localDW);
extern void I_backCalc_fixdt_Init(DW_I_backCalc_fixdt *localDW, int32_T
  rtp_yInit);
extern void I_backCalc_fixdt_Reset(DW_I_backCalc_fixdt *localDW, int32_T
  rtp_yInit);
extern void I_backCalc_fixdt(int16_T rtu_err, uint16_T rtu_I, uint16_T rtu_Kb,
  int16_T rtu_satMax, int16_T rtu_satMin, int16_T *rty_out, DW_I_backCalc_fixdt *
  localDW);
extern void PI_clamp_fixdt_Init(DW_PI_clamp_fixdt *localDW);
extern void PI_clamp_fixdt_Reset(DW_PI_clamp_fixdt *localDW);
extern void PI_clamp_fixdt(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I,
  int32_T rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
  rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt *localDW);
extern void PI_clamp_fixdt_d_Init(DW_PI_clamp_fixdt_m *localDW);
extern void PI_clamp_fixdt_b_Reset(DW_PI_clamp_fixdt_m *localDW);
extern void PI_clamp_fixdt_l(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I,
  int16_T rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
  rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt_m *localDW);
extern void PI_clamp_fixdt_f_Init(DW_PI_clamp_fixdt_g *localDW);
extern void PI_clamp_fixdt_g_Reset(DW_PI_clamp_fixdt_g *localDW);
extern void PI_clamp_fixdt_k(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I,
  int16_T rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
  rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt_g *localDW);
uint8_T plook_u8s16_evencka(int16_T u, int16_T bp0, uint16_T bpSpace, uint32_T
  maxIndex)
{
  uint8_T bpIndex;
  uint16_T fbpIndex;

  /* Prelookup - Index only
     Index Search method: 'even'
     Extrapolation method: 'Clip'
     Use previous index: 'off'
     Use last breakpoint for index at or above upper limit: 'on'
     Remove protection against out-of-range input in generated code: 'off'
   */
  if (u <= bp0) {
    bpIndex = 0U;
  } else {
    fbpIndex = (uint16_T)((uint32_T)(uint16_T)(u - bp0) / bpSpace);
    if (fbpIndex < maxIndex) {
      bpIndex = (uint8_T)fbpIndex;
    } else {
      bpIndex = (uint8_T)maxIndex;
    }
  }

  return bpIndex;
}

uint8_T plook_u8u16_evencka(uint16_T u, uint16_T bp0, uint16_T bpSpace, uint32_T
  maxIndex)
{
  uint8_T bpIndex;
  uint16_T fbpIndex;

  /* Prelookup - Index only
     Index Search method: 'even'
     Extrapolation method: 'Clip'
     Use previous index: 'off'
     Use last breakpoint for index at or above upper limit: 'on'
     Remove protection against out-of-range input in generated code: 'off'
   */
  if (u <= bp0) {
    bpIndex = 0U;
  } else {
    fbpIndex = (uint16_T)((uint32_T)(uint16_T)((uint32_T)u - bp0) / bpSpace);
    if (fbpIndex < maxIndex) {
      bpIndex = (uint8_T)fbpIndex;
    } else {
      bpIndex = (uint8_T)maxIndex;
    }
  }

  return bpIndex;
}

int32_T div_nde_s32_floor(int32_T numerator, int32_T denominator)
{
  return (((numerator < 0) != (denominator < 0)) && (numerator % denominator !=
           0) ? -1 : 0) + numerator / denominator;
}

/* System initialize for atomic system: '<S13>/Counter' */
void Counter_Init(DW_Counter *localDW, int16_T rtp_z_cntInit)
{
  /* InitializeConditions for UnitDelay: '<S18>/UnitDelay' */
  localDW->UnitDelay_DSTATE = rtp_z_cntInit;
}

/* Output and update for atomic system: '<S13>/Counter' */
int16_T Counter(int16_T rtu_inc, int16_T rtu_max, boolean_T rtu_rst, DW_Counter *
                localDW)
{
  int16_T rtu_rst_0;
  int16_T rty_cnt_0;

  /* Switch: '<S18>/Switch1' incorporates:
   *  Constant: '<S18>/Constant23'
   *  UnitDelay: '<S18>/UnitDelay'
   */
  if (rtu_rst) {
    rtu_rst_0 = 0;
  } else {
    rtu_rst_0 = localDW->UnitDelay_DSTATE;
  }

  /* End of Switch: '<S18>/Switch1' */

  /* Sum: '<S16>/Sum1' */
  rty_cnt_0 = (int16_T)(rtu_inc + rtu_rst_0);

  /* MinMax: '<S16>/MinMax' */
  if (rty_cnt_0 < rtu_max) {
    /* Update for UnitDelay: '<S18>/UnitDelay' */
    localDW->UnitDelay_DSTATE = rty_cnt_0;
  } else {
    /* Update for UnitDelay: '<S18>/UnitDelay' */
    localDW->UnitDelay_DSTATE = rtu_max;
  }

  /* End of MinMax: '<S16>/MinMax' */
  return rty_cnt_0;
}

/* System reset for atomic system: '<S50>/Low_Pass_Filter' */
void Low_Pass_Filter_Reset(DW_Low_Pass_Filter *localDW)
{
  /* InitializeConditions for UnitDelay: '<S56>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE[0] = 0;
  localDW->UnitDelay1_DSTATE[1] = 0;
}

/* Output and update for atomic system: '<S50>/Low_Pass_Filter' */
void Low_Pass_Filter(const int16_T rtu_u[2], uint16_T rtu_coef, int16_T rty_y[2],
                     DW_Low_Pass_Filter *localDW)
{
  int32_T rtb_Sum3_g;

  /* Sum: '<S56>/Sum2' incorporates:
   *  UnitDelay: '<S56>/UnitDelay1'
   */
  rtb_Sum3_g = rtu_u[0] - (localDW->UnitDelay1_DSTATE[0] >> 16);
  if (rtb_Sum3_g > 32767) {
    rtb_Sum3_g = 32767;
  } else {
    if (rtb_Sum3_g < -32768) {
      rtb_Sum3_g = -32768;
    }
  }

  /* Sum: '<S56>/Sum3' incorporates:
   *  Product: '<S56>/Divide3'
   *  Sum: '<S56>/Sum2'
   *  UnitDelay: '<S56>/UnitDelay1'
   */
  rtb_Sum3_g = rtu_coef * rtb_Sum3_g + localDW->UnitDelay1_DSTATE[0];

  /* DataTypeConversion: '<S56>/Data Type Conversion' */
  rty_y[0] = (int16_T)(rtb_Sum3_g >> 16);

  /* Update for UnitDelay: '<S56>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE[0] = rtb_Sum3_g;

  /* Sum: '<S56>/Sum2' incorporates:
   *  UnitDelay: '<S56>/UnitDelay1'
   */
  rtb_Sum3_g = rtu_u[1] - (localDW->UnitDelay1_DSTATE[1] >> 16);
  if (rtb_Sum3_g > 32767) {
    rtb_Sum3_g = 32767;
  } else {
    if (rtb_Sum3_g < -32768) {
      rtb_Sum3_g = -32768;
    }
  }

  /* Sum: '<S56>/Sum3' incorporates:
   *  Product: '<S56>/Divide3'
   *  Sum: '<S56>/Sum2'
   *  UnitDelay: '<S56>/UnitDelay1'
   */
  rtb_Sum3_g = rtu_coef * rtb_Sum3_g + localDW->UnitDelay1_DSTATE[1];

  /* DataTypeConversion: '<S56>/Data Type Conversion' */
  rty_y[1] = (int16_T)(rtb_Sum3_g >> 16);

  /* Update for UnitDelay: '<S56>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE[1] = rtb_Sum3_g;
}

/*
 * System initialize for atomic system:
 *    '<S25>/Counter'
 *    '<S24>/Counter'
 */
void Counter_b_Init(DW_Counter_b *localDW, uint16_T rtp_z_cntInit)
{
  /* InitializeConditions for UnitDelay: '<S30>/UnitDelay' */
  localDW->UnitDelay_DSTATE = rtp_z_cntInit;
}

/*
 * Output and update for atomic system:
 *    '<S25>/Counter'
 *    '<S24>/Counter'
 */
void Counter_n(uint16_T rtu_inc, uint16_T rtu_max, boolean_T rtu_rst, uint16_T
               *rty_cnt, DW_Counter_b *localDW)
{
  uint16_T rtu_rst_0;

  /* Switch: '<S30>/Switch1' incorporates:
   *  Constant: '<S30>/Constant23'
   *  UnitDelay: '<S30>/UnitDelay'
   */
  if (rtu_rst) {
    rtu_rst_0 = 0U;
  } else {
    rtu_rst_0 = localDW->UnitDelay_DSTATE;
  }

  /* End of Switch: '<S30>/Switch1' */

  /* Sum: '<S29>/Sum1' */
  *rty_cnt = (uint16_T)((uint32_T)rtu_inc + rtu_rst_0);

  /* MinMax: '<S29>/MinMax' */
  if (*rty_cnt < rtu_max) {
    /* Update for UnitDelay: '<S30>/UnitDelay' */
    localDW->UnitDelay_DSTATE = *rty_cnt;
  } else {
    /* Update for UnitDelay: '<S30>/UnitDelay' */
    localDW->UnitDelay_DSTATE = rtu_max;
  }

  /* End of MinMax: '<S29>/MinMax' */
}

/*
 * Output and update for atomic system:
 *    '<S21>/either_edge'
 *    '<S20>/either_edge'
 */
void either_edge(boolean_T rtu_u, boolean_T *rty_y, DW_either_edge *localDW)
{
  /* RelationalOperator: '<S26>/Relational Operator' incorporates:
   *  UnitDelay: '<S26>/UnitDelay'
   */
  *rty_y = (rtu_u != localDW->UnitDelay_DSTATE);

  /* Update for UnitDelay: '<S26>/UnitDelay' */
  localDW->UnitDelay_DSTATE = rtu_u;
}

/* System initialize for atomic system: '<S20>/Debounce_Filter' */
void Debounce_Filter_Init(DW_Debounce_Filter *localDW)
{
  /* SystemInitialize for IfAction SubSystem: '<S21>/Qualification' */

  /* SystemInitialize for Atomic SubSystem: '<S25>/Counter' */
  Counter_b_Init(&localDW->Counter_n1, 0U);

  /* End of SystemInitialize for SubSystem: '<S25>/Counter' */

  /* End of SystemInitialize for SubSystem: '<S21>/Qualification' */

  /* SystemInitialize for IfAction SubSystem: '<S21>/Dequalification' */

  /* SystemInitialize for Atomic SubSystem: '<S24>/Counter' */
  Counter_b_Init(&localDW->Counter_e, 0U);

  /* End of SystemInitialize for SubSystem: '<S24>/Counter' */

  /* End of SystemInitialize for SubSystem: '<S21>/Dequalification' */
}

/* Output and update for atomic system: '<S20>/Debounce_Filter' */
void Debounce_Filter(boolean_T rtu_u, uint16_T rtu_tAcv, uint16_T rtu_tDeacv,
                     boolean_T *rty_y, DW_Debounce_Filter *localDW)
{
  uint16_T rtb_Sum1_n;
  boolean_T rtb_RelationalOperator_g;

  /* Outputs for Atomic SubSystem: '<S21>/either_edge' */
  either_edge(rtu_u, &rtb_RelationalOperator_g, &localDW->either_edge_p);

  /* End of Outputs for SubSystem: '<S21>/either_edge' */

  /* If: '<S21>/If2' incorporates:
   *  Constant: '<S24>/Constant6'
   *  Constant: '<S25>/Constant6'
   *  Inport: '<S23>/yPrev'
   *  Logic: '<S21>/Logical Operator1'
   *  Logic: '<S21>/Logical Operator2'
   *  Logic: '<S21>/Logical Operator3'
   *  Logic: '<S21>/Logical Operator4'
   *  UnitDelay: '<S21>/UnitDelay'
   */
  if (rtu_u && (!localDW->UnitDelay_DSTATE)) {
    /* Outputs for IfAction SubSystem: '<S21>/Qualification' incorporates:
     *  ActionPort: '<S25>/Action Port'
     */

    /* Outputs for Atomic SubSystem: '<S25>/Counter' */
    Counter_n(1U, rtu_tAcv, rtb_RelationalOperator_g, &rtb_Sum1_n,
              &localDW->Counter_n1);

    /* End of Outputs for SubSystem: '<S25>/Counter' */

    /* Switch: '<S25>/Switch2' incorporates:
     *  Constant: '<S25>/Constant6'
     *  RelationalOperator: '<S25>/Relational Operator2'
     */
    *rty_y = ((rtb_Sum1_n > rtu_tAcv) || localDW->UnitDelay_DSTATE);

    /* End of Outputs for SubSystem: '<S21>/Qualification' */
  } else if ((!rtu_u) && localDW->UnitDelay_DSTATE) {
    /* Outputs for IfAction SubSystem: '<S21>/Dequalification' incorporates:
     *  ActionPort: '<S24>/Action Port'
     */

    /* Outputs for Atomic SubSystem: '<S24>/Counter' */
    Counter_n(1U, rtu_tDeacv, rtb_RelationalOperator_g, &rtb_Sum1_n,
              &localDW->Counter_e);

    /* End of Outputs for SubSystem: '<S24>/Counter' */

    /* Switch: '<S24>/Switch2' incorporates:
     *  Constant: '<S24>/Constant6'
     *  RelationalOperator: '<S24>/Relational Operator2'
     */
    *rty_y = ((!(rtb_Sum1_n > rtu_tDeacv)) && localDW->UnitDelay_DSTATE);

    /* End of Outputs for SubSystem: '<S21>/Dequalification' */
  } else {
    /* Outputs for IfAction SubSystem: '<S21>/Default' incorporates:
     *  ActionPort: '<S23>/Action Port'
     */
    *rty_y = localDW->UnitDelay_DSTATE;

    /* End of Outputs for SubSystem: '<S21>/Default' */
  }

  /* End of If: '<S21>/If2' */

  /* Update for UnitDelay: '<S21>/UnitDelay' */
  localDW->UnitDelay_DSTATE = *rty_y;
}

/*
 * System initialize for atomic system:
 *    '<S83>/I_backCalc_fixdt'
 *    '<S83>/I_backCalc_fixdt1'
 *    '<S82>/I_backCalc_fixdt'
 */
void I_backCalc_fixdt_Init(DW_I_backCalc_fixdt *localDW, int32_T rtp_yInit)
{
  /* InitializeConditions for UnitDelay: '<S90>/UnitDelay' */
  localDW->UnitDelay_DSTATE_m = rtp_yInit;
}

/*
 * System reset for atomic system:
 *    '<S83>/I_backCalc_fixdt'
 *    '<S83>/I_backCalc_fixdt1'
 *    '<S82>/I_backCalc_fixdt'
 */
void I_backCalc_fixdt_Reset(DW_I_backCalc_fixdt *localDW, int32_T rtp_yInit)
{
  /* InitializeConditions for UnitDelay: '<S88>/UnitDelay' */
  localDW->UnitDelay_DSTATE = 0;

  /* InitializeConditions for UnitDelay: '<S90>/UnitDelay' */
  localDW->UnitDelay_DSTATE_m = rtp_yInit;
}

/*
 * Output and update for atomic system:
 *    '<S83>/I_backCalc_fixdt'
 *    '<S83>/I_backCalc_fixdt1'
 *    '<S82>/I_backCalc_fixdt'
 */
void I_backCalc_fixdt(int16_T rtu_err, uint16_T rtu_I, uint16_T rtu_Kb, int16_T
                      rtu_satMax, int16_T rtu_satMin, int16_T *rty_out,
                      DW_I_backCalc_fixdt *localDW)
{
  int32_T rtb_Sum1_o;
  int16_T rtb_DataTypeConversion1_gf;

  /* Sum: '<S88>/Sum2' incorporates:
   *  Product: '<S88>/Divide2'
   *  UnitDelay: '<S88>/UnitDelay'
   */
  rtb_Sum1_o = (rtu_err * rtu_I) >> 4;
  if ((rtb_Sum1_o < 0) && (localDW->UnitDelay_DSTATE < MIN_int32_T - rtb_Sum1_o))
  {
    rtb_Sum1_o = MIN_int32_T;
  } else if ((rtb_Sum1_o > 0) && (localDW->UnitDelay_DSTATE > MAX_int32_T
              - rtb_Sum1_o)) {
    rtb_Sum1_o = MAX_int32_T;
  } else {
    rtb_Sum1_o += localDW->UnitDelay_DSTATE;
  }

  /* End of Sum: '<S88>/Sum2' */

  /* Sum: '<S90>/Sum1' incorporates:
   *  UnitDelay: '<S90>/UnitDelay'
   */
  rtb_Sum1_o += localDW->UnitDelay_DSTATE_m;

  /* DataTypeConversion: '<S90>/Data Type Conversion1' */
  rtb_DataTypeConversion1_gf = (int16_T)(rtb_Sum1_o >> 12);

  /* Switch: '<S91>/Switch2' incorporates:
   *  RelationalOperator: '<S91>/LowerRelop1'
   *  RelationalOperator: '<S91>/UpperRelop'
   *  Switch: '<S91>/Switch'
   */
  if (rtb_DataTypeConversion1_gf > rtu_satMax) {
    *rty_out = rtu_satMax;
  } else if (rtb_DataTypeConversion1_gf < rtu_satMin) {
    /* Switch: '<S91>/Switch' */
    *rty_out = rtu_satMin;
  } else {
    *rty_out = rtb_DataTypeConversion1_gf;
  }

  /* End of Switch: '<S91>/Switch2' */

  /* Update for UnitDelay: '<S88>/UnitDelay' incorporates:
   *  Product: '<S88>/Divide1'
   *  Sum: '<S88>/Sum3'
   */
  localDW->UnitDelay_DSTATE = (int16_T)(*rty_out - rtb_DataTypeConversion1_gf) *
    rtu_Kb;

  /* Update for UnitDelay: '<S90>/UnitDelay' */
  localDW->UnitDelay_DSTATE_m = rtb_Sum1_o;
}

/* System initialize for atomic system: '<S63>/PI_clamp_fixdt' */
void PI_clamp_fixdt_Init(DW_PI_clamp_fixdt *localDW)
{
  /* InitializeConditions for Delay: '<S77>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* System reset for atomic system: '<S63>/PI_clamp_fixdt' */
void PI_clamp_fixdt_Reset(DW_PI_clamp_fixdt *localDW)
{
  /* InitializeConditions for UnitDelay: '<S74>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE = false;

  /* InitializeConditions for Delay: '<S77>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* Output and update for atomic system: '<S63>/PI_clamp_fixdt' */
void PI_clamp_fixdt(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I, int32_T
                    rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
                    rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt
                    *localDW)
{
  boolean_T rtb_LowerRelop1_c0;
  boolean_T rtb_UpperRelop_f;
  int32_T rtb_Sum1_p0;
  int32_T q0;
  int32_T tmp;
  int16_T tmp_0;

  /* Sum: '<S74>/Sum2' incorporates:
   *  Product: '<S74>/Divide2'
   */
  q0 = rtu_err * rtu_I;
  if ((q0 < 0) && (rtu_ext_limProt < MIN_int32_T - q0)) {
    q0 = MIN_int32_T;
  } else if ((q0 > 0) && (rtu_ext_limProt > MAX_int32_T - q0)) {
    q0 = MAX_int32_T;
  } else {
    q0 += rtu_ext_limProt;
  }

  /* Delay: '<S77>/Resettable Delay' */
  if (localDW->icLoad != 0) {
    localDW->ResettableDelay_DSTATE = rtu_init;
  }

  /* Switch: '<S74>/Switch1' incorporates:
   *  Constant: '<S74>/Constant'
   *  Sum: '<S74>/Sum2'
   *  UnitDelay: '<S74>/UnitDelay1'
   */
  if (localDW->UnitDelay1_DSTATE) {
    tmp = 0;
  } else {
    tmp = q0;
  }

  /* End of Switch: '<S74>/Switch1' */

  /* Sum: '<S77>/Sum1' incorporates:
   *  Delay: '<S77>/Resettable Delay'
   */
  rtb_Sum1_p0 = tmp + localDW->ResettableDelay_DSTATE;

  /* Product: '<S74>/Divide5' */
  tmp = (rtu_err * rtu_P) >> 11;
  if (tmp > 32767) {
    tmp = 32767;
  } else {
    if (tmp < -32768) {
      tmp = -32768;
    }
  }

  /* Sum: '<S74>/Sum1' incorporates:
   *  DataTypeConversion: '<S77>/Data Type Conversion1'
   *  Product: '<S74>/Divide5'
   */
  tmp = (((rtb_Sum1_p0 >> 16) << 1) + tmp) >> 1;
  if (tmp > 32767) {
    tmp = 32767;
  } else {
    if (tmp < -32768) {
      tmp = -32768;
    }
  }

  /* RelationalOperator: '<S78>/LowerRelop1' incorporates:
   *  Sum: '<S74>/Sum1'
   */
  rtb_LowerRelop1_c0 = ((int16_T)tmp > rtu_satMax);

  /* RelationalOperator: '<S78>/UpperRelop' incorporates:
   *  Sum: '<S74>/Sum1'
   */
  rtb_UpperRelop_f = ((int16_T)tmp < rtu_satMin);

  /* Switch: '<S78>/Switch1' incorporates:
   *  Sum: '<S74>/Sum1'
   *  Switch: '<S78>/Switch3'
   */
  if (rtb_LowerRelop1_c0) {
    *rty_out = rtu_satMax;
  } else if (rtb_UpperRelop_f) {
    /* Switch: '<S78>/Switch3' */
    *rty_out = rtu_satMin;
  } else {
    *rty_out = (int16_T)tmp;
  }

  /* End of Switch: '<S78>/Switch1' */

  /* Signum: '<S76>/SignDeltaU2' incorporates:
   *  Sum: '<S74>/Sum2'
   */
  if (q0 < 0) {
    q0 = -1;
  } else {
    q0 = (q0 > 0);
  }

  /* End of Signum: '<S76>/SignDeltaU2' */

  /* Signum: '<S76>/SignDeltaU3' incorporates:
   *  Sum: '<S74>/Sum1'
   */
  if ((int16_T)tmp < 0) {
    tmp_0 = -1;
  } else {
    tmp_0 = (int16_T)((int16_T)tmp > 0);
  }

  /* End of Signum: '<S76>/SignDeltaU3' */

  /* Update for UnitDelay: '<S74>/UnitDelay1' incorporates:
   *  DataTypeConversion: '<S76>/DataTypeConv4'
   *  Logic: '<S74>/AND1'
   *  Logic: '<S76>/AND1'
   *  RelationalOperator: '<S76>/Equal1'
   */
  localDW->UnitDelay1_DSTATE = ((q0 == tmp_0) && (rtb_LowerRelop1_c0 ||
    rtb_UpperRelop_f));

  /* Update for Delay: '<S77>/Resettable Delay' */
  localDW->icLoad = 0U;
  localDW->ResettableDelay_DSTATE = rtb_Sum1_p0;
}

/* System initialize for atomic system: '<S61>/PI_clamp_fixdt' */
void PI_clamp_fixdt_d_Init(DW_PI_clamp_fixdt_m *localDW)
{
  /* InitializeConditions for Delay: '<S67>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* System reset for atomic system: '<S61>/PI_clamp_fixdt' */
void PI_clamp_fixdt_b_Reset(DW_PI_clamp_fixdt_m *localDW)
{
  /* InitializeConditions for UnitDelay: '<S65>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE = false;

  /* InitializeConditions for Delay: '<S67>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* Output and update for atomic system: '<S61>/PI_clamp_fixdt' */
void PI_clamp_fixdt_l(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I, int16_T
                      rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
                      rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt_m
                      *localDW)
{
  boolean_T rtb_LowerRelop1_l;
  boolean_T rtb_UpperRelop_l;
  int32_T rtb_Sum1_ni;
  int32_T q0;
  int32_T tmp;
  int16_T tmp_0;

  /* Sum: '<S65>/Sum2' incorporates:
   *  Product: '<S65>/Divide2'
   */
  q0 = rtu_err * rtu_I;
  if ((q0 < 0) && (rtu_ext_limProt < MIN_int32_T - q0)) {
    q0 = MIN_int32_T;
  } else if ((q0 > 0) && (rtu_ext_limProt > MAX_int32_T - q0)) {
    q0 = MAX_int32_T;
  } else {
    q0 += rtu_ext_limProt;
  }

  /* Delay: '<S67>/Resettable Delay' */
  if (localDW->icLoad != 0) {
    localDW->ResettableDelay_DSTATE = rtu_init << 16;
  }

  /* Switch: '<S65>/Switch1' incorporates:
   *  Constant: '<S65>/Constant'
   *  Sum: '<S65>/Sum2'
   *  UnitDelay: '<S65>/UnitDelay1'
   */
  if (localDW->UnitDelay1_DSTATE) {
    tmp = 0;
  } else {
    tmp = q0;
  }

  /* End of Switch: '<S65>/Switch1' */

  /* Sum: '<S67>/Sum1' incorporates:
   *  Delay: '<S67>/Resettable Delay'
   */
  rtb_Sum1_ni = tmp + localDW->ResettableDelay_DSTATE;

  /* Product: '<S65>/Divide5' */
  tmp = (rtu_err * rtu_P) >> 11;
  if (tmp > 32767) {
    tmp = 32767;
  } else {
    if (tmp < -32768) {
      tmp = -32768;
    }
  }

  /* Sum: '<S65>/Sum1' incorporates:
   *  DataTypeConversion: '<S67>/Data Type Conversion1'
   *  Product: '<S65>/Divide5'
   */
  tmp = (((rtb_Sum1_ni >> 16) << 1) + tmp) >> 1;
  if (tmp > 32767) {
    tmp = 32767;
  } else {
    if (tmp < -32768) {
      tmp = -32768;
    }
  }

  /* RelationalOperator: '<S68>/LowerRelop1' incorporates:
   *  Sum: '<S65>/Sum1'
   */
  rtb_LowerRelop1_l = ((int16_T)tmp > rtu_satMax);

  /* RelationalOperator: '<S68>/UpperRelop' incorporates:
   *  Sum: '<S65>/Sum1'
   */
  rtb_UpperRelop_l = ((int16_T)tmp < rtu_satMin);

  /* Switch: '<S68>/Switch1' incorporates:
   *  Sum: '<S65>/Sum1'
   *  Switch: '<S68>/Switch3'
   */
  if (rtb_LowerRelop1_l) {
    *rty_out = rtu_satMax;
  } else if (rtb_UpperRelop_l) {
    /* Switch: '<S68>/Switch3' */
    *rty_out = rtu_satMin;
  } else {
    *rty_out = (int16_T)tmp;
  }

  /* End of Switch: '<S68>/Switch1' */

  /* Signum: '<S66>/SignDeltaU2' incorporates:
   *  Sum: '<S65>/Sum2'
   */
  if (q0 < 0) {
    q0 = -1;
  } else {
    q0 = (q0 > 0);
  }

  /* End of Signum: '<S66>/SignDeltaU2' */

  /* Signum: '<S66>/SignDeltaU3' incorporates:
   *  Sum: '<S65>/Sum1'
   */
  if ((int16_T)tmp < 0) {
    tmp_0 = -1;
  } else {
    tmp_0 = (int16_T)((int16_T)tmp > 0);
  }

  /* End of Signum: '<S66>/SignDeltaU3' */

  /* Update for UnitDelay: '<S65>/UnitDelay1' incorporates:
   *  DataTypeConversion: '<S66>/DataTypeConv4'
   *  Logic: '<S65>/AND1'
   *  Logic: '<S66>/AND1'
   *  RelationalOperator: '<S66>/Equal1'
   */
  localDW->UnitDelay1_DSTATE = ((q0 == tmp_0) && (rtb_LowerRelop1_l ||
    rtb_UpperRelop_l));

  /* Update for Delay: '<S67>/Resettable Delay' */
  localDW->icLoad = 0U;
  localDW->ResettableDelay_DSTATE = rtb_Sum1_ni;
}

/* System initialize for atomic system: '<S62>/PI_clamp_fixdt' */
void PI_clamp_fixdt_f_Init(DW_PI_clamp_fixdt_g *localDW)
{
  /* InitializeConditions for Delay: '<S72>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* System reset for atomic system: '<S62>/PI_clamp_fixdt' */
void PI_clamp_fixdt_g_Reset(DW_PI_clamp_fixdt_g *localDW)
{
  /* InitializeConditions for UnitDelay: '<S69>/UnitDelay1' */
  localDW->UnitDelay1_DSTATE = false;

  /* InitializeConditions for Delay: '<S72>/Resettable Delay' */
  localDW->icLoad = 1U;
}

/* Output and update for atomic system: '<S62>/PI_clamp_fixdt' */
void PI_clamp_fixdt_k(int16_T rtu_err, uint16_T rtu_P, uint16_T rtu_I, int16_T
                      rtu_init, int16_T rtu_satMax, int16_T rtu_satMin, int32_T
                      rtu_ext_limProt, int16_T *rty_out, DW_PI_clamp_fixdt_g
                      *localDW)
{
  boolean_T rtb_LowerRelop1_i3;
  boolean_T rtb_UpperRelop_i;
  int16_T rtb_Sum1_bm;
  int16_T tmp;
  int32_T tmp_0;
  int32_T q0;

  /* Sum: '<S69>/Sum2' incorporates:
   *  Product: '<S69>/Divide2'
   */
  q0 = rtu_err * rtu_I;
  if ((q0 < 0) && (rtu_ext_limProt < MIN_int32_T - q0)) {
    q0 = MIN_int32_T;
  } else if ((q0 > 0) && (rtu_ext_limProt > MAX_int32_T - q0)) {
    q0 = MAX_int32_T;
  } else {
    q0 += rtu_ext_limProt;
  }

  /* Delay: '<S72>/Resettable Delay' */
  if (localDW->icLoad != 0) {
    localDW->ResettableDelay_DSTATE = rtu_init;
  }

  /* Switch: '<S69>/Switch1' incorporates:
   *  Constant: '<S69>/Constant'
   *  Sum: '<S69>/Sum2'
   *  UnitDelay: '<S69>/UnitDelay1'
   */
  if (localDW->UnitDelay1_DSTATE) {
    tmp = 0;
  } else {
    tmp = (int16_T)(((q0 < 0 ? 65535 : 0) + q0) >> 16);
  }

  /* End of Switch: '<S69>/Switch1' */

  /* Sum: '<S72>/Sum1' incorporates:
   *  Delay: '<S72>/Resettable Delay'
   */
  rtb_Sum1_bm = (int16_T)(tmp + localDW->ResettableDelay_DSTATE);

  /* Product: '<S69>/Divide5' */
  tmp_0 = (rtu_err * rtu_P) >> 11;
  if (tmp_0 > 32767) {
    tmp_0 = 32767;
  } else {
    if (tmp_0 < -32768) {
      tmp_0 = -32768;
    }
  }

  /* Sum: '<S69>/Sum1' incorporates:
   *  Product: '<S69>/Divide5'
   */
  tmp_0 = ((rtb_Sum1_bm << 1) + tmp_0) >> 1;
  if (tmp_0 > 32767) {
    tmp_0 = 32767;
  } else {
    if (tmp_0 < -32768) {
      tmp_0 = -32768;
    }
  }

  /* RelationalOperator: '<S73>/LowerRelop1' incorporates:
   *  Sum: '<S69>/Sum1'
   */
  rtb_LowerRelop1_i3 = ((int16_T)tmp_0 > rtu_satMax);

  /* RelationalOperator: '<S73>/UpperRelop' incorporates:
   *  Sum: '<S69>/Sum1'
   */
  rtb_UpperRelop_i = ((int16_T)tmp_0 < rtu_satMin);

  /* Switch: '<S73>/Switch1' incorporates:
   *  Sum: '<S69>/Sum1'
   *  Switch: '<S73>/Switch3'
   */
  if (rtb_LowerRelop1_i3) {
    *rty_out = rtu_satMax;
  } else if (rtb_UpperRelop_i) {
    /* Switch: '<S73>/Switch3' */
    *rty_out = rtu_satMin;
  } else {
    *rty_out = (int16_T)tmp_0;
  }

  /* End of Switch: '<S73>/Switch1' */

  /* Signum: '<S71>/SignDeltaU2' incorporates:
   *  Sum: '<S69>/Sum2'
   */
  if (q0 < 0) {
    q0 = -1;
  } else {
    q0 = (q0 > 0);
  }

  /* End of Signum: '<S71>/SignDeltaU2' */

  /* Signum: '<S71>/SignDeltaU3' incorporates:
   *  Sum: '<S69>/Sum1'
   */
  if ((int16_T)tmp_0 < 0) {
    tmp = -1;
  } else {
    tmp = (int16_T)((int16_T)tmp_0 > 0);
  }

  /* End of Signum: '<S71>/SignDeltaU3' */

  /* Update for UnitDelay: '<S69>/UnitDelay1' incorporates:
   *  DataTypeConversion: '<S71>/DataTypeConv4'
   *  Logic: '<S69>/AND1'
   *  Logic: '<S71>/AND1'
   *  RelationalOperator: '<S71>/Equal1'
   */
  localDW->UnitDelay1_DSTATE = ((q0 == tmp) && (rtb_LowerRelop1_i3 ||
    rtb_UpperRelop_i));

  /* Update for Delay: '<S72>/Resettable Delay' */
  localDW->icLoad = 0U;
  localDW->ResettableDelay_DSTATE = rtb_Sum1_bm;
}

/* Model step function */
void BLDC_controller_step(RT_MODEL *const rtM)
{
  P *rtP = ((P *) rtM->defaultParam);
  DW *rtDW = ((DW *) rtM->dwork);
  ExtU *rtU = (ExtU *) rtM->inputs;
  ExtY *rtY = (ExtY *) rtM->outputs;
  boolean_T rtb_LogicalOperator;
  int8_T rtb_Sum2_h;
  boolean_T rtb_RelationalOperator4_d;
  boolean_T rtb_UnitDelay5_e;
  uint8_T rtb_a_elecAngle_XA_g;
  boolean_T rtb_LogicalOperator1_j;
  boolean_T rtb_LogicalOperator2_p;
  boolean_T rtb_RelationalOperator1_mv;
  int16_T rtb_Switch1_l;
  int16_T rtb_Saturation;
  int16_T rtb_Saturation1;
  int32_T rtb_Sum1_jt;
  int16_T rtb_Merge_m;
  int16_T rtb_Merge1;
  uint16_T rtb_Divide14_e;
  uint16_T rtb_Divide1_f;
  int16_T rtb_TmpSignalConversionAtLow_Pa[2];
  int32_T rtb_Switch1;
  int32_T rtb_Sum1;
  int32_T rtb_Gain3;
  uint8_T Sum;
  int16_T Switch2;
  int16_T Abs5;
  int16_T DataTypeConversion2;
  int16_T tmp[4];
  int8_T UnitDelay3;

  /* Outputs for Atomic SubSystem: '<Root>/BLDC_controller' */
  /* Sum: '<S11>/Sum' incorporates:
   *  Gain: '<S11>/g_Ha'
   *  Gain: '<S11>/g_Hb'
   *  Inport: '<Root>/b_hallA '
   *  Inport: '<Root>/b_hallB'
   *  Inport: '<Root>/b_hallC'
   */
  Sum = (uint8_T)((uint32_T)(uint8_T)((uint32_T)(uint8_T)(rtU->b_hallA << 2) +
    (uint8_T)(rtU->b_hallB << 1)) + rtU->b_hallC);

  /* Logic: '<S10>/Logical Operator' incorporates:
   *  Inport: '<Root>/b_hallA '
   *  Inport: '<Root>/b_hallB'
   *  Inport: '<Root>/b_hallC'
   *  UnitDelay: '<S10>/UnitDelay1'
   *  UnitDelay: '<S10>/UnitDelay2'
   *  UnitDelay: '<S10>/UnitDelay3'
   */
  rtb_LogicalOperator = (boolean_T)((rtU->b_hallA != 0) ^ (rtU->b_hallB != 0) ^
    (rtU->b_hallC != 0) ^ (rtDW->UnitDelay3_DSTATE_fy != 0) ^
    (rtDW->UnitDelay1_DSTATE != 0)) ^ (rtDW->UnitDelay2_DSTATE_f != 0);

  /* If: '<S13>/If2' incorporates:
   *  If: '<S3>/If2'
   *  Inport: '<S17>/z_counterRawPrev'
   *  UnitDelay: '<S13>/UnitDelay3'
   */
  if (rtb_LogicalOperator) {
    /* Outputs for IfAction SubSystem: '<S3>/F01_03_Direction_Detection' incorporates:
     *  ActionPort: '<S12>/Action Port'
     */
    /* UnitDelay: '<S12>/UnitDelay3' */
    UnitDelay3 = rtDW->Switch2_e;

    /* Sum: '<S12>/Sum2' incorporates:
     *  Constant: '<S11>/vec_hallToPos'
     *  Selector: '<S11>/Selector'
     *  UnitDelay: '<S12>/UnitDelay2'
     */
    rtb_Sum2_h = (int8_T)(rtConstP.vec_hallToPos_Value[Sum] -
                          rtDW->UnitDelay2_DSTATE_b);

    /* Switch: '<S12>/Switch2' incorporates:
     *  Constant: '<S12>/Constant20'
     *  Constant: '<S12>/Constant23'
     *  Constant: '<S12>/Constant24'
     *  Constant: '<S12>/Constant8'
     *  Logic: '<S12>/Logical Operator3'
     *  RelationalOperator: '<S12>/Relational Operator1'
     *  RelationalOperator: '<S12>/Relational Operator6'
     */
    if ((rtb_Sum2_h == 1) || (rtb_Sum2_h == -5)) {
      rtDW->Switch2_e = 1;
    } else {
      rtDW->Switch2_e = -1;
    }

    /* End of Switch: '<S12>/Switch2' */

    /* Update for UnitDelay: '<S12>/UnitDelay2' incorporates:
     *  Constant: '<S11>/vec_hallToPos'
     *  Selector: '<S11>/Selector'
     */
    rtDW->UnitDelay2_DSTATE_b = rtConstP.vec_hallToPos_Value[Sum];

    /* End of Outputs for SubSystem: '<S3>/F01_03_Direction_Detection' */

    /* Outputs for IfAction SubSystem: '<S13>/Raw_Motor_Speed_Estimation' incorporates:
     *  ActionPort: '<S17>/Action Port'
     */
    rtDW->z_counterRawPrev = rtDW->UnitDelay3_DSTATE;

    /* Sum: '<S17>/Sum7' incorporates:
     *  Inport: '<S17>/z_counterRawPrev'
     *  UnitDelay: '<S13>/UnitDelay3'
     *  UnitDelay: '<S17>/UnitDelay4'
     */
    Switch2 = (int16_T)(rtDW->z_counterRawPrev - rtDW->UnitDelay4_DSTATE);

    /* Abs: '<S17>/Abs2' */
    if (Switch2 < 0) {
      rtb_Switch1_l = (int16_T)-Switch2;
    } else {
      rtb_Switch1_l = Switch2;
    }

    /* End of Abs: '<S17>/Abs2' */

    /* Relay: '<S17>/dz_cntTrnsDet' */
    if (rtb_Switch1_l >= rtP->dz_cntTrnsDetHi) {
      rtDW->dz_cntTrnsDet_Mode = true;
    } else {
      if (rtb_Switch1_l <= rtP->dz_cntTrnsDetLo) {
        rtDW->dz_cntTrnsDet_Mode = false;
      }
    }

    rtDW->dz_cntTrnsDet = rtDW->dz_cntTrnsDet_Mode;

    /* End of Relay: '<S17>/dz_cntTrnsDet' */

    /* RelationalOperator: '<S17>/Relational Operator4' */
    rtb_RelationalOperator4_d = (rtDW->Switch2_e != UnitDelay3);

    /* Switch: '<S17>/Switch3' incorporates:
     *  Constant: '<S17>/Constant4'
     *  Logic: '<S17>/Logical Operator1'
     *  Switch: '<S17>/Switch1'
     *  Switch: '<S17>/Switch2'
     *  UnitDelay: '<S17>/UnitDelay1'
     */
    if (rtb_RelationalOperator4_d && rtDW->UnitDelay1_DSTATE_n) {
      rtb_Switch1_l = 0;
    } else if (rtb_RelationalOperator4_d) {
      /* Switch: '<S17>/Switch2' incorporates:
       *  UnitDelay: '<S13>/UnitDelay4'
       */
      rtb_Switch1_l = rtDW->UnitDelay4_DSTATE_e;
    } else if (rtDW->dz_cntTrnsDet) {
      /* Switch: '<S17>/Switch1' incorporates:
       *  Constant: '<S17>/cf_speedCoef'
       *  Product: '<S17>/Divide14'
       *  Switch: '<S17>/Switch2'
       */
      rtb_Switch1_l = (int16_T)((rtP->cf_speedCoef << 4) /
        rtDW->z_counterRawPrev);
    } else {
      /* Switch: '<S17>/Switch1' incorporates:
       *  Constant: '<S17>/cf_speedCoef'
       *  Gain: '<S17>/g_Ha'
       *  Product: '<S17>/Divide13'
       *  Sum: '<S17>/Sum13'
       *  Switch: '<S17>/Switch2'
       *  UnitDelay: '<S17>/UnitDelay2'
       *  UnitDelay: '<S17>/UnitDelay3'
       *  UnitDelay: '<S17>/UnitDelay5'
       */
      rtb_Switch1_l = (int16_T)(((uint16_T)(rtP->cf_speedCoef << 2) << 4) /
        (int16_T)(((rtDW->UnitDelay2_DSTATE + rtDW->UnitDelay3_DSTATE_o) +
                   rtDW->UnitDelay5_DSTATE) + rtDW->z_counterRawPrev));
    }

    /* End of Switch: '<S17>/Switch3' */

    /* Product: '<S17>/Divide11' */
    rtDW->Divide11 = (int16_T)(rtb_Switch1_l * rtDW->Switch2_e);

    /* Update for UnitDelay: '<S17>/UnitDelay4' */
    rtDW->UnitDelay4_DSTATE = rtDW->z_counterRawPrev;

    /* Update for UnitDelay: '<S17>/UnitDelay2' incorporates:
     *  UnitDelay: '<S17>/UnitDelay3'
     */
    rtDW->UnitDelay2_DSTATE = rtDW->UnitDelay3_DSTATE_o;

    /* Update for UnitDelay: '<S17>/UnitDelay3' incorporates:
     *  UnitDelay: '<S17>/UnitDelay5'
     */
    rtDW->UnitDelay3_DSTATE_o = rtDW->UnitDelay5_DSTATE;

    /* Update for UnitDelay: '<S17>/UnitDelay5' */
    rtDW->UnitDelay5_DSTATE = rtDW->z_counterRawPrev;

    /* Update for UnitDelay: '<S17>/UnitDelay1' */
    rtDW->UnitDelay1_DSTATE_n = rtb_RelationalOperator4_d;

    /* End of Outputs for SubSystem: '<S13>/Raw_Motor_Speed_Estimation' */
  }

  /* End of If: '<S13>/If2' */

  /* Outputs for Atomic SubSystem: '<S13>/Counter' */

  /* Constant: '<S13>/Constant6' incorporates:
   *  Constant: '<S13>/z_maxCntRst2'
   */
  rtb_Switch1_l = (int16_T) Counter(1, rtP->z_maxCntRst, rtb_LogicalOperator,
    &rtDW->Counter_e);

  /* End of Outputs for SubSystem: '<S13>/Counter' */

  /* Switch: '<S13>/Switch2' incorporates:
   *  Constant: '<S13>/Constant4'
   *  Constant: '<S13>/z_maxCntRst'
   *  RelationalOperator: '<S13>/Relational Operator2'
   */
  if (rtb_Switch1_l > rtP->z_maxCntRst) {
    Switch2 = 0;
  } else {
    Switch2 = rtDW->Divide11;
  }

  /* End of Switch: '<S13>/Switch2' */

  /* Abs: '<S13>/Abs5' */
  if (Switch2 < 0) {
    Abs5 = (int16_T)-Switch2;
  } else {
    Abs5 = Switch2;
  }

  /* End of Abs: '<S13>/Abs5' */

  /* Relay: '<S13>/n_commDeacv' */
  if (Abs5 >= rtP->n_commDeacvHi) {
    rtDW->n_commDeacv_Mode = true;
  } else {
    if (Abs5 <= rtP->n_commAcvLo) {
      rtDW->n_commDeacv_Mode = false;
    }
  }

  /* Logic: '<S13>/Logical Operator3' incorporates:
   *  Constant: '<S13>/b_angleMeasEna'
   *  Logic: '<S13>/Logical Operator1'
   *  Logic: '<S13>/Logical Operator2'
   *  Relay: '<S13>/n_commDeacv'
   */
  rtb_LogicalOperator = (rtP->b_angleMeasEna || (rtDW->n_commDeacv_Mode &&
    (!rtDW->dz_cntTrnsDet)));

  /* UnitDelay: '<S2>/UnitDelay2' */
  rtb_RelationalOperator4_d = rtDW->UnitDelay2_DSTATE_c;

  /* UnitDelay: '<S2>/UnitDelay5' */
  rtb_UnitDelay5_e = rtDW->UnitDelay5_DSTATE_m;

  /* DataTypeConversion: '<S1>/Data Type Conversion2' incorporates:
   *  Inport: '<Root>/r_inpTgt'
   */
  DataTypeConversion2 = (int16_T)(rtU->r_inpTgt << 4);

  /* Saturate: '<S1>/Saturation' incorporates:
   *  Inport: '<Root>/i_phaAB'
   */
  rtb_Gain3 = rtU->i_phaAB << 4;
  if (rtb_Gain3 >= 27200) {
    rtb_Saturation = 27200;
  } else if (rtb_Gain3 <= -27200) {
    rtb_Saturation = -27200;
  } else {
    rtb_Saturation = (int16_T)(rtU->i_phaAB << 4);
  }

  /* End of Saturate: '<S1>/Saturation' */

  /* Saturate: '<S1>/Saturation1' incorporates:
   *  Inport: '<Root>/i_phaBC'
   */
  rtb_Gain3 = rtU->i_phaBC << 4;
  if (rtb_Gain3 >= 27200) {
    rtb_Saturation1 = 27200;
  } else if (rtb_Gain3 <= -27200) {
    rtb_Saturation1 = -27200;
  } else {
    rtb_Saturation1 = (int16_T)(rtU->i_phaBC << 4);
  }

  /* End of Saturate: '<S1>/Saturation1' */

  /* If: '<S3>/If1' incorporates:
   *  Constant: '<S3>/b_angleMeasEna'
   */
  if (!rtP->b_angleMeasEna) {
    /* Outputs for IfAction SubSystem: '<S3>/F01_05_Electrical_Angle_Estimation' incorporates:
     *  ActionPort: '<S14>/Action Port'
     */
    /* Switch: '<S14>/Switch2' incorporates:
     *  Constant: '<S14>/Constant16'
     *  Product: '<S14>/Divide1'
     *  Product: '<S14>/Divide3'
     *  RelationalOperator: '<S14>/Relational Operator7'
     *  Sum: '<S14>/Sum3'
     *  Switch: '<S14>/Switch3'
     */
    if (rtb_LogicalOperator) {
      /* MinMax: '<S14>/MinMax' */
      rtb_Merge_m = rtb_Switch1_l;
      if (!(rtb_Merge_m < rtDW->z_counterRawPrev)) {
        rtb_Merge_m = rtDW->z_counterRawPrev;
      }

      /* End of MinMax: '<S14>/MinMax' */

      /* Switch: '<S14>/Switch3' incorporates:
       *  Constant: '<S11>/vec_hallToPos'
       *  Constant: '<S14>/Constant16'
       *  RelationalOperator: '<S14>/Relational Operator7'
       *  Selector: '<S11>/Selector'
       *  Sum: '<S14>/Sum1'
       */
      if (rtDW->Switch2_e == 1) {
        rtb_Sum2_h = rtConstP.vec_hallToPos_Value[Sum];
      } else {
        rtb_Sum2_h = (int8_T)(rtConstP.vec_hallToPos_Value[Sum] + 1);
      }

      rtb_Merge_m = (int16_T)(((int16_T)((int16_T)((rtb_Merge_m << 14) /
        rtDW->z_counterRawPrev) * rtDW->Switch2_e) + (rtb_Sum2_h << 14)) >> 2);
    } else {
      if (rtDW->Switch2_e == 1) {
        /* Switch: '<S14>/Switch3' incorporates:
         *  Constant: '<S11>/vec_hallToPos'
         *  Selector: '<S11>/Selector'
         */
        rtb_Sum2_h = rtConstP.vec_hallToPos_Value[Sum];
      } else {
        /* Switch: '<S14>/Switch3' incorporates:
         *  Constant: '<S11>/vec_hallToPos'
         *  Selector: '<S11>/Selector'
         *  Sum: '<S14>/Sum1'
         */
        rtb_Sum2_h = (int8_T)(rtConstP.vec_hallToPos_Value[Sum] + 1);
      }

      rtb_Merge_m = (int16_T)(rtb_Sum2_h << 12);
    }

    /* End of Switch: '<S14>/Switch2' */

    /* MinMax: '<S14>/MinMax1' incorporates:
     *  Constant: '<S14>/Constant1'
     */
    if (!(rtb_Merge_m > 0)) {
      rtb_Merge_m = 0;
    }

    /* End of MinMax: '<S14>/MinMax1' */

    /* SignalConversion: '<S14>/Signal Conversion2' incorporates:
     *  Product: '<S14>/Divide2'
     */
    rtb_Merge_m = (int16_T)((15 * rtb_Merge_m) >> 4);

    /* End of Outputs for SubSystem: '<S3>/F01_05_Electrical_Angle_Estimation' */
  } else {
    /* Outputs for IfAction SubSystem: '<S3>/F01_06_Electrical_Angle_Measurement' incorporates:
     *  ActionPort: '<S15>/Action Port'
     */
    /* Sum: '<S15>/Sum1' incorporates:
     *  Constant: '<S15>/Constant2'
     *  Constant: '<S15>/n_polePairs'
     *  Inport: '<Root>/a_mechAngle'
     *  Product: '<S15>/Divide'
     */
    rtb_Sum1_jt = rtU->a_mechAngle * rtP->n_polePairs - 480;

    /* DataTypeConversion: '<S15>/Data Type Conversion20' incorporates:
     *  Constant: '<S15>/a_elecPeriod'
     *  Product: '<S19>/Divide2'
     *  Product: '<S19>/Divide3'
     *  Sum: '<S19>/Sum3'
     */
    rtb_Merge_m = (int16_T)((int16_T)(rtb_Sum1_jt - ((int16_T)((int16_T)
      div_nde_s32_floor(rtb_Sum1_jt, 5760) * 360) << 4)) << 2);

    /* End of Outputs for SubSystem: '<S3>/F01_06_Electrical_Angle_Measurement' */
  }

  /* End of If: '<S3>/If1' */

  /* If: '<S7>/If1' incorporates:
   *  Constant: '<S1>/z_ctrlTypSel'
   */
  rtb_Sum2_h = rtDW->If1_ActiveSubsystem;
  UnitDelay3 = -1;
  if (rtP->z_ctrlTypSel == 2) {
    UnitDelay3 = 0;
  }

  rtDW->If1_ActiveSubsystem = UnitDelay3;
  if ((rtb_Sum2_h != UnitDelay3) && (rtb_Sum2_h == 0)) {
    /* Disable for If: '<S45>/If2' */
    if (rtDW->If2_ActiveSubsystem_a == 0) {
      /* Disable for Outport: '<S50>/iq' */
      rtDW->DataTypeConversion[0] = 0;

      /* Disable for Outport: '<S50>/iqAbs' */
      rtDW->Abs5_h = 0;

      /* Disable for Outport: '<S50>/id' */
      rtDW->DataTypeConversion[1] = 0;
    }

    rtDW->If2_ActiveSubsystem_a = -1;

    /* End of Disable for If: '<S45>/If2' */

    /* Disable for Outport: '<S45>/r_sin' */
    rtDW->r_sin_M1 = 0;

    /* Disable for Outport: '<S45>/r_cos' */
    rtDW->r_cos_M1 = 0;

    /* Disable for Outport: '<S45>/iq' */
    rtDW->DataTypeConversion[0] = 0;

    /* Disable for Outport: '<S45>/id' */
    rtDW->DataTypeConversion[1] = 0;

    /* Disable for Outport: '<S45>/iqAbs' */
    rtDW->Abs5_h = 0;
  }

  if (UnitDelay3 == 0) {
    /* Outputs for IfAction SubSystem: '<S7>/Clarke_Park_Transform_Forward' incorporates:
     *  ActionPort: '<S45>/Action Port'
     */
    /* If: '<S49>/If1' incorporates:
     *  Constant: '<S49>/z_selPhaCurMeasABC'
     */
    if (rtP->z_selPhaCurMeasABC == 0) {
      /* Outputs for IfAction SubSystem: '<S49>/Clarke_PhasesAB' incorporates:
       *  ActionPort: '<S53>/Action Port'
       */
      /* Gain: '<S53>/Gain4' */
      rtb_Gain3 = 18919 * rtb_Saturation;

      /* Gain: '<S53>/Gain2' */
      rtb_Sum1_jt = 18919 * rtb_Saturation1;

      /* Sum: '<S53>/Sum1' incorporates:
       *  Gain: '<S53>/Gain2'
       *  Gain: '<S53>/Gain4'
       */
      rtb_Gain3 = (((rtb_Gain3 < 0 ? 32767 : 0) + rtb_Gain3) >> 15) + (int16_T)
        (((rtb_Sum1_jt < 0 ? 16383 : 0) + rtb_Sum1_jt) >> 14);
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      rtb_Merge1 = (int16_T)rtb_Gain3;

      /* End of Sum: '<S53>/Sum1' */
      /* End of Outputs for SubSystem: '<S49>/Clarke_PhasesAB' */
    } else if (rtP->z_selPhaCurMeasABC == 1) {
      /* Outputs for IfAction SubSystem: '<S49>/Clarke_PhasesBC' incorporates:
       *  ActionPort: '<S55>/Action Port'
       */
      /* Sum: '<S55>/Sum3' */
      rtb_Gain3 = rtb_Saturation - rtb_Saturation1;
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      /* Gain: '<S55>/Gain2' incorporates:
       *  Sum: '<S55>/Sum3'
       */
      rtb_Gain3 *= 18919;
      rtb_Merge1 = (int16_T)(((rtb_Gain3 < 0 ? 32767 : 0) + rtb_Gain3) >> 15);

      /* Sum: '<S55>/Sum1' */
      rtb_Gain3 = -rtb_Saturation - rtb_Saturation1;
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      rtb_Saturation = (int16_T)rtb_Gain3;

      /* End of Sum: '<S55>/Sum1' */
      /* End of Outputs for SubSystem: '<S49>/Clarke_PhasesBC' */
    } else {
      /* Outputs for IfAction SubSystem: '<S49>/Clarke_PhasesAC' incorporates:
       *  ActionPort: '<S54>/Action Port'
       */
      /* Gain: '<S54>/Gain4' */
      rtb_Gain3 = 18919 * rtb_Saturation;

      /* Gain: '<S54>/Gain2' */
      rtb_Sum1_jt = 18919 * rtb_Saturation1;

      /* Sum: '<S54>/Sum1' incorporates:
       *  Gain: '<S54>/Gain2'
       *  Gain: '<S54>/Gain4'
       */
      rtb_Gain3 = -(((rtb_Gain3 < 0 ? 32767 : 0) + rtb_Gain3) >> 15) - (int16_T)
        (((rtb_Sum1_jt < 0 ? 16383 : 0) + rtb_Sum1_jt) >> 14);
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      rtb_Merge1 = (int16_T)rtb_Gain3;

      /* End of Sum: '<S54>/Sum1' */
      /* End of Outputs for SubSystem: '<S49>/Clarke_PhasesAC' */
    }

    /* End of If: '<S49>/If1' */

    /* PreLookup: '<S52>/a_elecAngle_XA' */
    rtb_a_elecAngle_XA_g = plook_u8s16_evencka(rtb_Merge_m, 0, 128U, 180U);

    /* Interpolation_n-D: '<S52>/r_sin_M1' */
    rtDW->r_sin_M1 = rtConstP.r_sin_M1_Table[rtb_a_elecAngle_XA_g];

    /* Interpolation_n-D: '<S52>/r_cos_M1' */
    rtDW->r_cos_M1 = rtConstP.r_cos_M1_Table[rtb_a_elecAngle_XA_g];

    /* If: '<S45>/If2' incorporates:
     *  Constant: '<S50>/cf_currFilt'
     *  Inport: '<Root>/b_motEna'
     */
    rtb_Sum2_h = rtDW->If2_ActiveSubsystem_a;
    UnitDelay3 = -1;
    if (rtU->b_motEna) {
      UnitDelay3 = 0;
    }

    rtDW->If2_ActiveSubsystem_a = UnitDelay3;
    if ((rtb_Sum2_h != UnitDelay3) && (rtb_Sum2_h == 0)) {
      /* Disable for Outport: '<S50>/iq' */
      rtDW->DataTypeConversion[0] = 0;

      /* Disable for Outport: '<S50>/iqAbs' */
      rtDW->Abs5_h = 0;

      /* Disable for Outport: '<S50>/id' */
      rtDW->DataTypeConversion[1] = 0;
    }

    if (UnitDelay3 == 0) {
      if (0 != rtb_Sum2_h) {
        /* SystemReset for IfAction SubSystem: '<S45>/Current_Filtering' incorporates:
         *  ActionPort: '<S50>/Action Port'
         */

        /* SystemReset for Atomic SubSystem: '<S50>/Low_Pass_Filter' */

        /* SystemReset for If: '<S45>/If2' */
        Low_Pass_Filter_Reset(&rtDW->Low_Pass_Filter_m);

        /* End of SystemReset for SubSystem: '<S50>/Low_Pass_Filter' */

        /* End of SystemReset for SubSystem: '<S45>/Current_Filtering' */
      }

      /* Sum: '<S51>/Sum6' incorporates:
       *  Product: '<S51>/Divide1'
       *  Product: '<S51>/Divide4'
       */
      rtb_Gain3 = (int16_T)((rtb_Merge1 * rtDW->r_cos_M1) >> 14) - (int16_T)
        ((rtb_Saturation * rtDW->r_sin_M1) >> 14);
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      /* Outputs for IfAction SubSystem: '<S45>/Current_Filtering' incorporates:
       *  ActionPort: '<S50>/Action Port'
       */
      /* SignalConversion: '<S50>/TmpSignal ConversionAtLow_Pass_FilterInport1' incorporates:
       *  Sum: '<S51>/Sum6'
       */
      rtb_TmpSignalConversionAtLow_Pa[0] = (int16_T)rtb_Gain3;

      /* End of Outputs for SubSystem: '<S45>/Current_Filtering' */

      /* Sum: '<S51>/Sum1' incorporates:
       *  Product: '<S51>/Divide2'
       *  Product: '<S51>/Divide3'
       */
      rtb_Gain3 = (int16_T)((rtb_Saturation * rtDW->r_cos_M1) >> 14) + (int16_T)
        ((rtb_Merge1 * rtDW->r_sin_M1) >> 14);
      if (rtb_Gain3 > 32767) {
        rtb_Gain3 = 32767;
      } else {
        if (rtb_Gain3 < -32768) {
          rtb_Gain3 = -32768;
        }
      }

      /* Outputs for IfAction SubSystem: '<S45>/Current_Filtering' incorporates:
       *  ActionPort: '<S50>/Action Port'
       */
      /* SignalConversion: '<S50>/TmpSignal ConversionAtLow_Pass_FilterInport1' incorporates:
       *  Sum: '<S51>/Sum1'
       */
      rtb_TmpSignalConversionAtLow_Pa[1] = (int16_T)rtb_Gain3;

      /* Outputs for Atomic SubSystem: '<S50>/Low_Pass_Filter' */
      Low_Pass_Filter(rtb_TmpSignalConversionAtLow_Pa, rtP->cf_currFilt,
                      rtDW->DataTypeConversion, &rtDW->Low_Pass_Filter_m);

      /* End of Outputs for SubSystem: '<S50>/Low_Pass_Filter' */

      /* Abs: '<S50>/Abs5' incorporates:
       *  Constant: '<S50>/cf_currFilt'
       */
      if (rtDW->DataTypeConversion[0] < 0) {
        rtDW->Abs5_h = (int16_T)-rtDW->DataTypeConversion[0];
      } else {
        rtDW->Abs5_h = rtDW->DataTypeConversion[0];
      }

      /* End of Abs: '<S50>/Abs5' */
      /* End of Outputs for SubSystem: '<S45>/Current_Filtering' */
    }

    /* End of If: '<S45>/If2' */
    /* End of Outputs for SubSystem: '<S7>/Clarke_Park_Transform_Forward' */
  }

  /* End of If: '<S7>/If1' */

  /* Chart: '<S1>/Task_Scheduler' incorporates:
   *  UnitDelay: '<S2>/UnitDelay2'
   *  UnitDelay: '<S2>/UnitDelay5'
   *  UnitDelay: '<S2>/UnitDelay6'
   */
  if (rtDW->UnitDelay2_DSTATE_c) {
    /* Outputs for Function Call SubSystem: '<S1>/F02_Diagnostics' */
    /* If: '<S4>/If2' incorporates:
     *  Constant: '<S20>/CTRL_COMM2'
     *  Constant: '<S20>/t_errDequal'
     *  Constant: '<S20>/t_errQual'
     *  Constant: '<S4>/b_diagEna'
     *  RelationalOperator: '<S20>/Relational Operator2'
     */
    if (rtP->b_diagEna) {
      /* Outputs for IfAction SubSystem: '<S4>/Diagnostics_Enabled' incorporates:
       *  ActionPort: '<S20>/Action Port'
       */
      /* Switch: '<S20>/Switch3' incorporates:
       *  Abs: '<S20>/Abs4'
       *  Constant: '<S13>/n_stdStillDet'
       *  Constant: '<S20>/CTRL_COMM4'
       *  Constant: '<S20>/r_errInpTgtThres'
       *  Inport: '<Root>/b_motEna'
       *  Logic: '<S20>/Logical Operator1'
       *  RelationalOperator: '<S13>/Relational Operator9'
       *  RelationalOperator: '<S20>/Relational Operator7'
       *  S-Function (sfix_bitop): '<S20>/Bitwise Operator1'
       *  UnitDelay: '<S20>/UnitDelay'
       *  UnitDelay: '<S8>/UnitDelay4'
       */
      if ((rtDW->UnitDelay_DSTATE_e & 4) != 0) {
        rtb_RelationalOperator1_mv = true;
      } else {
        if (rtDW->UnitDelay4_DSTATE_eu < 0) {
          /* Abs: '<S20>/Abs4' incorporates:
           *  UnitDelay: '<S8>/UnitDelay4'
           */
          rtb_Saturation1 = (int16_T)-rtDW->UnitDelay4_DSTATE_eu;
        } else {
          /* Abs: '<S20>/Abs4' incorporates:
           *  UnitDelay: '<S8>/UnitDelay4'
           */
          rtb_Saturation1 = rtDW->UnitDelay4_DSTATE_eu;
        }

        rtb_RelationalOperator1_mv = (rtU->b_motEna && (Abs5 <
          rtP->n_stdStillDet) && (rtb_Saturation1 > rtP->r_errInpTgtThres));
      }

      /* End of Switch: '<S20>/Switch3' */

      /* Sum: '<S20>/Sum' incorporates:
       *  Constant: '<S20>/CTRL_COMM'
       *  Constant: '<S20>/CTRL_COMM1'
       *  DataTypeConversion: '<S20>/Data Type Conversion3'
       *  Gain: '<S20>/g_Hb'
       *  Gain: '<S20>/g_Hb1'
       *  RelationalOperator: '<S20>/Relational Operator1'
       *  RelationalOperator: '<S20>/Relational Operator3'
       */
      rtb_a_elecAngle_XA_g = (uint8_T)(((uint32_T)((Sum == 7) << 1) + (Sum == 0))
        + (rtb_RelationalOperator1_mv << 2));

      /* Outputs for Atomic SubSystem: '<S20>/Debounce_Filter' */
      Debounce_Filter(rtb_a_elecAngle_XA_g != 0, rtP->t_errQual,
                      rtP->t_errDequal, &rtDW->Merge_p, &rtDW->Debounce_Filter_k);

      /* End of Outputs for SubSystem: '<S20>/Debounce_Filter' */

      /* Outputs for Atomic SubSystem: '<S20>/either_edge' */
      either_edge(rtDW->Merge_p, &rtb_RelationalOperator1_mv,
                  &rtDW->either_edge_i);

      /* End of Outputs for SubSystem: '<S20>/either_edge' */

      /* Switch: '<S20>/Switch1' incorporates:
       *  Constant: '<S20>/CTRL_COMM2'
       *  Constant: '<S20>/t_errDequal'
       *  Constant: '<S20>/t_errQual'
       *  RelationalOperator: '<S20>/Relational Operator2'
       */
      if (rtb_RelationalOperator1_mv) {
        /* Outport: '<Root>/z_errCode' */
        rtY->z_errCode = rtb_a_elecAngle_XA_g;
      } else {
        /* Outport: '<Root>/z_errCode' incorporates:
         *  UnitDelay: '<S20>/UnitDelay'
         */
        rtY->z_errCode = rtDW->UnitDelay_DSTATE_e;
      }

      /* End of Switch: '<S20>/Switch1' */

      /* Update for UnitDelay: '<S20>/UnitDelay' incorporates:
       *  Outport: '<Root>/z_errCode'
       */
      rtDW->UnitDelay_DSTATE_e = rtY->z_errCode;

      /* End of Outputs for SubSystem: '<S4>/Diagnostics_Enabled' */
    }

    /* End of If: '<S4>/If2' */
    /* End of Outputs for SubSystem: '<S1>/F02_Diagnostics' */

    /* Outputs for Function Call SubSystem: '<S1>/F03_Control_Mode_Manager' */
    /* Logic: '<S31>/Logical Operator4' incorporates:
     *  Constant: '<S31>/constant8'
     *  Inport: '<Root>/b_motEna'
     *  Inport: '<Root>/z_ctrlModReq'
     *  Logic: '<S31>/Logical Operator7'
     *  RelationalOperator: '<S31>/Relational Operator10'
     */
    rtb_RelationalOperator1_mv = (rtDW->Merge_p || (!rtU->b_motEna) ||
      (rtU->z_ctrlModReq == 0));

    /* Logic: '<S31>/Logical Operator1' incorporates:
     *  Constant: '<S1>/b_cruiseCtrlEna'
     *  Constant: '<S31>/constant1'
     *  Inport: '<Root>/z_ctrlModReq'
     *  RelationalOperator: '<S31>/Relational Operator1'
     */
    rtb_LogicalOperator1_j = ((rtU->z_ctrlModReq == 2) || rtP->b_cruiseCtrlEna);

    /* Logic: '<S31>/Logical Operator2' incorporates:
     *  Constant: '<S1>/b_cruiseCtrlEna'
     *  Constant: '<S31>/constant'
     *  Inport: '<Root>/z_ctrlModReq'
     *  Logic: '<S31>/Logical Operator5'
     *  RelationalOperator: '<S31>/Relational Operator4'
     */
    rtb_LogicalOperator2_p = ((rtU->z_ctrlModReq == 3) && (!rtP->b_cruiseCtrlEna));

    /* Chart: '<S5>/F03_02_Control_Mode_Manager' incorporates:
     *  Constant: '<S31>/constant5'
     *  Inport: '<Root>/z_ctrlModReq'
     *  Logic: '<S31>/Logical Operator3'
     *  Logic: '<S31>/Logical Operator6'
     *  Logic: '<S31>/Logical Operator9'
     *  RelationalOperator: '<S31>/Relational Operator5'
     */
    if (rtDW->is_active_c1_BLDC_controller == 0U) {
      rtDW->is_active_c1_BLDC_controller = 1U;
      rtDW->is_c1_BLDC_controller = IN_OPEN;
      rtDW->z_ctrlMod = OPEN_MODE;
    } else if (rtDW->is_c1_BLDC_controller == IN_ACTIVE) {
      if (rtb_RelationalOperator1_mv) {
        rtDW->is_ACTIVE = IN_NO_ACTIVE_CHILD;
        rtDW->is_c1_BLDC_controller = IN_OPEN;
        rtDW->z_ctrlMod = OPEN_MODE;
      } else {
        switch (rtDW->is_ACTIVE) {
         case IN_SPEED_MODE:
          rtDW->z_ctrlMod = SPD_MODE;
          if (!rtb_LogicalOperator1_j) {
            rtDW->is_ACTIVE = IN_NO_ACTIVE_CHILD;
            if (rtb_LogicalOperator2_p) {
              rtDW->is_ACTIVE = IN_TORQUE_MODE;
              rtDW->z_ctrlMod = TRQ_MODE;
            } else {
              rtDW->is_ACTIVE = IN_VOLTAGE_MODE;
              rtDW->z_ctrlMod = VLT_MODE;
            }
          }
          break;

         case IN_TORQUE_MODE:
          rtDW->z_ctrlMod = TRQ_MODE;
          if (!rtb_LogicalOperator2_p) {
            rtDW->is_ACTIVE = IN_NO_ACTIVE_CHILD;
            if (rtb_LogicalOperator1_j) {
              rtDW->is_ACTIVE = IN_SPEED_MODE;
              rtDW->z_ctrlMod = SPD_MODE;
            } else {
              rtDW->is_ACTIVE = IN_VOLTAGE_MODE;
              rtDW->z_ctrlMod = VLT_MODE;
            }
          }
          break;

         default:
          rtDW->z_ctrlMod = VLT_MODE;
          if (rtb_LogicalOperator2_p || rtb_LogicalOperator1_j) {
            rtDW->is_ACTIVE = IN_NO_ACTIVE_CHILD;
            if (rtb_LogicalOperator2_p) {
              rtDW->is_ACTIVE = IN_TORQUE_MODE;
              rtDW->z_ctrlMod = TRQ_MODE;
            } else if (rtb_LogicalOperator1_j) {
              rtDW->is_ACTIVE = IN_SPEED_MODE;
              rtDW->z_ctrlMod = SPD_MODE;
            } else {
              rtDW->is_ACTIVE = IN_VOLTAGE_MODE;
              rtDW->z_ctrlMod = VLT_MODE;
            }
          }
          break;
        }
      }
    } else {
      rtDW->z_ctrlMod = OPEN_MODE;
      if ((!rtb_RelationalOperator1_mv) && ((rtU->z_ctrlModReq == 1) ||
           rtb_LogicalOperator1_j || rtb_LogicalOperator2_p)) {
        rtDW->is_c1_BLDC_controller = IN_ACTIVE;
        if (rtb_LogicalOperator2_p) {
          rtDW->is_ACTIVE = IN_TORQUE_MODE;
          rtDW->z_ctrlMod = TRQ_MODE;
        } else if (rtb_LogicalOperator1_j) {
          rtDW->is_ACTIVE = IN_SPEED_MODE;
          rtDW->z_ctrlMod = SPD_MODE;
        } else {
          rtDW->is_ACTIVE = IN_VOLTAGE_MODE;
          rtDW->z_ctrlMod = VLT_MODE;
        }
      }
    }

    /* End of Chart: '<S5>/F03_02_Control_Mode_Manager' */

    /* If: '<S33>/If1' incorporates:
     *  Constant: '<S1>/z_ctrlTypSel'
     *  Inport: '<S34>/r_inpTgt'
     *  Saturate: '<S33>/Saturation'
     */
    if (rtP->z_ctrlTypSel == 2) {
      /* Outputs for IfAction SubSystem: '<S33>/FOC_Control_Type' incorporates:
       *  ActionPort: '<S36>/Action Port'
       */
      /* SignalConversion: '<S36>/TmpSignal ConversionAtSelectorInport1' incorporates:
       *  Constant: '<S36>/Vd_max'
       *  Constant: '<S36>/constant1'
       *  Constant: '<S36>/i_max'
       *  Constant: '<S36>/n_max'
       */
      tmp[0] = 0;
      tmp[1] = rtP->Vd_max;
      tmp[2] = rtP->n_max;
      tmp[3] = rtP->i_max;

      /* End of Outputs for SubSystem: '<S33>/FOC_Control_Type' */

      /* Saturate: '<S33>/Saturation' */
      if (DataTypeConversion2 > 16000) {
        DataTypeConversion2 = 16000;
      } else {
        if (DataTypeConversion2 < -16000) {
          DataTypeConversion2 = -16000;
        }
      }

      /* Outputs for IfAction SubSystem: '<S33>/FOC_Control_Type' incorporates:
       *  ActionPort: '<S36>/Action Port'
       */
      /* Product: '<S36>/Divide1' incorporates:
       *  Inport: '<Root>/z_ctrlModReq'
       *  Product: '<S36>/Divide4'
       *  Selector: '<S36>/Selector'
       */
      rtb_Saturation = (int16_T)(((uint16_T)((tmp[rtU->z_ctrlModReq] << 5) / 125)
        * DataTypeConversion2) >> 12);

      /* End of Outputs for SubSystem: '<S33>/FOC_Control_Type' */
    } else if (DataTypeConversion2 > 16000) {
      /* Outputs for IfAction SubSystem: '<S33>/Default_Control_Type' incorporates:
       *  ActionPort: '<S34>/Action Port'
       */
      /* Saturate: '<S33>/Saturation' incorporates:
       *  Inport: '<S34>/r_inpTgt'
       */
      rtb_Saturation = 16000;

      /* End of Outputs for SubSystem: '<S33>/Default_Control_Type' */
    } else if (DataTypeConversion2 < -16000) {
      /* Outputs for IfAction SubSystem: '<S33>/Default_Control_Type' incorporates:
       *  ActionPort: '<S34>/Action Port'
       */
      /* Saturate: '<S33>/Saturation' incorporates:
       *  Inport: '<S34>/r_inpTgt'
       */
      rtb_Saturation = -16000;

      /* End of Outputs for SubSystem: '<S33>/Default_Control_Type' */
    } else {
      /* Outputs for IfAction SubSystem: '<S33>/Default_Control_Type' incorporates:
       *  ActionPort: '<S34>/Action Port'
       */
      rtb_Saturation = DataTypeConversion2;

      /* End of Outputs for SubSystem: '<S33>/Default_Control_Type' */
    }

    /* End of If: '<S33>/If1' */

    /* If: '<S33>/If2' incorporates:
     *  Inport: '<S35>/r_inpTgtScaRaw'
     */
    rtb_Sum2_h = rtDW->If2_ActiveSubsystem_f;
    UnitDelay3 = (int8_T)!(rtDW->z_ctrlMod == 0);
    rtDW->If2_ActiveSubsystem_f = UnitDelay3;
    switch (UnitDelay3) {
     case 0:
      if (UnitDelay3 != rtb_Sum2_h) {
        /* SystemReset for IfAction SubSystem: '<S33>/Open_Mode' incorporates:
         *  ActionPort: '<S37>/Action Port'
         */
        /* SystemReset for Atomic SubSystem: '<S37>/rising_edge_init' */
        /* SystemReset for If: '<S33>/If2' incorporates:
         *  UnitDelay: '<S39>/UnitDelay'
         *  UnitDelay: '<S40>/UnitDelay'
         */
        rtDW->UnitDelay_DSTATE_b = true;

        /* End of SystemReset for SubSystem: '<S37>/rising_edge_init' */

        /* SystemReset for Atomic SubSystem: '<S37>/Rate_Limiter' */
        rtDW->UnitDelay_DSTATE = 0;

        /* End of SystemReset for SubSystem: '<S37>/Rate_Limiter' */
        /* End of SystemReset for SubSystem: '<S33>/Open_Mode' */
      }

      /* Outputs for IfAction SubSystem: '<S33>/Open_Mode' incorporates:
       *  ActionPort: '<S37>/Action Port'
       */
      /* DataTypeConversion: '<S37>/Data Type Conversion' incorporates:
       *  UnitDelay: '<S8>/UnitDelay4'
       */
      rtb_Gain3 = rtDW->UnitDelay4_DSTATE_eu << 12;
      rtb_Sum1_jt = (rtb_Gain3 & 134217728) != 0 ? rtb_Gain3 | -134217728 :
        rtb_Gain3 & 134217727;

      /* Outputs for Atomic SubSystem: '<S37>/rising_edge_init' */
      /* UnitDelay: '<S39>/UnitDelay' */
      rtb_RelationalOperator1_mv = rtDW->UnitDelay_DSTATE_b;

      /* Update for UnitDelay: '<S39>/UnitDelay' incorporates:
       *  Constant: '<S39>/Constant'
       */
      rtDW->UnitDelay_DSTATE_b = false;

      /* End of Outputs for SubSystem: '<S37>/rising_edge_init' */

      /* Outputs for Atomic SubSystem: '<S37>/Rate_Limiter' */
      /* Switch: '<S40>/Switch1' incorporates:
       *  UnitDelay: '<S40>/UnitDelay'
       */
      if (rtb_RelationalOperator1_mv) {
        rtb_Switch1 = rtb_Sum1_jt;
      } else {
        rtb_Switch1 = rtDW->UnitDelay_DSTATE;
      }

      /* End of Switch: '<S40>/Switch1' */

      /* Sum: '<S38>/Sum1' */
      rtb_Gain3 = -rtb_Switch1;
      rtb_Sum1 = (rtb_Gain3 & 134217728) != 0 ? rtb_Gain3 | -134217728 :
        rtb_Gain3 & 134217727;

      /* Switch: '<S41>/Switch2' incorporates:
       *  Constant: '<S37>/dV_openRate'
       *  RelationalOperator: '<S41>/LowerRelop1'
       */
      if (rtb_Sum1 > rtP->dV_openRate) {
        rtb_Sum1 = rtP->dV_openRate;
      } else {
        /* Gain: '<S37>/Gain3' */
        rtb_Gain3 = -rtP->dV_openRate;
        rtb_Gain3 = (rtb_Gain3 & 134217728) != 0 ? rtb_Gain3 | -134217728 :
          rtb_Gain3 & 134217727;

        /* Switch: '<S41>/Switch' incorporates:
         *  RelationalOperator: '<S41>/UpperRelop'
         */
        if (rtb_Sum1 < rtb_Gain3) {
          rtb_Sum1 = rtb_Gain3;
        }

        /* End of Switch: '<S41>/Switch' */
      }

      /* End of Switch: '<S41>/Switch2' */

      /* Sum: '<S38>/Sum2' */
      rtb_Gain3 = rtb_Sum1 + rtb_Switch1;
      rtb_Switch1 = (rtb_Gain3 & 134217728) != 0 ? rtb_Gain3 | -134217728 :
        rtb_Gain3 & 134217727;

      /* Switch: '<S40>/Switch2' */
      if (rtb_RelationalOperator1_mv) {
        /* Update for UnitDelay: '<S40>/UnitDelay' */
        rtDW->UnitDelay_DSTATE = rtb_Sum1_jt;
      } else {
        /* Update for UnitDelay: '<S40>/UnitDelay' */
        rtDW->UnitDelay_DSTATE = rtb_Switch1;
      }

      /* End of Switch: '<S40>/Switch2' */
      /* End of Outputs for SubSystem: '<S37>/Rate_Limiter' */

      /* DataTypeConversion: '<S37>/Data Type Conversion1' */
      rtDW->Merge1 = (int16_T)(rtb_Switch1 >> 12);

      /* End of Outputs for SubSystem: '<S33>/Open_Mode' */
      break;

     case 1:
      /* Outputs for IfAction SubSystem: '<S33>/Default_Mode' incorporates:
       *  ActionPort: '<S35>/Action Port'
       */
      rtDW->Merge1 = rtb_Saturation;

      /* End of Outputs for SubSystem: '<S33>/Default_Mode' */
      break;
    }

    /* End of If: '<S33>/If2' */

    /* Abs: '<S5>/Abs1' */
    if (rtDW->Merge1 < 0) {
      rtDW->Abs1 = (int16_T)-rtDW->Merge1;
    } else {
      rtDW->Abs1 = rtDW->Merge1;
    }

    /* End of Abs: '<S5>/Abs1' */
    /* End of Outputs for SubSystem: '<S1>/F03_Control_Mode_Manager' */
  } else if (rtDW->UnitDelay5_DSTATE_m) {
    /* Outputs for Function Call SubSystem: '<S1>/F04_Field_Weakening' */
    /* If: '<S6>/If3' incorporates:
     *  Constant: '<S6>/b_fieldWeakEna'
     */
    if (rtP->b_fieldWeakEna) {
      /* Outputs for IfAction SubSystem: '<S6>/Field_Weakening_Enabled' incorporates:
       *  ActionPort: '<S42>/Action Port'
       */
      /* Abs: '<S42>/Abs5' */
      if (DataTypeConversion2 < 0) {
        DataTypeConversion2 = (int16_T)-DataTypeConversion2;
      }

      /* End of Abs: '<S42>/Abs5' */

      /* Switch: '<S44>/Switch2' incorporates:
       *  Constant: '<S42>/r_fieldWeakHi'
       *  Constant: '<S42>/r_fieldWeakLo'
       *  RelationalOperator: '<S44>/LowerRelop1'
       *  RelationalOperator: '<S44>/UpperRelop'
       *  Switch: '<S44>/Switch'
       */
      if (DataTypeConversion2 > rtP->r_fieldWeakHi) {
        DataTypeConversion2 = rtP->r_fieldWeakHi;
      } else {
        if (DataTypeConversion2 < rtP->r_fieldWeakLo) {
          /* Switch: '<S44>/Switch' incorporates:
           *  Constant: '<S42>/r_fieldWeakLo'
           */
          DataTypeConversion2 = rtP->r_fieldWeakLo;
        }
      }

      /* End of Switch: '<S44>/Switch2' */

      /* Product: '<S42>/Divide14' incorporates:
       *  Constant: '<S42>/r_fieldWeakHi'
       *  Constant: '<S42>/r_fieldWeakLo'
       *  Sum: '<S42>/Sum1'
       *  Sum: '<S42>/Sum3'
       */
      rtb_Divide14_e = (uint16_T)(((int16_T)(DataTypeConversion2 -
        rtP->r_fieldWeakLo) << 15) / (int16_T)(rtP->r_fieldWeakHi -
        rtP->r_fieldWeakLo));

      /* Switch: '<S43>/Switch2' incorporates:
       *  Constant: '<S42>/n_fieldWeakAuthHi'
       *  Constant: '<S42>/n_fieldWeakAuthLo'
       *  RelationalOperator: '<S43>/LowerRelop1'
       *  RelationalOperator: '<S43>/UpperRelop'
       *  Switch: '<S43>/Switch'
       */
      if (Abs5 > rtP->n_fieldWeakAuthHi) {
        rtb_Saturation = rtP->n_fieldWeakAuthHi;
      } else if (Abs5 < rtP->n_fieldWeakAuthLo) {
        /* Switch: '<S43>/Switch' incorporates:
         *  Constant: '<S42>/n_fieldWeakAuthLo'
         */
        rtb_Saturation = rtP->n_fieldWeakAuthLo;
      } else {
        rtb_Saturation = Abs5;
      }

      /* End of Switch: '<S43>/Switch2' */

      /* Product: '<S42>/Divide1' incorporates:
       *  Constant: '<S42>/n_fieldWeakAuthHi'
       *  Constant: '<S42>/n_fieldWeakAuthLo'
       *  Sum: '<S42>/Sum2'
       *  Sum: '<S42>/Sum4'
       */
      rtb_Divide1_f = (uint16_T)(((int16_T)(rtb_Saturation -
        rtP->n_fieldWeakAuthLo) << 15) / (int16_T)(rtP->n_fieldWeakAuthHi -
        rtP->n_fieldWeakAuthLo));

      /* Switch: '<S42>/Switch1' incorporates:
       *  MinMax: '<S42>/MinMax1'
       *  RelationalOperator: '<S42>/Relational Operator6'
       */
      if (rtb_Divide14_e < rtb_Divide1_f) {
        /* MinMax: '<S42>/MinMax' */
        if (!(rtb_Divide14_e > rtb_Divide1_f)) {
          rtb_Divide14_e = rtb_Divide1_f;
        }

        /* End of MinMax: '<S42>/MinMax' */
      } else {
        if (rtb_Divide1_f < rtb_Divide14_e) {
          /* MinMax: '<S42>/MinMax1' */
          rtb_Divide14_e = rtb_Divide1_f;
        }
      }

      /* End of Switch: '<S42>/Switch1' */

      /* Switch: '<S42>/Switch2' incorporates:
       *  Constant: '<S1>/z_ctrlTypSel'
       *  Constant: '<S42>/CTRL_COMM2'
       *  Constant: '<S42>/a_phaAdvMax'
       *  Constant: '<S42>/id_fieldWeakMax'
       *  RelationalOperator: '<S42>/Relational Operator1'
       */
      if (rtP->z_ctrlTypSel == 2) {
        rtb_Saturation1 = rtP->id_fieldWeakMax;
      } else {
        rtb_Saturation1 = rtP->a_phaAdvMax;
      }

      /* End of Switch: '<S42>/Switch2' */

      /* Product: '<S42>/Divide3' */
      rtDW->Divide3 = (int16_T)((rtb_Saturation1 * rtb_Divide14_e) >> 15);

      /* End of Outputs for SubSystem: '<S6>/Field_Weakening_Enabled' */
    }

    /* End of If: '<S6>/If3' */
    /* End of Outputs for SubSystem: '<S1>/F04_Field_Weakening' */

    /* Outputs for Function Call SubSystem: '<S7>/Motor_Limitations' */
    /* If: '<S48>/If1' incorporates:
     *  Constant: '<S1>/z_ctrlTypSel'
     *  Constant: '<S80>/Vd_max1'
     *  Constant: '<S80>/i_max'
     */
    rtb_Sum2_h = rtDW->If1_ActiveSubsystem_o;
    UnitDelay3 = -1;
    if (rtP->z_ctrlTypSel == 2) {
      UnitDelay3 = 0;
    }

    rtDW->If1_ActiveSubsystem_o = UnitDelay3;
    if ((rtb_Sum2_h != UnitDelay3) && (rtb_Sum2_h == 0)) {
      /* Disable for SwitchCase: '<S80>/Switch Case' */
      rtDW->SwitchCase_ActiveSubsystem_d = -1;
    }

    if (UnitDelay3 == 0) {
      /* Outputs for IfAction SubSystem: '<S48>/Motor_Limitations_Enabled' incorporates:
       *  ActionPort: '<S80>/Action Port'
       */
      rtDW->Vd_max1 = rtP->Vd_max;

      /* Gain: '<S80>/Gain3' incorporates:
       *  Constant: '<S80>/Vd_max1'
       */
      rtDW->Gain3 = (int16_T)-rtDW->Vd_max1;

      /* Interpolation_n-D: '<S80>/Vq_max_M1' incorporates:
       *  Abs: '<S80>/Abs5'
       *  PreLookup: '<S80>/Vq_max_XA'
       *  UnitDelay: '<S7>/UnitDelay4'
       */
      if (rtDW->Switch1 < 0) {
        rtb_Saturation1 = (int16_T)-rtDW->Switch1;
      } else {
        rtb_Saturation1 = rtDW->Switch1;
      }

      rtDW->Vq_max_M1 = rtP->Vq_max_M1[plook_u8s16_evencka(rtb_Saturation1,
        rtP->Vq_max_XA[0], (uint16_T)(rtP->Vq_max_XA[1] - rtP->Vq_max_XA[0]),
        45U)];

      /* End of Interpolation_n-D: '<S80>/Vq_max_M1' */

      /* Gain: '<S80>/Gain5' */
      rtDW->Gain5 = (int16_T)-rtDW->Vq_max_M1;
      rtDW->i_max = rtP->i_max;

      /* Interpolation_n-D: '<S80>/iq_maxSca_M1' incorporates:
       *  Constant: '<S80>/i_max'
       *  Product: '<S80>/Divide4'
       */
      rtb_Gain3 = rtDW->Divide3 << 16;
      rtb_Gain3 = (rtb_Gain3 == MIN_int32_T) && (rtDW->i_max == -1) ?
        MAX_int32_T : rtb_Gain3 / rtDW->i_max;
      if (rtb_Gain3 < 0) {
        rtb_Gain3 = 0;
      } else {
        if (rtb_Gain3 > 65535) {
          rtb_Gain3 = 65535;
        }
      }

      /* Product: '<S80>/Divide1' incorporates:
       *  Interpolation_n-D: '<S80>/iq_maxSca_M1'
       *  PreLookup: '<S80>/iq_maxSca_XA'
       *  Product: '<S80>/Divide4'
       */
      rtDW->Divide1_n = (int16_T)
        ((rtConstP.iq_maxSca_M1_Table[plook_u8u16_evencka((uint16_T)rtb_Gain3,
           0U, 1311U, 49U)] * rtDW->i_max) >> 16);

      /* Gain: '<S80>/Gain1' */
      rtDW->Gain1 = (int16_T)-rtDW->Divide1_n;

      /* SwitchCase: '<S80>/Switch Case' incorporates:
       *  Constant: '<S80>/n_max1'
       *  Constant: '<S82>/Constant1'
       *  Constant: '<S82>/cf_KbLimProt'
       *  Constant: '<S82>/cf_nKiLimProt'
       *  Constant: '<S83>/Constant'
       *  Constant: '<S83>/Constant1'
       *  Constant: '<S83>/cf_KbLimProt'
       *  Constant: '<S83>/cf_iqKiLimProt'
       *  Constant: '<S83>/cf_nKiLimProt'
       *  Sum: '<S82>/Sum1'
       *  Sum: '<S83>/Sum1'
       *  Sum: '<S83>/Sum2'
       */
      rtb_Sum2_h = rtDW->SwitchCase_ActiveSubsystem_d;
      UnitDelay3 = -1;
      switch (rtDW->z_ctrlMod) {
       case 1:
        UnitDelay3 = 0;
        break;

       case 2:
        UnitDelay3 = 1;
        break;

       case 3:
        UnitDelay3 = 2;
        break;
      }

      rtDW->SwitchCase_ActiveSubsystem_d = UnitDelay3;
      switch (UnitDelay3) {
       case 0:
        if (UnitDelay3 != rtb_Sum2_h) {
          /* SystemReset for IfAction SubSystem: '<S80>/Voltage_Mode_Protection' incorporates:
           *  ActionPort: '<S83>/Action Port'
           */

          /* SystemReset for Atomic SubSystem: '<S83>/I_backCalc_fixdt' */

          /* SystemReset for SwitchCase: '<S80>/Switch Case' */
          I_backCalc_fixdt_Reset(&rtDW->I_backCalc_fixdt_i, 65536000);

          /* End of SystemReset for SubSystem: '<S83>/I_backCalc_fixdt' */

          /* SystemReset for Atomic SubSystem: '<S83>/I_backCalc_fixdt1' */
          I_backCalc_fixdt_Reset(&rtDW->I_backCalc_fixdt1, 65536000);

          /* End of SystemReset for SubSystem: '<S83>/I_backCalc_fixdt1' */

          /* End of SystemReset for SubSystem: '<S80>/Voltage_Mode_Protection' */
        }

        /* Outputs for IfAction SubSystem: '<S80>/Voltage_Mode_Protection' incorporates:
         *  ActionPort: '<S83>/Action Port'
         */

        /* Outputs for Atomic SubSystem: '<S83>/I_backCalc_fixdt' */
        I_backCalc_fixdt((int16_T)(rtDW->Divide1_n - rtDW->Abs5_h),
                         rtP->cf_iqKiLimProt, rtP->cf_KbLimProt, rtDW->Abs1, 0,
                         &rtDW->Switch2_a, &rtDW->I_backCalc_fixdt_i);

        /* End of Outputs for SubSystem: '<S83>/I_backCalc_fixdt' */

        /* Outputs for Atomic SubSystem: '<S83>/I_backCalc_fixdt1' */
        I_backCalc_fixdt((int16_T)(rtP->n_max - Abs5), rtP->cf_nKiLimProt,
                         rtP->cf_KbLimProt, rtDW->Abs1, 0, &rtDW->Switch2_o,
                         &rtDW->I_backCalc_fixdt1);

        /* End of Outputs for SubSystem: '<S83>/I_backCalc_fixdt1' */

        /* End of Outputs for SubSystem: '<S80>/Voltage_Mode_Protection' */
        break;

       case 1:
        /* Outputs for IfAction SubSystem: '<S80>/Speed_Mode_Protection' incorporates:
         *  ActionPort: '<S81>/Action Port'
         */
        /* Switch: '<S84>/Switch2' incorporates:
         *  RelationalOperator: '<S84>/LowerRelop1'
         *  RelationalOperator: '<S84>/UpperRelop'
         *  Switch: '<S84>/Switch'
         */
        if (rtDW->DataTypeConversion[0] > rtDW->Divide1_n) {
          rtb_Saturation1 = rtDW->Divide1_n;
        } else if (rtDW->DataTypeConversion[0] < rtDW->Gain1) {
          /* Switch: '<S84>/Switch' */
          rtb_Saturation1 = rtDW->Gain1;
        } else {
          rtb_Saturation1 = rtDW->DataTypeConversion[0];
        }

        /* End of Switch: '<S84>/Switch2' */

        /* Product: '<S81>/Divide1' incorporates:
         *  Constant: '<S81>/cf_iqKiLimProt'
         *  Sum: '<S81>/Sum3'
         */
        rtDW->Divide1 = (int16_T)(rtb_Saturation1 - rtDW->DataTypeConversion[0])
          * rtP->cf_iqKiLimProt;

        /* End of Outputs for SubSystem: '<S80>/Speed_Mode_Protection' */
        break;

       case 2:
        if (UnitDelay3 != rtb_Sum2_h) {
          /* SystemReset for IfAction SubSystem: '<S80>/Torque_Mode_Protection' incorporates:
           *  ActionPort: '<S82>/Action Port'
           */

          /* SystemReset for Atomic SubSystem: '<S82>/I_backCalc_fixdt' */

          /* SystemReset for SwitchCase: '<S80>/Switch Case' */
          I_backCalc_fixdt_Reset(&rtDW->I_backCalc_fixdt_j, 58982400);

          /* End of SystemReset for SubSystem: '<S82>/I_backCalc_fixdt' */

          /* End of SystemReset for SubSystem: '<S80>/Torque_Mode_Protection' */
        }

        /* Outputs for IfAction SubSystem: '<S80>/Torque_Mode_Protection' incorporates:
         *  ActionPort: '<S82>/Action Port'
         */

        /* Outputs for Atomic SubSystem: '<S82>/I_backCalc_fixdt' */
        I_backCalc_fixdt((int16_T)(rtP->n_max - Abs5), rtP->cf_nKiLimProt,
                         rtP->cf_KbLimProt, rtDW->Vq_max_M1, 0, &rtDW->Switch2_i,
                         &rtDW->I_backCalc_fixdt_j);

        /* End of Outputs for SubSystem: '<S82>/I_backCalc_fixdt' */

        /* End of Outputs for SubSystem: '<S80>/Torque_Mode_Protection' */
        break;
      }

      /* End of SwitchCase: '<S80>/Switch Case' */

      /* Gain: '<S80>/Gain4' */
      rtDW->Gain4 = (int16_T)-rtDW->i_max;

      /* End of Outputs for SubSystem: '<S48>/Motor_Limitations_Enabled' */
    }

    /* End of If: '<S48>/If1' */
    /* End of Outputs for SubSystem: '<S7>/Motor_Limitations' */
  } else {
    if (rtDW->UnitDelay6_DSTATE) {
      /* Outputs for Function Call SubSystem: '<S7>/FOC' */
      /* If: '<S47>/If1' incorporates:
       *  Constant: '<S1>/z_ctrlTypSel'
       */
      rtb_Sum2_h = rtDW->If1_ActiveSubsystem_j;
      UnitDelay3 = -1;
      if (rtP->z_ctrlTypSel == 2) {
        UnitDelay3 = 0;
      }

      rtDW->If1_ActiveSubsystem_j = UnitDelay3;
      if ((rtb_Sum2_h != UnitDelay3) && (rtb_Sum2_h == 0)) {
        /* Disable for SwitchCase: '<S59>/Switch Case' */
        rtDW->SwitchCase_ActiveSubsystem = -1;

        /* Disable for If: '<S59>/If1' */
        rtDW->If1_ActiveSubsystem_a = -1;
      }

      if (UnitDelay3 == 0) {
        /* Outputs for IfAction SubSystem: '<S47>/FOC_Enabled' incorporates:
         *  ActionPort: '<S59>/Action Port'
         */
        /* SwitchCase: '<S59>/Switch Case' incorporates:
         *  Constant: '<S61>/cf_nKi'
         *  Constant: '<S61>/cf_nKp'
         *  Inport: '<S60>/r_inpTgtSca'
         *  Sum: '<S61>/Sum3'
         *  UnitDelay: '<S8>/UnitDelay4'
         */
        rtb_Sum2_h = rtDW->SwitchCase_ActiveSubsystem;
        switch (rtDW->z_ctrlMod) {
         case 1:
          break;

         case 2:
          UnitDelay3 = 1;
          break;

         case 3:
          UnitDelay3 = 2;
          break;

         default:
          UnitDelay3 = 3;
          break;
        }

        rtDW->SwitchCase_ActiveSubsystem = UnitDelay3;
        switch (UnitDelay3) {
         case 0:
          /* Outputs for IfAction SubSystem: '<S59>/Voltage_Mode' incorporates:
           *  ActionPort: '<S64>/Action Port'
           */
          /* MinMax: '<S64>/MinMax' */
          if (rtDW->Abs1 < rtDW->Switch2_a) {
            DataTypeConversion2 = rtDW->Abs1;
          } else {
            DataTypeConversion2 = rtDW->Switch2_a;
          }

          if (!(DataTypeConversion2 < rtDW->Switch2_o)) {
            DataTypeConversion2 = rtDW->Switch2_o;
          }

          /* End of MinMax: '<S64>/MinMax' */

          /* Signum: '<S64>/SignDeltaU2' */
          if (rtDW->Merge1 < 0) {
            rtb_Saturation1 = -1;
          } else {
            rtb_Saturation1 = (int16_T)(rtDW->Merge1 > 0);
          }

          /* End of Signum: '<S64>/SignDeltaU2' */

          /* Product: '<S64>/Divide1' */
          rtb_Saturation = (int16_T)(DataTypeConversion2 * rtb_Saturation1);

          /* Switch: '<S79>/Switch2' incorporates:
           *  RelationalOperator: '<S79>/LowerRelop1'
           *  RelationalOperator: '<S79>/UpperRelop'
           *  Switch: '<S79>/Switch'
           */
          if (rtb_Saturation > rtDW->Vq_max_M1) {
            /* SignalConversion: '<S64>/Signal Conversion2' */
            rtDW->Merge = rtDW->Vq_max_M1;
          } else if (rtb_Saturation < rtDW->Gain5) {
            /* Switch: '<S79>/Switch' incorporates:
             *  SignalConversion: '<S64>/Signal Conversion2'
             */
            rtDW->Merge = rtDW->Gain5;
          } else {
            /* SignalConversion: '<S64>/Signal Conversion2' incorporates:
             *  Switch: '<S79>/Switch'
             */
            rtDW->Merge = rtb_Saturation;
          }

          /* End of Switch: '<S79>/Switch2' */
          /* End of Outputs for SubSystem: '<S59>/Voltage_Mode' */
          break;

         case 1:
          if (UnitDelay3 != rtb_Sum2_h) {
            /* SystemReset for IfAction SubSystem: '<S59>/Speed_Mode' incorporates:
             *  ActionPort: '<S61>/Action Port'
             */

            /* SystemReset for Atomic SubSystem: '<S61>/PI_clamp_fixdt' */

            /* SystemReset for SwitchCase: '<S59>/Switch Case' */
            PI_clamp_fixdt_b_Reset(&rtDW->PI_clamp_fixdt_l4);

            /* End of SystemReset for SubSystem: '<S61>/PI_clamp_fixdt' */

            /* End of SystemReset for SubSystem: '<S59>/Speed_Mode' */
          }

          /* Outputs for IfAction SubSystem: '<S59>/Speed_Mode' incorporates:
           *  ActionPort: '<S61>/Action Port'
           */
          /* DataTypeConversion: '<S61>/Data Type Conversion2' incorporates:
           *  Constant: '<S61>/n_cruiseMotTgt'
           */
          rtb_Saturation = (int16_T)(rtP->n_cruiseMotTgt << 4);

          /* Switch: '<S61>/Switch4' incorporates:
           *  Constant: '<S1>/b_cruiseCtrlEna'
           *  Logic: '<S61>/Logical Operator1'
           *  RelationalOperator: '<S61>/Relational Operator3'
           */
          if (rtP->b_cruiseCtrlEna && (rtb_Saturation != 0)) {
            /* Switch: '<S61>/Switch3' incorporates:
             *  MinMax: '<S61>/MinMax4'
             */
            if (rtb_Saturation > 0) {
              rtb_TmpSignalConversionAtLow_Pa[0] = rtDW->Vq_max_M1;

              /* MinMax: '<S61>/MinMax3' */
              if (rtDW->Merge1 > rtDW->Gain5) {
                rtb_TmpSignalConversionAtLow_Pa[1] = rtDW->Merge1;
              } else {
                rtb_TmpSignalConversionAtLow_Pa[1] = rtDW->Gain5;
              }

              /* End of MinMax: '<S61>/MinMax3' */
            } else {
              if (rtDW->Vq_max_M1 < rtDW->Merge1) {
                /* MinMax: '<S61>/MinMax4' */
                rtb_TmpSignalConversionAtLow_Pa[0] = rtDW->Vq_max_M1;
              } else {
                rtb_TmpSignalConversionAtLow_Pa[0] = rtDW->Merge1;
              }

              rtb_TmpSignalConversionAtLow_Pa[1] = rtDW->Gain5;
            }

            /* End of Switch: '<S61>/Switch3' */
          } else {
            rtb_TmpSignalConversionAtLow_Pa[0] = rtDW->Vq_max_M1;
            rtb_TmpSignalConversionAtLow_Pa[1] = rtDW->Gain5;
          }

          /* End of Switch: '<S61>/Switch4' */

          /* Switch: '<S61>/Switch2' incorporates:
           *  Constant: '<S1>/b_cruiseCtrlEna'
           */
          if (!rtP->b_cruiseCtrlEna) {
            rtb_Saturation = rtDW->Merge1;
          }

          /* End of Switch: '<S61>/Switch2' */

          /* Sum: '<S61>/Sum3' */
          rtb_Gain3 = rtb_Saturation - Switch2;
          if (rtb_Gain3 > 32767) {
            rtb_Gain3 = 32767;
          } else {
            if (rtb_Gain3 < -32768) {
              rtb_Gain3 = -32768;
            }
          }

          /* Outputs for Atomic SubSystem: '<S61>/PI_clamp_fixdt' */
          PI_clamp_fixdt_l((int16_T)rtb_Gain3, rtP->cf_nKp, rtP->cf_nKi,
                           rtDW->UnitDelay4_DSTATE_eu,
                           rtb_TmpSignalConversionAtLow_Pa[0],
                           rtb_TmpSignalConversionAtLow_Pa[1], rtDW->Divide1,
                           &rtDW->Merge, &rtDW->PI_clamp_fixdt_l4);

          /* End of Outputs for SubSystem: '<S61>/PI_clamp_fixdt' */

          /* End of Outputs for SubSystem: '<S59>/Speed_Mode' */
          break;

         case 2:
          if (UnitDelay3 != rtb_Sum2_h) {
            /* SystemReset for IfAction SubSystem: '<S59>/Torque_Mode' incorporates:
             *  ActionPort: '<S62>/Action Port'
             */

            /* SystemReset for Atomic SubSystem: '<S62>/PI_clamp_fixdt' */

            /* SystemReset for SwitchCase: '<S59>/Switch Case' */
            PI_clamp_fixdt_g_Reset(&rtDW->PI_clamp_fixdt_kh);

            /* End of SystemReset for SubSystem: '<S62>/PI_clamp_fixdt' */

            /* End of SystemReset for SubSystem: '<S59>/Torque_Mode' */
          }

          /* Outputs for IfAction SubSystem: '<S59>/Torque_Mode' incorporates:
           *  ActionPort: '<S62>/Action Port'
           */
          /* Gain: '<S62>/Gain4' */
          rtb_Saturation = (int16_T)-rtDW->Switch2_i;

          /* Switch: '<S70>/Switch2' incorporates:
           *  RelationalOperator: '<S70>/LowerRelop1'
           *  RelationalOperator: '<S70>/UpperRelop'
           *  Switch: '<S70>/Switch'
           */
          if (rtDW->Merge1 > rtDW->Divide1_n) {
            rtb_Saturation1 = rtDW->Divide1_n;
          } else if (rtDW->Merge1 < rtDW->Gain1) {
            /* Switch: '<S70>/Switch' */
            rtb_Saturation1 = rtDW->Gain1;
          } else {
            rtb_Saturation1 = rtDW->Merge1;
          }

          /* End of Switch: '<S70>/Switch2' */

          /* Sum: '<S62>/Sum2' */
          rtb_Gain3 = rtb_Saturation1 - rtDW->DataTypeConversion[0];
          if (rtb_Gain3 > 32767) {
            rtb_Gain3 = 32767;
          } else {
            if (rtb_Gain3 < -32768) {
              rtb_Gain3 = -32768;
            }
          }

          /* MinMax: '<S62>/MinMax1' */
          if (rtDW->Vq_max_M1 < rtDW->Switch2_i) {
            rtb_Saturation1 = rtDW->Vq_max_M1;
          } else {
            rtb_Saturation1 = rtDW->Switch2_i;
          }

          /* End of MinMax: '<S62>/MinMax1' */

          /* MinMax: '<S62>/MinMax2' */
          if (!(rtb_Saturation > rtDW->Gain5)) {
            rtb_Saturation = rtDW->Gain5;
          }

          /* End of MinMax: '<S62>/MinMax2' */

          /* Outputs for Atomic SubSystem: '<S62>/PI_clamp_fixdt' */

          /* SignalConversion: '<S62>/Signal Conversion2' incorporates:
           *  Constant: '<S62>/cf_iqKi'
           *  Constant: '<S62>/cf_iqKp'
           *  Constant: '<S62>/constant2'
           *  Sum: '<S62>/Sum2'
           *  UnitDelay: '<S8>/UnitDelay4'
           */
          PI_clamp_fixdt_k((int16_T)rtb_Gain3, rtP->cf_iqKp, rtP->cf_iqKi,
                           rtDW->UnitDelay4_DSTATE_eu, rtb_Saturation1,
                           rtb_Saturation, 0, &rtDW->Merge,
                           &rtDW->PI_clamp_fixdt_kh);

          /* End of Outputs for SubSystem: '<S62>/PI_clamp_fixdt' */

          /* End of Outputs for SubSystem: '<S59>/Torque_Mode' */
          break;

         case 3:
          /* Outputs for IfAction SubSystem: '<S59>/Open_Mode' incorporates:
           *  ActionPort: '<S60>/Action Port'
           */
          rtDW->Merge = rtDW->Merge1;

          /* End of Outputs for SubSystem: '<S59>/Open_Mode' */
          break;
        }

        /* End of SwitchCase: '<S59>/Switch Case' */

        /* If: '<S59>/If1' incorporates:
         *  Constant: '<S63>/cf_idKi1'
         *  Constant: '<S63>/cf_idKp1'
         *  Constant: '<S63>/constant1'
         *  Constant: '<S63>/constant2'
         *  Sum: '<S63>/Sum3'
         */
        rtb_Sum2_h = rtDW->If1_ActiveSubsystem_a;
        UnitDelay3 = -1;
        if (rtb_LogicalOperator) {
          UnitDelay3 = 0;
        }

        rtDW->If1_ActiveSubsystem_a = UnitDelay3;
        if (UnitDelay3 == 0) {
          if (0 != rtb_Sum2_h) {
            /* SystemReset for IfAction SubSystem: '<S59>/Vd_Calculation' incorporates:
             *  ActionPort: '<S63>/Action Port'
             */

            /* SystemReset for Atomic SubSystem: '<S63>/PI_clamp_fixdt' */

            /* SystemReset for If: '<S59>/If1' */
            PI_clamp_fixdt_Reset(&rtDW->PI_clamp_fixdt_i);

            /* End of SystemReset for SubSystem: '<S63>/PI_clamp_fixdt' */

            /* End of SystemReset for SubSystem: '<S59>/Vd_Calculation' */
          }

          /* Outputs for IfAction SubSystem: '<S59>/Vd_Calculation' incorporates:
           *  ActionPort: '<S63>/Action Port'
           */
          /* Gain: '<S63>/toNegative' */
          rtb_Saturation = (int16_T)-rtDW->Divide3;

          /* Switch: '<S75>/Switch2' incorporates:
           *  RelationalOperator: '<S75>/LowerRelop1'
           *  RelationalOperator: '<S75>/UpperRelop'
           *  Switch: '<S75>/Switch'
           */
          if (rtb_Saturation > rtDW->i_max) {
            rtb_Saturation = rtDW->i_max;
          } else {
            if (rtb_Saturation < rtDW->Gain4) {
              /* Switch: '<S75>/Switch' */
              rtb_Saturation = rtDW->Gain4;
            }
          }

          /* End of Switch: '<S75>/Switch2' */

          /* Sum: '<S63>/Sum3' */
          rtb_Gain3 = rtb_Saturation - rtDW->DataTypeConversion[1];
          if (rtb_Gain3 > 32767) {
            rtb_Gain3 = 32767;
          } else {
            if (rtb_Gain3 < -32768) {
              rtb_Gain3 = -32768;
            }
          }

          /* Outputs for Atomic SubSystem: '<S63>/PI_clamp_fixdt' */
          PI_clamp_fixdt((int16_T)rtb_Gain3, rtP->cf_idKp, rtP->cf_idKi, 0,
                         rtDW->Vd_max1, rtDW->Gain3, 0, &rtDW->Switch1,
                         &rtDW->PI_clamp_fixdt_i);

          /* End of Outputs for SubSystem: '<S63>/PI_clamp_fixdt' */

          /* End of Outputs for SubSystem: '<S59>/Vd_Calculation' */
        }

        /* End of If: '<S59>/If1' */
        /* End of Outputs for SubSystem: '<S47>/FOC_Enabled' */
      }

      /* End of If: '<S47>/If1' */
      /* End of Outputs for SubSystem: '<S7>/FOC' */
    }
  }

  /* End of Chart: '<S1>/Task_Scheduler' */

  /* If: '<S7>/If2' incorporates:
   *  Constant: '<S1>/z_ctrlTypSel'
   *  Constant: '<S8>/CTRL_COMM1'
   *  RelationalOperator: '<S8>/Relational Operator6'
   *  Switch: '<S8>/Switch2'
   */
  rtb_Sum2_h = rtDW->If2_ActiveSubsystem;
  UnitDelay3 = -1;
  if (rtP->z_ctrlTypSel == 2) {
    rtb_Saturation = rtDW->Merge;
    UnitDelay3 = 0;
  } else {
    rtb_Saturation = rtDW->Merge1;
  }

  rtDW->If2_ActiveSubsystem = UnitDelay3;
  if ((rtb_Sum2_h != UnitDelay3) && (rtb_Sum2_h == 0)) {
    /* Disable for Outport: '<S46>/V_phaABC_FOC' */
    rtDW->Gain4_e[0] = 0;
    rtDW->Gain4_e[1] = 0;
    rtDW->Gain4_e[2] = 0;
  }

  if (UnitDelay3 == 0) {
    /* Outputs for IfAction SubSystem: '<S7>/Clarke_Park_Transform_Inverse' incorporates:
     *  ActionPort: '<S46>/Action Port'
     */
    /* Sum: '<S58>/Sum6' incorporates:
     *  Product: '<S58>/Divide1'
     *  Product: '<S58>/Divide4'
     */
    rtb_Gain3 = (int16_T)((rtDW->Switch1 * rtDW->r_cos_M1) >> 14) - (int16_T)
      ((rtDW->Merge * rtDW->r_sin_M1) >> 14);
    if (rtb_Gain3 > 32767) {
      rtb_Gain3 = 32767;
    } else {
      if (rtb_Gain3 < -32768) {
        rtb_Gain3 = -32768;
      }
    }

    /* Sum: '<S58>/Sum1' incorporates:
     *  Product: '<S58>/Divide2'
     *  Product: '<S58>/Divide3'
     */
    rtb_Sum1_jt = (int16_T)((rtDW->Switch1 * rtDW->r_sin_M1) >> 14) + (int16_T)
      ((rtDW->Merge * rtDW->r_cos_M1) >> 14);
    if (rtb_Sum1_jt > 32767) {
      rtb_Sum1_jt = 32767;
    } else {
      if (rtb_Sum1_jt < -32768) {
        rtb_Sum1_jt = -32768;
      }
    }

    /* Gain: '<S57>/Gain1' incorporates:
     *  Sum: '<S58>/Sum1'
     */
    rtb_Sum1_jt *= 14189;

    /* Sum: '<S57>/Sum6' incorporates:
     *  Gain: '<S57>/Gain1'
     *  Gain: '<S57>/Gain3'
     *  Sum: '<S58>/Sum6'
     */
    rtb_Sum1_jt = (((rtb_Sum1_jt < 0 ? 16383 : 0) + rtb_Sum1_jt) >> 14) -
      ((int16_T)(((int16_T)rtb_Gain3 < 0) + (int16_T)rtb_Gain3) >> 1);
    if (rtb_Sum1_jt > 32767) {
      rtb_Sum1_jt = 32767;
    } else {
      if (rtb_Sum1_jt < -32768) {
        rtb_Sum1_jt = -32768;
      }
    }

    /* Sum: '<S57>/Sum2' incorporates:
     *  Sum: '<S57>/Sum6'
     *  Sum: '<S58>/Sum6'
     */
    rtb_Switch1 = -(int16_T)rtb_Gain3 - (int16_T)rtb_Sum1_jt;
    if (rtb_Switch1 > 32767) {
      rtb_Switch1 = 32767;
    } else {
      if (rtb_Switch1 < -32768) {
        rtb_Switch1 = -32768;
      }
    }

    /* MinMax: '<S57>/MinMax1' incorporates:
     *  Sum: '<S57>/Sum2'
     *  Sum: '<S57>/Sum6'
     *  Sum: '<S58>/Sum6'
     */
    DataTypeConversion2 = (int16_T)rtb_Gain3;
    if (!((int16_T)rtb_Gain3 < (int16_T)rtb_Sum1_jt)) {
      DataTypeConversion2 = (int16_T)rtb_Sum1_jt;
    }

    if (!(DataTypeConversion2 < (int16_T)rtb_Switch1)) {
      DataTypeConversion2 = (int16_T)rtb_Switch1;
    }

    /* MinMax: '<S57>/MinMax2' incorporates:
     *  Sum: '<S57>/Sum2'
     *  Sum: '<S57>/Sum6'
     *  Sum: '<S58>/Sum6'
     */
    rtb_Saturation1 = (int16_T)rtb_Gain3;
    if (!((int16_T)rtb_Gain3 > (int16_T)rtb_Sum1_jt)) {
      rtb_Saturation1 = (int16_T)rtb_Sum1_jt;
    }

    if (!(rtb_Saturation1 > (int16_T)rtb_Switch1)) {
      rtb_Saturation1 = (int16_T)rtb_Switch1;
    }

    /* Sum: '<S57>/Add' incorporates:
     *  MinMax: '<S57>/MinMax1'
     *  MinMax: '<S57>/MinMax2'
     */
    rtb_Sum1 = DataTypeConversion2 + rtb_Saturation1;
    if (rtb_Sum1 > 32767) {
      rtb_Sum1 = 32767;
    } else {
      if (rtb_Sum1 < -32768) {
        rtb_Sum1 = -32768;
      }
    }

    /* Gain: '<S57>/Gain2' incorporates:
     *  Sum: '<S57>/Add'
     */
    rtb_Merge1 = (int16_T)(rtb_Sum1 >> 1);

    /* Sum: '<S57>/Add1' incorporates:
     *  Sum: '<S58>/Sum6'
     */
    rtb_Gain3 = (int16_T)rtb_Gain3 - rtb_Merge1;
    if (rtb_Gain3 > 32767) {
      rtb_Gain3 = 32767;
    } else {
      if (rtb_Gain3 < -32768) {
        rtb_Gain3 = -32768;
      }
    }

    /* Gain: '<S57>/Gain4' incorporates:
     *  Sum: '<S57>/Add1'
     */
    rtDW->Gain4_e[0] = (int16_T)((18919 * rtb_Gain3) >> 14);

    /* Sum: '<S57>/Add1' incorporates:
     *  Sum: '<S57>/Sum6'
     */
    rtb_Gain3 = (int16_T)rtb_Sum1_jt - rtb_Merge1;
    if (rtb_Gain3 > 32767) {
      rtb_Gain3 = 32767;
    } else {
      if (rtb_Gain3 < -32768) {
        rtb_Gain3 = -32768;
      }
    }

    /* Gain: '<S57>/Gain4' incorporates:
     *  Sum: '<S57>/Add1'
     */
    rtDW->Gain4_e[1] = (int16_T)((18919 * rtb_Gain3) >> 14);

    /* Sum: '<S57>/Add1' incorporates:
     *  Sum: '<S57>/Sum2'
     */
    rtb_Gain3 = (int16_T)rtb_Switch1 - rtb_Merge1;
    if (rtb_Gain3 > 32767) {
      rtb_Gain3 = 32767;
    } else {
      if (rtb_Gain3 < -32768) {
        rtb_Gain3 = -32768;
      }
    }

    /* Gain: '<S57>/Gain4' incorporates:
     *  Sum: '<S57>/Add1'
     */
    rtDW->Gain4_e[2] = (int16_T)((18919 * rtb_Gain3) >> 14);

    /* End of Outputs for SubSystem: '<S7>/Clarke_Park_Transform_Inverse' */
  }

  /* End of If: '<S7>/If2' */

  /* If: '<S8>/If' incorporates:
   *  Constant: '<S11>/vec_hallToPos'
   *  Constant: '<S1>/z_ctrlTypSel'
   *  Constant: '<S8>/CTRL_COMM2'
   *  Constant: '<S8>/CTRL_COMM3'
   *  Inport: '<S95>/V_phaABC_FOC_in'
   *  Logic: '<S8>/Logical Operator1'
   *  Logic: '<S8>/Logical Operator2'
   *  LookupNDDirect: '<S94>/z_commutMap_M1'
   *  RelationalOperator: '<S8>/Relational Operator1'
   *  RelationalOperator: '<S8>/Relational Operator2'
   *  Selector: '<S11>/Selector'
   *
   * About '<S94>/z_commutMap_M1':
   *  2-dimensional Direct Look-Up returning a Column
   */
  if (rtb_LogicalOperator && (rtP->z_ctrlTypSel == 2)) {
    /* Outputs for IfAction SubSystem: '<S8>/FOC_Method' incorporates:
     *  ActionPort: '<S95>/Action Port'
     */
    DataTypeConversion2 = rtDW->Gain4_e[0];
    rtb_Saturation1 = rtDW->Gain4_e[1];
    rtb_Merge1 = rtDW->Gain4_e[2];

    /* End of Outputs for SubSystem: '<S8>/FOC_Method' */
  } else if (rtb_LogicalOperator && (rtP->z_ctrlTypSel == 1)) {
    /* Outputs for IfAction SubSystem: '<S8>/SIN_Method' incorporates:
     *  ActionPort: '<S96>/Action Port'
     */
    /* Switch: '<S97>/Switch_PhaAdv' incorporates:
     *  Constant: '<S97>/b_fieldWeakEna'
     *  Product: '<S98>/Divide2'
     *  Product: '<S98>/Divide3'
     *  Sum: '<S98>/Sum3'
     */
    if (rtP->b_fieldWeakEna) {
      /* Sum: '<S97>/Sum3' incorporates:
       *  Product: '<S97>/Product2'
       */
      DataTypeConversion2 = (int16_T)((int16_T)((int16_T)(rtDW->Divide3 *
        rtDW->Switch2_e) << 2) + rtb_Merge_m);
      DataTypeConversion2 -= (int16_T)((int16_T)((int16_T)div_nde_s32_floor
        (DataTypeConversion2, 23040) * 360) << 6);
    } else {
      DataTypeConversion2 = rtb_Merge_m;
    }

    /* End of Switch: '<S97>/Switch_PhaAdv' */

    /* PreLookup: '<S96>/a_elecAngle_XA' */
    Sum = plook_u8s16_evencka(DataTypeConversion2, 0, 128U, 180U);

    /* Product: '<S96>/Divide2' incorporates:
     *  Interpolation_n-D: '<S96>/r_sin3PhaA_M1'
     *  Interpolation_n-D: '<S96>/r_sin3PhaB_M1'
     *  Interpolation_n-D: '<S96>/r_sin3PhaC_M1'
     */
    DataTypeConversion2 = (int16_T)((rtb_Saturation *
      rtConstP.r_sin3PhaA_M1_Table[Sum]) >> 14);
    rtb_Saturation1 = (int16_T)((rtb_Saturation *
      rtConstP.r_sin3PhaB_M1_Table[Sum]) >> 14);
    rtb_Merge1 = (int16_T)((rtb_Saturation * rtConstP.r_sin3PhaC_M1_Table[Sum]) >>
      14);

    /* End of Outputs for SubSystem: '<S8>/SIN_Method' */
  } else {
    /* Outputs for IfAction SubSystem: '<S8>/COM_Method' incorporates:
     *  ActionPort: '<S94>/Action Port'
     */
    if (rtConstP.vec_hallToPos_Value[Sum] > 5) {
      /* LookupNDDirect: '<S94>/z_commutMap_M1'
       *
       * About '<S94>/z_commutMap_M1':
       *  2-dimensional Direct Look-Up returning a Column
       */
      rtb_Sum2_h = 5;
    } else if (rtConstP.vec_hallToPos_Value[Sum] < 0) {
      /* LookupNDDirect: '<S94>/z_commutMap_M1'
       *
       * About '<S94>/z_commutMap_M1':
       *  2-dimensional Direct Look-Up returning a Column
       */
      rtb_Sum2_h = 0;
    } else {
      /* LookupNDDirect: '<S94>/z_commutMap_M1' incorporates:
       *  Constant: '<S11>/vec_hallToPos'
       *  Selector: '<S11>/Selector'
       *
       * About '<S94>/z_commutMap_M1':
       *  2-dimensional Direct Look-Up returning a Column
       */
      rtb_Sum2_h = rtConstP.vec_hallToPos_Value[Sum];
    }

    /* LookupNDDirect: '<S94>/z_commutMap_M1' incorporates:
     *  Constant: '<S11>/vec_hallToPos'
     *  Selector: '<S11>/Selector'
     *
     * About '<S94>/z_commutMap_M1':
     *  2-dimensional Direct Look-Up returning a Column
     */
    rtb_Sum1_jt = rtb_Sum2_h * 3;

    /* Product: '<S94>/Divide2' incorporates:
     *  LookupNDDirect: '<S94>/z_commutMap_M1'
     *
     * About '<S94>/z_commutMap_M1':
     *  2-dimensional Direct Look-Up returning a Column
     */
    DataTypeConversion2 = (int16_T)(rtb_Saturation *
      rtConstP.z_commutMap_M1_table[rtb_Sum1_jt]);
    rtb_Saturation1 = (int16_T)(rtConstP.z_commutMap_M1_table[1 + rtb_Sum1_jt] *
      rtb_Saturation);
    rtb_Merge1 = (int16_T)(rtConstP.z_commutMap_M1_table[2 + rtb_Sum1_jt] *
      rtb_Saturation);

    /* End of Outputs for SubSystem: '<S8>/COM_Method' */
  }

  /* End of If: '<S8>/If' */

  /* Outport: '<Root>/DC_phaA' incorporates:
   *  DataTypeConversion: '<S8>/Data Type Conversion6'
   */
  rtY->DC_phaA = (int16_T)(DataTypeConversion2 >> 4);

  /* Outport: '<Root>/DC_phaB' incorporates:
   *  DataTypeConversion: '<S8>/Data Type Conversion6'
   */
  rtY->DC_phaB = (int16_T)(rtb_Saturation1 >> 4);

  /* Update for UnitDelay: '<S10>/UnitDelay3' incorporates:
   *  Inport: '<Root>/b_hallA '
   */
  rtDW->UnitDelay3_DSTATE_fy = rtU->b_hallA;

  /* Update for UnitDelay: '<S10>/UnitDelay1' incorporates:
   *  Inport: '<Root>/b_hallB'
   */
  rtDW->UnitDelay1_DSTATE = rtU->b_hallB;

  /* Update for UnitDelay: '<S10>/UnitDelay2' incorporates:
   *  Inport: '<Root>/b_hallC'
   */
  rtDW->UnitDelay2_DSTATE_f = rtU->b_hallC;

  /* Update for UnitDelay: '<S13>/UnitDelay3' */
  rtDW->UnitDelay3_DSTATE = rtb_Switch1_l;

  /* Update for UnitDelay: '<S13>/UnitDelay4' */
  rtDW->UnitDelay4_DSTATE_e = Abs5;

  /* Update for UnitDelay: '<S2>/UnitDelay2' incorporates:
   *  UnitDelay: '<S2>/UnitDelay6'
   */
  rtDW->UnitDelay2_DSTATE_c = rtDW->UnitDelay6_DSTATE;

  /* Update for UnitDelay: '<S2>/UnitDelay5' */
  rtDW->UnitDelay5_DSTATE_m = rtb_RelationalOperator4_d;

  /* Update for UnitDelay: '<S2>/UnitDelay6' */
  rtDW->UnitDelay6_DSTATE = rtb_UnitDelay5_e;

  /* Update for UnitDelay: '<S8>/UnitDelay4' */
  rtDW->UnitDelay4_DSTATE_eu = rtb_Saturation;

  /* Outport: '<Root>/DC_phaC' incorporates:
   *  DataTypeConversion: '<S8>/Data Type Conversion6'
   */
  rtY->DC_phaC = (int16_T)(rtb_Merge1 >> 4);

  /* Outport: '<Root>/n_mot' incorporates:
   *  DataTypeConversion: '<S1>/Data Type Conversion1'
   */
  rtY->n_mot = (int16_T)(Switch2 >> 4);

  /* Outport: '<Root>/a_elecAngle' incorporates:
   *  DataTypeConversion: '<S1>/Data Type Conversion3'
   */
  rtY->a_elecAngle = (int16_T)(rtb_Merge_m >> 6);

  /* End of Outputs for SubSystem: '<Root>/BLDC_controller' */

  /* Outport: '<Root>/iq' */
  rtY->iq = rtDW->DataTypeConversion[0];

  /* Outport: '<Root>/id' */
  rtY->id = rtDW->DataTypeConversion[1];
}

/* Model initialize function */
void BLDC_controller_initialize(RT_MODEL *const rtM)
{
  P *rtP = ((P *) rtM->defaultParam);
  DW *rtDW = ((DW *) rtM->dwork);

  /* Start for Atomic SubSystem: '<Root>/BLDC_controller' */
  /* Start for If: '<S7>/If1' */
  rtDW->If1_ActiveSubsystem = -1;

  /* Start for IfAction SubSystem: '<S7>/Clarke_Park_Transform_Forward' */
  /* Start for If: '<S45>/If2' */
  rtDW->If2_ActiveSubsystem_a = -1;

  /* End of Start for SubSystem: '<S7>/Clarke_Park_Transform_Forward' */

  /* Start for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S1>/F03_Control_Mode_Manager'
   */
  /* Start for If: '<S33>/If2' */
  rtDW->If2_ActiveSubsystem_f = -1;

  /* Start for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S7>/Motor_Limitations'
   */
  /* Start for If: '<S48>/If1' */
  rtDW->If1_ActiveSubsystem_o = -1;

  /* Start for IfAction SubSystem: '<S48>/Motor_Limitations_Enabled' */
  /* Start for SwitchCase: '<S80>/Switch Case' */
  rtDW->SwitchCase_ActiveSubsystem_d = -1;

  /* End of Start for SubSystem: '<S48>/Motor_Limitations_Enabled' */

  /* Start for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S7>/FOC'
   */
  /* Start for If: '<S47>/If1' */
  rtDW->If1_ActiveSubsystem_j = -1;

  /* Start for IfAction SubSystem: '<S47>/FOC_Enabled' */
  /* Start for SwitchCase: '<S59>/Switch Case' */
  rtDW->SwitchCase_ActiveSubsystem = -1;

  /* Start for If: '<S59>/If1' */
  rtDW->If1_ActiveSubsystem_a = -1;

  /* End of Start for SubSystem: '<S47>/FOC_Enabled' */

  /* Start for If: '<S7>/If2' */
  rtDW->If2_ActiveSubsystem = -1;

  /* End of Start for SubSystem: '<Root>/BLDC_controller' */

  /* SystemInitialize for Atomic SubSystem: '<Root>/BLDC_controller' */
  /* InitializeConditions for UnitDelay: '<S13>/UnitDelay3' */
  rtDW->UnitDelay3_DSTATE = rtP->z_maxCntRst;

  /* InitializeConditions for UnitDelay: '<S2>/UnitDelay2' */
  rtDW->UnitDelay2_DSTATE_c = true;

  /* SystemInitialize for IfAction SubSystem: '<S13>/Raw_Motor_Speed_Estimation' */
  /* SystemInitialize for Outport: '<S17>/z_counter' */
  rtDW->z_counterRawPrev = rtP->z_maxCntRst;

  /* End of SystemInitialize for SubSystem: '<S13>/Raw_Motor_Speed_Estimation' */

  /* SystemInitialize for Atomic SubSystem: '<S13>/Counter' */
  Counter_Init(&rtDW->Counter_e, rtP->z_maxCntRst);

  /* End of SystemInitialize for SubSystem: '<S13>/Counter' */

  /* SystemInitialize for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S1>/F02_Diagnostics'
   */

  /* SystemInitialize for IfAction SubSystem: '<S4>/Diagnostics_Enabled' */

  /* SystemInitialize for Atomic SubSystem: '<S20>/Debounce_Filter' */
  Debounce_Filter_Init(&rtDW->Debounce_Filter_k);

  /* End of SystemInitialize for SubSystem: '<S20>/Debounce_Filter' */

  /* End of SystemInitialize for SubSystem: '<S4>/Diagnostics_Enabled' */

  /* SystemInitialize for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S1>/F03_Control_Mode_Manager'
   */
  /* SystemInitialize for IfAction SubSystem: '<S33>/Open_Mode' */
  /* SystemInitialize for Atomic SubSystem: '<S37>/rising_edge_init' */
  /* InitializeConditions for UnitDelay: '<S39>/UnitDelay' */
  rtDW->UnitDelay_DSTATE_b = true;

  /* End of SystemInitialize for SubSystem: '<S37>/rising_edge_init' */
  /* End of SystemInitialize for SubSystem: '<S33>/Open_Mode' */

  /* SystemInitialize for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S7>/Motor_Limitations'
   */
  /* SystemInitialize for IfAction SubSystem: '<S48>/Motor_Limitations_Enabled' */

  /* SystemInitialize for IfAction SubSystem: '<S80>/Voltage_Mode_Protection' */

  /* SystemInitialize for Atomic SubSystem: '<S83>/I_backCalc_fixdt' */
  I_backCalc_fixdt_Init(&rtDW->I_backCalc_fixdt_i, 65536000);

  /* End of SystemInitialize for SubSystem: '<S83>/I_backCalc_fixdt' */

  /* SystemInitialize for Atomic SubSystem: '<S83>/I_backCalc_fixdt1' */
  I_backCalc_fixdt_Init(&rtDW->I_backCalc_fixdt1, 65536000);

  /* End of SystemInitialize for SubSystem: '<S83>/I_backCalc_fixdt1' */

  /* End of SystemInitialize for SubSystem: '<S80>/Voltage_Mode_Protection' */

  /* SystemInitialize for IfAction SubSystem: '<S80>/Torque_Mode_Protection' */

  /* SystemInitialize for Atomic SubSystem: '<S82>/I_backCalc_fixdt' */
  I_backCalc_fixdt_Init(&rtDW->I_backCalc_fixdt_j, 58982400);

  /* End of SystemInitialize for SubSystem: '<S82>/I_backCalc_fixdt' */

  /* End of SystemInitialize for SubSystem: '<S80>/Torque_Mode_Protection' */

  /* SystemInitialize for Outport: '<S80>/Vd_max' */
  rtDW->Vd_max1 = 14400;

  /* SystemInitialize for Outport: '<S80>/Vd_min' */
  rtDW->Gain3 = -14400;

  /* SystemInitialize for Outport: '<S80>/Vq_max' */
  rtDW->Vq_max_M1 = 14400;

  /* SystemInitialize for Outport: '<S80>/Vq_min' */
  rtDW->Gain5 = -14400;

  /* SystemInitialize for Outport: '<S80>/id_max' */
  rtDW->i_max = 12000;

  /* SystemInitialize for Outport: '<S80>/id_min' */
  rtDW->Gain4 = -12000;

  /* SystemInitialize for Outport: '<S80>/iq_max' */
  rtDW->Divide1_n = 12000;

  /* SystemInitialize for Outport: '<S80>/iq_min' */
  rtDW->Gain1 = -12000;

  /* End of SystemInitialize for SubSystem: '<S48>/Motor_Limitations_Enabled' */

  /* SystemInitialize for Chart: '<S1>/Task_Scheduler' incorporates:
   *  SubSystem: '<S7>/FOC'
   */

  /* SystemInitialize for IfAction SubSystem: '<S47>/FOC_Enabled' */

  /* SystemInitialize for IfAction SubSystem: '<S59>/Speed_Mode' */

  /* SystemInitialize for Atomic SubSystem: '<S61>/PI_clamp_fixdt' */
  PI_clamp_fixdt_d_Init(&rtDW->PI_clamp_fixdt_l4);

  /* End of SystemInitialize for SubSystem: '<S61>/PI_clamp_fixdt' */

  /* End of SystemInitialize for SubSystem: '<S59>/Speed_Mode' */

  /* SystemInitialize for IfAction SubSystem: '<S59>/Torque_Mode' */

  /* SystemInitialize for Atomic SubSystem: '<S62>/PI_clamp_fixdt' */
  PI_clamp_fixdt_f_Init(&rtDW->PI_clamp_fixdt_kh);

  /* End of SystemInitialize for SubSystem: '<S62>/PI_clamp_fixdt' */

  /* End of SystemInitialize for SubSystem: '<S59>/Torque_Mode' */

  /* SystemInitialize for IfAction SubSystem: '<S59>/Vd_Calculation' */

  /* SystemInitialize for Atomic SubSystem: '<S63>/PI_clamp_fixdt' */
  PI_clamp_fixdt_Init(&rtDW->PI_clamp_fixdt_i);

  /* End of SystemInitialize for SubSystem: '<S63>/PI_clamp_fixdt' */

  /* End of SystemInitialize for SubSystem: '<S59>/Vd_Calculation' */

  /* End of SystemInitialize for SubSystem: '<S47>/FOC_Enabled' */

  /* End of SystemInitialize for SubSystem: '<Root>/BLDC_controller' */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
