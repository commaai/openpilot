#include "acado_common.h"
#include "acado_auxiliary_functions.h"

#include <stdio.h>

#define NX          ACADO_NX  /* Number of differential state variables.  */
#define NXA         ACADO_NXA /* Number of algebraic variables. */
#define NU          ACADO_NU  /* Number of control inputs. */
#define NOD         ACADO_NOD  /* Number of online data values. */

#define NY          ACADO_NY  /* Number of measurements/references on nodes 0..N - 1. */
#define NYN         ACADO_NYN /* Number of measurements/references on node N. */

#define N           ACADO_N   /* Number of intervals in the horizon. */

ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;

typedef struct {
  double x, y, psi, delta, t;
} state_t;


typedef struct {
  double x[N];
  double y[N];
  double psi[N];
	double delta[N];
} log_t;

void init(double steerRateCost){
  acado_initializeSolver();
  int    i;

  /* Initialize the states and controls. */
  for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[ i ] = 0.0;
  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.1;

  /* Initialize the measurements/reference. */
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;

  /* MPC: initialize the current state feedback. */
  for (i = 0; i < NX; ++i) acadoVariables.x0[ i ] = 0.0;

  for (i = 0; i < N; i++) {
    int f = 1;
    if (i > 4){
      f = 3;
    }
    acadoVariables.W[25 * i + 0] = 1.0 * f;
    acadoVariables.W[25 * i + 6] = 1.0 * f;
    acadoVariables.W[25 * i + 12] = 1.0 * f;
    acadoVariables.W[25 * i + 18] = 1.0 * f;
    acadoVariables.W[25 * i + 24] = steerRateCost * f;
  }
  acadoVariables.WN[0] = 1.0;
  acadoVariables.WN[5] = 1.0;
  acadoVariables.WN[10] = 1.0;
  acadoVariables.WN[15] = 1.0;
}

int run_mpc(state_t * x0, log_t * solution,
             double l_poly[4], double r_poly[4], double p_poly[4],
             double l_prob, double r_prob, double p_prob, double curvature_factor, double v_ref, double lane_width){

  int    i;

  for (i = 0; i <= NOD * N; i+= NOD){
    acadoVariables.od[i] = curvature_factor;
    acadoVariables.od[i+1] = v_ref;

    acadoVariables.od[i+2] = l_poly[0];
    acadoVariables.od[i+3] = l_poly[1];
    acadoVariables.od[i+4] = l_poly[2];
    acadoVariables.od[i+5] = l_poly[3];

    acadoVariables.od[i+6] = r_poly[0];
    acadoVariables.od[i+7] = r_poly[1];
    acadoVariables.od[i+8] = r_poly[2];
    acadoVariables.od[i+9] = r_poly[3];

    acadoVariables.od[i+10] = p_poly[0];
    acadoVariables.od[i+11] = p_poly[1];
    acadoVariables.od[i+12] = p_poly[2];
    acadoVariables.od[i+13] = p_poly[3];


    acadoVariables.od[i+14] = l_prob;
    acadoVariables.od[i+15] = r_prob;
    acadoVariables.od[i+16] = p_prob;
    acadoVariables.od[i+17] = lane_width;

  }

  acadoVariables.x0[0] = x0->x;
  acadoVariables.x0[1] = x0->y;
  acadoVariables.x0[2] = x0->psi;
  acadoVariables.x0[3] = x0->delta;


  acado_preparationStep();
  acado_feedbackStep();
  /* printf("lat its: %d\n", acado_getNWSR()); */

	for (i = 0; i <= N; i++){
		solution->x[i] = acadoVariables.x[i*NX];
		solution->y[i] = acadoVariables.x[i*NX+1];
		solution->psi[i] = acadoVariables.x[i*NX+2];
		solution->delta[i] = acadoVariables.x[i*NX+3];
	}

  acado_shiftStates(2, 0, 0);
  acado_shiftControls( 0 );


  return acado_getNWSR();
}
