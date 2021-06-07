#include "acado_common.h"
#include "acado_auxiliary_functions.h"
#include "common/modeldata.h"
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
  double x, y, psi, tire_angle, tire_angle_rate;
} state_t;

typedef struct {
  double x[N+1];
  double y[N+1];
  double psi[N+1];
  double curvature[N+1];
  double curvature_rate[N];
  double cost;
} log_t;

void set_weights(double pathCost, double headingCost, double steerRateCost){
  int    i;
  const int STEP_MULTIPLIER = 3.0;

  for (i = 0; i < N; i++) {
    double f = 20 * (T_IDXS[i+1] - T_IDXS[i]);
    // Setup diagonal entries
    acadoVariables.W[NY*NY*i + (NY+1)*0] = pathCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*1] = headingCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*2] = steerRateCost * f;
  }
  acadoVariables.WN[(NYN+1)*0] = pathCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*1] = headingCost * STEP_MULTIPLIER;
}

void init(){
  acado_initializeSolver();
  int    i;

  /* Initialize the states and controls. */
  for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[ i ] = 0.0;
  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;

  /* Initialize the measurements/reference. */
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;

  /* MPC: initialize the current state feedback. */
  for (i = 0; i < NX; ++i) acadoVariables.x0[ i ] = 0.0;
}

int run_mpc(state_t * x0, log_t * solution, double v_ego,
             double rotation_radius, double target_y[N+1], double target_psi[N+1]){

  int    i;

  for (i = 0; i <= NOD * N; i+= NOD){
    acadoVariables.od[i] = v_ego;
    acadoVariables.od[i+1] = rotation_radius;
  }
  for (i = 0; i < N; i+= 1){
    acadoVariables.y[NY*i + 0] = target_y[i];
    acadoVariables.y[NY*i + 1] = (v_ego + 5.0) * target_psi[i];
    acadoVariables.y[NY*i + 2] = 0.0;
  }
  acadoVariables.yN[0] = target_y[N];
  acadoVariables.yN[1] = (2.0 * v_ego + 5.0) * target_psi[N];

  acadoVariables.x0[0] = x0->x;
  acadoVariables.x0[1] = x0->y;
  acadoVariables.x0[2] = x0->psi;
  acadoVariables.x0[3] = x0->tire_angle;


  acado_preparationStep();
  acado_feedbackStep();

  /* printf("lat its: %d\n", acado_getNWSR());  // n iterations
  printf("Objective: %.6f\n", acado_getObjective());  // solution cost */

  for (i = 0; i <= N; i++){
    solution->x[i] = acadoVariables.x[i*NX];
    solution->y[i] = acadoVariables.x[i*NX+1];
    solution->psi[i] = acadoVariables.x[i*NX+2];
    solution->curvature[i] = acadoVariables.x[i*NX+3];
    if (i < N){
      solution->curvature_rate[i] = acadoVariables.u[i];
    }
  }
  solution->cost = acado_getObjective();

  // Dont shift states here. Current solution is closer to next timestep than if
  // we use the old solution as a starting point
  //acado_shiftStates(2, 0, 0);
  //acado_shiftControls( 0 );

  return acado_getNWSR();
}
