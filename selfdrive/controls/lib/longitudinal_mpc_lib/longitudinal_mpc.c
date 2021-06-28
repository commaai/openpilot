#include "acado_common.h"
#include "acado_auxiliary_functions.h"
#include "common/modeldata.h"

#include <stdio.h>
#include <math.h>

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
  double x_ego, v_ego, a_ego;
} state_t;


typedef struct {
  double x_ego[N+1];
  double v_ego[N+1];
  double a_ego[N+1];
  double t[N+1];
  double j_ego[N];
  double cost;
} log_t;

void init(double xCost, double vCost, double aCost, double jerkCost){
  acado_initializeSolver();
  int    i;
  const int STEP_MULTIPLIER = 3;

  /* Initialize the states and controls. */
  for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[ i ] = 0.0;
  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;

  /* Initialize the measurements/reference. */
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;

  /* MPC: initialize the current state feedback. */
  for (i = 0; i < NX; ++i) acadoVariables.x0[ i ] = 0.0;
  
  // Set weights
  for (i = 0; i < N; i++) {
    double f = 20 * (T_IDXS[i+1] - T_IDXS[i]);
    // Setup diagonal entries
    acadoVariables.W[NY*NY*i + (NY+1)*0] = xCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*1] = vCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*2] = aCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*3] = jerkCost * f;
  }
  acadoVariables.WN[(NYN+1)*0] = xCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*1] = vCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*2] = aCost * STEP_MULTIPLIER;

}


int run_mpc(state_t * x0, log_t * solution,
            double target_x[N+1], double target_v[N+1], double target_a[N+1],
            double min_a, double max_a){
  int i;
  for (i = 0; i < N + 1; ++i){
    acadoVariables.od[i*NOD] = min_a;
    acadoVariables.od[i*NOD+1] = max_a;
  }
  for (i = 0; i < N; i+= 1){
    acadoVariables.y[NY*i + 0] = target_x[i];
    acadoVariables.y[NY*i + 1] = target_v[i];
    acadoVariables.y[NY*i + 2] = target_a[i];
    acadoVariables.y[NY*i + 3] = 0.0;
  }
  acadoVariables.yN[0] = target_x[N];
  acadoVariables.yN[1] = target_v[N];
  acadoVariables.yN[2] = target_a[N];

  acadoVariables.x0[0] = x0->x_ego;
  acadoVariables.x0[1] = x0->v_ego;
  acadoVariables.x0[2] = x0->a_ego;

  acado_preparationStep();
  acado_feedbackStep();

  for (i = 0; i <= N; i++) {
    solution->x_ego[i] = acadoVariables.x[i*NX];
    solution->v_ego[i] = acadoVariables.x[i*NX+1];
    solution->a_ego[i] = acadoVariables.x[i*NX+2];

    if (i < N) {
      solution->j_ego[i] = acadoVariables.u[i];
    }
  }
  solution->cost = acado_getObjective();

  // Dont shift states here. Current solution is closer to next timestep than if
  // we shift by 0.1 seconds.
  return acado_getNWSR();
}
