#include "acado_common.h"
#include "acado_auxiliary_functions.h"

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

void init(double xCost, double vCost, double aCost, double accelCost, double jerkCost) {
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
    int f = 1;
    if (i > 4) {
      f = STEP_MULTIPLIER;
    }
    // Setup diagonal entries
    acadoVariables.W[NY*NY*i + (NY+1)*0] = xCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*1] = vCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*2] = aCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*3] = accelCost * f;
    acadoVariables.W[NY*NY*i + (NY+1)*4] = jerkCost * f;
  }
  acadoVariables.WN[(NYN+1)*0] = xCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*1] = vCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*2] = aCost * STEP_MULTIPLIER;
  acadoVariables.WN[(NYN+1)*3] = accelCost * STEP_MULTIPLIER;

}

void init_with_simulation(double v_ego) {
  int i;

  double x_ego = 0.0;

  double dt = 0.2;
  double t = 0.0;

  for (i = 0; i < N + 1; ++i) {
    if (i > 4) {
      dt = 0.6;
    }

    acadoVariables.x[i*NX] = x_ego;
    acadoVariables.x[i*NX+1] = v_ego;
    acadoVariables.x[i*NX+2] = 0;
    acadoVariables.x[i*NX+3] = t;

    x_ego += v_ego * dt;
    t += dt;
  }

  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;
}

int run_mpc(state_t * x0, log_t * solution,
            double x_poly[4], double v_poly[4], double a_poly[4]) {
  int i;

  for (i = 0; i < N + 1; ++i) {
    acadoVariables.od[i*NOD+0] = x_poly[0];
    acadoVariables.od[i*NOD+1] = x_poly[1];
    acadoVariables.od[i*NOD+2] = x_poly[2];
    acadoVariables.od[i*NOD+3] = x_poly[3];

    acadoVariables.od[i*NOD+4] = v_poly[0];
    acadoVariables.od[i*NOD+5] = v_poly[1];
    acadoVariables.od[i*NOD+6] = v_poly[2];
    acadoVariables.od[i*NOD+7] = v_poly[3];

    acadoVariables.od[i*NOD+8] = a_poly[0];
    acadoVariables.od[i*NOD+9] = a_poly[1];
    acadoVariables.od[i*NOD+10] = a_poly[2];
    acadoVariables.od[i*NOD+11] = a_poly[3];
  }

  acadoVariables.x[0] = acadoVariables.x0[0] = x0->x_ego;
  acadoVariables.x[1] = acadoVariables.x0[1] = x0->v_ego;
  acadoVariables.x[2] = acadoVariables.x0[2] = x0->a_ego;
  acadoVariables.x[3] = acadoVariables.x0[3] = 0;

  acado_preparationStep();
  acado_feedbackStep();

  for (i = 0; i <= N; i++) {
    solution->x_ego[i] = acadoVariables.x[i*NX];
    solution->v_ego[i] = acadoVariables.x[i*NX+1];
    solution->a_ego[i] = acadoVariables.x[i*NX+2];
    solution->t[i] = acadoVariables.x[i*NX+3];

    if (i < N) {
      solution->j_ego[i] = acadoVariables.u[i];
    }
  }
  solution->cost = acado_getObjective();

  // Dont shift states here. Current solution is closer to next timestep than if
  // we shift by 0.1 seconds.
  return acado_getNWSR();
}
