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
  double x_ego, v_ego, a_ego, x_l, v_l, a_l;
} state_t;


typedef struct {
  double x_ego[N+1];
  double v_ego[N+1];
  double a_ego[N+1];
  double j_ego[N];
  double x_l[N+1];
  double v_l[N+1];
  double a_l[N+1];
  double t[N+1];
  double cost;
} log_t;

void init(double ttcCost, double distanceCost, double accelerationCost, double jerkCost){
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
    if (i > 4){
      f = STEP_MULTIPLIER;
    }
    // Setup diagonal entries
    acadoVariables.W[NY*NY*i + (NY+1)*0] = ttcCost * f; // exponential cost for time-to-collision (ttc)
    acadoVariables.W[NY*NY*i + (NY+1)*1] = distanceCost * f; // desired distance
    acadoVariables.W[NY*NY*i + (NY+1)*2] = accelerationCost * f; // acceleration
    acadoVariables.W[NY*NY*i + (NY+1)*3] = jerkCost * f; // jerk
  }
  acadoVariables.WN[(NYN+1)*0] = ttcCost * STEP_MULTIPLIER; // exponential cost for danger zone
  acadoVariables.WN[(NYN+1)*1] = distanceCost * STEP_MULTIPLIER; // desired distance
  acadoVariables.WN[(NYN+1)*2] = accelerationCost * STEP_MULTIPLIER; // acceleration

}

void init_with_simulation(double v_ego, double x_l_0, double v_l_0, double a_l_0, double l){
  int i;

  double x_l = x_l_0;
  double v_l = v_l_0;
  double a_l = a_l_0;

  double x_ego = 0.0;
  double a_ego = -(v_ego - v_l) * (v_ego - v_l) / (2.0 * x_l + 0.01) + a_l;

  if (a_ego > 0){
    a_ego = 0.0;
  }


  double dt = 0.2;
  double t = 0.;

  for (i = 0; i < N + 1; ++i){
    if (i > 4){
      dt = 0.6;
    }

    /* printf("%.2f\t%.2f\t%.2f\t%.2f\n", t, x_ego, v_ego, a_l); */
    acadoVariables.x[i*NX] = x_ego;
    acadoVariables.x[i*NX+1] = v_ego;
    acadoVariables.x[i*NX+2] = a_ego;

    v_ego += a_ego * dt;

    if (v_ego <= 0.0) {
      v_ego = 0.0;
      a_ego = 0.0;
    }

    x_ego += v_ego * dt;
    t += dt;
  }

  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;
}

int run_mpc(state_t * x0, log_t * solution, double l, double a_l_0){
  // Calculate lead vehicle predictions
  int i;
  double t = 0.;
  double dt = 0.2;
  double x_l = x0->x_l;
  double v_l = x0->v_l;
  double a_l = a_l_0;

  /* printf("t\tx_l\t_v_l\t_al\n"); */
  for (i = 0; i < N + 1; ++i){
    if (i > 4){
      dt = 0.6;
    }

    /* printf("%.2f\t%.2f\t%.2f\t%.2f\n", t, x_l, v_l, a_l); */

    acadoVariables.od[i*NOD] = x_l;
    acadoVariables.od[i*NOD+1] = v_l;

    solution->x_l[i] = x_l;
    solution->v_l[i] = v_l;
    solution->a_l[i] = a_l;
    solution->t[i] = t;

    a_l = a_l_0 * exp(-l * t * t / 2);
    x_l += v_l * dt;
    v_l += a_l * dt;
    if (v_l < 0.0){
      a_l = 0.0;
      v_l = 0.0;
    }

    t += dt;
  }

  acadoVariables.x[0] = acadoVariables.x0[0] = x0->x_ego;
  acadoVariables.x[1] = acadoVariables.x0[1] = x0->v_ego;
  acadoVariables.x[2] = acadoVariables.x0[2] = x0->a_ego;

  acado_preparationStep();
  acado_feedbackStep();

  for (i = 0; i <= N; i++){
    solution->x_ego[i] = acadoVariables.x[i*NX];
    solution->v_ego[i] = acadoVariables.x[i*NX+1];
    solution->a_ego[i] = acadoVariables.x[i*NX+2];

    if (i < N){
      solution->j_ego[i] = acadoVariables.u[i];
    }
  }
  solution->cost = acado_getObjective();

  // Dont shift states here. Current solution is closer to next timestep than if
  // we shift by 0.2 seconds.

  return acado_getNWSR();
}
