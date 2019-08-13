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
  double j_ego[N+1];
	double x_l[N+1];
	double v_l[N+1];
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
    acadoVariables.W[16 * i + 0] = ttcCost * f; // exponential cost for time-to-collision (ttc)
    acadoVariables.W[16 * i + 5] = distanceCost * f; // desired distance
    acadoVariables.W[16 * i + 10] = accelerationCost * f; // acceleration
    acadoVariables.W[16 * i + 15] = jerkCost * f; // jerk
  }
  acadoVariables.WN[0] = ttcCost * STEP_MULTIPLIER; // exponential cost for danger zone
  acadoVariables.WN[4] = distanceCost * STEP_MULTIPLIER; // desired distance
  acadoVariables.WN[8] = accelerationCost * STEP_MULTIPLIER; // acceleration

}

void init_with_simulation(double v_ego, double x_l_0, double v_l_0, double a_l_0, double l){
  int i;
  double x_ego = 0.0;
  double a_ego = 0.0;

  double x_l = x_l_0;
  double v_l = v_l_0;
  double a_l = a_l_0;


  if (v_ego > v_l){
    a_ego = -(v_ego - v_l) * (v_ego - v_l) / (2.0 * x_l + 0.01) + a_l;
  }
  double dt = 0.2;
  double t = 0.;

  for (i = 0; i < N + 1; ++i){
    if (i > 4){
      dt = 0.6;
    }

    // Directly calculate a_l to prevent instabilites due to discretization
    a_l = a_l_0 * exp(-l * t * t / 2);

    /* printf("x_e: %.2f\t v_e: %.2f\t a_e: %.2f\t", x_ego, v_ego, a_ego); */
    /* printf("x_l: %.2f\t v_l: %.2f\t a_l: %.2f\n", x_l, v_l, a_l); */

    acadoVariables.x[i*NX] = x_ego;
    acadoVariables.x[i*NX+1] = v_ego;
    acadoVariables.x[i*NX+2] = a_ego;

    acadoVariables.x[i*NX+3] = x_l;
    acadoVariables.x[i*NX+4] = v_l;
    acadoVariables.x[i*NX+5] = t;

    x_ego += v_ego * dt;
    v_ego += a_ego * dt;

    x_l += v_l * dt;
    v_l += a_l * dt;
    t += dt;

    if (v_ego <= 0.0) {
      v_ego = 0.0;
      a_ego = 0.0;
    }
  }
  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.0;
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;
}

int run_mpc(state_t * x0, log_t * solution, double l, double a_l_0, double TR){
  int i;

  for (i = 0; i <= NOD * N; i+= NOD){
    acadoVariables.od[i] = l;
    acadoVariables.od[i+1] = a_l_0;
  }

  acadoVariables.x[0] = acadoVariables.x0[0] = x0->x_ego;
  acadoVariables.x[1] = acadoVariables.x0[1] = x0->v_ego;
  acadoVariables.x[2] = acadoVariables.x0[2] = x0->a_ego;
  acadoVariables.x[3] = acadoVariables.x0[3] = x0->x_l;
  acadoVariables.x[4] = acadoVariables.x0[4] = x0->v_l;
  acadoVariables.x[5] = acadoVariables.x0[5] = 0.0;

  acado_preparationStep(TR);
  acado_feedbackStep();

	for (i = 0; i <= N; i++){
    solution->x_ego[i] = acadoVariables.x[i*NX];
		solution->v_ego[i] = acadoVariables.x[i*NX+1];
    solution->a_ego[i] = acadoVariables.x[i*NX+2];
		solution->x_l[i] = acadoVariables.x[i*NX+3];
		solution->v_l[i] = acadoVariables.x[i*NX+4];
		solution->t[i] = acadoVariables.x[i*NX+5];

    solution->j_ego[i] = acadoVariables.u[i];
	}
  solution->cost = acado_getObjective(TR);

  // Dont shift states here. Current solution is closer to next timestep than if
  // we shift by 0.2 seconds.

  return acado_getNWSR();
}
