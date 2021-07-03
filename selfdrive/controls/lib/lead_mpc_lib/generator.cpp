#include <acado_code_generation.hpp>

using namespace std;

#define MAX_BRAKE 7.
#define T_REACT 1.8
#define MIN_ACCEL -3.0

int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState x_ego, v_ego, a_ego;
  DifferentialState dummy, dummy3;
  OnlineData x_l, v_l;

  Control j_ego, time_slack, accel_slack;

  auto ego_stop_time = v_ego/MAX_BRAKE;
  auto ego_stop_dist = v_ego*ego_stop_time - (MAX_BRAKE * ego_stop_time*ego_stop_time)/2;
  auto lead_stop_time = v_l/MAX_BRAKE;
  auto lead_stop_dist = v_l*lead_stop_time - (MAX_BRAKE * lead_stop_time*lead_stop_time)/2;
  auto reaction_distance = ((x_l - x_ego)- 4.0  + lead_stop_dist - ego_stop_dist);

  // Equations of motion
  f << dot(x_ego) == v_ego;
  f << dot(v_ego) == a_ego;
  f << dot(a_ego) == j_ego;
  f << dot(dummy) == time_slack;
  f << dot(dummy3) == accel_slack;

  // Running cost
  Function h;
  h << j_ego/(v_ego + 1.0);
  h << time_slack;
  h << accel_slack;
  h << accel_slack;
  h << reaction_distance - T_REACT*v_ego;

  // Weights are defined in mpc.
  BMatrix Q(5,5); Q.setAll(true);

  // Terminal cost
  Function hN;
  hN << reaction_distance - T_REACT*v_ego;

  // Weights are defined in mpc.
  BMatrix QN(1,1); QN.setAll(true);

  // Non uniform time grid
  // First 5 timesteps are 0.2, after that it's 0.6
  DMatrix numSteps(20, 1);
  for (int i = 0; i < 5; i++){
    numSteps(i) = 1;
  }
  for (int i = 5; i < 20; i++){
    numSteps(i) = 3;
  }

  // Setup Optimal Control Problem
  const double tStart = 0.0;
  const double tEnd   = 10.0;

  OCP ocp( tStart, tEnd, numSteps);
  ocp.subjectTo(f);

  ocp.minimizeLSQ(Q, h);
  ocp.minimizeLSQEndTerm(QN, hN);

  ocp.subjectTo( 0.0 <= v_ego);
  ocp.subjectTo( MIN_ACCEL <= a_ego + accel_slack);
  ocp.subjectTo( 0.0 <= reaction_distance - T_REACT*v_ego + time_slack);
  ocp.setNOD(2);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, 50);
  mpc.set( MAX_NUM_QP_ITERATIONS, 50);
  mpc.set( CG_USE_VARIABLE_WEIGHTING_MATRIX, YES);

  mpc.set( SPARSE_QP_SOLUTION, CONDENSING );
  mpc.set( QP_SOLVER, QP_QPOASES );
  mpc.set( HOTSTART_QP, YES );
  mpc.set( GENERATE_TEST_FILE, NO);
  mpc.set( GENERATE_MAKE_FILE, NO );
  mpc.set( GENERATE_MATLAB_INTERFACE, NO );
  mpc.set( GENERATE_SIMULINK_INTERFACE, NO );

  if (mpc.exportCode( "lib_mpc_export" ) != SUCCESSFUL_RETURN)
    exit( EXIT_FAILURE );

  mpc.printDimensionsQP( );

  return EXIT_SUCCESS;
}
