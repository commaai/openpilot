#include <acado_code_generation.hpp>
#include "selfdrive/common/modeldata.h"

using namespace std;


int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState x_ego, v_ego, a_ego;
  DifferentialState dummy_0;
  OnlineData min_a, max_a;

  Control j_ego, accel_slack;

  // Equations of motion
  f << dot(x_ego) == v_ego;
  f << dot(v_ego) == a_ego;
  f << dot(a_ego) == j_ego;
  f << dot(dummy_0) == accel_slack;

  // Running cost
  Function h;
  h << x_ego;
  h << v_ego;
  h << a_ego;
  h << j_ego;
  h << accel_slack;

  // Weights are defined in mpc.
  BMatrix Q(5,5); Q.setAll(true);

  // Terminal cost
  Function hN;
  hN << x_ego;
  hN << v_ego;
  hN << a_ego;

  // Weights are defined in mpc.
  BMatrix QN(3,3); QN.setAll(true);

  double T_IDXS_ARR[LON_MPC_N + 1];
  memcpy(T_IDXS_ARR, T_IDXS, (LON_MPC_N + 1) * sizeof(double));
  Grid times(LON_MPC_N + 1, T_IDXS_ARR);
  OCP ocp(times);
  ocp.subjectTo(f);

  ocp.minimizeLSQ(Q, h);
  ocp.minimizeLSQEndTerm(QN, hN);

  ocp.subjectTo( 0.0 <= v_ego);
  ocp.subjectTo( 0.0 <= a_ego - min_a + accel_slack);
  ocp.subjectTo( a_ego - max_a + accel_slack <= 0.0);
  ocp.setNOD(2);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, 1000);
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
