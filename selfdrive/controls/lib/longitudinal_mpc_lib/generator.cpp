#include <acado_code_generation.hpp>

const int controlHorizon = 50;

using namespace std;


int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState x_ego, v_ego, a_ego, t;
  OnlineData min_a, max_a;

  Control j_ego;

  // Equations of motion
  f << dot(x_ego) == v_ego;
  f << dot(v_ego) == a_ego;
  f << dot(a_ego) == j_ego;
  f << dot(t) == 1;

  // Running cost
  Function h;
  h << x_ego;
  h << v_ego;
  h << a_ego;
  h << j_ego * (0.1 * v_ego + 1.0);

  // Weights are defined in mpc.
  BMatrix Q(4,4); Q.setAll(true);

  // Terminal cost
  Function hN;
  hN << x_ego;
  hN << v_ego;
  hN << a_ego;

  // Weights are defined in mpc.
  BMatrix QN(3,3); QN.setAll(true);

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
  ocp.subjectTo( 0.0 <= a_ego - min_a);
  ocp.subjectTo( a_ego - max_a <= 0.0);
  ocp.setNOD(2);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, controlHorizon);
  mpc.set( MAX_NUM_QP_ITERATIONS, 500);
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
