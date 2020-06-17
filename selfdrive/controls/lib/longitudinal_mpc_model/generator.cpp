#include <acado_code_generation.hpp>

const int controlHorizon = 50;

using namespace std;


int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState x_ego, v_ego, a_ego, t;

  OnlineData x_poly_r0, x_poly_r1, x_poly_r2, x_poly_r3;
  OnlineData v_poly_r0, v_poly_r1, v_poly_r2, v_poly_r3;
  OnlineData a_poly_r0, a_poly_r1, a_poly_r2, a_poly_r3;

  Control j_ego;

  // Equations of motion
  f << dot(x_ego) == v_ego;
  f << dot(v_ego) == a_ego;
  f << dot(a_ego) == j_ego;
  f << dot(t) == 1;

  auto poly_x = x_poly_r0*(t*t*t) + x_poly_r1*(t*t) + x_poly_r2*t + x_poly_r3;
  auto poly_v = v_poly_r0*(t*t*t) + v_poly_r1*(t*t) + v_poly_r2*t + v_poly_r3;
  auto poly_a = a_poly_r0*(t*t*t) + a_poly_r1*(t*t) + a_poly_r2*t + a_poly_r3;

  // Running cost
  Function h;
  h << x_ego - poly_x;
  h << v_ego - poly_v;
  h << a_ego - poly_a;
  h << a_ego * (0.1 * v_ego + 1.0);
  h << j_ego * (0.1 * v_ego + 1.0);

  // Weights are defined in mpc.
  BMatrix Q(5,5); Q.setAll(true);

  // Terminal cost
  Function hN;
  hN << x_ego - poly_x;
  hN << v_ego - poly_v;
  hN << a_ego - poly_a;
  hN << a_ego * (0.1 * v_ego + 1.0);

  // Weights are defined in mpc.
  BMatrix QN(4,4); QN.setAll(true);

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

  //ocp.subjectTo( 0.0 <= v_ego);
  ocp.setNOD(12);

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
