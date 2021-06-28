#include <math.h>
#include <acado_code_generation.hpp>
#include "selfdrive/common/modeldata.h"

#define deg2rad(d) (d/180.0*M_PI)

using namespace std;

int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState xx; // x position
  DifferentialState yy; // y position
  DifferentialState psi; // vehicle heading
  DifferentialState curvature;

  OnlineData v_ego;
  OnlineData rotation_radius;

  Control curvature_rate;


  // Equations of motion
  f << dot(xx) == v_ego * cos(psi) - rotation_radius * sin(psi) * (v_ego * curvature);
  f << dot(yy) == v_ego * sin(psi) + rotation_radius * cos(psi) * (v_ego * curvature);
  f << dot(psi) == v_ego * curvature;
  f << dot(curvature) == curvature_rate;

  // Running cost
  Function h;

  // Distance errors
  h << yy;

  // Heading error
  h << (v_ego + 5.0 ) * psi;

  // Angular rate error
  h << (v_ego + 5.0) * 4 * curvature_rate;

  BMatrix Q(3,3); Q.setAll(true);
  // Q(0,0) = 1.0;
  // Q(1,1) = 1.0;
  // Q(2,2) = 1.0;
  // Q(3,3) = 1.0;
  // Q(4,4) = 2.0;

  // Terminal cost
  Function hN;

  // Distance errors
  hN << yy;

  // Heading errors
  hN << (2.0 * v_ego + 5.0 ) * psi;

  BMatrix QN(2,2); QN.setAll(true);
  // QN(0,0) = 1.0;
  // QN(1,1) = 1.0;
  // QN(2,2) = 1.0;
  // QN(3,3) = 1.0;

  double T_IDXS_ARR[LAT_MPC_N + 1];
  memcpy(T_IDXS_ARR, T_IDXS, (LAT_MPC_N + 1) * sizeof(double));
  Grid times(LAT_MPC_N + 1, T_IDXS_ARR);
  OCP ocp(times);
  ocp.subjectTo(f);

  ocp.minimizeLSQ(Q, h);
  ocp.minimizeLSQEndTerm(QN, hN);

  // car can't go backward to avoid "circles"
  ocp.subjectTo( deg2rad(-90) <= psi <= deg2rad(90));
  // more than absolute max steer angle
  ocp.subjectTo( deg2rad(-50) <= curvature <= deg2rad(50));
  ocp.setNOD(2);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, 2500);
  mpc.set( MAX_NUM_QP_ITERATIONS, 1000);
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
