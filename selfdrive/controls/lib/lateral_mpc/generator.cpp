#include <acado_code_generation.hpp>

#define PI 3.1415926536
#define deg2rad(d) (d/180.0*PI)

const int controlHorizon = 50;

using namespace std;
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState xx; // x position
  DifferentialState yy; // y position
  DifferentialState psi; // vehicle heading
  DifferentialState dpsi;

  OnlineData v_ref; // m/s
  OnlineData d_poly_r0, d_poly_r1, d_poly_r2, d_poly_r3;
  OnlineData dpsi_poly_r0, dpsi_poly_r1, dpsi_poly_r2, dpsi_poly_r3;

  Control ddpsi;

  // Equations of motion
  f << dot(xx) == v_ref * cos(psi);
  f << dot(yy) == v_ref * sin(psi);
  f << dot(psi) == dpsi;
  f << dot(dpsi) == ddpsi;

  auto poly_d = d_poly_r0*(xx*xx*xx) + d_poly_r1*(xx*xx) + d_poly_r2*xx + d_poly_r3;
  auto poly_dpsi = dpsi_poly_r0*(xx*xx*xx) + dpsi_poly_r1*(xx*xx) + dpsi_poly_r2*xx + dpsi_poly_r3;
  auto poly_ddpsi = 3*dpsi_poly_r0*(xx*xx) + 2*dpsi_poly_r1*(xx) + dpsi_poly_r2;

  // Running cost
  Function h;

  // Distance errors
  h << (poly_d - yy);

  // Yaw rate trajectory error
  h << (poly_dpsi - dpsi);
  
  // Angular rate error
  h << (poly_ddpsi - ddpsi);

  BMatrix Q(3,3); Q.setAll(true);
  // Q(0,0) = 1.0;
  // Q(1,1) = 1.0;
  // Q(2,2) = 1.0;

  // Terminal cost
  Function hN;

  // Distance errors
  hN << (poly_d - yy);

  BMatrix QN(1,1); QN.setAll(true);
  // QN(0,0) = 1.0;

  // Non uniform time grid
  // First 5 timesteps are 0.05, after that it's 0.15
  DMatrix numSteps(20, 1);
  for (int i = 0; i < 5; i++){
    numSteps(i) = 1;
  }
  for (int i = 5; i < 20; i++){
    numSteps(i) = 3;
  }

  // Setup Optimal Control Problem
  const double tStart = 0.0;
  const double tEnd   = 2.5;

  OCP ocp( tStart, tEnd, numSteps);
  ocp.subjectTo(f);

  ocp.minimizeLSQ(Q, h);
  ocp.minimizeLSQEndTerm(QN, hN);

  // car can't go backward to avoid "circles"
  ocp.subjectTo( deg2rad(-90) <= psi <= deg2rad(90));
  // more than absolute max steer angle
  ocp.subjectTo( deg2rad(-50) <= ddpsi <= deg2rad(50));
  ocp.setNOD(9);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, 1 * controlHorizon);
  mpc.set( MAX_NUM_QP_ITERATIONS, 500);
  mpc.set( CG_USE_VARIABLE_WEIGHTING_MATRIX, YES);

  mpc.set( SPARSE_QP_SOLUTION, CONDENSING );
  mpc.set( QP_SOLVER, QP_QPOASES );
  mpc.set( HOTSTART_QP, YES );
  mpc.set( GENERATE_TEST_FILE, NO);
  mpc.set( GENERATE_MAKE_FILE, NO );
  mpc.set( GENERATE_MATLAB_INTERFACE, NO );
  mpc.set( GENERATE_SIMULINK_INTERFACE, NO );

  if (mpc.exportCode("selfdrive/controls/lib/lateral_mpc/lib_mpc_export") != SUCCESSFUL_RETURN)
    exit( EXIT_FAILURE );

  mpc.printDimensionsQP( );

  return EXIT_SUCCESS;
}
