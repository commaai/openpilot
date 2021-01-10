#ifndef ACADO_TOOLKIT_TEMPLATES_HPP
#define ACADO_TOOLKIT_TEMPLATES_HPP

#define TEMPLATE_PATHS "/home/batman/openpilot/phonelibs/acado/acado/acado/code_generation/templates;/home/batman/openpilot/phonelibs/acado/include/acado/code_generation/templates"
 
#define INTEGRATOR_MEX_TEMPLATE "integrator_mex.c.in"
#define RHS_MEX_TEMPLATE "rhs_mex.c.in"

#define HESSIAN_REG_SOURCE "acado_hessian_regularization.c.in"

#define FORCES_TEMPLATE  "forces_interface.in"
#define FORCES_GENERATOR "acado_forces_generator.m.in"
#define FORCES_GENERATOR_PYTHON "acado_forces_generator.py.in"
#define QPOASES_HEADER   "qpoases_interface.hpp.in"
#define QPOASES_SOURCE   "qpoases_interface.cpp.in"
#define QPOASES3_HEADER  "qpoases3_interface.h.in"
#define QPOASES3_SOURCE  "qpoases3_interface.c.in"
#define QPDUNES_TEMPLATE "qpdunes_interface.in"
#define QPDUNES_SPLIT_TEMPLATE "qpdunes_split_interface.in"

#define AUXILIARY_FUNCTIONS_HEADER "acado_auxiliary_functions.h.in"
#define AUXILIARY_FUNCTIONS_SOURCE "acado_auxiliary_functions.c.in"
#define AUXILIARY_SIM_FUNCTIONS_HEADER "acado_auxiliary_sim_functions.h.in"
#define AUXILIARY_SIM_FUNCTIONS_SOURCE "acado_auxiliary_sim_functions.c.in"
#define SOLVER_MEX "acado_solver_mex.c.in"
#define EH_SOLVER_MEX "acado_EH_solver_mex.c.in"

#define MAKE_MEX_FORCES "make_acado_solver_forces.m.in"
#define MAKE_MEX_HPMPC "make_acado_solver_hpmpc.m.in"
#define MAKE_MEX_QPOASES "make_acado_solver_qpoases.m.in"
#define MAKE_MEX_EH_QPOASES "make_acado_EH_solver_qpoases.m.in"
#define MAKE_MEX_QPOASES3 "make_acado_solver_qpoases3.m.in"
#define MAKE_MEX_EH_QPOASES3 "make_acado_EH_solver_qpoases3.m.in"
#define MAKE_MEX_QPDUNES "make_acado_solver_qpdunes.m.in"
#define MAKE_MEX_EH_QPDUNES "make_acado_EH_solver_qpdunes.m.in"
#define MAKE_MEX_BLOCK_QPDUNES "make_acado_block_solver_qpdunes.m.in"
#define MAKE_MEX_INTEGRATOR "make_acado_integrator.m.in"
#define MAKE_MEX_MODEL "make_acado_model.m.in"

#define MAKEFILE_FORCES "makefile.forces.in"
#define MAKEFILE_QPOASES "makefile.qpoases.in"
#define MAKEFILE_QPOASES3 "makefile.qpoases3.in"
#define MAKEFILE_QPDUNES "makefile.qpdunes.in"
#define MAKEFILE_EH_QPOASES "makefile.EH_qpoases.in"
#define MAKEFILE_EH_QPOASES3 ""
#define MAKEFILE_EH_QPDUNES "makefile.EH_qpdunes.in"
#define MAKEFILE_HPMPC "makefile.hpmpc.in"
#define MAKEFILE_INTEGRATOR "makefile.integrator.in"

#define MAKEFILE_SFUN_QPOASES "make_acado_solver_sfunction.m.in"
#define MAKEFILE_SFUN_QPOASES3 "make_acado_solver_sfunction.m.in"
#define SOLVER_SFUN_SOURCE "acado_solver_sfunction.c.in"
#define SOLVER_SFUN_HEADER "acado_solver_sfunction.h.in"

#define DUMMY_TEST_FILE "dummy_test_file.in"

#define COMMON_HEADER_TEMPLATE "acado_common_header.h.in"

#define HPMPC_INTERFACE "hpmpc_interface.c.in"

#endif // ACADO_TOOLKIT_TEMPLATES_HPP
