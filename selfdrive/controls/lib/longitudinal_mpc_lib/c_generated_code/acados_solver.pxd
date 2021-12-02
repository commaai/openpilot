cimport acados_solver_common

cdef extern from "acados_solver_long.h":
    ctypedef struct nlp_solver_capsule "long_solver_capsule":
        pass

    nlp_solver_capsule * acados_create_capsule "long_acados_create_capsule"()
    int acados_free_capsule "long_acados_free_capsule"(nlp_solver_capsule *capsule)

    int acados_create "long_acados_create"(nlp_solver_capsule * capsule)
    int acados_update_params "long_acados_update_params"(nlp_solver_capsule * capsule, int stage, double *value, int np_)
    int acados_solve "long_acados_solve"(nlp_solver_capsule * capsule)
    int acados_free "long_acados_free"(nlp_solver_capsule * capsule)
    void acados_print_stats "long_acados_print_stats"(nlp_solver_capsule * capsule)

    acados_solver_common.ocp_nlp_in *acados_get_nlp_in "long_acados_get_nlp_in"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_nlp_out "long_acados_get_nlp_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_solver *acados_get_nlp_solver "long_acados_get_nlp_solver"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_config *acados_get_nlp_config "long_acados_get_nlp_config"(nlp_solver_capsule * capsule)
    void *acados_get_nlp_opts "long_acados_get_nlp_opts"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_dims *acados_get_nlp_dims "long_acados_get_nlp_dims"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_plan *acados_get_nlp_plan "long_acados_get_nlp_plan"(nlp_solver_capsule * capsule)
