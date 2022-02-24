cimport acados_solver_common

cdef extern from "acados_solver_lat.h":
    ctypedef struct nlp_solver_capsule "lat_solver_capsule":
        pass

    nlp_solver_capsule * acados_create_capsule "lat_acados_create_capsule"()
    int acados_free_capsule "lat_acados_free_capsule"(nlp_solver_capsule *capsule)

    int acados_create "lat_acados_create"(nlp_solver_capsule * capsule)
    int acados_update_params "lat_acados_update_params"(nlp_solver_capsule * capsule, int stage, double *value, int np_)
    int acados_solve "lat_acados_solve"(nlp_solver_capsule * capsule)
    int acados_free "lat_acados_free"(nlp_solver_capsule * capsule)
    void acados_print_stats "lat_acados_print_stats"(nlp_solver_capsule * capsule)

    acados_solver_common.ocp_nlp_in *acados_get_nlp_in "lat_acados_get_nlp_in"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_nlp_out "lat_acados_get_nlp_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_solver *acados_get_nlp_solver "lat_acados_get_nlp_solver"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_config *acados_get_nlp_config "lat_acados_get_nlp_config"(nlp_solver_capsule * capsule)
    void *acados_get_nlp_opts "lat_acados_get_nlp_opts"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_dims *acados_get_nlp_dims "lat_acados_get_nlp_dims"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_plan *acados_get_nlp_plan "lat_acados_get_nlp_plan"(nlp_solver_capsule * capsule)
