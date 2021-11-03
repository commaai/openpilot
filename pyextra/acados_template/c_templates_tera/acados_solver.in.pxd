cimport acados_solver_common

cdef extern from "acados_solver_{{ model.name }}.h":
    ctypedef struct nlp_solver_capsule:
        pass

    nlp_solver_capsule * acados_create_capsule "{{ model.name }}_acados_create_capsule"()
    int acados_free_capsule "{{ model.name }}_acados_free_capsule"(nlp_solver_capsule *capsule)

    int acados_create "{{ model.name }}_acados_create"(nlp_solver_capsule * capsule)
    int acados_update_params "{{ model.name }}_acados_update_params"(nlp_solver_capsule * capsule, int stage, double *value, int np_)
    int acados_solve "{{ model.name }}_acados_solve"(nlp_solver_capsule * capsule)
    int acados_free "{{ model.name }}_acados_free"(nlp_solver_capsule * capsule)
    void acados_print_stats "{{ model.name }}_acados_print_stats"(nlp_solver_capsule * capsule)

    acados_solver_common.ocp_nlp_in *acados_get_nlp_in "{{ model.name }}_acados_get_nlp_in"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_nlp_out "{{ model.name }}_acados_get_nlp_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_solver *acados_get_nlp_solver "{{ model.name }}_acados_get_nlp_solver"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_config *acados_get_nlp_config "{{ model.name }}_acados_get_nlp_config"(nlp_solver_capsule * capsule)
    void *acados_get_nlp_opts "{{ model.name }}_acados_get_nlp_opts"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_dims *acados_get_nlp_dims "{{ model.name }}_acados_get_nlp_dims"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_plan *acados_get_nlp_plan "{{ model.name }}_acados_get_nlp_plan"(nlp_solver_capsule * capsule)
