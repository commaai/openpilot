#ifndef {{ model.name }}_PHI_E_CONSTRAINT
#define {{ model.name }}_PHI_E_CONSTRAINT

#ifdef __cplusplus
extern "C" {
#endif

{% if dims.nphi_e > 0 %}
int {{ model.name }}_phi_e_constraint(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_phi_e_constraint_work(int *, int *, int *, int *);
const int *{{ model.name }}_phi_e_constraint_sparsity_in(int);
const int *{{ model.name }}_phi_e_constraint_sparsity_out(int);
int {{ model.name }}_phi_e_constraint_n_in(void);
int {{ model.name }}_phi_e_constraint_n_out(void);
{% endif %}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // {{ model.name }}_PHI_E_CONSTRAINT
