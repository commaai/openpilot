/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_{{ model.name }}.h"


int main()
{
    int status = 0;
    sim_solver_capsule *capsule = {{ model.name }}_acados_sim_solver_create_capsule();
    status = {{ model.name }}_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = {{ model.name }}_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = {{ model.name }}_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = {{ model.name }}_acados_get_sim_out(capsule);
    void *acados_sim_dims = {{ model.name }}_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[{{ dims.nx }}];
    {%- for i in range(end=dims.nx) %}
    x_current[{{ i }}] = 0.0;
    {%- endfor %}

  {% if constraints.lbx_0 %}
    {%- for i in range(end=dims.nbx_0) %}
    x_current[{{ constraints.idxbx_0[i] }}] = {{ constraints.lbx_0[i] }};
    {%- endfor %}
    {% if dims.nbx_0 != dims.nx %}
    printf("main_sim: NOTE: initial state not fully defined via lbx_0, using 0.0 for indices that are not in idxbx_0.");
    {%- endif %}
  {% else %}
    printf("main_sim: initial state not defined, should be in lbx_0, using zero vector.");
  {%- endif %}


    // initial value for control input
    double u0[{{ dims.nu }}];
    {%- for i in range(end=dims.nu) %}
    u0[{{ i }}] = 0.0;
    {%- endfor %}

  {%- if dims.np > 0 %}
    // set parameters
    double p[{{ dims.np }}];
    {% for item in parameter_values %}
    p[{{ loop.index0 }}] = {{ item }};
    {% endfor %}

    {{ model.name }}_acados_sim_update_params(capsule, p, {{ dims.np }});
  {% endif %}{# if np > 0 #}

    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        status = {{ model.name }}_acados_sim_solve(capsule);

        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);
        
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < {{ dims.nx }}; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = {{ model.name }}_acados_sim_free(capsule);
    if (status) {
        printf("{{ model.name }}_acados_sim_free() returned status %d. \n", status);
    }

    {{ model.name }}_acados_sim_solver_free_capsule(capsule);

    return status;
}
