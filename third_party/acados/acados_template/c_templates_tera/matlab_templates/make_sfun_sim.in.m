%
% Copyright (c) The acados authors.
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;

%

SOURCES = [ 'acados_sim_solver_sfunction_{{ model.name }}.c ', ...
            'acados_sim_solver_{{ model.name }}.c ', ...
            {%- if  solver_options.integrator_type == 'ERK' %}
            '{{ model.name }}_model/{{ model.name }}_expl_ode_fun.c ',...
            '{{ model.name }}_model/{{ model.name }}_expl_vde_forw.c ',...
            '{{ model.name }}_model/{{ model.name }}_expl_vde_adj.c ',...
                {%- if solver_options.hessian_approx == 'EXACT' %}
            '{{ model.name }}_model/{{ model.name }}_expl_ode_hess.c ',...
                {%- endif %}
            {%- elif solver_options.integrator_type == "IRK" %}
            '{{ model.name }}_model/{{ model.name }}_impl_dae_fun.c ', ...
            '{{ model.name }}_model/{{ model.name }}_impl_dae_fun_jac_x_xdot_z.c ', ...
            '{{ model.name }}_model/{{ model.name }}_impl_dae_jac_x_xdot_u_z.c ', ...
                {%- if solver_options.hessian_approx == 'EXACT' %}
            '{{ model.name }}_model/{{ model.name }}_impl_dae_hess.c ',...
                {%- endif %}
            {%- elif solver_options.integrator_type == "GNSF" %}
                {%- if model.gnsf.purely_linear != 1 %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun.c ',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_fun_jac_y.c ',...
            '{{ model.name }}_model/{{ model.name }}_gnsf_phi_jac_y_uhat.c ',...
                {%- if model.gnsf.nontrivial_f_LO == 1 %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz.c ',...
                {%- endif %}
                {%- endif %}
            '{{ model.name }}_model/{{ model.name }}_gnsf_get_matrices_fun.c ',...
            {%- endif %}
          ];

INC_PATH = '{{ acados_include_path }}';

INCS = [ ' -I', fullfile(INC_PATH, 'blasfeo', 'include'), ...
         ' -I', fullfile(INC_PATH, 'hpipm', 'include'), ...
        ' -I', INC_PATH, ' -I', fullfile(INC_PATH, 'acados'), ' '];

CFLAGS  = ' -O';

LIB_PATH = '{{ acados_lib_path }}';

LIBS = '-lacados -lblasfeo -lhpipm';

try
    % eval( [ 'mex -v -output  acados_sim_solver_sfunction_{{ model.name }} ', ...
    eval( [ 'mex -output  acados_sim_solver_sfunction_{{ model.name }} ', ...
        CFLAGS, INCS, ' ', SOURCES, ' -L', LIB_PATH, ' ', LIBS ]);

catch exception
    disp('make_sfun failed with the following exception:')
    disp(exception);
    disp('Try adding -v to the mex command above to get more information.')
    keyboard
end


fprintf( [ '\n\nSuccessfully created sfunction:\nacados_sim_solver_sfunction_{{ model.name }}', '.', ...
    eval('mexext')] );


global sfun_sim_input_names
sfun_sim_input_names = {};

%% print note on usage of s-function
fprintf('\n\nNote: Usage of Sfunction is as follows:\n')
input_note = 'Inputs are:\n1) x0, initial state, size [{{ dims.nx }}]\n ';
i_in = 2;
sfun_sim_input_names = [sfun_sim_input_names; 'x0 [{{ dims.nx }}]'];

{%- if dims.nu > 0 %}
input_note = strcat(input_note, num2str(i_in), ') u, size [{{ dims.nu }}]\n ');
i_in = i_in + 1;
sfun_sim_input_names = [sfun_sim_input_names; 'u [{{ dims.nu }}]'];
{%- endif %}

{%- if dims.np > 0 %}
input_note = strcat(input_note, num2str(i_in), ') parameters, size [{{ dims.np }}]\n ');
i_in = i_in + 1;
sfun_sim_input_names = [sfun_sim_input_names; 'p [{{ dims.np }}]'];
{%- endif %}


fprintf(input_note)

disp(' ')

global sfun_sim_output_names
sfun_sim_output_names = {};

output_note = strcat('Outputs are:\n', ...
                '1) x1 - simulated state, size [{{ dims.nx }}]\n');
sfun_sim_output_names = [sfun_sim_output_names; 'x1 [{{ dims.nx }}]'];

fprintf(output_note)


% The mask drawing command is:
% ---
% global sfun_sim_input_names sfun_sim_output_names
% for i = 1:length(sfun_sim_input_names)
% 	port_label('input', i, sfun_sim_input_names{i})
% end
% for i = 1:length(sfun_sim_output_names)
% 	port_label('output', i, sfun_sim_output_names{i})
% end
% ---
% It can be used by copying it in sfunction/Mask/Edit mask/Icon drawing commands
%   (you can access it wirth ctrl+M on the s-function)