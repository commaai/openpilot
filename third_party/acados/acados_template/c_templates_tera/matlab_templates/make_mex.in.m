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

function make_mex_{{ model.name }}()

    opts.output_dir = pwd;

    % get acados folder
    acados_folder = getenv('ACADOS_INSTALL_DIR');

    % set paths
    acados_include = ['-I' fullfile(acados_folder, 'include')];
    template_lib_include = ['-l' 'acados_ocp_solver_{{ model.name }}'];
    template_lib_path = ['-L' fullfile(pwd)];

    acados_lib_path = ['-L' fullfile(acados_folder, 'lib')];
    external_include = ['-I', fullfile(acados_folder, 'external')];
    blasfeo_include = ['-I', fullfile(acados_folder, 'external', 'blasfeo', 'include')];
    hpipm_include = ['-I', fullfile(acados_folder, 'external', 'hpipm', 'include')];

    % load linking information of compiled acados
    link_libs_core_filename = fullfile(acados_folder, 'lib', 'link_libs.json');
    addpath(fullfile(acados_folder, 'external', 'jsonlab'));
    link_libs = loadjson(link_libs_core_filename);

    % add necessary link instructs
    acados_lib_extra = {};
    lib_names = fieldnames(link_libs);
    for idx = 1 : numel(lib_names)
        lib_name = lib_names{idx};
        link_arg = link_libs.(lib_name);
        if ~isempty(link_arg)
            acados_lib_extra = [acados_lib_extra, link_arg];
        end
    end


    mex_include = ['-I', fullfile(acados_folder, 'interfaces', 'acados_matlab_octave')];

    mex_names = { ...
        'acados_mex_create_{{ model.name }}' ...
        'acados_mex_free_{{ model.name }}' ...
        'acados_mex_solve_{{ model.name }}' ...
        'acados_mex_set_{{ model.name }}' ...
    };

    mex_files = cell(length(mex_names), 1);
    for k=1:length(mex_names)
        mex_files{k} = fullfile([mex_names{k}, '.c']);
    end

    %% octave C flags
    if is_octave()
        if ~exist(fullfile(opts.output_dir, 'cflags_octave.txt'), 'file')
            diary(fullfile(opts.output_dir, 'cflags_octave.txt'));
            diary on
            mkoctfile -p CFLAGS
            diary off
            input_file = fopen(fullfile(opts.output_dir, 'cflags_octave.txt'), 'r');
            cflags_tmp = fscanf(input_file, '%[^\n]s');
            fclose(input_file);
            if ~ismac()
                cflags_tmp = [cflags_tmp, ' -std=c99 -fopenmp'];
            else
                cflags_tmp = [cflags_tmp, ' -std=c99'];
            end
            input_file = fopen(fullfile(opts.output_dir, 'cflags_octave.txt'), 'w');
            fprintf(input_file, '%s', cflags_tmp);
            fclose(input_file);
        end
        % read cflags from file
        input_file = fopen(fullfile(opts.output_dir, 'cflags_octave.txt'), 'r');
        cflags_tmp = fscanf(input_file, '%[^\n]s');
        fclose(input_file);
        setenv('CFLAGS', cflags_tmp);
    end

    %% compile mex
    for ii=1:length(mex_files)
        disp(['compiling ', mex_files{ii}])
        if is_octave()
    %        mkoctfile -p CFLAGS
            mex(acados_include, template_lib_include, external_include, blasfeo_include, hpipm_include,...
                template_lib_path, mex_include, acados_lib_path, '-lacados', '-lhpipm', '-lblasfeo',...
                acados_lib_extra{:}, mex_files{ii})
        else
            if ismac()
                FLAGS = 'CFLAGS=$CFLAGS -std=c99';
            else
                FLAGS = 'CFLAGS=$CFLAGS -std=c99 -fopenmp';
            end
            mex(FLAGS, acados_include, template_lib_include, external_include, blasfeo_include, hpipm_include,...
                template_lib_path, mex_include, acados_lib_path, '-lacados', '-lhpipm', '-lblasfeo',...
                acados_lib_extra{:}, mex_files{ii})
        end
    end


end