# -*- coding: future_fstrings -*-
#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import os
import sys
from subprocess import call


class CMakeBuilder:
    """
    Class to work with the `CMake` build system.
    """
    def __init__(self):
        self._source_dir = None  # private source directory, this is set to code_export_dir
        self.build_dir = 'build'
        self._build_dir = None  # private build directory, usually rendered to abspath(build_dir)
        self.generator = None
        """Defines the generator, options can be found via `cmake --help` under 'Generator'. Type: string. Linux default 'Unix Makefiles', Windows 'Visual Studio 15 2017 Win64'; default value: `None`."""
        # set something for Windows
        if os.name == 'nt':
            self.generator = 'Visual Studio 15 2017 Win64'
        self.build_targets = None
        """A comma-separated list of the build targets, if `None` then all targets will be build; type: List of strings; default: `None`."""
        self.options_on = None
        """List of strings as CMake options which are translated to '-D Opt[0]=ON -D Opt[1]=ON ...'; default: `None`."""

    # Generate the command string for handling the cmake command.
    def get_cmd1_cmake(self):
        defines_str = ''
        if self.options_on is not None:
            defines_arr = [f' -D{opt}=ON' for opt in self.options_on]
            defines_str = ' '.join(defines_arr)
        generator_str = ''
        if self.generator is not None:
            generator_str = f' -G"{self.generator}"'
        return f'cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="{self._source_dir}"{defines_str}{generator_str} -Wdev -S"{self._source_dir}" -B"{self._build_dir}"'

    # Generate the command string for handling the build.
    def get_cmd2_build(self):
        import multiprocessing
        cmd = f'cmake --build "{self._build_dir}" --config Release -j{multiprocessing.cpu_count()}'
        if self.build_targets is not None:
            cmd += f' -t {self.build_targets}'
        return cmd

    # Generate the command string for handling the install command.
    def get_cmd3_install(self):
        return f'cmake --install "{self._build_dir}"'

    def exec(self, code_export_directory):
        """
        Execute the compilation using `CMake` with the given settings.
        :param code_export_directory: must be the absolute path to the directory where the code was exported to
        """
        if(os.path.isabs(code_export_directory) is False):
            print(f'(W) the code export directory "{code_export_directory}" is not an absolute path!')
        self._source_dir = code_export_directory
        self._build_dir = os.path.abspath(self.build_dir)
        try:
            os.mkdir(self._build_dir)
        except FileExistsError as e:
            pass

        try:
            os.chdir(self._build_dir)
            cmd_str = self.get_cmd1_cmake()
            print(f'call("{cmd_str})"')
            retcode = call(cmd_str, shell=True)
            if retcode != 0:
                raise RuntimeError(f'CMake command "{cmd_str}" was terminated by signal {retcode}')
            cmd_str = self.get_cmd2_build()
            print(f'call("{cmd_str}")')
            retcode = call(cmd_str, shell=True)
            if retcode != 0:
                raise RuntimeError(f'Build command "{cmd_str}" was terminated by signal {retcode}')
            cmd_str = self.get_cmd3_install()
            print(f'call("{cmd_str}")')
            retcode = call(cmd_str, shell=True)
            if retcode != 0:
                raise RuntimeError(f'Install command "{cmd_str}" was terminated by signal {retcode}')
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)
        except Exception as e:
            print("Execution failed:", e, file=sys.stderr)
            exit(1)
