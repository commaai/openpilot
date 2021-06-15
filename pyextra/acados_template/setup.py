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

from setuptools import setup, find_packages

import sys
print(sys.version_info)

if sys.version_info < (3,5):
    sys.exit('Python version 3.5 or later required. Exiting.')

setup(name='acados_template',
   version='0.1',
   python_requires='>=3.5', # < 3.9
   description='A templating framework for acados',
   url='http://github.com/zanellia/acados',
   author='Andrea Zanelli',
   license='BSD',
   packages = find_packages(),
   include_package_data = True,
   setup_requires=['setuptools_scm'],
   use_scm_version={
     "fallback_version": "0.1-local",
     "root": "../..",
     "relative_to": __file__
   },
   install_requires=[
      'numpy',
      'scipy',
      'casadi>=3.5.1',
      'matplotlib',
      'future-fstrings'
   ],
   package_data={'': [
       'c_templates_tera/main.in.c',
       'c_templates_tera/main_sim.in.c',
       'c_templates_tera/Makefile.in',
       'c_templates_tera/model.in.h',
       'c_templates_tera/acados_solver.in.c',
       'c_templates_tera/acados_solver.in.h',
       'c_templates_tera/acados_sim_solver.in.c',
       'c_templates_tera/acados_sim_solver.in.h',
       'c_templates_tera/acados_solver_sfun.in.c',
       'c_templates_tera/p_constraint.in.h',
       'c_templates_tera/h_constraint.in.h',
       'c_templates_tera/h_e_constraint.in.h',
       'c_templates_tera/phi_e_constraint.in.h',
       'c_templates_tera/phi_constraint.in.h',
       'c_templates_tera/make_sfun.in.m',
       'c_templates_tera/cost_y_fun.in.h',
       'c_templates_tera/cost_y_e_fun.in.h',
       'c_templates_tera/cost_y_0_fun.in.h',
       'c_templates_tera/external_cost.in.h',
       'c_templates_tera/external_cost_e.in.h',
       'c_templates_tera/external_cost_0.in.h',
       'acados_layout.json',
       'acados_sim_layout.json',
       ]},
   zip_safe=False
)
