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






ACADOS_FLAGS = -fPIC -std=c99   #-fno-diagnostics-show-line-numbers -g
# # Debugging
# ACADOS_FLAGS += -g3

MODEL_OBJ=
MODEL_OBJ+= lat_model/lat_expl_ode_fun.o
MODEL_OBJ+= lat_model/lat_expl_vde_forw.o


OCP_OBJ=
OCP_OBJ+= lat_cost/lat_cost_y_0_fun.c
OCP_OBJ+= lat_cost/lat_cost_y_0_fun_jac_ut_xt.c
OCP_OBJ+= lat_cost/lat_cost_y_0_hess.c
OCP_OBJ+= lat_cost/lat_cost_y_fun.c
OCP_OBJ+= lat_cost/lat_cost_y_fun_jac_ut_xt.c
OCP_OBJ+= lat_cost/lat_cost_y_hess.c
OCP_OBJ+= lat_cost/lat_cost_y_e_fun.c
OCP_OBJ+= lat_cost/lat_cost_y_e_fun_jac_ut_xt.c
OCP_OBJ+= lat_cost/lat_cost_y_e_hess.c
OCP_OBJ+= acados_solver_lat.o


SIM_OBJ=
SIM_OBJ+= acados_sim_solver_lat.o

EX_OBJ=
EX_OBJ+= main_lat.o

EX_SIM_OBJ=
EX_SIM_OBJ+= main_sim_lat.o

OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(SIM_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /data/openpilot/third_party/acados/acados/include
LIB_PATH = /data/openpilot/third_party/acados/acados/lib
all: clean casadi_fun example_sim example
shared_lib: bundled_shared_lib ocp_shared_lib sim_shared_lib

CASADI_MODEL_SOURCE=
CASADI_MODEL_SOURCE+= lat_expl_ode_fun.c
CASADI_MODEL_SOURCE+= lat_expl_vde_forw.c
CASADI_COST_Y_SOURCE=
CASADI_COST_Y_SOURCE+= lat_cost_y_fun.c
CASADI_COST_Y_SOURCE+= lat_cost_y_fun_jac_ut_xt.c
CASADI_COST_Y_SOURCE+= lat_cost_y_hess.c
CASADI_COST_Y_E_SOURCE=
CASADI_COST_Y_E_SOURCE+= lat_cost_y_e_fun.c
CASADI_COST_Y_E_SOURCE+= lat_cost_y_e_fun_jac_ut_xt.c
CASADI_COST_Y_E_SOURCE+= lat_cost_y_e_hess.c
CASADI_COST_Y_0_SOURCE=
CASADI_COST_Y_0_SOURCE+= lat_cost_y_0_fun.c
CASADI_COST_Y_0_SOURCE+= lat_cost_y_0_fun_jac_ut_xt.c
CASADI_COST_Y_0_SOURCE+= lat_cost_y_0_hess.c

casadi_fun:
	( cd lat_model ; gcc $(ACADOS_FLAGS) -c $(CASADI_MODEL_SOURCE))
	( cd lat_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_SOURCE))
	( cd lat_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_E_SOURCE))
	( cd lat_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_0_SOURCE))

main:
	gcc $(ACADOS_FLAGS) -c main_lat.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

main_sim:
	gcc $(ACADOS_FLAGS) -c main_sim_lat.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/

ocp_solver:
	gcc $(ACADOS_FLAGS) -c acados_solver_lat.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

sim_solver:
	gcc $(ACADOS_FLAGS) -c acados_sim_solver_lat.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

example: ocp_solver main
	gcc $(ACADOS_FLAGS) -o main_lat $(EX_OBJ) $(OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-I $(INCLUDE_PATH)/acados/ \


example_sim: sim_solver main_sim
	gcc $(ACADOS_FLAGS) -o main_sim_lat $(EX_SIM_OBJ) $(MODEL_OBJ) $(SIM_OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/acados/ \

bundled_shared_lib: casadi_fun ocp_solver sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_solver_lat.so $(OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \

ocp_shared_lib: casadi_fun ocp_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_ocp_solver_lat.so $(OCP_OBJ) $(MODEL_OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	 \
	-lm \

ocp_cython_c: ocp_shared_lib
	cython \
	-o acados_ocp_solver_pyx.c \
	-I $(INCLUDE_PATH)/../interfaces/acados_template/acados_template \
	$(INCLUDE_PATH)/../interfaces/acados_template/acados_template/acados_ocp_solver_pyx.pyx \

ocp_cython_o: ocp_cython_c
	clang $(ACADOS_FLAGS) -c -O2 \
	-o acados_ocp_solver_pyx.o \
	-I /usr/include/python3.8 \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	acados_ocp_solver_pyx.c \

ocp_cython: ocp_cython_o
	clang $(ACADOS_FLAGS) -shared \
	-o acados_ocp_solver_pyx.so \
	-Wl,-rpath=$(LIB_PATH) \
	acados_ocp_solver_pyx.o \
	$(abspath .)/libacados_ocp_solver_lat.so \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo -lqpOASES_e \
	 \
	-lm \

sim_shared_lib: casadi_fun sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_sim_solver_lat.so $(SIM_OBJ) $(MODEL_OBJ) -L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	 \
	-lm \

clean:
	rm -f *.o
	rm -f *.so
	rm -f main_lat

clean_ocp_shared_lib:
	rm -f libacados_ocp_solver_lat.so
	rm -f acados_solver_lat.o
