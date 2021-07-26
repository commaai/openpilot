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
MODEL_OBJ+= lead_model/lead_expl_ode_fun.o
MODEL_OBJ+= lead_model/lead_expl_vde_forw.o


OCP_OBJ=
OCP_OBJ+= lead_cost/lead_cost_y_0_fun.c
OCP_OBJ+= lead_cost/lead_cost_y_0_fun_jac_ut_xt.c
OCP_OBJ+= lead_cost/lead_cost_y_0_hess.c
OCP_OBJ+= lead_cost/lead_cost_y_fun.c
OCP_OBJ+= lead_cost/lead_cost_y_fun_jac_ut_xt.c
OCP_OBJ+= lead_cost/lead_cost_y_hess.c
OCP_OBJ+= lead_cost/lead_cost_y_e_fun.c
OCP_OBJ+= lead_cost/lead_cost_y_e_fun_jac_ut_xt.c
OCP_OBJ+= lead_cost/lead_cost_y_e_hess.c
OCP_OBJ+= acados_solver_lead.o


SIM_OBJ=
SIM_OBJ+= acados_sim_solver_lead.o

EX_OBJ=
EX_OBJ+= main_lead.o

EX_SIM_OBJ=
EX_SIM_OBJ+= main_sim_lead.o

OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(SIM_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /home/batman/openpilot/phonelibs/acados/acados/include
LIB_PATH = /home/batman/openpilot/phonelibs/acados/acados/lib
all: clean casadi_fun example_sim example
shared_lib: bundled_shared_lib ocp_shared_lib sim_shared_lib

CASADI_MODEL_SOURCE=
CASADI_MODEL_SOURCE+= lead_expl_ode_fun.c
CASADI_MODEL_SOURCE+= lead_expl_vde_forw.c
CASADI_COST_Y_SOURCE=
CASADI_COST_Y_SOURCE+= lead_cost_y_fun.c
CASADI_COST_Y_SOURCE+= lead_cost_y_fun_jac_ut_xt.c
CASADI_COST_Y_SOURCE+= lead_cost_y_hess.c
CASADI_COST_Y_E_SOURCE=
CASADI_COST_Y_E_SOURCE+= lead_cost_y_e_fun.c
CASADI_COST_Y_E_SOURCE+= lead_cost_y_e_fun_jac_ut_xt.c
CASADI_COST_Y_E_SOURCE+= lead_cost_y_e_hess.c
CASADI_COST_Y_0_SOURCE=
CASADI_COST_Y_0_SOURCE+= lead_cost_y_0_fun.c
CASADI_COST_Y_0_SOURCE+= lead_cost_y_0_fun_jac_ut_xt.c
CASADI_COST_Y_0_SOURCE+= lead_cost_y_0_hess.c

casadi_fun:
	( cd lead_model ; gcc $(ACADOS_FLAGS) -c $(CASADI_MODEL_SOURCE))
	( cd lead_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_SOURCE))
	( cd lead_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_E_SOURCE))
	( cd lead_cost ; gcc $(ACADOS_FLAGS) -c $(CASADI_COST_Y_0_SOURCE))

main:
	gcc $(ACADOS_FLAGS) -c main_lead.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

main_sim:
	gcc $(ACADOS_FLAGS) -c main_sim_lead.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/

ocp_solver:
	gcc $(ACADOS_FLAGS) -c acados_solver_lead.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

sim_solver:
	gcc $(ACADOS_FLAGS) -c acados_sim_solver_lead.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

example: ocp_solver main
	gcc $(ACADOS_FLAGS) -o main_lead $(EX_OBJ) $(OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-I $(INCLUDE_PATH)/acados/ \


example_sim: sim_solver main_sim
	gcc $(ACADOS_FLAGS) -o main_sim_lead $(EX_SIM_OBJ) $(MODEL_OBJ) $(SIM_OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/acados/ \

bundled_shared_lib: casadi_fun ocp_solver sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_solver_lead.so $(OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	 \
	-lm \

ocp_shared_lib: casadi_fun ocp_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_ocp_solver_lead.so $(OCP_OBJ) $(MODEL_OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	 \
	-lm \

sim_shared_lib: casadi_fun sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_sim_solver_lead.so $(SIM_OBJ) $(MODEL_OBJ) -L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	 \
	-lm \

clean:
	rm -f *.o
	rm -f *.so
	rm -f main_lead

clean_ocp_shared_lib:
	rm -f libacados_ocp_solver_lead.so
	rm -f acados_solver_lead.o
