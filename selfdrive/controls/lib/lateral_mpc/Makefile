
CC = clang
CXX = clang++

PHONELIBS = ../../../../phonelibs

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CFLAGS = -O3 -fPIC -I.
CXXFLAGS = -O3 -fPIC -I.

QPOASES_FLAGS = -I$(PHONELIBS)/qpoases -I$(PHONELIBS)/qpoases/INCLUDE -I$(PHONELIBS)/qpoases/SRC

ACADO_FLAGS = -I$(PHONELIBS)/acado/include -I$(PHONELIBS)/acado/include/acado

ifeq ($(UNAME_S),Darwin)
	ACADO_LIBS := -lacado_toolkit_s
else ifeq ($(UNAME_M),aarch64)
	ACADO_LIBS := -L $(PHONELIBS)/acado/aarch64/lib -l:libacado_toolkit.a -l:libacado_casadi.a -l:libacado_csparse.a
else
	ACADO_LIBS := -L $(PHONELIBS)/acado/x64/lib -l:libacado_toolkit.a -l:libacado_casadi.a -l:libacado_csparse.a
endif

OBJS = \
  lib_qp/Bounds.o \
  lib_qp/Constraints.o \
  lib_qp/CyclingManager.o \
  lib_qp/Indexlist.o \
  lib_qp/MessageHandling.o \
  lib_qp/QProblem.o \
  lib_qp/QProblemB.o \
  lib_qp/SubjectTo.o \
  lib_qp/Utils.o \
  lib_qp/EXTRAS/SolutionAnalysis.o \
	lib_mpc_export/acado_qpoases_interface.o \
  lib_mpc_export/acado_integrator.o \
  lib_mpc_export/acado_solver.o \
  lib_mpc_export/acado_auxiliary_functions.o \
  lateral_mpc.o

DEPS := $(OBJS:.o=.d)

.PHONY: all
all: libmpc.so

libmpc.so: $(OBJS)
	$(CXX) -shared -o '$@' $^ -lm

lib_qp/%.o: $(PHONELIBS)/qpoases/SRC/%.cpp
	@echo "[ CXX ] $@"
	mkdir -p lib_qp/EXTRAS
	$(CXX) $(CXXFLAGS) -MMD \
           -I lib_mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

%.o: %.cpp
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -I lib_mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
           -I lib_mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

generator: generator.cpp
	$(CXX) -v -Wall -std=c++11 \
         generator.cpp \
         -o generator \
         $(ACADO_FLAGS) \
         $(ACADO_LIBS)

.PHONY: generate
generate: generator
	./generator

.PHONY: clean
clean:
	rm -f *.so generator $(OBJS) $(DEPS)

-include $(DEPS)
