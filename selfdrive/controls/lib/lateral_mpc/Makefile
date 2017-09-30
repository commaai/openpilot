CC = clang
CXX = clang++

PHONELIBS = ../../../../phonelibs

UNAME_M := $(shell uname -m)

CFLAGS = -O3 -fPIC -I.
CXXFLAGS = -O3 -fPIC -I.

QPOASES_FLAGS = -I$(PHONELIBS)/qpoases -I$(PHONELIBS)/qpoases/INCLUDE -I$(PHONELIBS)/qpoases/SRC 

ACADO_FLAGS = -I$(PHONELIBS)/acado/include -I$(PHONELIBS)/acado/include/acado

ifeq ($(UNAME_M),aarch64)
ACADO_LIBS := -L $(PHONELIBS)/acado/aarch64/lib -l:libacado_toolkit.a -l:libacado_casadi.a -l:libacado_csparse.a
else
ACADO_LIBS := -L $(PHONELIBS)/acado/x64/lib -l:libacado_toolkit.a -l:libacado_casadi.a -l:libacado_csparse.a
endif

OBJS = \
  $(PHONELIBS)/qpoases/SRC/Bounds.o \
  $(PHONELIBS)/qpoases/SRC/Constraints.o \
  $(PHONELIBS)/qpoases/SRC/CyclingManager.o \
  $(PHONELIBS)/qpoases/SRC/Indexlist.o \
  $(PHONELIBS)/qpoases/SRC/MessageHandling.o \
  $(PHONELIBS)/qpoases/SRC/QProblem.o \
  $(PHONELIBS)/qpoases/SRC/QProblemB.o \
  $(PHONELIBS)/qpoases/SRC/SubjectTo.o \
  $(PHONELIBS)/qpoases/SRC/Utils.o \
  $(PHONELIBS)/qpoases/SRC/EXTRAS/SolutionAnalysis.o \
  mpc_export/acado_qpoases_interface.o \
  mpc_export/acado_integrator.o \
  mpc_export/acado_solver.o \
  mpc_export/acado_auxiliary_functions.o \
  mpc.o

DEPS := $(OBJS:.o=.d)

.PHONY: all
all: libcommampc.so

libcommampc.so: $(OBJS)
	$(CXX) -shared -o '$@' $^ -lm

%.o: %.cpp
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -I mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
           -I mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

generator: generator.cpp
	$(CXX) -Wall -std=c++11 \
         generator.cpp \
         -o generator \
         $(ACADO_FLAGS) \
         $(ACADO_LIBS)

.PHONY: generate
generate: generator
	./generator

.PHONY: clean
clean:
	rm -f libcommampc.so generator $(OBJS) $(DEPS)

-include $(DEPS)
