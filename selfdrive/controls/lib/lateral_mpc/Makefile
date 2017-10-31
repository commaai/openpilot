
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
  qp/Bounds.o \
  qp/Constraints.o \
  qp/CyclingManager.o \
  qp/Indexlist.o \
  qp/MessageHandling.o \
  qp/QProblem.o \
  qp/QProblemB.o \
  qp/SubjectTo.o \
  qp/Utils.o \
  qp/EXTRAS/SolutionAnalysis.o \
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


qp/%.o: $(PHONELIBS)/qpoases/SRC/%.cpp
	@echo "[ CXX ] $@"
	mkdir -p qp
	$(CXX) $(CXXFLAGS) -MMD \
           -I mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

qp/EXTRAS/%.o: $(PHONELIBS)/qpoases/SRC/EXTRAS/%.cpp
	@echo "[ CXX ] $@"
	mkdir -p qp/EXTRAS
	$(CXX) $(CXXFLAGS) -MMD \
           -I mpc_export/ \
           $(QPOASES_FLAGS) \
           -c -o '$@' '$<'

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
