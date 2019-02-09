#!/usr/bin/env bash
sed -i -e 's/1\.2345678[^)]*)/TR)/g' -e 's/out)/out, double TR)/g' -e 's/Objective(  )/Objective(double TR)/g' -e 's/ValueOut )/ValueOut, TR )/g' -e 's/ionStep(  )/ionStep(double TR)/g' -e 's/Objective(double TR);/Objective(TR);/g' mpc_export/acado_solver.c
