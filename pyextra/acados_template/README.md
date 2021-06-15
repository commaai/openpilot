`acados_template` is a Python package that can be used to specify optimal control problems from Python and to generate self-contained C code that uses the acados solvers to solve them.
The pip package is based on templated code (C files, Header files and Makefiles), which are rendered from Python using the templating engine `Tera`.
The genereated C code can be compiled into a self-contained C library that can be deployed on an embedded system.

# Usage

## Optimal Control Problem description
The Python interface relies on the same problem formulation as the MATLAB interface [see here](https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf).

## Installation

### Linux/macOS

1. Compile and install `acados` by running:
```bash
cd <acados_root>/build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

2. Install acados_template Python package by running
```
pip3 install <acados_root>/interfaces/acados_template
```
Note: If you are working with a virtual Python environment, use the `pip` corresponding to this Python environment instead of `pip3`.

3. Add the path to the compiled shared libraries `libacados.so, libblasfeo.so, libhpipm.so` to `LD_LIBRARY_PATH` (default path is `<acados_root/lib>`) by running:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
```
Tipp: you can add this line to your `.bashrc`/`.zshrc`.

4. Run a Python example to check that everything works.
We suggest to get started with the example
`<acados_root>/examples/acados_python/getting_started/minimal_example_ocp.py`.

5. Optional: Can be done automatically through the interface:
In order to be able to successfully render C code templates, you need to download the `t_renderer` binaries for your platform from <https://github.com/acados/tera_renderer/releases/> and place them in `<acados_root>/bin` (please strip the version and platform from the binaries (e.g.`t_renderer-v0.0.34 -> t_renderer`).
Notice that you might need to make `t_renderer` executable.
Run `export ACADOS_SOURCE_DIR=<acados_root>` such that the location of acados will be known to the Python package at run time.

6. Optional: Set `acados_lib_path`, `acados_include_path`.
If you want the generated Makefile to refer to a specific path (e.g. when cross-compiling or compiling from a location different from the one where you generate the C code), you will have to set these paths accordingly in the generating Python code.

### Windows
You can in principle install the acados_template package within your native Python shell, but we highly recommend 
using Windows Subsystems for Linux (https://docs.microsoft.com/en-us/windows/wsl/about) and to follow the 
Linux/macOS installation instruction.

For more information visit
https://docs.acados.org/interfaces/
