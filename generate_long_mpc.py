#!/usr/bin/env python3
import sys
import os

# Add paths to python path
sys.path.insert(0, '.')
sys.path.insert(0, 'third_party/acados')

# Set environment variables
os.environ['ACADOS_SOURCE_DIR'] = '/Users/tom/Documents/apps/openpilot/third_party/acados'
os.environ['ACADOS_PYTHON_INTERFACE_PATH'] = '/Users/tom/Documents/apps/openpilot/third_party/acados/acados_template'
os.environ['TERA_PATH'] = '/Users/tom/Documents/apps/openpilot/third_party/acados/Darwin/t_renderer'



# Create the c_generated_code directory if it doesn't exist
cgen_dir = 'selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code'
os.makedirs(cgen_dir, exist_ok=True)

# Create __init__.py files to make directories into modules
with open(os.path.join(cgen_dir, '__init__.py'), 'w') as f:
    f.write('# Generated for importing\n')

# Now try to run the generation part directly
os.chdir('selfdrive/controls/lib/longitudinal_mpc_lib')

# Import the generation function directly
from long_mpc import gen_long_ocp
from openpilot.third_party.acados.acados_template import AcadosOcpSolver

# Generate the OCP and create the code
ocp = gen_long_ocp()
JSON_FILE = "acados_ocp_long.json"
AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
print("MPC code generation completed successfully!")
