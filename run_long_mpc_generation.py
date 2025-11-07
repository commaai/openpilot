#!/usr/bin/env python3
import sys
import os

# Add paths to python path
sys.path.insert(0, '/Users/tom/Documents/apps/openpilot')
sys.path.insert(0, '/Users/tom/Documents/apps/openpilot/third_party/acados')

# Set environment variables
os.environ['ACADOS_SOURCE_DIR'] = '/Users/tom/Documents/apps/openpilot/third_party/acados'
os.environ['ACADOS_PYTHON_INTERFACE_PATH'] = '/Users/tom/Documents/apps/openpilot/third_party/acados/acados_template'
os.environ['TERA_PATH'] = '/Users/tom/Documents/apps/openpilot/third_party/acados/Darwin/t_renderer'

# Change to the directory where the script is located
os.chdir('/Users/tom/Documents/apps/openpilot/selfdrive/controls/lib/longitudinal_mpc_lib')

# Now run the long_mpc.py as main
if __name__ == '__main__':
    # Execute the original file content with __name__ set to '__main__'
    with open('long_mpc.py', 'r') as f:
        content = f.read()
    
    # Execute the content in the __main__ context
    exec(content, {'__name__': '__main__', '__file__': 'long_mpc.py'})