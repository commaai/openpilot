import os

DBC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbc')

# -I include path for e.g. "#include <opendbc/safety/safety.h>"
INCLUDE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
