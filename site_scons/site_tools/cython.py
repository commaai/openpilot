import SCons
from SCons.Action import Action

cythonAction = Action("$CYTHONCOM")

def create_builder(env):
    try:
        cython = env['BUILDERS']['Cython']
    except KeyError:
        cython = SCons.Builder.Builder(
                  action = cythonAction,
                  emitter = {},
                  suffix = cython_suffix_emitter,
                  single_source = 1)
        env['BUILDERS']['Cython'] = cython
    return cython

def cython_suffix_emitter(env, source):
    return "$CYTHONCFILESUFFIX"

def generate(env):
    env["CYTHON"] = "cythonize"
    env["CYTHONCOM"] = "$CYTHON $CYTHONFLAGS $SOURCE"
    env["CYTHONCFILESUFFIX"] = ".cpp"

    c_file, _ = SCons.Tool.createCFileBuilders(env)

    c_file.suffix['.pyx'] = cython_suffix_emitter
    c_file.add_action('.pyx', cythonAction)

    c_file.suffix['.py'] = cython_suffix_emitter
    c_file.add_action('.py', cythonAction)

    create_builder(env)

def exists(env):
  return True
