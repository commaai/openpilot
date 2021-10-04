import re
import SCons
from SCons.Action import Action
from SCons.Scanner import Scanner

pyx_from_import_re = re.compile(r'^from\s+(\S+)\s+cimport', re.M)
pyx_import_re = re.compile(r'^cimport\s+(\S+)', re.M)
cdef_import_re = re.compile(r'^cdef extern from\s+.(\S+).:', re.M)


def pyx_scan(node, env, path, arg=None):
  contents = node.get_text_contents()

  # from <module> cimport ...
  matches = pyx_from_import_re.findall(contents)
  # cimport <module>
  matches += pyx_import_re.findall(contents)

  # Modules can be either .pxd or .pyx files
  files = [f"{m.replace('.', '/')}.pxd" for m in matches]
  files += [f"{m.replace('.', '/')}.pyx" for m in matches]

  # cdef extern from <file>
  files += cdef_import_re.findall(contents)

  # Handle relative imports
  cur_dir = str(node.get_dir())
  files = [cur_dir + f if f.startswith('/') else f for f in files]

  # Filter out non-existing files (probably system imports)
  files = [f for f in files if env.File(f).exists()]
  return env.File(files)


pyxscanner = Scanner(function=pyx_scan, skeys=['.pyx', '.pxd'], recursive=True)
cythonAction = Action("$CYTHONCOM")


def create_builder(env):
  try:
    cython = env['BUILDERS']['Cython']
  except KeyError:
    cython = SCons.Builder.Builder(
      action=cythonAction,
      emitter={},
      suffix=cython_suffix_emitter,
      single_source=1
    )
    env.Append(SCANNERS=pyxscanner)
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
