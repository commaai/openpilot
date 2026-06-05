import platform

from SCons.Script import Dir, File


def compile_single_filter(env, target, filter_gen_script, output_dir, extra_gen_artifacts, script_deps):
  generated_src_files = [File(f) for f in [f'{output_dir}/{target}.cpp', f'{output_dir}/{target}.h']]
  extra_generated_files = [File(f'{output_dir}/{x}') for x in extra_gen_artifacts]
  generator_file = File(filter_gen_script)

  env.Command(generated_src_files + extra_generated_files,
              [generator_file] + script_deps, f"{File(generator_file).relpath} {target} {Dir(output_dir).relpath}")

  generated_cc_file = File(generated_src_files[:1])

  return generated_cc_file


class BaseRednoseCompileMethod:
  def __init__(self, base_py_deps, base_cc_deps):
    self.base_py_deps = base_py_deps
    self.base_cc_deps = base_cc_deps


class CompileFilterMethod(BaseRednoseCompileMethod):
  def __call__(self, env, target, filter_gen_script, output_dir, extra_gen_artifacts=[], gen_script_deps=[]):
    objects = compile_single_filter(env, target, filter_gen_script, output_dir, extra_gen_artifacts, self.base_py_deps + gen_script_deps)
    linker_flags = env.get("LINKFLAGS", [])
    if platform.system() == "Darwin":
      linker_flags = ["-undefined", "dynamic_lookup"]
    lib_target = env.SharedLibrary(f'{output_dir}/{target}', [self.base_cc_deps, objects], LINKFLAGS=linker_flags)

    return lib_target


def generate(env):
  templates = env.Glob("$REDNOSE_ROOT/rednose/templates/*")
  sympy_helpers = env.File("$REDNOSE_ROOT/rednose/helpers/sympy_helpers.py")
  ekf_sym = env.File("$REDNOSE_ROOT/rednose/helpers/ekf_sym.py")

  gen_script_deps = templates + [sympy_helpers, ekf_sym]
  filter_lib_deps = []

  env.AddMethod(CompileFilterMethod(gen_script_deps, filter_lib_deps), "RednoseCompileFilter")


def exists(env):
  return True
