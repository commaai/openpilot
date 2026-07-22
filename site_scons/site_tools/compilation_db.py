import fnmatch
import itertools
import json

import SCons
from SCons.Platform import TempFileMunge
from SCons.Tool.asm import ASSuffixes, ASPPSuffixes
from SCons.Tool.cc import CSuffixes
from SCons.Tool.cxx import CXXSuffixes


DEFAULT_DB_NAME = "compile_commands.json"
_entries = []


class CompDBTEMPFILE(TempFileMunge):
  def __call__(self, target, source, env, for_signature):
    return self.cmd


def make_entry_emitter(command):
  action = SCons.Action.Action(command)

  def emitter(target, source, env):
    _entries.append((action, list(target), list(source), env))
    return target, source

  return emitter


def render_database(env):
  entries = []
  use_abspath = env["COMPILATIONDB_USE_ABSPATH"] in [True, 1, "True", "true"]
  path_filter = env.subst("$COMPILATIONDB_PATH_FILTER")

  for action, target, source, entry_env in _entries:
    source_file = source[0]
    output_file = target[0]
    if not source_file.is_derived():
      source_file = source_file.srcnode()

    if use_abspath:
      source_file = source_file.abspath
      output_file = output_file.abspath
    else:
      source_file = source_file.path
      output_file = output_file.path

    if path_filter and not fnmatch.fnmatch(output_file, path_filter):
      continue

    command = action.strfunction(
      target=target,
      source=source,
      env=entry_env,
      overrides={"TEMPFILE": CompDBTEMPFILE},
    )
    entries.append({
      "directory": entry_env.Dir("#").abspath,
      "command": command,
      "file": source_file,
      "output": output_file,
    })

  return json.dumps(entries, sort_keys=True, indent=4, separators=(",", ": ")) + "\n"


def write_database(target, source, env):
  with open(target[0].path, "w") as output_file:
    output_file.write(source[0].read())


def database_emitter(target, source, env):
  if not target and len(source) == 1:
    target = source
  if not target:
    target = [DEFAULT_DB_NAME]
  return target, [SCons.Node.Python.Value(render_database(env))]


def generate(env, **kwargs):
  static_obj, shared_obj = SCons.Tool.createObjBuilders(env)
  components = itertools.chain(
    itertools.product(CSuffixes, [
      (static_obj, "$CCCOM"),
      (shared_obj, "$SHCCCOM"),
    ]),
    itertools.product(CXXSuffixes, [
      (static_obj, "$CXXCOM"),
      (shared_obj, "$SHCXXCOM"),
    ]),
    itertools.product(ASSuffixes, [
      (static_obj, "$ASCOM"),
      (shared_obj, "$ASCOM"),
    ]),
    itertools.product(ASPPSuffixes, [
      (static_obj, "$ASPPCOM"),
      (shared_obj, "$ASPPCOM"),
    ]),
  )

  for suffix, (builder, command) in components:
    emitter = builder.emitter.get(suffix, False)
    if emitter:
      builder.emitter[suffix] = SCons.Builder.ListEmitter([
        emitter,
        make_entry_emitter(command),
      ])

  env["BUILDERS"]["CompilationDatabase"] = SCons.Builder.Builder(
    action=SCons.Action.Action(write_database, kwargs.get("COMPILATIONDB_COMSTR", "Building compilation database $TARGET")),
    emitter=database_emitter,
    suffix="json",
  )
  env.SetDefault(
    COMPILATIONDB_USE_ABSPATH=False,
    COMPILATIONDB_PATH_FILTER="",
  )


def exists(env):
  return True
