import SCons.Action
import SCons.Builder
import SCons.Util


def generate(env):
  # The s flag writes the full archive's symbol index during member insertion,
  # avoiding SCons' default second pass with ranlib.
  archive = SCons.Builder.Builder(
    action=SCons.Action.Action("$ARCOM", "$ARCOMSTR"),
    emitter="$LIBEMITTER",
    prefix="$LIBPREFIX",
    suffix="$LIBSUFFIX",
    src_suffix="$OBJSUFFIX",
    src_builder="StaticObject",
  )
  env["BUILDERS"]["StaticLibrary"] = archive
  env["BUILDERS"]["Library"] = archive

  env["AR"] = "ar"
  env["ARFLAGS"] = SCons.Util.CLVar("rcs")
  env["ARCOM"] = "$AR $ARFLAGS $TARGET $SOURCES"
  env["LIBPREFIX"] = "lib"
  env["LIBSUFFIX"] = ".a"


def exists(env):
  return env.Detect("ar")
