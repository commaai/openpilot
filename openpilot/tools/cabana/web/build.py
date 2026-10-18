#!/usr/bin/env python3
"""Builds cabana for the browser with emscripten.

Third-party sources (imgui, implot, json11, capnproto built for wasm) live in CABANA_WEB_DEPS.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CABANA = HERE.parent
OPENPILOT = CABANA.parent.parent
ROOT = OPENPILOT.parent
DEPS = Path(os.environ.get("CABANA_WEB_DEPS", Path.home() / "cabana-web-deps"))
OBJ = HERE / "obj"
DIST = HERE / "dist"

IMGUI = DEPS / "imgui"
IMPLOT = DEPS / "implot"
JSON11 = DEPS / "json11"
CAPNP = DEPS / "capnp-wasm"

CXXFLAGS = [
  "-std=c++20", "-O2", "-g1", "-pthread", "-fexceptions",
  "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-function", "-Wno-deprecated-declarations",
  "-DIMGUI_IMPL_OPENGL_ES3", "-DGLFW_INCLUDE_NONE",
  '-DOPENDBC_FILE_PATH="/opendbc"',
  '-DCABANA_FONTS_DIR="/fonts"',
  '-DBOOTSTRAP_ICONS_TTF="/fonts/bootstrap-icons.ttf"',
  "--use-port=contrib.glfw3", "--use-port=bzip2",
]
INCLUDES = [ROOT, OPENPILOT, ROOT / "msgq_repo", ROOT / "opendbc_repo", OPENPILOT / "cereal/gen/cpp",
            CAPNP / "include", IMGUI, IMGUI / "backends", IMPLOT, DEPS / "include"]

LINKFLAGS = [
  "-pthread", "-fexceptions",
  "-sUSE_WEBGL2=1", "-sFULL_ES3=1", "-sMIN_WEBGL_VERSION=2", "-sMAX_WEBGL_VERSION=2",
  "-sALLOW_MEMORY_GROWTH=1", "-sMAXIMUM_MEMORY=4gb", "-sINITIAL_MEMORY=256mb",
  "-sPTHREAD_POOL_SIZE=8", "-sPTHREAD_POOL_SIZE_STRICT=0",
  "-sSTACK_SIZE=1mb", "-sDEFAULT_PTHREAD_STACK_SIZE=1mb",
  "-sEXIT_RUNTIME=0", "-sNO_DISABLE_EXCEPTION_CATCHING",
  "-sENVIRONMENT=web,worker", "-sFETCH=1",
  "-sEXPORTED_FUNCTIONS=_main,_malloc,_free,_web_video_frame",
  "--use-port=contrib.glfw3", "--use-port=bzip2",
  "--shell-file", str(HERE / "shell.html"),
]

THIRD_PARTY = [
  *(IMGUI / f for f in ["imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp", "imgui_widgets.cpp",
                        "backends/imgui_impl_glfw.cpp", "backends/imgui_impl_opengl3.cpp"]),
  IMPLOT / "implot.cpp", IMPLOT / "implot_items.cpp",
  JSON11 / "json11.cpp",
  DEPS / "zstd/build/single_file_libs/zstddeclib.c",
]

CEREAL = sorted((OPENPILOT / "cereal/gen/cpp").glob("*.c++"))
COMMON = [OPENPILOT / "common" / f for f in ["util.cc"]]
REPLAY = [OPENPILOT / "tools/replay" / f for f in ["replay.cc", "logreader.cc", "seg_mgr.cc", "route.cc", "util.cc", "timeline.cc", "filereader.cc"]]
CORE = [CABANA / f for f in ["streams/abstractstream.cc", "streams/replaystream.cc",
                              "dbc/dbc.cc", "dbc/dbcfile.cc", "dbc/dbcmanager.cc",
                              "utils/export.cc", "utils/util.cc", "utils/strings.cc",
                              "commands.cc", "settings.cc", "routes.cc"]]
UI = sorted(p for d in ["ui", "ui/widgets", "ui/dialogs", "ui/chart", "ui/tools"] for p in (CABANA / d).glob("*.cc")
            if p.name != "main.cc")
WEB = sorted(HERE.glob("*.cc"))


def bootstrap_icons_src():
  import bootstrap_icons
  out = OBJ / "bootstrap_icons.cc"
  data = Path(bootstrap_icons.SVG_PATH).read_bytes()
  src = ["#include <cstddef>", "extern const unsigned char bootstrap_icons_svg[];", "extern const size_t bootstrap_icons_svg_len;",
         "const unsigned char bootstrap_icons_svg[] = {"]
  src += [",".join(str(b) for b in data[i:i + 32]) + "," for i in range(0, len(data), 32)]
  src += ["};", "const size_t bootstrap_icons_svg_len = sizeof(bootstrap_icons_svg);", ""]
  text = "\n".join(src)
  if not out.exists() or out.read_text() != text:
    out.write_text(text)
  return out, bootstrap_icons.TTF_PATH


def compile_one(src: Path, extra=()):
  obj = OBJ / (hashlib.sha1((str(src) + " ".join(CXXFLAGS + list(extra))).encode()).hexdigest()[:8] + "_" + src.stem + ".o")
  dep = obj.with_suffix(".d")
  if obj.exists() and dep.exists():
    deps = dep.read_text().replace("\\\n", " ").split(":", 1)[1].split()
    if all(Path(d).exists() and Path(d).stat().st_mtime <= obj.stat().st_mtime for d in deps):
      return obj, None
  flags = [f for f in CXXFLAGS if not f.startswith("-std=")] if src.suffix == ".c" else CXXFLAGS
  cmd = ["em++", *flags, *extra, *(f"-I{i}" for i in INCLUDES), "-MMD", "-MF", str(dep), "-c", str(src), "-o", str(obj)]
  r = subprocess.run(cmd, capture_output=True, text=True)
  return obj, (r.returncode, r.stderr)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--compile-only", action="store_true")
  ap.add_argument("-j", type=int, default=os.cpu_count())
  args = ap.parse_args()

  OBJ.mkdir(exist_ok=True)
  icons_src, icons_ttf = bootstrap_icons_src()
  sources = THIRD_PARTY + CEREAL + COMMON + REPLAY + CORE + UI + WEB + [icons_src]

  failed = False
  objs = []
  with ThreadPoolExecutor(args.j) as ex:
    for src, (obj, res) in zip(sources, ex.map(compile_one, sources), strict=True):
      objs.append(obj)
      if res and res[0] != 0:
        failed = True
        print(f"=== {src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}\n{res[1]}")
      elif res and res[1].strip():
        print(res[1])
  if failed:
    sys.exit(1)
  if args.compile_only:
    return

  DIST.mkdir(exist_ok=True)
  assets = HERE / "assets"
  shutil.rmtree(assets, ignore_errors=True)
  (assets / "fonts").mkdir(parents=True)
  for f in (OPENPILOT / "selfdrive/assets/fonts").glob("*.ttf"):
    shutil.copy(f, assets / "fonts" / f.name)
  shutil.copy(icons_ttf, assets / "fonts" / "bootstrap-icons.ttf")
  (assets / "dbc").mkdir()
  fingerprints = CABANA / "dbc/car_fingerprint_to_dbc.json"
  if not fingerprints.exists():
    subprocess.run([sys.executable, str(CABANA / "dbc/generate_dbc_json.py"), "--out", str(fingerprints)], check=True, cwd=ROOT)
  shutil.copy(fingerprints, assets / "dbc" / fingerprints.name)
  (assets / "opendbc").mkdir()
  for f in (ROOT / "opendbc_repo/opendbc/dbc").glob("*.dbc"):
    shutil.copy(f, assets / "opendbc" / f.name)

  cmd = ["em++", *LINKFLAGS, *map(str, objs), str(CAPNP / "lib/libcapnp.a"), str(CAPNP / "lib/libkj.a"),
         "--preload-file", f"{assets}/fonts@/fonts", "--preload-file", f"{assets}/opendbc@/opendbc",
         "--preload-file", f"{assets}/dbc@/dbc",
         "-o", str(DIST / "cabana.html")]
  subprocess.run(cmd, check=True)
  print(f"built {DIST / 'cabana.html'}")


if __name__ == "__main__":
  main()
