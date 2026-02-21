#!/usr/bin/env python3
"""openpilot installer — self-extracting zipapp with raylib UI."""
import os
import sys
import zipfile

# ---------------------------------------------------------------------------
# Phase 1: extract native .so libs from the zip; read assets into memory.
# Python cannot load .so extensions from inside a zip archive.
# ---------------------------------------------------------------------------
CACHE = "/tmp/.installer_libs"
os.makedirs(CACHE, exist_ok=True)

FONTS = {}       # basename -> bytes
CONTINUE_SH = b""

with zipfile.ZipFile(sys.argv[0]) as zf:
    for name in zf.namelist():
        if name.endswith(".ttf"):
            FONTS[os.path.basename(name)] = zf.read(name)
        elif name == "assets/continue_openpilot.sh":
            CONTINUE_SH = zf.read(name)
        elif (name.startswith(("raylib/", "pyray/", "cffi/"))
                or name.endswith(".so")):
            dest = os.path.join(CACHE, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(zf.read(name))

sys.path.insert(0, CACHE)

# ---------------------------------------------------------------------------
# Phase 2: imports (native libs are now on sys.path)
# ---------------------------------------------------------------------------
import re        # noqa: E402
import subprocess  # noqa: E402
import time       # noqa: E402

import pyray as rl  # noqa: E402

import config  # noqa: E402  — generated at build time (BRANCH, INTERNAL, GIT_URL)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GIT_SSH_URL = "git@github.com:commaai/openpilot.git"
CONTINUE_PATH = "/data/continue.sh"
INSTALL_PATH = "/data/openpilot"
VALID_CACHE_PATH = "/data/.openpilot_cache"
TMP_INSTALL_PATH = "/data/tmppilot"
FONT_SIZE = 120

# globals set by detect_device()
font_inter = font_roman = font_display = None
tici_device = False
device_type = None  # "tici" | "tizi" | "mici" | None


# ---------------------------------------------------------------------------
# Device detection & branch migration
# ---------------------------------------------------------------------------
def detect_device():
    global tici_device, device_type
    if not os.path.isfile("/TICI"):
        device_type = None
        tici_device = False
        return
    try:
        model = open("/sys/firmware/devicetree/base/model").read().strip("\x00").lower()
    except Exception:
        model = ""
    if "tizi" in model:
        device_type = "tizi"
    elif "mici" in model:
        device_type = "mici"
    else:
        device_type = "tici"
    tici_device = device_type in ("tici", "tizi")


def branch_migration():
    branch = config.BRANCH
    tici_prebuilt = ["release3", "release-tizi", "release3-staging", "nightly", "nightly-dev"]
    if device_type == "tici":
        if branch in tici_prebuilt:
            return "release-tici"
        if branch == "master":
            return "master-tici"
    elif device_type == "tizi":
        if branch == "release3":
            return "release-tizi"
        if branch == "release3-staging":
            return "release-tizi-staging"
    elif device_type == "mici":
        if branch == "release3":
            return "release-mici"
        if branch == "release3-staging":
            return "release-mici-staging"
    return branch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd):
    ret = os.system(cmd)
    assert ret == 0, f"Command failed with {ret}: {cmd}"


def system_time_valid():
    return time.time() > 1704067200  # 2024-01-01 UTC


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_progress(progress):
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)
    if tici_device:
        rl.draw_text_ex(font_inter, "Installing...", rl.Vector2(150, 290), 110, 0, rl.WHITE)
        bar = rl.Rectangle(150, 570, rl.get_screen_width() - 300, 72)
        rl.draw_rectangle_rec(bar, rl.Color(41, 41, 41, 255))
        progress = max(0, min(progress, 100))
        bar_fill = rl.Rectangle(150, 570, bar.width * progress / 100.0, 72)
        rl.draw_rectangle_rec(bar_fill, rl.Color(70, 91, 234, 255))
        rl.draw_text_ex(font_inter, f"{progress}%", rl.Vector2(150, 670), 85, 0, rl.WHITE)
    else:
        rl.draw_text_ex(font_display, "installing", rl.Vector2(8, 10), 82, 0, rl.WHITE)
        rl.draw_text_ex(font_roman, f"{progress}%",
                        rl.Vector2(6, rl.get_screen_height() - 128 + 18), 128, 0,
                        rl.Color(255, 255, 255, int(255 * 0.9 * 0.35)))
    rl.end_drawing()


def finish_install():
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)
    if tici_device:
        m = "Finishing install..."
        text_width = rl.measure_text(m, FONT_SIZE)
        rl.draw_text_ex(font_display, m,
                        rl.Vector2((rl.get_screen_width() - text_width) / 2 + FONT_SIZE,
                                   (rl.get_screen_height() - FONT_SIZE) / 2),
                        FONT_SIZE, 0, rl.WHITE)
    else:
        rl.draw_text_ex(font_display, "finishing setup", rl.Vector2(8, 10), 82, 0, rl.WHITE)
    rl.end_drawing()
    time.sleep(60)


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------
def execute_git_command(cmd):
    stages = [
        ("Receiving objects: ", 91),
        ("Resolving deltas: ", 2),
        ("Updating files: ", 7),
    ]
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        data = os.read(proc.stdout.fileno(), 512)
        if not data:
            break
        for line in re.split(rb'[\r\n]+', data):
            line = line.decode(errors='replace')
            base = 0
            for prefix, weight in stages:
                if prefix in line:
                    m = re.search(r'(\d{1,3})%', line)
                    if m:
                        percent = int(m.group(1))
                        render_progress(base + int(percent / 100.0 * weight))
                    break
                base += weight
    return proc.wait()


def fresh_clone(branch):
    cmd = (f"git clone --progress {config.GIT_URL} -b {branch} "
           f"--depth=1 --recurse-submodules {TMP_INSTALL_PATH} 2>&1")
    return execute_git_command(cmd)


def cached_fetch(cache, branch):
    run(f"cp -rp {cache} {TMP_INSTALL_PATH}")
    run(f"cd {TMP_INSTALL_PATH} && git remote set-branches --add origin {branch}")
    render_progress(10)
    return execute_git_command(
        f"cd {TMP_INSTALL_PATH} && git fetch --progress origin {branch} 2>&1")


def do_install(branch):
    while not system_time_valid():
        time.sleep(0.5)
    run(f"rm -rf {TMP_INSTALL_PATH}")
    if os.path.exists(INSTALL_PATH) and os.path.exists(VALID_CACHE_PATH):
        return cached_fetch(INSTALL_PATH, branch)
    return fresh_clone(branch)


def clone_finished(exit_code, branch):
    assert exit_code == 0
    render_progress(100)

    os.chdir(TMP_INSTALL_PATH)
    run(f"git checkout {branch}")
    run(f"git reset --hard origin/{branch}")
    run("git submodule update --init")

    run(f"rm -f {VALID_CACHE_PATH}")
    run(f"rm -rf {INSTALL_PATH}")
    run(f"mv {TMP_INSTALL_PATH} {INSTALL_PATH}")

    if config.INTERNAL:
        run("mkdir -p /data/params/d/")
        ssh_keys = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMX2kU8eBZyEWmbq0tjMPxksWWVuIV/5l64GabcYbdpI"
        for key, value in {"SshEnabled": "1", "RecordFrontLock": "1", "GithubSshKeys": ssh_keys}.items():
            with open(f"/data/params/d/{key}", "w") as f:
                f.write(value)
        run(f"cd {INSTALL_PATH} && "
            f"git remote set-url origin --push {GIT_SSH_URL} && "
            f'git config --replace-all remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"')

    with open("/data/continue.sh.new", "wb") as f:
        f.write(CONTINUE_SH)
    run("chmod +x /data/continue.sh.new")
    run(f"mv /data/continue.sh.new {CONTINUE_PATH}")

    finish_install()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global font_inter, font_roman, font_display

    detect_device()

    if tici_device:
        rl.init_window(2160, 1080, "Installer")
    else:
        rl.init_window(536, 240, "Installer")

    font_inter = rl.load_font_from_memory(".ttf", FONTS["inter-ascii.ttf"], len(FONTS["inter-ascii.ttf"]), FONT_SIZE, None, 0)
    font_roman = rl.load_font_from_memory(".ttf", FONTS["Inter-Light.ttf"], len(FONTS["Inter-Light.ttf"]), FONT_SIZE, None, 0)
    font_display = rl.load_font_from_memory(".ttf", FONTS["Inter-Bold.ttf"], len(FONTS["Inter-Bold.ttf"]), FONT_SIZE, None, 0)
    rl.set_texture_filter(font_inter.texture, rl.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_filter(font_roman.texture, rl.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_filter(font_display.texture, rl.TEXTURE_FILTER_BILINEAR)

    branch = branch_migration()

    if os.path.exists(CONTINUE_PATH):
        finish_install()
    else:
        render_progress(0)
        result = do_install(branch)
        clone_finished(result, branch)

    rl.close_window()
    rl.unload_font(font_inter)
    rl.unload_font(font_roman)
    rl.unload_font(font_display)


main()
