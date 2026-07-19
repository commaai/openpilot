import subprocess

from scripts.lint.check_added_large_files import check_added_large_files
from scripts.lint.check_shebang_scripts_are_executable import check_shebang_scripts_are_executable


def git(cwd, *args):
  subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def test_check_added_large_files(tmp_path, monkeypatch, capsys):
  git(tmp_path, "init")
  (tmp_path / "small").write_bytes(b"x" * 1024)
  (tmp_path / "large").write_bytes(b"x" * 1025)
  monkeypatch.chdir(tmp_path)

  assert check_added_large_files(["small", "large"], 1) == 1
  assert capsys.readouterr().out == "large (2 KB) exceeds 1 KB.\n"


def test_check_added_large_files_ignores_lfs(tmp_path, monkeypatch):
  git(tmp_path, "init")
  (tmp_path / ".gitattributes").write_text("large filter=lfs\n")
  (tmp_path / "large").write_bytes(b"x" * 1025)
  monkeypatch.chdir(tmp_path)

  assert check_added_large_files(["large"], 1) == 0


def test_check_shebang_scripts_are_executable(tmp_path, monkeypatch, capsys):
  git(tmp_path, "init")
  executable = tmp_path / "executable"
  executable.write_text("#!/bin/sh\n")
  executable.chmod(0o755)
  non_executable = tmp_path / "non executable"
  non_executable.write_text("#!/bin/sh\n")
  git(tmp_path, "add", "executable", "non executable")
  monkeypatch.chdir(tmp_path)

  assert check_shebang_scripts_are_executable(["executable", "non executable"]) == 1
  assert "non executable: has a shebang but is not marked executable!" in capsys.readouterr().err


def test_check_shebang_scripts_allows_non_scripts(tmp_path, monkeypatch):
  git(tmp_path, "init")
  filename = tmp_path / "README"
  filename.write_text("not a script\n")
  git(tmp_path, "add", "README")
  monkeypatch.chdir(tmp_path)

  assert check_shebang_scripts_are_executable(["README"]) == 0
