import subprocess
from unittest.mock import Mock, PropertyMock, patch

import pytest

from openpilot.system.updated import updated


def written(params: Mock) -> list[str]:
  return [c.args[1] for c in params.put.call_args_list]


class TestProgressReporter:
  def test_percent_and_eta(self):
    params = Mock()
    with patch.object(updated.time, "monotonic") as t:
      t.return_value = 100
      p = updated.ProgressReporter(params, "updating OS")
      p.update(0.0)
      t.return_value = 110
      p.update(0.1)  # 10%/10s -> 90s left
      t.return_value = 110.5
      p.update(0.2)  # within a second of the last write
      t.return_value = 170
      p.update(0.8)  # 20s left
    assert written(params) == ["updating OS...", "updating OS... 0%", "updating OS... 10%, ~2 min left", "updating OS... 80%, <1 min left"]

  def test_restart_resets_eta(self):
    params = Mock()
    with patch.object(updated.time, "monotonic") as t:
      t.return_value = 0
      p = updated.ProgressReporter(params, "updating OS")
      p.update(0.5)
      t.return_value = 10
      p.update(0.1)  # retried from an earlier point, no rate yet
    assert written(params)[-1] == "updating OS... 10%"


class TestFetch:
  @pytest.fixture
  def repo(self, tmp_path):
    remote, local = tmp_path / "remote", tmp_path / "local"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "master", str(remote)], check=True)
    (remote / "big").write_bytes(subprocess.check_output(["head", "-c", "3000000", "/dev/urandom"]))
    subprocess.run([*git, "-C", str(remote), "add", "big"], check=True)
    subprocess.run([*git, "-C", str(remote), "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "init", "-q", str(local)], check=True)
    subprocess.run(["git", "-C", str(local), "remote", "add", "origin", f"file://{remote}"], check=True)
    with patch.object(updated, "OVERLAY_MERGED", str(local)):
      yield local

  def test_fetch(self, repo):
    params = Mock()
    output = updated.fetch_with_progress(params, "master")
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "FETCH_HEAD"])
    assert written(params)[0] == "downloading..."
    assert all(s == "downloading..." or s.endswith(" MB") for s in written(params))
    assert "%" not in output  # progress lines aren't logged

  def test_fetch_failure(self, repo):
    with pytest.raises(subprocess.CalledProcessError) as e:
      updated.fetch_with_progress(Mock(), "missing")
    assert "missing" in e.value.output


class TestUpdater:
  def test_check_uses_one_remote_request(self):
    updater = updated.Updater()
    updater.params = Mock()
    updater.params.get.return_value = "master"
    with (
      patch.object(updated, "setup_git_options"),
      patch.object(updated, "run", return_value="a" * 40 + "\trefs/heads/master\n") as run,
      patch.object(updater, "get_branch", return_value="master"),
      patch.object(updater, "get_commit_hash", return_value="a" * 40),
    ):
      updater.check_for_update()
    run.assert_called_once_with(["git", "ls-remote", "--heads", "origin"], updated.OVERLAY_MERGED)
    assert updater.has_internet
    assert updater.branches["master"] == "a" * 40

  def test_check_failure_clears_connectivity(self):
    updater = updated.Updater()
    updater._has_internet = True
    with patch.object(updated, "setup_git_options"), patch.object(updated, "run", side_effect=subprocess.CalledProcessError(1, "git")):
      with pytest.raises(subprocess.CalledProcessError):
        updater.check_for_update()
    assert not updater.has_internet

  @pytest.mark.parametrize("installed, available, ready, expected", [
    ("target", False, False, False),  # already installed
    ("old", False, True, False),  # downloaded, waiting for a reboot
    ("old", False, False, True),  # fetched, but finalizing was interrupted
    ("old", True, False, True),  # new remote commit
  ])
  def test_needs_fetch(self, installed, available, ready, expected):
    updater = updated.Updater()
    updater.params = Mock()
    updater.params.get.return_value = "master"
    updater.branches["master"] = "target"
    with (
      patch.object(updater, "get_commit_hash", return_value=installed),
      patch.object(updater, "get_branch", return_value="master"),
      patch.object(updated.Updater, "update_available", new_callable=PropertyMock, return_value=available),
      patch.object(updated.Updater, "update_ready", new_callable=PropertyMock, return_value=ready),
    ):
      assert updater.needs_fetch == expected
