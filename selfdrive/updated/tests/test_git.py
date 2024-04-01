import contextlib
from openpilot.selfdrive.updated.tests.test_base import BaseUpdateTest, run, update_release


class TestUpdateDGitStrategy(BaseUpdateTest):
  def update_remote_release(self, release):
    update_release(self.remote_dir, release, *self.MOCK_RELEASES[release])
    run(["git", "add", "."], cwd=self.remote_dir)
    run(["git", "commit", "-m", f"openpilot release {release}"], cwd=self.remote_dir)

  def setup_remote_release(self, release):
    run(["git", "init"], cwd=self.remote_dir)
    run(["git", "checkout", "-b", release], cwd=self.remote_dir)
    self.update_remote_release(release)

  def setup_basedir_release(self, release):
    super().setup_basedir_release(release)
    run(["git", "clone", "-b", release, self.remote_dir, self.basedir])

  @contextlib.contextmanager
  def additional_context(self):
    yield
