from openpilot.common.swaglog import SwaglogRotatingFileHandler


def seed(base: str, indexes) -> None:
  for i in indexes:
    with open(f"{base}.{i:010}", "w") as f:
      f.write("x")


def existing(tmp_path) -> list[str]:
  return sorted(f.name for f in tmp_path.iterdir())


class TestSwaglogRotation:
  def test_rollover_deletes_oldest_first(self, tmp_path):
    base = str(tmp_path / "swaglog")
    seed(base, range(8))  # swaglog.0000000000 .. swaglog.0000000007, exceeds backup_count

    # restart over pre-existing files: __init__ rolls over, opening .8 and pruning to backup_count
    handler = SwaglogRotatingFileHandler(base, backup_count=4)
    try:
      assert existing(tmp_path) == [f"swaglog.{i:010}" for i in (5, 6, 7, 8)]

      # subsequent rollovers keep deleting oldest-first
      handler.doRollover()
      assert existing(tmp_path) == [f"swaglog.{i:010}" for i in (6, 7, 8, 9)]
      handler.doRollover()
      assert existing(tmp_path) == [f"swaglog.{i:010}" for i in (7, 8, 9, 10)]
    finally:
      handler.close()

    # a second restart over the survivors keeps the same invariant
    handler = SwaglogRotatingFileHandler(base, backup_count=4)
    try:
      assert existing(tmp_path) == [f"swaglog.{i:010}" for i in (8, 9, 10, 11)]
    finally:
      handler.close()
