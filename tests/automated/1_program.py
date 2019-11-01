from .helpers import test_all_pandas, panda_connect_and_init

@test_all_pandas
@panda_connect_and_init
def test_recover(p):
  assert p.recover(timeout=30)

@test_all_pandas
@panda_connect_and_init
def test_flash(p):
  p.flash()
