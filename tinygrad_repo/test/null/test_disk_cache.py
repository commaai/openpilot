import unittest
import pickle
from tinygrad.helpers import diskcache_get, diskcache_put, diskcache, CACHELEVEL

def remote_get(table,q,k): q.put(diskcache_get(table, k))
def remote_put(table,k,v): diskcache_put(table, k, v)

@unittest.skipIf(CACHELEVEL < 1, "disk cache is off")
class DiskCache(unittest.TestCase):
  def test_putget(self):
    table = "test_putget"
    diskcache_put(table, "hello", "world")
    self.assertEqual(diskcache_get(table, "hello"), "world")
    diskcache_put(table, "hello", "world2")
    self.assertEqual(diskcache_get(table, "hello"), "world2")

  def test_putcomplex(self):
    table = "test_putcomplex"
    diskcache_put(table, "k", ("complex", 123, "object"))
    ret = diskcache_get(table, "k")
    self.assertEqual(ret, ("complex", 123, "object"))

  def test_getotherprocess(self):
    table = "test_getotherprocess"
    from multiprocessing import Process, Queue
    diskcache_put(table, "k", "getme")
    q = Queue()
    p = Process(target=remote_get, args=(table,q,"k"))
    p.start()
    p.join()
    self.assertEqual(q.get(), "getme")

  def test_putotherprocess(self):
    table = "test_putotherprocess"
    from multiprocessing import Process
    p = Process(target=remote_put, args=(table,"k", "remote"))
    p.start()
    p.join()
    self.assertEqual(diskcache_get(table, "k"), "remote")

  def test_no_table(self):
    self.assertIsNone(diskcache_get("faketable", "k"))

  def test_ret(self):
    table = "test_ret"
    self.assertEqual(diskcache_put(table, "key", ("vvs",)), ("vvs",))

  def test_non_str_key(self):
    table = "test_non_str_key"
    diskcache_put(table, 4, 5)
    self.assertEqual(diskcache_get(table, 4), 5)
    self.assertEqual(diskcache_get(table, "4"), 5)

  def test_decorator(self):
    calls = 0
    @diskcache
    def hello(x):
      nonlocal calls
      calls += 1
      return "world"+x
    self.assertEqual(hello("bob"), "worldbob")
    self.assertEqual(hello("billy"), "worldbilly")
    kcalls = calls
    self.assertEqual(hello("bob"), "worldbob")
    self.assertEqual(hello("billy"), "worldbilly")
    self.assertEqual(kcalls, calls)

  def test_dict_key(self):
    table = "test_dict_key"
    fancy_key = {"hello": "world", "goodbye": 7, "good": True, "pkl": pickle.dumps("cat")}
    fancy_key2 = {"hello": "world", "goodbye": 8, "good": True, "pkl": pickle.dumps("cat")}
    fancy_key3 = {"hello": "world", "goodbye": 8, "good": True, "pkl": pickle.dumps("dog")}
    diskcache_put(table, fancy_key, 5)
    self.assertEqual(diskcache_get(table, fancy_key), 5)
    diskcache_put(table, fancy_key2, 8)
    self.assertEqual(diskcache_get(table, fancy_key2), 8)
    self.assertEqual(diskcache_get(table, fancy_key), 5)
    self.assertEqual(diskcache_get(table, fancy_key3), None)

  def test_table_name(self):
    table = "test_gfx1010:xnack-"
    diskcache_put(table, "key", "test")
    self.assertEqual(diskcache_get(table, "key"), "test")

if __name__ == "__main__":
  unittest.main()
