import fcntl
import multiprocessing
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest

from openpilot.common.shm_queue import ShmQueue


def produce(path, producer, connection):
  queue = ShmQueue(path, 4096)
  accepted = []
  for i in range(1000):
    data = f'{producer}:{i}'.encode() * 10
    if queue.send(data):
      accepted.append(data)
    time.sleep(0.0001)
  queue.close()
  connection.send(accepted)
  connection.close()


def die_mid_write(path, connection):
  queue = ShmQueue(path, 4096)
  fcntl.flock(queue.fd, fcntl.LOCK_EX)
  _, write = struct.unpack_from('<QQ', queue.mem)
  queue._copy(write, struct.pack('<I', 100) + b'partial')
  connection.send(True)
  time.sleep(60)


class TestShmQueue(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.path = str(Path(self.directory.name) / 'queue')

  def queue(self, capacity=4096):
    queue = ShmQueue(self.path, capacity)
    self.addCleanup(queue.close)
    return queue

  def test_wrap_full_and_empty(self):
    queue = self.queue(64)
    reader = self.queue(64)
    for _ in range(100):
      self.assertIsNone(reader.receive())
      self.assertTrue(queue.send(b'a' * 30))
      self.assertTrue(queue.send(b'b' * 20))
      self.assertFalse(queue.send(b'full'))
      self.assertEqual(reader.receive(), b'a' * 30)
      self.assertEqual(reader.receive(), b'b' * 20)
    self.assertFalse(queue.send(b'x' * 60))
    self.assertTrue(queue.send(b'x' * 59))
    self.assertEqual(reader.receive(), b'x' * 59)
    self.assertTrue(queue.send(b''))
    self.assertEqual(reader.receive(), b'')

  def test_large_messages_and_reopen(self):
    queue = ShmQueue(self.path)
    message = b'a' * (3 * 1024 * 1024)
    try:
      for _ in range(10):
        self.assertTrue(queue.send(message))
    finally:
      queue.close()
    reader = self.queue(ShmQueue.CAPACITY)
    for _ in range(10):
      self.assertEqual(reader.receive(), message)
    self.assertIsNone(reader.receive())

  def test_contention_and_writer_death(self):
    queue = self.queue()
    self.assertTrue(queue.send(b'committed'))
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    self.addCleanup(parent.close)
    self.addCleanup(child.close)
    process = ctx.Process(target=die_mid_write, args=(self.path, child))
    process.start()
    try:
      self.assertTrue(parent.poll(10))
      self.assertTrue(parent.recv())
      self.assertFalse(queue.send(b'busy'))
      with self.assertRaises(BlockingIOError):
        ShmQueue(self.path, 4096, blocking=False)
    finally:
      process.kill()
      process.join(10)
    self.assertEqual(queue.receive(), b'committed')
    self.assertIsNone(queue.receive())
    self.assertTrue(queue.send(b'after crash'))
    self.assertEqual(queue.receive(), b'after crash')

  def test_concurrent_producers(self):
    reader = self.queue()
    ctx = multiprocessing.get_context('spawn')
    processes, connections = [], []
    for i in range(4):
      parent, child = ctx.Pipe()
      self.addCleanup(parent.close)
      self.addCleanup(child.close)
      process = ctx.Process(target=produce, args=(self.path, i, child))
      process.start()
      processes.append(process)
      connections.append(parent)
    received = []
    deadline = time.monotonic() + 15
    try:
      while not all(c.poll() for c in connections) and time.monotonic() < deadline:
        data = reader.receive()
        if data is not None:
          received.append(data)
      self.assertTrue(all(c.poll() for c in connections))
      accepted = [c.recv() for c in connections]
      while (data := reader.receive()) is not None:
        received.append(data)
      self.assertCountEqual(received, [data for group in accepted for data in group])
      for i, group in enumerate(accepted):
        self.assertTrue(group)
        self.assertEqual([data for data in received if data.startswith(f'{i}:'.encode())], group)
    finally:
      for process in processes:
        process.join(2)
        if process.is_alive():
          process.kill()
          process.join()

  def test_cpp_interoperability_and_fork(self):
    compiler = shutil.which('c++')
    if compiler is None:
      self.skipTest('C++ compiler required')
    source = Path(self.directory.name) / 'producer.cc'
    binary = source.with_suffix('')
    source.write_text('''
#include "openpilot/common/shm_queue.h"
#include <sys/wait.h>
#include <iterator>
#include <iostream>
int main(int argc, char **argv) {
  ShmQueue queue(argv[1], 8 * 1024 * 1024);
  std::string data{std::istreambuf_iterator<char>(std::cin), {}};
  if (!queue.send(data)) return 1;
  pid_t child = fork();
  if (child == 0) _exit(queue.send("child") ? 0 : 2);
  if (child < 0) return 3;
  int status;
  waitpid(child, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 4;
}
''')
    root = Path(__file__).resolve().parents[3]
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Werror', '-I', str(root), str(source), '-o', str(binary)], check=True)
    queue = self.queue(8 * 1024 * 1024)
    data = bytes(range(256)) * (3 * 1024 * 4)
    # Repeated runs exercise wrapped C++ payloads.
    for _ in range(5):
      self.assertTrue(queue.send(b'python'))
      subprocess.run([str(binary), self.path], input=data, check=True)
      self.assertEqual(queue.receive(), b'python')
      self.assertEqual(queue.receive(), data)
      self.assertEqual(queue.receive(), b'child')
      self.assertIsNone(queue.receive())


if __name__ == '__main__':
  unittest.main()
