import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpilot.selfdrive.modeld.helpers import chestnut_compiled, modeld_pkl_path
from openpilot.selfdrive.modeld.precompile_modeld import compiler_hashes


class TestPrecompiledModel(unittest.TestCase):
  def test_compiler_sources_match(self):
    metadata = json.loads(modeld_pkl_path(True).with_suffix('.json').read_text())
    self.assertEqual(compiler_hashes(), metadata['compiler_sha256'], 'Recompile the big model after changing its compiler sources or tinygrad')

  def test_requires_complete_download(self):
    with tempfile.TemporaryDirectory() as directory, patch('openpilot.selfdrive.modeld.helpers.MODELS_DIR', Path(directory)):
      path = modeld_pkl_path(True)
      chunks = [Path(f'{path}.chunk01of02'), Path(f'{path}.chunk02of02')]
      data = [b'compiled model', b'weights']
      path.with_suffix('.json').write_text(json.dumps({'chunks': {c.name: len(d) for c, d in zip(chunks, data, strict=True)}}))
      self.assertFalse(chestnut_compiled())
      Path(f'{path}.chunkmanifest').write_text('2')
      self.assertFalse(chestnut_compiled())
      for chunk, content in zip(chunks, data, strict=True):
        chunk.write_bytes(content)
      self.assertTrue(chestnut_compiled())
      chunks[0].write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
      self.assertFalse(chestnut_compiled())
      chunks[0].write_bytes(data[0][:-1])
      self.assertFalse(chestnut_compiled())
      chunks[0].write_bytes(data[0])
      chunks[1].unlink()
      self.assertFalse(chestnut_compiled())


if __name__ == '__main__':
  unittest.main()
