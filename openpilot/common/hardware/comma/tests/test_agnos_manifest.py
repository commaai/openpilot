import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from openpilot.common.hardware.comma import agnos


class TestAgnosManifest(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.manifest = Path(self.temp.name) / 'agnos.json'
    self.log = Mock()
    self.verify = self.enterContext(patch.object(agnos, 'verify_partition', return_value=True))
    self.flash = self.enterContext(patch.object(agnos, 'flash_partition'))
    self.clear = self.enterContext(patch.object(agnos, 'clear_partition_hash'))
    self.run = self.enterContext(patch.object(agnos.subprocess, 'run'))
    self.check_output = self.enterContext(patch.object(agnos.subprocess, 'check_output', return_value='lun as boot lun'))

  def assert_no_partition_actions(self):
    for action in (self.verify, self.flash, self.clear, self.run, self.check_output):
      action.assert_not_called()

  def test_invalid_manifests(self):
    operations = (agnos.verify_agnos_update,
                  lambda path, slot: agnos.flash_agnos_update(path, slot, self.log),
                  lambda path, slot: agnos.swap(path, slot, self.log))
    for value in ([], {}, None, 0, 'invalid'):
      self.manifest.write_text(json.dumps(value))
      for operation in operations:
        with self.subTest(manifest=value, operation=operation):
          with self.assertRaises(ValueError):
            operation(str(self.manifest), 1)
          self.assert_no_partition_actions()

  def test_invalid_json(self):
    self.manifest.write_text('{')
    for operation in (agnos.verify_agnos_update,
                      lambda path, slot: agnos.flash_agnos_update(path, slot, self.log),
                      lambda path, slot: agnos.swap(path, slot, self.log)):
      with self.subTest(operation=operation):
        with self.assertRaises(json.JSONDecodeError):
          operation(str(self.manifest), 1)
        self.assert_no_partition_actions()

  def write_valid_manifest(self):
    entries = [{'name': 'boot', 'url': 'https://example.com/boot.img.xz'},
               {'name': 'system', 'url': 'https://example.com/system.img.xz', 'full_check': True}]
    self.manifest.write_text(json.dumps(entries))
    return entries

  def test_valid_verification(self):
    entries = self.write_valid_manifest()
    self.assertTrue(agnos.verify_agnos_update(str(self.manifest), 1))
    self.assertEqual(self.verify.call_args_list, [call(1, entry) for entry in entries])
    self.verify.reset_mock()
    self.verify.return_value = False
    self.assertFalse(agnos.verify_agnos_update(str(self.manifest), 1))
    self.verify.assert_called_once_with(1, entries[0])
    for action in (self.flash, self.clear, self.run, self.check_output):
      action.assert_not_called()

  def test_valid_flash(self):
    entries = self.write_valid_manifest()
    agnos.flash_agnos_update(str(self.manifest), 1, self.log)
    self.run.assert_called_once_with('abctl --set_unbootable 1', shell=True)
    self.assertEqual(self.flash.call_args_list, [call(1, entry, self.log, False) for entry in entries])
    self.verify.assert_not_called()
    self.clear.assert_not_called()
    self.check_output.assert_not_called()

  def test_valid_swap(self):
    entries = self.write_valid_manifest()
    agnos.swap(str(self.manifest), 1, self.log)
    self.clear.assert_called_once_with(1, entries[0])
    self.check_output.assert_called_once_with('abctl --set_active 1', shell=True, stderr=agnos.subprocess.STDOUT, encoding='utf8')
    self.verify.assert_not_called()
    self.flash.assert_not_called()
    self.run.assert_not_called()


if __name__ == '__main__':
  unittest.main()
