#!/usr/bin/env python3
import contextlib
import functools
import multiprocessing
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import usb1

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import HARDWARE
from openpilot.common.mock import mock_messages
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.helpers import chestnut_compiled, chestnut_present, modeld_pkl_path
from openpilot.selfdrive.test.helpers import processes_context
from openpilot.system.manager.process import launcher
from openpilot.system.manager.process_config import managed_processes
from opendbc.car.car_helpers import get_demo_car_params


def fault_launcher(module, name, fault, trigger, injected, models):
  # Patch only the child: normal model loading, camera input and inference still run on hardware.
  from openpilot.selfdrive.modeld import modeld

  original_init, original_run, original_warmup = modeld.ModelState.__init__, modeld.ModelState.run, modeld.ModelState.warmup

  def init(self, width, height, chestnut):
    if chestnut and fault in ('load_usb', 'load_timeout'):
      injected.set()
      if fault == 'load_timeout':
        time.sleep(modeld.BIG_MODEL_TIMEOUT + 10)
      raise usb1.USBErrorNoDevice()
    original_init(self, width, height, chestnut)

  def warmup(self):
    if self.chestnut and fault == 'warmup_usb':
      injected.set()
      raise usb1.USBErrorNoDevice()
    return original_warmup(self)

  def run(self, *args, **kwargs):
    if self.chestnut and trigger.is_set() and not injected.is_set():
      injected.set()
      if fault == 'run_usb':
        raise usb1.USBErrorNoDevice()
      if fault == 'run_timeout':
        raise RuntimeError('injected GPU wait timeout')
      if fault in ('nan', 'inf'):
        run_model = self.run_model

        def nonfinite(**inputs):
          outs, = run_model(**inputs)
          values = outs.numpy().copy()
          values.flat[0] = np.nan if fault == 'nan' else np.inf
          return (Mock(numpy=Mock(return_value=values)),)

        with patch.object(self, 'run_model', nonfinite):
          return original_run(self, *args, **kwargs)
    return original_run(self, *args, **kwargs)

  with contextlib.ExitStack() as stack:
    stack.enter_context(patch.object(modeld.ModelState, '__init__', init))
    stack.enter_context(patch.object(modeld.ModelState, 'warmup', warmup))
    stack.enter_context(patch.object(modeld.ModelState, 'run', run))
    if models is not None:
      stack.enter_context(patch.object(modeld, 'modeld_pkl_path', lambda chestnut: Path(models) / modeld_pkl_path(chestnut).name))
      stack.enter_context(patch.object(modeld, 'chestnut_compiled', lambda: (Path(models) / 'big_driving_tinygrad.pkl.chunkmanifest').is_file()))
    launcher(module, name)


@unittest.skipUnless(HARDWARE.get_device_type() == 'mici', 'requires MICI')
class TestChestnutFaults(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  def setUp(self):
    assert chestnut_present() and chestnut_compiled(), 'Chestnut hardware and compiled big model are required'
    self.params = Params()
    self.params.put('CarParams', get_demo_car_params().to_bytes(), block=True)
    self.pm = messaging.PubMaster(['deviceState'])

  def tick(self, sm):
    msg = messaging.new_message('deviceState')
    msg.deviceState.deviceType = HARDWARE.get_device_type()
    self.pm.send('deviceState', msg)
    sm.update(100)

  @contextlib.contextmanager
  def model(self, fault='', models=None):
    trigger, injected = multiprocessing.Event(), multiprocessing.Event()
    proc = managed_processes['modeld']
    sm = messaging.SubMaster(['modelV2'])
    target = functools.partial(fault_launcher, fault=fault, trigger=trigger, injected=injected, models=models)
    with patch.object(proc, 'launcher', target), processes_context(['modeld'], ignore_stopped=['modeld']):
      yield sm, proc, trigger, injected

  def frames(self, sm, proc, big, count=40, timeout=90):
    deadline = time.monotonic() + timeout
    frames = 0
    last_frame = -1
    while frames < count:
      assert time.monotonic() < deadline, f'no sustained valid model output: {big=}, {frames=}'
      assert proc.proc.is_alive(), f'modeld exited: {proc.proc.exitcode}'
      self.tick(sm)
      if not sm.updated['modelV2']:
        continue
      m = sm['modelV2']
      assert sm.valid['modelV2'], 'invalid model output'
      assert m.frameId > last_frame, 'repeated or reordered model frame'
      assert all(np.isfinite(v).all() for v in (m.position.x, m.position.y, m.position.z,
                                               [m.action.desiredAcceleration, m.action.desiredCurvature]))
      last_frame = m.frameId
      if m.big == big:
        frames += 1
      elif frames:
        raise AssertionError('model switched back after fallback')
    assert not self.params.get_bool('ChestnutLoading')
    assert self.params.get_bool('ChestnutActive') == big

  def recovery(self):
    with self.model() as (sm, proc, _, _):
      self.frames(sm, proc, True)

  @mock_messages(['deviceMotion'])
  def test_load_failures(self):
    with processes_context(['camerad', 'calibrationd']):
      for fault in ('load_usb', 'warmup_usb', 'load_timeout'):
        with self.subTest(fault=fault):
          with self.model(fault) as (sm, proc, _, injected):
            self.frames(sm, proc, False)
            assert injected.is_set(), 'fault was not exercised'
          self.recovery()

  @mock_messages(['deviceMotion'])
  def test_inference_failures(self):
    with processes_context(['camerad', 'calibrationd']):
      for fault in ('run_usb', 'run_timeout', 'nan', 'inf'):
        with self.subTest(fault=fault):
          with self.model(fault) as (sm, proc, trigger, injected):
            self.frames(sm, proc, True)
            trigger.set()
            self.frames(sm, proc, False, timeout=15)
            assert injected.is_set(), 'fault was not exercised'
            # A recovered link must not silently switch back to the big model mid-drive.
            self.frames(sm, proc, False)
          self.recovery()

  @contextlib.contextmanager
  def damaged_models(self, fault):
    # Keep build artifacts untouched, including on interruption of the test.
    with tempfile.TemporaryDirectory() as tmp:
      models = Path(tmp)
      for p in modeld_pkl_path(True).parent.glob('*driving_tinygrad.pkl*'):
        (models / p.name).symlink_to(p)
      manifest = models / 'big_driving_tinygrad.pkl.chunkmanifest'
      chunk = sorted(models.glob('big_driving_tinygrad.pkl.chunk[0-9]*'))[0]
      if fault in ('missing_manifest', 'invalid_manifest'):
        manifest.unlink()
        if fault == 'invalid_manifest':
          manifest.write_text('invalid')
      else:
        if fault == 'truncated_buffer':
          chunk = sorted(models.glob('big_driving_tinygrad.pkl.chunk[0-9]*'))[-1]
        chunk.unlink()
        if fault == 'corrupt_pickle':
          chunk.write_bytes(b'\x01\x00\x00\x00\x00\x00\x00\x00!')
        elif fault == 'truncated_buffer':
          shutil.copyfile(modeld_pkl_path(True).parent / chunk.name, chunk)
          with chunk.open('r+b') as f:
            f.truncate(chunk.stat().st_size - 1)
      yield tmp

  @mock_messages(['deviceMotion'])
  def test_model_files(self):
    with processes_context(['camerad', 'calibrationd']):
      for fault in ('missing_manifest', 'invalid_manifest', 'missing_chunk', 'corrupt_pickle', 'truncated_buffer'):
        with self.subTest(fault=fault):
          with self.damaged_models(fault) as models, self.model(models=models) as (sm, proc, _, _):
            self.frames(sm, proc, False)
          self.recovery()

  @mock_messages(['deviceMotion'])
  def test_no_working_model(self):
    with processes_context(['camerad', 'calibrationd']), self.damaged_models('corrupt_pickle') as models:
      for path in Path(models).glob('driving_tinygrad.pkl*'):
        path.unlink()
      with self.model(models=models) as (sm, proc, _, _):
        deadline = time.monotonic() + 30
        while proc.proc.is_alive():
          assert time.monotonic() < deadline, 'modeld hung with no working model'
          self.tick(sm)
          assert not sm.updated['modelV2'], 'published a model without a working model file'
        assert proc.proc.exitcode != 0
        assert not self.params.get_bool('ChestnutActive')
    with processes_context(['camerad', 'calibrationd']):
      self.recovery()

  @mock_messages(['deviceMotion'])
  def test_stop_during_load(self):
    with processes_context(['camerad', 'calibrationd']):
      for _ in range(3):
        with self.model('load_timeout') as (sm, proc, _, injected):
          deadline = time.monotonic() + 20
          while not injected.is_set():
            assert time.monotonic() < deadline, 'model load did not start'
            assert proc.proc.is_alive()
            self.tick(sm)
          assert self.params.get_bool('ChestnutLoading')
        # The normal process stop must reap the loader along with modeld.
        assert proc.proc is None
        self.recovery()


if __name__ == '__main__':
  unittest.main()
