import subprocess, unittest, os, sys
from tinygrad.device import Device

class TestTinygradSlow(unittest.TestCase):
  def test_env_overwrite_default_device(self):
    subprocess.run([f'{Device.DEFAULT}=1 python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                    shell=True, check=True)
    subprocess.run([f'DISK=1 {Device.DEFAULT}=1 python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                    shell=True, check=True)
    subprocess.run([f'NPY=1 {Device.DEFAULT}=1 python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                    shell=True, check=True)

    if Device.DEFAULT != "CPU":
      # setting multiple devices fail
      with self.assertRaises(subprocess.CalledProcessError):
        subprocess.run([f'{Device.DEFAULT}=1 CPU=1 python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                        shell=True, check=True)

      # setting device via DEV
      subprocess.run([f'DEV={Device.DEFAULT.capitalize()} python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                      shell=True, check=True)
      subprocess.run([f'DEV={Device.DEFAULT.lower()} python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                      shell=True, check=True)
      subprocess.run([f'DEV={Device.DEFAULT.upper()} python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                      shell=True, check=True)

      with self.assertRaises(subprocess.CalledProcessError):
        subprocess.run([f'DEV={Device.DEFAULT} CPU=1 python3 -c "from tinygrad import Device; assert Device.DEFAULT == \\"{Device.DEFAULT}\\""'],
                        shell=True, check=True)

class TestRunAsModule(unittest.TestCase):
  def test_module_runs(self):
    p = subprocess.run([sys.executable, "-m", "tinygrad.device"],stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      env={**os.environ, "DEBUG": "1"}, timeout=40,)
    out = (p.stdout + p.stderr).decode()
    self.assertEqual(p.returncode, 0, msg=out)

if __name__ == '__main__':
  unittest.main()
