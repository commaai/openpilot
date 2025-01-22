import unittest, subprocess, platform
from tinygrad.runtime.support.elf import elf_loader

class TestElfLoader(unittest.TestCase):
  def test_load_clang_jit_strtab(self):
    src = '''
      void relocation(int); // will be a jump to relocation (needed for .rela.text to exist)
      void test(int x) {
        relocation(x+1);
      }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode('utf-8'))
    _, sections, _ = elf_loader(obj)
    section_names = [sh.name for sh in sections]
    assert '.text' in section_names and '.rela.text' in section_names, str(section_names)

if __name__ == '__main__':
  unittest.main()
