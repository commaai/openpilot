import unittest, subprocess, platform
from tinygrad.runtime.ops_cpu import ClangJITCompiler
from tinygrad.runtime.support.elf import elf_loader

class TestElfLoader(unittest.TestCase):
  def test_load_clang_jit_strtab(self):
    src = '''
      int something; // will be a load from a relocation (needed for .rela.text to exist)
      int test(int x) {
        return something + x;
      }
    '''
    args = ('-x', 'c', '-c', '-target', f'{platform.machine()}-none-unknown-elf', '-march=native', '-fPIC', '-O2', '-ffreestanding', '-nostdlib')
    obj = subprocess.check_output(('clang',) + args + ('-', '-o', '-'), input=src.encode('utf-8'))
    _, sections, _ = elf_loader(obj)
    section_names = [sh.name for sh in sections]
    assert '.text' in section_names and '.rela.text' in section_names, str(section_names)
  def test_clang_jit_compiler_external_raise(self):
    src = '''
      int evil_external_function(int);
      int test(int x) {
        return evil_external_function(x+2)*2;
      }
    '''
    with self.assertRaisesRegex(RuntimeError, 'evil_external_function'):
      ClangJITCompiler().compile(src)

if __name__ == '__main__':
  unittest.main()
