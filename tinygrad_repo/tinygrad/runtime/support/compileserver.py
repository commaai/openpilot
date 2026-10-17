import ast, struct, sys
from tinygrad.helpers import fromimport

if __name__ == "__main__":
  assert len(sys.argv) >= 3, f"usage: {sys.argv[0]} <compiler> <arch> [<args>]"
  compiler = fromimport(*sys.argv[1].split(':'))(sys.argv[2], *(ast.literal_eval(arg) for arg in sys.argv[3:]))
  while (amt:=sys.stdin.buffer.read(4)):
    try: lib = compiler.compile(sys.stdin.buffer.read(struct.unpack("I", amt)[0]).decode())
    except Exception as e:
      lib = b""
      print(e, file=sys.stderr, flush=True)
    sys.stdout.buffer.write(struct.pack("I", len(lib)) + lib)
    sys.stdout.buffer.flush()
