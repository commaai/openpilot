from tinygrad import Device

# PATH=/opt/homebrew/opt/llvm/bin:$PATH python3 extra/dsp/opt.py

if __name__ == "__main__":
  compiler = Device["DSP"].compiler

  lib = compiler.compile("""
typedef long HVX_Vector __attribute__((__vector_size__(128))) __attribute__ ((aligned(128)));
typedef long HVX_VectorPair __attribute__((__vector_size__(256))) __attribute__ ((aligned(256)));

void test(unsigned char *c, unsigned char *a, unsigned char *b) {
  HVX_Vector t0 = *(HVX_Vector*)a;
  //HVX_VectorPair t1 = *((HVX_VectorPair*)b);
  HVX_Vector acc = __builtin_HEXAGON_V6_vd0_128B();
  for (int i = 0; i < 128; i++) {
    //__builtin_HEXAGON_V6_lvsplatb_128B(t0[i])
    //acc += __builtin_HEXAGON_V6_lvsplatb_128B(t0[i]) * t1;
    //acc += t0[i] * t1;
    unsigned int t1 = ((unsigned int *)b)[i];
    //acc = __builtin_HEXAGON_V6_vrmpyub_acc_128B(acc, t0, t1);
    acc = __builtin_HEXAGON_V6_vrmpybus_acc_128B(acc, t0, t1);
  }
  *((HVX_Vector*)c) = acc;
}""")

  compiler.disassemble(lib)
