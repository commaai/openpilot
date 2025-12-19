# [<buf device:HIP size:1605632 dtype:dtypes.float>, <buf device:HIP size:301506 dtype:dtypes.float>, <buf device:HIP size:9408 dtype:dtypes.float>]
from tinygrad import Device, dtypes
from tinygrad.device import Buffer, CompiledRunner

import ctypes
import gpuctypes.hip as hip
from tinygrad.helpers import to_char_p_p, init_c_var
def get_bytes(arg, get_sz, get_str, check) -> bytes: return (sz := init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x)))), ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value))[1]  # noqa: E501
def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")
def compile_hip(prg:str, arch="gfx1100") -> bytes:
  check(hip.hiprtcCreateProgram(ctypes.byref(prog := hip.hiprtcProgram()), prg.encode(), "<null>".encode(), 0, None, None))
  compile_options = [f'--offload-arch={arch}', '-I/opt/rocm/include']
  status = hip.hiprtcCompileProgram(prog, len(compile_options), to_char_p_p([o.encode() for o in compile_options]))
  if status != 0: raise RuntimeError(f"compile failed: {get_bytes(prog, hip.hiprtcGetProgramLogSize, hip.hiprtcGetProgramLog, check).decode()}")
  return get_bytes(prog, hip.hiprtcGetCodeSize, hip.hiprtcGetCode, check)

prefix = """
typedef long unsigned int size_t;
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
typedef float float2 __attribute__((ext_vector_type(2)));
static inline __attribute__((device)) float2 make_float2(float x, float y) { return {x, y}; }
"""

code = """
extern "C" __attribute__((global))void r_2_8_7_7_4_8_3_7_7_4_4_2_2(float* data0, const float* data1, const float* data2) {
  int gidx0 = __ockl_get_group_id(2); /* 2 */
  int gidx1 = __ockl_get_group_id(1); /* 8 */
  int gidx2 = __ockl_get_group_id(0); /* 49 */
  int lidx4 = __ockl_get_local_id(1); /* 4 */
  int lidx5 = __ockl_get_local_id(0); /* 8 */
  float2 acc0 = make_float2(0.0f,0.0f);
  float2 acc1 = make_float2(0.0f,0.0f);
  float2 acc2 = make_float2(0.0f,0.0f);
  float2 acc3 = make_float2(0.0f,0.0f);
  float2 acc4 = make_float2(0.0f,0.0f);
  float2 acc5 = make_float2(0.0f,0.0f);
  float2 acc6 = make_float2(0.0f,0.0f);
  float2 acc7 = make_float2(0.0f,0.0f);
  float2 acc8 = make_float2(0.0f,0.0f);
  float2 acc9 = make_float2(0.0f,0.0f);
  float2 acc10 = make_float2(0.0f,0.0f);
  float2 acc11 = make_float2(0.0f,0.0f);
  float2 acc12 = make_float2(0.0f,0.0f);
  float2 acc13 = make_float2(0.0f,0.0f);
  float2 acc14 = make_float2(0.0f,0.0f);
  float2 acc15 = make_float2(0.0f,0.0f);
  float2 acc16 = make_float2(0.0f,0.0f);
  float2 acc17 = make_float2(0.0f,0.0f);
  float2 acc18 = make_float2(0.0f,0.0f);
  float2 acc19 = make_float2(0.0f,0.0f);
  float2 acc20 = make_float2(0.0f,0.0f);
  float2 acc21 = make_float2(0.0f,0.0f);
  float2 acc22 = make_float2(0.0f,0.0f);
  float2 acc23 = make_float2(0.0f,0.0f);
  float2 acc24 = make_float2(0.0f,0.0f);
  float2 acc25 = make_float2(0.0f,0.0f);
  float2 acc26 = make_float2(0.0f,0.0f);
  float2 acc27 = make_float2(0.0f,0.0f);
  float2 acc28 = make_float2(0.0f,0.0f);
  float2 acc29 = make_float2(0.0f,0.0f);
  float2 acc30 = make_float2(0.0f,0.0f);
  float2 acc31 = make_float2(0.0f,0.0f);
  int alu0 = (gidx2/7);
  int alu1 = (gidx2%7);
  int alu2 = (alu1*32);
  int alu3 = (lidx5*4);
  int alu4 = ((gidx0*802816)+(gidx1*100352)+(alu0*1792)+(alu1*16)+(lidx4*448)+(lidx5*2));
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 7; ridx1++) {
      int alu5 = ((alu0*(-32))+(lidx4*(-8))+(ridx1*(-1)));
      bool alu6 = (alu5<(-2));
      bool alu7 = (alu5<0);
      bool alu8 = (((alu0*32)+(lidx4*8)+ridx1)<221);
      for (int ridx2 = 0; ridx2 < 7; ridx2++) {
        int alu9 = ((gidx0*150528)+(ridx0*50176)+(alu0*7168)+(lidx4*1792)+(ridx1*224)+alu2+alu3+ridx2);
        int alu10 = ((alu1*(-32))+(lidx5*(-4))+(ridx2*(-1)));
        bool alu11 = (alu10<(-2));
        float val0 = 0.0f;
        if ((alu6*alu11)) { val0 = data1[alu9+(-675)]; }
        float val1 = 0.0f;
        if ((alu7*alu11)) { val1 = data1[alu9+(-227)]; }
        float val2 = 0.0f;
        if (alu11) { val2 = data1[alu9+221]; }
        float val3 = 0.0f;
        if ((alu8*alu11)) { val3 = data1[alu9+669]; }
        bool alu12 = (alu10<0);
        bool alu13 = ((alu2+alu3+ridx2)<225);
        float val4 = 0.0f;
        if ((alu6*alu12*alu13)) { val4 = data1[alu9+(-673)]; }
        float val5 = 0.0f;
        if ((alu7*alu12*alu13)) { val5 = data1[alu9+(-225)]; }
        float val6 = 0.0f;
        if ((alu12*alu13)) { val6 = data1[alu9+223]; }
        float val7 = 0.0f;
        if ((alu8*alu12*alu13)) { val7 = data1[alu9+671]; }
        int alu14 = ((gidx1*1176)+(ridx0*49)+(ridx1*7)+ridx2);
        float val8 = data2[alu14];
        float val9 = data2[alu14+147];
        float val10 = data2[alu14+294];
        float val11 = data2[alu14+441];
        float val12 = data2[alu14+588];
        float val13 = data2[alu14+735];
        float val14 = data2[alu14+882];
        float val15 = data2[alu14+1029];
        (acc0).x = ((val0*val8)+(acc0).x);
        (acc1).x = ((val0*val9)+(acc1).x);
        (acc2).x = ((val0*val10)+(acc2).x);
        (acc3).x = ((val0*val11)+(acc3).x);
        (acc4).x = ((val1*val8)+(acc4).x);
        (acc5).x = ((val1*val9)+(acc5).x);
        (acc6).x = ((val1*val10)+(acc6).x);
        (acc7).x = ((val1*val11)+(acc7).x);
        (acc8).x = ((val2*val8)+(acc8).x);
        (acc9).x = ((val2*val9)+(acc9).x);
        (acc10).x = ((val2*val10)+(acc10).x);
        (acc11).x = ((val2*val11)+(acc11).x);
        (acc12).x = ((val3*val8)+(acc12).x);
        (acc13).x = ((val3*val9)+(acc13).x);
        (acc14).x = ((val3*val10)+(acc14).x);
        (acc15).x = ((val3*val11)+(acc15).x);
        (acc16).x = ((val0*val12)+(acc16).x);
        (acc17).x = ((val0*val13)+(acc17).x);
        (acc18).x = ((val0*val14)+(acc18).x);
        (acc19).x = ((val0*val15)+(acc19).x);
        (acc20).x = ((val1*val12)+(acc20).x);
        (acc21).x = ((val1*val13)+(acc21).x);
        (acc22).x = ((val1*val14)+(acc22).x);
        (acc23).x = ((val1*val15)+(acc23).x);
        (acc24).x = ((val2*val12)+(acc24).x);
        (acc25).x = ((val2*val13)+(acc25).x);
        (acc26).x = ((val2*val14)+(acc26).x);
        (acc27).x = ((val2*val15)+(acc27).x);
        (acc28).x = ((val3*val12)+(acc28).x);
        (acc29).x = ((val3*val13)+(acc29).x);
        (acc30).x = ((val3*val14)+(acc30).x);
        (acc31).x = ((val3*val15)+(acc31).x);
        (acc0).y = ((val4*val8)+(acc0).y);
        (acc1).y = ((val4*val9)+(acc1).y);
        (acc2).y = ((val4*val10)+(acc2).y);
        (acc3).y = ((val4*val11)+(acc3).y);
        (acc4).y = ((val5*val8)+(acc4).y);
        (acc5).y = ((val5*val9)+(acc5).y);
        (acc6).y = ((val5*val10)+(acc6).y);
        (acc7).y = ((val5*val11)+(acc7).y);
        (acc8).y = ((val6*val8)+(acc8).y);
        (acc9).y = ((val6*val9)+(acc9).y);
        (acc10).y = ((val6*val10)+(acc10).y);
        (acc11).y = ((val6*val11)+(acc11).y);
        (acc12).y = ((val7*val8)+(acc12).y);
        (acc13).y = ((val7*val9)+(acc13).y);
        (acc14).y = ((val7*val10)+(acc14).y);
        (acc15).y = ((val7*val11)+(acc15).y);
        (acc16).y = ((val4*val12)+(acc16).y);
        (acc17).y = ((val4*val13)+(acc17).y);
        (acc18).y = ((val4*val14)+(acc18).y);
        (acc19).y = ((val4*val15)+(acc19).y);
        (acc20).y = ((val5*val12)+(acc20).y);
        (acc21).y = ((val5*val13)+(acc21).y);
        (acc22).y = ((val5*val14)+(acc22).y);
        (acc23).y = ((val5*val15)+(acc23).y);
        (acc24).y = ((val6*val12)+(acc24).y);
        (acc25).y = ((val6*val13)+(acc25).y);
        (acc26).y = ((val6*val14)+(acc26).y);
        (acc27).y = ((val6*val15)+(acc27).y);
        (acc28).y = ((val7*val12)+(acc28).y);
        (acc29).y = ((val7*val13)+(acc29).y);
        (acc30).y = ((val7*val14)+(acc30).y);
        (acc31).y = ((val7*val15)+(acc31).y);
      }
    }
  }
  *((float2*)(data0+alu4)) = acc0;
  *((float2*)(data0+alu4+12544)) = acc1;
  *((float2*)(data0+alu4+25088)) = acc2;
  *((float2*)(data0+alu4+37632)) = acc3;
  *((float2*)(data0+alu4+112)) = acc4;
  *((float2*)(data0+alu4+12656)) = acc5;
  *((float2*)(data0+alu4+25200)) = acc6;
  *((float2*)(data0+alu4+37744)) = acc7;
  *((float2*)(data0+alu4+224)) = acc8;
  *((float2*)(data0+alu4+12768)) = acc9;
  *((float2*)(data0+alu4+25312)) = acc10;
  *((float2*)(data0+alu4+37856)) = acc11;
  *((float2*)(data0+alu4+336)) = acc12;
  *((float2*)(data0+alu4+12880)) = acc13;
  *((float2*)(data0+alu4+25424)) = acc14;
  *((float2*)(data0+alu4+37968)) = acc15;
  *((float2*)(data0+alu4+50176)) = acc16;
  *((float2*)(data0+alu4+62720)) = acc17;
  *((float2*)(data0+alu4+75264)) = acc18;
  *((float2*)(data0+alu4+87808)) = acc19;
  *((float2*)(data0+alu4+50288)) = acc20;
  *((float2*)(data0+alu4+62832)) = acc21;
  *((float2*)(data0+alu4+75376)) = acc22;
  *((float2*)(data0+alu4+87920)) = acc23;
  *((float2*)(data0+alu4+50400)) = acc24;
  *((float2*)(data0+alu4+62944)) = acc25;
  *((float2*)(data0+alu4+75488)) = acc26;
  *((float2*)(data0+alu4+88032)) = acc27;
  *((float2*)(data0+alu4+50512)) = acc28;
  *((float2*)(data0+alu4+63056)) = acc29;
  *((float2*)(data0+alu4+75600)) = acc30;
  *((float2*)(data0+alu4+88144)) = acc31;
}
"""

dev = "HIP"
lib = Device[dev].compiler.compile(prefix+code)
#lib = compile_hip(code)
b0 = Buffer(dev, 1605632, dtypes.float)
b1 = Buffer(dev, 301506, dtypes.float)
b2 = Buffer(dev, 9408, dtypes.float)
print(hex(b0._buf.value), hex(b0._buf.value+1605632*4))
print(hex(b1._buf.value))
print(hex(b2._buf.value))
#prg = CompiledRunner("r_2_8_7_7_4_8_3_7_7_4_4_2_2", "", dev, [7, 1, 1], [8, 4, 1], precompiled=lib)
prg = CompiledRunner("r_2_8_7_7_4_8_3_7_7_4_4_2_2", "", dev, [49, 8, 2], [8, 4, 1], precompiled=lib)
print("compiled")
prg([b0, b1, b2], {})
print("ran")
Device[dev].synchronize()
print("sync")
