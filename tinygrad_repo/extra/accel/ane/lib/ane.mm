#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sstream>

#import <IOSurface/IOSurfaceRef.h>

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

#include "h11ane.h"
using namespace H11ANE;

//#define DEBUG printf
#define DEBUG(x, ...)

extern "C" {

// global vars
H11ANEDevice *dev = NULL;

int MyH11ANEDeviceControllerNotification(H11ANEDeviceController *param_1, void *param_2, H11ANEDevice *param_3) {
  DEBUG("MyH11ANEDeviceControllerNotification %p %p %p\n", param_1, param_2, param_3);
  dev = param_3;
  return 0;
}

int MyH11ANEDeviceMessageNotification(H11ANE::H11ANEDevice* dev, unsigned int param_1, void* param_2, void* param_3) {
  DEBUG("MyH11ANEDeviceMessageNotification %d %p %p\n", param_1, param_2, param_3);
  return 0;
}

int ANE_Open() {
  int ret;
  H11ANEDeviceController dc(MyH11ANEDeviceControllerNotification, NULL);
  dc.SetupDeviceController();
  assert(dev != NULL);
  dev->EnableDeviceMessages();

  char empty[0x90] = {0};
  H11ANEDeviceInfoStruct dis = {0};
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty, UsageCompile, &dis);
  DEBUG("open 0x%x %p\n", ret, dev);

  ret = dev->ANE_PowerOn();
  DEBUG("power on: %d\n", ret);

  ret = dev->ANE_IsPowered();
  DEBUG("powered? %d\n", ret);

  return 0;
}

int stride_for_width(int width) {
  int ret = width*2;
  ret += (64-(ret % 64))%64;
  return ret;
}

void *ANE_TensorCreate(int width, int height) {
  // all float16
  // input buffer

  NSDictionary* dict = [NSDictionary dictionaryWithObjectsAndKeys:
                           [NSNumber numberWithInt:width], kIOSurfaceWidth,
                           [NSNumber numberWithInt:height], kIOSurfaceHeight,
                           [NSNumber numberWithInt:2], kIOSurfaceBytesPerElement,
                           [NSNumber numberWithInt:stride_for_width(width)], kIOSurfaceBytesPerRow,
                           [NSNumber numberWithInt:1278226536], kIOSurfacePixelFormat,
                           nil];
  IOSurfaceRef in_surf = IOSurfaceCreate((CFDictionaryRef)dict);
  IOSurfaceLock((IOSurfaceRef)in_surf, 0, nil);

  return (void *)in_surf;
}

void* ANE_TensorData(void *out_surf) {
  void *ret = (void *)IOSurfaceGetBaseAddress((IOSurfaceRef)out_surf);
  //IOSurfaceUnlock((IOSurfaceRef)out_surf, 0, nil);
  DEBUG("TensorData %p -> %p\n", out_surf, ret);
  return ret;
}

uint64_t ANE_Compile(char *iprog, int sz) {
  int ret;
  int cksum = 0;
  for (int i = 0; i < sz; i++) cksum += iprog[i];
  DEBUG("ANE_Compile %p with checksum %x size %d\n", iprog, cksum, sz);

  char *prog = (char*)aligned_alloc(0x1000, sz);
  memcpy(prog, iprog, sz);

  H11ANEProgramCreateArgsStruct mprog = {0};
  mprog.program = prog;
  mprog.program_length = sz;

  H11ANEProgramCreateArgsStructOutput *out = new H11ANEProgramCreateArgsStructOutput;
  memset(out, 0, sizeof(H11ANEProgramCreateArgsStructOutput));
  ret = dev->ANE_ProgramCreate(&mprog, out);
  uint64_t program_handle = out->program_handle;
  delete out;
  DEBUG("program create: %lx %lx\n", ret, program_handle);
  // early failure
  if (ret != 0) return 0;

  H11ANEProgramPrepareArgsStruct pas = {0};
  pas.program_handle = program_handle;
  pas.flags = 0x0000000100010001;
  ret = dev->ANE_ProgramPrepare(&pas);
  DEBUG("program prepare: %lx\n", ret);

  return program_handle;
}

int ANE_Run(uint64_t program_handle, void *in_surf, void *out_surf, void *w_surf) {
  int ret;
  DEBUG("ANE_Run %p %p\n", in_surf, out_surf);
  H11ANEProgramRequestArgsStruct *pras = new H11ANEProgramRequestArgsStruct;
  memset(pras, 0, sizeof(H11ANEProgramRequestArgsStruct));

  // TODO: make real struct
  pras->args[0] = program_handle;
  pras->args[4] = 0x0000002100000003;

  // inputs
  int in_surf_id = IOSurfaceGetID((IOSurfaceRef)in_surf);
  int out_surf_id = IOSurfaceGetID((IOSurfaceRef)out_surf);

  if (w_surf != NULL) {
    pras->args[0x28/8] = 0x0000010000000002;
    int w_surf_id = IOSurfaceGetID((IOSurfaceRef)w_surf);
    pras->args[0x130/8] = (long long)w_surf_id;
  } else {
    pras->args[0x28/8] = 1;
  }
  pras->args[0x128/8] = (long long)in_surf_id<<32LL;

  // outputs
  pras->args[0x528/8] = 1;
  // 0x628 = outputBufferSurfaceId
  pras->args[0x628/8] = (long long)out_surf_id<<32LL;

  mach_port_t recvPort = 0;
  IOCreateReceivePort(kOSAsyncCompleteMessageID, &recvPort);
  DEBUG("recv port: 0x%x\n", recvPort);

  // run program
  ret = dev->ANE_ProgramSendRequest(pras, recvPort);
  DEBUG("send 0x%x\n", ret);

  struct {
    mach_msg_header_t header;
    char data[256];
  } message;

  ret = mach_msg(&message.header,
          MACH_RCV_MSG,
          0, sizeof(message),
          recvPort,
          MACH_MSG_TIMEOUT_NONE,
          MACH_PORT_NULL);
  DEBUG("got message: %d sz %d\n", ret, message.header.msgh_size);
  delete pras;

  return 0;
}

int ANECCompile(CFDictionaryRef param_1, CFDictionaryRef param_2, unsigned long param_3);
int ANE_CompilePlist(char *path, bool debug=false) {
  CFTypeRef ikeys[2];
  ikeys[0] = CFSTR("NetworkPlistName");
  ikeys[1] = CFSTR("NetworkPlistPath");

  CFTypeRef ivalues[2];
  ivalues[0] = CFStringCreateWithCString(kCFAllocatorDefault, path, kCFStringEncodingUTF8);
  ivalues[1] = CFSTR("./");

  CFDictionaryRef iDictionary = CFDictionaryCreate(kCFAllocatorDefault, ikeys, ivalues, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFArrayRef array = CFArrayCreate(kCFAllocatorDefault, (const void**)&iDictionary, 1, &kCFTypeArrayCallBacks);

  CFMutableDictionaryRef optionsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFMutableDictionaryRef flagsDictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

  // h11 (or anything?) works here too, and creates different outputs that don't run
  CFDictionaryAddValue(flagsDictionary, CFSTR("TargetArchitecture"), CFSTR("h13"));
  CFDictionaryAddValue(optionsDictionary, CFSTR("OutputFileName"), CFSTR("model.hwx"));

  if (debug) {
    CFDictionaryAddValue(flagsDictionary, CFSTR("CompileANEProgramForDebugging"), kCFBooleanTrue);
    int debug_mask = 0x7fffffff;
    CFDictionaryAddValue(flagsDictionary, CFSTR("DebugMask"), CFNumberCreate(kCFAllocatorDefault, 3, &debug_mask));
  }

  return ANECCompile(optionsDictionary, flagsDictionary, 0);
}

/*void _Z24ZinIrRegBitPrintOutDebugILj7EE11ZinIrStatusjRN11ZinHWTraitsIXT_EE6HwTypeEiRNSt3__113basic_ostreamIcNS5_11char_traitsIcEEEE(
  unsigned long param_1, void *param_2,int param_3, std::basic_ostream<char> *param_4);
char *ANE_RegDebug(int a1, void *dat, int a2) {
  std::ostringstream ss;
  _Z24ZinIrRegBitPrintOutDebugILj7EE11ZinIrStatusjRN11ZinHWTraitsIXT_EE6HwTypeEiRNSt3__113basic_ostreamIcNS5_11char_traitsIcEEEE(a1, dat, a2, &ss);
  std::string cppstr = ss.str();
  const char *str = cppstr.c_str();
  char *ret = (char *)malloc(strlen(str)+1);
  strcpy(ret, str);
  return ret;
}*/

}

