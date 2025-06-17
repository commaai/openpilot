#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#import <IOSurface/IOSurfaceRef.h>

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

void hexdump(void *vdat, int l) {
  unsigned char *dat = (unsigned char *)vdat;
  for (int i = 0; i < l; i++) {
    if (i!=0 && (i%0x10) == 0) printf("\n");
    printf("%02X ", dat[i]);
  }
  printf("\n");
}

#include "h11ane.h"

using namespace H11ANE;

H11ANEDevice *device = NULL;

int MyH11ANEDeviceControllerNotification(H11ANEDeviceController *param_1, void *param_2, H11ANEDevice *param_3) {
  printf("MyH11ANEDeviceControllerNotification %p %p %p\n", param_1, param_2, param_3);
  device = param_3;
  return 0;
}

int MyH11ANEDeviceMessageNotification(H11ANE::H11ANEDevice* dev, unsigned int param_1, void* param_2, void* param_3) {
  printf("MyH11ANEDeviceMessageNotification %d %p %p\n", param_1, param_2, param_3);
  return 0;
}

int main() {
  int ret;
  printf("hello %d\n", getpid());

  H11ANEDeviceController dc(MyH11ANEDeviceControllerNotification, NULL);
  dc.SetupDeviceController();
  assert(device != NULL);
  H11ANEDevice *dev = device;
  dev->EnableDeviceMessages();

  char empty[0x90] = {0};
  H11ANEDeviceInfoStruct dis = {0};
  //dis.nothing = 0x87c15a20a;
  //dis.sleep_timer = 5000;
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty, UsageCompile, &dis);
  printf("open 0x%x %p\n", ret, dev);

  /*ret = dev->ANE_AddPersistentClient();
  printf("add persistent %x\n", ret);*/

  H11ANEStatusStruct blah = {0};
  ret = dev->ANE_GetStatus(&blah);
  printf("get status %x\n", ret);

  // this isn't callable anymore, it requires debugger
  ret = dev->ANE_PowerOn();
  printf("power on: %x\n", ret);

  ret = dev->ANE_IsPowered();
  printf("powered? %d\n", ret);

  /*if (ret == 0) {
    printf("POWER ON FAILED\n");
    return -1;
  }*/

  H11ANEProgramCreateArgsStruct mprog = {0};
  mprog.program_length = 0xc000;
  char *prog = (char*)aligned_alloc(0x1000, mprog.program_length);
  mprog.program = prog;

  FILE *f = fopen("../2_compile/model.hwx", "rb");
  assert(f);
  int sz = fread(prog, 1, mprog.program_length, f);
  printf("read %x %p\n", sz, prog);
  fclose(f);

  H11ANEProgramCreateArgsStructOutput *out = new H11ANEProgramCreateArgsStructOutput;
  memset(out, 0, sizeof(H11ANEProgramCreateArgsStructOutput));
  ret = dev->ANE_ProgramCreate(&mprog, out);
  uint64_t program_handle = out->program_handle;
  printf("program create: %lx %lx\n", ret, program_handle);

  H11ANEProgramPrepareArgsStruct pas = {0};
  pas.program_handle = program_handle;
  pas.flags = 0x0000000100010001;
  //pas.flags = 0x0000000102010001;
  ret = dev->ANE_ProgramPrepare(&pas);
  printf("program prepare: %lx\n", ret);

  // input buffer
  NSDictionary* dict = [NSDictionary dictionaryWithObjectsAndKeys:
                           [NSNumber numberWithInt:16], kIOSurfaceWidth,
                           [NSNumber numberWithInt:16], kIOSurfaceHeight,
                           [NSNumber numberWithInt:1], kIOSurfaceBytesPerElement,
                           [NSNumber numberWithInt:64], kIOSurfaceBytesPerRow,
                           [NSNumber numberWithInt:1278226536], kIOSurfacePixelFormat,
                           nil];
  IOSurfaceRef in_surf = IOSurfaceCreate((CFDictionaryRef)dict);
  int in_surf_id = IOSurfaceGetID(in_surf);
  printf("we have surface %p with id 0x%x\n", in_surf, in_surf_id);

  // load inputs
  IOSurfaceLock(in_surf, 0, nil);
  unsigned char *inp = (unsigned char *)IOSurfaceGetBaseAddress(in_surf);
  for (int i = 0; i < 16; i++) inp[i] = (i+1)*0x10;
  /*inp[0] = 0x39;
  inp[1] = 0x65;*/
  hexdump(inp, 0x20);
  IOSurfaceUnlock(in_surf, 0, nil);

  // output buffer
  NSDictionary* odict = [NSDictionary dictionaryWithObjectsAndKeys:
                           [NSNumber numberWithInt:16], kIOSurfaceWidth,
                           [NSNumber numberWithInt:16], kIOSurfaceHeight,
                           [NSNumber numberWithInt:1], kIOSurfaceBytesPerElement,
                           [NSNumber numberWithInt:64], kIOSurfaceBytesPerRow,
                           [NSNumber numberWithInt:1278226536], kIOSurfacePixelFormat,
                           nil];
  IOSurfaceRef out_surf = IOSurfaceCreate((CFDictionaryRef)odict);
  int out_surf_id = IOSurfaceGetID(out_surf);
  printf("we have surface %p with id 0x%x\n", out_surf, out_surf_id);

  H11ANEProgramRequestArgsStruct *pras = new H11ANEProgramRequestArgsStruct;
  memset(pras, 0, sizeof(H11ANEProgramRequestArgsStruct));

  // TODO: make real struct
  pras->args[0] = program_handle;
  pras->args[4] = 0x0000002100000003;

  // inputs
  pras->args[0x28/8] = 1;
  pras->args[0x128/8] = (long long)in_surf_id<<32LL;

  // outputs
  pras->args[0x528/8] = 1;
  // 0x628 = outputBufferSurfaceId
  pras->args[0x628/8] = (long long)out_surf_id<<32LL;

  mach_port_t recvPort = 0;
  IOCreateReceivePort(kOSAsyncCompleteMessageID, &recvPort);
  printf("recv port: 0x%x\n", recvPort);

  // *** reopen with other client ***
  H11ANEDeviceController dc2(MyH11ANEDeviceControllerNotification, NULL);
  dc2.SetupDeviceController();
  assert(device != NULL);
  dev = device;
  dev->EnableDeviceMessages();

  char empty2[0x90] = {0};
  dis.program_handle = program_handle;
  dis.program_auth_code = 0;
  ret = dev->H11ANEDeviceOpen(MyH11ANEDeviceMessageNotification, empty2, UsageWithProgram, &dis);
  printf("reopen 0x%x %p\n", ret, dev);

  // run program (i think we need the other client for this)
  ret = dev->ANE_ProgramSendRequest(pras, recvPort);
  printf("send 0x%x\n", ret);

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
  printf("got message: %d sz %d\n", ret, message.header.msgh_size);

  unsigned char *dat = (unsigned char *)IOSurfaceGetBaseAddress(out_surf);
  printf("%p\n", dat);
  hexdump(dat, 0x100);
}


