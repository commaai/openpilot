#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <stdio.h>
#include <inttypes.h>

static io_connect_t open_uc_by_name(const char *svc_name) {
  io_connect_t conn = IO_OBJECT_NULL;
  io_service_t service = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceNameMatching(svc_name));
  if (!service) { fprintf(stderr, "service not found: %s\n", svc_name); return IO_OBJECT_NULL; }
  kern_return_t kr = IOServiceOpen(service, mach_task_self(), /*type*/0, &conn);
  IOObjectRelease(service);
  if (kr) { fprintf(stderr, "IOServiceOpen 0x%x\n", kr); return IO_OBJECT_NULL; }
  return conn;
}

int main(int argc, char **argv) {
  uint32_t bar = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 0) : 0;  // pick BAR index
  io_connect_t conn = open_uc_by_name("tinygpu");
  if (!conn) return 2;

  mach_vm_address_t addr = 0;
  mach_vm_size_t    size = 0;
  kern_return_t kr = IOConnectMapMemory64(conn, bar, mach_task_self(), &addr, &size, kIOMapAnywhere);
  if (kr) { fprintf(stderr, "Map BAR%u failed 0x%x\n", bar, kr); IOServiceClose(conn); return 3; }

  printf("BAR%u mapped at 0x%llx, size 0x%llx\n", bar, (unsigned long long)addr, (unsigned long long)size);

  // example: read a 32-bit register at offset 0x0 (make sure it’s safe!)
  volatile uint32_t *mmio = (volatile uint32_t*)(uintptr_t)addr;
  uint32_t v = mmio[0];
  printf("mmio[0]=0x%08x\n", v);

  kr = IOConnectUnmapMemory64(conn, bar, mach_task_self(), addr);
  if (kr) fprintf(stderr, "Unmap failed 0x%x\n", kr);

  IOServiceClose(conn);
  return 0;
}