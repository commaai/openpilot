#include "TinyGPUDriver.h"
#include "TinyGPUDriverUserClient.h"
#include <AudioDriverKit/AudioDriverKit.h>
#include <DriverKit/IOUserServer.h>
#include <DriverKit/IOLib.h>
#include <DriverKit/OSString.h>
#include <DriverKit/IOMemoryMap.h>
#include <DriverKit/IODMACommand.h>
#include <DriverKit/IODispatchQueue.h>
#include <PCIDriverKit/PCIDriverKit.h>
#include <DriverKit/OSAction.h>

struct TinyGPUDriver_IVars
{
	IOPCIDevice *pci = nullptr;
};

bool TinyGPUDriver::init()
{
	os_log(OS_LOG_DEFAULT, "tinygpu: init");

	auto answer = super::init();
	if (!answer) {
		return false;
	}

	ivars = new TinyGPUDriver_IVars();
	if (ivars == nullptr) {
		return false;
	}

	return true;
}

void TinyGPUDriver::free()
{
	if (ivars != nullptr) {
		
	}
	IOSafeDeleteNULL(ivars, TinyGPUDriver_IVars, 1);
	super::free();
}

kern_return_t TinyGPUDriver::Start_Impl(IOService* in_provider)
{
	IOServiceName service_name;
	os_log(OS_LOG_DEFAULT, "tinygpu: on gpu detected");

	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err) return err;

	ivars->pci = OSDynamicCast(IOPCIDevice, in_provider);
	if (!ivars->pci) return kIOReturnNoDevice;

	err = ivars->pci->Open(this, 0);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: Open() failed 0x%08x", err);
		ivars->pci = nullptr;
		return err;
	}

	uint16_t ven = 0, dev = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetVendorID, &ven);
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetDeviceID, &dev);
	os_log(OS_LOG_DEFAULT, "tinygpu: opened device ven=0x%04x dev=0x%04x", ven, dev);

#if 0
	uint32_t off = 0x100;
	while (off) {
		uint32_t hdr = 0, next = 0, cap_id = 0;
		ivars->pci->ConfigurationRead32(off, &hdr);
		cap_id = hdr & 0xFFFFu;
		next = (hdr >> 20) & 0xFFCu;
		os_log(OS_LOG_DEFAULT, "tinygpu: cap: %u", cap_id);
		if (cap_id == 0x15) {
			uint32_t cap = 0, ctrl = 0;
			ivars->pci->ConfigurationRead32(off+0x4, &cap);
			ivars->pci->ConfigurationRead32(off+0x8, &ctrl);

			uint32_t new_bar_size = 31 - __builtin_clz(cap >> 4);
			uint32_t new_ctrl = (ctrl & ~0x1f00) | (new_bar_size << 8);
			ivars->pci->ConfigurationWrite32(off+0x8, new_ctrl);

			os_log(OS_LOG_DEFAULT, "tinygpu: rebar: cap=%u ctrl=%u new_bar_size=%u new_ctrl=%u", cap, ctrl, new_bar_size, new_ctrl);
			ivars->pci->Reset(0);
			break;
		}
		off = next;
	}
	ivars->pci->Reset(kIOPCIDeviceResetTypeHotReset);
#endif

	uint16_t commandRegister;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &commandRegister);
	commandRegister |= (kIOPCICommandIOSpace | kIOPCICommandBusMaster | kIOPCICommandMemorySpace);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, commandRegister);

	memcpy((void*)service_name, (void*)"tinygpu\0", 8);
	SetName(service_name);

	os_log(OS_LOG_DEFAULT, "tinygpu: will register service %s", service_name);
	RegisterService();

	os_log(OS_LOG_DEFAULT, "tinygpu: service started %s", service_name);
	return 0;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* in_provider)
{
	ivars->pci->Close(this, 0);
	return 0;
}

kern_return_t TinyGPUDriver::NewUserClient_Impl(uint32_t in_type, IOUserClient** out_user_client)
{
	kern_return_t err = 0;

	IOService* user_client_service = nullptr;
	err = Create(this, "TinyGPUDriverUserClientProperties", &user_client_service);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to create NewUserClient");
		goto error;
	}
	*out_user_client = OSDynamicCast(IOUserClient, user_client_service);
	os_log(OS_LOG_DEFAULT, "tinygpu: NewUserClient created");

error:
	return err;
}

kern_return_t TinyGPUDriver::MapBar(uint32_t bar, IOMemoryDescriptor** memory)
{
	kern_return_t err = 0;
	uint8_t barMemoryIndex, barMemoryType;
	uint64_t barMemorySize;
	err = ivars->pci->GetBARInfo(bar, &barMemoryIndex, &barMemorySize, &barMemoryType);
	if (err) return err;

	os_log(OS_LOG_DEFAULT, "tinygpu: requested bar mapping %d, %d", bar, (uint32_t)barMemoryIndex);
	err = ivars->pci->_CopyDeviceMemoryWithIndex(barMemoryIndex, memory, this);
	return err;
}

kern_return_t TinyGPUDriver::CreateDMA(size_t size, TinyGPUCreateDMAResp* dmaDesc)
{
	kern_return_t err = 0;
	IOMemoryMap* memoryMap = nullptr;
	IOBufferMemoryDescriptor* sharedBuf = nullptr;
	IODMACommand* dmaCmd = nullptr;
	uint64_t flags = kIOMemoryDirectionInOut;
	uint32_t segCount = 32;
	IOAddressSegment segments[32];
	IODMACommandSpecification dmaSpec = {
		.options = 0,
		.maxAddressBits = 40,
	};

	err = IOBufferMemoryDescriptor::Create(kIOMemoryDirectionInOut, size, IOVMPageSize, &sharedBuf);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to alloc user buffer, err=%d", err);
		goto error;
	}
	
	err = IODMACommand::Create(ivars->pci, kIODMACommandCreateNoOptions, &dmaSpec, &dmaCmd);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to create dma command, err=%d", err);
		goto error;
	}

	err = dmaCmd->PrepareForDMA(kIODMACommandPrepareForDMANoOptions, sharedBuf, 0, size,
								&flags, &segCount, segments);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to prepare for dma, err=%d", err);
		goto error;
	}

	// pass addresses to userland
	{
		// debug
		for (int i = 0; i < segCount; i++) {
			os_log(OS_LOG_DEFAULT, "tinygpu: new dma mapping (sz=0x%zx) %d 0x%llx 0x%llx", size, i, segments[i].address, segments[i].length);
		}

		err = sharedBuf->CreateMapping(0, 0, 0, IOVMPageSize, IOVMPageSize, &memoryMap); // one page should be fine
		if (err) {
			os_log(OS_LOG_DEFAULT, "tinygpu: failed to map memory, err=%d", err);
			goto error;
		}

		// Send back gpu addresses
		uint64_t* addr = (uint64_t*)memoryMap->GetAddress();
		for (int i = 0; i < segCount; i++) {
			addr[i * 2] = segments[i].address;
			addr[i * 2 + 1] = segments[i].length;
		}
		addr[segCount * 2] = 0;
		addr[segCount * 2 + 1] = 0;

		// free memoryMap
		memoryMap->release();
		memoryMap = nullptr;
	}

	dmaDesc->sharedBuf = sharedBuf;
	dmaDesc->dmaCmd = dmaCmd;
	return 0;

error:
	if (memoryMap) {
		memoryMap->release();
		memoryMap = nullptr;
	}
	if (dmaCmd) {
		dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions);
		dmaCmd->release();
		dmaCmd = nullptr;
	}
	if (sharedBuf) {
		sharedBuf->release();
		sharedBuf = nullptr;
	}
	return err;
}

kern_return_t TinyGPUDriver::CfgRead(uint32_t off, uint32_t size, uint32_t* outVal)
{
  if (!ivars->pci || !outVal) return kIOReturnNotReady;

  if (size == 1) {
	uint8_t v8 = 0;
	ivars->pci->ConfigurationRead8(off, &v8);
	*outVal = v8;
  } else if (size == 2) {
	uint16_t v16 = 0;
	ivars->pci->ConfigurationRead16(off, &v16);
	*outVal = v16;
  } else if (size == 4) {
	uint32_t v32 = 0;
	ivars->pci->ConfigurationRead32(off, &v32);
	*outVal = v32;
  }
  return 0;
}

kern_return_t TinyGPUDriver::CfgWrite(uint32_t off, uint32_t size, uint32_t val)
{
  if (!ivars->pci) return kIOReturnNotReady;
  if (size == 1) ivars->pci->ConfigurationWrite8 (off, (uint8_t)val);
  else if (size == 2) ivars->pci->ConfigurationWrite16(off, (uint16_t)val);
  else if (size == 4) ivars->pci->ConfigurationWrite32(off, (uint32_t)val);
  return 0;
}

kern_return_t TinyGPUDriver::ResetDevice()
{
	if (!ivars->pci) return kIOReturnNotReady;
	ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
	return 0;
}
