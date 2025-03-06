enum ANEDeviceUsageType {
  UsageNoProgram,
  UsageWithProgram,  // used in running process
  UsageCompile       // used in aned
};

struct H11ANEDeviceInfoStruct {
  uint64_t program_handle;
  uint64_t program_auth_code;
  uint64_t sleep_timer;
  uint64_t junk[0x100];
};

struct H11ANEStatusStruct {
  uint64_t junk[0x100];
};

struct H11ANEProgramCreateArgsStruct {
  void *program;
  uint64_t program_length;
  uint64_t empty[4];
  char has_signature;
};

struct H11ANEProgramCreateArgsStructOutput {
  uint64_t program_handle;
  int unknown[0x2000];
};

struct H11ANEProgramPrepareArgsStruct {
  uint64_t program_handle;
  uint64_t flags;
  uint64_t empty[0x100];
};

struct H11ANEProgramRequestArgsStruct {
  uint64_t args[0x1000];
};

namespace H11ANE {
  class H11ANEDevice;

  class H11ANEDeviceController {
    public:
      H11ANEDeviceController(
        int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*),
        void *arg);
      int SetupDeviceController();
    private:   // size is 0x50
      CFArrayRef array_ref;
      mach_port_t *master_port;
      IONotificationPortRef port_ref;
      CFRunLoopSourceRef source_ref;
      int (*callback)(H11ANE::H11ANEDeviceController*, void*, H11ANE::H11ANEDevice*);
      void *callback_arg;
      CFRunLoopRef run_loop_ref;
      io_iterator_t io_iterator;
      pthread_t thread_self;
      uint64_t unused;
  };

  // we should switch to the IOKit kernel interface, it's likely a lot more stable
  // actually this probably isn't true. ANEServices is normal dynamic links
  // https://googleprojectzero.blogspot.com/2020/11/oops-i-missed-it-again.html

  // H11ANEInDirectPathClient
  // _ANE_DeviceOpen
  // _ANE_DeviceClose
  // _ANE_ProgramSendRequest

  // * if they need kernel debugger attached
  // H11ANEInUserClient
  // _ANE_DeviceOpen
  // _ANE_DeviceClose
  // _ANE_ProgramSendRequest
  // _ANE_ProgramCreate
  // _ANE_ProgramPrepare
  // _ANE_ProgramUnprepare
  // _ANE_ProgramDestroy
  // _ANE_GetStatus
  // _ANE_PowerOn
  // _ANE_PowerOff
  // _ANE_IsPowered
  // * _ANE_LoadFirmware
  // * _ANE_ForgetFirmware
  // * _ANE_SendCommand
  // _ANE_SetPowerManagement
  // _ANE_GetTime
  // * _ANE_SetDriverLoggingFlags
  // * _ANE_ShowSharedMemoryAllocations
  // * _ANE_SetDARTCacheTTL
  // * _ANE_SetFirmwareBootArg
  // * _ANE_SetThrottlingPercentage
  // * _ANE_AddPersistentClient
  // * _ANE_RemovePersistentClient
  // * _ANE_CreateClientLoggingSession
  // * _ANE_TerminateClientLoggingSession
  // _ANE_GetDriverLoggingFlags
  // * _ANE_FlushInactiveDARTMappings
  // _ANE_GetVersion
  // _ANE_RegisterFirmwareWorkProcessor
  // _ANE_UnregisterFirmwareWorkProcessor
  // * _ANE_GetFirmwareWorkProcessorItem
  // _ANE_CompleteFirmwareWorkProcessorItem
  // _ANE_ReleaseFirmwareWorkProcessorBuffers
  // * _ANE_ReadANERegister
  // * _ANE_WriteANERegister
  // _ANE_ProgramCreateInstance

  // note, this is not the raw IOKit class, it's in ANEServices.framework
  class H11ANEDevice {
    public:
      H11ANEDevice(H11ANE::H11ANEDeviceController *param_1, unsigned int param_2);

      unsigned long H11ANEDeviceOpen(
        int (*callback)(H11ANE::H11ANEDevice*, unsigned int, void*, void*),
        void *param_2, ANEDeviceUsageType param_3, H11ANEDeviceInfoStruct *param_4);

      void EnableDeviceMessages();
      int ANE_AddPersistentClient();
      int ANE_GetStatus(H11ANEStatusStruct *param_1);

      // power management
      int ANE_IsPowered();
      int ANE_PowerOn();
      int ANE_PowerOff();

      // logging (e00002c7 error, needs PE_i_can_has_debugger)
      int ANE_CreateClientLoggingSession(unsigned int log_iosurface);
      int ANE_TerminateClientLoggingSession(unsigned int log_iosurface);
      int ANE_GetDriverLoggingFlags(unsigned int *flags);
      int ANE_SetDriverLoggingFlags(unsigned int flags);

      // program creation
      int ANE_ProgramCreate(H11ANEProgramCreateArgsStruct*,
                            H11ANEProgramCreateArgsStructOutput*);
      int ANE_ProgramPrepare(H11ANEProgramPrepareArgsStruct*);
      int ANE_ProgramSendRequest(H11ANEProgramRequestArgsStruct*, mach_port_t);

      // need PE_i_can_has_debugger
      int ANE_ReadANERegister(unsigned int param_1, unsigned int *param_2);
      int ANE_ForgetFirmware();


    private:   // size is 0x88 
      unsigned char unknown[0x88];
  };

};

