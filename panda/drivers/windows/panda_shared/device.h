#ifndef __PANDA_DEVICE
#define __PANDA_DEVICE

//
// Define below GUIDs
//
#include <initguid.h>
#include <unordered_map>

#if defined(UNICODE)
#define _tcout std::wcout
#define tstring std::wstring
#else
#define _tcout std::cout
#define tstring std::string
#endif

//
// Device Interface GUID.
// Used by all WinUsb devices that this application talks to.
// Must match "DeviceInterfaceGUIDs" registry value specified in the INF file.
// cce5291c-a69f-4995-a4c2-2ae57a51ade9
//
DEFINE_GUID(GUID_DEVINTERFACE_panda,
    0xcce5291c,0xa69f,0x4995,0xa4,0xc2,0x2a,0xe5,0x7a,0x51,0xad,0xe9);

tstring GetLastErrorAsString();

namespace panda {
	std::unordered_map<std::string, tstring> __declspec(dllexport) detect_pandas();
}
#endif
