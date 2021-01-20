#include "stdafx.h"

#include <stdio.h>

LONG __cdecl
_tmain(
    LONG     Argc,
    LPTSTR * Argv
    )
/*++

Routine description:

    Sample program that communicates with a USB device using WinUSB

--*/
{
    DEVICE_DATA           deviceData;
    HRESULT               hr;
    USB_DEVICE_DESCRIPTOR deviceDesc;
    BOOL                  bResult;
    BOOL                  noDevice;
    ULONG                 lengthReceived;

    UNREFERENCED_PARAMETER(Argc);
    UNREFERENCED_PARAMETER(Argv);

    //
    // Find a device connected to the system that has WinUSB installed using our
    // INF
    //
    hr = OpenDevice(&deviceData, &noDevice);

    if (FAILED(hr)) {

        if (noDevice) {

            printf(_T("Device not connected or driver not installed\n"));

        } else {

            printf(_T("Failed looking for device, HRESULT 0x%x\n"), hr);
        }

        return 0;
    }

    //
    // Get device descriptor
    //
    bResult = WinUsb_GetDescriptor(deviceData.WinusbHandle,
                                   USB_DEVICE_DESCRIPTOR_TYPE,
                                   0,
                                   0,
                                   (PBYTE) &deviceDesc,
                                   sizeof(deviceDesc),
                                   &lengthReceived);

    if (FALSE == bResult || lengthReceived != sizeof(deviceDesc)) {

        printf(_T("Error among LastError %d or lengthReceived %d\n"),
               FALSE == bResult ? GetLastError() : 0,
               lengthReceived);
        CloseDevice(&deviceData);
        return 0;
    }

    //
    // Print a few parts of the device descriptor
    //
	printf(_T("Device found: VID_%04X&PID_%04X; bcdUsb %04X; path: %s\n"),
           deviceDesc.idVendor,
           deviceDesc.idProduct,
           deviceDesc.bcdUSB,
		deviceData.DevicePath);

    CloseDevice(&deviceData);
    return 0;
}
