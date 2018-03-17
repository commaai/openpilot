#include "stdafx.h"

#include <SetupAPI.h>
#include <Devpkey.h>

#include <unordered_map>
#include <string>

#include <winusb.h>

#include "device.h"

using namespace panda;

//Returns the last Win32 error, in string format. Returns an empty string if there is no error.
tstring GetLastErrorAsString(){
	//Get the error message, if any.
	DWORD errorMessageID = ::GetLastError();
	if (errorMessageID == 0)
		return tstring(); //No error message has been recorded

	_TCHAR *messageBuffer = nullptr;
	size_t size = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (_TCHAR*)&messageBuffer, 0, NULL);

	tstring message(messageBuffer, size);

	//Free the buffer.
	LocalFree(messageBuffer);

	return message;
}

std::unordered_map<std::string, tstring> panda::detect_pandas() {
	HDEVINFO                        deviceInfo;
	HRESULT                         hr;
	SP_DEVINFO_DATA					deviceInfoData;
	SP_DEVICE_INTERFACE_DATA		interfaceData;
	unsigned int					deviceIndex;

	std::unordered_map<std::string, tstring> map_sn_to_devpath;

	deviceInfo = SetupDiGetClassDevs(&GUID_DEVINTERFACE_panda,
		NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE); //DIGCF_ALLCLASSES

	if (deviceInfo == INVALID_HANDLE_VALUE) {
		hr = HRESULT_FROM_WIN32(GetLastError());
		_tprintf(_T("Failed to get dev handle. HR: %d\n"), hr);
		return map_sn_to_devpath;
	}

	ZeroMemory(&deviceInfoData, sizeof(SP_DEVINFO_DATA));
	deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
	deviceIndex = 0;

	while (SetupDiEnumDeviceInfo(deviceInfo, deviceIndex, &deviceInfoData)) {
		deviceIndex++;
		_tprintf(_T("Device info index %d\n"), deviceIndex);

		interfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
		if (SetupDiEnumDeviceInterfaces(deviceInfo, &deviceInfoData,
			&GUID_DEVINTERFACE_panda, 0, &interfaceData) == FALSE) {
			_tprintf(_T("    Got unexpected error while accessing interface %d\n"), GetLastError());
			continue;
		}

		DWORD requiredLength;
		if (SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, NULL, 0, &requiredLength, NULL) == FALSE
			&& ERROR_INSUFFICIENT_BUFFER != GetLastError()) {
			_tprintf(_T("    Got unexpected error while reading interface details %d\n"), GetLastError());
			continue;
		}

		PSP_DEVICE_INTERFACE_DETAIL_DATA detailData = (PSP_DEVICE_INTERFACE_DETAIL_DATA)LocalAlloc(LMEM_FIXED, requiredLength);
		if (NULL == detailData) {
			_tprintf(_T("    Failed to allocate %d bytes for interface data\n"), requiredLength);
			continue;
		}
		detailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

		DWORD length = requiredLength;
		if (SetupDiGetDeviceInterfaceDetail(deviceInfo, &interfaceData, detailData, length, &requiredLength, NULL) == FALSE) {
			_tprintf(_T("    Got unexpected error while reading interface details (2nd time) %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			LocalFree(detailData);
			continue;
		}

		//_tprintf(_T("    Path: '%s'\n"), detailData->DevicePath);
		HANDLE deviceHandle = CreateFile(detailData->DevicePath,
			GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE | FILE_SHARE_READ,
			NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);

		if (INVALID_HANDLE_VALUE == deviceHandle) {
			_tprintf(_T("    Error opening Device Handle %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			LocalFree(detailData);
			continue;
		}

		WINUSB_INTERFACE_HANDLE winusbHandle;
		if (WinUsb_Initialize(deviceHandle, &winusbHandle) == FALSE) {
			_tprintf(_T("    Error initializing WinUSB %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			CloseHandle(deviceHandle);
			LocalFree(detailData);
			continue;
		}

		USB_DEVICE_DESCRIPTOR deviceDesc;
		unsigned long lengthReceived;
		if (WinUsb_GetDescriptor(winusbHandle, USB_DEVICE_DESCRIPTOR_TYPE, 0, 0,
			(PBYTE)&deviceDesc, sizeof(deviceDesc), &lengthReceived) == FALSE
			|| lengthReceived != sizeof(deviceDesc)) {
			_tprintf(_T("    Error getting device descriptor %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			WinUsb_Free(winusbHandle);
			CloseHandle(deviceHandle);
			LocalFree(detailData);
			continue;
		}

		#define SNDESCLEN 64
		PUSB_STRING_DESCRIPTOR psnDesc = (PUSB_STRING_DESCRIPTOR)LocalAlloc(LMEM_FIXED, SNDESCLEN);
		if (NULL == psnDesc) {
			_tprintf(_T("    Failed to allocate %d bytes for sn data\n"), SNDESCLEN);
			continue;
		}

		if (WinUsb_GetDescriptor(winusbHandle, USB_STRING_DESCRIPTOR_TYPE, deviceDesc.iSerialNumber,
			0x0409 /*Eng*/, (PBYTE)psnDesc, SNDESCLEN, &lengthReceived) == FALSE || lengthReceived == 0) {
			_tprintf(_T("    Error getting serial number %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			LocalFree(psnDesc);
			WinUsb_Free(winusbHandle);
			CloseHandle(deviceHandle);
			LocalFree(detailData);
			continue;
		}
		//The minus 2 is for the two numbers, not the null term.
		psnDesc->bString[(psnDesc->bLength - 2) / sizeof(_TCHAR)] = 0;

		char w_to_m_buff[256];
		size_t mbuff_len;
		if (wcstombs_s(&mbuff_len, w_to_m_buff, sizeof(w_to_m_buff), psnDesc->bString, 24) != 0) {
			_tprintf(_T("    Error generating mb SN string %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
			LocalFree(psnDesc);
			WinUsb_Free(winusbHandle);
			CloseHandle(deviceHandle);
			LocalFree(detailData);
			continue;
		}
		std::string serialnum(w_to_m_buff, mbuff_len-1);
		printf("    Device found: seriallen: %d; serial: %s\n", lengthReceived, serialnum.c_str());

		map_sn_to_devpath[serialnum] = tstring(detailData->DevicePath);

		LocalFree(psnDesc);
		WinUsb_Free(winusbHandle);
		CloseHandle(deviceHandle);
		LocalFree(detailData);
	}

	if(deviceInfo)
		SetupDiDestroyDeviceInfoList(deviceInfo);

	return map_sn_to_devpath;
}
