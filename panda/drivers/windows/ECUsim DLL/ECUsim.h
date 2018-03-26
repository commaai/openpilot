#pragma once

#include <string>
#include "panda\panda.h"
#include <queue>

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the ECUSIMDLL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// ECUSIMDLL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef ECUSIMDLL_EXPORTS
#define ECUSIMDLL_API __declspec(dllexport)
#else
#define ECUSIMDLL_API __declspec(dllimport)
#endif

// This class is exported from the ECUsim DLL.dll
class ECUSIMDLL_API ECUsim {
public:
	ECUsim(std::string sn, unsigned long can_baud, bool ext_addr = FALSE);
	ECUsim(panda::Panda && p, unsigned long can_baud, bool ext_addr = FALSE);
	~ECUsim();

	void stop();
	void join();

	// Flag determines if verbose output is enabled
	volatile bool verbose;
	BOOL ext_addr;
private:
	std::unique_ptr<panda::Panda> panda;

	static DWORD WINAPI _canthreadBootstrap(LPVOID This);
	DWORD can_recv_thread_function();

	BOOL _can_addr_matches(panda::PANDA_CAN_MSG & msg);

	void _CAN_process_msg(panda::PANDA_CAN_MSG & msg);

	std::string process_obd_msg(UCHAR mode, UCHAR pid, bool& return_data);

	HANDLE thread_can;
	volatile bool doloop;
	std::queue<uint8_t> can_multipart_data;

	BOOL can11b_enabled;
	BOOL can29b_enabled;
};
