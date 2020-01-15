// panda_playground.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "pandaJ2534DLL Test\Loader4.h"
#include "ECUsim DLL\ECUsim.h"
#include <chrono>


int _tmain(int Argc, _TCHAR *Argv) {
	UNREFERENCED_PARAMETER(Argc);
	UNREFERENCED_PARAMETER(Argv);

	ECUsim sim("", 500000);

	//if (LoadJ2534Dll("C:\\WINDOWS\\SysWOW64\\op20pt32.dll") != 0) {
	if (LoadJ2534Dll("pandaJ2534.dll") != 0) {
		auto err = GetLastError();
		return 1;
	}
	unsigned long did, cid, fid;
	PassThruOpen("", &did);
	PassThruConnect(did, ISO15765, CAN_29BIT_ID, 500000, &cid);

	PASSTHRU_MSG mask, pattern, flow;

	memcpy(mask.Data, "\xff\xff\xff\xff", 4);
	mask.DataSize = 4;
	mask.ProtocolID = ISO15765;
	mask.TxFlags = CAN_29BIT_ID;
	mask.ExtraDataIndex = 0;
	mask.RxStatus = 0;

	////////////////////////18//DA//F1//EF
	memcpy(pattern.Data, "\x18\xda\xf1\xef", 4);
	pattern.DataSize = 4;
	pattern.ProtocolID = ISO15765;
	pattern.TxFlags = CAN_29BIT_ID;
	pattern.ExtraDataIndex = 0;
	pattern.RxStatus = 0;

	memcpy(flow.Data, "\x18\xda\xef\xf1", 4);
	flow.DataSize = 4;
	flow.ProtocolID = ISO15765;
	flow.TxFlags = CAN_29BIT_ID;
	flow.ExtraDataIndex = 0;
	flow.RxStatus = 0;

	auto res = PassThruStartMsgFilter(cid, FLOW_CONTROL_FILTER, &mask, &pattern, &flow, &fid);
	if (res != STATUS_NOERROR)
		return 1;

	SCONFIG_LIST list;
	SCONFIG config;
	config.Parameter = LOOPBACK;
	config.Value = 0;
	list.ConfigPtr = &config;
	list.NumOfParams = 1;

	res = PassThruIoctl(cid, SET_CONFIG, &list, NULL);
	if (res != STATUS_NOERROR)
		return 1;

	PASSTHRU_MSG outmsg;
	memcpy(outmsg.Data, "\x18\xda\xef\xf1""\xAA\xBB\xCC\xDD\xEE\xFF\x11\x22\x33\x44", 4 + 10);
	outmsg.DataSize = 4 + 10;
	outmsg.ProtocolID = ISO15765;
	outmsg.TxFlags = CAN_29BIT_ID;
	outmsg.ExtraDataIndex = 0;
	outmsg.RxStatus = 0;

	unsigned long msgoutcount = 1;

	res = PassThruWriteMsgs(cid, &outmsg, &msgoutcount, 0);
	if (res != STATUS_NOERROR)
		return 1;

	PASSTHRU_MSG inmsg[8];
	unsigned long msgincount = 8;

	res = PassThruReadMsgs(cid, inmsg, &msgincount, 1000);
	if (res != STATUS_NOERROR)
		return 1;

	return 0;
}
