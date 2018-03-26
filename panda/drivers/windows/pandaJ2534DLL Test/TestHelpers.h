#pragma once
#include "stdafx.h"
#include "pandaJ2534DLL/J2534_v0404.h"
#include "panda/panda.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

extern void write_ioctl(unsigned int chanid, unsigned int param, unsigned int val, const __LineInfo* pLineInfo = NULL);

extern std::vector<panda::PANDA_CAN_MSG> panda_recv_loop_loose(std::unique_ptr<panda::Panda>& p, unsigned int min_num, unsigned long timeout_ms = 100);

extern std::vector<panda::PANDA_CAN_MSG> panda_recv_loop(std::unique_ptr<panda::Panda>& p, unsigned int num_expected, unsigned long timeout_ms = 100);

extern void check_panda_can_msg(panda::PANDA_CAN_MSG& msgin, uint8_t bus, unsigned long addr, bool addr_29b,
	bool is_receipt, std::string dat, const __LineInfo* pLineInfo = NULL);

extern unsigned long J2534_start_periodic_msg_checked(unsigned long chanid, unsigned long ProtocolID, unsigned long TxFlags, unsigned long DataSize,
	unsigned long ExtraDataIndex, const char * Data, unsigned long TimeInterval, const __LineInfo * pLineInfo);

extern unsigned long J2534_start_periodic_msg(unsigned long chanid, unsigned long ProtocolID, unsigned long TxFlags, unsigned long DataSize,
	unsigned long ExtraDataIndex, const char* Data, unsigned long TimeInterval, unsigned long* msgID, const __LineInfo* pLineInfo = NULL);

extern void J2534_send_msg_checked(unsigned long chanid, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long Timestamp, unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data, const __LineInfo* pLineInfo = NULL);

extern long J2534_send_msg(unsigned long chanid, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long Timestamp, unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data);

extern std::vector<PASSTHRU_MSG> j2534_recv_loop_loose(unsigned int chanid, unsigned int min_num, unsigned long timeout_ms = 100);

extern std::vector<PASSTHRU_MSG> j2534_recv_loop(unsigned int chanid, unsigned int num_expected, unsigned long timeout_ms = 100);

extern void check_J2534_can_msg(PASSTHRU_MSG& msgin, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data, const __LineInfo* pLineInfo = NULL);

extern unsigned long J2534_set_PASS_filter(unsigned long chanid, unsigned long ProtocolID, unsigned long tx,
	unsigned long len, char* mask, char* pattern, const __LineInfo* pLineInfo = NULL);

extern unsigned long J2534_set_BLOCK_filter(unsigned long chanid, unsigned long ProtocolID, unsigned long tx,
	unsigned long len, char* mask, char* pattern, const __LineInfo* pLineInfo = NULL);

extern unsigned long J2534_set_flowctrl_filter(unsigned long chanid, unsigned long tx,
	unsigned long len, char* mask, char* pattern, char* flow, const __LineInfo* pLineInfo = NULL);

extern std::unique_ptr<panda::Panda> getPanda(unsigned long kbaud = 500, BOOL loopback = FALSE);

extern std::vector<panda::PANDA_CAN_MSG> checked_panda_send(std::unique_ptr<panda::Panda>& p, uint32_t addr, bool is_29b,
	char* msg, uint8_t len, unsigned int num_expected=0, const __LineInfo* pLineInfo = NULL, unsigned long timeout_ms = 100);
