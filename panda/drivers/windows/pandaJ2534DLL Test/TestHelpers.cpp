#include "stdafx.h"
#include "TestHelpers.h"
#include "Loader4.h"
#include "pandaJ2534DLL/J2534_v0404.h"
#include "panda_shared/panda.h"
#include "Timer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void write_ioctl(unsigned int chanid, unsigned int param, unsigned int val, const __LineInfo* pLineInfo) {
	SCONFIG config = { param, val };
	SCONFIG_LIST inconfig = { 1, &config };

	Assert::AreEqual<long>(STATUS_NOERROR, PassThruIoctl(chanid, SET_CONFIG, &inconfig, NULL), _T("Failed to set IOCTL."), pLineInfo);
}

std::vector<panda::PANDA_CAN_MSG> panda_recv_loop_loose(std::unique_ptr<panda::Panda>& p, unsigned int min_num, unsigned long timeout_ms) {
	std::vector<panda::PANDA_CAN_MSG> ret_messages;
	Timer t = Timer();

	while (t.getTimePassed() < timeout_ms) {
		Sleep(10);
		std::vector<panda::PANDA_CAN_MSG>msg_recv = p->can_recv();
		if (msg_recv.size() > 0) {
			ret_messages.insert(std::end(ret_messages), std::begin(msg_recv), std::end(msg_recv));
		}
	}

	Assert::IsTrue(min_num <= ret_messages.size(), _T("Received too few messages."));
	return ret_messages;
}

std::vector<panda::PANDA_CAN_MSG> panda_recv_loop(std::unique_ptr<panda::Panda>& p, unsigned int num_expected, unsigned long timeout_ms) {
	std::vector<panda::PANDA_CAN_MSG> ret_messages;
	Timer t = Timer();

	while (t.getTimePassed() < timeout_ms) {
		Sleep(10);
		std::vector<panda::PANDA_CAN_MSG>msg_recv = p->can_recv();
		if (msg_recv.size() > 0) {
			ret_messages.insert(std::end(ret_messages), std::begin(msg_recv), std::end(msg_recv));
		}
		if (ret_messages.size() >= num_expected) break;
	}

	std::ostringstream stringStream;

	stringStream << "j2534_recv_loop Broke at " << t.getTimePassed() << " ms size is " << ret_messages.size() << std::endl;

	if (num_expected != ret_messages.size()) {
		stringStream << "Incorrect number of messages received. Displaying the messages:" << std::endl;
		for (auto msg : ret_messages) {
			stringStream << "    TS: " << msg.recv_time << "; Dat: ";
			for (int i = 0; i < msg.len; i++) stringStream << std::hex << std::setw(2) << std::setfill('0') << int(msg.dat[i] & 0xFF) << " ";
			stringStream << std::endl;
		}
	}

	Logger::WriteMessage(stringStream.str().c_str());

	Assert::AreEqual<unsigned long>(num_expected, ret_messages.size(), _T("Received wrong number of messages."));
	return ret_messages;
}

void check_panda_can_msg(panda::PANDA_CAN_MSG& msgin, uint8_t bus, unsigned long addr, bool addr_29b,
	bool is_receipt, std::string dat, const __LineInfo* pLineInfo) {
	Assert::AreEqual<uint8_t>(bus, msgin.bus, _T("Wrong msg bus"), pLineInfo);
	Assert::AreEqual<unsigned long>(addr, msgin.addr, _T("Wrong msg addr"), pLineInfo);
	Assert::AreEqual<bool>(addr_29b, msgin.addr_29b, _T("Wrong msg 29b flag"), pLineInfo);
	Assert::AreEqual<bool>(is_receipt, msgin.is_receipt, _T("Wrong msg receipt flag"), pLineInfo);

	std::ostringstream logmsg;
	logmsg << "Expected Hex (";
	for (int i = 0; i < dat.size(); i++) logmsg << std::hex << std::setw(2) << std::setfill('0') << int(dat[i] & 0xFF) << " ";
	logmsg << "); Actual Hex (";
	for (int i = 0; i < msgin.len; i++) logmsg << std::hex << std::setw(2) << std::setfill('0') << int(((char*)msgin.dat)[i] & 0xFF) << " ";
	logmsg << ")";
	Logger::WriteMessage(logmsg.str().c_str());

	Assert::AreEqual<size_t>(dat.size(), msgin.len, _T("Wrong msg len"), pLineInfo);
	Assert::AreEqual<std::string>(dat, std::string((char*)msgin.dat, msgin.len), _T("Wrong msg payload"), pLineInfo);
}

unsigned long J2534_start_periodic_msg_checked(unsigned long chanid, unsigned long ProtocolID, unsigned long TxFlags, unsigned long DataSize,
	unsigned long ExtraDataIndex, const char * Data, unsigned long TimeInterval, const __LineInfo * pLineInfo) {
	PASSTHRU_MSG msg = { ProtocolID, 0, TxFlags, 0, DataSize, ExtraDataIndex };
	memcpy_s(msg.Data, 4128, Data, DataSize);
	unsigned long msgID;
	Assert::AreEqual<long>(STATUS_NOERROR, J2534_start_periodic_msg(chanid, ProtocolID, TxFlags, DataSize,
		ExtraDataIndex, Data, TimeInterval, &msgID, pLineInfo), _T("Failed to start Periodic Message."), pLineInfo);
	return msgID;
}

unsigned long J2534_start_periodic_msg(unsigned long chanid, unsigned long ProtocolID, unsigned long TxFlags, unsigned long DataSize,
	unsigned long ExtraDataIndex, const char * Data, unsigned long TimeInterval, unsigned long* msgID, const __LineInfo * pLineInfo) {
	PASSTHRU_MSG msg = { ProtocolID, 0, TxFlags, 0, DataSize, ExtraDataIndex };
	memcpy_s(msg.Data, 4128, Data, DataSize);
	return PassThruStartPeriodicMsg(chanid, &msg, msgID, TimeInterval);
}

void J2534_send_msg_checked(unsigned long chanid, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long Timestamp, unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data, const __LineInfo* pLineInfo) {

	PASSTHRU_MSG msg = { ProtocolID, RxStatus, TxFlags, Timestamp, DataSize, ExtraDataIndex };
	memcpy_s(msg.Data, 4128, Data, DataSize);
	unsigned long msgcount = 1;
	Assert::AreEqual<long>(STATUS_NOERROR, PassThruWriteMsgs(chanid, &msg, &msgcount, 0), _T("Failed to write message."), pLineInfo);
	Assert::AreEqual<unsigned long>(1, msgcount, _T("Wrong message count after tx."), pLineInfo);
}

long J2534_send_msg(unsigned long chanid, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long Timestamp, unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data) {

	PASSTHRU_MSG msg = { ProtocolID, RxStatus, TxFlags, Timestamp, DataSize, ExtraDataIndex };
	memcpy_s(msg.Data, 4128, Data, DataSize);
	unsigned long msgcount = 1;
	return PassThruWriteMsgs(chanid, &msg, &msgcount, 0);
}

//Allow more messages to come in than the min.
std::vector<PASSTHRU_MSG> j2534_recv_loop_loose(unsigned int chanid, unsigned int min_num, unsigned long timeout_ms) {
	std::vector<PASSTHRU_MSG> ret_messages;
	PASSTHRU_MSG recvbuff[4] = {};
	Timer t = Timer();

	while (t.getTimePassed() < timeout_ms) {
		unsigned long msgcount = 4;
		unsigned int res = PassThruReadMsgs(chanid, recvbuff, &msgcount, 0);
		if (res == ERR_BUFFER_EMPTY) continue;
		Assert::IsFalse(msgcount > 4, _T("PassThruReadMsgs returned more data than the buffer could hold."));
		Assert::AreEqual<long>(STATUS_NOERROR, res, _T("Failed to read message."));
		if (msgcount > 0) {
			for (unsigned int i = 0; i < msgcount; i++) {
				ret_messages.push_back(recvbuff[i]);
			}
		}
	}

	Assert::IsTrue(min_num <= ret_messages.size(), _T("Received too few messages."));
	return ret_messages;
}

std::vector<PASSTHRU_MSG> j2534_recv_loop(unsigned int chanid, unsigned int num_expected, unsigned long timeout_ms) {
	std::vector<PASSTHRU_MSG> ret_messages;
	PASSTHRU_MSG recvbuff[4] = {};
	Timer t = Timer();

	while (t.getTimePassed() < timeout_ms) {
		unsigned long msgcount = 4;
		unsigned int res = PassThruReadMsgs(chanid, recvbuff, &msgcount, 0);
		if (res == ERR_BUFFER_EMPTY) continue;
		Assert::IsFalse(msgcount > 4, _T("PassThruReadMsgs returned more data than the buffer could hold."));
		Assert::AreEqual<long>(STATUS_NOERROR, res, _T("Failed to read message."));
		if (msgcount > 0) {
			for (unsigned int i = 0; i < msgcount; i++) {
				ret_messages.push_back(recvbuff[i]);
			}
		}
		if (ret_messages.size() >= num_expected) break;
	}

	std::ostringstream stringStream;
	stringStream << "j2534_recv_loop Broke at " << t.getTimePassed() << " ms size is " << ret_messages.size() << std::endl;

	if (num_expected != ret_messages.size()) {
		stringStream << "Incorrect number of messages received. Displaying the messages:" << std::endl;
		for (auto msg : ret_messages) {
			stringStream << "    TS: " << msg.Timestamp << "; Dat: ";
			for (int i = 0; i < msg.DataSize; i++) stringStream << std::hex << std::setw(2) << std::setfill('0') << int(msg.Data[i] & 0xFF) << " ";
			stringStream << std::endl;
		}
	}

	Logger::WriteMessage(stringStream.str().c_str());

	Assert::AreEqual<unsigned long>(num_expected, ret_messages.size(), _T("Received wrong number of messages."));
	return ret_messages;
}

void check_J2534_can_msg(PASSTHRU_MSG& msgin, unsigned long ProtocolID, unsigned long RxStatus, unsigned long TxFlags,
	unsigned long DataSize, unsigned long ExtraDataIndex, const char* Data, const __LineInfo* pLineInfo) {
	Assert::AreEqual<size_t>(DataSize, msgin.DataSize, _T("Wrong msg len"), pLineInfo);

	std::ostringstream logmsg;
	logmsg << "Expected Hex (";
	for (int i = 0; i < DataSize; i++) logmsg << std::hex << std::setw(2) << std::setfill('0') << int(Data[i] & 0xFF) << " ";
	logmsg << "); Actual Hex (";
	for (int i = 0; i < msgin.DataSize; i++) logmsg << std::hex << std::setw(2) << std::setfill('0') << int(((char*)msgin.Data)[i] & 0xFF) << " ";
	logmsg << ")";
	Logger::WriteMessage(logmsg.str().c_str());
	Assert::AreEqual<std::string>(std::string(Data, DataSize), std::string((char*)msgin.Data, msgin.DataSize), _T("Wrong msg payload"), pLineInfo);

	Assert::AreEqual<unsigned long>(ProtocolID, msgin.ProtocolID, _T("Wrong msg protocol"), pLineInfo);
	Assert::AreEqual<unsigned long>(RxStatus, msgin.RxStatus, _T("Wrong msg receipt rxstatus"), pLineInfo);
	Assert::AreEqual<unsigned long>(TxFlags, msgin.TxFlags, _T("Wrong msg receipt txflag"), pLineInfo);
	Assert::AreEqual<unsigned long>(ExtraDataIndex, msgin.ExtraDataIndex, _T("Wrong msg ExtraDataIndex"), pLineInfo);
}

unsigned long J2534_set_PASS_filter(unsigned long chanid, unsigned long ProtocolID, unsigned long tx,
	unsigned long len, char* mask, char* pattern, const __LineInfo* pLineInfo) {
	unsigned long filterid;
	PASSTHRU_MSG mask_msg = { ProtocolID, 0, tx, 0, len, 0, 0 };
	PASSTHRU_MSG pattern_msg = { ProtocolID, 0, tx, 0, len, 0, 0 };
	memcpy(mask_msg.Data, mask, len);
	memcpy(pattern_msg.Data, pattern, len);
	Assert::AreEqual<long>(STATUS_NOERROR, PassThruStartMsgFilter(chanid, PASS_FILTER, &mask_msg, &pattern_msg, NULL, &filterid),
		_T("Failed to create filter."), pLineInfo);
	return filterid;
}

unsigned long J2534_set_BLOCK_filter(unsigned long chanid, unsigned long ProtocolID, unsigned long tx,
	unsigned long len, char* mask, char* pattern, const __LineInfo* pLineInfo) {
	unsigned long filterid;
	PASSTHRU_MSG mask_msg = { ProtocolID, 0, tx, 0, len, 0, 0 };
	PASSTHRU_MSG pattern_msg = { ProtocolID, 0, tx, 0, len, 0, 0 };
	memcpy(mask_msg.Data, mask, len);
	memcpy(pattern_msg.Data, pattern, len);
	Assert::AreEqual<long>(STATUS_NOERROR, PassThruStartMsgFilter(chanid, BLOCK_FILTER, &mask_msg, &pattern_msg, NULL, &filterid),
		_T("Failed to create filter."), pLineInfo);
	return filterid;
}

unsigned long J2534_set_flowctrl_filter(unsigned long chanid, unsigned long tx,
	unsigned long len, char* mask, char* pattern, char* flow, const __LineInfo* pLineInfo) {
	unsigned long filterid;
	PASSTHRU_MSG mask_msg = { ISO15765, 0, tx, 0, len, 0, 0 };
	PASSTHRU_MSG pattern_msg = { ISO15765, 0, tx, 0, len, 0, 0 };
	PASSTHRU_MSG flow_msg = { ISO15765, 0, tx, 0, len, 0, 0 };
	memcpy(mask_msg.Data, mask, len);
	memcpy(pattern_msg.Data, pattern, len);
	memcpy(flow_msg.Data, flow, len);
	Assert::AreEqual<long>(STATUS_NOERROR, PassThruStartMsgFilter(chanid, FLOW_CONTROL_FILTER, &mask_msg, &pattern_msg, &flow_msg, &filterid),
		_T("Failed to create filter."), pLineInfo);
	return filterid;
}

std::unique_ptr<panda::Panda> getPanda(unsigned long kbaud, BOOL loopback) {
	auto p = panda::Panda::openPanda("");
	Assert::IsTrue(p != nullptr, _T("Could not open raw panda device to test communication."));
	p->set_can_speed_kbps(panda::PANDA_CAN1, kbaud);
	p->set_safety_mode(panda::SAFETY_ALLOUTPUT);
	p->set_can_loopback(loopback);
	p->can_clear(panda::PANDA_CAN_RX);
	return p;
}

std::vector<panda::PANDA_CAN_MSG> checked_panda_send(std::unique_ptr<panda::Panda>& p, uint32_t addr, bool is_29b,
	char* msg, uint8_t len, unsigned int num_expected, const __LineInfo* pLineInfo, unsigned long timeout_ms) {
	Assert::IsTrue(p->can_send(addr, is_29b, (const uint8_t*)msg, len, panda::PANDA_CAN1), _T("Panda send says it failed."), pLineInfo);
	auto panda_msg_recv = panda_recv_loop(p, 1 + num_expected, timeout_ms);
	check_panda_can_msg(panda_msg_recv[0], 0, addr, is_29b, TRUE, std::string(msg, len), pLineInfo);
	panda_msg_recv.erase(panda_msg_recv.begin());
	return panda_msg_recv;
}
