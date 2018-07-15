#include "stdafx.h"
#include "ECUsim.h"

ECUsim::ECUsim(std::string sn, unsigned long can_baud, bool ext_addr) :
	doloop(TRUE), verbose(TRUE), can11b_enabled(TRUE), can29b_enabled(TRUE), ext_addr(ext_addr){
	this->panda = panda::Panda::openPanda(sn);
	this->panda->set_can_speed_cbps(panda::PANDA_CAN1, can_baud / 100); //Don't pass in baud where baud%100 != 0
	this->panda->set_safety_mode(panda::SAFETY_ALLOUTPUT);
	this->panda->set_can_loopback(FALSE);
	this->panda->can_clear(panda::PANDA_CAN_RX);

	DWORD threadid;
	this->thread_can = CreateThread(NULL, 0, _canthreadBootstrap, (LPVOID)this, 0, &threadid);
}

ECUsim::~ECUsim() {
	this->stop();
	this->join();
}

void ECUsim::stop() {
	this->doloop = FALSE;
}

void ECUsim::join() {
	WaitForSingleObject(this->thread_can, INFINITE);
}

DWORD WINAPI ECUsim::_canthreadBootstrap(LPVOID This) {
	return ((ECUsim*)This)->can_recv_thread_function();
}

DWORD ECUsim::can_recv_thread_function() {
	while (this->doloop) {
		auto msgs = this->panda->can_recv();
		for (auto& msg : msgs) {
			if (msg.is_receipt) continue;
			if (msg.bus == 0 && !msg.is_receipt /*&& msg.len == 8*/ && msg.dat[0] >= 2) {
				if (this->verbose) {
					printf("Processing message (bus: %d; addr: %X; 29b: %d):\n    ", msg.bus, msg.addr, msg.addr_29b);
					for (int i = 0; i < msg.len; i++) printf("%02X ", msg.dat[i]);
					printf("\n");
				}
				this->_CAN_process_msg(msg);
			} else {
				if (this->verbose) {
					printf("Rejecting message (bus: %d; addr: %X; 29b: %d):\n    ", msg.bus, msg.addr, msg.addr_29b);
					for (int i = 0; i < msg.len; i++) printf("%02X ", msg.dat[i]);
					printf("\n");
				}
			}
		}
	}

	return 0;
}

BOOL ECUsim::_can_addr_matches(panda::PANDA_CAN_MSG& msg) {
	if (this->can11b_enabled && !msg.addr_29b && (msg.addr == 0x7DF || (msg.addr & 0x7F8) == 0x7E0)) {
		if (!this->ext_addr) {
			return TRUE;
		} else {
			return msg.len >= 1 && msg.dat[0] == 0x13;//13 is an arbitrary address picked to test ext addresses
		}
	}
	if (this->can29b_enabled && msg.addr_29b && ((msg.addr & 0x1FFF00FF) == 0x18DB00F1 || (msg.addr & 0x1FFF00FF) == 0x18da00f1)) {
		if (!this->ext_addr) {
			return TRUE;
		} else {
			return msg.len >= 1 && msg.dat[0] == 0x13;//13 is an arbitrary address picked to test ext addresses
		}
	}
	return FALSE;
}

void ECUsim::_CAN_process_msg(panda::PANDA_CAN_MSG& msg) {
	std::string outmsg;
	uint32_t outaddr;
	uint8_t formatted_msg_buff[8];
	bool doreply = FALSE;

	if (this->_can_addr_matches(msg)) {// && msg.len == 8) {
		uint8_t *dat = (this->ext_addr) ? &msg.dat[1] : &msg.dat[0];
		if ((dat[0] & 0xF0) == 0x10) {
			printf("Got a multiframe write request\n");
			outaddr = (msg.addr_29b) ? 0x18DAF1EF : 0x7E8;
			this->panda->can_send(outaddr, msg.addr_29b, (const uint8_t*)"\x30\x00\x00", 3, panda::PANDA_CAN1);
			return;
		}

		/////////// Check if Flow Control Msg
		if ((dat[0] & 0xF0) == 0x30 && msg.len >= 3 && this->can_multipart_data.size() > 0) {
			if (this->verbose) printf("More data requested\n");
			uint8_t block_size = dat[1], sep_time_min = dat[2];
			outaddr = (msg.addr == 0x7DF || msg.addr == 0x7E0) ? 0x7E8 : 0x18DAF1EF; //ext addr 5th byte is just always 0x13 for simplicity

			unsigned int msgnum = 1;
			while (this->can_multipart_data.size()) {
				unsigned int datalen = this->ext_addr ?
					min(6, this->can_multipart_data.size()): //EXT ADDR VALUE
					min(7, this->can_multipart_data.size()); //NORMAL ADDR VALUE

				unsigned int idx = 0;
				if (this->ext_addr)
					formatted_msg_buff[idx++] = 0x13; //EXT ADDR
				formatted_msg_buff[idx++] = 0x20 | msgnum;
				for (int i = 0; i < datalen; i++) {
					formatted_msg_buff[i + idx] = this->can_multipart_data.front();
					this->can_multipart_data.pop();
				}
				for (int i = datalen + idx; i < sizeof(formatted_msg_buff); i++)
					formatted_msg_buff[i] = 0;

				if (this->verbose) {
					printf("Multipart reply to %X.\n    ", outaddr);
					for (int i = 0; i < datalen + idx; i++) printf("%02X ", formatted_msg_buff[i]);
					printf("\n");
				}

				this->panda->can_send(outaddr, msg.addr_29b, formatted_msg_buff, datalen + idx, panda::PANDA_CAN1);
				msgnum = (msgnum + 1) % 0x10;
				Sleep(10);
			}
			return;
		}

		/////////// Normal message in
		outmsg = this->process_obd_msg(dat[1], dat[2], doreply);
		if (doreply) {
			outaddr = (msg.addr_29b) ? 0x18DAF1EF : 0x7E8;

			if (outmsg.size() <= (this->ext_addr ? 4 : 5)) {
				unsigned int idx = 0;
				if(this->ext_addr)
					formatted_msg_buff[idx++] = 0x13; //EXT ADDR
				formatted_msg_buff[idx++] = outmsg.size() + 2;
				formatted_msg_buff[idx++] = 0x40 | dat[1];
				formatted_msg_buff[idx++] = dat[2]; //PID
				memcpy_s(&formatted_msg_buff[idx], sizeof(formatted_msg_buff) - idx, outmsg.c_str(), outmsg.size());
				for (int i = idx + outmsg.size(); i < 8; i++)
					formatted_msg_buff[i] = 0;

				if (this->verbose) {
					printf("Replying to %X.\n    ", outaddr);
					for (int i = 0; i < 8; i++) printf("%02X ", formatted_msg_buff[i]);
					printf("\n");
				}

				this->panda->can_send(outaddr, msg.addr_29b, formatted_msg_buff, 8, panda::PANDA_CAN1); //outmsg.size() + 3
			} else {
				uint8_t first_msg_len = this->ext_addr ?
					min(2, outmsg.size() % 7) : //EXT ADDR VALUES
					min(3, outmsg.size() % 7);  //NORMAL ADDR VALUES
				uint8_t payload_len = outmsg.size() + 3;

				unsigned int idx = 0;
				if (this->ext_addr)
					formatted_msg_buff[idx++] = 0x13; //EXT ADDR
				formatted_msg_buff[idx++] = 0x10 | ((payload_len >> 8) & 0xF);
				formatted_msg_buff[idx++] = payload_len & 0xFF;
				formatted_msg_buff[idx++] = 0x40 | dat[1];
				formatted_msg_buff[idx++] = dat[2]; //PID
				formatted_msg_buff[idx++] = 1;
				memcpy_s(&formatted_msg_buff[idx], sizeof(formatted_msg_buff) - idx, outmsg.c_str(), first_msg_len);

				if (this->verbose) {
					printf("Replying FIRST FRAME to %X.\n    ", outaddr);
					for (int i = 0; i < 8; i++) printf("%02X ", formatted_msg_buff[i]);
					printf("\n");
				}

				this->panda->can_send(outaddr, msg.addr_29b, formatted_msg_buff, 8, panda::PANDA_CAN1);
				for (int i = first_msg_len; i < outmsg.size(); i++)
					this->can_multipart_data.push(outmsg[i]);
			}
		}
	}
}

std::string ECUsim::process_obd_msg(UCHAR mode, UCHAR pid, bool& return_data) {
	std::string tmp;
	return_data = TRUE;

	switch (mode) {
	case 0x01: // Mode : Show current data
		switch (pid) {
		case 0x00: //List supported things
			return "\xff\xff\xff\xfe"; //b"\xBE\x1F\xB8\x10" #Bitfield, random features
		case 0x01: // Monitor Status since DTC cleared
			return std::string("\x00\x00\x00\x00", 4); //Bitfield, random features
		case 0x04: // Calculated engine load
			return "\x2f";
		case 0x05: // Engine coolant temperature
			return "\x3c";
		case 0x0B: // Intake manifold absolute pressure
			return "\x90";
		case 0x0C: // Engine RPM
			return "\x1A\xF8";
		case 0x0D: // Vehicle Speed
			return "\x53";
		case 0x10: // MAF air flow rate
			return "\x01\xA0";
		case 0x11: // Throttle Position
			return "\x90";
		case 0x33: // Absolute Barometric Pressure
			return "\x90";
		default:
			return_data = FALSE;
			return "";
		}
	case 0x09: // Mode : Request vehicle information
		switch (pid) {
		case 0x02: // Show VIN
			return "1D4GP00R55B123456";
		case 0xFC: // test long multi message.Ligned up for LIN responses
			for (int i = 0; i < 80; i++) {
				tmp += "\xAA\xAA";
			}
			return tmp;//">BBH", 0xAA, 0xAA, num + 1)
		case 0xFD: // test long multi message
			for (int i = 0; i < 80; i++) {
				tmp += "\xAA\xAA\xAA";
				tmp.push_back(i >> 24);
				tmp.push_back((i >> 16) & 0xFF);
				tmp.push_back((i >> 8) & 0xFF);
				tmp.push_back(i & 0xFF);
			}
			return "\xAA\xAA\xAA" + tmp;
		case 0xFE: // test very long multi message
			tmp = "\xAA\xAA\xAA";
			for (int i = 0; i < 584; i++) {
				tmp += "\xAA\xAA\xAA";
				tmp.push_back(i >> 24);
				tmp.push_back((i >> 16) & 0xFF);
				tmp.push_back((i >> 8) & 0xFF);
				tmp.push_back(i & 0xFF);
			}
			return tmp + "\xAA";
		case 0xFF:
			for (int i = 0; i < 584; i++) {
				tmp += "\xAA\xAA\xAA\xAA\xAA";
				tmp.push_back(((i + 1) >> 8) & 0xFF);
				tmp.push_back((i + 1) & 0xFF);
			}
			return std::string("\xAA\x00\x00", 3) + tmp;
		default:
			return_data = FALSE;
			return "";
		}
	case 0x3E:
		if (pid == 0) {
			return_data = TRUE;
			return "";
		}
		return_data = FALSE;
		return "";
	default:
		return_data = FALSE;
		return "";
	}
}
