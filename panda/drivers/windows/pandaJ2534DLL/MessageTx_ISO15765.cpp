#include "stdafx.h"
#include "MessageTx_ISO15765.h"
#include "constants_ISO15765.h"

//in microseconsa
#define TIMEOUT_FC 250000 //Flow Control
#define TIMEOUT_CF 250000 //Consecutive Frames

MessageTx_ISO15765::MessageTx_ISO15765(
	std::shared_ptr<J2534Connection> connection_in,
	PASSTHRU_MSG& to_send,
	std::shared_ptr<J2534MessageFilter> filter
) : MessageTxTimeoutable(connection_in, to_send), filter(filter), frames_sent(0),
consumed_count(0), txInFlight(FALSE), sendAll(FALSE), block_size(0), numWaitFrames(0), didtimeout(FALSE), issuspended(FALSE){

	CANid = ((uint8_t)fullmsg.Data[0]) << 24 | ((uint8_t)fullmsg.Data[1]) << 16 |
		((uint8_t)fullmsg.Data[2]) << 8 | ((uint8_t)fullmsg.Data[3]);

	payload = fullmsg.Data.substr(addressLength());

	if (check_bmask(fullmsg.TxFlags, ISO15765_ADDR_TYPE))
		data_prefix = fullmsg.Data[4];

	if (payload.size() <= (7 - data_prefix.size())) {
		isMultipart = FALSE;
		auto framepayload = data_prefix + std::string(1, (char)payload.size()) + payload;
		if (check_bmask(this->fullmsg.TxFlags, ISO15765_FRAME_PAD))
			framepayload += std::string(8 - framepayload.size(), '\x00');
		framePayloads.push_back(framepayload);
	} else {
		isMultipart = TRUE;
		unsigned long first_payload_len = 6 - data_prefix.size(); // 5 or 6
		std::string framepayload = data_prefix +
			(char)(0x10 | ((payload.size() >> 8) & 0xF)) +
			(char)(payload.size() & 0xFF) +
			payload.substr(0, first_payload_len);
		framePayloads.push_back(framepayload);

		unsigned int pktnum = 1;
		uint8_t CFDatSize = 7 - data_prefix.size();
		while (TRUE) {
			framepayload = data_prefix + (char)(0x20 | (pktnum % 0x10)) +
				payload.substr(first_payload_len + (CFDatSize * (pktnum-1)), CFDatSize);

			if (check_bmask(this->fullmsg.TxFlags, ISO15765_FRAME_PAD))
				framepayload += std::string(8 - framepayload.size(), '\x00');
			framePayloads.push_back(framepayload);
			if (first_payload_len + (CFDatSize * pktnum) >= payload.size()) break;
			pktnum++;
		}

	}
};

unsigned int MessageTx_ISO15765::addressLength() {
	return check_bmask(fullmsg.TxFlags, ISO15765_ADDR_TYPE) ? 5 : 4;
}

void MessageTx_ISO15765::execute() {
	if (didtimeout || issuspended) return;
	if (this->frames_sent >= this->framePayloads.size()) return;
	if (block_size == 0 && !sendAll && this->frames_sent > 0) return;
	if (block_size > 0 && !sendAll) block_size--;

	if (auto conn_sp = this->connection.lock()) {
		if (auto panda_dev_sp = conn_sp->getPandaDev()) {
			auto& outFramePayload = this->framePayloads[this->frames_sent];
			if (panda_dev_sp->panda->can_send(this->CANid, check_bmask(this->fullmsg.TxFlags, CAN_29BIT_ID),
				(const uint8_t*)outFramePayload.c_str(), (uint8_t)outFramePayload.size(), panda::PANDA_CAN1) == FALSE) {
				return;
			}

			this->txInFlight = TRUE;
			this->frames_sent++;
			panda_dev_sp->txMsgsAwaitingEcho.push(shared_from_this());
		}
	}
}

//Returns TRUE if receipt is consumed by the msg, FALSE otherwise.
BOOL MessageTx_ISO15765::checkTxReceipt(J2534Frame frame) {
	if (!txInFlight) return FALSE;
	if (frame.Data.size() >= addressLength() + 1 && (frame.Data[addressLength()] & 0xF0) == FRAME_FLOWCTRL) return FALSE;

	if (frame.Data == fullmsg.Data.substr(0, 4) + framePayloads[frames_sent - 1] &&
		((this->fullmsg.TxFlags & CAN_29BIT_ID) == (frame.RxStatus & CAN_29BIT_ID))) { //Check receipt is expected
		txInFlight = FALSE; //Received the expected receipt. Allow another msg to be sent.

		if (this->recvCount == 0 && this->framePayloads.size() > 1)
			scheduleTimeout(TIMEOUT_FC);

		if (frames_sent == framePayloads.size()) { //Check message done
			if (auto conn_sp = std::static_pointer_cast<J2534Connection_ISO15765>(this->connection.lock())) {
				unsigned long flags = (filter == nullptr) ? fullmsg.TxFlags : this->filter->flags;

				J2534Frame outframe(ISO15765);
				outframe.Timestamp = frame.Timestamp;
				outframe.RxStatus = TX_INDICATION | (flags & (ISO15765_ADDR_TYPE | CAN_29BIT_ID));
				outframe.Data = frame.Data.substr(0, addressLength());
				conn_sp->addMsgToRxQueue(outframe);

				if (conn_sp->loopback) {
					J2534Frame outframe(ISO15765);
					outframe.Timestamp = frame.Timestamp;
					outframe.RxStatus = TX_MSG_TYPE | (flags & (ISO15765_ADDR_TYPE | CAN_29BIT_ID));
					outframe.Data = this->fullmsg.Data;
					conn_sp->addMsgToRxQueue(outframe);
				}

			} //TODO what if fails
		} else {
			//Restart timeout if we are waiting for a flow control frame.
			//FC frames are required when we are not sending all, the
			//current block_size batch has not been sent, a FC message has
			//already been received (differentiating from first frame), the
			//message is not finished, and there is more than one frame in
			//the message.
			if (block_size == 0 && recvCount != 0 && !sendAll && !this->isFinished() && this->framePayloads.size() > 1)
				scheduleTimeout(TIMEOUT_CF);
		}
		return TRUE;
	}
	return FALSE;
}

BOOL MessageTx_ISO15765::isFinished() {
	return this->frames_sent == this->framePayloads.size() && !txInFlight;
}

BOOL MessageTx_ISO15765::txReady() {
	return block_size > 0 || sendAll || this->frames_sent == 0;
}

void MessageTx_ISO15765::reset() {
	frames_sent = 0;
	consumed_count = 0;
	block_size = 0;
	txInFlight = FALSE;
	sendAll = FALSE;
	numWaitFrames = 0;
	didtimeout = FALSE;
}

void MessageTx_ISO15765::onTimeout() {
	didtimeout = TRUE;
	if (auto conn_sp = std::static_pointer_cast<J2534Connection_ISO15765>(this->connection.lock())) {
		if (auto panda_dev_sp = conn_sp->getPandaDev()) {
			panda_dev_sp->removeConnectionTopAction(conn_sp, shared_from_this());
		}
	}
}

void MessageTx_ISO15765::flowControlContinue(uint8_t block_size, std::chrono::microseconds separation_time) {
	this->issuspended = FALSE;
	this->block_size = block_size;
	this->delay = separation_time;
	this->sendAll = block_size == 0;
	this->recvCount++;
}

void MessageTx_ISO15765::flowControlWait(unsigned long N_WFTmax) {
	this->issuspended = TRUE;
	this->recvCount++;
	this->numWaitFrames++;
	this->sendAll = FALSE;
	this->block_size = block_size;
	this->delay = std::chrono::microseconds(0);
	//Docs are vague on if 0 means NO WAITS ALLOWED or NO LIMIT TO WAITS.
	//It is less likely to cause issue if NO LIMIT is assumed.
	if (N_WFTmax > 0 && this->numWaitFrames > N_WFTmax) {
		this->onTimeout(); //Trigger self destruction of message.
	} else {
		scheduleTimeout(TIMEOUT_FC);
	}
}

void MessageTx_ISO15765::flowControlAbort() {
	this->recvCount++; //Invalidate future timeout actions.
	this->onTimeout(); //Trigger self destruction of message.
}
