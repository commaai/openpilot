#include "stdafx.h"
#include "MessageTx_CAN.h"
#include "J2534Connection_CAN.h"

MessageTx_CAN::MessageTx_CAN(
	std::shared_ptr<J2534Connection> connection_in,
	PASSTHRU_MSG& to_send
) : MessageTx(connection_in, to_send), sentyet(FALSE), txInFlight(FALSE) {};

void MessageTx_CAN::execute() {
	uint32_t addr = ((uint8_t)fullmsg.Data[0]) << 24 | ((uint8_t)fullmsg.Data[1]) << 16 |
		((uint8_t)fullmsg.Data[2]) << 8 | ((uint8_t)fullmsg.Data[3]);

	if (auto conn_sp = std::static_pointer_cast<J2534Connection_CAN>(this->connection.lock())) {
		if (auto panda_dev_sp = conn_sp->getPandaDev()) {
			auto payload = fullmsg.Data.substr(4);
			if (panda_dev_sp->panda->can_send(addr, check_bmask(this->fullmsg.TxFlags, CAN_29BIT_ID),
				(const uint8_t*)payload.c_str(), (uint8_t)payload.size(), panda::PANDA_CAN1) == FALSE) {
				return;
			}
			this->txInFlight = TRUE;
			this->sentyet = TRUE;
			panda_dev_sp->txMsgsAwaitingEcho.push(shared_from_this());
		}
	}
}

//Returns TRUE if receipt is consumed by the msg, FALSE otherwise.
BOOL MessageTx_CAN::checkTxReceipt(J2534Frame frame) {
	if (txReady()) return FALSE;
	if (frame.Data == fullmsg.Data && ((this->fullmsg.TxFlags & CAN_29BIT_ID) == (frame.RxStatus & CAN_29BIT_ID))) {
		txInFlight = FALSE;
		if (auto conn_sp = std::static_pointer_cast<J2534Connection_CAN>(this->connection.lock()))
			if (conn_sp->loopback)
				conn_sp->addMsgToRxQueue(frame);
		return TRUE;
	}
	return FALSE;
}

void MessageTx_CAN::reset() {
	sentyet = FALSE;
	txInFlight = FALSE;
}
