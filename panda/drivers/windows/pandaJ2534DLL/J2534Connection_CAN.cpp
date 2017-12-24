#include "stdafx.h"
#include "J2534Connection_CAN.h"
#include "MessageTx_CAN.h"
#include "Timer.h"

J2534Connection_CAN::J2534Connection_CAN(
		std::shared_ptr<PandaJ2534Device> panda_dev,
		unsigned long ProtocolID,
		unsigned long Flags,
		unsigned long BaudRate
	) : J2534Connection(panda_dev, ProtocolID, Flags, BaudRate) {
	this->port = 0;

	if (BaudRate % 100 || BaudRate < 10000 || BaudRate > 5000000)
		throw ERR_INVALID_BAUDRATE;

	panda_dev->panda->set_can_speed_cbps(panda::PANDA_CAN1, BaudRate/100);
};

unsigned long J2534Connection_CAN::validateTxMsg(PASSTHRU_MSG* msg) {
	if ((msg->DataSize < this->getMinMsgLen() || msg->DataSize > this->getMaxMsgLen() ||
		(val_is_29bit(msg->TxFlags) != this->_is_29bit() && !check_bmask(this->Flags, CAN_ID_BOTH))))
		return ERR_INVALID_MSG;
	return STATUS_NOERROR;
}

std::shared_ptr<MessageTx> J2534Connection_CAN::parseMessageTx(PASSTHRU_MSG& msg) {
	return std::dynamic_pointer_cast<MessageTx>(std::make_shared<MessageTx_CAN>(shared_from_this(), msg));
}

void J2534Connection_CAN::setBaud(unsigned long BaudRate) {
	if (auto panda_dev = this->getPandaDev()) {
		if (BaudRate % 100 || BaudRate < 10000 || BaudRate > 5000000)
			throw ERR_NOT_SUPPORTED;

		panda_dev->panda->set_can_speed_cbps(panda::PANDA_CAN1, (uint16_t)(BaudRate / 100));
		return J2534Connection::setBaud(BaudRate);
	} else {
		throw ERR_DEVICE_NOT_CONNECTED;
	}
}
