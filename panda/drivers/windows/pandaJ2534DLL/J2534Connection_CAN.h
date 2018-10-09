#pragma once

#include "J2534Connection.h"
#include "panda_shared/panda.h"

#define val_is_29bit(num) check_bmask(num, CAN_29BIT_ID)

class J2534Connection_CAN : public J2534Connection {
public:
	J2534Connection_CAN(
		std::shared_ptr<PandaJ2534Device> panda_dev,
		unsigned long ProtocolID,
		unsigned long Flags,
		unsigned long BaudRate
	);

	virtual unsigned long validateTxMsg(PASSTHRU_MSG* msg);

	virtual std::shared_ptr<MessageTx> parseMessageTx(PASSTHRU_MSG& pMsg);

	virtual void setBaud(unsigned long baud);

	virtual unsigned long getMinMsgLen() {
		return 4;
	}

	virtual unsigned long getMaxMsgLen() {
		return 12;
	}

	virtual unsigned long getMaxMsgSingleFrameLen() {
		return 12;
	}

	virtual bool isProtoCan() {
		return TRUE;
	}

	bool _is_29bit() {
		return (this->Flags & CAN_29BIT_ID) == CAN_29BIT_ID;
	}

};