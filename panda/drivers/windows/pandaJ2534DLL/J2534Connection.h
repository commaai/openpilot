#pragma once
#include "panda_shared/panda.h"
#include "J2534_v0404.h"
#include "synchronize.h"
#include "J2534Frame.h"
#include "PandaJ2534Device.h"
#include "J2534MessageFilter.h"
#include "MessagePeriodic.h"

class J2534Frame;
class Action;
class PandaJ2534Device;
class J2534MessageFilter;

#define check_bmask(num, mask)(((num) & mask) == mask)

/**
Class representing a generic J2534 Connection created by PassThruConnect,
and is associated with a channelID given to the J2534 API user.
Subclasses implement specific J2534 supported protocols.
*/
class J2534Connection : public std::enable_shared_from_this<J2534Connection> {
	friend class PandaJ2534Device;

public:
	J2534Connection(
		std::shared_ptr<PandaJ2534Device> panda_dev,
		unsigned long ProtocolID,
		unsigned long Flags,
		unsigned long BaudRate
	);
	virtual ~J2534Connection() {};

	//J2534 API functions

	virtual long PassThruReadMsgs(PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
	long PassThruWriteMsgs(PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout);
	virtual long PassThruStartPeriodicMsg(PASSTHRU_MSG *pMsg, unsigned long *pMsgID, unsigned long TimeInterval);
	virtual long PassThruStopPeriodicMsg(unsigned long MsgID);

	virtual long PassThruStartMsgFilter(unsigned long FilterType, PASSTHRU_MSG *pMaskMsg, PASSTHRU_MSG *pPatternMsg,
		PASSTHRU_MSG *pFlowControlMsg, unsigned long *pFilterID);

	virtual long PassThruStopMsgFilter(unsigned long FilterID);
	virtual long PassThruIoctl(unsigned long IoctlID, void *pInput, void *pOutput);

	//Functions for parsing messages to be send with PassThruWriteMsgs.

	virtual unsigned long  validateTxMsg(PASSTHRU_MSG* msg);
	virtual std::shared_ptr<MessageTx> parseMessageTx(PASSTHRU_MSG& msg) { return nullptr; };

	//IOCTL functions

	virtual long init5b(SBYTE_ARRAY* pInput, SBYTE_ARRAY* pOutput);
	virtual long initFast(PASSTHRU_MSG* pInput, PASSTHRU_MSG* pOutput);
	long clearTXBuff();
	long clearRXBuff();
	long clearPeriodicMsgs();
	long clearMsgFilters();

	virtual void setBaud(unsigned long baud);
	virtual void setParity(unsigned long parity);

	unsigned long getBaud() {
		return this->BaudRate;
	}

	unsigned long getProtocol() {
		return this->ProtocolID;
	};

	virtual bool isProtoCan() {
		return FALSE;
	}

	//Port is used in a protocol specific way to differentiate tranceivers.
	unsigned long getPort() {
		return this->port;
	}

	virtual void processIOCTLSetConfig(unsigned long Parameter, unsigned long Value);

	virtual unsigned long processIOCTLGetConfig(unsigned long Parameter);

	//Called when the passthru device has received a message for this connection
	//Loopback messages are processed separately.
	virtual void processMessage(const J2534Frame& msg);

	//Limitations on message size. Override in every subclass.

	virtual unsigned long getMinMsgLen() {
		return 1;
	}

	virtual unsigned long getMaxMsgLen() {
		return 4128;
	}

	virtual unsigned long getMaxMsgSingleFrameLen() {
		return 12;
	}

	//Add an Action to the Task Queue for future processing.
	//The task should be set its expire time before being submitted.
	void schedultMsgTx(std::shared_ptr<Action> msgout);

	void rescheduleExistingTxMsgs();

	std::shared_ptr<PandaJ2534Device> getPandaDev() {
		if (auto panda_dev_sp = this->panda_dev.lock())
			return panda_dev_sp;
		return nullptr;
	}

	//Add a message to the queue read by PassThruReadMsgs().
	void addMsgToRxQueue(const J2534Frame& frame) {
		synchronized(messageRxBuff_mutex) {
			messageRxBuff.push(frame);
		}
	}

	bool loopback = FALSE;

protected:
	unsigned long ProtocolID;
	unsigned long Flags;
	unsigned long BaudRate;
	unsigned long Parity;
	unsigned long port;

	std::weak_ptr<PandaJ2534Device> panda_dev;

	Mutex messageRxBuff_mutex;
	std::queue<J2534Frame> messageRxBuff;

	std::array<std::shared_ptr<J2534MessageFilter>, 10> filters;
	std::queue<std::shared_ptr<Action>> txbuff;

	std::array<std::shared_ptr<MessagePeriodic>, 10> periodicMessages;

private:
	Mutex staged_writes_lock;
};
