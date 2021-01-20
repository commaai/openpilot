#pragma once
#include "Action.h"
#include "J2534Frame.h"

class J2534Connection;

class MessageTx : public Action, public std::enable_shared_from_this<MessageTx>
{
public:
	MessageTx(
		std::weak_ptr<J2534Connection> connection_in,
		PASSTHRU_MSG& to_send
	) : Action(connection_in), fullmsg(to_send) { };

	virtual BOOL checkTxReceipt(J2534Frame frame) = 0;

	virtual BOOL isFinished() = 0;

	virtual BOOL txReady() = 0;

	virtual void reset() = 0;

protected:
	J2534Frame fullmsg;
};