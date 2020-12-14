#pragma once
#include "Action.h"
#include "MessageTx.h"

class J2534Connection;

/* A message that is resent on a given period. Created with calls to PassThruStartPeriodicMessage.

Instead of making each J2534 protocol implementation have to implement periodic message
functionality, this class takes a message to be sent, and passes along the execute call
to the message, then reschedules itself.
*/
class MessagePeriodic : public Action, public std::enable_shared_from_this<Action>
{
public:
	MessagePeriodic(
		std::chrono::microseconds delay,
		std::shared_ptr<MessageTx> msg
	);

	virtual void execute();

	void cancel() {
		this->active = FALSE;
	}

protected:
	std::shared_ptr<MessageTx> msg;

private:
	BOOL runyet;
	BOOL active;
};
