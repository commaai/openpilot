#include "stdafx.h"
#include "MessagePeriodic.h"
#include "J2534Connection.h"

MessagePeriodic::MessagePeriodic(
	std::chrono::microseconds delay,
	std::shared_ptr<MessageTx> msg
) : Action(msg->connection, delay), msg(msg), runyet(FALSE), active(TRUE) { };

void MessagePeriodic::execute() {
	if (!this->active) return;
	if (this->runyet) {
		if (msg->isFinished()) {
			msg->reset();
			msg->execute();
		}
	} else {
		this->runyet = TRUE;
		msg->execute();
	}

	if (auto conn_sp = this->connection.lock()) {
		if (auto panda_dev_sp = conn_sp->getPandaDev()) {
			//Scheduling must be relative to now incase there was a long stall that
			//would case it to be super far behind and try to catch up forever.
			this->scheduleImmediateDelay();
			panda_dev_sp->insertActionIntoTaskList(shared_from_this());
		}
	}
}
