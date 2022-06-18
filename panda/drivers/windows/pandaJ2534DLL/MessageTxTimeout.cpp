#include "stdafx.h"
#include "J2534Connection.h"
#include "MessageTxTimeout.h"

MessageTxTimeoutable::MessageTxTimeoutable(
	std::weak_ptr<J2534Connection> connection,
	PASSTHRU_MSG& to_send
) : MessageTx(connection, to_send), recvCount(0) { };

void MessageTxTimeoutable::scheduleTimeout(std::chrono::microseconds timeoutus) {
	if (auto conn_sp = this->connection.lock()) {
		if (auto panda_dev_sp = conn_sp->getPandaDev()) {
			auto timeoutobj = std::make_shared<MessageTxTimeout>(std::static_pointer_cast<MessageTxTimeoutable>(shared_from_this()), timeoutus);
			panda_dev_sp->scheduleAction(std::static_pointer_cast<Action>(timeoutobj), TRUE);
		}
	}
}

void MessageTxTimeoutable::scheduleTimeout(unsigned long timeoutus) {
	scheduleTimeout(std::chrono::microseconds(timeoutus));
}



MessageTxTimeout::MessageTxTimeout(
	std::shared_ptr<MessageTxTimeoutable> msg,
	std::chrono::microseconds timeout
) : Action(msg->connection), msg(msg), lastRecvCount(msg->getRecvCount()) {
	delay = timeout;
};

MessageTxTimeout::MessageTxTimeout(
	std::shared_ptr<MessageTxTimeoutable> msg,
	unsigned long timeout
) : MessageTxTimeout(msg, std::chrono::microseconds(timeout * 1000)) { };

void MessageTxTimeout::execute() {
	if (auto msg_sp = this->msg.lock()) {
		if (msg_sp->getRecvCount() == this->lastRecvCount) {
			msg_sp->onTimeout();
		}
	}
}
