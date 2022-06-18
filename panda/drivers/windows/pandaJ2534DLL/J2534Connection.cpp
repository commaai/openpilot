#include "stdafx.h"
#include "J2534Connection.h"
#include "Timer.h"

J2534Connection::J2534Connection(
	std::shared_ptr<PandaJ2534Device> panda_dev,
	unsigned long ProtocolID,
	unsigned long Flags,
	unsigned long BaudRate
) : panda_dev(panda_dev), ProtocolID(ProtocolID), Flags(Flags), BaudRate(BaudRate), Parity(0), port(0) { }

unsigned long J2534Connection::validateTxMsg(PASSTHRU_MSG* msg) {
	if (msg->DataSize < this->getMinMsgLen() || msg->DataSize > this->getMaxMsgLen())
		return ERR_INVALID_MSG;
	return STATUS_NOERROR;
}

long J2534Connection::PassThruReadMsgs(PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout) {
	//Timeout of 0 means return immediately. Non zero means WAIT for that time then return. Dafuk.
	long err_code = STATUS_NOERROR;
	Timer t = Timer();

	unsigned long msgnum = 0;
	while (msgnum < *pNumMsgs) {
		if (Timeout > 0 && t.getTimePassed() >= Timeout) {
			err_code = ERR_TIMEOUT;
			break;
		}

		//Synchronized won't work where we have to break out of a loop
		messageRxBuff_mutex.lock();
		if (this->messageRxBuff.empty()) {
			messageRxBuff_mutex.unlock();
			if (Timeout == 0)
				break;
			Sleep(2);
			continue;
		}

		auto msg_in = this->messageRxBuff.front();
		this->messageRxBuff.pop();
		messageRxBuff_mutex.unlock();

		PASSTHRU_MSG *msg_out = &pMsg[msgnum++];
		msg_out->ProtocolID = this->ProtocolID;
		msg_out->DataSize = msg_in.Data.size();
		memcpy(msg_out->Data, msg_in.Data.c_str(), msg_in.Data.size());
		msg_out->Timestamp = msg_in.Timestamp;
		msg_out->RxStatus = msg_in.RxStatus;
		msg_out->ExtraDataIndex = msg_in.ExtraDataIndex;
		msg_out->TxFlags = 0;
		if (msgnum == *pNumMsgs) break;
	}

	if (msgnum == 0)
		err_code = ERR_BUFFER_EMPTY;
	*pNumMsgs = msgnum;
	return err_code;
}

long J2534Connection::PassThruWriteMsgs(PASSTHRU_MSG *pMsg, unsigned long *pNumMsgs, unsigned long Timeout) {
	//There doesn't seem to be much reason to implement the timeout here.
	for (unsigned int msgnum = 0; msgnum < *pNumMsgs; msgnum++) {
		PASSTHRU_MSG* msg = &pMsg[msgnum];
		if (msg->ProtocolID != this->ProtocolID) {
			*pNumMsgs = msgnum;
			return ERR_MSG_PROTOCOL_ID;
		}

		auto retcode = this->validateTxMsg(msg);
		if (retcode != STATUS_NOERROR) {
			*pNumMsgs = msgnum;
			return retcode;
		}

		auto msgtx = this->parseMessageTx(*pMsg);
		if (msgtx != nullptr) //Nullptr is supported for unimplemented connection types.
			this->schedultMsgTx(std::dynamic_pointer_cast<Action>(msgtx));
	}
	return STATUS_NOERROR;
}

//The docs say that a device has to support 10 periodic messages, though more is ok.
//It is easier to store them on the connection, so 10 per connection it is.
long J2534Connection::PassThruStartPeriodicMsg(PASSTHRU_MSG *pMsg, unsigned long *pMsgID, unsigned long TimeInterval) {
	if (pMsg->DataSize < getMinMsgLen() || pMsg->DataSize > getMaxMsgSingleFrameLen()) return ERR_INVALID_MSG;
	if (pMsg->ProtocolID != this->ProtocolID) return ERR_MSG_PROTOCOL_ID;
	if (TimeInterval < 5 || TimeInterval > 65535) return ERR_INVALID_TIME_INTERVAL;

	for (unsigned int i = 0; i < this->periodicMessages.size(); i++) {
		if (periodicMessages[i] != nullptr) continue;

		*pMsgID = i;
		auto msgtx = this->parseMessageTx(*pMsg);
		if (msgtx != nullptr) {
			periodicMessages[i] = std::make_shared<MessagePeriodic>(std::chrono::microseconds(TimeInterval*1000), msgtx);
			periodicMessages[i]->scheduleImmediate();
			if (auto panda_dev = this->getPandaDev()) {
				panda_dev->insertActionIntoTaskList(periodicMessages[i]);
			}
		}
		return STATUS_NOERROR;
	}
	return ERR_EXCEEDED_LIMIT;
}

long J2534Connection::PassThruStopPeriodicMsg(unsigned long MsgID) {
	if (MsgID >= this->periodicMessages.size() || this->periodicMessages[MsgID] == nullptr)
		return ERR_INVALID_MSG_ID;
	this->periodicMessages[MsgID]->cancel();
	this->periodicMessages[MsgID] = nullptr;
	return STATUS_NOERROR;
}

long J2534Connection::PassThruStartMsgFilter(unsigned long FilterType, PASSTHRU_MSG *pMaskMsg, PASSTHRU_MSG *pPatternMsg,
	PASSTHRU_MSG *pFlowControlMsg, unsigned long *pFilterID) {
	for (unsigned int i = 0; i < this->filters.size(); i++) {
		if (filters[i] == nullptr) {
			try {
				auto newfilter = std::make_shared<J2534MessageFilter>(this, FilterType, pMaskMsg, pPatternMsg, pFlowControlMsg);
				for (unsigned int check_idx = 0; check_idx < filters.size(); check_idx++) {
					if (filters[check_idx] == nullptr) continue;
					if (filters[check_idx] == newfilter) {
						filters[i] = nullptr;
						return ERR_NOT_UNIQUE;
					}
				}
				*pFilterID = i;
				filters[i] = newfilter;
				return STATUS_NOERROR;
			} catch (int e) {
				return e;
			}
		}
	}
	return ERR_EXCEEDED_LIMIT;
}

long J2534Connection::PassThruStopMsgFilter(unsigned long FilterID) {
	if (FilterID >= this->filters.size() || this->filters[FilterID] == nullptr)
		return ERR_INVALID_FILTER_ID;
	this->filters[FilterID] = nullptr;
	return STATUS_NOERROR;
}

long J2534Connection::PassThruIoctl(unsigned long IoctlID, void *pInput, void *pOutput) {
	return STATUS_NOERROR;
}

long J2534Connection::init5b(SBYTE_ARRAY* pInput, SBYTE_ARRAY* pOutput) {
	if (pInput->NumOfBytes == 1) {
		if (auto panda_ps = this->panda_dev.lock()) {
			auto resp = panda_ps->kline_five_baud_init(pInput->BytePtr[0]);
			if (resp.size() > 0) {
				auto key_bytes = resp.c_str();
				if (pOutput->NumOfBytes >= 1) {
					pOutput->BytePtr[0] = key_bytes[0];
				}
				if (pOutput->NumOfBytes >= 2) {
					pOutput->BytePtr[1] = key_bytes[1];
				}
				return STATUS_NOERROR;
			}
		}
	}

	return ERR_FAILED;
}
long J2534Connection::initFast(PASSTHRU_MSG* pInput, PASSTHRU_MSG* pOutput) {
	if (auto panda_ps = this->panda_dev.lock()) {
		auto start_comm = std::string((char*)pInput->Data, pInput->DataSize);
		auto resp = panda_ps->kline_wakeup_start_comm(start_comm);
		if (resp.size() > 0) {
			pOutput->ProtocolID = pInput->ProtocolID;
			pOutput->RxStatus = 0;
			pOutput->TxFlags = 0;
			pOutput->Timestamp = pInput->Timestamp;
			pOutput->ExtraDataIndex = resp.size();
			memcpy(pOutput->Data, resp.c_str(), resp.size());
			pOutput->DataSize = resp.size();
			return STATUS_NOERROR;
		}
	}

	return ERR_FAILED;
}
long J2534Connection::clearTXBuff() {
	if (auto panda_ps = this->panda_dev.lock()) {
		synchronized(staged_writes_lock) {
			this->txbuff = {};
			switch (this->ProtocolID)
			{
				case CAN:
				case CAN_PS:
				case ISO15765:
				case ISO15765_PS:
					panda_ps->panda->can_clear(panda::PANDA_CAN1_TX);
					break;
				case ISO9141:
				case ISO9141_PS:
				case ISO14230:
				case ISO14230_PS:
					panda_ps->panda->serial_clear(panda::SERIAL_LIN1);
					panda_ps->panda->serial_clear(panda::SERIAL_LIN2);
					break;
				default:
					break;
			}
		}
		return STATUS_NOERROR;
	}
	return ERR_FAILED;
}
long J2534Connection::clearRXBuff() {
	if (auto panda_ps = this->panda_dev.lock()) {
		synchronized(messageRxBuff_mutex) {
			this->messageRxBuff = {};
			switch (this->ProtocolID)
			{
			case CAN:
			case CAN_PS:
			case ISO15765:
			case ISO15765_PS:
				panda_ps->panda->can_clear(panda::PANDA_CAN_RX);
				break;
			case ISO9141:
			case ISO9141_PS:
			case ISO14230:
			case ISO14230_PS:
				panda_ps->panda->serial_clear(panda::SERIAL_LIN1);
				panda_ps->panda->serial_clear(panda::SERIAL_LIN2);
				break;
			default:
				break;
			}
		}
		return STATUS_NOERROR;
	}
	return ERR_FAILED;
}
long J2534Connection::clearPeriodicMsgs() {
	for (unsigned int i = 0; i < this->periodicMessages.size(); i++) {
		if (periodicMessages[i] == nullptr) continue;
		this->periodicMessages[i]->cancel();
		this->periodicMessages[i] = nullptr;
	}

	return STATUS_NOERROR;
}
long J2534Connection::clearMsgFilters() {
	for (auto& filter : this->filters) filter = nullptr;
	return STATUS_NOERROR;
}

void J2534Connection::setBaud(unsigned long baud) {
	this->BaudRate = baud;
}

void J2534Connection::setParity(unsigned long parity) {
	this->Parity = parity;
}

void J2534Connection::schedultMsgTx(std::shared_ptr<Action> msgout) {
	if (auto panda_ps = this->panda_dev.lock()) {
		synchronized(staged_writes_lock) {
			this->txbuff.push(msgout);
			panda_ps->registerConnectionTx(shared_from_this());
		}
	}
}

void J2534Connection::rescheduleExistingTxMsgs() {
	if (auto panda_ps = this->panda_dev.lock()) {
		synchronized(staged_writes_lock) {
			panda_ps->unstallConnectionTx(shared_from_this());
		}
	}
}

//Works well as long as the protocol doesn't support flow control.
void J2534Connection::processMessage(const J2534Frame& msg) {
	FILTER_RESULT filter_res = FILTER_RESULT_NEUTRAL;

	for (auto filter : this->filters) {
		if (filter == nullptr) continue;
		FILTER_RESULT current_check_res = filter->check(msg);
		if (current_check_res == FILTER_RESULT_BLOCK) return;
		if (current_check_res == FILTER_RESULT_PASS) filter_res = FILTER_RESULT_PASS;
	}

	if (filter_res == FILTER_RESULT_PASS) {
		addMsgToRxQueue(msg);
	}
}

void J2534Connection::processIOCTLSetConfig(unsigned long Parameter, unsigned long Value) {
	switch (Parameter) {
	case DATA_RATE:			// 5-500000
		this->setBaud(Value);
		break;
	case LOOPBACK:			// 0 (OFF), 1 (ON) [0]
		this->loopback = (Value != 0);
		break;
	case PARITY:
		this->setParity(Value);
		break;
	case ISO15765_WFT_MAX:
		break;
	case NODE_ADDRESS:		// J1850PWM Related (Not supported by panda). HDS requires these to 'work'.
	case NETWORK_LINE:
	case P1_MIN:			// A bunch of stuff relating to ISO9141 and ISO14230 that the panda
	case P1_MAX:			// currently doesn't support. Don't let HDS know we can't use these.
	case P2_MIN:
	case P2_MAX:
	case P3_MIN:
	case P3_MAX:
	case P4_MIN:
	case P4_MAX:
	case W0:
	case W1:
	case W2:
	case W3:
	case W4:
	case W5:
	case TIDLE:
	case TINIL:
	case TWUP:
	case T1_MAX:			// SCI related options. The panda does not appear to support this
	case T2_MAX:
	case T3_MAX:
	case T4_MAX:
	case T5_MAX:
		break;				// Just smile and nod.
	default:
		printf("Got unknown SET code %X\n", Parameter);
	}

	// reserved parameters usually mean special equiptment is required
	//if (Parameter >= 0x20) {
	//	throw ERR_NOT_SUPPORTED;
	//}
}

unsigned long J2534Connection::processIOCTLGetConfig(unsigned long Parameter) {
	switch (Parameter) {
	case DATA_RATE:
		return this->getBaud();
	case LOOPBACK:
		return this->loopback;
		break;
	case BIT_SAMPLE_POINT:
		return 80;
	case SYNC_JUMP_WIDTH:
		return 15;
	default:
		// HDS rarely reads off values through ioctl GET_CONFIG, but it often
		// just wants the call to pass without erroring, so just don't do anything.
		printf("Got unknown code %X\n", Parameter);
	}
}
