#include "stdafx.h"
#include "J2534MessageFilter.h"
#include "J2534Frame.h"

J2534MessageFilter::J2534MessageFilter(
	J2534Connection *const conn,
	unsigned int filtertype,
	PASSTHRU_MSG *pMaskMsg,
	PASSTHRU_MSG *pPatternMsg,
	PASSTHRU_MSG *pFlowControlMsg
) : filtertype(filtertype), flags(0), conn(conn) {
	switch (filtertype) {
	case PASS_FILTER:
	case BLOCK_FILTER:
		if (pMaskMsg == NULL || pPatternMsg == NULL)
			throw ERR_NULL_PARAMETER;
		if (pFlowControlMsg != NULL)
			throw ERR_INVALID_FILTER_ID;
		if (pMaskMsg->DataSize != pPatternMsg->DataSize)
			throw ERR_INVALID_MSG;
		break;
	case FLOW_CONTROL_FILTER:
		if (conn->getProtocol() != ISO15765) throw ERR_MSG_PROTOCOL_ID; //CHECK
		if (pFlowControlMsg == NULL || pMaskMsg == NULL || pPatternMsg == NULL)
			throw ERR_NULL_PARAMETER;
		break;
	default:
		throw ERR_INVALID_MSG;
	}

	if (!(conn->getMinMsgLen() < pMaskMsg->DataSize || pMaskMsg->DataSize < conn->getMaxMsgLen()))
		throw ERR_INVALID_MSG;
	if (conn->getProtocol() != pMaskMsg->ProtocolID)
		throw ERR_MSG_PROTOCOL_ID;
	this->maskMsg = std::string((char*)pMaskMsg->Data, pMaskMsg->DataSize);

	if (!(conn->getMinMsgLen() < pPatternMsg->DataSize || pPatternMsg->DataSize < conn->getMaxMsgLen()))
		throw ERR_INVALID_MSG;
	if (conn->getProtocol() != pPatternMsg->ProtocolID)
		throw ERR_MSG_PROTOCOL_ID;
	this->patternMsg = std::string((char*)pPatternMsg->Data, pPatternMsg->DataSize);
	if (this->maskMsg.size() != this->patternMsg.size())
		throw ERR_INVALID_MSG;

	if (pFlowControlMsg) {
		if (!(conn->getMinMsgLen() < pFlowControlMsg->DataSize || pFlowControlMsg->DataSize < conn->getMaxMsgLen()))
			throw ERR_INVALID_MSG;
		if (conn->getProtocol() != pFlowControlMsg->ProtocolID)
			throw ERR_MSG_PROTOCOL_ID;
		if (pMaskMsg->TxFlags != pPatternMsg->TxFlags || pMaskMsg->TxFlags != pFlowControlMsg->TxFlags)
			throw ERR_INVALID_MSG;
		if(pFlowControlMsg->TxFlags & ~(ISO15765_FRAME_PAD | CAN_29BIT_ID | ISO15765_ADDR_TYPE))
			throw ERR_INVALID_MSG;
		if ((pFlowControlMsg->TxFlags & ISO15765_ADDR_TYPE) == ISO15765_ADDR_TYPE) {
			if(pFlowControlMsg->DataSize != 5)
				throw ERR_INVALID_MSG;
		} else {
			if (pFlowControlMsg->DataSize != 4)
				throw ERR_INVALID_MSG;
		}
		this->flowCtrlMsg = std::string((char*)pFlowControlMsg->Data, pFlowControlMsg->DataSize);
		if (this->flowCtrlMsg.size() != this->patternMsg.size())
			throw ERR_INVALID_MSG;
		this->flags = pFlowControlMsg->TxFlags;
	}
}

bool J2534MessageFilter::operator ==(const J2534MessageFilter &b) const {
	if (this->filtertype != b.filtertype) return FALSE;
	if (this->maskMsg != b.maskMsg) return FALSE;
	if (this->patternMsg != b.patternMsg) return FALSE;
	if (this->flowCtrlMsg != b.flowCtrlMsg) return FALSE;
	if (this->flags != b.flags) return FALSE;
	return TRUE;
}

FILTER_RESULT J2534MessageFilter::check(const J2534Frame& msg) {
	bool matches = TRUE;
	if (msg.Data.size() < this->maskMsg.size()) {
		matches = FALSE;
	} else {
		for (unsigned int i = 0; i < this->maskMsg.size(); i++) {
			if (this->patternMsg[i] != (msg.Data[i] & this->maskMsg[i])) {
				matches = FALSE;
				break;
			}
		}
	}

	switch (this->filtertype) {
	case PASS_FILTER:
		return matches ? FILTER_RESULT_PASS : FILTER_RESULT_NEUTRAL;
	case BLOCK_FILTER:
		return matches ? FILTER_RESULT_BLOCK: FILTER_RESULT_NEUTRAL;
	case FLOW_CONTROL_FILTER:
		return matches ? FILTER_RESULT_MATCH : FILTER_RESULT_NOMATCH;
	default:
		throw std::out_of_range("Filtertype should not be able to be anything but PASS, BLOCK, or FLOW_CONTROL");
	}
}

std::string J2534MessageFilter::get_flowctrl() {
	return std::string(this->flowCtrlMsg);
}
