#pragma once
#include "J2534_v0404.h"
#include "J2534Connection.h"
#include "J2534Frame.h"

typedef enum {
	FILTER_RESULT_BLOCK,
	FILTER_RESULT_NEUTRAL,
	FILTER_RESULT_PASS,
	FILTER_RESULT_NOMATCH = FILTER_RESULT_BLOCK,
	FILTER_RESULT_MATCH = FILTER_RESULT_PASS,
} FILTER_RESULT;

//Forward declare
class J2534Connection;

/* Represents a J2534 Message Filter created by PassThruStartMsgFilter.

J2534 uses filters to sort out messages in a simple and sane way. Except for
flow control filters. J2534 v04.04 uses filters to manage 'conversations' in
protocols that support flow control like ISO15765. The whole solution is a
hack, and J2534 v05.00 greatly simplifies this concept. But we are using
v04.04 so, here we are.
*/
class J2534MessageFilter {
public:
	J2534MessageFilter(
		J2534Connection *const conn,
		unsigned int filtertype,
		PASSTHRU_MSG *pMaskMsg,
		PASSTHRU_MSG *pPatternMsg,
		PASSTHRU_MSG *pFlowControlMsg
	);

	bool J2534MessageFilter::operator ==(const J2534MessageFilter &b) const;

	FILTER_RESULT check(const J2534Frame& msg);
	std::string get_flowctrl();

	unsigned long flags;
	J2534Connection *const conn;
private:
	unsigned int filtertype;
	std::string maskMsg;
	std::string patternMsg;
	std::string flowCtrlMsg;
};