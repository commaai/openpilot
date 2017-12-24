#pragma once
#include "MessageTxTimeout.h"
#include "J2534Connection_ISO15765.h"

class J2534Connection_ISO15765;

/**
A specialized message type that can handle J2534 single and multi
frame (with flow control) writes.
*/
class MessageTx_ISO15765 : public MessageTxTimeoutable
{
public:
	MessageTx_ISO15765(
		std::shared_ptr<J2534Connection> connection,
		PASSTHRU_MSG& to_send,
		std::shared_ptr<J2534MessageFilter> filter
	);

	unsigned int addressLength();

	virtual void execute();

	virtual BOOL checkTxReceipt(J2534Frame frame);

	virtual BOOL isFinished();

	virtual BOOL txReady();

	virtual void reset();

	virtual void onTimeout();

	//Functions for ISO15765 flow control

	void MessageTx_ISO15765::flowControlContinue(uint8_t block_size, std::chrono::microseconds separation_time);
	void MessageTx_ISO15765::flowControlWait(unsigned long N_WFTmax);
	void MessageTx_ISO15765::flowControlAbort();

	std::shared_ptr<J2534MessageFilter> filter;
	unsigned long frames_sent;
	unsigned long consumed_count;
	uint8_t block_size;
	unsigned long CANid;
	std::string data_prefix;
	std::string payload;
	BOOL isMultipart;
	std::vector<std::string> framePayloads;
	BOOL txInFlight;
	BOOL sendAll;
	unsigned int numWaitFrames;
	BOOL didtimeout;
	BOOL issuspended;
};
