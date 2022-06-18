#pragma once
#include <memory>
#include "MessageTx.h"

class J2534Connection;

class MessageTx_ISO14230 : public MessageTx
{
public:
    MessageTx_ISO14230(
        std::shared_ptr<J2534Connection> connection_in,
        PASSTHRU_MSG& to_send
    );

    virtual void execute();

    virtual BOOL checkTxReceipt(J2534Frame frame);

    virtual BOOL isFinished() {
        return !txInFlight && sentyet;
    };

    virtual BOOL txReady() {
        return !sentyet;
    };

    virtual void reset();

private:
    BOOL sentyet;
    BOOL txInFlight;
};
