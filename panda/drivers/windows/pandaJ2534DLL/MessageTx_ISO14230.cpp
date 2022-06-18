#include "stdafx.h"
#include "MessageTx_ISO14230.h"
#include "J2534Connection_ISO14230.h"

MessageTx_ISO14230::MessageTx_ISO14230(
    std::shared_ptr<J2534Connection> connection_in,
    PASSTHRU_MSG& to_send
) : MessageTx(connection_in, to_send), sentyet(FALSE), txInFlight(FALSE) {};

void MessageTx_ISO14230::execute() {
    if (auto conn_sp = this->connection.lock()) {
        if (auto panda_dev_sp = conn_sp->getPandaDev()) {
            if (panda_dev_sp->kline_send(this->fullmsg.Data)) {
                if (auto conn_sp = this->connection.lock())
                {
                    if (conn_sp->loopback) {
                        auto echo = J2534Frame(conn_sp->getProtocol(), TX_MSG_TYPE, 0, this->fullmsg.Timestamp);
                        echo.Data = std::string(this->fullmsg.Data);
                        echo.ExtraDataIndex = this->fullmsg.Data.size();
                        conn_sp->addMsgToRxQueue(J2534Frame(conn_sp->getProtocol(), START_OF_MESSAGE, 0, this->fullmsg.Timestamp));
                        conn_sp->addMsgToRxQueue(echo);
                    }
                }
                this->txInFlight = FALSE;
                this->sentyet = TRUE;
            }
            // remove action since echo was read back in kline_send()
            panda_dev_sp->removeConnectionTopAction(conn_sp, shared_from_this());
        }
    }
}

BOOL MessageTx_ISO14230::checkTxReceipt(J2534Frame frame) {
    throw "not implemented!";
}

void MessageTx_ISO14230::reset() {
    sentyet = FALSE;
    txInFlight = FALSE;
}
