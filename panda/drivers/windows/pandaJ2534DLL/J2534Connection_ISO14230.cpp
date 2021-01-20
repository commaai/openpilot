#include "stdafx.h"
#include "J2534Connection_ISO14230.h"
#include "MessageTx_ISO14230.h"
#include "Timer.h"

J2534Connection_ISO14230::J2534Connection_ISO14230(
        std::shared_ptr<PandaJ2534Device> panda_dev,
        unsigned long ProtocolID,
        unsigned long Flags,
        unsigned long BaudRate
    ) : J2534Connection(panda_dev, ProtocolID, Flags, BaudRate) {
    this->port = 0;

    if (BaudRate % 100 || BaudRate < 2400 || BaudRate > 115200)
        throw ERR_INVALID_BAUDRATE;

    panda_dev->panda->set_uart_baud(panda::SERIAL_LIN1, BaudRate);
    panda_dev->panda->set_uart_baud(panda::SERIAL_LIN2, BaudRate);
};

unsigned long J2534Connection_ISO14230::validateTxMsg(PASSTHRU_MSG* msg) {
    if (msg->DataSize < this->getMinMsgLen() || msg->DataSize > this->getMaxMsgLen())
        return ERR_INVALID_MSG;
    return STATUS_NOERROR;
}

std::shared_ptr<MessageTx> J2534Connection_ISO14230::parseMessageTx(PASSTHRU_MSG& msg) {
    return std::dynamic_pointer_cast<MessageTx>(std::make_shared<MessageTx_ISO14230>(shared_from_this(), msg));
}

void J2534Connection_ISO14230::setBaud(unsigned long BaudRate) {
    if (auto panda_dev = this->getPandaDev()) {
        if (BaudRate % 100 || BaudRate < 2400 || BaudRate > 115200)
            throw ERR_NOT_SUPPORTED;

        panda_dev->panda->set_uart_baud(panda::SERIAL_LIN1, BaudRate);
        panda_dev->panda->set_uart_baud(panda::SERIAL_LIN2, BaudRate);
        return J2534Connection::setBaud(BaudRate);
    } else {
        throw ERR_DEVICE_NOT_CONNECTED;
    }
}

void J2534Connection_ISO14230::setParity(unsigned long Parity) {
    if (auto panda_dev = this->getPandaDev()) {
        panda::PANDA_SERIAL_PORT_PARITY parity;
        switch (Parity) {
            case 0:
                parity = panda::PANDA_PARITY_OFF;
                break;
            case 1:
                parity = panda::PANDA_PARITY_ODD;
                break;
            case 2:
                parity = panda::PANDA_PARITY_EVEN;
                break;
            default:
                throw ERR_NOT_SUPPORTED;
        }
        panda_dev->panda->set_uart_parity(panda::SERIAL_LIN1, parity);
        panda_dev->panda->set_uart_parity(panda::SERIAL_LIN2, parity);
        return J2534Connection::setParity(Parity);
    }
    else {
        throw ERR_DEVICE_NOT_CONNECTED;
    }
}

void J2534Connection_ISO14230::processMessage(const J2534Frame& msg) {
    FILTER_RESULT filter_res = FILTER_RESULT_NEUTRAL;

    for (auto filter : this->filters) {
        if (filter == nullptr) continue;
        FILTER_RESULT current_check_res = filter->check(msg);
        if (current_check_res == FILTER_RESULT_BLOCK) return;
        if (current_check_res == FILTER_RESULT_PASS) filter_res = FILTER_RESULT_PASS;
    }

    if (filter_res == FILTER_RESULT_PASS) {
        addMsgToRxQueue(J2534Frame(msg.ProtocolID, START_OF_MESSAGE, 0, 0));
        addMsgToRxQueue(msg);
    }
}
