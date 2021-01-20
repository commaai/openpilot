#pragma once

class MessageRx
{
public:
	MessageRx(
		unsigned long size,
		std::string piece,
		unsigned long rxFlags,
		std::shared_ptr<J2534MessageFilter> filter
	) : expected_size(size & 0xFFF), flags(rxFlags) {
		msg.reserve(expected_size);
		msg = piece;
		next_part = 1;
	};

	bool rx_add_frame(uint8_t pci_byte, unsigned int max_packet_size, const std::string piece) {
		if ((pci_byte & 0x0F) != this->next_part) {
			//TODO: Maybe this should instantly fail the transaction.
			return TRUE;
		}

		this->next_part = (this->next_part + 1) % 0x10;
		unsigned int payload_len = min(expected_size - msg.size(), max_packet_size);
		if (piece.size() < payload_len) {
			//A frame was received that could have held more data.
			//No examples of this protocol show that happening, so
			//it will be assumed that it is grounds to reset rx.
			return FALSE;
		}
		msg += piece.substr(0, payload_len);

		return TRUE;
	}

	unsigned int bytes_remaining() {
		return this->expected_size - this->msg.size();
	}

	bool is_ready() {
		return this->msg.size() == this->expected_size;
	}

	bool flush_result(std::string& final_msg) {
		if (this->msg.size() == this->expected_size) {
			final_msg = this->msg;
			return TRUE;
		}
		return FALSE;
	}

	uint8_t getNextConsecutiveFrameId() {
		return this->next_part++;
	}

	std::weak_ptr<J2534MessageFilter> filter;
	unsigned long flags;
	unsigned long expected_size;
	std::string msg;
	unsigned char next_part;
};
