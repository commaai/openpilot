// panda.cpp : Defines the exported functions for the DLL application.
//
#include "stdafx.h"

#include "device.h"
#include "panda.h"

#define REQUEST_IN 0xC0
#define REQUEST_OUT 0x40

#define CAN_TRANSMIT 1
#define CAN_EXTENDED 4

#define KLINE_HEADER_FMT_ADDR_MASK 0xC0
#define KLINE_HEADER_FMT_LEN_MASK 0x3F

using namespace panda;

Panda::Panda(
	WINUSB_INTERFACE_HANDLE WinusbHandle,
	HANDLE DeviceHandle,
	tstring devPath_,
	std::string sn_
) : usbh(WinusbHandle), devh(DeviceHandle), devPath(devPath_), sn(sn_) {
	printf("CREATED A PANDA %s\n", this->sn.c_str());
	this->set_can_loopback(FALSE);
	this->set_raw_io(TRUE);
	this->set_alt_setting(0);
}

Panda::~Panda() {
	WinUsb_Free(this->usbh);
	CloseHandle(this->devh);
	printf("Cleanup Panda %s\n", this->sn.c_str());
}

std::vector<std::string> Panda::listAvailablePandas() {
	std::vector<std::string> ret;
	auto map_sn_to_devpath = detect_pandas();

	for (auto kv : map_sn_to_devpath) {
		ret.push_back(std::string(kv.first));
	}

	return ret;
}

std::unique_ptr<Panda> Panda::openPanda(std::string sn)
{
	auto map_sn_to_devpath = detect_pandas();

	if (map_sn_to_devpath.empty()) return nullptr;
	if (map_sn_to_devpath.find(sn) == map_sn_to_devpath.end() && sn != "") return nullptr;

	tstring devpath;
	if (sn.empty()) {
		sn = map_sn_to_devpath.begin()->first;
		devpath = map_sn_to_devpath.begin()->second;
	} else {
		devpath = map_sn_to_devpath[sn];
	}

	HANDLE deviceHandle = CreateFile(devpath.c_str(),
		GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);

	if (INVALID_HANDLE_VALUE == deviceHandle) {
		_tprintf(_T("    Error opening Device Handle %d.\n"),// Msg: '%s'\n"),
			GetLastError());// , GetLastErrorAsString().c_str());
		return nullptr;
	}

	WINUSB_INTERFACE_HANDLE winusbHandle;
	if (WinUsb_Initialize(deviceHandle, &winusbHandle) == FALSE) {
		_tprintf(_T("    Error initializing WinUSB %d.\n"),// Msg: '%s'\n"),
			GetLastError());// , GetLastErrorAsString().c_str());
		CloseHandle(deviceHandle);
		return nullptr;
	}

	return std::unique_ptr<Panda>(new Panda(winusbHandle, deviceHandle, map_sn_to_devpath[sn], sn));
}

std::string Panda::get_usb_sn() {
	return std::string(this->sn);
}

int Panda::control_transfer(
	uint8_t			bmRequestType,
	uint8_t  		bRequest,
	uint16_t  		wValue,
	uint16_t  		wIndex,
	void *			data,
	uint16_t		wLength,
	unsigned int  	timeout
) {
	UNREFERENCED_PARAMETER(timeout);

	WINUSB_SETUP_PACKET SetupPacket;
	ZeroMemory(&SetupPacket, sizeof(WINUSB_SETUP_PACKET));
	ULONG cbSent = 0;

	//Create the setup packet
	SetupPacket.RequestType = bmRequestType;
	SetupPacket.Request = bRequest;
	SetupPacket.Value = wValue;
	SetupPacket.Index = wIndex;
	SetupPacket.Length = wLength;

	//ULONG timeout = 10; // ms
	//WinUsb_SetPipePolicy(interfaceHandle, pipeID, PIPE_TRANSFER_TIMEOUT, sizeof(ULONG), &timeout);

	if (WinUsb_ControlTransfer(this->usbh, SetupPacket, (PUCHAR)data, wLength, &cbSent, 0) == FALSE) {
		return -1;
	}

	return cbSent;
}

int Panda::bulk_write(UCHAR endpoint, const void * buff, ULONG length, PULONG transferred, ULONG timeout) {
	if (this->usbh == INVALID_HANDLE_VALUE || !buff || !length || !transferred) return FALSE;

	if (WinUsb_WritePipe(this->usbh, endpoint, (PUCHAR)buff, length, transferred, NULL) == FALSE) {
		_tprintf(_T("    Got error during bulk xfer: %d. Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
		return FALSE;
	}
	return TRUE;
}

int Panda::bulk_read(UCHAR endpoint, void * buff, ULONG buff_size, PULONG transferred, ULONG timeout) {
	if (this->usbh == INVALID_HANDLE_VALUE || !buff || !buff_size || !transferred) return FALSE;

	if (WinUsb_ReadPipe(this->usbh, endpoint, (PUCHAR)buff, buff_size, transferred, NULL) == FALSE) {
		_tprintf(_T("    Got error during bulk xfer: %d. Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
		return FALSE;
	}
	return TRUE;
}

bool Panda::set_alt_setting(UCHAR alt_setting) {
	if (WinUsb_AbortPipe(this->usbh, 0x81) == FALSE) {
		_tprintf(_T("    Error abobrting pipe before setting altsetting. continue. %d, Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
	}
	if (WinUsb_SetCurrentAlternateSetting(this->usbh, alt_setting) == FALSE) {
		_tprintf(_T("    Error setting usb altsetting %d, Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
		return FALSE;
	}

	// Either the panda or the windows usb stack can drop messages
	// if an odd number of messages are sent before an interrupt IN
	// message is canceled. There are some other odd behaviors, but
	// the best solution so far has been to send a few messages
	// before using the device to clear out the pipe. No, the windows
	// functions for clearing/resetting/etc the pipe did not work.
	// This took way too to figure out a workaround.
	// New info. The most repeatable behavior is losing the first
	// message sent after setting alt setting to 1 (even without
	// receiving). Something like this happened on linux sometimes.
	bool loopback_backup = this->loopback;
	this->set_can_loopback(TRUE);
	Sleep(20); // Give time for any sent messages to appear in the RX buffer.
	this->can_clear(PANDA_CAN_RX);
	// send 4 messages becaus can_recv reads 4 messages at a time
	for (int i = 0; i < 4; i++) {
		printf("Sending PAD %d\n", i);
		if (this->can_send(0x7FF, FALSE, {}, 0, PANDA_CAN1) == FALSE) {
			auto err = GetLastError();
			printf("Got err on first send: %d\n", err);
		}
	}
	Sleep(10);
	//this->can_clear(PANDA_CAN_RX);

	//Read the messages so they do not contaimnate the real message stream.
	this->can_recv();

	//this->set_can_loopback(FALSE);
	this->set_can_loopback(loopback_backup);

	return TRUE;
}

UCHAR Panda::get_current_alt_setting() {
	UCHAR alt_setting;
	if (WinUsb_GetCurrentAlternateSetting(this->usbh, &alt_setting) == FALSE) {
		_tprintf(_T("    Error getting usb altsetting %d, Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
		return FALSE;
	}

	return alt_setting;
}

bool Panda::set_raw_io(bool val) {
	UCHAR raw_io = val;
	if (!WinUsb_SetPipePolicy(this->usbh, 0x81, RAW_IO, sizeof(raw_io), &raw_io)) {
		_tprintf(_T("    Error setting usb raw I/O pipe policy %d, Msg: '%s'\n"),
			GetLastError(), GetLastErrorAsString().c_str());
		return FALSE;
	}

	return TRUE;
}

PANDA_HEALTH Panda::get_health()
{
	WINUSB_SETUP_PACKET SetupPacket;
	ZeroMemory(&SetupPacket, sizeof(WINUSB_SETUP_PACKET));
	ULONG cbSent = 0;

	//Create the setup packet
	SetupPacket.RequestType = REQUEST_IN;
	SetupPacket.Request = 0xD2;
	SetupPacket.Value = 0;
	SetupPacket.Index = 0;
	SetupPacket.Length = sizeof(UCHAR);

	//uint8_t health[13];
	PANDA_HEALTH health;

	if (WinUsb_ControlTransfer(this->usbh, SetupPacket, (PUCHAR)&health, sizeof(health), &cbSent, 0) == FALSE) {
		_tprintf(_T("    Got unexpected error while reading panda health (2nd time) %d. Msg: '%s'\n"),
				GetLastError(), GetLastErrorAsString().c_str());
	}

	return health;
}

bool Panda::enter_bootloader() {
	return this->control_transfer(REQUEST_OUT, 0xd1, 0, 0, NULL, 0, 0) != -1;
}

std::string Panda::get_version() {
	char buff[0x40];
	ZeroMemory(&buff, sizeof(buff));

	int xferCount = this->control_transfer(REQUEST_IN, 0xd6, 0, 0, buff, 0x40, 0);
	if (xferCount == -1) return std::string();
	return std::string(buff);
}

//TODO: Do hash stuff for calculating the serial.
std::string Panda::get_serial() {
	char buff[0x20];
	ZeroMemory(&buff, sizeof(buff));

	int xferCount = this->control_transfer(REQUEST_IN, 0xD0, 0, 0, buff, 0x20, 0);
	if (xferCount == -1) return std::string();
	return std::string(buff);

	//dat = self._handle.controlRead(REQUEST_IN, 0xd0, 0, 0, 0x20);
	//hashsig, calc_hash = dat[0x1c:], hashlib.sha1(dat[0:0x1c]).digest()[0:4]
	//	assert(hashsig == calc_hash)
	//	return[dat[0:0x10], dat[0x10:0x10 + 10]]
}

//Secret appears to by raw bytes, not a string. TODO: Change returned type.
std::string Panda::get_secret() {
	char buff[0x10];
	ZeroMemory(&buff, sizeof(buff));

	int xferCount = this->control_transfer(REQUEST_IN, 0xd0, 1, 0, buff, 0x10, 0);
	if (xferCount == -1) return std::string();
	return std::string(buff);
}

bool Panda::set_usb_power(bool on) {
	return this->control_transfer(REQUEST_OUT, 0xe6, (int)on, 0, NULL, 0, 0) != -1;
}

bool Panda::set_esp_power(bool on) {
	return this->control_transfer(REQUEST_OUT, 0xd9, (int)on, 0, NULL, 0, 0) != -1;
}

bool Panda::esp_reset(uint16_t bootmode = 0) {
	return this->control_transfer(REQUEST_OUT, 0xda, bootmode, 0, NULL, 0, 0) != -1;
}

bool Panda::set_safety_mode(PANDA_SAFETY_MODE mode = SAFETY_SILENT) {
	return this->control_transfer(REQUEST_OUT, 0xdc, mode, 0, NULL, 0, 0) != -1;
}

bool Panda::set_can_forwarding(PANDA_CAN_PORT from_bus, PANDA_CAN_PORT to_bus) {
	if (from_bus == PANDA_CAN_UNK) return FALSE;
	return this->control_transfer(REQUEST_OUT, 0xdd, from_bus, to_bus, NULL, 0, 0) != -1;
}

bool Panda::set_gmlan(PANDA_GMLAN_HOST_PORT bus = PANDA_GMLAN_CAN3) {
	return this->control_transfer(REQUEST_OUT, 0xdb, 1, (bus == PANDA_GMLAN_CLEAR) ? 0 : bus, NULL, 0, 0) != -1;
}

bool Panda::set_can_loopback(bool enable) {
	this->loopback = enable;
	return this->control_transfer(REQUEST_OUT, 0xe5, enable, 0, NULL, 0, 0) != -1;
}

//Can not use the full range of 16 bit speed.
//cbps means centa bits per second (tento of kbps)
bool Panda::set_can_speed_cbps(PANDA_CAN_PORT bus, uint16_t speed) {
	if (bus == PANDA_CAN_UNK) return FALSE;
	return this->control_transfer(REQUEST_OUT, 0xde, bus, speed, NULL, 0, 0) != -1;
}

//Can not use the full range of 16 bit speed.
bool Panda::set_can_speed_kbps(PANDA_CAN_PORT bus, uint16_t speed) {
	return set_can_speed_cbps(bus, speed * 10);
}

//Can not use full 32 bit range of rate
bool Panda::set_uart_baud(PANDA_SERIAL_PORT uart, uint32_t rate) {
	return this->control_transfer(REQUEST_OUT, 0xe4, uart, rate / 300, NULL, 0, 0) != -1;
}

bool Panda::set_uart_parity(PANDA_SERIAL_PORT uart, PANDA_SERIAL_PORT_PARITY parity) {
	return this->control_transfer(REQUEST_OUT, 0xe2, uart, parity, NULL, 0, 0) != -1;
}

bool Panda::can_send_many(const std::vector<PANDA_CAN_MSG>& can_msgs) {
	std::vector<PANDA_CAN_MSG_INTERNAL> formatted_msgs;
	formatted_msgs.reserve(can_msgs.size());

	for (auto msg : can_msgs) {
		if (msg.bus == PANDA_CAN_UNK) continue;
		if (msg.len > 8) continue;
		PANDA_CAN_MSG_INTERNAL tmpmsg = {};
		tmpmsg.rir = (msg.addr_29b) ?
			((msg.addr << 3) | CAN_TRANSMIT | CAN_EXTENDED) :
			(((msg.addr & 0x7FF) << 21) | CAN_TRANSMIT);
		tmpmsg.f2 = msg.len | (msg.bus << 4);
		memcpy(tmpmsg.dat, msg.dat, msg.len);
		formatted_msgs.push_back(tmpmsg);
	}

	if (formatted_msgs.size() == 0) return FALSE;

	unsigned int retcount;
	return this->bulk_write(3, formatted_msgs.data(),
		sizeof(PANDA_CAN_MSG_INTERNAL)*formatted_msgs.size(), (PULONG)&retcount, 0);
}

bool Panda::can_send(uint32_t addr, bool addr_29b, const uint8_t *dat, uint8_t len, PANDA_CAN_PORT bus) {
	if (bus == PANDA_CAN_UNK) return FALSE;
	if (len > 8) return FALSE;
	PANDA_CAN_MSG msg;
	msg.addr_29b = addr_29b;
	msg.addr = addr;
	msg.len = len;
	memcpy(msg.dat, dat, msg.len);
	msg.bus = bus;
	return this->can_send_many(std::vector<PANDA_CAN_MSG>{msg});
}

PANDA_CAN_MSG Panda::parse_can_recv(PANDA_CAN_MSG_INTERNAL *in_msg_raw) {
	PANDA_CAN_MSG in_msg;

	in_msg.addr_29b = (bool)(in_msg_raw->rir & CAN_EXTENDED);
	in_msg.addr = (in_msg.addr_29b) ? (in_msg_raw->rir >> 3) : (in_msg_raw->rir >> 21);
	in_msg.recv_time = this->runningTime.getTimePassedUS();
	in_msg.recv_time_point = std::chrono::steady_clock::now();
	//The timestamp from the device is (in_msg_raw->f2 >> 16),
	//but this 16 bit value is a little hard to use. Using a
	//timer since the initialization of this device.
	in_msg.len = in_msg_raw->f2 & 0xF;
	memcpy(in_msg.dat, in_msg_raw->dat, 8);

	in_msg.is_receipt = ((in_msg_raw->f2 >> 4) & 0x80) == 0x80;
	switch ((in_msg_raw->f2 >> 4) & 0x7F) {
	case PANDA_CAN1:
		in_msg.bus = PANDA_CAN1;
		break;
	case PANDA_CAN2:
		in_msg.bus = PANDA_CAN2;
		break;
	case PANDA_CAN3:
		in_msg.bus = PANDA_CAN3;
		break;
	default:
		in_msg.bus = PANDA_CAN_UNK;
	}
	return in_msg;
}

bool Panda::can_rx_q_push(HANDLE kill_event, DWORD timeoutms) {
	while (1) {
		auto w_ptr = this->w_ptr;
		auto n_ptr = w_ptr + 1;
		if (n_ptr == CAN_RX_QUEUE_LEN) {
			n_ptr = 0;
		}

		// Pause if there is not a slot available in the queue
		if (n_ptr == this->r_ptr) {
			printf("RX queue full!\n");
			Sleep(1);
			continue;
		}

		if (this->can_rx_q[n_ptr].complete) {
			// TODO: is ResetEvent() faster?
			CloseHandle(this->can_rx_q[n_ptr].complete);
		}

		// Overlapped structure required for async read.
		this->can_rx_q[n_ptr].complete = CreateEvent(NULL, TRUE, TRUE, NULL);
		memset(&this->can_rx_q[n_ptr].overlapped, sizeof(OVERLAPPED), 0);
		this->can_rx_q[n_ptr].overlapped.hEvent = this->can_rx_q[n_ptr].complete;
		this->can_rx_q[n_ptr].error = 0;

		if (!WinUsb_ReadPipe(this->usbh, 0x81, this->can_rx_q[n_ptr].data, sizeof(this->can_rx_q[n_ptr].data), &this->can_rx_q[n_ptr].count, &this->can_rx_q[n_ptr].overlapped)) {
			// An overlapped read will return true if done, or false with an
			// error of ERROR_IO_PENDING if the transfer is still in process.
			this->can_rx_q[n_ptr].error = GetLastError();
		}

		// Process the pipe read call from the previous invocation of this function
		if (this->can_rx_q[w_ptr].error == ERROR_IO_PENDING) {
			HANDLE phSignals[2] = { this->can_rx_q[w_ptr].complete, kill_event };
			auto dwError = WaitForMultipleObjects(kill_event ? 2 : 1, phSignals, FALSE, timeoutms);

			// Check if packet, timeout (nope), or break
			if (dwError == WAIT_OBJECT_0) {
				// Signal came from our usb object. Read the returned data.
				if (!GetOverlappedResult(this->usbh, &this->can_rx_q[w_ptr].overlapped, &this->can_rx_q[w_ptr].count, TRUE)) {
					// TODO: handle other error cases better.
					dwError = GetLastError();
					printf("Got overlap error %d\n", dwError);

					continue;
				}
			}
			else {
				WinUsb_AbortPipe(this->usbh, 0x81);

				// Return FALSE to show that the optional signal
				// was set instead of the wait breaking from a
				// message or recoverable error.
				if (dwError == (WAIT_OBJECT_0 + 1)) {
					return FALSE;
				}
				continue;
			}
		}
		else if (this->can_rx_q[w_ptr].error != 0) { // ERROR_BAD_COMMAND happens when device is unplugged.
			return FALSE;
		}

		this->w_ptr = n_ptr;
	}

	return TRUE;
}

void Panda::can_rx_q_pop(PANDA_CAN_MSG msg_out[], int& count) {
	count = 0;

	// No data left in queue
	if (this->r_ptr == this->w_ptr) {
		Sleep(1);
		return;
	}

	auto r_ptr = this->r_ptr;
	for (unsigned long i = 0; i < this->can_rx_q[r_ptr].count; i += sizeof(PANDA_CAN_MSG_INTERNAL)) {
		auto in_msg_raw = (PANDA_CAN_MSG_INTERNAL*)(this->can_rx_q[r_ptr].data + i);
		msg_out[count] = parse_can_recv(in_msg_raw);
		++count;
	}

	// Advance read pointer (wrap around if needed)
	++r_ptr;
	this->r_ptr = (r_ptr == CAN_RX_QUEUE_LEN ? 0 : r_ptr);
}

std::vector<PANDA_CAN_MSG> Panda::can_recv() {
	std::vector<PANDA_CAN_MSG> msg_recv;
	int retcount;
	char buff[sizeof(PANDA_CAN_MSG_INTERNAL) * 4];

	if (this->bulk_read(0x81, buff, sizeof(buff), (PULONG)&retcount, 0) == FALSE)
		return msg_recv;

	for (int i = 0; i < retcount; i += sizeof(PANDA_CAN_MSG_INTERNAL)) {
		PANDA_CAN_MSG_INTERNAL* in_msg_raw = (PANDA_CAN_MSG_INTERNAL*)(buff + i);
		auto in_msg = parse_can_recv(in_msg_raw);
		msg_recv.push_back(in_msg);
	}

	return msg_recv;
}

bool Panda::can_clear(PANDA_CAN_PORT_CLEAR bus) {
	/*Clears all messages from the specified internal CAN ringbuffer as though it were drained.
	bus(int) : can bus number to clear a tx queue, or 0xFFFF to clear the global can rx queue.*/
	return this->control_transfer(REQUEST_OUT, 0xf1, bus, 0, NULL, 0, 0) != -1;
}

std::string Panda::serial_read(PANDA_SERIAL_PORT port_number) {
	std::string result;
	char buff[0x40];
	while (TRUE) {
		int retlen = this->control_transfer(REQUEST_IN, 0xe0, port_number, 0, &buff, 0x40, 0);
		if (retlen <= 0)
			break;
		result += std::string(buff, retlen);
		if (retlen < 0x40) break;
	}
	return result;
}

std::string Panda::serial_read(PANDA_SERIAL_PORT port_number, unsigned int len, unsigned int timeout_ms) {
	std::string result = std::string();
	auto ms_remaining = timeout_ms;
	char buff[0x40];
	while (len > 0 && ms_remaining > 0) {
		int retlen = this->control_transfer(REQUEST_IN, 0xe0, port_number, 0, &buff, min(len, 0x40), 0);
		if (retlen <= 0) {
			ms_remaining -= 1;
			Sleep(1);
			continue;
		}
		result += std::string(buff, retlen);
		len -= retlen;
		ms_remaining = timeout_ms;
	}
	return result;
}

int Panda::serial_write(PANDA_SERIAL_PORT port_number, const std::string& data) {
	int retcount = 0;
	for (int i = 0; i < data.size(); i += 0x3F) {
		std::string slice = std::string(1, (char)port_number) + data.substr(i, min(data.size() - i, 0x3F));
		int retlen;
		if (this->bulk_write(2, slice.c_str(), slice.size(), (PULONG)&retlen, 0) == FALSE) return -1;
		if (retlen != slice.size()) return -1;
		retcount += retlen - 1;
	}
	return retcount;
}

bool Panda::serial_clear(PANDA_SERIAL_PORT port_number) {
	return this->control_transfer(REQUEST_OUT, 0xf2, port_number, 0, NULL, 0, 0) != -1;
}

uint8_t Panda::kline_checksum(const char* data, size_t size) {
	unsigned int checksum = 0;
	for (int i = 0; i < size; i++) {
		checksum += (uint8_t)data[i];
	}
	return (uint8_t)(checksum % 0x100);
}

PANDA_KLINE_MSG Panda::kline_parse(const std::string& data, bool add_checksum) {
	auto bytes = data.c_str();
	auto size = data.size();
	PANDA_KLINE_MSG msg_in;
	ZeroMemory(&msg_in, sizeof(PANDA_KLINE_MSG));
	msg_in.data = std::string(data);

	unsigned int i = 0;
	unsigned int len = 0;
	unsigned int expected_len = 2;
	if (size > i) {
		// data layout: Fmt [Tgt] [Src] [Len] Data CS <- [ ] indicates optional
		msg_in.addr_type = (PANDA_KLINE_ADDR_TYPE)(bytes[i] & KLINE_HEADER_FMT_ADDR_MASK);
		len = bytes[i++] & KLINE_HEADER_FMT_LEN_MASK;
		if (msg_in.addr_type != 0 && size > i + 2) {
			expected_len += 2;
			msg_in.target = bytes[i++];
			msg_in.source = bytes[i++];
		}
		if (len == 0 && size > i + 1) {
			expected_len += 1;
			len = bytes[i++];
		}
		expected_len += len;
		if (expected_len == size) {
			auto checksum = this->kline_checksum(bytes, size - 1);
			if (msg_in.checksum == checksum) {
				msg_in.valid = true;
			}
		}
		else if (add_checksum && expected_len == size + 1) {
			msg_in.checksum = this->kline_checksum(bytes, size);
			msg_in.data += std::string(1, (char)msg_in.checksum);
			msg_in.valid = true;
		}
	}

	return msg_in;
}

bool Panda::kline_slow_init(bool k, bool l, uint8_t addr) {
	return this->control_transfer(REQUEST_OUT, 0xf4, k && l ? 2 : (uint16_t)l, (uint16_t)addr, NULL, 0, 0) != -1;
}

bool Panda::kline_fast_init(bool k, bool l) {
	return this->control_transfer(REQUEST_OUT, 0xf0, k && l ? 2 : (uint16_t)l, 0, NULL, 0, 0) != -1;
}

std::vector<PANDA_KLINE_MSG> Panda::kline_recv(PANDA_SERIAL_PORT port_number) {
	if (port_number != SERIAL_LIN1 && port_number != SERIAL_LIN2) {
		throw "invalid serial port number";
	}

	std::vector<PANDA_KLINE_MSG> msg_recv;

	while (1) {
		// P1/P4 max time between bytes is 20ms
		auto result = this->serial_read(port_number, KLINE_MSG_MAX_LEN, 20);
		if (result.size() == 0) {
			break;
		}

		auto msg_in = this->kline_parse(result, false);
		// TODO: only add if msg_in.valid ???
		msg_recv.push_back(msg_in);
	}

	return msg_recv;
}

bool Panda::kline_send(PANDA_SERIAL_PORT port_number, const std::string& data) {
	if (port_number != SERIAL_LIN1 && port_number != SERIAL_LIN2) {
		throw "invalid serial port number";
	}
	auto msg_out = this->kline_parse(data, true);
	auto result = this->serial_write(port_number, msg_out.data);
	auto echo = this->serial_read(port_number, msg_out.data.size(), 5);
	if (echo != msg_out.data) {
		return false;
	}
	return true;
}
