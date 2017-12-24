#include "stdafx.h"
#include "panda/panda.h"
#include "TestHelpers.h"

#include <tchar.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace panda;

namespace pandaTestNative
{
	TEST_CLASS(DeviceDiscovery)
	{
	public:

		TEST_METHOD(Panda_DevDiscover_ListDevices)
		{
			auto pandas_available = Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas were found."));
			for (auto sn : pandas_available) {
				Assert::IsTrue(sn.size() == 24, _T("panda Serial Number not 24 characters long."));
			}
		}

		TEST_METHOD(Panda_DevDiscover_OpenFirstDevice)
		{
			auto pandas_available = Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas were found."));

			auto p1 = Panda::openPanda(pandas_available[0]);
			Assert::IsFalse(p1 == nullptr, _T("Could not open panda."));
		}

		TEST_METHOD(Panda_DevDiscover_OpenDeviceNoName)
		{
			auto pandas_available = Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas were found."));

			auto p1 = Panda::openPanda("");
			Assert::IsFalse(p1 == nullptr, _T("Could not open panda."));
			Assert::IsTrue(p1->get_usb_sn() == pandas_available[0], _T("Could not open panda."));
		}

		TEST_METHOD(Panda_DevDiscover_OpenDeviceUnavailable)
		{
			auto p1 = Panda::openPanda("ZZZZZZZZZZZZZZZZZZZZZZZZ");
			Assert::IsTrue(p1 == nullptr, _T("Invalid sn still worked."));
		}

		TEST_METHOD(Panda_DevDiscover_WillNotOpenAlreadyOpenedDevice)
		{
			auto pandas_available = Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas were found."));

			auto p1 = Panda::openPanda(pandas_available[0]);
			Assert::IsFalse(p1 == nullptr, _T("Could not open panda."));

			auto p2 = Panda::openPanda(pandas_available[0]);
			Assert::IsTrue(p2 == nullptr, _T("Opened an already open panda."));
		}

		TEST_METHOD(Panda_DevDiscover_OpenedDeviceNotListed)
		{
			auto pandas_available = Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas were found."));

			auto p1 = Panda::openPanda(pandas_available[0]);
			Assert::IsFalse(p1 == nullptr, _T("Could not open panda."));

			auto pandas_available2 = Panda::listAvailablePandas();
			for (auto sn : pandas_available2) {
				Assert::IsFalse(p1->get_usb_sn() == sn, _T("Opened panda appears in list of available pandas."));
			}

		}
	};

	TEST_CLASS(CANOperations)
	{
	public:

		TEST_METHOD(Panda_CAN_Echo)
		{
			auto p0 = getPanda(500, TRUE);

			uint32_t addr = 0xAA;
			bool is_29b = FALSE;
			uint8_t candata[8];

			for (auto canbus : { PANDA_CAN1, PANDA_CAN2, PANDA_CAN3 }) {
				uint8_t len = (rand() % 8) + 1;
				for (size_t i = 0; i < len; i++)
					candata[i] = rand() % 256;

				p0->can_send(addr, is_29b, candata, len, canbus);
				Sleep(10);

				auto can_msgs = p0->can_recv();

				Assert::AreEqual<size_t>(2, can_msgs.size(), _T("Received the wrong number of CAN messages."), LINE_INFO());

				for (auto msg : can_msgs) {
					Assert::IsTrue(msg.addr == addr, _T("Wrong addr."));
					Assert::IsTrue(msg.bus == canbus, _T("Wrong bus."));
					Assert::IsTrue(msg.len == len, _T("Wrong len."));
					Assert::AreEqual(memcmp(msg.dat, candata, msg.len), 0, _T("Received CAN data not equal"));
					for (int i = msg.len; i < 8; i++)
						Assert::IsTrue(msg.dat[i] == 0, _T("Received CAN data not trailed by 0s"));
				}

				Assert::IsTrue(can_msgs[0].is_receipt, _T("Didn't get receipt."));
				Assert::IsFalse(can_msgs[1].is_receipt, _T("Didn't get echo."));
			}
		}

		TEST_METHOD(Panda_CAN_ChangeBaud)
		{
			auto p0 = getPanda(250);
			auto p1 = getPanda(500);

			p0->can_send(0xAA, FALSE, (const uint8_t*)"\x1\x2\x3\x4\x5\x6\x7\x8", 8, panda::PANDA_CAN1);
			panda_recv_loop(p0, 0);
			panda_recv_loop(p1, 0);

			p0->set_can_speed_kbps(panda::PANDA_CAN1, 500);

			auto panda_msg_recv = panda_recv_loop(p0, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0xAA, FALSE, TRUE, "\x1\x2\x3\x4\x5\x6\x7\x8", LINE_INFO());
			panda_msg_recv = panda_recv_loop(p1, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0xAA, FALSE, FALSE, "\x1\x2\x3\x4\x5\x6\x7\x8", LINE_INFO());

			//////////////////

			p0->set_can_speed_kbps(panda::PANDA_CAN1, 250);
			p0->can_send(0xC4, FALSE, (const uint8_t*)"\xA\B\xC\xD\xE\xF\x10\x11", 8, panda::PANDA_CAN1);
			panda_recv_loop(p0, 0);
			panda_recv_loop(p1, 0);

			p1->set_can_speed_kbps(panda::PANDA_CAN1, 250);

			panda_msg_recv = panda_recv_loop(p0, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0xC4, FALSE, TRUE, "\xA\B\xC\xD\xE\xF\x10\x11", LINE_INFO());
			panda_msg_recv = panda_recv_loop(p1, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0xC4, FALSE, FALSE, "\xA\B\xC\xD\xE\xF\x10\x11", LINE_INFO());
		}

		TEST_METHOD(Panda_CAN_ClearClears)
		{
			auto p0 = getPanda(500, TRUE);
			p0->can_send(0xAA, FALSE, (const uint8_t*)"\x0\x1\x2\x3\x4\x5\x6\x7", 8, panda::PANDA_CAN1);
			Sleep(100);
			p0->can_clear(PANDA_CAN_RX);

			auto can_msgs = p0->can_recv();
			Assert::IsTrue(can_msgs.size() == 0, _T("Received messages after a clear."));
		}
	};

	TEST_CLASS(SerialOperations)
	{
	public:

		TEST_METHOD(Panda_LIN_Echo)
		{
			auto p0 = getPanda(500);

			for (auto lin_port : { SERIAL_LIN1, SERIAL_LIN2 }) {
				p0->serial_clear(lin_port);

				for (int i = 0; i < 10; i++) {
					uint8_t len = (rand() % LIN_MSG_MAX_LEN) + 1;
					std::string lindata;
					lindata.reserve(len);

					for (size_t j = 0; j < len; j++)
						lindata += (const char)(rand() % 256);

					p0->serial_write(lin_port, lindata.c_str(), len);
					Sleep(10);

					auto retdata = p0->serial_read(lin_port);
					Assert::AreEqual(retdata, lindata);
				}
			}
		}
	};
}