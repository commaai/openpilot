#include "stdafx.h"
#include "Loader4.h"
#include "pandaJ2534DLL/J2534_v0404.h"
#include "panda_shared/panda.h"
#include "Timer.h"
#include "ECUsim DLL\ECUsim.h"
#include "TestHelpers.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace pandaJ2534DLLTest
{
	TEST_CLASS(J2534DLLInitialization)
	{
	public:

		TEST_CLASS_CLEANUP(deinit) {
			UnloadJ2534Dll();
		}

		TEST_METHOD(J2534_Driver_Init)
		{
			long err = LoadJ2534Dll("pandaJ2534_0404_32.dll");
			Assert::IsTrue(err == 0, _T("Library failed to load properly. Check the export names and library location."));
		}

	};

	TEST_CLASS(J2534DeviceInitialization)
	{
	public:

		TEST_METHOD_INITIALIZE(init) {
			LoadJ2534Dll("pandaJ2534_0404_32.dll");
		}

		TEST_METHOD_CLEANUP(deinit) {
			if (didopen) {
				PassThruClose(devid);
				didopen = FALSE;
			}
			UnloadJ2534Dll();
		}

		TEST_METHOD(J2534_Device_OpenDevice__Empty)
		{
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(""), _T("Failed to open device."), LINE_INFO());
		}

		TEST_METHOD(J2534_Device_OpenDevice__J2534_2)
		{
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev("J2534-2:"), _T("Failed to open device."), LINE_INFO());
		}

		TEST_METHOD(J2534_Device_OpenDevice__SN)
		{
			auto pandas_available = panda::Panda::listAvailablePandas();
			Assert::IsTrue(pandas_available.size() > 0, _T("No pandas detected."));

			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(pandas_available[0].c_str()), _T("Failed to open device."), LINE_INFO());

			auto pandas_available_2 = panda::Panda::listAvailablePandas();
			for (auto panda_sn : pandas_available_2)
				Assert::AreNotEqual(panda_sn, pandas_available[0]);
		}

		TEST_METHOD(J2534_Device_CloseDevice)
		{
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(""), _T("Failed to open device."), LINE_INFO());
			Assert::AreEqual<long>(STATUS_NOERROR, close_dev(devid), _T("Failed to close device."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_DEVICE_ID, PassThruClose(devid), _T("The 2nd close should have failed with ERR_INVALID_DEVICE_ID."), LINE_INFO());
		}

		TEST_METHOD(J2534_Device_ConnectDisconnect)
		{
			unsigned long chanid;
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(""), _T("Failed to open device."), LINE_INFO());
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruConnect(devid, CAN, 0, 500000, &chanid), _T("Failed to open channel."), LINE_INFO());

			Assert::AreEqual<long>(STATUS_NOERROR, PassThruDisconnect(chanid), _T("Failed to close channel."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_CHANNEL_ID, PassThruDisconnect(chanid), _T("The 2nd disconnect should have failed with ERR_INVALID_CHANNEL_ID."), LINE_INFO());
		}

		TEST_METHOD(J2534_Device_ConnectInvalidProtocol)
		{
			unsigned long chanid;
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(""), _T("Failed to open device."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_PROTOCOL_ID, PassThruConnect(devid, 999, 0, 500000, &chanid),
				_T("Did not report ERR_INVALID_PROTOCOL_ID."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_CHANNEL_ID, PassThruDisconnect(chanid), _T("The channel should not have been created."), LINE_INFO());
		}

		bool didopen = FALSE;
		unsigned long devid;

		unsigned long open_dev(const char* name, long assert_err = STATUS_NOERROR, TCHAR* failmsg = _T("Failed to open device.")) {
			unsigned int res = PassThruOpen((void*)name, &devid);
			if (res == STATUS_NOERROR) didopen = TRUE;
			return res;
		}

		unsigned long close_dev(unsigned long devid) {
			unsigned long res = PassThruClose(devid);
			if (res == STATUS_NOERROR) didopen = FALSE;
			return res;
		}

	};

	TEST_CLASS(J2534DeviceCAN)
	{
	public:

		TEST_METHOD_INITIALIZE(init) {
			LoadJ2534Dll("pandaJ2534_0404_32.dll");
		}

		TEST_METHOD_CLEANUP(deinit) {
			if (didopen) {
				PassThruClose(devid);
				didopen = FALSE;
			}
			UnloadJ2534Dll();
		}

		//Test that the BAUD rate of a CAN connection can be changed.
		TEST_METHOD(J2534_CAN_SetBaud)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO()); // ENABLE J2534 ECHO/LOOPBACK
			auto p = getPanda(250);

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 4 + 2, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			j2534_recv_loop(chanid, 0);
			panda_recv_loop(p, 0);

			write_ioctl(chanid, DATA_RATE, 250000, LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, TX_MSG_TYPE, 0, 4 + 2, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_11b_Tx)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 6, 6, "\x0\x0\x3\xAB""HI", LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());

			j2534_recv_loop(chanid, 0, 50); // Check no message is returned (since loopback is off)
		}

		TEST_METHOD(J2534_CAN_29b_Tx)
		{
			auto chanid = J2534_open_and_connect("", CAN, CAN_29BIT_ID, 500000, LINE_INFO());
			auto p = getPanda(500);

			Assert::AreEqual<long>(ERR_INVALID_MSG, J2534_send_msg(chanid, CAN, 0, 0, 0, 6, 6, "\x0\x0\x3\xAB""HI"), _T("11b address should fail to tx."), LINE_INFO());
			J2534_send_msg_checked(chanid, CAN, 0, CAN_29BIT_ID, 0, 6, 6, "\x0\x0\x3\xAB""YO", LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, TRUE, FALSE, "YO", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_11b29b_Tx)
		{
			auto chanid = J2534_open_and_connect("", CAN, CAN_ID_BOTH, 500000, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 6, 6, "\x0\x0\x3\xAB""HI", LINE_INFO());
			J2534_send_msg_checked(chanid, CAN, 0, CAN_29BIT_ID, 0, 6, 6, "\x0\x0\x3\xAB""YO", LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x3AB, TRUE, FALSE, "YO", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_TxEcho)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 9, 9, "\x0\x0\x3\xAB""HIDOG", LINE_INFO());

			auto msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HIDOG", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 0);

			/////////////////////////////////
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO()); // ENABLE J2534 ECHO/LOOPBACK

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 7, 7, "\x0\x0\x3\xAB""SUP", LINE_INFO());

			msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "SUP", LINE_INFO());

			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, TX_MSG_TYPE, 0, 3 + 4, 0, "\x0\x0\x3\xAB""SUP", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_RxAndPassAllFilters)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			J2534_set_PASS_filter(chanid, CAN, 0, 4, "\x0\x0\x0\x0", "\x0\x0\x0\x0", LINE_INFO());
			auto p = getPanda(500);

			p->can_send(0x1FA, FALSE, (const uint8_t*)"ABCDE", 5, panda::PANDA_CAN1);
			p->can_send(0x2AC, FALSE, (const uint8_t*)"HIJKL", 5, panda::PANDA_CAN1);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x1\xFA""ABCDE", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x2\xAC""HIJKL", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_RxAndLimitedPassFilter)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			J2534_set_PASS_filter(chanid, CAN, 0, 4, "\xFF\xFF\xFF\xFF", "\x0\x0\x02\xAC", LINE_INFO());
			auto p = getPanda(500);

			p->can_send(0x1FA, FALSE, (const uint8_t*)"ABCDE", 5, panda::PANDA_CAN1);
			p->can_send(0x2AC, FALSE, (const uint8_t*)"HIJKL", 5, panda::PANDA_CAN1);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x2\xAC""HIJKL", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_RxAndPassBlockFilter)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			J2534_set_PASS_filter(chanid, CAN, 0, 4, "\x0\x0\x0\x0", "\x0\x0\x0\x0", LINE_INFO());
			J2534_set_BLOCK_filter(chanid, CAN, 0, 4, "\xFF\xFF\xFF\xFF", "\x0\x0\x02\xAC", LINE_INFO());
			auto p = getPanda(500);

			p->can_send(0x1FA, FALSE, (const uint8_t*)"ABCDE", 5, panda::PANDA_CAN1);
			p->can_send(0x2AC, FALSE, (const uint8_t*)"HIJKL", 5, panda::PANDA_CAN1);
			p->can_send(0x3FA, FALSE, (const uint8_t*)"MNOPQ", 5, panda::PANDA_CAN1);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2, 1000);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x1\xFA""ABCDE", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x3\xFA""MNOPQ", LINE_INFO());
		}

		//Check that the order of the pass and block filter do not matter
		TEST_METHOD(J2534_CAN_RxAndFilterBlockPass)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			J2534_set_BLOCK_filter(chanid, CAN, 0, 4, "\xFF\xFF\xFF\xFF", "\x0\x0\x02\xAC", LINE_INFO());
			J2534_set_PASS_filter(chanid, CAN, 0, 4, "\x0\x0\x0\x0", "\x0\x0\x0\x0", LINE_INFO());
			auto p = getPanda(500);

			p->can_send(0x1FA, FALSE, (const uint8_t*)"ABCDE", 5, panda::PANDA_CAN1);
			p->can_send(0x2AC, FALSE, (const uint8_t*)"HIJKL", 5, panda::PANDA_CAN1); // Should not pass filter
			p->can_send(0x3FA, FALSE, (const uint8_t*)"MNOPQ", 5, panda::PANDA_CAN1);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2, 2000);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x1\xFA""ABCDE", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x3\xFA""MNOPQ", LINE_INFO());
		}

		//Check that the order of the pass and block filter do not matter
		TEST_METHOD(J2534_CAN_RxAndFilterRemoval)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto filterid0 = J2534_set_BLOCK_filter(chanid, CAN, 0, 4, "\xFF\xFF\xFF\xFF", "\x0\x0\x02\xAC", LINE_INFO());
			auto filterid1 = J2534_set_PASS_filter(chanid, CAN, 0, 4, "\x0\x0\x0\x0", "\x0\x0\x0\x0", LINE_INFO());

			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopMsgFilter(chanid, filterid0), _T("Failed to delete filter."), LINE_INFO());

			auto p = getPanda(500);

			p->can_send(0x1FA, FALSE, (const uint8_t*)"ABCDE", 5, panda::PANDA_CAN1);
			p->can_send(0x2AC, FALSE, (const uint8_t*)"HIJKL", 5, panda::PANDA_CAN1);
			p->can_send(0x3FA, FALSE, (const uint8_t*)"MNOPQ", 5, panda::PANDA_CAN1);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 3, 1000);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x1\xFA""ABCDE", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x2\xAC""HIJKL", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[2], CAN, 0, 0, 5 + 4, 0, "\x0\x0\x3\xFA""MNOPQ", LINE_INFO());
		}

		//Check that the order of the pass and block filter do not matter
		TEST_METHOD(J2534_CAN_RxWithTimeout)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			J2534_set_PASS_filter(chanid, CAN, 0, 4, "\x0\x0\x0\x0", "\x0\x0\x0\x0", LINE_INFO());
			auto p = getPanda(500);

			PASSTHRU_MSG recvbuff;
			unsigned long msgcount = 1;
			unsigned int res = PassThruReadMsgs(chanid, &recvbuff, &msgcount, 100); // Here is where we test the timeout
			Assert::AreEqual<long>(ERR_BUFFER_EMPTY, res, _T("No message should be found"), LINE_INFO());
			Assert::AreEqual<unsigned long>(0, msgcount, _T("Received wrong number of messages."));

			//TODO Test that the timings work right instead of just testing it doesn't crash.
		}

		TEST_METHOD(J2534_CAN_Baud)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 250000, LINE_INFO());
			auto p = getPanda(250);

			J2534_send_msg_checked(chanid, CAN, 0, 0, 0, 6, 6, "\x0\x0\x3\xAB""HI", LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
		}

		TEST_METHOD(J2534_CAN_PeriodicMessageStartStop)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto p = getPanda(500);

			auto msgid = J2534_start_periodic_msg_checked(chanid, CAN, 0, 6, 0, "\x0\x0\x3\xAB""HI", 100, LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 3, 250);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid), _T("Failed to delete filter."), LINE_INFO());
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[2], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());

			auto timediff_1_0 = msg_recv[1].recv_time - msg_recv[0].recv_time;
			auto timediff_2_1 = msg_recv[2].recv_time - msg_recv[1].recv_time;

			std::ostringstream stringStream1;
			stringStream1 << "times1: " << timediff_1_0 << ", " << timediff_2_1 << std::endl;
			Logger::WriteMessage(stringStream1.str().c_str());

			Assert::IsTrue(timediff_1_0 > 90000);
			Assert::IsTrue(timediff_1_0 < 110000);
			Assert::IsTrue(timediff_2_1 > 90000);
			Assert::IsTrue(timediff_2_1 < 110000);

			msg_recv = panda_recv_loop(p, 0, 300);
		}

		TEST_METHOD(J2534_CAN_PeriodicMessageMultipleStartStop)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto p = getPanda(500);

			auto msgid0 = J2534_start_periodic_msg_checked(chanid, CAN, 0, 6, 0, "\x0\x0\x3\xAB""HI", 100, LINE_INFO());
			auto msgid1 = J2534_start_periodic_msg_checked(chanid, CAN, 0, 6, 0, "\x0\x0\x1\x23""YO", 80, LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 9, 370);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid0), _T("Failed to delete filter."), LINE_INFO());
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid1), _T("Failed to delete filter."), LINE_INFO());
			//time diagram. 10 ms per character. * is send event. : is termination of periodic messages.
			//*---------*---------*---------*-----:----* HI
			//*-------*-------*-------*-------*---:----* YO
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x123, FALSE, FALSE, "YO", LINE_INFO());
			check_panda_can_msg(msg_recv[2], 0, 0x123, FALSE, FALSE, "YO", LINE_INFO());
			check_panda_can_msg(msg_recv[3], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[4], 0, 0x123, FALSE, FALSE, "YO", LINE_INFO());
			check_panda_can_msg(msg_recv[5], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[6], 0, 0x123, FALSE, FALSE, "YO", LINE_INFO());
			check_panda_can_msg(msg_recv[7], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[8], 0, 0x123, FALSE, FALSE, "YO", LINE_INFO());

			auto timediff_HI_3_0 = msg_recv[3].recv_time - msg_recv[0].recv_time;
			auto timediff_HI_5_3 = msg_recv[5].recv_time - msg_recv[3].recv_time;
			auto timediff_HI_7_5 = msg_recv[7].recv_time - msg_recv[5].recv_time;

			auto timediff_YO_2_1 = msg_recv[2].recv_time - msg_recv[1].recv_time;
			auto timediff_YO_4_2 = msg_recv[4].recv_time - msg_recv[2].recv_time;
			auto timediff_YO_6_4 = msg_recv[6].recv_time - msg_recv[4].recv_time;
			auto timediff_YO_8_6 = msg_recv[8].recv_time - msg_recv[6].recv_time;

			std::ostringstream stringStreamHi;
			stringStreamHi << "HiTimes: " << timediff_HI_3_0 << ", " << timediff_HI_5_3 << ", " << timediff_HI_7_5 << std::endl;
			Logger::WriteMessage(stringStreamHi.str().c_str());

			std::ostringstream stringStreamYo;
			stringStreamYo << "HiTimes: " << timediff_YO_2_1 << ", " << timediff_YO_4_2 << ", " << timediff_YO_6_4 << ", " << timediff_YO_8_6 << std::endl;
			Logger::WriteMessage(stringStreamYo.str().c_str());

			Assert::IsTrue(timediff_HI_3_0 > 90000);
			Assert::IsTrue(timediff_HI_3_0 < 110000);
			Assert::IsTrue(timediff_HI_5_3 > 90000);
			Assert::IsTrue(timediff_HI_5_3 < 110000);
			Assert::IsTrue(timediff_HI_7_5 > 90000);
			Assert::IsTrue(timediff_HI_7_5 < 110000);

			Assert::IsTrue(timediff_YO_2_1 > 80000-10000);
			Assert::IsTrue(timediff_YO_2_1 < 80000+1000);
			Assert::IsTrue(timediff_YO_4_2 > 80000 - 10000);
			Assert::IsTrue(timediff_YO_4_2 < 80000 + 10000);
			Assert::IsTrue(timediff_YO_6_4 > 80000 - 10000);
			Assert::IsTrue(timediff_YO_6_4 < 80000 + 10000);
			Assert::IsTrue(timediff_YO_8_6 > 80000 - 10000);
			Assert::IsTrue(timediff_YO_8_6 < 80000 + 10000);

			msg_recv = panda_recv_loop(p, 0, 300);
		}

		TEST_METHOD(J2534_CAN_PeriodicMessageStartStop_Loopback)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO()); // ENABLE J2534 ECHO/LOOPBACK
			auto p = getPanda(500);
			auto msgid = J2534_start_periodic_msg_checked(chanid, CAN, 0, 6, 0, "\x0\x0\x3\xAB""HI", 100, LINE_INFO());

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 3, 250);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid), _T("Failed to delete filter."), LINE_INFO());
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[2], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 3);
			check_J2534_can_msg(j2534_msg_recv[0], CAN, TX_MSG_TYPE, 0, 6, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], CAN, TX_MSG_TYPE, 0, 6, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[2], CAN, TX_MSG_TYPE, 0, 6, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());

			auto timediff_1_0 = j2534_msg_recv[1].Timestamp - j2534_msg_recv[0].Timestamp;
			auto timediff_2_1 = j2534_msg_recv[2].Timestamp - j2534_msg_recv[1].Timestamp;

			std::ostringstream stringStream1;
			stringStream1 << "times1: " << timediff_1_0 << ", " << timediff_2_1 << std::endl;
			Logger::WriteMessage(stringStream1.str().c_str());

			Assert::IsTrue(timediff_1_0 > 90000);
			Assert::IsTrue(timediff_1_0 < 110000);
			Assert::IsTrue(timediff_2_1 > 90000);
			Assert::IsTrue(timediff_2_1 < 110000);

			msg_recv = panda_recv_loop(p, 0, 300);
		}

		TEST_METHOD(J2534_CAN_PeriodicMessageWithTx)
		{
			auto chanid = J2534_open_and_connect("", CAN, 0, 500000, LINE_INFO());
			auto p = getPanda(500);
			auto msgid = J2534_start_periodic_msg_checked(chanid, CAN, 0, 6, 0, "\x0\x0\x3\xAB""HI", 100, LINE_INFO());

			J2534_send_msg(chanid, CAN, 0, 0, 0, 7, 0, "\x0\x0\x3\xAB""LOL");

			std::vector<panda::PANDA_CAN_MSG> msg_recv = panda_recv_loop(p, 4, 250);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid), _T("Failed to delete filter."), LINE_INFO());
			check_panda_can_msg(msg_recv[0], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x3AB, FALSE, FALSE, "LOL", LINE_INFO());//Staggered write inbetween multiple scheduled TXs
			check_panda_can_msg(msg_recv[2], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());
			check_panda_can_msg(msg_recv[3], 0, 0x3AB, FALSE, FALSE, "HI", LINE_INFO());

			auto timediff_2_0 = msg_recv[2].recv_time - msg_recv[0].recv_time;
			auto timediff_3_2 = msg_recv[3].recv_time - msg_recv[2].recv_time;

			std::ostringstream stringStream1;
			stringStream1 << "times1: " << timediff_2_0 << ", " << timediff_3_2 << std::endl;
			Logger::WriteMessage(stringStream1.str().c_str());

			Assert::IsTrue(timediff_2_0 > 90000);
			Assert::IsTrue(timediff_2_0 < 110000);
			Assert::IsTrue(timediff_3_2 > 90000);
			Assert::IsTrue(timediff_3_2 < 110000);

			msg_recv = panda_recv_loop(p, 0, 300);
		}

		TEST_METHOD(J2534_CAN_BaudInvalid)
		{
			unsigned long chanid;
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(""), _T("Failed to open device."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_BAUDRATE, PassThruConnect(devid, CAN, 0, 6000000, &chanid), _T("Baudrate should have been invalid."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_BAUDRATE, PassThruConnect(devid, CAN, 0, 200, &chanid), _T("Baudrate should have been invalid."), LINE_INFO());
			Assert::AreEqual<long>(ERR_INVALID_BAUDRATE, PassThruConnect(devid, CAN, 0, 250010, &chanid), _T("Baudrate should have been invalid."), LINE_INFO());
		}

		bool didopen = FALSE;
		unsigned long devid;

		unsigned long open_dev(const char* name, long assert_err = STATUS_NOERROR, TCHAR* failmsg = _T("Failed to open device.")) {
			unsigned int res = PassThruOpen((void*)name, &devid);
			if (res == STATUS_NOERROR) didopen = TRUE;
			return res;
		}

		unsigned long J2534_open_and_connect(const char* name, unsigned long ProtocolID, unsigned long Flags, unsigned long bps, const __LineInfo* pLineInfo = NULL) {
			unsigned long chanid;
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(name), _T("Failed to open device."), pLineInfo);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruConnect(devid, ProtocolID, Flags, bps, &chanid), _T("Failed to open channel."), pLineInfo);
			write_ioctl(chanid, LOOPBACK, FALSE, LINE_INFO()); // DISABLE J2534 ECHO/LOOPBACK
			return chanid;
		}

	};

	TEST_CLASS(J2534DeviceISO15765)
	{
	public:

		TEST_METHOD_INITIALIZE(init) {
			LoadJ2534Dll("pandaJ2534_0404_32.dll");
		}

		TEST_METHOD_CLEANUP(deinit) {
			if (didopen) {
				PassThruClose(devid);
				didopen = FALSE;
			}
			UnloadJ2534Dll();
		}

		//Test that the BAUD rate of a ISO15765 connection can be changed.
		TEST_METHOD(J2534_ISO15765_SetBaud)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, 0, 500000, LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO()); // ENABLE J2534 ECHO/LOOPBACK
			auto p = getPanda(250);

			J2534_send_msg_checked(chanid, ISO15765, 0, 0, 0, 4 + 2, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			j2534_recv_loop(chanid, 0);
			panda_recv_loop(p, 0);

			write_ioctl(chanid, DATA_RATE, 250000, LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, TX_INDICATION, 0, 4, 0, "\x0\x0\x3\xAB", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, TX_MSG_TYPE, 0, 4 + 2, 0, "\x0\x0\x3\xAB""HI", LINE_INFO());
			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x3AB, FALSE, FALSE, "\x2""HI", LINE_INFO());
		}

		///////////////////// Tests checking things don't send/receive /////////////////////

		//Check tx PASSES and rx FAIL WITHOUT a filter. 29 bit. NO Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_PassTxFailRx_29b_NoFilter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			auto p = getPanda(500);

			//TX: works because all single frame writes should work (with or without a flow contorl filter)
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x07""TX_TEST", LINE_INFO());

			//RX: Reads require a flow control filter, and should fail without one.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x06\x41\x00\xff\xff\xff\xfe", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}

		//Check tx and rx FAIL WITHOUT a filter. 29 bit. NO Filter. NoPadding. STD address. First Frame.
		TEST_METHOD(J2534_ISO15765_FailTxRx_29b_NoFilter_NoPad_STD_FF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			auto p = getPanda(500);

			//TX
			Assert::AreEqual<long>(ERR_NO_FLOW_CONTROL, J2534_send_msg(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 12, 0, "\x18\xda\xef\xf1\xA1\xB2\xC3\xD4\xE5\xF6\x09\x1A"),
				_T("Should fail to tx without a filter."), LINE_INFO());
			j2534_recv_loop(chanid, 0);
			panda_recv_loop(p, 0);

			//RX; Send full response and check didn't receive flow control from J2534 device
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x14\x49\x02\x01""1D4", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""GP00R55", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""B123456", 8, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);//Check a full message is not accepted.
		}

		//Check tx PASSES and rx FAIL with a MISMATCHED filter. 29 bit. Mismatch Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_PassTxFailRx_29b_MismatchFilter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//TX: works because all single frame writes should work (with or without a flow contorl filter)
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 6, 0, "\x18\xda\xe0\xf1""\x11\x22", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xe0\xf1", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAE0F1, TRUE, FALSE, "\x02""\x11\x22", LINE_INFO());

			//RX. Send ISO15765 single frame to device. Address still doesn't match filter, so should not be received.
			checked_panda_send(p, 0x18DAF1E0, TRUE, "\x06\x41\x00\xff\xff\xff\xfe", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}

		//Check tx and rx FAIL with a MISMATCHED filter. 29 bit. Mismatch Filter. NoPadding. STD address. First Frame.
		TEST_METHOD(J2534_ISO15765_FailTxRx_29b_MismatchFilter_NoPad_STD_FF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//TX
			Assert::AreEqual<long>(ERR_NO_FLOW_CONTROL, J2534_send_msg(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 12, 0, "\x18\xda\xe0\xf1""USELESS STUFF"),
				_T("Should fail to tx without a filter."), LINE_INFO());
			j2534_recv_loop(chanid, 0);
			panda_recv_loop(p, 0);

			//RX; Send a full response and check didn't receive flow control from J2534 device
			checked_panda_send(p, 0x18DAF1E0, TRUE, "\x10\x14\x49\x02\x01""1D4", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1E0, TRUE, "\x21""GP00R55", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1E0, TRUE, "\x22""B123456", 8, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);//Check a full message is not accepted.
		}

		//Check tx FAILS with a MISMATCHED filter 29bit flag. 29 bit. Mismatch Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_FailTxRx_29b_MismatchFilterFlag29b_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x0\x0\x1\xab", "\x0\x0\x1\xcd", LINE_INFO());
			auto p = getPanda(500);

			//TX
			Assert::AreEqual<long>(ERR_INVALID_MSG, J2534_send_msg(chanid, ISO15765, 0, 0, 0, 6, 0, "\x0/x0/x1/xcd\x01\x00"),
				_T("mismatched address should fail to tx."), LINE_INFO());
			j2534_recv_loop(chanid, 0);
			panda_recv_loop(p, 0);

			//RX. Send ISO15765 single frame to device. Address still doesn't match filter, so should not be received.
			checked_panda_send(p, 0x1ab, FALSE, "\x06\x41\x00\xff\xff\xff\xfe", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}

		///////////////////// Tests checking things actually send/receive. Standard Addressing /////////////////////

		//Check rx passes with filter. 29 bit. Good Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessRx_29b_Filter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x07""ABCD123", 8, 0, LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID, 0, 11, 11, "\x18\xda\xf1\xef""ABCD123", LINE_INFO());
		}

		//Check tx passes with filter. 29 bit. Good Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x07""TX_TEST", LINE_INFO());
		}

		//Check tx passes with filter. 29 bit. Good Filter. NoPadding. STD address. Single Frame. Loopback.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_SF_LOOPBACK)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x07""TX_TEST", LINE_INFO());
		}

		//Check rx passes with filter. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_SuccessRx_29b_Filter_NoPad_STD_FFCF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ninete", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""en byte", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""s here", 7, 0, LINE_INFO());

			//Check J2534 constructed the whole message
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID, 0, 4 + 0x13, 4 + 0x13, "\x18\xda\xf1\xef""nineteen bytes here", LINE_INFO());
		}

		//Check multi frame tx passes with filter. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_FFCF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 14, 0, "\x18\xda\xef\xf1""\xAA\xBB\xCC\xDD\xEE\xFF\x11\x22\x33\x44", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 0); // No TxDone msg until after the final tx frame is sent

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x0A""\xAA\xBB\xCC\xDD\xEE\xFF", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x0\x0", 3, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""\x11\x22\x33\x44", LINE_INFO());

			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
		}

		//Check rx passes with filter. 11 bit. Good Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessRx_11b_Filter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, 0, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, 0, 4, "\xff\xff\xff\xff", "\x0\x0\x1\xab", "\x0\x0\x1\xcd", LINE_INFO());
			auto p = getPanda(500);

			checked_panda_send(p, 0x1ab, FALSE, "\x07""ABCD123", 8, 0, LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, 0, 0, 11, 11, "\x0\x0\x1\xab""ABCD123", LINE_INFO());
		}

		//Check tx passes with filter. 11 bit. Good Filter. NoPadding. STD address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_11b_Filter_NoPad_STD_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, 0, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, 0, 4, "\xff\xff\xff\xff", "\x0\x0\x1\xab", "\x0\x0\x1\xcd", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, 0, 0, 11, 0, "\x0\x0\x1\xcd""TX_TEST", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, TX_INDICATION, 0, 4, 0, "\x0\x0\x1\xcd", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x1CD, FALSE, FALSE, "\x07""TX_TEST", LINE_INFO());
		}

		//Check tx passes with filter multiple times. 29 bit. Good Filter. NoPadding. STD address. Multiple Single Frames.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MultipleSF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 9, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 4);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[2], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[3], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 9, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x07""TX_TEST", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x05""HELLO", LINE_INFO());
		}

		//Check that receiver's flow control block size requests are respected. 29 bit. Good Filter. NoPadding. STD address. Multiple Frames with multiple flow control.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MF_FLOWCONTROLBlockSize)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 52, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXXYYZZ", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x34""AABBCC", LINE_INFO());

			// [flow_status, block_size, st_min]
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x01\x00", 3, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x02\x00", 3, 2, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x01\x00", 3, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x00\x00", 3, 3, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x27""YYZZ", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
		}

		//Check that receiver's flow control separation time requests are respected. 29 bit. Good Filter. NoPadding. STD address. Multiple Frames with multiple flow control.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MF_FLOWCONTROLSTMinMultiFc)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 52, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXXYYZZ", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x34""AABBCC", LINE_INFO());

			// [flow_status, block_size, st_min]
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x03\x0A", 3, 3, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			auto timediff0_1_0 = panda_msg_recv[1].recv_time - panda_msg_recv[0].recv_time;
			auto timediff0_2_1 = panda_msg_recv[2].recv_time - panda_msg_recv[1].recv_time;

			std::ostringstream stringStream0;
			stringStream0 << "times0: " << timediff0_1_0 << ", " << timediff0_2_1 << std::endl;
			Logger::WriteMessage(stringStream0.str().c_str());

			Assert::IsTrue(timediff0_1_0 > 10000);
			Assert::IsTrue(timediff0_1_0 < 32000);//Flexible, but trying to make sure things don't just all lag for a second or something
			Assert::IsTrue(timediff0_2_1 > 10000);
			Assert::IsTrue(timediff0_2_1 < 32000);

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x04\x20", 3, 4, LINE_INFO(), 500);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x27""YYZZ", LINE_INFO());
			auto timediff1_1_0 = panda_msg_recv[1].recv_time - panda_msg_recv[0].recv_time;
			auto timediff1_2_1 = panda_msg_recv[2].recv_time - panda_msg_recv[1].recv_time;
			auto timediff1_3_2 = panda_msg_recv[3].recv_time - panda_msg_recv[2].recv_time;

			std::ostringstream stringStream1;
			stringStream1 << "times1: " << timediff1_1_0 << ", " << timediff1_2_1 << ", " << timediff1_3_2 << std::endl;
			Logger::WriteMessage(stringStream1.str().c_str());

			Assert::IsTrue(timediff1_1_0 > 32000);
			Assert::IsTrue(timediff1_2_1 > 32000);
			Assert::IsTrue(timediff1_3_2 > 32000);

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
		}

		//Check that receiver's flow control separation time requests are respected 2. 29 bit. Good Filter. NoPadding. STD address. Multiple Frames with one flow control.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MF_FLOWCONTROLSTMinSingleFc)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 52, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXXYYZZ", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x34""AABBCC", LINE_INFO());

			// [flow_status, block_size, st_min]
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x07\x0A", 3, 7, LINE_INFO(), 500);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[4], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[5], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[6], 0, 0x18DAEFF1, TRUE, FALSE, "\x27""YYZZ", LINE_INFO());

			auto timediff_1_0 = panda_msg_recv[1].recv_time - panda_msg_recv[0].recv_time;
			auto timediff_2_1 = panda_msg_recv[2].recv_time - panda_msg_recv[1].recv_time;
			auto timediff_3_2 = panda_msg_recv[3].recv_time - panda_msg_recv[2].recv_time;
			auto timediff_4_3 = panda_msg_recv[4].recv_time - panda_msg_recv[3].recv_time;
			auto timediff_5_4 = panda_msg_recv[5].recv_time - panda_msg_recv[4].recv_time;
			auto timediff_6_5 = panda_msg_recv[6].recv_time - panda_msg_recv[5].recv_time;

			std::ostringstream stringStream1;
			stringStream1 << "times1: " << timediff_1_0 << ", " << timediff_2_1 << ", " << timediff_3_2 <<
				", " << timediff_4_3 << ", " << timediff_5_4 << ", " << timediff_6_5 << std::endl;
			Logger::WriteMessage(stringStream1.str().c_str());

			Assert::IsTrue(timediff_1_0 > 10000);
			Assert::IsTrue(timediff_2_1 > 10000);
			Assert::IsTrue(timediff_3_2 > 10000);
			Assert::IsTrue(timediff_4_3 > 10000);
			Assert::IsTrue(timediff_5_4 > 10000);
			Assert::IsTrue(timediff_6_5 > 10000);
		}

		//Check that tx works for messages with more than 16 frames. 29 bit. Good Filter. NoPadding. STD address. Large multiframe message.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MF_LotsOfFrames)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 125, 0,
				"\x18\xda\xef\xf1"
				"AABBCC""DDEEFFG""GHHIIJJ""KKLLMMN""NOOPPQQ""RRSSTTU""UVVWWXX""YYZZ112""2334455""6677889"
				"900abcd""efghijk""lmnopqr""stuvwxy""z!@#$%^""&*()_+-""=`~ABCD""EFGHIJK", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x7D""AABBCC", LINE_INFO());

			// [flow_status, block_size, st_min]
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x00\x00", 3, 17, LINE_INFO(), 1000);
			check_panda_can_msg(panda_msg_recv[0],  0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1],  0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2],  0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[3],  0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[4],  0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[5],  0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[6],  0, 0x18DAEFF1, TRUE, FALSE, "\x27""YYZZ112", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[7],  0, 0x18DAEFF1, TRUE, FALSE, "\x28""2334455", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[8],  0, 0x18DAEFF1, TRUE, FALSE, "\x29""6677889", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[9],  0, 0x18DAEFF1, TRUE, FALSE, "\x2A""900abcd", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[10], 0, 0x18DAEFF1, TRUE, FALSE, "\x2B""efghijk", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[11], 0, 0x18DAEFF1, TRUE, FALSE, "\x2C""lmnopqr", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[12], 0, 0x18DAEFF1, TRUE, FALSE, "\x2D""stuvwxy", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[13], 0, 0x18DAEFF1, TRUE, FALSE, "\x2E""z!@#$%^", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[14], 0, 0x18DAEFF1, TRUE, FALSE, "\x2F""&*()_+-", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[15], 0, 0x18DAEFF1, TRUE, FALSE, "\x20""=`~ABCD", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[16], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""EFGHIJK", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
		}

		//Check tx passes with filter multiple times. 29 bit. Good Filter. NoPadding. STD address. Multiple Single Frames.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MultipleMFSF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 23, 0, "\x18\xda\xef\xf1""Long data because I can", LINE_INFO());
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 9, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x17""Long d", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x00\x00", 3, 4, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""ata bec", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""ause I ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""can", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x05""HELLO", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 4);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 23, 0, "\x18\xda\xef\xf1""Long data because I can", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[2], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[3], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 5, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());
		}

		//Check tx passes after message timeout. 29 bit. Good Filter. NoPadding. STD address. Multiple Frame timeout then Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MFTimeoutSFSuccess)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 23, 0, "\x18\xda\xef\xf1""Long data because I can", LINE_INFO());
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 9, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 2, 1000);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x17""Long d", LINE_INFO()); //First Frame. Not replying so it needs to time out.
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x05""HELLO", LINE_INFO()); //Reply to the next message.

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 5, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());
		}

		//Check tx passes after mid-message timeout. 29 bit. Good Filter. NoPadding. STD address. Multiple Frame mid-timeout then Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MFMidTimeoutSFSuccess)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 23, 0, "\x18\xda\xef\xf1""Long data because I can", LINE_INFO());
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 9, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x17""Long d", LINE_INFO()); //First Frame. Not replying so it needs to time out.

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x01\x00", 3, 2, LINE_INFO(), 1000);//Start a conversation
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""ata bec", LINE_INFO());//Check passthru device sent more data, but don't reply to it
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x05""HELLO", LINE_INFO()); //Reply to the next message.

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 5, 0, "\x18\xda\xef\xf1""HELLO", LINE_INFO());
		}

		//Check slow tx passes without hitting FC timeout. 29 bit. Good Filter. NoPadding. STD address. Long STmin, catches if FC timeout applies before needed.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_SLOWMF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x06\x7F", 3, 6, LINE_INFO(), 3000);//Start a conversation... but slow. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());//Check this convo doesn't trigger that timeout.
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[4], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());//Some of these should fail to recv if there is an issue.
			check_panda_can_msg(panda_msg_recv[5], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());
		}

		//Check MF tx can be sent along side of a periodic message. 29 bit. Good Filter. NoPadding. STD address. Long STmin, checks that MF tx and periodic TX don't break each other.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_SLOWMF_WithPeriodicMsg)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			//Timing diagram of this test.
			//* is a periodic msg transfer; F is first frame, L is Flow control, C is Consecutive Frame.
			// *~~~~~~~*~~~~~~~*~~~~~~~* (The alignment here is unimportant. The exact order is not checked.
			//F C----C----C----C----C----C (100 ms between Cs)
			// L

			auto msgid = J2534_start_periodic_msg_checked(chanid, ISO15765, CAN_29BIT_ID, 6, 0, "\x18\xda\xef\xf1""HI", 130, LINE_INFO());
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x02""HI", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			Assert::IsTrue(p->can_send(0x18DAF1EF, TRUE, (const uint8_t*)"\x30\x06\x64", 3, panda::PANDA_CAN1), _T("Panda send says it failed."), LINE_INFO());

			Timer t_permsg = Timer();
			Timer t_MFmsg = Timer();
			unsigned int MFframesReceived = 0;
			unsigned int PeriodicMsgReceived = 1; //Because of the first panda_recv_loop above.
			std::array<std::string, 6> const mfMsgExpectedParts{ "\x21""DDEEFFG", "\x22""GHHIIJJ", "\x23""KKLLMMN", "\x24""NOOPPQQ", "\x25""RRSSTTU", "\x26""UVVWWXX" };

			while (TRUE) {
				std::vector<panda::PANDA_CAN_MSG>msg_recv = p->can_recv();
				for (auto msg : msg_recv) {
					if (msg.is_receipt) continue;
					if ((msg.dat[0] & 0xf0) == 0x20) {
						Assert::AreEqual<std::string>(mfMsgExpectedParts[MFframesReceived], std::string((const char*)msg.dat, msg.len), _T("Got wrong part of MF msg."), LINE_INFO());
						MFframesReceived++;
						t_MFmsg.reset();
					} else if (std::string((const char*)msg.dat, msg.len) == "\x02HI") {
						PeriodicMsgReceived++;
						t_permsg.reset();
					} else {
						Assert::IsTrue(FALSE, _T("Got impossible message. Something is very wrong. Check other tests."), LINE_INFO());
					}
				}

				if (MFframesReceived >= 6) break;
				Assert::IsTrue(300 > t_permsg.getTimePassed(), _T("Timed out waiting for periodic msessage frame."), LINE_INFO());
				Assert::IsTrue(300 > t_MFmsg.getTimePassed(), _T("Timed out waiting for multiframe msessage frame."), LINE_INFO());

				if (msg_recv.size() == 0)
					Sleep(10);
			}

			//Stop the periodic message and grab any data it may have sent since we last checked.
			//Not sure if this is needed.
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruStopPeriodicMsg(chanid, msgid), _T("Failed to delete filter."), LINE_INFO());
			auto extra_panda_msg = panda_recv_loop_loose(p, 0, 200);
			for (auto msg : extra_panda_msg) {
				 if (std::string((const char*)msg.dat, msg.len) == "\x02HI") {
					PeriodicMsgReceived++;
					Logger::WriteMessage("Received extra periodic message.");
				} else {
					Assert::IsTrue(FALSE, _T("Got impossible message. Something is very wrong. Check other tests."), LINE_INFO());
				}
			}

			Assert::IsTrue(PeriodicMsgReceived > 3, _T("Did not receive enough periodic messages. Likely canceled or delayed."), LINE_INFO());

			std::ostringstream stringStream;
			stringStream << "PeriodicMsgReceived = " << PeriodicMsgReceived << std::endl;
			Logger::WriteMessage(stringStream.str().c_str());

			unsigned int periodicTxIndicationCount = 0;
			unsigned int TxIndicationCount = 0;
			auto j2534_msg_recv = j2534_recv_loop(chanid, 2 + (PeriodicMsgReceived * 2));
			for (int i = 0; i < PeriodicMsgReceived + 1; i++) {
				check_J2534_can_msg(j2534_msg_recv[(i * 2) + 0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
				switch (j2534_msg_recv[(i * 2) + 1].DataSize) {
				case 4 + 2:
					check_J2534_can_msg(j2534_msg_recv[(i * 2) + 1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 2, 0, "\x18\xda\xef\xf1""HI", LINE_INFO());
					break;
				case 4 + 48:
					check_J2534_can_msg(j2534_msg_recv[(i * 2) + 1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());
					break;
				default:
					Assert::IsTrue(FALSE, _T("Got unexpected data!"), LINE_INFO());
				}
			}

			Assert::AreNotEqual<unsigned int>(PeriodicMsgReceived, periodicTxIndicationCount, _T("Wrong number of periodic msgs reported by passthru device."), LINE_INFO());
			Assert::AreNotEqual<unsigned int>(1, TxIndicationCount, _T("Wrong number of TX msgs reported by passthru device."), LINE_INFO());
		}

		///////////////////// Tests checking things break or recover during send/receive /////////////////////

		//Check rx FAILS when frame is dropped. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_FailRx_29b_Filter_NoPad_STD_FFCF_DropFrame)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ninete", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the rest of the message
			//Missing the 2nd frame "\x21""en byte"
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""s here", 7, 0, LINE_INFO());

			//Check J2534 DOES NOT construct the incomplete message
			j2534_recv_loop(chanid, 0);
		}

		//Check rx ignores frames that arrive out of order. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_PassRx_29b_Filter_NoPad_STD_FFCF_FrameNumSkip)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ABCDEF", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""XXXXXX", 7, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""GHIJKLM", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x23""ZZZZZZ", 7, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""NOPQRS", 7, 0, LINE_INFO());

			//Check J2534 constructa the complete message from the correctly numbered frames.
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID, 0, 4 + 0x13, 4 + 0x13, "\x18\xda\xf1\xef""ABCDEFGHIJKLMNOPQRS", LINE_INFO());
		}

		//Check Single Frame rx RESETS ongoing multiframe transmission. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_PassRx_29b_Filter_NoPad_STD_SFRxResetsMFRx)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ABCDEF", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the next part of the message multi message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""GHIJKLM", 8, 0, LINE_INFO());

			//ABORTING MESSAGE
			//Send a NEW single frame message and check the J2534 device gets it (but not the original message it was receiving.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x06""ABC123", 7, 0, LINE_INFO());
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID, 0, 10, 10, "\x18\xda\xf1\xef""ABC123", LINE_INFO());

			//Resume sending the old message, and check th eJ2534 device didn't get a message.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""NOPQRS", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}

		//The documentation says that a s ingle channel can not send and receive messages trhough a
		//single conversation (flow control filter) at the same time. However, the required behavior
		//when this is detected is not described. This test was my best understanding of how it was
		//wanted, but I no longer see the point. For now I am disabling it.
		/*//Check Single Frame tx RESETS ongoing multiframe rx transmission. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_PassRx_29b_Filter_NoPad_STD_SFTxResetsMFRx)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ABCDEF", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the next part of the message multi message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""GHIJKLM", 8, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);

			//ABORTING MESSAGE
			//Send a NEW single frame message and check the J2534 device gets it (but not the original message it was receiving.
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 11, 0, "\x18\xda\xef\xf1""TX_TEST", LINE_INFO());
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());

			panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x07""TX_TEST", LINE_INFO());
			///////////////////////////

			//Resume sending the old message, and check th eJ2534 device didn't get a message.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""NOPQRS", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}*/

		//TODO check rx is cleared by tx (multi). Or not.... read above note.

		//Check multiframe rx RESETS ongoing multiframe transmission. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_PassRx_29b_Filter_NoPad_STD_FFCF_MFRxResetsMFRx)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ABCDEF", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the next part of the multi message A
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""GHIJKLM", 8, 0, LINE_INFO());

			//ABORTING MESSAGE A
			//Send a NEW multi frame message (B) and check the J2534 device gets it (but not the original message it was receiving.
			//Send first frame, then check we get a flow control frame
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ninete", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""en byte", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""s here", 7, 0, LINE_INFO());

			//Check J2534 constructed the whole message
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID, 0, 4 + 0x13, 4 + 0x13, "\x18\xda\xf1\xef""nineteen bytes here", LINE_INFO());
			//////////////////////// End sending B

			//Resume sending the multi message A, and check th eJ2534 device didn't get a message.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""NOPQRS", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);
		}

		//Check rx fails gracefully if final CF of MF rx is too short. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_FailRxFinalCFTooShort_29b_Filter_NoPad_STD_FFCF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ninete", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x30\x00\x00", 3), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE, 0, 4, 0, "\x18\xda\xf1\xef", LINE_INFO());

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""en byte", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""s her", 6, 0, LINE_INFO()); //The transaction should reset here because more data could have been sent.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x23""e", 2, 0, LINE_INFO());

			//Check J2534 constructed the whole message
			j2534_msg_recv = j2534_recv_loop(chanid, 0);
		}

		//Check rx fails gracefully if first frame is too short. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_FailRxFFTooShort_29b_Filter_NoPad_STD_FFCF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame. The transaction should reset immediately because more data could have been sent in this frame.
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x10\x13""ninet", 7, 0, LINE_INFO());
			j2534_recv_loop(chanid, 0);

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x21""een byt", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x22""es here", 8, 0, LINE_INFO());

			//Check J2534 constructed the whole message
			j2534_recv_loop(chanid, 0);
		}

		//Check MF tx will stop upon receiving a flow control ABORT. 29 bit. Good Filter. NoPadding. STD address. Large STmin, then abort, then send SF.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD__MF_FCAbort_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x02\x20", 3, 2, LINE_INFO());//Start a conversation. FC timeout is 32 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x32\x0\x0", 3, 0, LINE_INFO());//Abort the conversation

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());
		}

		//Check MF tx will stop upon receiving a flow control ABORT during valid blocksize. 29 bit. Good Filter. NoPadding. STD address. Large STmin, then mid tx abort, then send SF.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD__MF_FCMixTXAbort_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x06\x7F", 3, 1, LINE_INFO(), 200);//Start a conversation. FC timeout is 127 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x32\x0\x0", 3, 0, LINE_INFO());//Abort the conversation
			panda_recv_loop(p, 0, 200);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());
		}

		//Check slow tx can be stalled past timeout with CF WAIT frames. 29 bit. Good Filter. NoPadding. STD address. MF tx that would timeout without WAIT frames.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MFWithWaitFrames)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			write_ioctl(chanid, ISO15765_WFT_MAX, 10, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x02\x40", 3, 2, LINE_INFO(), 3000);//Start a conversation. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());

			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x0\x0", 3, 4, LINE_INFO(), 3000);//Start a conversation. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());//Some of these should fail to recv if there is an issue.
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());
		}

		//Check slow tx can be stalled past timeout with CF WAIT frames during normal TX. 29 bit. Good Filter. NoPadding. STD address. Stalling working MF tx.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MFWithMidTXWaitFrames)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			write_ioctl(chanid, ISO15765_WFT_MAX, 10, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x06\x64", 3, 2, LINE_INFO(), 120);//Start a conversation. STmin 100. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x0\x0", 3, 4, LINE_INFO(), 3000);//Start a conversation. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[2], 0, 0x18DAEFF1, TRUE, FALSE, "\x25""RRSSTTU", LINE_INFO());//Some of these should fail to recv if there is an issue.
			check_panda_can_msg(panda_msg_recv[3], 0, 0x18DAEFF1, TRUE, FALSE, "\x26""UVVWWXX", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());
		}

		//Check that too many WAIT frames will abort the transfer. 29 bit. Good Filter. NoPadding. STD address. Too much stalling causes abort.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_STD_MFTooManyWaitFrames)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID, 4, "\xff\xff\xff\xff", "\x18\xda\xf1\xef", "\x18\xda\xef\xf1", LINE_INFO());
			write_ioctl(chanid, LOOPBACK, TRUE, LINE_INFO());
			write_ioctl(chanid, ISO15765_WFT_MAX, 2, LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 48, 0, "\x18\xda\xef\xf1""AABBCCDDEEFFGGHHIIJJKKLLMMNNOOPPQQRRSSTTUUVVWWXX", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x10\x30""AABBCC", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x02\x64", 3, 2, LINE_INFO(), 120);//Start a conversation. STmin 100. FC timeout is 250 ms.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x21""DDEEFFG", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x22""GHHIIJJ", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x02\x64", 3, 2, LINE_INFO(), 120);//Resume the conversation.
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x23""KKLLMMN", LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, "\x24""NOOPPQQ", LINE_INFO());

			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x31\x0\x0", 3, 0, LINE_INFO(), 100);//Delay the conversation.
			Sleep(100);

			//Should not resume because the conversation has been delayed too long.
			panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x30\x0\x0", 3, 0, LINE_INFO(), 300);

			//Send a SF message to check the tubes are not clogged.
			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 2);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION, 0, 4, 0, "\x18\xda\xef\xf1", LINE_INFO());
			check_J2534_can_msg(j2534_msg_recv[1], ISO15765, CAN_29BIT_ID | TX_MSG_TYPE, 0, 4 + 4, 0, "\x18\xda\xef\xf1""SUP!", LINE_INFO());
		}

		///////////////////// Tests checking things actually send/receive. Ext 5 byte Addressing /////////////////////

		//Check rx passes with filter. 29 bit. Good Filter. NoPadding. EXT address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessRx_29b_Filter_NoPad_EXT_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 5, "\xff\xff\xff\xff\xff", "\x18\xda\xf1\xef\x13", "\x18\xda\xef\xf1\x13", LINE_INFO());
			auto p = getPanda(500);

			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x06""ABC123", 8, 0, LINE_INFO());

			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 0, 11, 11, "\x18\xda\xf1\xef\x13""ABC123", LINE_INFO());
		}

		//Check tx passes with filter. 29 bit. Good Filter. NoPadding. EXT address. Single Frame.
		TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_EXT_SF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 5, "\xff\xff\xff\xff\xff", "\x18\xda\xf1\xef\x13", "\x18\xda\xef\xf1\x13", LINE_INFO());
			auto p = getPanda(500);

			J2534_send_msg_checked(chanid, ISO15765, 0, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 0, 11, 0, "\x18\xda\xef\xf1\x13""DERP!!", LINE_INFO());
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | TX_INDICATION | ISO15765_ADDR_TYPE, 0, 5, 0, "\x18\xda\xef\xf1\x13", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 1);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, "\x13""\x06""DERP!!", LINE_INFO());
		}

		//Check rx passes with filter. 29 bit. Good Filter. NoPadding. EXT address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_SuccessRx_29b_Filter_NoPad_EXT_FFCF)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 5, "\xff\xff\xff\xff\xff", "\x18\xda\xf1\xef\x13", "\x18\xda\xef\xf1\x13", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			Assert::IsTrue(p->can_send(0x18DAF1EF, TRUE, (const uint8_t*)"\x13""\x10\x13""ninet", 8, panda::PANDA_CAN1), _T("Panda send says it failed."), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE | ISO15765_ADDR_TYPE, 0, 5, 0, "\x18\xda\xf1\xef\x13", LINE_INFO());

			auto panda_msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAF1EF, TRUE, TRUE, std::string("\x13""\x10\x13""ninet", 8), LINE_INFO());
			check_panda_can_msg(panda_msg_recv[1], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x13""\x30\x00\x00", 4), LINE_INFO());

			//Send the rest of the message
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x21""een by", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x22""tes he", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x23""re", 8, 0, LINE_INFO());

			//Check J2534 constructed the whole message
			j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 0, 5 + 0x13, 5 + 0x13, "\x18\xda\xf1\xef\x13""nineteen bytes here", LINE_INFO());
		}

		//Check tx passes with filter. 29 bit. Good Filter. NoPadding. EXT address. Multi Frame.
		/*TEST_METHOD(J2534_ISO15765_SuccessTx_29b_Filter_NoPad_EXT_FFCF)
		{ //TODO when TX works with flow control}*/

		///////////////////// Tests checking things break or recover during send/receive. Ext 5 byte Addressing /////////////////////

		//Check rx FAILS when frame is dropped. 29 bit. Good Filter. NoPadding. STD address. Multi Frame.
		TEST_METHOD(J2534_ISO15765_FailRx_29b_Filter_NoPad_EXT_FFCF_DropFrame)
		{
			auto chanid = J2534_open_and_connect("", ISO15765, CAN_29BIT_ID, 500000, LINE_INFO());
			J2534_set_flowctrl_filter(chanid, CAN_29BIT_ID | ISO15765_ADDR_TYPE, 5, "\xff\xff\xff\xff\xff", "\x18\xda\xf1\xef\x13", "\x18\xda\xef\xf1\x13", LINE_INFO());
			auto p = getPanda(500);

			//Send first frame, then check we get a flow control frame
			auto panda_msg_recv = checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13\x10\x13""ninet", 8, 1, LINE_INFO());
			check_panda_can_msg(panda_msg_recv[0], 0, 0x18DAEFF1, TRUE, FALSE, std::string("\x13\x30\x00\x00", 4), LINE_INFO());

			//Check first frame is registered with J2534
			auto j2534_msg_recv = j2534_recv_loop(chanid, 1);
			check_J2534_can_msg(j2534_msg_recv[0], ISO15765, CAN_29BIT_ID | START_OF_MESSAGE | ISO15765_ADDR_TYPE, 0, 5, 0, "\x18\xda\xf1\xef\x13", LINE_INFO());

			//Send the rest of the message
			//Missing the 2nd frame "\x13""\x21""een by"
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x22""tes he", 8, 0, LINE_INFO());
			checked_panda_send(p, 0x18DAF1EF, TRUE, "\x13""\x23""re", 8, 0, LINE_INFO());

			//Check J2534 DOES NOT construct the incomplete message
			j2534_recv_loop(chanid, 0);
		}

		bool didopen = FALSE;
		unsigned long devid;

		unsigned long open_dev(const char* name, long assert_err = STATUS_NOERROR, TCHAR* failmsg = _T("Failed to open device.")) {
			unsigned int res = PassThruOpen((void*)name, &devid);
			if (res == STATUS_NOERROR) didopen = TRUE;
			return res;
		}

		unsigned long J2534_open_and_connect(const char* name, unsigned long ProtocolID, unsigned long Flags, unsigned long bps, const __LineInfo* pLineInfo = NULL) {
			unsigned long chanid;
			Assert::AreEqual<long>(STATUS_NOERROR, open_dev(name), _T("Failed to open device."), pLineInfo);
			Assert::AreEqual<long>(STATUS_NOERROR, PassThruConnect(devid, ProtocolID, Flags, bps, &chanid), _T("Failed to open channel."), pLineInfo);
			write_ioctl(chanid, LOOPBACK, FALSE, LINE_INFO()); // DISABLE J2534 ECHO/LOOPBACK
			return chanid;
		}

	};
}
