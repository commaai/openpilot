#include "stdafx.h"
#include "Loader4.h"
#include "pandaJ2534DLL/J2534_v0404.h"
#include "panda_shared/panda.h"
#include "Timer.h"
#include "ECUsim DLL\ECUsim.h"
#include "TestHelpers.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace pandaWCUsimTest
{

	TEST_CLASS(ECUsimTests)
	{
	public:

		TEST_METHOD(ECUsim_ISO15765_SingleFrameTx_29bStandardAddrPad500k)
		{
			ECUsim sim("", 500000);
			auto p = getPanda(500);

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x02\x01\x00", 3, panda::PANDA_CAN1);
			auto msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x02\x01\x00", 3), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x06\x41\x00\xff\xff\xff\xfe\x00", 8), LINE_INFO());
		}

		TEST_METHOD(ECUsim_ISO15765_SingleFrameTx_29bStandardAddrPad250k)
		{
			ECUsim sim("", 250000);
			auto p = getPanda(250);

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x02\x01\x00", 3, panda::PANDA_CAN1);
			auto msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x02\x01\x00", 3), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x06""\x41\x00""\xff\xff\xff\xfe""\x00", 8), LINE_INFO());
		}

		TEST_METHOD(ECUsim_ISO15765_SingleFrameTx_29bExtAddrPad500k)
		{
			ECUsim sim("", 500000, TRUE);
			auto p = getPanda(500);

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x13""\x02\x01\x00", 4, panda::PANDA_CAN1);
			auto msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x13""\x02\x01\x00", 4), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x13""\x06""\x41\x00""\xff\xff\xff\xfe", 8), LINE_INFO());
		}

		TEST_METHOD(ECUsim_ISO15765_MultiFrameTx_29bStandardAddrPad500k)
		{
			ECUsim sim("", 500000);
			auto p = getPanda(500);

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x02\x09\x02", 3, panda::PANDA_CAN1);
			auto msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x02\x09\x02", 3), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x10\x14""\x49\x02\x01""1D4", 8), LINE_INFO());

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x30\x00\x00", 3, panda::PANDA_CAN1);
			msg_recv = panda_recv_loop(p, 3);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x30\x0\x0", 3), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x21""GP00R55", 8), LINE_INFO());
			check_panda_can_msg(msg_recv[2], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x22""B123456", 8), LINE_INFO());
		}

		TEST_METHOD(ECUsim_ISO15765_MultiFrameTx_29bExtAddrPad500k)
		{
			ECUsim sim("", 500000, TRUE);
			auto p = getPanda(500);

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x13""\x02\x09\x02", 4, panda::PANDA_CAN1);
			auto msg_recv = panda_recv_loop(p, 2);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x13""\x02\x09\x02", 4), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x13""\x10\x14""\x49\x02\x01""1D", 8), LINE_INFO());

			p->can_send(0x18daeff1, TRUE, (const uint8_t*)"\x13""\x30\x00\x00", 4, panda::PANDA_CAN1);
			msg_recv = panda_recv_loop(p, 4);
			check_panda_can_msg(msg_recv[0], 0, 0x18daeff1, TRUE, TRUE, std::string("\x13""\x30\x0\x0", 4), LINE_INFO());
			check_panda_can_msg(msg_recv[1], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x13""\x21""4GP00R", 8), LINE_INFO());
			check_panda_can_msg(msg_recv[2], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x13""\x22""55B123", 8), LINE_INFO());
			check_panda_can_msg(msg_recv[3], 0, 0x18daf1ef, TRUE, FALSE, std::string("\x13""\x23""456", 5), LINE_INFO());
		}
	};

}