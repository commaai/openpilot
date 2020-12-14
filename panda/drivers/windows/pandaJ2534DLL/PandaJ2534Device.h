#pragma once
#include <memory>
#include <list>
#include <queue>
#include <set>
#include <chrono>
#include "J2534_v0404.h"
#include "panda_shared/panda.h"
#include "synchronize.h"
#include "Action.h"
#include "MessageTx.h"
#include "J2534Connection.h"

class J2534Connection;
class Action;
class MessageTx;

/**
Class representing a physical panda adapter. Instances are created by
PassThruOpen in the J2534 API. A Device can create one or more
J2534Connections.
*/
class PandaJ2534Device {
public:
	PandaJ2534Device(std::unique_ptr<panda::Panda> new_panda);

	~PandaJ2534Device();

	static std::shared_ptr<PandaJ2534Device> openByName(std::string sn);

	DWORD closeChannel(unsigned long ChannelID);
	DWORD addChannel(std::shared_ptr<J2534Connection>& conn, unsigned long* channel_id);

	std::unique_ptr<panda::Panda> panda;
	std::vector<std::shared_ptr<J2534Connection>> connections;

	//Place the Action in the task queue based on the Action's expiration time,
	//then signal the thread that processes actions.
	void insertActionIntoTaskList(std::shared_ptr<Action> action);

	void scheduleAction(std::shared_ptr<Action> msg, BOOL startdelayed=FALSE);

	void registerConnectionTx(std::shared_ptr<J2534Connection> conn);

	//Resume sending messages from the provided Connection's TX queue.
	void unstallConnectionTx(std::shared_ptr<J2534Connection> conn);

	//Cleans up several queues after a message completes, is canceled, or otherwise goes away.
	void removeConnectionTopAction(std::shared_ptr<J2534Connection> conn, std::shared_ptr<MessageTx> msg);

	//Messages that have been sent on the wire will be echoed by the panda when
	//transmission is complete. This tracks what is still waiting to hear an echo.
	std::queue<std::shared_ptr<MessageTx>> txMsgsAwaitingEcho;

	std::string kline_five_baud_init(uint8_t addr);
	std::string kline_wakeup_start_comm(std::string& start_comm);
	BOOL kline_send(std::string& data);

private:
	HANDLE thread_kill_event;

	HANDLE kline_recv_handle;
	static DWORD WINAPI _kline_recv_threadBootstrap(LPVOID This) {
		return ((PandaJ2534Device*)This)->kline_recv_thread();
	}
	DWORD kline_recv_thread();

	HANDLE can_recv_handle;
	static DWORD WINAPI _can_recv_threadBootstrap(LPVOID This) {
		return ((PandaJ2534Device*)This)->can_recv_thread();
	}
	DWORD can_recv_thread();

	HANDLE can_process_handle;
	static DWORD WINAPI _can_process_threadBootstrap(LPVOID This) {
		return ((PandaJ2534Device*)This)->can_process_thread();
	}
	DWORD can_process_thread();

	HANDLE flow_control_wakeup_event;
	HANDLE flow_control_thread_handle;
	static DWORD WINAPI _msg_tx_threadBootstrap(LPVOID This) {
		return ((PandaJ2534Device*)This)->msg_tx_thread();
	}
	DWORD msg_tx_thread();
	std::list<std::shared_ptr<Action>> task_queue;
	Mutex task_queue_mutex;

	std::queue<std::shared_ptr<J2534Connection>> ConnTxQueue;
	std::set<std::shared_ptr<J2534Connection>> ConnTxSet;
	Mutex connTXSet_mutex;
	BOOL txInProgress;

	Mutex kline_rx_mutex;
};
