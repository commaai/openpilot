#pragma once
#include <memory>

#include "J2534Frame.h"

class J2534Connection;

/**
An Action represents a unit of work that can be scheduled for execution at a later time.
Actions are not guaranteed to be run at their specified time, but a best effort is made.
An Action will never execute early, but can execute later depending on what is in the
queus.
Many different operations are based on this base class. Instead of making a thread,
consider if the work can be offloaded to the Task Queue.
*/
class Action
{
public:
	Action(
		std::weak_ptr<J2534Connection> connection,
		std::chrono::microseconds delay
	) : connection(connection), delay(delay) { };

	Action(
		std::weak_ptr<J2534Connection> connection
	) : connection(connection), delay(std::chrono::microseconds(0)) { };

	//The function called by the task runner when this action is to be invoked.
	virtual void execute() = 0;

	//Reschedule this Action for now().
	void scheduleImmediate() {
		expire = std::chrono::steady_clock::now();
	}

	//Reschedule this Action relative to its last expiration time.
	void scheduleDelay() {
		expire += this->delay;
	}

	//Reschedule this action {delay} after now().
	void scheduleImmediateDelay() {
		expire = std::chrono::steady_clock::now() + this->delay;
	}

	//Reschedule this Action based on a specific base time.
	void schedule(std::chrono::time_point<std::chrono::steady_clock> starttine, BOOL adddelayed) {
		this->expire = starttine;
		if (adddelayed)
			expire += this->delay;
	}

	std::weak_ptr<J2534Connection> connection;
	std::chrono::microseconds delay;
	//The timestamp at which point this Action is ready to be executed.
	std::chrono::time_point<std::chrono::steady_clock> expire;
};
