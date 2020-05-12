#include "stdafx.h"
#include "Timer.h"


Timer::Timer()
{
	reset();
}

// gets the time elapsed from construction.
unsigned long long /*milliseconds*/ Timer::getTimePassed(){
	// get the new time
	auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(clock::now());

	// return the difference of the times
	return (end - start).count();
}

void Timer::reset() {
	start = std::chrono::time_point_cast<std::chrono::milliseconds>(clock::now());
}