#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

//Inspired/directly copied from https://www.codeproject.com/Articles/12362/A-quot-synchronized-quot-statement-for-C-like-in-J
//Enables easier synchronization
class Mutex {
public:
	Mutex() {
		InitializeCriticalSectionAndSpinCount(&critSection, 0x00000400);
		//InitializeCriticalSection(&critSection);
	}

	~Mutex() {
		DeleteCriticalSection(&critSection);
	}

	void lock() {
		EnterCriticalSection(&critSection);
	}

	void unlock() {
		LeaveCriticalSection(&critSection);
	}

private:
	CRITICAL_SECTION critSection;
};

//Synchronization Controller Object
class Lock {
public:
	Lock(Mutex &m) : mutex(m), locked(TRUE) {
		m.lock();
	}

	~Lock() {
		mutex.unlock();
	}

	operator bool() const {
		return locked;
	}

	void setUnlock() {
		locked = FALSE;
	}

private:
	Mutex& mutex;
	bool locked;
};

//A useful shorthand for locking and unlocking a mutex over a scope.
//CAUTION, implemented with a for loop, so break/continue are consumed.
#define synchronized(M) for(Lock M##_lock = M; M##_lock; M##_lock.setUnlock())
