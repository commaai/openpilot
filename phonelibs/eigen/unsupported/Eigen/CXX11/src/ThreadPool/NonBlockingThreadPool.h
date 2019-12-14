// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H


namespace Eigen {

template <typename Environment>
class NonBlockingThreadPoolTempl : public Eigen::ThreadPoolInterface {
 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  NonBlockingThreadPoolTempl(int num_threads, Environment env = Environment())
      : env_(env),
        threads_(num_threads),
        queues_(num_threads),
        coprimes_(num_threads),
        waiters_(num_threads),
        blocked_(0),
        spinning_(0),
        done_(false),
        ec_(waiters_) {
    waiters_.resize(num_threads);

    // Calculate coprimes of num_threads.
    // Coprimes are used for a random walk over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a walk starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    for (int i = 1; i <= num_threads; i++) {
      unsigned a = i;
      unsigned b = num_threads;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes_.push_back(i);
      }
    }
    for (int i = 0; i < num_threads; i++) {
      queues_.push_back(new Queue());
    }
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(env_.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  ~NonBlockingThreadPoolTempl() {
    done_ = true;
    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    ec_.Notify(true);

    // Join threads explicitly to avoid destruction order issues.
    for (size_t i = 0; i < threads_.size(); i++) delete threads_[i];
    for (size_t i = 0; i < threads_.size(); i++) delete queues_[i];
  }

  void Schedule(std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue* q = queues_[pt->thread_id];
      t = q->PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      Queue* q = queues_[Rand(&pt->rand) % queues_.size()];
      t = q->PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f)
      ec_.Notify(false);
    else
      env_.ExecuteTask(t);  // Push failed, execute directly.
  }

  int NumThreads() const final {
    return static_cast<int>(threads_.size());
  }

  int CurrentThreadId() const final {
    const PerThread* pt =
        const_cast<NonBlockingThreadPoolTempl*>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 private:
  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) { }
    NonBlockingThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;  // Random generator state.
    int thread_id;  // Worker thread index in pool.
  };

  Environment env_;
  MaxSizeVector<Thread*> threads_;
  MaxSizeVector<Queue*> queues_;
  MaxSizeVector<unsigned> coprimes_;
  MaxSizeVector<EventCount::Waiter> waiters_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  EventCount ec_;

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
    pt->thread_id = thread_id;
    Queue* q = queues_[thread_id];
    EventCount::Waiter* waiter = &waiters_[thread_id];
    for (;;) {
      Task t = q->PopFront();
      if (!t.f) {
        t = Steal();
        if (!t.f) {
          // Leave one thread spinning. This reduces latency.
          // TODO(dvyukov): 1000 iterations is based on fair dice roll, tune it.
          // Also, the time it takes to attempt to steal work 1000 times depends
          // on the size of the thread pool. However the speed at which the user
          // of the thread pool submit tasks is independent of the size of the
          // pool. Consider a time based limit instead.
          if (!spinning_ && !spinning_.exchange(true)) {
            for (int i = 0; i < 1000 && !t.f; i++) {
              t = Steal();
            }
            spinning_ = false;
          }
          if (!t.f) {
            if (!WaitForWork(waiter, &t)) {
              return;
            }
          }
        }
      }
      if (t.f) {
        env_.ExecuteTask(t);
      }
    }
  }

  // Steal tries to steal work from other worker threads in best-effort manner.
  Task Steal() {
    PerThread* pt = GetPerThread();
    const size_t size = queues_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = coprimes_[r % coprimes_.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      Task t = queues_[victim]->PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }

  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    eigen_assert(!t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait(waiter);
    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait(waiter);
      *t = queues_[victim]->PopBack();
      return true;
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    if (done_ && blocked_ == threads_.size()) {
      ec_.CancelWait(waiter);
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    const size_t size = queues_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = coprimes_[r % coprimes_.size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!queues_[victim]->Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static EIGEN_STRONG_INLINE PerThread* GetPerThread() {
    EIGEN_THREAD_LOCAL PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  static EIGEN_STRONG_INLINE unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
  }
};

typedef NonBlockingThreadPoolTempl<StlThreadEnvironment> NonBlockingThreadPool;

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
