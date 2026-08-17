#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
    CUDA-specific ThreadPool

    Example usage

    // Construction
    KittensClub club(device_ids, NUM_DEVICES);

    // Dispatch work to all threads (no need to set device)
    club.execute([&](int dev_idx) {
        int dev;
        CUDACHECK(cudaGetDevice(&dev));
        if (dev != dev_idx) {
            fprintf(stderr, "Device mismatch: expected %d, got %d\n", dev_idx, dev);
            exit(1);
        }
    });
*/
class KittensClub {
public:
    __host__ inline KittensClub(const int *device_ids, const int num_devices);
    __host__ inline KittensClub(const int *device_ids, const cudaStream_t *streams, const int num_devices);
    __host__ inline ~KittensClub();

    // Dispatches `task` to all threads, and waits for all threads to finish (using cv)
    __host__ inline void execute(std::function<void(int, cudaStream_t)> task);

private:
    // Condition indicators
    bool stop;
    std::vector<bool> task_available;
    int n_task_done;

    // Threadpool
    std::vector<std::thread> workers;
    
    // Streams for each device
    std::vector<cudaStream_t> streams;
    
    // Main entry point for each thread
    __host__ inline void worker(int worker_id, int device_id);

    // Used to dispatch work to all threads
    std::function<void(int, cudaStream_t)> current_task;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};
    
__host__ inline KittensClub::KittensClub(const int *device_ids, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        task_available.push_back(false);
        streams.push_back(0); // Use default stream (null stream)
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}

__host__ inline KittensClub::KittensClub(const int *device_ids, const cudaStream_t *streams_in, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        task_available.push_back(false);
        streams.push_back(streams_in[dev_idx]);
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}
    
__host__ inline KittensClub::~KittensClub() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}
    
__host__ inline void KittensClub::execute(std::function<void(int, cudaStream_t)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

__host__ inline void KittensClub::worker(int worker_id, int device_id) {
    cudaSetDevice(device_id); // done once and never again! This saves a LOT of time
    while (true) {
        std::function<void(int, cudaStream_t)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, worker_id] { return stop || task_available[worker_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[worker_id] = false;
        }
        task(worker_id, streams[worker_id]);
        {
            std::lock_guard<std::mutex> lock(mutex); // adds about 10 microseconds overhead
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}
