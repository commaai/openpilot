#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/loggerd/loggerd.h"

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

int get_synced_frame_id(LoggerdState *s, CameraType cam_type, int start_frame_id) {
  int frame_id = start_frame_id;
  while (!sync_encoders(s, cam_type, frame_id)) {
    ++frame_id;
    usleep(0);
  }
  return frame_id;
};

TEST_CASE("sync_encoders") {
  const int max_waiting = GENERATE(1, 2, 3);

  for (int test_cnt = 0; test_cnt < 10; ++test_cnt) {
    LoggerdState s{.max_waiting = max_waiting};
    std::vector<int> start_frames;
    std::vector<std::future<int>> futures;

    for (int i = 0; i < max_waiting; ++i) {
      int start_frame_id = random_int(0, 20);
      start_frames.push_back(start_frame_id);
      futures.emplace_back(std::async(std::launch::async, get_synced_frame_id, &s, (CameraType)i, start_frame_id));
    }

    // get results
    int synced_frame_id = 0;
    for (int i = 0; i < max_waiting; ++i) {
      if (i == 0) {
        synced_frame_id = futures[i].get();
        // require synced_frame_id equal start_frame_id if max_waiting == 1
        if (max_waiting == 1) {
          REQUIRE(synced_frame_id == start_frames[0]);
        }
      } else {
        REQUIRE(futures[i].get() == synced_frame_id);
      }
    }
  }
}

const int MAX_SEGMENT_CNT = 100;

std::pair<int, uint32_t> encoder_thread(LoggerdState *s) {
  int cur_seg = 0;
  uint32_t frame_id = s->start_frame_id;

  while (cur_seg < MAX_SEGMENT_CNT) {
    ++frame_id;
    if (trigger_rotate_if_needed(s, cur_seg, frame_id)) {
      cur_seg = s->rotate_segment;
    }
    util::sleep_for(0);
  }

  return {cur_seg, frame_id};
}

TEST_CASE("trigger_rotate") {
  const int encoders = GENERATE(1, 2, 3);
  const int start_frame_id = random_int(0, 20);

  LoggerdState s{
      .max_waiting = encoders,
      .start_frame_id = start_frame_id,
  };

  std::vector<std::future<std::pair<int, uint32_t>>> futures;
  for (int i = 0; i < encoders; ++i) {
    futures.emplace_back(std::async(std::launch::async, encoder_thread, &s));
  }

  while (s.rotate_segment < MAX_SEGMENT_CNT) {
    rotate_if_needed(&s);
    util::sleep_for(10);
  }

  for (auto &f : futures) {
    auto [encoder_seg, frame_id] = f.get();
    REQUIRE(encoder_seg == MAX_SEGMENT_CNT);
    REQUIRE(frame_id == start_frame_id + encoder_seg * (SEGMENT_LENGTH * MAIN_FPS));
  }
}

TEST_CASE("clear_locks") {
  std::vector<std::string> dirs;
  std::string current_segment;
  for (int i = 0; i < 10; ++i) {
    std::string &path = dirs.emplace_back(LOG_ROOT + "/" + std::to_string(i));
    REQUIRE(util::create_directories(path, 0775));
    std::ofstream{path + "/.lock"};
    REQUIRE(util::file_exists(path + "/.lock"));
    if (i == 0) current_segment = path;
  }
  {
    std::future<bool> clear_locks_future = std::async(std::launch::async, [=] {
      clear_locks(LOG_ROOT);
      return true;
    });
  }
  for (const auto &dir : dirs) {
    std::string lock_file = dir + "/.lock";
    if (dir == current_segment) {
      REQUIRE(util::file_exists(lock_file));
      ::unlink(lock_file.c_str());
    } else {
      REQUIRE(util::file_exists(lock_file) == false);
    }
    rmdir(dir.c_str());
  }
}
