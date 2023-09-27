
#include "logging_definitions.h"

#define BANK_SIZE LOGGING_FLASH_SECTOR_SIZE
#define BANK_LOG_CAPACITY (BANK_SIZE / sizeof(log_t))
#define TOTAL_LOG_CAPACITY (BANK_LOG_CAPACITY * 2U)

#define LOGGING_MAX_LOGS_PER_MINUTE 10U

struct logging_state_t {
  uint16_t read_index;
  uint16_t write_index;
  uint16_t last_id;

  uint8_t rate_limit_counter;
  uint8_t rate_limit_log_count;
};
struct logging_state_t log_state = { 0 };
log_t *log_arr = (log_t *) LOGGING_FLASH_BASE_A;

uint16_t logging_next_id(uint16_t id) {
  return (id + 1U) % 0xFFFEU;
}

uint16_t logging_next_index(uint16_t index) {
  return (index + 1U) % TOTAL_LOG_CAPACITY;
}

void logging_erase_bank(uint8_t flash_sector) {
  print("erasing sector "); puth(flash_sector); print("\n");
  flash_unlock();
  if (!flash_erase_sector(flash_sector)) {
    print("failed to erase sector "); puth(flash_sector); print("\n");
  }
  flash_lock();
}

void logging_erase(void) {
  logging_erase_bank(LOGGING_FLASH_SECTOR_A);
  logging_erase_bank(LOGGING_FLASH_SECTOR_B);
  log_state.read_index = 0U;
  log_state.write_index = 0U;
}

void logging_find_read_index(uint16_t last_id) {
  // Figure out the read index by the last empty slot
  log_state.read_index = BANK_LOG_CAPACITY;
  for (uint16_t i = 0U; i < TOTAL_LOG_CAPACITY; i++) {
    if (log_arr[i].id == last_id) {
      log_state.read_index = logging_next_index(i);
    }
  }
}

void logging_init_read_index(void) {
  return logging_find_read_index(0xFFFFU);
}

void logging_init(void) {
  COMPILE_TIME_ASSERT(sizeof(log_t) == 64U);
  COMPILE_TIME_ASSERT((LOGGING_FLASH_BASE_A + BANK_SIZE) == LOGGING_FLASH_BASE_B);

  // Make sure all empty-ID logs are fully empty
  log_t empty_log;
  (void) memset(&empty_log, 0xFF, sizeof(log_t));

  for (uint16_t i = 0U; i < TOTAL_LOG_CAPACITY; i++) {
    if ((log_arr[i].id == 0xFFFFU) && (memcmp(&log_arr[i], &empty_log, sizeof(log_t)) != 0)) {
      logging_erase();
      break;
    }
  }

  logging_init_read_index();

  // At initialization, the read index should always be at the beginning of a bank
  // If not, clean slate
  if ((log_state.read_index != 0U) && (log_state.read_index != BANK_LOG_CAPACITY)) {
    logging_erase();
  }

  // Figure out the write index
  log_state.write_index = log_state.read_index;
  log_state.last_id = log_arr[log_state.write_index].id - 1U;
  for (uint16_t i = 0U; i < TOTAL_LOG_CAPACITY; i++) {
    bool done = false;
    if (log_arr[log_state.write_index].id == 0xFFFFU) {
      // Found the first empty slot after the read pointer
      done = true;
    } else if (log_arr[log_state.write_index].id != logging_next_id(log_state.last_id)) {
      // Discontinuity in the index, shouldn't happen!
      logging_erase();
      done = true;
    } else {
      log_state.last_id = log_arr[log_state.write_index].id;
      log_state.write_index = logging_next_index(log_state.write_index);
    }

    if (done) {
      break;
    }
  }

  // Reset rate limit
  log_state.rate_limit_counter = 0U;
  log_state.rate_limit_log_count = 0U;
}

// Call at 1Hz
void logging_tick(void) {
  flush_write_buffer();

  log_state.rate_limit_counter++;
  if (log_state.rate_limit_counter >= 60U) {
    log_state.rate_limit_counter = 0U;
    log_state.rate_limit_log_count = 0U;
  }
}

void log(const char* msg){
  if (log_state.rate_limit_log_count < LOGGING_MAX_LOGS_PER_MINUTE) {
    ENTER_CRITICAL();
    log_t new_log = {0};
    new_log.id = logging_next_id(log_state.last_id);
    log_state.last_id = new_log.id;
    new_log.uptime = uptime_cnt;
    if (current_board->has_rtc_battery) {
      new_log.timestamp = rtc_get_time();
    }

    uint8_t i = 0U;
    for (const char *in = msg; *in; in++) {
      new_log.msg[i] = *in;
      i++;
      if (i >= sizeof(new_log.msg)) {
        print("log message too long\n");
        break;
      }
    }

    // If we are at the beginning of a bank, erase it first and move the read pointer if needed
    switch (log_state.write_index) {
      case ((2U * BANK_LOG_CAPACITY) - 1U):
        logging_erase_bank(LOGGING_FLASH_SECTOR_A);
        if ((log_state.read_index < BANK_LOG_CAPACITY)) {
          log_state.read_index = BANK_LOG_CAPACITY;
        }
        break;
      case (BANK_LOG_CAPACITY - 1U):
        // beginning to write in bank B
        logging_erase_bank(LOGGING_FLASH_SECTOR_B);
        if ((log_state.read_index > BANK_LOG_CAPACITY)) {
          log_state.read_index = 0U;
        }
        break;
      default:
        break;
    }

    // Write!
    void *addr = &log_arr[log_state.write_index];
    uint32_t data[sizeof(log_t) / sizeof(uint32_t)];
    (void) memcpy(data, &new_log, sizeof(log_t));

    flash_unlock();
    for (uint8_t j = 0U; j < sizeof(log_t) / sizeof(uint32_t); j++) {
      flash_write_word(&((uint32_t *) addr)[j], data[j]);
    }
    flash_lock();

    // Update the write index
    log_state.write_index = logging_next_index(log_state.write_index);
    EXIT_CRITICAL();

    log_state.rate_limit_log_count++;
  } else {
    fault_occurred(FAULT_LOGGING_RATE_LIMIT);
  }
}

uint8_t logging_read(uint8_t *buffer) {
  uint8_t ret = 0U;
  if ((log_arr[log_state.read_index].id != 0xFFFFU) && (log_state.read_index != log_state.write_index)) {
    // Read the log
    (void) memcpy(buffer, &log_arr[log_state.read_index], sizeof(log_t));

    // Update the read index
    log_state.read_index = logging_next_index(log_state.read_index);

    ret = sizeof(log_t);
  }
  return ret;
}
