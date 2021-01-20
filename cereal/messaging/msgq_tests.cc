#include "catch2/catch.hpp"
#include "msgq.hpp"

TEST_CASE("ALIGN"){
  REQUIRE(ALIGN(0) == 0);
  REQUIRE(ALIGN(1) == 8);
  REQUIRE(ALIGN(7) == 8);
  REQUIRE(ALIGN(8) == 8);
  REQUIRE(ALIGN(99999) == 100000);
}

TEST_CASE("msgq_msg_init_size"){
  const size_t msg_size = 30;
  msgq_msg_t msg;

  msgq_msg_init_size(&msg, msg_size);
  REQUIRE(msg.size == msg_size);

  msgq_msg_close(&msg);
}

TEST_CASE("msgq_msg_init_data"){
  const size_t msg_size = 30;
  char * data = new char[msg_size];

  for (size_t i = 0; i < msg_size; i++){
    data[i] = i;
  }

  msgq_msg_t msg;
  msgq_msg_init_data(&msg, data, msg_size);

  REQUIRE(msg.size == msg_size);
  REQUIRE(memcmp(msg.data, data, msg_size) == 0);

  delete[] data;
  msgq_msg_close(&msg);
}


TEST_CASE("msgq_init_subscriber"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q;
  msgq_new_queue(&q, "test_queue", 1024);
  REQUIRE(*q.num_readers == 0);

  q.reader_id = 1;
  *q.read_valids[0] = false;
  *q.read_pointers[0] = ((uint64_t)1 << 32);

  *q.write_pointer = 255;

  msgq_init_subscriber(&q);
  REQUIRE(q.read_conflate == false);
  REQUIRE(*q.read_valids[0] == true);
  REQUIRE((*q.read_pointers[0] >> 32) == 0);
  REQUIRE((*q.read_pointers[0]  & 0xFFFFFFFF) == 255);
}

TEST_CASE("msgq_msg_send first message"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q;
  msgq_new_queue(&q, "test_queue", 1024);
  msgq_init_publisher(&q);

  REQUIRE(*q.write_pointer == 0);

  size_t msg_size = 128;

  SECTION("Aligned message size"){
  }
  SECTION("Unaligned message size"){
    msg_size--;
  }

  char * data = new char[msg_size];

  for (size_t i = 0; i < msg_size; i++){
    data[i] = i;
  }

  msgq_msg_t msg;
  msgq_msg_init_data(&msg, data, msg_size);


  msgq_msg_send(&msg, &q);
  REQUIRE(*(int64_t*)q.data == msg_size); // Check size tag
  REQUIRE(*q.write_pointer == 128 + sizeof(int64_t));
  REQUIRE(memcmp(q.data + sizeof(int64_t), data, msg_size) == 0);

  delete[] data;
  msgq_msg_close(&msg);
}

TEST_CASE("msgq_msg_send test wraparound"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q;
  msgq_new_queue(&q, "test_queue", 1024);
  msgq_init_publisher(&q);

  REQUIRE((*q.write_pointer & 0xFFFFFFFF) == 0);
  REQUIRE((*q.write_pointer >> 32) == 0);

  const size_t msg_size = 120;
  msgq_msg_t msg;
  msgq_msg_init_size(&msg, msg_size);

  for (int i = 0; i < 8; i++) {
    msgq_msg_send(&msg, &q);
  }
  // Check 8th message was written at the beginning
  REQUIRE((*q.write_pointer & 0xFFFFFFFF) == msg_size + sizeof(int64_t));

  // Check cycle count
  REQUIRE((*q.write_pointer >> 32) == 1);

  // Check wraparound tag
  char * tag_location = q.data;
  tag_location += 7 * (msg_size + sizeof(int64_t));
  REQUIRE(*(int64_t*)tag_location == -1);

  msgq_msg_close(&msg);
}

TEST_CASE("msgq_msg_recv test wraparound"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q_pub, q_sub;
  msgq_new_queue(&q_pub, "test_queue", 1024);
  msgq_new_queue(&q_sub, "test_queue", 1024);

  msgq_init_publisher(&q_pub);
  msgq_init_subscriber(&q_sub);

  REQUIRE((*q_pub.write_pointer >> 32) == 0);
  REQUIRE((*q_sub.read_pointers[0] >> 32) == 0);

  const size_t msg_size = 120;
  msgq_msg_t msg1;
  msgq_msg_init_size(&msg1, msg_size);


  SECTION("Check cycle counter after reset") {
    for (int i = 0; i < 8; i++) {
      msgq_msg_send(&msg1, &q_pub);
    }

    msgq_msg_t msg2;
    msgq_msg_recv(&msg2, &q_sub);
    REQUIRE(msg2.size == 0); // Reader had to reset
    msgq_msg_close(&msg2);

  }
  SECTION("Check cycle counter while keeping up with writer") {
    for (int i = 0; i < 8; i++) {
      msgq_msg_send(&msg1, &q_pub);

      msgq_msg_t msg2;
      msgq_msg_recv(&msg2, &q_sub);
      REQUIRE(msg2.size > 0);
      msgq_msg_close(&msg2);
    }
  }

  REQUIRE((*q_sub.read_pointers[0] >> 32) == 1);
  msgq_msg_close(&msg1);
}

TEST_CASE("msgq_msg_send test invalidation"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q_pub, q_sub;
  msgq_new_queue(&q_pub, "test_queue", 1024);
  msgq_new_queue(&q_sub, "test_queue", 1024);

  msgq_init_publisher(&q_pub);
  msgq_init_subscriber(&q_sub);
  *q_sub.write_pointer = (uint64_t)1 << 32;

  REQUIRE(*q_sub.read_valids[0] == true);

  SECTION("read pointer in tag"){
    *q_sub.read_pointers[0] = 0;
  }
  SECTION("read pointer in data section"){
    *q_sub.read_pointers[0] = 64;
  }
  SECTION("read pointer in wraparound section"){
    *q_pub.write_pointer = ((uint64_t)1 << 32) | 1000; // Writer is one cycle ahead
    *q_sub.read_pointers[0] = 1020;
  }

  msgq_msg_t msg;
  msgq_msg_init_size(&msg, 128);
  msgq_msg_send(&msg, &q_pub);

  REQUIRE(*q_sub.read_valids[0] == false);

  msgq_msg_close(&msg);
}

TEST_CASE("msgq_init_subscriber init 2 subscribers"){
  remove("/dev/shm/test_queue");
  msgq_queue_t q1, q2;
  msgq_new_queue(&q1, "test_queue", 1024);
  msgq_new_queue(&q2, "test_queue", 1024);

  *q1.num_readers = 0;

  REQUIRE(*q1.num_readers == 0);
  REQUIRE(*q2.num_readers == 0);

  msgq_init_subscriber(&q1);
  REQUIRE(*q1.num_readers == 1);
  REQUIRE(*q2.num_readers == 1);
  REQUIRE(q1.reader_id == 0);

  msgq_init_subscriber(&q2);
  REQUIRE(*q1.num_readers == 2);
  REQUIRE(*q2.num_readers == 2);
  REQUIRE(q2.reader_id == 1);
}


TEST_CASE("Write 1 msg, read 1 msg", "[integration]"){
  remove("/dev/shm/test_queue");
  const size_t msg_size = 128;
  msgq_queue_t writer, reader;

  msgq_new_queue(&writer, "test_queue", 1024);
  msgq_new_queue(&reader, "test_queue", 1024);

  msgq_init_publisher(&writer);
  msgq_init_subscriber(&reader);

  // Build 128 byte message
  msgq_msg_t outgoing_msg;
  msgq_msg_init_size(&outgoing_msg, msg_size);

  for (size_t i = 0; i < msg_size; i++){
    outgoing_msg.data[i] = i;
  }

  REQUIRE(msgq_msg_send(&outgoing_msg, &writer) == msg_size);

  msgq_msg_t incoming_msg1;
  REQUIRE(msgq_msg_recv(&incoming_msg1, &reader) == msg_size);
  REQUIRE(memcmp(incoming_msg1.data, outgoing_msg.data, msg_size) == 0);

  // Verify that there are no more messages
  msgq_msg_t incoming_msg2;
  REQUIRE(msgq_msg_recv(&incoming_msg2, &reader) == 0);

  msgq_msg_close(&outgoing_msg);
  msgq_msg_close(&incoming_msg1);
  msgq_msg_close(&incoming_msg2);
}

TEST_CASE("Write 2 msg, read 2 msg - conflate = false", "[integration]"){
  remove("/dev/shm/test_queue");
  const size_t msg_size = 128;
  msgq_queue_t writer, reader;

  msgq_new_queue(&writer, "test_queue", 1024);
  msgq_new_queue(&reader, "test_queue", 1024);

  msgq_init_publisher(&writer);
  msgq_init_subscriber(&reader);

  // Build 128 byte message
  msgq_msg_t outgoing_msg;
  msgq_msg_init_size(&outgoing_msg, msg_size);

  for (size_t i = 0; i < msg_size; i++){
    outgoing_msg.data[i] = i;
  }

  REQUIRE(msgq_msg_send(&outgoing_msg, &writer) == msg_size);
  REQUIRE(msgq_msg_send(&outgoing_msg, &writer) == msg_size);

  msgq_msg_t incoming_msg1;
  REQUIRE(msgq_msg_recv(&incoming_msg1, &reader) == msg_size);
  REQUIRE(memcmp(incoming_msg1.data, outgoing_msg.data, msg_size) == 0);

  msgq_msg_t incoming_msg2;
  REQUIRE(msgq_msg_recv(&incoming_msg2, &reader) == msg_size);
  REQUIRE(memcmp(incoming_msg2.data, outgoing_msg.data, msg_size) == 0);

  msgq_msg_close(&outgoing_msg);
  msgq_msg_close(&incoming_msg1);
  msgq_msg_close(&incoming_msg2);
}

TEST_CASE("Write 2 msg, read 2 msg - conflate = true", "[integration]"){
  remove("/dev/shm/test_queue");
  const size_t msg_size = 128;
  msgq_queue_t writer, reader;

  msgq_new_queue(&writer, "test_queue", 1024);
  msgq_new_queue(&reader, "test_queue", 1024);

  msgq_init_publisher(&writer);
  msgq_init_subscriber(&reader);
  reader.read_conflate = true;

  // Build 128 byte message
  msgq_msg_t outgoing_msg;
  msgq_msg_init_size(&outgoing_msg, msg_size);

  for (size_t i = 0; i < msg_size; i++){
    outgoing_msg.data[i] = i;
  }

  REQUIRE(msgq_msg_send(&outgoing_msg, &writer) == msg_size);
  REQUIRE(msgq_msg_send(&outgoing_msg, &writer) == msg_size);

  msgq_msg_t incoming_msg1;
  REQUIRE(msgq_msg_recv(&incoming_msg1, &reader) == msg_size);
  REQUIRE(memcmp(incoming_msg1.data, outgoing_msg.data, msg_size) == 0);

  // Verify that there are no more messages
  msgq_msg_t incoming_msg2;
  REQUIRE(msgq_msg_recv(&incoming_msg2, &reader) == 0);

  msgq_msg_close(&outgoing_msg);
  msgq_msg_close(&incoming_msg1);
  msgq_msg_close(&incoming_msg2);
}

TEST_CASE("1 publisher, 1 slow subscriber", "[integration]"){
  remove("/dev/shm/test_queue");
  msgq_queue_t writer, reader;

  msgq_new_queue(&writer, "test_queue", 1024);
  msgq_new_queue(&reader, "test_queue", 1024);

  msgq_init_publisher(&writer);
  msgq_init_subscriber(&reader);

  int n_received = 0;
  int n_skipped = 0;

  for (uint64_t i = 0; i < 1e5; i++) {
    msgq_msg_t outgoing_msg;
    msgq_msg_init_data(&outgoing_msg, (char*)&i, sizeof(uint64_t));
    msgq_msg_send(&outgoing_msg, &writer);
    msgq_msg_close(&outgoing_msg);

    if (i % 10 == 0){
      msgq_msg_t msg1;
      msgq_msg_recv(&msg1, &reader);

      if (msg1.size == 0){
        n_skipped++;
      } else {
        n_received++;
      }
      msgq_msg_close(&msg1);
    }
  }

  // TODO: verify these numbers by hand
  REQUIRE(n_received == 8572);
  REQUIRE(n_skipped == 1428);
}

TEST_CASE("1 publisher, 2 subscribers", "[integration]"){
  remove("/dev/shm/test_queue");
  msgq_queue_t writer, reader1, reader2;

  msgq_new_queue(&writer, "test_queue", 1024);
  msgq_new_queue(&reader1, "test_queue", 1024);
  msgq_new_queue(&reader2, "test_queue", 1024);

  msgq_init_publisher(&writer);
  msgq_init_subscriber(&reader1);
  msgq_init_subscriber(&reader2);

  for (uint64_t i = 0; i < 1024 * 3; i++) {
    msgq_msg_t outgoing_msg;
    msgq_msg_init_data(&outgoing_msg, (char*)&i, sizeof(uint64_t));
    msgq_msg_send(&outgoing_msg, &writer);

    msgq_msg_t msg1, msg2;
    msgq_msg_recv(&msg1, &reader1);
    msgq_msg_recv(&msg2, &reader2);

    REQUIRE(msg1.size == sizeof(uint64_t));
    REQUIRE(msg2.size == sizeof(uint64_t));
    REQUIRE(*(uint64_t*)msg1.data == i);
    REQUIRE(*(uint64_t*)msg2.data == i);

    msgq_msg_close(&outgoing_msg);
    msgq_msg_close(&msg1);
    msgq_msg_close(&msg2);
  }
}
