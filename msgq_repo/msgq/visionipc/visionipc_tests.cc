#include <thread>
#include <chrono>

#include "catch2/catch.hpp"

#include "msgq/visionipc/visionipc_server.h"
#include "msgq/visionipc/visionipc_client.h"


static void zmq_sleep(int milliseconds=1000){
  if (messaging_use_zmq()){
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  }
}

TEST_CASE("Connecting"){
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, 1, 100, 100);
  server.start_listener();

  VisionIpcClient client = VisionIpcClient("camerad", VISION_STREAM_ROAD, false);
  REQUIRE(client.connect());

  REQUIRE(client.connected);
}

TEST_CASE("getAvailableStreams"){
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, 1, 100, 100);
  server.create_buffers(VISION_STREAM_WIDE_ROAD, 1, 100, 100);
  server.start_listener();
  auto available_streams = VisionIpcClient::getAvailableStreams("camerad");
  REQUIRE(available_streams.size() == 2);
  REQUIRE(available_streams.count(VISION_STREAM_ROAD) == 1);
  REQUIRE(available_streams.count(VISION_STREAM_WIDE_ROAD) == 1);
}

TEST_CASE("Check buffers"){
  size_t width = 100, height = 200, num_buffers = 5;
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, num_buffers, width, height);
  server.start_listener();

  VisionIpcClient client = VisionIpcClient("camerad", VISION_STREAM_ROAD, false);
  REQUIRE(client.connect());

  REQUIRE(client.buffers[0].width == width);
  REQUIRE(client.buffers[0].height == height);
  REQUIRE(client.buffers[0].len);
  REQUIRE(client.num_buffers == num_buffers);
}

TEST_CASE("Send single buffer"){
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, 1, 100, 100);
  server.start_listener();

  VisionIpcClient client = VisionIpcClient("camerad", VISION_STREAM_ROAD, false);
  REQUIRE(client.connect());
  zmq_sleep();

  VisionBuf * buf = server.get_buffer(VISION_STREAM_ROAD);
  REQUIRE(buf != nullptr);

  *((uint64_t*)buf->addr) = 1234;

  VisionIpcBufExtra extra = {0};
  extra.frame_id = 1337;
  buf->set_frame_id(extra.frame_id);

  server.send(buf, &extra);

  VisionIpcBufExtra extra_recv = {0};
  VisionBuf * recv_buf = client.recv(&extra_recv);
  REQUIRE(recv_buf != nullptr);
  REQUIRE(*(uint64_t*)recv_buf->addr == 1234);
  REQUIRE(extra_recv.frame_id == extra.frame_id);
  REQUIRE(recv_buf->get_frame_id() == extra.frame_id);
}


TEST_CASE("Test no conflate"){
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, 1, 100, 100);
  server.start_listener();

  VisionIpcClient client = VisionIpcClient("camerad", VISION_STREAM_ROAD, false);
  REQUIRE(client.connect());
  zmq_sleep();

  VisionBuf * buf = server.get_buffer(VISION_STREAM_ROAD);
  REQUIRE(buf != nullptr);

  VisionIpcBufExtra extra = {0};
  extra.frame_id = 1;
  server.send(buf, &extra);
  extra.frame_id = 2;
  server.send(buf, &extra);

  VisionIpcBufExtra extra_recv = {0};
  VisionBuf * recv_buf = client.recv(&extra_recv);
  REQUIRE(recv_buf != nullptr);
  REQUIRE(extra_recv.frame_id == 1);

  recv_buf = client.recv(&extra_recv);
  REQUIRE(recv_buf != nullptr);
  REQUIRE(extra_recv.frame_id == 2);
}

TEST_CASE("Test conflate"){
  VisionIpcServer server("camerad");
  server.create_buffers(VISION_STREAM_ROAD, 1, 100, 100);
  server.start_listener();

  VisionIpcClient client = VisionIpcClient("camerad", VISION_STREAM_ROAD, true);
  REQUIRE(client.connect());
  zmq_sleep();

  VisionBuf * buf = server.get_buffer(VISION_STREAM_ROAD);
  REQUIRE(buf != nullptr);

  VisionIpcBufExtra extra = {0};
  extra.frame_id = 1;
  server.send(buf, &extra);
  extra.frame_id = 2;
  server.send(buf, &extra);

  VisionIpcBufExtra extra_recv = {0};
  VisionBuf * recv_buf = client.recv(&extra_recv);
  REQUIRE(recv_buf != nullptr);
  REQUIRE(extra_recv.frame_id == 2);

  recv_buf = client.recv(&extra_recv);
  REQUIRE(recv_buf == nullptr);
}
