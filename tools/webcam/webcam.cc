#include <iostream>
#include <chrono>
#include <thread>
#include <cv.h>
#include <highgui.h>

#include "cereal/gen/cpp/log.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include <kj/io.h>

#include "messaging.hpp"

#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874

Context* context = Context::create();

void* health_function(void*) {
  PubSocket* sock = PubSocket::create(context, "health");

  std::chrono::seconds sec(1);
  std::chrono::steady_clock::time_point next_time = std::chrono::steady_clock::now() + sec;
  while (1) {
    // Build capnp message
    capnp::MallocMessageBuilder message;

    cereal::Event::Builder event_builder = message.initRoot<cereal::Event>();
    event_builder.setLogMonoTime(0); // TODO

    cereal::HealthData::Builder health_builder = event_builder.initHealth();
    health_builder.setIgnitionLine(true);
    health_builder.setHwType(cereal::HealthData::HwType::WHITE_PANDA);
    health_builder.setControlsAllowed(true);

    // Serialize message
    kj::VectorOutputStream stream;
    capnp::writeMessage(stream, message);
    kj::ArrayPtr<unsigned char> stream_array = stream.getArray();

    // Send serialized message through cereal messaging 
    sock->send(reinterpret_cast<char*>(stream_array.begin()), stream_array.size());
    
    std::this_thread::sleep_until(next_time);
    next_time += sec;
  }

  return 0;
}

int frame_function() {
  cv::VideoCapture cap(0);
  PubSocket* sock = PubSocket::create(context, "frame");

  if (!cap.isOpened()) {
    std::cout << "Error opening VideoCapture" << std::endl;
    return -1;
  }

  while (1) {
    cv::Mat frame_mat;

    cap >> frame_mat;
    if (frame_mat.empty())
      break;

    cv::resize(frame_mat, frame_mat, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));

    // Get image data size
    int frame_size = frame_mat.total() * frame_mat.elemSize();

    // Get raw image data
    unsigned char* frame_buffer = new unsigned char[frame_size];
    memcpy(frame_buffer, frame_mat.data, frame_size * sizeof(char));

    // Build capnp message
    capnp::MallocMessageBuilder message;

    cereal::Event::Builder event_builder = message.initRoot<cereal::Event>();
    event_builder.setLogMonoTime(0); // TODO

    cereal::FrameData::Builder frame_builder = event_builder.initFrame();
    frame_builder.setImage(kj::arrayPtr(frame_buffer, frame_size));

    // Serialize message
    kj::VectorOutputStream stream;
    capnp::writeMessage(stream, message);
    kj::ArrayPtr<unsigned char> stream_array = stream.getArray();

    // Send serialized message through cereal messaging 
    sock->send(reinterpret_cast<char*>(stream_array.begin()), stream_array.size());

    // Clean up memory
    delete[] frame_buffer;
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}

int main() {
  pthread_t thread_handle;
  pthread_create(&thread_handle, 0, health_function, 0);
  pthread_detach(thread_handle);

  frame_function();

  return 0;
}
