#include <stdio.h>
#include <SDL.h>

#include <cudaGL.h>
#include <assert.h>
#include <GL/glut.h>

#include "simple_decoder.h"

#include "cereal/messaging/messaging.h"

//#define SLOW


SDL_Window *window;
SDL_Surface *window_surface;
SDL_Renderer* renderer;
SDL_Texture* texture;

uint64_t st;
bool parsed;


CUgraphicsResource res[2];

int main() {
  SimpleDecoder decoder;

  SDL_Init(SDL_INIT_VIDEO);
  SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
  window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1928, 1208, 0);
  assert(window != NULL);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  assert(renderer != NULL);
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_NV12, SDL_TEXTUREACCESS_STREAMING, 2048, 1216);
  assert(texture != NULL);


  // get opengl texture number
  SDL_GL_BindTexture(texture, NULL, NULL);
  GLint whichID;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &whichID);
  SDL_GL_UnbindTexture(texture);

  // link the texture to CUDA
  CHECK(cuGraphicsGLRegisterImage(&res[0], whichID, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
  CHECK(cuGraphicsGLRegisterImage(&res[1], whichID+1, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

  setenv("ZMQ", "1", 1);
  Context * c = Context::create();
  SubSocket *sub_sock = SubSocket::create(c, "roadEncodeData", "192.168.3.188");
  bool seen_header = false;
  while (1) {
    Message *msg = sub_sock->receive();
    capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
    auto event = cmsg.getRoot<cereal::Event>();
    auto edata = event.getRoadEncodeData();
    if (!seen_header) {
      auto header = edata.getHeader();
      if (header.size() > 0) {
        printf("got header\n");
        seen_header = true;
        decoder.decode(header.begin(), header.size(), true);
      } else {
        continue;
      }
    }
    st = nanos_since_boot();

    auto data = edata.getData();
    //printf("got data %d\n", data.size());
    CUdeviceptr dpSrcFrame = decoder.decode(data.begin(), data.size());

    uint64_t ct1 = nanos_since_boot();

    void *pixels;
    int pitch;
    SDL_LockTexture(texture, NULL, &pixels, &pitch);
    cuMemcpy((CUdeviceptr)pixels, dpSrcFrame, 2048*1216*3/2);
    SDL_UnlockTexture(texture);

    uint64_t ct2 = nanos_since_boot();

    SDL_Rect screen{0, 0, 1928, 1208};
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, &screen, NULL);
    SDL_RenderPresent(renderer);

    decoder.free_frame();

    uint64_t et = nanos_since_boot()-st;
    printf("pc latency: %.2f ms (copy %.2f ms)\n", et/1e6, (ct2-ct1)/1e6);

    delete msg;
  }
}
