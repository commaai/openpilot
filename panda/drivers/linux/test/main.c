#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>

const char *ifname = "can0";

static unsigned char payload[] = {0xAA, 0xAA, 0xAA, 0xAA, 0x07, 0x00, 0x00, 0x00};
int packet_len = 8;
int dir = 0;

void *write_thread( void *dat ){
  int nbytes;
  struct can_frame frame;
  int s = *((int*) dat);

  while(1){
    for(int i = 0; i < 1; i ++){
    if(packet_len % 2){
      frame.can_id  = 0x8AA | CAN_EFF_FLAG;
    }else{
      frame.can_id  = 0xAA;
    }

    frame.can_dlc = packet_len;
    memcpy(frame.data, payload, frame.can_dlc);

    nbytes = write(s, &frame, sizeof(struct can_frame));

    printf("Wrote %d bytes; addr: %lx; datlen: %d\n", nbytes, frame.can_id, frame.can_dlc);

    if(dir){
      packet_len++;
      if(packet_len >= 8)
	dir = 0;
    }else{
      packet_len--;
      if(packet_len <= 0)
	dir = 1;
    }
    }
    sleep(2);
  }
}


int main(void)
{
  pthread_t sndthread;
  int err, s, nbytes;
  struct sockaddr_can addr;
  struct can_frame frame;
  struct ifreq ifr;

  if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
    perror("Error while opening socket");
    return -1;
  }

  strcpy(ifr.ifr_name, ifname);
  ioctl(s, SIOCGIFINDEX, &ifr);

  addr.can_family  = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

  if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("Error in socket bind");
    return -2;
  }

  /////// Create Write Thread

  err = pthread_create( &sndthread, NULL, write_thread, (void*) &s);
  if(err){
    fprintf(stderr,"Error - pthread_create() return code: %d\n", err);
    exit(EXIT_FAILURE);
  }

  /////// Listen to socket
  while (1) {
    struct can_frame framein;

    // Read in a CAN frame
    int numBytes = read(s, &framein, CANFD_MTU);
    switch (numBytes) {
    case CAN_MTU:
      if(framein.can_id & 0x80000000)
	printf("Received %u byte payload; canid 0x%lx (EXT)\n",
	       framein.can_dlc, framein.can_id & 0x7FFFFFFF);
      else
	printf("Received %u byte payload; canid 0x%lx\n", framein.can_dlc, framein.can_id);
      break;
    case CANFD_MTU:
      // TODO: Should make an example for CAN FD
      break;
    case -1:
      // Check the signal value on interrupt
      //if (EINTR == errno)
      //  continue;

      // Delay before continuing
      sleep(1);
    default:
      continue;
    }
  }

  return 0;
}
