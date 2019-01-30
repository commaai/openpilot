#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> /* for strncpy */
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <time.h>

#include "jsmn.c"

struct input_event {
	struct timeval time;
	unsigned short type;
	unsigned short code;
	unsigned int value;
};

typedef struct DeviceState {
  // external
  char ipAddress[16];
  unsigned tx_throughput;
  int statePwr, stateVol;
  int tetherOn;
  int logOn;
  int buttonsOn;

  // internal
  int fdPwr, fdVol;
  unsigned long tx_bytes;
  time_t tx_time;
} DeviceState;
static DeviceState ds;

void ds_getIPAddress(char *buffer)
{
 int fd;
 struct ifreq ifr;

 fd = socket(AF_INET, SOCK_DGRAM, 0);
 ifr.ifr_addr.sa_family = AF_INET;
 strncpy(ifr.ifr_name, "wlan0", IFNAMSIZ-1);
 ioctl(fd, SIOCGIFADDR, &ifr);
 close(fd);

 /* display result */
 sprintf(buffer, "%s", inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr));
}

void toggleTether() {
  ds.tetherOn = 1-ds.tetherOn;
  char cmd[36];
  sprintf(cmd, "service call wifi 37 i32 0 i32 %d", ds.tetherOn);
  system(cmd);
}

void toggleLog() {
  ds.logOn = 1-ds.logOn;
}

void toggleButtons() {
  ds.buttonsOn = 1-ds.buttonsOn;
}

int ds_getTXBytes() {
    char str[64];
    int fd = open("/sys/class/net/wlan0/statistics/tx_bytes", O_RDONLY);
    int len = read(fd, str, 64);    
    close(fd);
    if(len>0) {
        char *ptr;
        unsigned long new_tx_bytes = strtoul(str, &ptr, 10);
        return new_tx_bytes;
    }
    return 0;
}


int ds_evt_init(char *evt) {
    int fdPwr = open(evt, 0);
    int flags = fcntl(fdPwr, F_GETFL, 0);
    fcntl(fdPwr, F_SETFL, flags | O_NONBLOCK);
    return fdPwr;
}

int ds_evt_read(fdPwr) {
    struct input_event data;
    ssize_t len = read(fdPwr, &data, sizeof(data));
    if(len==-1)
      return 0;
    //if(data.type!=0)
    //  printf("%d %d %d\n", data.type, data.code, data.value);
    return data.type!=0 && data.value==1?data.code:0;
}

void ds_init() {
  ds_getIPAddress(ds.ipAddress); // initialize IP address (refreshed only on touch)
  ds.tx_bytes = ds_getTXBytes();
  ds.tx_time = time(NULL);
  ds.tx_throughput = 0;
  ds.fdPwr = ds_evt_init("/dev/input/event0");
  ds.fdVol = ds_evt_init("/dev/input/event4");
  ds.tetherOn = 0;
  ds.logOn = 0;
}

void ds_update(isStopped, isAwake) {
  if(isStopped) {
    // check once a second
    time_t current_time = time(NULL);
    if (current_time!=ds.tx_time) {
      // update ip address
      ds_getIPAddress(ds.ipAddress);
      // update throughput
      unsigned long current_bytes = ds_getTXBytes();
      ds.tx_throughput = (current_bytes-ds.tx_bytes) / (current_time-ds.tx_time) / 1000; // KB/s
      ds.tx_bytes = current_bytes;
      ds.tx_time = current_time;
    }
  }
  ds.statePwr = ds_evt_read(ds.fdPwr);
  ds.stateVol = ds_evt_read(ds.fdVol);

  if(isAwake && ds.stateVol==114) 
    toggleTether();

  if(isAwake && ds.stateVol==115) 
    toggleLog();
}

char *_jsonstring(const char *js, jsmntok_t *token) {
  char *dst = malloc(token->end-token->start+1);
  strncpy(dst, js+token->start, token->end-token->start);
  return dst;
}

int _jsoneq(const char *json, jsmntok_t *tok, const char *s) {
	if (tok->type == JSMN_STRING && (int) strlen(s) == tok->end - tok->start &&
			strncmp(json + tok->start, s, tok->end - tok->start) == 0) {
		return 1;
	}
	return 0;
}


char *parseLogMessage(const char *js) {
  jsmntok_t tokens[64];
  jsmn_parser parser;
  jsmn_init(&parser);
  int r = jsmn_parse(&parser, js, strlen(js), tokens, 64);
  //for(i=1;i<r;i+=2) 
  //  printf("tok %.*s\n", tokens[i].end-tokens[i].start, js+tokens[i].start);
  if(r>0) {
    int i;
    // find exception first
    for(i=1;i<r;i+=2) 
      if(_jsoneq(js, &tokens[i], "exc_info")) 
        return _jsonstring(js, &tokens[i+1]);
    // return msg, unless it's "running"
    for(i=1;i<r;i+=2)
      if(_jsoneq(js, &tokens[i], "msg")) {
        char *substr = _jsonstring(js, &tokens[i+1]);
        if(strstr(substr, "running")==NULL)
          return substr;
        else {
          free(substr);
          return NULL;
        }
      }
  }
  return NULL;
}

unsigned long getCurrentDateIndex() {
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  return (tm.tm_year+1900)*10000000000+(tm.tm_mon)*100000000+tm.tm_mday*1000000+tm.tm_hour*10000+tm.tm_min*100+tm.tm_sec;           
}