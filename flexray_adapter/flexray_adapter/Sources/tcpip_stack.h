/*
 * tcpip_stack.h
 */

#ifndef TCPIP_STACK_H_
#define TCPIP_STACK_H_

typedef void (*tcpip_app_func)();
typedef struct{
	tcpip_app_func app_func;
}tcpip_stack_start_params;

void tcpip_stack_start(tcpip_stack_start_params *params);

#endif /* LWIP_CLIENT_H_ */
