/*
 * tcpip_stack.c
 *
 *  Tcp stack implementation based on LWIP
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if defined(USING_OS_FREERTOS)
/* FreeRTOS kernel includes. */
#include "FreeRTOS.h"
#include "task.h"
#endif /* defined(USING_OS_FREERTOS) */

#include "osif.h"

/* lwIP core includes */
#include "lwip/opt.h"

#include "lwip/sys.h"
#include "lwip/timeouts.h"
#include "lwip/debug.h"
#include "lwip/stats.h"
#include "lwip/init.h"
#include "lwip/tcpip.h"
#include "lwip/netif.h"
#include "lwip/api.h"
#include "lwip/arch.h"

#include "lwip/tcp.h"
#include "lwip/udp.h"
#include "lwip/dns.h"
#include "lwip/dhcp.h"
#include "lwip/autoip.h"

/* lwIP netif includes */
#include "lwip/etharp.h"
#include "netif/ethernet.h"

/* include the port-dependent configuration */
#include "lwipcfg.h"
#include "enetif.h"

#include "tcpip_stack.h"

typedef struct{
	sys_sem_t* init_sem;
	tcpip_app_func app_func;
}tcpip_init_done_callback_params;

/* The ethernet interface */
struct netif netif;

static void mainLoopTask(void *pvParameters);

/* This function initializes all network interfaces
 * Implements enetif_init_Activity
 */
static void enetif_init(void)
{
  ip4_addr_t ipaddr, netmask, gw;
#define NETIF_ADDRS &ipaddr, &netmask, &gw,
  ip4_addr_set_zero(&gw);
  ip4_addr_set_zero(&ipaddr);
  ip4_addr_set_zero(&netmask);
#if (!LWIP_DHCP) && (!LWIP_AUTOIP)
  LWIP_PORT_INIT_GW(&gw);
  LWIP_PORT_INIT_IPADDR(&ipaddr);
  LWIP_PORT_INIT_NETMASK(&netmask);
#endif /* (!LWIP_DHCP) && (!LWIP_AUTOIP) */

  netif_set_default(netif_add(&netif, NETIF_ADDRS NULL, enet_ethernetif_init, tcpip_input));
  netif_set_up(&netif);
}

static void
post_tcpip_init(void* arg)
{
	tcpip_init_done_callback_params *params = (tcpip_init_done_callback_params *)arg;
	LWIP_ASSERT("init_sem != NULL", params->init_sem != NULL);
	LWIP_ASSERT("app_func != NULL", params->app_func != NULL);
	/* init network interfaces */
	enetif_init();

	params->app_func();

	sys_sem_signal(params->init_sem);
}

static void mainLoopTask(void* pvParameters)
{
	tcpip_stack_start_params *start_params = (tcpip_stack_start_params *)pvParameters;
	err_t err;
	sys_sem_t init_sem;
	tcpip_init_done_callback_params params;

	/* initialize lwIP stack, network interfaces and applications */
	err = sys_sem_new(&init_sem, 0);
	LWIP_ASSERT("failed to create init_sem", err == (err_t)ERR_OK);
	LWIP_UNUSED_ARG(err);
	params.init_sem = &init_sem;
	params.app_func = start_params->app_func;
	tcpip_init(post_tcpip_init, (void*)&params);
	/* we have to wait for initialization to finish before
	* calling update_adapter()! */
	(void)sys_sem_wait(&init_sem);
	sys_sem_free(&init_sem);
#if (LWIP_SOCKET || LWIP_NETCONN) && LWIP_NETCONN_SEM_PER_THREAD
	netconn_thread_init();
#endif

  while (1) {
	  sys_msleep(5000);
  }

#if (LWIP_SOCKET || LWIP_NETCONN) && LWIP_NETCONN_SEM_PER_THREAD
  netconn_thread_cleanup();
#endif
  enet_ethernetif_shutdown(&netif);
}

void tcpip_stack_start(tcpip_stack_start_params *params) {
#if defined(USING_OS_FREERTOS)
  BaseType_t ret = xTaskCreate(mainLoopTask, "tcpip_stack_loop", 256U, (void *)params, DEFAULT_THREAD_PRIO, NULL);
                                     /* Start the tasks and timer running. */
  LWIP_ASSERT("failed to create tcpip_stack_loop", ret == pdPASS);
#else
    mainLoopTask((void *)func);
#endif /* defined(USING_OS_FREERTOS) */
}
