/*
 * Copyright (c) 2001-2003 Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 * This file is part of the lwIP TCP/IP stack.
 *
 * Author: Adam Dunkels <adam@sics.se>
 *
 */
#ifndef LWIP_LWIPOPTS_H
#define LWIP_LWIPOPTS_H

/****************************************************************************
 * General: OS, Memory, etc.
 ****************************************************************************/
#define LWIP_NETIF_HOSTNAME           1
#define LWIP_NETIF_HOSTNAME_TEXT      ("tcpipBoard1")

#define TCPIP_MBOX_SIZE               10

#define DEFAULT_UDP_RECVMBOX_SIZE     10
#define DEFAULT_TCP_RECVMBOX_SIZE     10
#define DEFAULT_RAW_RECVMBOX_SIZE     10
#define DEFAULT_ACCEPTMBOX_SIZE       10

#define LWIP_NETIF_TX_SINGLE_PBUF     1
#define LWIP_SUPPORT_CUSTOM_PBUF      1
#define MEMP_USE_CUSTOM_POOLS         0
#define MEM_USE_POOLS                 0

/*! @brief OS options of the module */
#define NO_SYS                        0                           /* 0: using OS with multi-thread --> can use raw, netconn or socket API*/

/*! @brief RAW API options of the module */
#define LWIP_RAW                      0

/*! @brief NETCONN API options of the module */
#define LWIP_NETCONN                  1
#define LWIP_NETCONN_SEM_PER_THREAD   (LWIP_NETCONN || LWIP_SOCKET)

/*! @brief SOCKET API options of the module */
#define LWIP_SOCKET                   0
#define LWIP_SOCKET_SET_ERRNO         (LWIP_SOCKET)


/** SYS_LIGHTWEIGHT_PROT
 * define SYS_LIGHTWEIGHT_PROT in lwipopts.h if you want inter-task protection
 * for certain critical regions during buffer allocation, deallocation and memory
 * allocation and deallocation.
 */
#define SYS_LIGHTWEIGHT_PROT          (NO_SYS==0)

#define LWIP_SNMP                     LWIP_UDP
#define MIB2_STATS                    LWIP_SNMP
#define LWIP_DNS                      LWIP_UDP
#define LWIP_MDNS_RESPONDER           LWIP_UDP
#define LWIP_NUM_NETIF_CLIENT_DATA    (LWIP_MDNS_RESPONDER)
#define LWIP_HAVE_LOOPIF              0
#define LWIP_NETIF_LOOPBACK           0
#define TCP_LISTEN_BACKLOG            1
#define LWIP_COMPAT_SOCKETS           1
#define LWIP_SO_RCVTIMEO              1
#define LWIP_SO_RCVBUF                1
#define LWIP_TCPIP_CORE_LOCKING       1
#define LWIP_NETIF_LINK_CALLBACK      0
#define LWIP_NETIF_STATUS_CALLBACK    0

/*! @brief Memory options of the module */
/* MEM_ALIGNMENT: should be set to the alignment of the CPU for which
   lwIP is compiled. 4 byte alignment -> define MEM_ALIGNMENT to 4, 2
   byte alignment -> define MEM_ALIGNMENT to 2. */
#define MEM_ALIGNMENT                 4

/* MEM_SIZE: the size of the heap memory. If the application will send
a lot of data that needs to be copied, this should be set high. */
#define MEM_SIZE                      20480

/* MEMP_NUM_PBUF: the number of memp struct pbufs. If the application
   sends a lot of data out of ROM (or other static memory), this
   should be set high. */
#define MEMP_NUM_PBUF                 8
/* MEMP_NUM_RAW_PCB: the number of UDP protocol control blocks. One
   per active RAW "connection". */
#define MEMP_NUM_RAW_PCB              4
/* MEMP_NUM_UDP_PCB: the number of UDP protocol control blocks. One
   per active UDP "connection". */
#define MEMP_NUM_UDP_PCB              8
/* MEMP_NUM_TCP_PCB: the number of simultaneously active TCP
   connections. */
#define MEMP_NUM_TCP_PCB              8
/* MEMP_NUM_TCP_PCB_LISTEN: the number of listening TCP
   connections. */
#define MEMP_NUM_TCP_PCB_LISTEN       8
/* MEMP_NUM_TCP_SEG: the number of simultaneously queued TCP
   segments. */
#define MEMP_NUM_TCP_SEG              16
/* MEMP_NUM_SYS_TIMEOUT: the number of simultaneously active
   timeouts. */
#define MEMP_NUM_SYS_TIMEOUT          15

/* The following four are used only with the sequential API and can be
   set to 0 if the application only will use the raw API. */
/* MEMP_NUM_NETBUF: the number of struct netbufs. */
#define MEMP_NUM_NETBUF               16
/* MEMP_NUM_NETCONN: the number of struct netconns. */
#define MEMP_NUM_NETCONN              32
/* MEMP_NUM_TCPIP_MSG_*: the number of struct tcpip_msg, which is used
   for sequential API communication and incoming packets. Used in
   src/api/tcpip.c. */
#define MEMP_NUM_TCPIP_MSG_API        8
#define MEMP_NUM_TCPIP_MSG_INPKT      8

#define PBUF_POOL_SIZE                0                           /* The number of buffers in the pbuf pool. */

/****************************************************************************
 * Data Link Layer
 ****************************************************************************/
/*! @brief MAC address of the module */
#ifndef LWIP_MAC_ADDR_BASE
#define LWIP_MAC_ADDR_BASE            0x22,0x33,0x44,0x55,0x66,0x77
#endif /* ifndef LWIP_MAC_ADDR_BASE */

/*! @brief (LWIP_ETHERNET == 1) Enable ETHERNET support even though ARP might be disabled */
#define LWIP_ETHERNET                 1

/*! @brief ARP options */
#define LWIP_ARP                      1                           /* 1: Enable ARP, 0: Disable ARP */
#define ARP_TABLE_SIZE                10                          /* ARP Table size */
#define ARP_QUEUEING                  0                           /* ARP Queueing */

/****************************************************************************
 * IP Network Layer
 ****************************************************************************/
/*! @brief IPv4 options of the module */
#define LWIP_IPV4                     1                           /* 1: Enable IPv4, 0: Disable IPv4 */
/*! @brief Dynamic IP address settings of the module */
#define LWIP_DHCP                     0                           /* 1: Enable DHCP, 0: Disable DHCP */
#define DHCP_DOES_ARP_CHECK           (LWIP_DHCP)                 /* 1: Do an ARP check on the offered address (recommended) */
#define LWIP_AUTOIP                   0                           /* 1: Enable AUTOIP, 0: Disable AUTOIP */
#define LWIP_DHCP_AUTOIP_COOP         (LWIP_DHCP && LWIP_AUTOIP)  /* AUTOIP and DHCP Cooperation */
#define LWIP_DHCP_AUTOIP_COOP_TRIES   3                           /* Number of DHCP Discover tries before switching to
                                                                     AUTOIP. Set it to a low value will get a IP quickly.
                                                                     But you should prepare to handle changing
                                                                     IP address when DHCP overrides AUTOIP */

/*! @brief Static IP address settings of the module */
#define LWIP_PORT_INIT_IPADDR(addr)   IP4_ADDR((addr), 192,168,5,10)
#define LWIP_PORT_INIT_NETMASK(addr)  IP4_ADDR((addr), 255,255,255,0)
#define LWIP_PORT_INIT_GW(addr)       IP4_ADDR((addr), 192,168,5,1)

/*! @brief IP forward ability of the module */
#define IP_FORWARD                    0                           /* 1: Forward IP packets across network interfaces,
                                                                     0: Run lwIP on a device with only 1 network interface */

/*! @brief IP reassembly and segmentation. These are orthogonal even if they both deal with IP fragments */
#define IP_REASSEMBLY                 1
#define IP_REASS_MAX_PBUFS            10
#define MEMP_NUM_REASSDATA            10
#define IP_FRAG                       1

/*! @brief ICMP options of the module */
#define LWIP_ICMP                     1
#define ICMP_TTL                      255                         /* ICMP Time to live value */

/*! @brief IGMP options of the module */
#define LWIP_IGMP                     0

/*! @brief IPv6 options of the module */
#define LWIP_IPV6                     0                           /* 1: Enable IPv6, 0: Disable IPv6 */

/****************************************************************************
 * Transport Layer
 ****************************************************************************/
/*! @brief TCP options of the module */
#define LWIP_TCP                      1                           /* 1: Enable TCP, 0: Disable TCP */
#define TCP_TTL                       255                         /* TCP Time to live value */
#define TCP_QUEUE_OOSEQ               0                           /* Controls if TCP should queue segments that arrive out of order.
                                                                     Defines to 0 if your device is low on memory. */
#define TCP_MSS                       1460                        /* TCP Maximum segment size */
#define TCP_SND_BUF                   2920                        /* TCP sender buffer space (bytes). */

/* TCP sender buffer space (pbufs). This must be at least = 2 *
   TCP_SND_BUF/TCP_MSS for things to work. */
#define TCP_SND_QUEUELEN              (4 * TCP_SND_BUF/TCP_MSS)

/* TCP writable space (bytes). This must be less than or equal
   to TCP_SND_BUF. It is the amount of space which must be
   available in the tcp snd_buf for select to return writable */
#define TCP_SNDLOWAT                  (TCP_SND_BUF/2)

#define TCP_WND                       65535                       /* TCP receive window. */
#define TCP_MAXRTX                    12                          /* Maximum number of retransmissions of data segments. */
#define TCP_SYNMAXRTX                 4                           /* Maximum number of retransmissions of SYN segments. */

/*! @brief UDP options of the module */
#define LWIP_UDP                      0                           /* 1: Enable UDP, 0: Disable UDP */

/****************************************************************************
 * Debug, Statistics, PPP, Checksum
 ****************************************************************************/
/*! @brief Debugging options of the module */
#ifdef LWIP_DEBUG
#define LWIP_DBG_MIN_LEVEL            0
#define PPP_DEBUG                     LWIP_DBG_OFF
#define MEM_DEBUG                     LWIP_DBG_OFF
#define MEMP_DEBUG                    LWIP_DBG_OFF
#define PBUF_DEBUG                    LWIP_DBG_OFF
#define API_LIB_DEBUG                 LWIP_DBG_OFF
#define API_MSG_DEBUG                 LWIP_DBG_OFF
#define TCPIP_DEBUG                   LWIP_DBG_OFF
#define NETIF_DEBUG                   LWIP_DBG_OFF
#define SOCKETS_DEBUG                 LWIP_DBG_OFF
#define DNS_DEBUG                     LWIP_DBG_OFF
#define AUTOIP_DEBUG                  LWIP_DBG_OFF
#define DHCP_DEBUG                    LWIP_DBG_OFF
#define IP_DEBUG                      LWIP_DBG_OFF
#define IP_REASS_DEBUG                LWIP_DBG_OFF
#define ICMP_DEBUG                    LWIP_DBG_OFF
#define IGMP_DEBUG                    LWIP_DBG_OFF
#define UDP_DEBUG                     LWIP_DBG_OFF
#define TCP_DEBUG                     LWIP_DBG_OFF
#define TCP_INPUT_DEBUG               LWIP_DBG_OFF
#define TCP_OUTPUT_DEBUG              LWIP_DBG_OFF
#define TCP_RTO_DEBUG                 LWIP_DBG_OFF
#define TCP_CWND_DEBUG                LWIP_DBG_OFF
#define TCP_WND_DEBUG                 LWIP_DBG_OFF
#define TCP_FR_DEBUG                  LWIP_DBG_OFF
#define TCP_QLEN_DEBUG                LWIP_DBG_OFF
#define TCP_RST_DEBUG                 LWIP_DBG_OFF
#endif

#define LWIP_DBG_TYPES_ON             (LWIP_DBG_ON|LWIP_DBG_TRACE|LWIP_DBG_STATE|LWIP_DBG_FRESH|LWIP_DBG_HALT)

/*! @brief Statistics options of the module */
#define LWIP_STATS                    1                           /* 1: collect statistics; 0: not collect statistics */
#define LWIP_STATS_DISPLAY            0

#if LWIP_STATS
#define LINK_STATS                    1
#define IP_STATS                      1
#define ICMP_STATS                    1
#define IGMP_STATS                    1
#define IPFRAG_STATS                  1
#define UDP_STATS                     1
#define TCP_STATS                     1
#define MEM_STATS                     1
#define MEMP_STATS                    1
#define PBUF_STATS                    1
#define SYS_STATS                     1
#endif /* LWIP_STATS */

/*! @brief PPP options of the module */
#define PPP_SUPPORT                   0

/*! @brief Checksum options of the module */
#define LWIP_CHECKSUM_CTRL_PER_NETIF  0
#define CHECKSUM_GEN_IP               0
#define CHECKSUM_GEN_UDP              0
#define CHECKSUM_GEN_TCP              0
#define CHECKSUM_GEN_ICMP             0
#define CHECKSUM_GEN_ICMP6            LWIP_IPV6
#define CHECKSUM_CHECK_IP             0
#define CHECKSUM_CHECK_UDP            0
#define CHECKSUM_CHECK_TCP            0
#define CHECKSUM_CHECK_ICMP           0
#define CHECKSUM_CHECK_ICMP6          LWIP_IPV6
#define LWIP_CHECKSUM_ON_COPY         0

#endif /* LWIP_LWIPOPTS_H */
