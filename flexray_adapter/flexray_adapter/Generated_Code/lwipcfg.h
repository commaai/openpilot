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
#ifndef LWIP_LWIPCFG_H
#define LWIP_LWIPCFG_H

/****************************************************************************
 * FREERTOS
 ****************************************************************************/
/*************************************
 * Raw applications
 *************************************/

/*************************************
 * Netconn applications
 *************************************/
#define LWIP_DNS_APP                  0
#define LWIP_SHELL_APP                0

/*************************************
 * Socket applications
 *************************************/

/****************************************************************************
 * ENET RX/TX BUFFER DESCRIPTOR
 ****************************************************************************/
#define ENET_RXBD_NUM                 6
#define ENET_TXBD_NUM                 2

/****************************************************************************
 * MEDIA INDEPENDENT INTERFACE
 ****************************************************************************/
#define ENET_MIIMODE                  ENET_RMII_MODE
#define ENET_MIISPEED                 ENET_MII_SPEED_100M

/****************************************************************************
 * ADDITIONAL USERS SETTINGS
 ****************************************************************************/

#endif /* LWIP_LWIPCFG_H */
