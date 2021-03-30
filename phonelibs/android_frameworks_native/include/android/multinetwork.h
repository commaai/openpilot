/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_MULTINETWORK_H
#define ANDROID_MULTINETWORK_H

#include <netdb.h>
#include <stdlib.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * The corresponding C type for android.net.Network#getNetworkHandle() return
 * values.  The Java signed long value can be safely cast to a net_handle_t:
 *
 *     [C]    ((net_handle_t) java_long_network_handle)
 *     [C++]  static_cast<net_handle_t>(java_long_network_handle)
 *
 * as appropriate.
 */
typedef uint64_t net_handle_t;

/**
 * The value NETWORK_UNSPECIFIED indicates no specific network.
 *
 * For some functions (documented below), a previous binding may be cleared
 * by an invocation with NETWORK_UNSPECIFIED.
 *
 * Depending on the context it may indicate an error.  It is expressly
 * not used to indicate some notion of the "current default network".
 */
#define NETWORK_UNSPECIFIED  ((net_handle_t)0)


/**
 * All functions below that return an int return 0 on success or -1
 * on failure with an appropriate errno value set.
 */


/**
 * Set the network to be used by the given socket file descriptor.
 *
 * To clear a previous socket binding invoke with NETWORK_UNSPECIFIED.
 *
 * This is the equivalent of:
 *
 *     [ android.net.Network#bindSocket() ]
 *     https://developer.android.com/reference/android/net/Network.html#bindSocket(java.net.Socket)
 */
int android_setsocknetwork(net_handle_t network, int fd);


/**
 * Binds the current process to |network|.  All sockets created in the future
 * (and not explicitly bound via android_setsocknetwork()) will be bound to
 * |network|.  All host name resolutions will be limited to |network| as well.
 * Note that if the network identified by |network| ever disconnects, all
 * sockets created in this way will cease to work and all host name
 * resolutions will fail.  This is by design so an application doesn't
 * accidentally use sockets it thinks are still bound to a particular network.
 *
 * To clear a previous process binding invoke with NETWORK_UNSPECIFIED.
 *
 * This is the equivalent of:
 *
 *     [ android.net.ConnectivityManager#setProcessDefaultNetwork() ]
 *     https://developer.android.com/reference/android/net/ConnectivityManager.html#setProcessDefaultNetwork(android.net.Network)
 */
int android_setprocnetwork(net_handle_t network);


/**
 * Perform hostname resolution via the DNS servers associated with |network|.
 *
 * All arguments (apart from |network|) are used identically as those passed
 * to getaddrinfo(3).  Return and error values are identical to those of
 * getaddrinfo(3), and in particular gai_strerror(3) can be used as expected.
 * Similar to getaddrinfo(3):
 *     - |hints| may be NULL (in which case man page documented defaults apply)
 *     - either |node| or |service| may be NULL, but not both
 *     - |res| must not be NULL
 *
 * This is the equivalent of:
 *
 *     [ android.net.Network#getAllByName() ]
 *     https://developer.android.com/reference/android/net/Network.html#getAllByName(java.lang.String)
 */
int android_getaddrinfofornetwork(net_handle_t network,
        const char *node, const char *service,
        const struct addrinfo *hints, struct addrinfo **res);

__END_DECLS

#endif  // ANDROID_MULTINETWORK_H
