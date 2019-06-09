/*  =========================================================================
    zsockopt - get/set 0MQ socket options (deprecated)

            ****************************************************
            *   GENERATED SOURCE CODE, DO NOT EDIT!!           *
            *   TO CHANGE THIS, EDIT src/zsockopt.gsl          *
            *   AND RUN `gsl sockopts` in src/.                *
            ****************************************************

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZSOCKOPT_H_INCLUDED__
#define __ZSOCKOPT_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
#if (ZMQ_VERSION_MAJOR == 4)
//  Get socket options
CZMQ_EXPORT int zsocket_heartbeat_ivl (void *zocket);
CZMQ_EXPORT int zsocket_heartbeat_ttl (void *zocket);
CZMQ_EXPORT int zsocket_heartbeat_timeout (void *zocket);
CZMQ_EXPORT int zsocket_use_fd (void *zocket);
CZMQ_EXPORT int zsocket_tos (void *zocket);
CZMQ_EXPORT char * zsocket_zap_domain (void *zocket);
CZMQ_EXPORT int zsocket_mechanism (void *zocket);
CZMQ_EXPORT int zsocket_plain_server (void *zocket);
CZMQ_EXPORT char * zsocket_plain_username (void *zocket);
CZMQ_EXPORT char * zsocket_plain_password (void *zocket);
CZMQ_EXPORT int zsocket_curve_server (void *zocket);
CZMQ_EXPORT char * zsocket_curve_publickey (void *zocket);
CZMQ_EXPORT char * zsocket_curve_secretkey (void *zocket);
CZMQ_EXPORT char * zsocket_curve_serverkey (void *zocket);
CZMQ_EXPORT int zsocket_gssapi_server (void *zocket);
CZMQ_EXPORT int zsocket_gssapi_plaintext (void *zocket);
CZMQ_EXPORT char * zsocket_gssapi_principal (void *zocket);
CZMQ_EXPORT char * zsocket_gssapi_service_principal (void *zocket);
CZMQ_EXPORT int zsocket_ipv6 (void *zocket);
CZMQ_EXPORT int zsocket_immediate (void *zocket);
CZMQ_EXPORT int zsocket_ipv4only (void *zocket);
CZMQ_EXPORT int zsocket_type (void *zocket);
CZMQ_EXPORT int zsocket_sndhwm (void *zocket);
CZMQ_EXPORT int zsocket_rcvhwm (void *zocket);
CZMQ_EXPORT int zsocket_affinity (void *zocket);
CZMQ_EXPORT char * zsocket_identity (void *zocket);
CZMQ_EXPORT int zsocket_rate (void *zocket);
CZMQ_EXPORT int zsocket_recovery_ivl (void *zocket);
CZMQ_EXPORT int zsocket_sndbuf (void *zocket);
CZMQ_EXPORT int zsocket_rcvbuf (void *zocket);
CZMQ_EXPORT int zsocket_linger (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl_max (void *zocket);
CZMQ_EXPORT int zsocket_backlog (void *zocket);
CZMQ_EXPORT int zsocket_maxmsgsize (void *zocket);
CZMQ_EXPORT int zsocket_multicast_hops (void *zocket);
CZMQ_EXPORT int zsocket_rcvtimeo (void *zocket);
CZMQ_EXPORT int zsocket_sndtimeo (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_idle (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_cnt (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_intvl (void *zocket);
CZMQ_EXPORT char * zsocket_tcp_accept_filter (void *zocket);
CZMQ_EXPORT int zsocket_rcvmore (void *zocket);
CZMQ_EXPORT SOCKET zsocket_fd (void *zocket);
CZMQ_EXPORT int zsocket_events (void *zocket);
CZMQ_EXPORT char * zsocket_last_endpoint (void *zocket);

//  Set socket options
CZMQ_EXPORT void zsocket_set_heartbeat_ivl (void *zocket, int heartbeat_ivl);
CZMQ_EXPORT void zsocket_set_heartbeat_ttl (void *zocket, int heartbeat_ttl);
CZMQ_EXPORT void zsocket_set_heartbeat_timeout (void *zocket, int heartbeat_timeout);
CZMQ_EXPORT void zsocket_set_use_fd (void *zocket, int use_fd);
CZMQ_EXPORT void zsocket_set_tos (void *zocket, int tos);
CZMQ_EXPORT void zsocket_set_router_handover (void *zocket, int router_handover);
CZMQ_EXPORT void zsocket_set_router_mandatory (void *zocket, int router_mandatory);
CZMQ_EXPORT void zsocket_set_probe_router (void *zocket, int probe_router);
CZMQ_EXPORT void zsocket_set_req_relaxed (void *zocket, int req_relaxed);
CZMQ_EXPORT void zsocket_set_req_correlate (void *zocket, int req_correlate);
CZMQ_EXPORT void zsocket_set_conflate (void *zocket, int conflate);
CZMQ_EXPORT void zsocket_set_zap_domain (void *zocket, const char * zap_domain);
CZMQ_EXPORT void zsocket_set_plain_server (void *zocket, int plain_server);
CZMQ_EXPORT void zsocket_set_plain_username (void *zocket, const char * plain_username);
CZMQ_EXPORT void zsocket_set_plain_password (void *zocket, const char * plain_password);
CZMQ_EXPORT void zsocket_set_curve_server (void *zocket, int curve_server);
CZMQ_EXPORT void zsocket_set_curve_publickey (void *zocket, const char * curve_publickey);
CZMQ_EXPORT void zsocket_set_curve_publickey_bin (void *zocket, const byte *curve_publickey);
CZMQ_EXPORT void zsocket_set_curve_secretkey (void *zocket, const char * curve_secretkey);
CZMQ_EXPORT void zsocket_set_curve_secretkey_bin (void *zocket, const byte *curve_secretkey);
CZMQ_EXPORT void zsocket_set_curve_serverkey (void *zocket, const char * curve_serverkey);
CZMQ_EXPORT void zsocket_set_curve_serverkey_bin (void *zocket, const byte *curve_serverkey);
CZMQ_EXPORT void zsocket_set_gssapi_server (void *zocket, int gssapi_server);
CZMQ_EXPORT void zsocket_set_gssapi_plaintext (void *zocket, int gssapi_plaintext);
CZMQ_EXPORT void zsocket_set_gssapi_principal (void *zocket, const char * gssapi_principal);
CZMQ_EXPORT void zsocket_set_gssapi_service_principal (void *zocket, const char * gssapi_service_principal);
CZMQ_EXPORT void zsocket_set_ipv6 (void *zocket, int ipv6);
CZMQ_EXPORT void zsocket_set_immediate (void *zocket, int immediate);
CZMQ_EXPORT void zsocket_set_router_raw (void *zocket, int router_raw);
CZMQ_EXPORT void zsocket_set_ipv4only (void *zocket, int ipv4only);
CZMQ_EXPORT void zsocket_set_delay_attach_on_connect (void *zocket, int delay_attach_on_connect);
CZMQ_EXPORT void zsocket_set_sndhwm (void *zocket, int sndhwm);
CZMQ_EXPORT void zsocket_set_rcvhwm (void *zocket, int rcvhwm);
CZMQ_EXPORT void zsocket_set_affinity (void *zocket, int affinity);
CZMQ_EXPORT void zsocket_set_subscribe (void *zocket, const char * subscribe);
CZMQ_EXPORT void zsocket_set_unsubscribe (void *zocket, const char * unsubscribe);
CZMQ_EXPORT void zsocket_set_identity (void *zocket, const char * identity);
CZMQ_EXPORT void zsocket_set_rate (void *zocket, int rate);
CZMQ_EXPORT void zsocket_set_recovery_ivl (void *zocket, int recovery_ivl);
CZMQ_EXPORT void zsocket_set_sndbuf (void *zocket, int sndbuf);
CZMQ_EXPORT void zsocket_set_rcvbuf (void *zocket, int rcvbuf);
CZMQ_EXPORT void zsocket_set_linger (void *zocket, int linger);
CZMQ_EXPORT void zsocket_set_reconnect_ivl (void *zocket, int reconnect_ivl);
CZMQ_EXPORT void zsocket_set_reconnect_ivl_max (void *zocket, int reconnect_ivl_max);
CZMQ_EXPORT void zsocket_set_backlog (void *zocket, int backlog);
CZMQ_EXPORT void zsocket_set_maxmsgsize (void *zocket, int maxmsgsize);
CZMQ_EXPORT void zsocket_set_multicast_hops (void *zocket, int multicast_hops);
CZMQ_EXPORT void zsocket_set_rcvtimeo (void *zocket, int rcvtimeo);
CZMQ_EXPORT void zsocket_set_sndtimeo (void *zocket, int sndtimeo);
CZMQ_EXPORT void zsocket_set_xpub_verbose (void *zocket, int xpub_verbose);
CZMQ_EXPORT void zsocket_set_tcp_keepalive (void *zocket, int tcp_keepalive);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_idle (void *zocket, int tcp_keepalive_idle);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_cnt (void *zocket, int tcp_keepalive_cnt);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_intvl (void *zocket, int tcp_keepalive_intvl);
CZMQ_EXPORT void zsocket_set_tcp_accept_filter (void *zocket, const char * tcp_accept_filter);
#endif

#if (ZMQ_VERSION_MAJOR == 3)
//  Get socket options
CZMQ_EXPORT int zsocket_ipv4only (void *zocket);
CZMQ_EXPORT int zsocket_type (void *zocket);
CZMQ_EXPORT int zsocket_sndhwm (void *zocket);
CZMQ_EXPORT int zsocket_rcvhwm (void *zocket);
CZMQ_EXPORT int zsocket_affinity (void *zocket);
CZMQ_EXPORT char * zsocket_identity (void *zocket);
CZMQ_EXPORT int zsocket_rate (void *zocket);
CZMQ_EXPORT int zsocket_recovery_ivl (void *zocket);
CZMQ_EXPORT int zsocket_sndbuf (void *zocket);
CZMQ_EXPORT int zsocket_rcvbuf (void *zocket);
CZMQ_EXPORT int zsocket_linger (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl_max (void *zocket);
CZMQ_EXPORT int zsocket_backlog (void *zocket);
CZMQ_EXPORT int zsocket_maxmsgsize (void *zocket);
CZMQ_EXPORT int zsocket_multicast_hops (void *zocket);
CZMQ_EXPORT int zsocket_rcvtimeo (void *zocket);
CZMQ_EXPORT int zsocket_sndtimeo (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_idle (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_cnt (void *zocket);
CZMQ_EXPORT int zsocket_tcp_keepalive_intvl (void *zocket);
CZMQ_EXPORT char * zsocket_tcp_accept_filter (void *zocket);
CZMQ_EXPORT int zsocket_rcvmore (void *zocket);
CZMQ_EXPORT SOCKET zsocket_fd (void *zocket);
CZMQ_EXPORT int zsocket_events (void *zocket);
CZMQ_EXPORT char * zsocket_last_endpoint (void *zocket);

//  Set socket options
CZMQ_EXPORT void zsocket_set_router_raw (void *zocket, int router_raw);
CZMQ_EXPORT void zsocket_set_ipv4only (void *zocket, int ipv4only);
CZMQ_EXPORT void zsocket_set_delay_attach_on_connect (void *zocket, int delay_attach_on_connect);
CZMQ_EXPORT void zsocket_set_sndhwm (void *zocket, int sndhwm);
CZMQ_EXPORT void zsocket_set_rcvhwm (void *zocket, int rcvhwm);
CZMQ_EXPORT void zsocket_set_affinity (void *zocket, int affinity);
CZMQ_EXPORT void zsocket_set_subscribe (void *zocket, const char * subscribe);
CZMQ_EXPORT void zsocket_set_unsubscribe (void *zocket, const char * unsubscribe);
CZMQ_EXPORT void zsocket_set_identity (void *zocket, const char * identity);
CZMQ_EXPORT void zsocket_set_rate (void *zocket, int rate);
CZMQ_EXPORT void zsocket_set_recovery_ivl (void *zocket, int recovery_ivl);
CZMQ_EXPORT void zsocket_set_sndbuf (void *zocket, int sndbuf);
CZMQ_EXPORT void zsocket_set_rcvbuf (void *zocket, int rcvbuf);
CZMQ_EXPORT void zsocket_set_linger (void *zocket, int linger);
CZMQ_EXPORT void zsocket_set_reconnect_ivl (void *zocket, int reconnect_ivl);
CZMQ_EXPORT void zsocket_set_reconnect_ivl_max (void *zocket, int reconnect_ivl_max);
CZMQ_EXPORT void zsocket_set_backlog (void *zocket, int backlog);
CZMQ_EXPORT void zsocket_set_maxmsgsize (void *zocket, int maxmsgsize);
CZMQ_EXPORT void zsocket_set_multicast_hops (void *zocket, int multicast_hops);
CZMQ_EXPORT void zsocket_set_rcvtimeo (void *zocket, int rcvtimeo);
CZMQ_EXPORT void zsocket_set_sndtimeo (void *zocket, int sndtimeo);
CZMQ_EXPORT void zsocket_set_xpub_verbose (void *zocket, int xpub_verbose);
CZMQ_EXPORT void zsocket_set_tcp_keepalive (void *zocket, int tcp_keepalive);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_idle (void *zocket, int tcp_keepalive_idle);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_cnt (void *zocket, int tcp_keepalive_cnt);
CZMQ_EXPORT void zsocket_set_tcp_keepalive_intvl (void *zocket, int tcp_keepalive_intvl);
CZMQ_EXPORT void zsocket_set_tcp_accept_filter (void *zocket, const char * tcp_accept_filter);
#endif

#if (ZMQ_VERSION_MAJOR == 2)
//  Get socket options
CZMQ_EXPORT int zsocket_hwm (void *zocket);
CZMQ_EXPORT int zsocket_swap (void *zocket);
CZMQ_EXPORT int zsocket_affinity (void *zocket);
CZMQ_EXPORT char * zsocket_identity (void *zocket);
CZMQ_EXPORT int zsocket_rate (void *zocket);
CZMQ_EXPORT int zsocket_recovery_ivl (void *zocket);
CZMQ_EXPORT int zsocket_recovery_ivl_msec (void *zocket);
CZMQ_EXPORT int zsocket_mcast_loop (void *zocket);
#   if (ZMQ_VERSION_MINOR == 2)
CZMQ_EXPORT int zsocket_rcvtimeo (void *zocket);
#   endif
#   if (ZMQ_VERSION_MINOR == 2)
CZMQ_EXPORT int zsocket_sndtimeo (void *zocket);
#   endif
CZMQ_EXPORT int zsocket_sndbuf (void *zocket);
CZMQ_EXPORT int zsocket_rcvbuf (void *zocket);
CZMQ_EXPORT int zsocket_linger (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl (void *zocket);
CZMQ_EXPORT int zsocket_reconnect_ivl_max (void *zocket);
CZMQ_EXPORT int zsocket_backlog (void *zocket);
CZMQ_EXPORT int zsocket_type (void *zocket);
CZMQ_EXPORT int zsocket_rcvmore (void *zocket);
CZMQ_EXPORT SOCKET zsocket_fd (void *zocket);
CZMQ_EXPORT int zsocket_events (void *zocket);

//  Set socket options
CZMQ_EXPORT void zsocket_set_hwm (void *zocket, int hwm);
CZMQ_EXPORT void zsocket_set_swap (void *zocket, int swap);
CZMQ_EXPORT void zsocket_set_affinity (void *zocket, int affinity);
CZMQ_EXPORT void zsocket_set_identity (void *zocket, const char * identity);
CZMQ_EXPORT void zsocket_set_rate (void *zocket, int rate);
CZMQ_EXPORT void zsocket_set_recovery_ivl (void *zocket, int recovery_ivl);
CZMQ_EXPORT void zsocket_set_recovery_ivl_msec (void *zocket, int recovery_ivl_msec);
CZMQ_EXPORT void zsocket_set_mcast_loop (void *zocket, int mcast_loop);
#   if (ZMQ_VERSION_MINOR == 2)
CZMQ_EXPORT void zsocket_set_rcvtimeo (void *zocket, int rcvtimeo);
#   endif
#   if (ZMQ_VERSION_MINOR == 2)
CZMQ_EXPORT void zsocket_set_sndtimeo (void *zocket, int sndtimeo);
#   endif
CZMQ_EXPORT void zsocket_set_sndbuf (void *zocket, int sndbuf);
CZMQ_EXPORT void zsocket_set_rcvbuf (void *zocket, int rcvbuf);
CZMQ_EXPORT void zsocket_set_linger (void *zocket, int linger);
CZMQ_EXPORT void zsocket_set_reconnect_ivl (void *zocket, int reconnect_ivl);
CZMQ_EXPORT void zsocket_set_reconnect_ivl_max (void *zocket, int reconnect_ivl_max);
CZMQ_EXPORT void zsocket_set_backlog (void *zocket, int backlog);
CZMQ_EXPORT void zsocket_set_subscribe (void *zocket, const char * subscribe);
CZMQ_EXPORT void zsocket_set_unsubscribe (void *zocket, const char * unsubscribe);
#endif

//  Self test of this class
CZMQ_EXPORT void zsockopt_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
