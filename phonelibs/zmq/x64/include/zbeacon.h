/*  =========================================================================
    zbeacon - LAN discovery and presence
    
    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZBEACON_H_INCLUDED__
#define __ZBEACON_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  Create new zbeacon actor instance:
//
//      zactor_t *beacon = zactor_new (zbeacon, NULL);
//
//  Destroy zbeacon instance:
//
//      zactor_destroy (&beacon);
//
//  Enable verbose logging of commands and activity:
//
//      zstr_send (beacon, "VERBOSE");
//
//  Configure beacon to run on specified UDP port, and return the name of
//  the host, which can be used as endpoint for incoming connections. To
//  force the beacon to operate on a given interface, set the environment
//  variable ZSYS_INTERFACE, or call zsys_set_interface() before creating
//  the beacon. If the system does not support UDP broadcasts (lacking a
//  workable interface), returns an empty hostname:
//
//      //  Pictures: 's' = C string, 'i' = int
//      zsock_send (beacon, "si", "CONFIGURE", port_number);
//      char *hostname = zstr_recv (beacon);
//
//  Start broadcasting a beacon at a specified interval in msec. The beacon
//  data can be at most UDP_FRAME_MAX bytes; this constant is defined in
//  zsys.h to be 255:
//
//      //  Pictures: 'b' = byte * data + size_t size
//      zsock_send (beacon, "sbi", "PUBLISH", data, size, interval);
//
//  Stop broadcasting the beacon:
//
//      zstr_sendx (beacon, "SILENCE", NULL);
//
//  Start listening to beacons from peers. The filter is used to do a prefix
//  match on received beacons, to remove junk. Note that any received data
//  that is identical to our broadcast beacon_data is discarded in any case.
//  If the filter size is zero, we get all peer beacons:
//  
//      zsock_send (beacon, "sb", "SUBSCRIBE", filter_data, filter_size);
//
//  Stop listening to other peers
//
//      zstr_sendx (beacon, "UNSUBSCRIBE", NULL);
//
//  Receive next beacon from a peer. Received beacons are always a 2-frame
//  message containing the ipaddress of the sender, and then the binary
//  beacon data as published by the sender:
//
//      zmsg_t *msg = zmsg_recv (beacon);
//
//  This is the zbeacon constructor as a zactor_fn:
CZMQ_EXPORT void
    zbeacon (zsock_t *pipe, void *unused);

//  Self test of this class
CZMQ_EXPORT void
    zbeacon_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
