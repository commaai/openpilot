/*  =========================================================================
    zsys - system-level methods

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZSYS_H_INCLUDED__
#define __ZSYS_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
#define UDP_FRAME_MAX   255         //  Max size of UDP frame

//  Callback for interrupt signal handler
typedef void (zsys_handler_fn) (int signal_value);

//  Initialize CZMQ zsys layer; this happens automatically when you create
//  a socket or an actor; however this call lets you force initialization
//  earlier, so e.g. logging is properly set-up before you start working.
//  Not threadsafe, so call only from main thread. Safe to call multiple
//  times. Returns global CZMQ context.
CZMQ_EXPORT void *
    zsys_init (void);

//  Optionally shut down the CZMQ zsys layer; this normally happens automatically
//  when the process exits; however this call lets you force a shutdown
//  earlier, avoiding any potential problems with atexit() ordering, especially
//  with Windows dlls.
CZMQ_EXPORT void
    zsys_shutdown (void);

//  Get a new ZMQ socket, automagically creating a ZMQ context if this is
//  the first time. Caller is responsible for destroying the ZMQ socket
//  before process exits, to avoid a ZMQ deadlock. Note: you should not use
//  this method in CZMQ apps, use zsock_new() instead.
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT void *
    zsys_socket (int type, const char *filename, size_t line_nbr);

//  Destroy/close a ZMQ socket. You should call this for every socket you
//  create using zsys_socket().
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT int
    zsys_close (void *handle, const char *filename, size_t line_nbr);

//  Return ZMQ socket name for socket type
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT char *
    zsys_sockname (int socktype);
    
//  Create a pipe, which consists of two PAIR sockets connected over inproc.
//  The pipe is configured to use the zsys_pipehwm setting. Returns the
//  frontend socket successful, NULL if failed.
CZMQ_EXPORT zsock_t *
    zsys_create_pipe (zsock_t **backend_p);
    
//  Set interrupt handler; this saves the default handlers so that a
//  zsys_handler_reset () can restore them. If you call this multiple times
//  then the last handler will take affect. If handler_fn is NULL, disables
//  default SIGINT/SIGTERM handling in CZMQ.
CZMQ_EXPORT void
    zsys_handler_set (zsys_handler_fn *handler_fn);

//  Reset interrupt handler, call this at exit if needed
CZMQ_EXPORT void
    zsys_handler_reset (void);

//  Set default interrupt handler, so Ctrl-C or SIGTERM will set
//  zsys_interrupted. Idempotent; safe to call multiple times.
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT void
    zsys_catch_interrupts (void);

//  Return 1 if file exists, else zero
CZMQ_EXPORT bool
    zsys_file_exists (const char *filename);

//  Return size of file, or -1 if not found
CZMQ_EXPORT ssize_t
    zsys_file_size (const char *filename);

//  Return file modification time. Returns 0 if the file does not exist.
CZMQ_EXPORT time_t
    zsys_file_modified (const char *filename);

//  Return file mode; provides at least support for the POSIX S_ISREG(m)
//  and S_ISDIR(m) macros and the S_IRUSR and S_IWUSR bits, on all boxes.
//  Returns a mode_t cast to int, or -1 in case of error.
CZMQ_EXPORT int
    zsys_file_mode (const char *filename);

//  Delete file. Does not complain if the file is absent
CZMQ_EXPORT int
    zsys_file_delete (const char *filename);

//  Check if file is 'stable'
CZMQ_EXPORT bool
    zsys_file_stable (const char *filename);

//  Create a file path if it doesn't exist. The file path is treated as a 
//  printf format.
CZMQ_EXPORT int
    zsys_dir_create (const char *pathname, ...);

//  Remove a file path if empty; the pathname is treated as printf format.
CZMQ_EXPORT int
    zsys_dir_delete (const char *pathname, ...);

//  Move to a specified working directory. Returns 0 if OK, -1 if this failed.
CZMQ_EXPORT int
    zsys_dir_change (const char *pathname);

//  Set private file creation mode; all files created from here will be
//  readable/writable by the owner only.
CZMQ_EXPORT void
    zsys_file_mode_private (void);

//  Reset default file creation mode; all files created from here will use
//  process file mode defaults.
CZMQ_EXPORT void
    zsys_file_mode_default (void);

//  Return the CZMQ version for run-time API detection; returns version
//  number into provided fields, providing reference isn't null in each case.
CZMQ_EXPORT void
    zsys_version (int *major, int *minor, int *patch);

//  Format a string using printf formatting, returning a freshly allocated
//  buffer. If there was insufficient memory, returns NULL. Free the returned
//  string using zstr_free().
CZMQ_EXPORT char *
    zsys_sprintf (const char *format, ...);

//  Format a string with a va_list argument, returning a freshly allocated
//  buffer. If there was insufficient memory, returns NULL. Free the returned
//  string using zstr_free().
CZMQ_EXPORT char *
    zsys_vprintf (const char *format, va_list argptr);

//  Create UDP beacon socket; if the routable option is true, uses
//  multicast (not yet implemented), else uses broadcast. This method
//  and related ones might _eventually_ be moved to a zudp class.
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT SOCKET
    zsys_udp_new (bool routable);

//  Close a UDP socket
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT int
    zsys_udp_close (SOCKET handle);

//  Send zframe to UDP socket, return -1 if sending failed due to
//  interface having disappeared (happens easily with WiFi)
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT int
    zsys_udp_send (SOCKET udpsock, zframe_t *frame, inaddr_t *address, int addrlen);

//  Receive zframe from UDP socket, and set address of peer that sent it
//  The peername must be a char [INET_ADDRSTRLEN] array.
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT zframe_t *
    zsys_udp_recv (SOCKET udpsock, char *peername, int peerlen);

//  Handle an I/O error on some socket operation; will report and die on
//  fatal errors, and continue silently on "try again" errors.
//  *** This is for CZMQ internal use only and may change arbitrarily ***
CZMQ_EXPORT void
    zsys_socket_error (const char *reason);

//  Return current host name, for use in public tcp:// endpoints. Caller gets
//  a freshly allocated string, should free it using zstr_free(). If the host
//  name is not resolvable, returns NULL.
CZMQ_EXPORT char *
    zsys_hostname (void);

//  Move the current process into the background. The precise effect depends
//  on the operating system. On POSIX boxes, moves to a specified working
//  directory (if specified), closes all file handles, reopens stdin, stdout,
//  and stderr to the null device, and sets the process to ignore SIGHUP. On
//  Windows, does nothing. Returns 0 if OK, -1 if there was an error.
CZMQ_EXPORT int
    zsys_daemonize (const char *workdir);

//  Drop the process ID into the lockfile, with exclusive lock, and switch
//  the process to the specified group and/or user. Any of the arguments
//  may be null, indicating a no-op. Returns 0 on success, -1 on failure.
//  Note if you combine this with zsys_daemonize, run after, not before
//  that method, or the lockfile will hold the wrong process ID.
CZMQ_EXPORT int
    zsys_run_as (const char *lockfile, const char *group, const char *user);

//  Returns true if the underlying libzmq supports CURVE security.
//  Uses a heuristic probe according to the version of libzmq being used.
CZMQ_EXPORT bool
    zsys_has_curve (void);

//  Configure the number of I/O threads that ZeroMQ will use. A good
//  rule of thumb is one thread per gigabit of traffic in or out. The
//  default is 1, sufficient for most applications. If the environment
//  variable ZSYS_IO_THREADS is defined, that provides the default.
//  Note that this method is valid only before any socket is created.
CZMQ_EXPORT void
    zsys_set_io_threads (size_t io_threads);

//  Configure the number of sockets that ZeroMQ will allow. The default
//  is 1024. The actual limit depends on the system, and you can query it
//  by using zsys_socket_limit (). A value of zero means "maximum".
//  Note that this method is valid only before any socket is created.
CZMQ_EXPORT void
    zsys_set_max_sockets (size_t max_sockets);

//  Return maximum number of ZeroMQ sockets that the system will support.
CZMQ_EXPORT size_t
    zsys_socket_limit (void);

//  Configure the maximum allowed size of a message sent.
//  The default is INT_MAX.
CZMQ_EXPORT void
    zsys_set_max_msgsz (int max_msgsz);

//  Return maximum message size.
CZMQ_EXPORT int
    zsys_max_msgsz (void);

//  Configure the default linger timeout in msecs for new zsock instances.
//  You can also set this separately on each zsock_t instance. The default
//  linger time is zero, i.e. any pending messages will be dropped. If the
//  environment variable ZSYS_LINGER is defined, that provides the default.
//  Note that process exit will typically be delayed by the linger time.
CZMQ_EXPORT void
    zsys_set_linger (size_t linger);

//  Configure the default outgoing pipe limit (HWM) for new zsock instances.
//  You can also set this separately on each zsock_t instance. The default
//  HWM is 1,000, on all versions of ZeroMQ. If the environment variable
//  ZSYS_SNDHWM is defined, that provides the default. Note that a value of
//  zero means no limit, i.e. infinite memory consumption.
CZMQ_EXPORT void
    zsys_set_sndhwm (size_t sndhwm);

//  Configure the default incoming pipe limit (HWM) for new zsock instances.
//  You can also set this separately on each zsock_t instance. The default
//  HWM is 1,000, on all versions of ZeroMQ. If the environment variable
//  ZSYS_RCVHWM is defined, that provides the default. Note that a value of
//  zero means no limit, i.e. infinite memory consumption.
CZMQ_EXPORT void
    zsys_set_rcvhwm (size_t rcvhwm);

//  Configure the default HWM for zactor internal pipes; this is set on both
//  ends of the pipe, for outgoing messages only (sndhwm). The default HWM is
//  1,000, on all versions of ZeroMQ. If the environment var ZSYS_ACTORHWM is
//  defined, that provides the default. Note that a value of zero means no
//  limit, i.e. infinite memory consumption.
CZMQ_EXPORT void
    zsys_set_pipehwm (size_t pipehwm);

//  Return the HWM for zactor internal pipes.
CZMQ_EXPORT size_t
    zsys_pipehwm (void);

//  Configure use of IPv6 for new zsock instances. By default sockets accept
//  and make only IPv4 connections. When you enable IPv6, sockets will accept
//  and connect to both IPv4 and IPv6 peers. You can override the setting on
//  each zsock_t instance. The default is IPv4 only (ipv6 set to 0). If the
//  environment variable ZSYS_IPV6 is defined (as 1 or 0), this provides the
//  default. Note: has no effect on ZMQ v2.
CZMQ_EXPORT void
    zsys_set_ipv6 (int ipv6);

//  Return use of IPv6 for zsock instances.
CZMQ_EXPORT int
    zsys_ipv6 (void);

//  Set network interface name to use for broadcasts, particularly zbeacon.
//  This lets the interface be configured for test environments where required.
//  For example, on Mac OS X, zbeacon cannot bind to 255.255.255.255 which is
//  the default when there is no specified interface. If the environment
//  variable ZSYS_INTERFACE is set, use that as the default interface name.
//  Setting the interface to "*" means "use all available interfaces".
CZMQ_EXPORT void
    zsys_set_interface (const char *value);

//  Return network interface to use for broadcasts, or "" if none was set.
CZMQ_EXPORT const char *
    zsys_interface (void);

//  Set IPv6 address to use zbeacon socket, particularly for receiving zbeacon.
//  This needs to be set IPv6 is enabled as IPv6 can have multiple addresses
//  on a given interface. If the environment variable ZSYS_IPV6_ADDRESS is set,
//  use that as the default IPv6 address.
CZMQ_EXPORT void
    zsys_set_ipv6_address (const char *value);

//  Return IPv6 address to use for zbeacon reception, or "" if none was set.
CZMQ_EXPORT const char *
    zsys_ipv6_address (void);

//  Set IPv6 milticast address to use for sending zbeacon messages. This needs
//  to be set if IPv6 is enabled. If the environment variable
//  ZSYS_IPV6_MCAST_ADDRESS is set, use that as the default IPv6 multicast
//  address.
CZMQ_EXPORT void
    zsys_set_ipv6_mcast_address (const char *value);

//  Return IPv6 multicast address to use for sending zbeacon, or "" if none was
//  set.
CZMQ_EXPORT const char *
    zsys_ipv6_mcast_address (void);

//  Configure the automatic use of pre-allocated FDs when creating new sockets.
//  If 0 (default), nothing will happen. Else, when a new socket is bound, the
//  system API will be used to check if an existing pre-allocated FD with a
//  matching port (if TCP) or path (if IPC) exists, and if it does it will be
//  set via the ZMQ_USE_FD socket option so that the library will use it
//  instead of creating a new socket.
CZMQ_EXPORT void
    zsys_set_auto_use_fd (int auto_use_fd);

//  Return use of automatic pre-allocated FDs for zsock instances.
CZMQ_EXPORT int
    zsys_auto_use_fd (void);

//  Set log identity, which is a string that prefixes all log messages sent
//  by this process. The log identity defaults to the environment variable
//  ZSYS_LOGIDENT, if that is set.
CZMQ_EXPORT void
    zsys_set_logident (const char *value);

//  Set stream to receive log traffic. By default, log traffic is sent to
//  stdout. If you set the stream to NULL, no stream will receive the log
//  traffic (it may still be sent to the system facility).
CZMQ_EXPORT void
    zsys_set_logstream (FILE *stream);
    
//  Sends log output to a PUB socket bound to the specified endpoint. To
//  collect such log output, create a SUB socket, subscribe to the traffic
//  you care about, and connect to the endpoint. Log traffic is sent as a
//  single string frame, in the same format as when sent to stdout. The
//  log system supports a single sender; multiple calls to this method will
//  bind the same sender to multiple endpoints. To disable the sender, call
//  this method with a null argument.
CZMQ_EXPORT void
    zsys_set_logsender (const char *endpoint);

//  Enable or disable logging to the system facility (syslog on POSIX boxes,
//  event log on Windows). By default this is disabled.
CZMQ_EXPORT void
    zsys_set_logsystem (bool logsystem);
    
//  Log error condition - highest priority
CZMQ_EXPORT void
    zsys_error (const char *format, ...);

//  Log warning condition - high priority
CZMQ_EXPORT void
    zsys_warning (const char *format, ...);
    
//  Log normal, but significant, condition - normal priority
CZMQ_EXPORT void
    zsys_notice (const char *format, ...);
    
//  Log informational message - low priority
CZMQ_EXPORT void
    zsys_info (const char *format, ...);
    
//  Log debug-level message - lowest priority
CZMQ_EXPORT void
    zsys_debug (const char *format, ...);

//  Self test of this class
CZMQ_EXPORT void
    zsys_test (bool verbose);
    
//  Global signal indicator, TRUE when user presses Ctrl-C or the process
//  gets a SIGTERM signal.
CZMQ_EXPORT extern volatile int zsys_interrupted;
//  Deprecated name for this variable
CZMQ_EXPORT extern volatile int zctx_interrupted;
//  @end

#ifdef __cplusplus
}
#endif

#endif
