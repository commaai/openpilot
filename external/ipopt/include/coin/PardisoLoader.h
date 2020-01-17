/* Copyright (C) 2008 GAMS Development and others
 All Rights Reserved.
 This code is published under the Eclipse Public License.

 $Id: PardisoLoader.h 2204 2013-04-13 13:49:26Z stefan $

 Author: Stefan Vigerske
*/

#ifndef PARDISOLOADER_H_
#define PARDISOLOADER_H_

#ifdef __cplusplus
extern "C" {
#endif
  /** Tries to load a dynamically linked library with Pardiso.
   * Return a failure if the library cannot be loaded or not all Pardiso symbols are found.
   * @param libname The name under which the Pardiso lib can be found, or NULL to use a default name (libpardiso.SHAREDLIBEXT).
   * @param msgbuf A buffer where we can store a failure message. Assumed to be NOT NULL!
   * @param msglen Length of the message buffer.
   * @return Zero on success, nonzero on failure.
   */
  int LSL_loadPardisoLib(const char* libname, char* msgbuf, int msglen);

  /** Unloads a loaded Pardiso library.
   * @return Zero on success, nonzero on failure.
   */
  int LSL_unloadPardisoLib();

  /** Indicates whether a Pardiso library has been successfully loaded.
   * @return Zero if not loaded, nonzero if handle is loaded
   */
  int LSL_isPardisoLoaded();

  /** Returns name of the shared library that should contain Pardiso */
  char* LSL_PardisoLibraryName();
#ifdef __cplusplus
}
#endif

#endif /*PARADISOLOADER_H_*/
