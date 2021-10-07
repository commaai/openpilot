#***************************************************************************
#                                  _   _ ____  _
#  Project                     ___| | | |  _ \| |
#                             / __| | | | |_) | |
#                            | (__| |_| |  _ <| |___
#                             \___|\___/|_| \_\_____|
#
# Copyright (C) 1998 - 2011, Daniel Stenberg, <daniel@haxx.se>, et al.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution. The terms
# are also available at http://curl.haxx.se/docs/copyright.html.
#
# You may opt to use, copy, modify, merge, publish, distribute and/or sell
# copies of the Software, and permit persons to whom the Software is
# furnished to do so, under the terms of the COPYING file.
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# KIND, either express or implied.
#
###########################################################################
pkginclude_HEADERS = \
	curl.h curlver.h easy.h mprintf.h stdcheaders.h multi.h \
	typecheck-gcc.h curlbuild.h curlrules.h

pkgincludedir= $(includedir)/curl

# curlbuild.h does not exist in the git tree. When the original libcurl
# source code distribution archive file is created, curlbuild.h.dist is
# renamed to curlbuild.h and included in the tarball so that it can be
# used directly on non-configure systems.
#
# The distributed curlbuild.h will be overwritten on configure systems
# when the configure script runs, with one that is suitable and specific
# to the library being configured and built.
#
# curlbuild.h.in is the distributed template file from which the configure
# script creates curlbuild.h at library configuration time, overwiting the
# one included in the distribution archive.
#
# curlbuild.h.dist is not included in the source code distribution archive.

EXTRA_DIST = curlbuild.h.in

DISTCLEANFILES = curlbuild.h

checksrc:
	@@PERL@ $(top_srcdir)/lib/checksrc.pl -Wcurlbuild.h -D$(top_srcdir)/include/curl $(pkginclude_HEADERS) $(EXTRA_DIST)

if CURLDEBUG
# for debug builds, we scan the sources on all regular make invokes
all-local: checksrc
endif
