/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*  This file is deprecated, and all its functionality provided by zmq.h     */
/*  Note that -Wpedantic compilation requires GCC to avoid using its custom
    extensions such as #warning, hence the trick below. Also, pragmas for
    warnings or other messages are not standard, not portable, and not all
    compilers even have an equivalent concept.
    So in the worst case, this include file is treated as silently empty. */

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || defined(_MSC_VER)
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcpp"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wall"
#endif
#pragma message("Warning: zmq_utils.h is deprecated. All its functionality is provided by zmq.h.")
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
#endif
