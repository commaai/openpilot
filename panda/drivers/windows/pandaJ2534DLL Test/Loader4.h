// Loader4.h
// (c) 2005 National Control Systems, Inc.
// Portions (c) 2004 Drew Technologies, Inc.

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to:
// the Free Software Foundation, Inc.
// 51 Franklin Street, Fifth Floor
// Boston, MA  02110-1301, USA

// National Control Systems, Inc.
// 10737 Hamburg Rd
// Hamburg, MI 48139
// 810-231-2901

// Drew Technologies, Inc.
// 7012  E.M -36, Suite 3B
// Whitmore Lake, MI 48189
// 810-231-3171

#include "pandaJ2534DLL/J2534_v0404.h"

//Other Functions
long WINAPI LoadJ2534Dll(char *);
long WINAPI UnloadJ2534Dll();

// NCS Returns of any functions not found
#define ERR_NO_PTOPEN					0x0001
#define ERR_NO_PTCLOSE					0x0002
#define ERR_NO_PTCONNECT				0x0004
#define ERR_NO_PTDISCONNECT				0x0008
#define ERR_NO_PTREADMSGS				0x0010
#define ERR_NO_PTWRITEMSGS				0x0020
#define ERR_NO_PTSTARTPERIODICMSG		0x0040
#define ERR_NO_PTSTOPPERIODICMSG		0x0080
#define ERR_NO_PTSTARTMSGFILTER			0x0100
#define ERR_NO_PTSTOPMSGFILTER			0x0200
#define ERR_NO_PTSETPROGRAMMINGVOLTAGE	0x0400
#define ERR_NO_PTREADVERSION			0x0800
#define ERR_NO_PTGETLASTERROR			0x1000
#define ERR_NO_PTIOCTL					0x2000
#define ERR_NO_FUNCTIONS				0x3fff
#define ERR_NO_DLL						-1
#define ERR_WRONG_DLL_VER				-2
#define ERR_FUNC_MISSING				-3
