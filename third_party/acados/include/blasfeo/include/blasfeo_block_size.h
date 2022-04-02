/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#ifndef BLASFEO_BLOCK_SIZE_H_
#define BLASFEO_BLOCK_SIZE_H_



#define D_EL_SIZE 8 // double precision
#define S_EL_SIZE 4 // single precision



#if defined( TARGET_X64_INTEL_SKYLAKE_X )
// common
#define CACHE_LINE_SIZE 64 // data cache size: 64 bytes
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 8-way
#define L2_CACHE_SIZE (256*1024) //(1024*1024) // L2 data cache size: 1 MB ; DTLB1 64*4 kB = 256 kB
#define LLC_CACHE_SIZE (6*1024*1024) //(8*1024*1024) // LLC cache size: 8 MB ; TLB 1536*4 kB = 6 MB
// double
#define D_PS 8 // panel size
#define D_PLD 8 // 4 // GCD of panel length
#define D_M_KERNEL 24 // max kernel size
#define D_N_KERNEL 8 // max kernel size
#define D_KC 128 //256 // 192
#define D_NC 144 //72 //96 //72 // 120 // 512
#define D_MC 2400 // 6000
// single
#define S_PS 16 // panel size
#define S_PLD 4 // GCD of panel length TODO probably 16 when writing assebly
#define S_M_KERNEL 32 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 128 //256
#define S_NC 128 //144
#define S_MC 3000

#elif defined( TARGET_X64_INTEL_HASWELL )
// common
#define CACHE_LINE_SIZE 64 // data cache size: 64 bytes
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 8-way
#define L2_CACHE_SIZE (256*1024) // L2 data cache size: 256 kB ; DTLB1 64*4 kB = 256 kB
#define LLC_CACHE_SIZE (6*1024*1024) // LLC cache size: 6 MB ; TLB 1024*4 kB = 4 MB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 12 // max kernel size
#define D_N_KERNEL 8 // max kernel size
#define D_KC 256 // 192
#define D_NC 64 //96 //72 // 120 // 512
#define D_MC 1500
// single
#define S_PS 8 // panel size
#define S_PLD 4 // 2 // GCD of panel length
#define S_M_KERNEL 24 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 144
#define S_MC 3000

#elif defined( TARGET_X64_INTEL_SANDY_BRIDGE )
// common
#define CACHE_LINE_SIZE 64 // data cache size: 64 bytes
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 8-way
#define L2_CACHE_SIZE (256*1024) // L2 data cache size: 256 kB ; DTLB1 64*4 kB = 256 kB
#define LLC_CACHE_SIZE (4*1024*1024) // LLC cache size: 4 MB ; TLB 1024*4 kB = 4 MB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 8 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256 //320 //256 //320
#define D_NC 72 //64 //72 //60 // 120
#define D_MC 1000 // 800
// single
#define S_PS 8 // panel size
#define S_PLD 4 // 2 // GCD of panel length
#define S_M_KERNEL 16 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 144
#define S_MC 2000

#elif defined( TARGET_X64_INTEL_CORE )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy

#elif defined( TARGET_X64_AMD_BULLDOZER )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_X86_AMD_JAGUAR )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_X86_AMD_BARCELONA )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined(TARGET_ARMV8A_APPLE_M1)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (128*1024) // L1 data cache size (big cores): 64 kB, ?-way ; DTLB1 ?
#define LLC_CACHE_SIZE (12*1024*1024) // LLC (L2) cache size (big cores): 12 MB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 8 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 512 //256
#define D_NC 128 //256
#define D_MC 6000
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 512
#define S_NC 256
#define S_MC 6000


#elif defined(TARGET_ARMV8A_ARM_CORTEX_A76)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (64*1024) // L1 data cache size: 64 kB, 4-way ; DTLB1 48*4 kB = 192 kB
#define LLC_CACHE_SIZE (1*1024*1024) // LLC cache size: 1 MB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 8 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 512 //256
#define D_NC 128 //256
#define D_MC 6000
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 512
#define S_NC 256
#define S_MC 6000


#elif defined(TARGET_ARMV8A_ARM_CORTEX_A73)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 (64?) kB, 4-way, seen as 8-(16-)way ; DTLB1 48*4 kB = 192 kB
#define LLC_CACHE_SIZE (1*1024*1024) // LLC cache size: 1 MB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 8 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 320
#define D_NC 256
#define D_MC 6000
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined(TARGET_ARMV8A_ARM_CORTEX_A57)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 2-way ; DTLB1 32*4 kB = 128 kB
#define LLC_CACHE_SIZE (1*1024*1024) // LLC cache size: 1 MB // 2 MB ???
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 8 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 128 //224 //256 //192
#define D_NC 72 //40 //36 //48
#define D_MC (4*192) //512 //488 //600
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined(TARGET_ARMV8A_ARM_CORTEX_A55)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 4-way ; DTLB1 16*4 kB = 64 kB
#define LLC_CACHE_SIZE (512*1024) // LLC cache size: 512 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 12 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 224
#define D_NC 160
#define D_MC 6000
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined(TARGET_ARMV8A_ARM_CORTEX_A53)
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB, 4-way ??? ; DTLB1 10*4 kB = 40 kB
#define LLC_CACHE_SIZE (256*1024) // LLC cache size: 256 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 12 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 160
#define D_NC 128
#define D_MC 6000
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 8 // max kernel size
#define S_N_KERNEL 8 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_ARMV7A_ARM_CORTEX_A15 )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_ARMV7A_ARM_CORTEX_A7 )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_ARMV7A_ARM_CORTEX_A9 )
// common
#define CACHE_LINE_SIZE 32
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy
// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#elif defined( TARGET_GENERIC )
// common
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE (32*1024) // L1 data cache size: 32 kB
// double
#define D_PS 4 // panel size
#define D_PLD 4 // 2 // GCD of panel length
#define D_M_KERNEL 4 // max kernel size
#define D_N_KERNEL 4 // max kernel size
#define D_KC 256
#define D_NC 128 // TODO these are just dummy
#define D_MC 3000 // TODO these are just dummy

// single
#define S_PS 4
#define S_PLD 4 //2
#define S_M_KERNEL 4 // max kernel size
#define S_N_KERNEL 4 // max kernel size
#define S_KC 256
#define S_NC 128 // TODO these are just dummy
#define S_MC 3000 // TODO these are just dummy


#else
#error "Unknown architecture"
#endif



#define D_CACHE_LINE_EL (CACHE_LINE_SIZE/D_EL_SIZE)
#define D_L1_CACHE_EL (L1_CACHE_SIZE/D_EL_SIZE)
#define D_L2_CACHE_EL (L2_CACHE_SIZE/D_EL_SIZE)
#define D_LLC_CACHE_EL (LLC_CACHE_SIZE/D_EL_SIZE)

#define S_CACHE_LINE_EL (CACHE_LINE_SIZE/S_EL_SIZE)
#define S_L1_CACHE_EL (L1_CACHE_SIZE/S_EL_SIZE)
#define S_L2_CACHE_EL (L2_CACHE_SIZE/S_EL_SIZE)
#define S_LLC_CACHE_EL (LLC_CACHE_SIZE/S_EL_SIZE)



#endif  // BLASFEO_BLOCK_SIZE_H_
