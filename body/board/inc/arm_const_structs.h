/* ----------------------------------------------------------------------
* Copyright (C) 2010-2014 ARM Limited. All rights reserved.
*
* $Date:        19. March 2015
* $Revision: 	V.1.4.5
*
* Project: 	    CMSIS DSP Library
* Title:	    arm_const_structs.h
*
* Description:	This file has constant structs that are initialized for
*              user convenience.  For example, some can be given as
*              arguments to the arm_cfft_f32() function.
*
* Target Processor: Cortex-M4/Cortex-M3
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of ARM LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
* -------------------------------------------------------------------- */

#ifndef _ARM_CONST_STRUCTS_H
#define _ARM_CONST_STRUCTS_H

#include "arm_math.h"
#include "arm_common_tables.h"

   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len16;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len32;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len64;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len128;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len256;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len512;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len1024;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len2048;
   extern const arm_cfft_instance_f32 arm_cfft_sR_f32_len4096;

   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len16;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len32;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len64;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len128;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len256;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len512;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len1024;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len2048;
   extern const arm_cfft_instance_q31 arm_cfft_sR_q31_len4096;

   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len16;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len32;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len64;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len128;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len256;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len512;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len1024;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len2048;
   extern const arm_cfft_instance_q15 arm_cfft_sR_q15_len4096;

#endif
