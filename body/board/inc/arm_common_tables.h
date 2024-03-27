/* ----------------------------------------------------------------------
* Copyright (C) 2010-2014 ARM Limited. All rights reserved.
*
* $Date:        19. October 2015
* $Revision: 	V.1.4.5 a
*
* Project: 	    CMSIS DSP Library
* Title:	    arm_common_tables.h
*
* Description:	This file has extern declaration for common tables like Bitreverse, reciprocal etc which are used across different functions
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

#ifndef _ARM_COMMON_TABLES_H
#define _ARM_COMMON_TABLES_H

#include "arm_math.h"

extern const uint16_t armBitRevTable[1024];
extern const q15_t armRecipTableQ15[64];
extern const q31_t armRecipTableQ31[64];
/* extern const q31_t realCoefAQ31[1024]; */
/* extern const q31_t realCoefBQ31[1024]; */
extern const float32_t twiddleCoef_16[32];
extern const float32_t twiddleCoef_32[64];
extern const float32_t twiddleCoef_64[128];
extern const float32_t twiddleCoef_128[256];
extern const float32_t twiddleCoef_256[512];
extern const float32_t twiddleCoef_512[1024];
extern const float32_t twiddleCoef_1024[2048];
extern const float32_t twiddleCoef_2048[4096];
extern const float32_t twiddleCoef_4096[8192];
#define twiddleCoef twiddleCoef_4096
extern const q31_t twiddleCoef_16_q31[24];
extern const q31_t twiddleCoef_32_q31[48];
extern const q31_t twiddleCoef_64_q31[96];
extern const q31_t twiddleCoef_128_q31[192];
extern const q31_t twiddleCoef_256_q31[384];
extern const q31_t twiddleCoef_512_q31[768];
extern const q31_t twiddleCoef_1024_q31[1536];
extern const q31_t twiddleCoef_2048_q31[3072];
extern const q31_t twiddleCoef_4096_q31[6144];
extern const q15_t twiddleCoef_16_q15[24];
extern const q15_t twiddleCoef_32_q15[48];
extern const q15_t twiddleCoef_64_q15[96];
extern const q15_t twiddleCoef_128_q15[192];
extern const q15_t twiddleCoef_256_q15[384];
extern const q15_t twiddleCoef_512_q15[768];
extern const q15_t twiddleCoef_1024_q15[1536];
extern const q15_t twiddleCoef_2048_q15[3072];
extern const q15_t twiddleCoef_4096_q15[6144];
extern const float32_t twiddleCoef_rfft_32[32];
extern const float32_t twiddleCoef_rfft_64[64];
extern const float32_t twiddleCoef_rfft_128[128];
extern const float32_t twiddleCoef_rfft_256[256];
extern const float32_t twiddleCoef_rfft_512[512];
extern const float32_t twiddleCoef_rfft_1024[1024];
extern const float32_t twiddleCoef_rfft_2048[2048];
extern const float32_t twiddleCoef_rfft_4096[4096];


/* floating-point bit reversal tables */
#define ARMBITREVINDEXTABLE__16_TABLE_LENGTH ((uint16_t)20  )
#define ARMBITREVINDEXTABLE__32_TABLE_LENGTH ((uint16_t)48  )
#define ARMBITREVINDEXTABLE__64_TABLE_LENGTH ((uint16_t)56  )
#define ARMBITREVINDEXTABLE_128_TABLE_LENGTH ((uint16_t)208 )
#define ARMBITREVINDEXTABLE_256_TABLE_LENGTH ((uint16_t)440 )
#define ARMBITREVINDEXTABLE_512_TABLE_LENGTH ((uint16_t)448 )
#define ARMBITREVINDEXTABLE1024_TABLE_LENGTH ((uint16_t)1800)
#define ARMBITREVINDEXTABLE2048_TABLE_LENGTH ((uint16_t)3808)
#define ARMBITREVINDEXTABLE4096_TABLE_LENGTH ((uint16_t)4032)

extern const uint16_t armBitRevIndexTable16[ARMBITREVINDEXTABLE__16_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable32[ARMBITREVINDEXTABLE__32_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable64[ARMBITREVINDEXTABLE__64_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable128[ARMBITREVINDEXTABLE_128_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable256[ARMBITREVINDEXTABLE_256_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable512[ARMBITREVINDEXTABLE_512_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable1024[ARMBITREVINDEXTABLE1024_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable2048[ARMBITREVINDEXTABLE2048_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable4096[ARMBITREVINDEXTABLE4096_TABLE_LENGTH];

/* fixed-point bit reversal tables */
#define ARMBITREVINDEXTABLE_FIXED___16_TABLE_LENGTH ((uint16_t)12  )
#define ARMBITREVINDEXTABLE_FIXED___32_TABLE_LENGTH ((uint16_t)24  )
#define ARMBITREVINDEXTABLE_FIXED___64_TABLE_LENGTH ((uint16_t)56  )
#define ARMBITREVINDEXTABLE_FIXED__128_TABLE_LENGTH ((uint16_t)112 )
#define ARMBITREVINDEXTABLE_FIXED__256_TABLE_LENGTH ((uint16_t)240 )
#define ARMBITREVINDEXTABLE_FIXED__512_TABLE_LENGTH ((uint16_t)480 )
#define ARMBITREVINDEXTABLE_FIXED_1024_TABLE_LENGTH ((uint16_t)992 )
#define ARMBITREVINDEXTABLE_FIXED_2048_TABLE_LENGTH ((uint16_t)1984)
#define ARMBITREVINDEXTABLE_FIXED_4096_TABLE_LENGTH ((uint16_t)4032)

extern const uint16_t armBitRevIndexTable_fixed_16[ARMBITREVINDEXTABLE_FIXED___16_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_32[ARMBITREVINDEXTABLE_FIXED___32_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_64[ARMBITREVINDEXTABLE_FIXED___64_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_128[ARMBITREVINDEXTABLE_FIXED__128_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_256[ARMBITREVINDEXTABLE_FIXED__256_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_512[ARMBITREVINDEXTABLE_FIXED__512_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_1024[ARMBITREVINDEXTABLE_FIXED_1024_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_2048[ARMBITREVINDEXTABLE_FIXED_2048_TABLE_LENGTH];
extern const uint16_t armBitRevIndexTable_fixed_4096[ARMBITREVINDEXTABLE_FIXED_4096_TABLE_LENGTH];

/* Tables for Fast Math Sine and Cosine */
extern const float32_t sinTable_f32[FAST_MATH_TABLE_SIZE + 1];
extern const q31_t sinTable_q31[FAST_MATH_TABLE_SIZE + 1];
extern const q15_t sinTable_q15[FAST_MATH_TABLE_SIZE + 1];

#endif /*  ARM_COMMON_TABLES_H */
