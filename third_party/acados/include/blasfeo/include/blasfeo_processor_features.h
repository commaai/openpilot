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
* Author: Ian McInerney                                                                           *
*                                                                                                 *
**************************************************************************************************/

#ifndef BLASFEO_PROCESSOR_FEATURES_H_
#define BLASFEO_PROCESSOR_FEATURES_H_

/**
 * Flags to indicate the different processor features
 */
enum
{
    // x86-64 CPU features
    BLASFEO_PROCESSOR_FEATURE_AVX  = 0x0001,    /// AVX instruction set
    BLASFEO_PROCESSOR_FEATURE_AVX2 = 0x0002,    /// AVX2 instruction set
    BLASFEO_PROCESSOR_FEATURE_FMA  = 0x0004,    /// FMA instruction set
    BLASFEO_PROCESSOR_FEATURE_SSE3 = 0x0008,    /// SSE3 instruction set

    // ARM CPU features
    BLASFEO_PROCESSOR_FEATURE_VFPv3  = 0x0100,  /// VFPv3 instruction set
    BLASFEO_PROCESSOR_FEATURE_NEON   = 0x0100,  /// NEON instruction set
    BLASFEO_PROCESSOR_FEATURE_VFPv4  = 0x0100,  /// VFPv4 instruction set
    BLASFEO_PROCESSOR_FEATURE_NEONv2 = 0x0100,  /// NEONv2 instruction set
} BLASFEO_PROCESSOR_FEATURES;

/**
 * Test the features that this processor provides against what the library was compiled with.
 *
 * @param features - Pointer to an integer to store the supported feature set (using the flags in the BLASFEO_PROCESSOR_FEATURES enum)
 * @return 0 if current processor doesn't support all features required for this library, 1 otherwise
 */
int blasfeo_processor_cpu_features( int* features );

/**
 * Test the features that this processor provides against what the library was compiled with.
 *
 * @param features - Pointer to an integer to store the supported feature set (using the flags in the BLASFEO_PROCESSOR_FEATURES enum)
 * @return 0 if current processor doesn't support all features required for this library, 1 otherwise
 */
void blasfeo_processor_library_features( int* features );

/**
 * Create a string listing the features the current processor supports.
 *
 * @param features - Flags from the BLASFEO_PROCESSOR_FEATURES enum indicating the features supported
 * @param featureString - Character array to store the feature string in
 */
void blasfeo_processor_feature_string( int features, char* featureString );

/**
 * Get a string listing the processor features that this library version needs to run.
 *
 * @param featureString - Character array to store the feature string in
 */
void blasfeo_processor_library_string( char* featureString );

#endif  // BLASFEO_PROCESSOR_FEATURES_H_
