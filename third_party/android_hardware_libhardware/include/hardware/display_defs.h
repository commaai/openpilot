/* Copyright (c) 2014-2015, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of The Linux Foundation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ANDROID_INCLUDE_DISPLAY_DEFS_H
#define ANDROID_INCLUDE_DISPLAY_DEFS_H

#include <stdint.h>
#include <sys/cdefs.h>

#include <hardware/hwcomposer.h>

__BEGIN_DECLS

/* Will need to update below enums if hwcomposer_defs.h is updated */

/* Extended events for hwc_methods::eventControl() */
enum {
    HWC_EVENT_ORIENTATION             = HWC_EVENT_VSYNC + 1
};


/* Extended hwc_layer_t::compositionType values */
enum {
    /* this layer will be handled in the HWC, using a blit engine */
    HWC_BLIT                          = 0xFF
};

/* Extended hwc_layer_t::flags values
 * Flags are set by SurfaceFlinger and read by the HAL
 */
enum {
    /*
     * HWC_SCREENSHOT_ANIMATOR_LAYER is set by surfaceflinger to indicate
     * that this layer is a screenshot animating layer. HWC uses this
     * info to disable rotation animation on External Display
     */
    HWC_SCREENSHOT_ANIMATOR_LAYER     = 0x00000004
};

__END_DECLS

#endif /* ANDROID_INCLUDE_DISPLAY_DEFS_H*/
