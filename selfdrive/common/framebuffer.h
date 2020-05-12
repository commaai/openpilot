#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FramebufferState FramebufferState;

FramebufferState* framebuffer_init(
    const char* name, int32_t layer, int alpha,
    int *out_w, int *out_h);

void framebuffer_set_power(FramebufferState *s, int mode);
void framebuffer_swap(FramebufferState *s);

/* Display power modes */
enum {
    /* The display is turned off (blanked). */
    HWC_POWER_MODE_OFF      = 0,
    /* The display is turned on and configured in a low power state
     * that is suitable for presenting ambient information to the user,
     * possibly with lower fidelity than normal but greater efficiency. */
    HWC_POWER_MODE_DOZE     = 1,
    /* The display is turned on normally. */
    HWC_POWER_MODE_NORMAL   = 2,
    /* The display is configured as in HWC_POWER_MODE_DOZE but may
     * stop applying frame buffer updates from the graphics subsystem.
     * This power mode is effectively a hint from the doze dream to
     * tell the hardware that it is done drawing to the display for the
     * time being and that the display should remain on in a low power
     * state and continue showing its current contents indefinitely
     * until the mode changes.
     *
     * This mode may also be used as a signal to enable hardware-based doze
     * functionality.  In this case, the doze dream is effectively
     * indicating that the hardware is free to take over the display
     * and manage it autonomously to implement low power always-on display
     * functionality. */
    HWC_POWER_MODE_DOZE_SUSPEND  = 3,
};


#ifdef __cplusplus
}
#endif

#endif
