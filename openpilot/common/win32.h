#pragma once

// Use instead of <windows.h>: lean headers, no min/max/MessageBox/NO_ERROR macros (cabana's MessageBox, capnp's NO_ERROR).
// Only from .cc files that never see common/params.h: its BOOL/INT/FLOAT enumerators clash with the Win32 typedefs.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#undef NO_ERROR
#undef MessageBox
#endif
