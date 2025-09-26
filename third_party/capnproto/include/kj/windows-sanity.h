// Copyright (c) 2013-2018 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// This file replaces poorly-named #defines from windows.h with properly-namespaced versions, so
// that they no longer conflict with similarly-named identifiers in other namespaces.
//
// This file must be #included some time after windows.h has been #included but before any attempt
// to use the names for other purposes. However, this can be difficult to determine in header
// files. Typically KJ / Cap'n Proto headers avoid including windows.h at all, but may use
// conflicting identifiers. In order to relieve application developers from the need to include
// windows-sanity.h themselves, we would like these headers to conditionally apply the fixes if
// and only if windows.h was already included. Therefore, this header checks if windows.h has been
// included and only applies fixups if this is the case. Furthermore, this header is designed such
// that it can be included multiple times, and the fixups will be applied the first time it is
// included *after* windows.h.
//
// Now, as long as any headers which need to use conflicting identifier names be sure to #include
// windows-sanity.h, we can be sure that no conflicts will occur regardless of in what order the
// application chooses to include these headers vs. windows.h.

#if !_WIN32 && !__CYGWIN__

// Not on Windows. Tell the compiler never to try to include this again.
#pragma once

#elif defined(_INC_WINDOWS)

// We're on Windows and windows.h has been included. We need to fixup the namespace. We only need
// to do this once, but we can't do it until windows.h has been included. Since that has happened
// now, we use `#pragma once` to tell the compiler never to include this file again.
#pragma once

namespace kj_win32_workarounds {
  // Namespace containing constant definitions intended to replace constants that are defined as
  // macros in the Windows headers. Do not refer to this namespace directly, we'll import it into
  // the global scope below.

#ifdef ERROR  // This could be absent if e.g. NOGDI was used.
  const auto ERROR_ = ERROR;
#undef ERROR
  const auto ERROR = ERROR_;
#endif

  typedef VOID VOID_;
#undef VOID
  typedef VOID_ VOID;
}

// Pull our constant definitions into the global namespace -- but only if they don't already exist
// in the global namespace.
using namespace kj_win32_workarounds;

#endif
