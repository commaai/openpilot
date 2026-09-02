// its own file: the CoreFoundation header defines a Rect that clashes with the one in common/util.h
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>

// the app menu takes its name from the main bundle, and a bare binary gets an info dictionary with its
// file name in it. That dictionary is mutable, so the name is set before glfw brings up cocoa
void setMacAppName(const char *name) {
  auto info = (CFMutableDictionaryRef)CFBundleGetInfoDictionary(CFBundleGetMainBundle());
  if (info == nullptr) return;
  CFStringRef value = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
  CFDictionarySetValue(info, CFSTR("CFBundleName"), value);
  CFRelease(value);
}
#endif
