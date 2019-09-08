#ifndef NONCOPYABLE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define NONCOPYABLE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"

namespace YAML {
// this is basically boost::noncopyable
class YAML_CPP_API noncopyable {
 protected:
  noncopyable() {}
  ~noncopyable() {}

 private:
  noncopyable(const noncopyable&);
  const noncopyable& operator=(const noncopyable&);
};
}

#endif  // NONCOPYABLE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
