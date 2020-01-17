#ifndef BASE64_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define BASE64_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <string>
#include <vector>

#include "yaml-cpp/dll.h"

namespace YAML {
YAML_CPP_API std::string EncodeBase64(const unsigned char *data,
                                      std::size_t size);
YAML_CPP_API std::vector<unsigned char> DecodeBase64(const std::string &input);

class YAML_CPP_API Binary {
 public:
  Binary() : m_unownedData(0), m_unownedSize(0) {}
  Binary(const unsigned char *data_, std::size_t size_)
      : m_unownedData(data_), m_unownedSize(size_) {}

  bool owned() const { return !m_unownedData; }
  std::size_t size() const { return owned() ? m_data.size() : m_unownedSize; }
  const unsigned char *data() const {
    return owned() ? &m_data[0] : m_unownedData;
  }

  void swap(std::vector<unsigned char> &rhs) {
    if (m_unownedData) {
      m_data.swap(rhs);
      rhs.clear();
      rhs.resize(m_unownedSize);
      std::copy(m_unownedData, m_unownedData + m_unownedSize, rhs.begin());
      m_unownedData = 0;
      m_unownedSize = 0;
    } else {
      m_data.swap(rhs);
    }
  }

  bool operator==(const Binary &rhs) const {
    const std::size_t s = size();
    if (s != rhs.size())
      return false;
    const unsigned char *d1 = data();
    const unsigned char *d2 = rhs.data();
    for (std::size_t i = 0; i < s; i++) {
      if (*d1++ != *d2++)
        return false;
    }
    return true;
  }

  bool operator!=(const Binary &rhs) const { return !(*this == rhs); }

 private:
  std::vector<unsigned char> m_data;
  const unsigned char *m_unownedData;
  std::size_t m_unownedSize;
};
}

#endif  // BASE64_H_62B23520_7C8E_11DE_8A39_0800200C9A66
