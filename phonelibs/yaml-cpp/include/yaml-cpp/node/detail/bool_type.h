#ifndef NODE_DETAIL_BOOL_TYPE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define NODE_DETAIL_BOOL_TYPE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

namespace YAML {
namespace detail {
struct unspecified_bool {
  struct NOT_ALLOWED;
  static void true_value(NOT_ALLOWED*) {}
};
typedef void (*unspecified_bool_type)(unspecified_bool::NOT_ALLOWED*);
}
}

#define YAML_CPP_OPERATOR_BOOL()                                            \
  operator YAML::detail::unspecified_bool_type() const {                    \
    return this->operator!() ? 0                                            \
                             : &YAML::detail::unspecified_bool::true_value; \
  }

#endif  // NODE_DETAIL_BOOL_TYPE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
