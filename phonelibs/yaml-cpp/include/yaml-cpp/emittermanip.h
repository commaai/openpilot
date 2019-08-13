#ifndef EMITTERMANIP_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define EMITTERMANIP_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <string>

namespace YAML {
enum EMITTER_MANIP {
  // general manipulators
  Auto,
  TagByKind,
  Newline,

  // output character set
  EmitNonAscii,
  EscapeNonAscii,

  // string manipulators
  // Auto, // duplicate
  SingleQuoted,
  DoubleQuoted,
  Literal,

  // bool manipulators
  YesNoBool,      // yes, no
  TrueFalseBool,  // true, false
  OnOffBool,      // on, off
  UpperCase,      // TRUE, N
  LowerCase,      // f, yes
  CamelCase,      // No, Off
  LongBool,       // yes, On
  ShortBool,      // y, t

  // int manipulators
  Dec,
  Hex,
  Oct,

  // document manipulators
  BeginDoc,
  EndDoc,

  // sequence manipulators
  BeginSeq,
  EndSeq,
  Flow,
  Block,

  // map manipulators
  BeginMap,
  EndMap,
  Key,
  Value,
  // Flow, // duplicate
  // Block, // duplicate
  // Auto, // duplicate
  LongKey
};

struct _Indent {
  _Indent(int value_) : value(value_) {}
  int value;
};

inline _Indent Indent(int value) { return _Indent(value); }

struct _Alias {
  _Alias(const std::string& content_) : content(content_) {}
  std::string content;
};

inline _Alias Alias(const std::string content) { return _Alias(content); }

struct _Anchor {
  _Anchor(const std::string& content_) : content(content_) {}
  std::string content;
};

inline _Anchor Anchor(const std::string content) { return _Anchor(content); }

struct _Tag {
  struct Type {
    enum value { Verbatim, PrimaryHandle, NamedHandle };
  };

  explicit _Tag(const std::string& prefix_, const std::string& content_,
                Type::value type_)
      : prefix(prefix_), content(content_), type(type_) {}
  std::string prefix;
  std::string content;
  Type::value type;
};

inline _Tag VerbatimTag(const std::string content) {
  return _Tag("", content, _Tag::Type::Verbatim);
}

inline _Tag LocalTag(const std::string content) {
  return _Tag("", content, _Tag::Type::PrimaryHandle);
}

inline _Tag LocalTag(const std::string& prefix, const std::string content) {
  return _Tag(prefix, content, _Tag::Type::NamedHandle);
}

inline _Tag SecondaryTag(const std::string content) {
  return _Tag("", content, _Tag::Type::NamedHandle);
}

struct _Comment {
  _Comment(const std::string& content_) : content(content_) {}
  std::string content;
};

inline _Comment Comment(const std::string content) { return _Comment(content); }

struct _Precision {
  _Precision(int floatPrecision_, int doublePrecision_)
      : floatPrecision(floatPrecision_), doublePrecision(doublePrecision_) {}

  int floatPrecision;
  int doublePrecision;
};

inline _Precision FloatPrecision(int n) { return _Precision(n, -1); }

inline _Precision DoublePrecision(int n) { return _Precision(-1, n); }

inline _Precision Precision(int n) { return _Precision(n, n); }
}

#endif  // EMITTERMANIP_H_62B23520_7C8E_11DE_8A39_0800200C9A66
