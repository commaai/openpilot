#ifndef EMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define EMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

#include "yaml-cpp/binary.h"
#include "yaml-cpp/dll.h"
#include "yaml-cpp/emitterdef.h"
#include "yaml-cpp/emittermanip.h"
#include "yaml-cpp/noncopyable.h"
#include "yaml-cpp/null.h"
#include "yaml-cpp/ostream_wrapper.h"

namespace YAML {
class Binary;
struct _Null;
}  // namespace YAML

namespace YAML {
class EmitterState;

class YAML_CPP_API Emitter : private noncopyable {
 public:
  Emitter();
  explicit Emitter(std::ostream& stream);
  ~Emitter();

  // output
  const char* c_str() const;
  std::size_t size() const;

  // state checking
  bool good() const;
  const std::string GetLastError() const;

  // global setters
  bool SetOutputCharset(EMITTER_MANIP value);
  bool SetStringFormat(EMITTER_MANIP value);
  bool SetBoolFormat(EMITTER_MANIP value);
  bool SetIntBase(EMITTER_MANIP value);
  bool SetSeqFormat(EMITTER_MANIP value);
  bool SetMapFormat(EMITTER_MANIP value);
  bool SetIndent(std::size_t n);
  bool SetPreCommentIndent(std::size_t n);
  bool SetPostCommentIndent(std::size_t n);
  bool SetFloatPrecision(std::size_t n);
  bool SetDoublePrecision(std::size_t n);

  // local setters
  Emitter& SetLocalValue(EMITTER_MANIP value);
  Emitter& SetLocalIndent(const _Indent& indent);
  Emitter& SetLocalPrecision(const _Precision& precision);

  // overloads of write
  Emitter& Write(const std::string& str);
  Emitter& Write(bool b);
  Emitter& Write(char ch);
  Emitter& Write(const _Alias& alias);
  Emitter& Write(const _Anchor& anchor);
  Emitter& Write(const _Tag& tag);
  Emitter& Write(const _Comment& comment);
  Emitter& Write(const _Null& n);
  Emitter& Write(const Binary& binary);

  template <typename T>
  Emitter& WriteIntegralType(T value);

  template <typename T>
  Emitter& WriteStreamable(T value);

 private:
  template <typename T>
  void SetStreamablePrecision(std::stringstream&) {}
  std::size_t GetFloatPrecision() const;
  std::size_t GetDoublePrecision() const;

  void PrepareIntegralStream(std::stringstream& stream) const;
  void StartedScalar();

 private:
  void EmitBeginDoc();
  void EmitEndDoc();
  void EmitBeginSeq();
  void EmitEndSeq();
  void EmitBeginMap();
  void EmitEndMap();
  void EmitNewline();
  void EmitKindTag();
  void EmitTag(bool verbatim, const _Tag& tag);

  void PrepareNode(EmitterNodeType::value child);
  void PrepareTopNode(EmitterNodeType::value child);
  void FlowSeqPrepareNode(EmitterNodeType::value child);
  void BlockSeqPrepareNode(EmitterNodeType::value child);

  void FlowMapPrepareNode(EmitterNodeType::value child);

  void FlowMapPrepareLongKey(EmitterNodeType::value child);
  void FlowMapPrepareLongKeyValue(EmitterNodeType::value child);
  void FlowMapPrepareSimpleKey(EmitterNodeType::value child);
  void FlowMapPrepareSimpleKeyValue(EmitterNodeType::value child);

  void BlockMapPrepareNode(EmitterNodeType::value child);

  void BlockMapPrepareLongKey(EmitterNodeType::value child);
  void BlockMapPrepareLongKeyValue(EmitterNodeType::value child);
  void BlockMapPrepareSimpleKey(EmitterNodeType::value child);
  void BlockMapPrepareSimpleKeyValue(EmitterNodeType::value child);

  void SpaceOrIndentTo(bool requireSpace, std::size_t indent);

  const char* ComputeFullBoolName(bool b) const;
  bool CanEmitNewline() const;

 private:
  std::unique_ptr<EmitterState> m_pState;
  ostream_wrapper m_stream;
};

template <typename T>
inline Emitter& Emitter::WriteIntegralType(T value) {
  if (!good())
    return *this;

  PrepareNode(EmitterNodeType::Scalar);

  std::stringstream stream;
  PrepareIntegralStream(stream);
  stream << value;
  m_stream << stream.str();

  StartedScalar();

  return *this;
}

template <typename T>
inline Emitter& Emitter::WriteStreamable(T value) {
  if (!good())
    return *this;

  PrepareNode(EmitterNodeType::Scalar);

  std::stringstream stream;
  SetStreamablePrecision<T>(stream);
  stream << value;
  m_stream << stream.str();

  StartedScalar();

  return *this;
}

template <>
inline void Emitter::SetStreamablePrecision<float>(std::stringstream& stream) {
  stream.precision(static_cast<std::streamsize>(GetFloatPrecision()));
}

template <>
inline void Emitter::SetStreamablePrecision<double>(std::stringstream& stream) {
  stream.precision(static_cast<std::streamsize>(GetDoublePrecision()));
}

// overloads of insertion
inline Emitter& operator<<(Emitter& emitter, const std::string& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, bool v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, char v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, unsigned char v) {
  return emitter.Write(static_cast<char>(v));
}
inline Emitter& operator<<(Emitter& emitter, const _Alias& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, const _Anchor& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, const _Tag& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, const _Comment& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, const _Null& v) {
  return emitter.Write(v);
}
inline Emitter& operator<<(Emitter& emitter, const Binary& b) {
  return emitter.Write(b);
}

inline Emitter& operator<<(Emitter& emitter, const char* v) {
  return emitter.Write(std::string(v));
}

inline Emitter& operator<<(Emitter& emitter, int v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, unsigned int v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, short v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, unsigned short v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, long v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, unsigned long v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, long long v) {
  return emitter.WriteIntegralType(v);
}
inline Emitter& operator<<(Emitter& emitter, unsigned long long v) {
  return emitter.WriteIntegralType(v);
}

inline Emitter& operator<<(Emitter& emitter, float v) {
  return emitter.WriteStreamable(v);
}
inline Emitter& operator<<(Emitter& emitter, double v) {
  return emitter.WriteStreamable(v);
}

inline Emitter& operator<<(Emitter& emitter, EMITTER_MANIP value) {
  return emitter.SetLocalValue(value);
}

inline Emitter& operator<<(Emitter& emitter, _Indent indent) {
  return emitter.SetLocalIndent(indent);
}

inline Emitter& operator<<(Emitter& emitter, _Precision precision) {
  return emitter.SetLocalPrecision(precision);
}
}

#endif  // EMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
