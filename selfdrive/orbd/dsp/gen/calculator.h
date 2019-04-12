#ifndef _CALCULATOR_H
#define _CALCULATOR_H

#include <stdint.h>
typedef  uint8_t           uint8;
typedef  uint32_t           uint32;

#ifndef __QAIC_HEADER
#define __QAIC_HEADER(ff) ff
#endif //__QAIC_HEADER

#ifndef __QAIC_HEADER_EXPORT
#define __QAIC_HEADER_EXPORT
#endif // __QAIC_HEADER_EXPORT

#ifndef __QAIC_HEADER_ATTRIBUTE
#define __QAIC_HEADER_ATTRIBUTE
#endif // __QAIC_HEADER_ATTRIBUTE

#ifndef __QAIC_IMPL
#define __QAIC_IMPL(ff) ff
#endif //__QAIC_IMPL

#ifndef __QAIC_IMPL_EXPORT
#define __QAIC_IMPL_EXPORT
#endif // __QAIC_IMPL_EXPORT

#ifndef __QAIC_IMPL_ATTRIBUTE
#define __QAIC_IMPL_ATTRIBUTE
#endif // __QAIC_IMPL_ATTRIBUTE
#ifdef __cplusplus
extern "C" {
#endif
__QAIC_HEADER_EXPORT int __QAIC_HEADER(calculator_init)(uint32* leet) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(calculator_extract_and_match)(const uint8* img, int imgLen, uint8* features, int featuresLen) __QAIC_HEADER_ATTRIBUTE;
#ifdef __cplusplus
}
#endif
#endif //_CALCULATOR_H
