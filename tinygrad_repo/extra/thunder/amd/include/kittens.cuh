/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */

#pragma once

#if defined(KITTENS_CDNA4)
#include "cdna4/includes.cuh"
#elif defined(KITTENS_UDNA1)
#include "udna1/includes.cuh"
#endif

#include "pyutils/util.cuh"


// #include "pyutils/pyutils.cuh" // for simple binding without including torch