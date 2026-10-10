/**
 * @file
 * @brief An aggregate header of warp memory operations on vectors, where a single warp loads or stores data on its own.
 */

#pragma once

#include "shared_to_register.cuh"
#include "global_to_register.cuh"
#include "global_to_shared.cuh"

#include "assembly/vec.cuh"