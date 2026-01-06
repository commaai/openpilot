/**
 * @file
 * @brief An aggregate header for all group-scope MMA operations.
 */

// All compilation targets can use the warp-scope MMA operations.
#include "warp/warp.cuh"

// Hopper has its own warpgroup-scope MMA operations.
#ifdef KITTENS_HOPPER
#include "warpgroup/warpgroup.cuh"
#endif

// Blackwell has its own tensor-scope MMA operations.
#ifdef KITTENS_BLACKWELL
#include "tensor/tensor.cuh"
#endif