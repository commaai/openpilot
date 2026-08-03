/**
 * @file
 * @brief An aggregate header of colaborative group memory movement operations
 */

#include "util/util.cuh"
#include "tile/tile.cuh"
#include "vec/vec.cuh"

#ifdef KITTENS_HOPPER
struct tma {
#include "util/tma.cuh"
#include "tile/tma.cuh"
#include "vec/tma.cuh"
struct cluster {
#include "util/tma_cluster.cuh"
#include "tile/tma_cluster.cuh"
#include "vec/tma_cluster.cuh"
};
};
#endif